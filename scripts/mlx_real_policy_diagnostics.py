#!/usr/bin/env python3
"""Real-policy PrefixIG/A-TGPO diagnostics with MLX.

This keeps the same no-training comparison as the synthetic smoke test, but
replaces mock old-policy scores with real MLX model token log-probs.

The default mode scores the controlled synthetic trajectories from
prefix_ig_tpo_synthetic.py, which gives an apples-to-apples version of the
paper table. It does not update model weights.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prefix_ig_tpo_smoke import (  # noqa: E402
    Sample,
    compute_diagnostics,
    mean_std,
    softmax,
)
from prefix_ig_tpo_synthetic import (  # noqa: E402
    atgpo_component_one_step_proxy,
    collect_metrics,
    make_case,
    print_baseline_comparison,
    print_component_diagnostics,
)


class MLXPolicyScorer:
    """MLX scorer for answer likelihood and completion log-probs."""

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        load_in_4bits: bool = False,
    ):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load

        self.mx = mx
        self.nn = nn
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        if load_in_4bits:
            nn.quantize(self.model, bits=4, group_size=128)
        self.model.eval()

    def encode(self, text: str) -> list[int]:
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            return self.tokenizer.encode(text)

    def decode_token(self, token_id: int) -> str:
        try:
            return self.tokenizer.decode([token_id])
        except TypeError:
            return str(token_id)

    def score(self, context: str, answer: str) -> float:
        prefix = context.rstrip() + "\nAnswer: "
        logps, _ = self.token_logprobs(prefix, answer)
        if not logps:
            return float("-inf")
        return sum(logps) / len(logps)

    def completion_mean_logp(self, prompt: str, completion: str) -> float:
        logps, _ = self.token_logprobs(prompt, completion)
        if not logps:
            return float("-inf")
        return sum(logps) / len(logps)

    def token_logprobs(self, prefix: str, suffix: str) -> tuple[list[float], list[str]]:
        mx = self.mx
        nn = self.nn
        prefix_ids = self.encode(prefix)
        suffix_ids = self.encode(suffix)
        if not prefix_ids or not suffix_ids:
            return [], []

        ids = prefix_ids + suffix_ids
        if len(ids) < 2:
            return [], []

        input_ids = mx.array([ids], dtype=mx.int32)
        logits = self.model(input_ids[:, :-1])
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.astype(mx.float32)
        targets = input_ids[:, 1:]
        log_probs = nn.log_softmax(logits, axis=-1)
        selected = mx.take_along_axis(
            log_probs, mx.expand_dims(targets, axis=-1), axis=-1
        ).squeeze(-1)

        start = max(len(prefix_ids) - 1, 0)
        end = start + len(suffix_ids)
        suffix_logps = selected[0, start:end]
        mx.eval(suffix_logps)
        return [float(value) for value in suffix_logps.tolist()], [
            self.decode_token(token_id) for token_id in suffix_ids
        ]


def math_log(value: float) -> float:
    return math.log(value)


def grpo_one_step_proxy(diags, step_scale: float):
    utilities = [diag.utility for diag in diags]
    utility_mean, utility_std = mean_std(utilities)
    advantages = [(value - utility_mean) / utility_std for value in utilities]
    proxy_logits = [
        math_log(max(diag.old_policy_prob, 1e-12)) + step_scale * advantage
        for diag, advantage in zip(diags, advantages)
    ]
    proxy_weights = softmax(proxy_logits)
    return [
        replace(diag, target_weight=proxy_weight)
        for diag, proxy_weight in zip(diags, proxy_weights)
    ]


def score_case_old_logps(case, scorer: MLXPolicyScorer) -> list[Sample]:
    scored = []
    for sample in case.samples:
        old_logp = scorer.completion_mean_logp(sample.prompt, sample.completion)
        scored.append(replace(sample, old_logp=old_logp))
    return scored


def print_real_policy_old_logp_summary(samples: list[Sample]) -> None:
    print()
    print("Real MLX old-policy scores")
    print("=" * 52)
    for sample in samples[:8]:
        print(f"{sample.sample_id}: mean completion logp={sample.old_logp:+.3f}")
    if len(samples) > 8:
        print(f"... {len(samples) - 8} more")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Local MLX model directory or Hugging Face repo visible to mlx_lm.load.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional LoRA adapter directory to load with the model.",
    )
    parser.add_argument(
        "--load-in-4bits",
        action="store_true",
        help="Quantize the loaded model to 4 bits before scoring/generation.",
    )
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lambda-ig", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--curvature-eps", type=float, default=1e-3)
    parser.add_argument("--grpo-step-scale", type=float, default=1.0)
    parser.add_argument("--atgpo-alpha", type=float, default=0.3)
    parser.add_argument("--atgpo-gamma", type=float, default=1.0)
    parser.add_argument("--atgpo-clip-low", type=float, default=3e-3)
    parser.add_argument("--atgpo-clip-high", type=float, default=4e-3)
    parser.add_argument("--atgpo-no-dynamic-clip", action="store_true")
    parser.add_argument("--atgpo-sim-step", type=float, default=0.05)
    parser.add_argument(
        "--show-old-logps",
        action="store_true",
        help="Print a short summary of real model old-policy log-probs.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    import random

    rng = random.Random(args.seed)
    scorer = MLXPolicyScorer(
        args.model,
        adapter_path=args.adapter_path,
        load_in_4bits=args.load_in_4bits,
    )
    cases = [make_case(case_id, rng) for case_id in range(args.num_examples)]

    scored_cases = []
    all_scored_samples = []
    for case in cases:
        scored_samples = score_case_old_logps(case, scorer)
        all_scored_samples.extend(scored_samples)
        scored_cases.append(replace(case, samples=scored_samples))

    if args.show_old_logps:
        print_real_policy_old_logp_summary(all_scored_samples)
        print()

    configs = [
        ("OriginalPolicy", "old", 0.0, False, "old_policy_prob"),
        ("FinalReward-TPO", "tpo", 0.0, False, "target_weight"),
        ("PrefixIG-GRPO", "grpo", args.lambda_ig, False, "target_weight"),
        ("A-TGPO-components", "atgpo", args.lambda_ig, False, "target_weight"),
        ("PrefixIG-TPO", "tpo", args.lambda_ig, False, "target_weight"),
        ("PrefixIG-TPO+curv", "tpo", args.lambda_ig, True, "target_weight"),
    ]

    results = []
    all_component_rows = []
    for name, mode, lambda_ig, apply_curvature, weight_field in configs:
        all_diags = []
        for case in scored_cases:
            case_diags = compute_diagnostics(
                samples=case.samples,
                scorer=scorer,
                lambda_ig=lambda_ig,
                tau=args.tau,
                curvature_eps=args.curvature_eps,
                apply_curvature=apply_curvature,
            )
            if mode == "grpo":
                case_diags = grpo_one_step_proxy(
                    case_diags, step_scale=args.grpo_step_scale
                )
            if mode == "atgpo":
                case_diags, component_rows = atgpo_component_one_step_proxy(
                    diags=case_diags,
                    samples=case.samples,
                    scorer=scorer,
                    step_scale=args.grpo_step_scale,
                    alpha=args.atgpo_alpha,
                    gamma=args.atgpo_gamma,
                    clip_low=args.atgpo_clip_low,
                    clip_high=args.atgpo_clip_high,
                    dynamic_clip=not args.atgpo_no_dynamic_clip,
                    sim_step=args.atgpo_sim_step,
                    token_logprob_scorer=scorer,
                )
                all_component_rows.extend(component_rows)
            all_diags.extend(case_diags)
        results.append((name, collect_metrics(all_diags, weight_field=weight_field)))

    print_baseline_comparison(results)
    if all_component_rows:
        print_component_diagnostics(all_component_rows)


if __name__ == "__main__":
    main()
