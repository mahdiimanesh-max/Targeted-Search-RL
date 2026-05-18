#!/usr/bin/env python3
"""Mac-scale online PrefixIG-TPO LoRA training.

This is the online counterpart to the offline TPO pilot. Each outer iteration
samples fresh search trajectories from the current policy, builds a PrefixIG-TPO
target distribution over the sampled group, and updates the LoRA adapter with a
grouped TPO loss plus a small token/action behavior anchor.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_offline_rg_eff_targets import (  # noqa: E402
    build_oracle_anchor_completion,
    case_regime,
    compute_target_diags,
)
from mlx_generated_policy_diagnostics import (  # noqa: E402
    build_flexible_tool_loop_prompt,
    build_generation_prompt,
    build_regime_case,
    build_singlehop_flexible_tool_loop_prompt,
    build_tool_loop_prompt,
    classify_generated_sample,
    extract_answer,
    extract_boxed,
    format_prompt,
    generate_completion,
    rollout_with_oracle_search,
    write_jsonl,
)
from prefix_ig_tpo_smoke import Sample  # noqa: E402
from prefix_ig_tpo_synthetic import make_case  # noqa: E402
from train_offline_tpo_lora import (  # noqa: E402
    encode_group_records,
    make_group_batch,
    offline_tpo_loss,
    sequence_log_scores,
)


class CurrentPolicyScorer:
    """Policy scorer backed by the same model object that is being updated."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.mx = mx
        self.nn = nn

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
        logps, _tokens = self.token_logprobs(prefix, answer)
        if not logps:
            return float("-inf")
        return sum(logps) / len(logps)

    def completion_mean_logp(self, prompt: str, completion: str) -> float:
        logps, _tokens = self.token_logprobs(prompt, completion)
        if not logps:
            return float("-inf")
        return sum(logps) / len(logps)

    def token_logprobs(self, prefix: str, suffix: str) -> tuple[list[float], list[str]]:
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument(
        "--resume-adapter-file",
        default=None,
        help="Warm-start LoRA weights, usually the best weighted-SFT adapter.",
    )
    parser.add_argument("--load-in-4bits", action="store_true")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--online-iters", type=int, default=5)
    parser.add_argument("--prompts-per-iter", type=int, default=2)
    parser.add_argument("--samples-per-prompt", type=int, default=3)
    parser.add_argument("--updates-per-iter", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--anchor-beta", type=float, default=0.05)
    parser.add_argument(
        "--target-method",
        default="prefixig_tpo",
        choices=[
            "prefixig_tpo",
            "prefixig_tpo_curv",
            "prefixig_tpo_eff",
            "prefixig_tpo_rg_eff",
            "prefixig_tpo_curv_eff",
            "prefixig_tpo_curv_rg_eff",
        ],
    )
    parser.add_argument(
        "--eval-regime",
        choices=["multihop", "singlehop", "mixedhop", "noisy", "mixedhop_noisy"],
        default="mixedhop_noisy",
    )
    parser.add_argument("--rollout-mode", choices=["guided", "flexible"], default="flexible")
    parser.add_argument("--one-shot", action="store_true")
    parser.add_argument("--oracle-anchor-count", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--action-max-tokens", type=int, default=32)
    parser.add_argument("--answer-max-tokens", type=int, default=24)
    parser.add_argument("--max-search-turns", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--lambda-ig", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--curvature-eps", type=float, default=1e-3)
    parser.add_argument("--eff-optimal-turns", type=int, default=2)
    parser.add_argument("--eff-repeat-threshold", type=float, default=0.55)
    parser.add_argument("--eff-min-turn-ig", type=float, default=0.05)
    parser.add_argument("--eff-lambda-extra-turn", type=float, default=0.6)
    parser.add_argument("--eff-lambda-repeat-query", type=float, default=0.4)
    parser.add_argument("--eff-lambda-low-ig", type=float, default=0.0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--score-normalization", choices=["mean", "sum"], default="mean")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--show-samples", action="store_true")
    return parser


def prompt_for_case(case, active_regime: str, search_index: list[str], scorer, args) -> tuple[str, str]:
    user_case_prompt, search_index = build_regime_case(case, active_regime, random.Random(args.seed))
    if args.one_shot:
        user_prompt = build_generation_prompt(user_case_prompt, search_index)
    elif args.rollout_mode == "flexible":
        if active_regime == "singlehop":
            user_prompt = build_singlehop_flexible_tool_loop_prompt(
                user_case_prompt, max_search_turns=args.max_search_turns
            )
        else:
            user_prompt = build_flexible_tool_loop_prompt(
                user_case_prompt, max_search_turns=args.max_search_turns
            )
    else:
        user_prompt = build_tool_loop_prompt(user_case_prompt)
    generation_prompt = format_prompt(
        scorer=scorer,
        user_prompt=user_prompt,
        use_chat_template=True,
    )
    return user_prompt, generation_prompt


def sample_online_groups(model, tokenizer, case_offset: int, args) -> tuple[list[list[dict]], list[dict]]:
    scorer = CurrentPolicyScorer(model, tokenizer)
    rng = random.Random(args.seed + 1009 * case_offset)
    groups: list[list[dict]] = []
    records: list[dict] = []
    cases = [make_case(case_offset + idx, rng) for idx in range(args.prompts_per_iter)]

    model.eval()
    for local_idx, case in enumerate(cases):
        case_idx = case_offset + local_idx
        active_regime = case_regime(args.eval_regime, case_idx)
        user_case_prompt, search_index = build_regime_case(case, active_regime, rng)
        if args.one_shot:
            user_prompt = build_generation_prompt(user_case_prompt, search_index)
        elif args.rollout_mode == "flexible":
            if active_regime == "singlehop":
                user_prompt = build_singlehop_flexible_tool_loop_prompt(
                    user_case_prompt, max_search_turns=args.max_search_turns
                )
            else:
                user_prompt = build_flexible_tool_loop_prompt(
                    user_case_prompt, max_search_turns=args.max_search_turns
                )
        else:
            user_prompt = build_tool_loop_prompt(user_case_prompt)
        generation_prompt = format_prompt(
            scorer=scorer,
            user_prompt=user_prompt,
            use_chat_template=True,
        )

        samples: list[Sample] = []
        for sample_idx in range(args.samples_per_prompt):
            if args.one_shot:
                completion = generate_completion(
                    scorer=scorer,
                    prompt=generation_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    min_p=args.min_p,
                    top_k=args.top_k,
                )
            else:
                completion = rollout_with_oracle_search(
                    scorer=scorer,
                    prompt=generation_prompt,
                    search_index=search_index,
                    max_turns=args.max_search_turns,
                    action_max_tokens=args.action_max_tokens,
                    answer_max_tokens=args.answer_max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    min_p=args.min_p,
                    top_k=args.top_k,
                )
            old_logp = scorer.completion_mean_logp(generation_prompt, completion)
            samples.append(
                Sample(
                    sample_id=f"online{case_idx:04d}_gen{sample_idx:02d}",
                    prompt=generation_prompt,
                    completion=completion,
                    answer=case.answer,
                    old_logp=old_logp,
                )
            )

        for anchor_idx in range(args.oracle_anchor_count):
            completion = build_oracle_anchor_completion(case, active_regime, search_index)
            old_logp = scorer.completion_mean_logp(generation_prompt, completion)
            samples.append(
                Sample(
                    sample_id=f"online{case_idx:04d}_oracle{anchor_idx:02d}",
                    prompt=generation_prompt,
                    completion=completion,
                    answer=case.answer,
                    old_logp=old_logp,
                )
            )

        diags = compute_target_diags(args.target_method, samples, scorer, args)
        group_records = []
        for diag, sample in zip(diags, samples):
            bucket = classify_generated_sample(sample, diag)
            prediction = extract_answer(sample.completion) or extract_boxed(sample.completion) or ""
            record = {
                "target_method": args.target_method,
                "online_iter_case_offset": case_offset,
                "eval_regime": active_regime,
                "case_id": case_idx,
                "sample_id": sample.sample_id,
                "prompt": user_prompt,
                "generation_prompt": generation_prompt,
                "completion": sample.completion,
                "answer": case.answer,
                "prediction": prediction,
                "bucket": bucket,
                "old_logp": sample.old_logp,
                "reward": diag.final_reward,
                "prefix_ig": diag.prefix_ig,
                "turn_igs": list(diag.turn_igs),
                "curvature": diag.curvature,
                "utility": diag.utility,
                "target_weight": diag.target_weight,
            }
            group_records.append(record)
            records.append(record)
        groups.append(group_records)

    mx.clear_cache()
    return groups, records


def online_tpo_anchor_loss(
    model,
    inputs,
    completion_mask,
    q_target,
    score_normalization: str,
    anchor_beta: float,
):
    tpo_loss, ntoks, policy_entropy, target_entropy, raw_tpo_loss, _kl_anchor = offline_tpo_loss(
        model,
        inputs,
        completion_mask,
        q_target,
        score_normalization,
        None,
        0.0,
    )
    mean_scores, _lengths = sequence_log_scores(
        model=model,
        inputs=inputs,
        completion_mask=completion_mask,
        score_normalization="mean",
    )
    anchor_loss = -(mx.stop_gradient(q_target) * mean_scores).sum()
    loss = tpo_loss + anchor_beta * anchor_loss
    return loss, ntoks, policy_entropy, target_entropy, raw_tpo_loss, anchor_loss


def bucket_summary(records: list[dict]) -> dict[str, float]:
    grouped: dict[int, list[dict]] = {}
    for record in records:
        grouped.setdefault(int(record["case_id"]), []).append(record)
    mass: dict[str, float] = {}
    for record in records:
        mass[str(record["bucket"])] = mass.get(str(record["bucket"]), 0.0) + float(
            record["target_weight"]
        )
    denom = max(len(grouped), 1)
    return {bucket: value / denom for bucket, value in sorted(mass.items())}


def save_adapter(model, adapter_file: Path, iteration: int | None = None) -> None:
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_file), adapter_weights)
    if iteration is not None:
        checkpoint = adapter_file.parent / f"{iteration:07d}_adapters.safetensors"
        mx.save_safetensors(str(checkpoint), adapter_weights)
        print(f"Saved adapter to {adapter_file} and {checkpoint}")
    else:
        print(f"Saved adapter to {adapter_file}")


def main() -> None:
    args = build_arg_parser().parse_args()

    from mlx_lm.tuner.utils import print_trainable_parameters
    from mlx_lm.utils import save_config
    from mlx_lm_lora.utils import from_pretrained

    random.seed(args.seed)
    mx.random.seed(args.seed)

    lora_config = {
        "rank": args.rank,
        "dropout": args.dropout,
        "scale": args.scale,
        "use_dora": False,
        "num_layers": args.num_layers,
    }
    quantized_load = {"bits": 4} if args.load_in_4bits else None
    model, tokenizer, adapter_file = from_pretrained(
        model=args.model,
        new_adapter_path=args.adapter_path,
        lora_config=lora_config,
        quantized_load=quantized_load,
    )
    if args.resume_adapter_file:
        print(f"Loading warm-start adapter from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    adapter_file = Path(adapter_file or adapter_path / "adapters.safetensors")
    save_config(
        {
            "fine_tune_type": "lora",
            "num_layers": args.num_layers,
            "lora_parameters": lora_config,
        },
        adapter_path / "adapter_config.json",
    )
    print_trainable_parameters(model)

    optimizer = optim.AdamW(learning_rate=args.learning_rate)
    loss_value_and_grad = nn.value_and_grad(model, online_tpo_anchor_loss)
    all_records: list[dict] = []
    start = time.perf_counter()

    print()
    print("Online PrefixIG-TPO")
    print("=" * 56)
    print(f"target_method: {args.target_method}")
    print(f"eval_regime: {args.eval_regime}")
    print(f"anchor_beta: {args.anchor_beta}")
    print(
        f"schedule: {args.online_iters} iters x {args.prompts_per_iter} prompts "
        f"x {args.samples_per_prompt} samples, {args.updates_per_iter} updates/iter"
    )

    for online_iter in range(1, args.online_iters + 1):
        case_offset = (online_iter - 1) * args.prompts_per_iter
        raw_groups, records = sample_online_groups(
            model=model,
            tokenizer=tokenizer,
            case_offset=case_offset,
            args=args,
        )
        all_records.extend(records)
        encoded_groups = encode_group_records(
            raw_groups,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
        if not encoded_groups:
            print(f"Iter {online_iter}: no trainable groups; skipping update")
            continue

        print()
        print(f"Iter {online_iter}: sampled {len(records)} trajectories")
        print("target mass:", bucket_summary(records))
        if args.show_samples:
            for record in records[: min(3, len(records))]:
                print(
                    f"- {record['sample_id']} {record['bucket']} "
                    f"q={float(record['target_weight']):.3f}: "
                    f"{record['completion'][:180]!r}"
                )

        model.train()
        losses = []
        anchor_losses = []
        for update_idx in tqdm(
            range(args.updates_per_iter),
            desc=f"online_update_{online_iter}",
        ):
            group = encoded_groups[update_idx % len(encoded_groups)]
            inputs, mask, q, _buckets, norm = make_group_batch(group, args.score_normalization)
            (
                loss,
                ntoks,
                policy_entropy,
                target_entropy,
                tpo_loss,
                anchor_loss,
            ), grad = loss_value_and_grad(
                model,
                inputs,
                mask,
                q,
                norm,
                args.anchor_beta,
            )
            optimizer.update(model, grad)
            mx.eval(
                model.state,
                optimizer.state,
                loss,
                ntoks,
                policy_entropy,
                target_entropy,
                tpo_loss,
                anchor_loss,
            )
            losses.append(float(loss.item()))
            anchor_losses.append(float(anchor_loss.item()))

        elapsed = max(time.perf_counter() - start, 1e-8)
        print(
            f"Iter {online_iter}: loss={sum(losses) / len(losses):.3f}, "
            f"anchor_loss={sum(anchor_losses) / len(anchor_losses):.3f}, "
            f"policy_entropy={float(policy_entropy.item()):.3f}, "
            f"target_entropy={float(target_entropy.item()):.3f}, "
            f"peak_mem={mx.get_peak_memory() / 1e9:.3f}GB, elapsed={elapsed:.1f}s"
        )
        start = time.perf_counter()

        if online_iter % args.save_every == 0:
            save_adapter(model, adapter_file, online_iter)
        mx.clear_cache()

    save_adapter(model, adapter_file)
    if args.output_jsonl:
        write_jsonl(Path(args.output_jsonl), all_records)
        print(f"Wrote online rollouts to {args.output_jsonl}")


if __name__ == "__main__":
    main()
