#!/usr/bin/env python3
"""Build offline rg_eff trajectory targets for a small Qwen LoRA pilot.

This script turns the generated-policy diagnostic into a trainable offline
dataset:

1. Generate K search trajectories per synthetic QA case with a real MLX model.
2. Score each group with PrefixIG-TPO plus reward-gated efficiency.
3. Write all candidates with target weights for later offline TPO.
4. Sample an SFT-style train/valid/test JSONL from the target distribution.

The SFT JSONL is a practical first laptop experiment: it distills the rg_eff
trajectory distribution into a LoRA adapter using the existing mlx-lm-lora SFT
trainer. The full candidate JSONL preserves the group weights for a later true
offline TPO trainer.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mlx_generated_policy_diagnostics import (  # noqa: E402
    apply_efficiency_penalty,
    build_flexible_tool_loop_prompt,
    build_generation_prompt,
    build_tool_loop_prompt,
    classify_generated_sample,
    format_prompt,
    generate_completion,
    rollout_with_oracle_search,
    unique_results,
    write_jsonl,
)
from mlx_real_policy_diagnostics import MLXPolicyScorer  # noqa: E402
from prefix_ig_tpo_smoke import Sample, compute_diagnostics, extract_answer, extract_boxed  # noqa: E402
from prefix_ig_tpo_synthetic import make_case  # noqa: E402


def split_records(
    records: list[dict],
    rng: random.Random,
    valid_fraction: float,
    test_fraction: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_valid = int(round(n_total * valid_fraction))
    n_test = int(round(n_total * test_fraction))
    n_valid = min(n_valid, max(n_total - 1, 0))
    n_test = min(n_test, max(n_total - n_valid - 1, 0))
    valid = shuffled[:n_valid]
    test = shuffled[n_valid : n_valid + n_test]
    train = shuffled[n_valid + n_test :]
    return train, valid, test


def write_sft_dataset(output_dir: Path, train: list[dict], valid: list[dict], test: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "valid.jsonl", valid)
    write_jsonl(output_dir / "test.jsonl", test)


def weighted_choice(items: list[dict], weights: list[float], rng: random.Random) -> dict:
    total = sum(max(weight, 0.0) for weight in weights)
    if total <= 0.0:
        return rng.choice(items)
    threshold = rng.random() * total
    running = 0.0
    for item, weight in zip(items, weights):
        running += max(weight, 0.0)
        if running >= threshold:
            return item
    return items[-1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
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
    parser.add_argument("--output-dir", default="outputs/offline_rg_eff_qwen_sft")
    parser.add_argument(
        "--candidates-jsonl",
        default="outputs/offline_rg_eff_qwen_candidates.jsonl",
    )
    parser.add_argument("--num-examples", type=int, default=16)
    parser.add_argument("--samples-per-example", type=int, default=4)
    parser.add_argument("--sft-samples-per-example", type=int, default=8)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--action-max-tokens", type=int, default=64)
    parser.add_argument("--answer-max-tokens", type=int, default=48)
    parser.add_argument("--max-search-turns", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="Use one-shot generation instead of the oracle search loop.",
    )
    parser.add_argument(
        "--rollout-mode",
        choices=["guided", "flexible"],
        default="flexible",
    )
    parser.add_argument("--lambda-ig", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--curvature-eps", type=float, default=1e-3)
    parser.add_argument("--eff-optimal-turns", type=int, default=2)
    parser.add_argument("--eff-repeat-threshold", type=float, default=0.55)
    parser.add_argument("--eff-min-turn-ig", type=float, default=0.05)
    parser.add_argument("--eff-lambda-extra-turn", type=float, default=0.6)
    parser.add_argument("--eff-lambda-repeat-query", type=float, default=0.4)
    parser.add_argument("--eff-lambda-low-ig", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = random.Random(args.seed)
    scorer = MLXPolicyScorer(
        args.model,
        adapter_path=args.adapter_path,
        load_in_4bits=args.load_in_4bits,
    )

    candidate_records: list[dict] = []
    sft_records: list[dict] = []
    bucket_counts: dict[str, int] = {}
    target_mass: dict[str, float] = {}

    cases = [make_case(case_id, rng) for case_id in range(args.num_examples)]

    for case_idx, case in enumerate(cases):
        search_index = unique_results([sample.completion for sample in case.samples])
        if args.one_shot:
            user_prompt = build_generation_prompt(case.prompt, search_index)
        elif args.rollout_mode == "flexible":
            user_prompt = build_flexible_tool_loop_prompt(
                case.prompt, max_search_turns=args.max_search_turns
            )
        else:
            user_prompt = build_tool_loop_prompt(case.prompt)

        generation_prompt = format_prompt(
            scorer=scorer,
            user_prompt=user_prompt,
            use_chat_template=not args.no_chat_template,
        )

        generated_samples: list[Sample] = []
        for sample_idx in range(args.samples_per_example):
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
            generated_samples.append(
                Sample(
                    sample_id=f"case{case_idx:04d}_gen{sample_idx:02d}",
                    prompt=generation_prompt,
                    completion=completion,
                    answer=case.answer,
                    old_logp=old_logp,
                )
            )

        base_diags = compute_diagnostics(
            samples=generated_samples,
            scorer=scorer,
            lambda_ig=args.lambda_ig,
            tau=args.tau,
            curvature_eps=args.curvature_eps,
            apply_curvature=False,
        )
        target_diags = apply_efficiency_penalty(
            diags=base_diags,
            samples=generated_samples,
            tau=args.tau,
            apply_curvature=False,
            reward_gated=True,
            curvature_eps=args.curvature_eps,
            optimal_turns=args.eff_optimal_turns,
            repeat_threshold=args.eff_repeat_threshold,
            min_turn_ig=args.eff_min_turn_ig,
            lambda_extra_turn=args.eff_lambda_extra_turn,
            lambda_repeat_query=args.eff_lambda_repeat_query,
            lambda_low_ig=args.eff_lambda_low_ig,
        )

        case_candidates: list[dict] = []
        for diag, sample in zip(target_diags, generated_samples):
            bucket = classify_generated_sample(sample, diag)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            target_mass[bucket] = target_mass.get(bucket, 0.0) + float(diag.target_weight)

            prediction = extract_answer(sample.completion) or extract_boxed(sample.completion) or ""
            record = {
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
            case_candidates.append(record)
            candidate_records.append(record)

        weights = [float(record["target_weight"]) for record in case_candidates]
        for draw_idx in range(args.sft_samples_per_example):
            chosen = weighted_choice(case_candidates, weights, rng)
            sft_records.append(
                {
                    "prompt": chosen["prompt"],
                    "completion": chosen["completion"],
                    "case_id": chosen["case_id"],
                    "source_sample_id": chosen["sample_id"],
                    "draw_id": draw_idx,
                    "bucket": chosen["bucket"],
                    "target_weight": chosen["target_weight"],
                    "answer": chosen["answer"],
                }
            )

        print(
            f"case {case_idx + 1:03d}/{args.num_examples}: "
            f"best={max(case_candidates, key=lambda item: item['target_weight'])['bucket']} "
            f"mass_sum={sum(weights):.3f}"
        )

    train, valid, test = split_records(
        sft_records,
        rng=rng,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
    )
    write_jsonl(Path(args.candidates_jsonl), candidate_records)
    write_sft_dataset(Path(args.output_dir), train=train, valid=valid, test=test)

    denom = max(args.num_examples, 1)
    print()
    print("Offline rg_eff target build complete")
    print("=" * 44)
    print(f"candidate trajectories: {len(candidate_records)} -> {args.candidates_jsonl}")
    print(f"sft records: {len(sft_records)} -> {args.output_dir}")
    print(f"split: train={len(train)} valid={len(valid)} test={len(test)}")
    print()
    print("Bucket counts")
    for bucket in sorted(bucket_counts):
        print(f"- {bucket}: {bucket_counts[bucket]}")
    print()
    print("Average target mass per case")
    for bucket in sorted(target_mass):
        print(f"- {bucket}: {target_mass[bucket] / denom:.3f}")


if __name__ == "__main__":
    main()
