#!/usr/bin/env python3
"""Build offline trajectory targets for small Qwen LoRA pilots.

This script turns the generated-policy diagnostic into trainable offline
datasets:

1. Generate K search trajectories per synthetic QA case with a real MLX model.
2. Score each group with one or more target policies.
3. Write all candidates with target weights for later offline TPO.
4. Sample SFT-style train/valid/test JSONL files from each target distribution.

The SFT JSONL is a practical laptop experiment: it distills a target trajectory
distribution into a LoRA adapter using the existing mlx-lm-lora SFT trainer. The
full candidate JSONL preserves the group weights for a later true offline TPO
trainer.
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
from prefix_ig_tpo_smoke import (  # noqa: E402
    Sample,
    compute_diagnostics,
    extract_answer,
    extract_boxed,
    mean_std,
    softmax,
)
from prefix_ig_tpo_synthetic import atgpo_component_one_step_proxy, make_case  # noqa: E402

TARGET_METHODS = {
    "atgpo_proxy",
    "reward_tpo",
    "prefixig_tpo",
    "prefixig_tpo_rg_eff",
}


def parse_target_methods(raw: str) -> list[str]:
    aliases = {
        "a_tgpo_proxy": "atgpo_proxy",
        "atgpo_components": "atgpo_proxy",
        "a-tgpo-components": "atgpo_proxy",
        "final_reward_tpo": "reward_tpo",
        "rg_eff": "prefixig_tpo_rg_eff",
        "prefixig_tpo+rg_eff": "prefixig_tpo_rg_eff",
    }
    methods = []
    for item in raw.split(","):
        method = aliases.get(item.strip(), item.strip())
        if not method:
            continue
        if method not in TARGET_METHODS:
            valid = ", ".join(sorted(TARGET_METHODS))
            raise ValueError(f"Unknown target method {method!r}; expected one of: {valid}")
        methods.append(method)
    if not methods:
        raise ValueError("At least one target method is required.")
    return methods


def method_output_dir(base_output_dir: Path, method: str, methods: list[str]) -> Path:
    if len(methods) == 1:
        return base_output_dir
    return base_output_dir / method


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


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    parser.add_argument("--model", default=None)
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
    parser.add_argument(
        "--input-candidates-jsonl",
        default=None,
        help=(
            "Reuse an existing candidates JSONL instead of generating new "
            "trajectories. This is useful for building comparable LoRA "
            "datasets from the same cached Qwen rollouts. The atgpo_proxy "
            "method additionally requires --model because it recomputes "
            "token-level component scores."
        ),
    )
    parser.add_argument(
        "--target-methods",
        default="prefixig_tpo_rg_eff",
        help=(
            "Comma-separated target datasets to build. Supported: "
            "reward_tpo,prefixig_tpo,prefixig_tpo_rg_eff,atgpo_proxy. When "
            "more than one method is requested, each dataset is written under "
            "output-dir/METHOD."
        ),
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
    parser.add_argument("--grpo-step-scale", type=float, default=1.0)
    parser.add_argument("--atgpo-alpha", type=float, default=0.3)
    parser.add_argument("--atgpo-gamma", type=float, default=1.0)
    parser.add_argument("--atgpo-clip-low", type=float, default=3e-3)
    parser.add_argument("--atgpo-clip-high", type=float, default=4e-3)
    parser.add_argument("--atgpo-sim-step", type=float, default=0.05)
    return parser


def compute_target_diags(method: str, generated_samples: list[Sample], scorer, args):
    if method == "reward_tpo":
        return compute_diagnostics(
            samples=generated_samples,
            scorer=scorer,
            lambda_ig=0.0,
            tau=args.tau,
            curvature_eps=args.curvature_eps,
            apply_curvature=False,
        )

    base_diags = compute_diagnostics(
        samples=generated_samples,
        scorer=scorer,
        lambda_ig=args.lambda_ig,
        tau=args.tau,
        curvature_eps=args.curvature_eps,
        apply_curvature=False,
    )
    if method == "prefixig_tpo":
        return base_diags
    if method == "prefixig_tpo_rg_eff":
        return apply_efficiency_penalty(
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
    raise ValueError(f"Unknown target method: {method}")


def group_records_by_case(records: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for record in records:
        grouped.setdefault(int(record["case_id"]), []).append(record)
    return grouped


def cached_atgpo_proxy_records(records: list[dict], scorer, args) -> list[dict]:
    out: list[dict] = []
    for _case_id, case_records in sorted(group_records_by_case(records).items()):
        samples = [
            Sample(
                sample_id=str(record["sample_id"]),
                prompt=str(record.get("generation_prompt") or record["prompt"]),
                completion=str(record["completion"]),
                answer=str(record["answer"]),
                old_logp=float(record["old_logp"]),
            )
            for record in case_records
        ]
        base_diags = compute_diagnostics(
            samples=samples,
            scorer=scorer,
            lambda_ig=args.lambda_ig,
            tau=args.tau,
            curvature_eps=args.curvature_eps,
            apply_curvature=False,
        )
        proxy_diags, component_rows = atgpo_component_one_step_proxy(
            diags=base_diags,
            samples=samples,
            scorer=scorer,
            step_scale=args.grpo_step_scale,
            alpha=args.atgpo_alpha,
            gamma=args.atgpo_gamma,
            clip_low=args.atgpo_clip_low,
            clip_high=args.atgpo_clip_high,
            dynamic_clip=True,
            sim_step=args.atgpo_sim_step,
            token_logprob_scorer=scorer,
        )
        for record, diag, component in zip(case_records, proxy_diags, component_rows):
            copied = dict(record)
            copied["target_method"] = "atgpo_proxy"
            copied["utility"] = diag.utility
            copied["target_weight"] = diag.target_weight
            copied["atgpo_proxy_weight"] = component.proxy_weight
            copied["atgpo_weighted_pg_loss"] = component.weighted_pg_loss
            copied["atgpo_clip_fraction"] = component.clip_fraction
            copied["atgpo_sampled_kl"] = component.sampled_kl
            copied["atgpo_token_advantage"] = component.token_advantage
            out.append(copied)
    return out


def cached_method_records(method: str, records: list[dict], args, scorer=None) -> list[dict]:
    if method == "atgpo_proxy":
        if scorer is None:
            raise ValueError("atgpo_proxy requires --model when using cached candidates.")
        return cached_atgpo_proxy_records(records, scorer=scorer, args=args)

    by_case = group_records_by_case(records)
    out: list[dict] = []
    for _case_id, case_records in sorted(by_case.items()):
        prefix_igs = [float(record.get("prefix_ig", 0.0)) for record in case_records]
        ig_mean, ig_std = mean_std(prefix_igs)
        utilities = []
        for record in case_records:
            reward = float(record.get("reward", 0.0))
            if method == "reward_tpo":
                utility = reward
            elif method == "prefixig_tpo":
                z_ig = (float(record.get("prefix_ig", 0.0)) - ig_mean) / ig_std
                utility = reward + args.lambda_ig * z_ig
            elif method == "prefixig_tpo_rg_eff":
                # Existing pilot candidates were produced by the rg_eff builder.
                # Reuse the stored utility so all baselines share the same rollouts.
                utility = float(record.get("utility", reward))
            else:
                raise ValueError(f"Unknown target method: {method}")
            utilities.append(utility)

        weights = softmax(
            [
                float(record["old_logp"]) + utility / max(args.tau, 1e-8)
                for record, utility in zip(case_records, utilities)
            ]
        )
        for record, utility, weight in zip(case_records, utilities, weights):
            copied = dict(record)
            copied["target_method"] = method
            copied["utility"] = utility
            copied["target_weight"] = weight
            out.append(copied)
    return out


def build_sft_records(
    method_records: list[dict],
    sft_samples_per_example: int,
    rng: random.Random,
) -> list[dict]:
    sft_records: list[dict] = []
    for _case_id, case_candidates in sorted(group_records_by_case(method_records).items()):
        weights = [float(record["target_weight"]) for record in case_candidates]
        for draw_idx in range(sft_samples_per_example):
            chosen = weighted_choice(case_candidates, weights, rng)
            sft_records.append(
                {
                    "prompt": chosen["prompt"],
                    "completion": chosen["completion"],
                    "case_id": chosen["case_id"],
                    "source_sample_id": chosen["sample_id"],
                    "draw_id": draw_idx,
                    "target_method": chosen["target_method"],
                    "bucket": chosen["bucket"],
                    "target_weight": chosen["target_weight"],
                    "answer": chosen["answer"],
                }
            )
    return sft_records


def summarize_method(method: str, records: list[dict], sft_records: list[dict], output_dir: Path, args) -> None:
    train, valid, test = split_records(
        sft_records,
        rng=random.Random(args.seed + 10_003 + sorted(TARGET_METHODS).index(method)),
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
    )
    write_sft_dataset(output_dir, train=train, valid=valid, test=test)

    by_case = group_records_by_case(records)
    denom = max(len(by_case), 1)
    bucket_counts: dict[str, int] = {}
    target_mass: dict[str, float] = {}
    for record in records:
        bucket = str(record["bucket"])
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        target_mass[bucket] = target_mass.get(bucket, 0.0) + float(record["target_weight"])

    print()
    print(method)
    print("-" * len(method))
    print(f"sft records: {len(sft_records)} -> {output_dir}")
    print(f"split: train={len(train)} valid={len(valid)} test={len(test)}")
    print("Bucket counts")
    for bucket in sorted(bucket_counts):
        print(f"- {bucket}: {bucket_counts[bucket]}")
    print("Average target mass per case")
    for bucket in sorted(target_mass):
        print(f"- {bucket}: {target_mass[bucket] / denom:.3f}")


def main() -> None:
    args = build_arg_parser().parse_args()
    target_methods = parse_target_methods(args.target_methods)
    rng = random.Random(args.seed)
    if args.input_candidates_jsonl:
        source_records = read_jsonl(Path(args.input_candidates_jsonl))
        candidate_records: list[dict] = []
        print(f"Reusing cached candidates from {args.input_candidates_jsonl}")
        print(f"source candidate trajectories: {len(source_records)}")
        scorer = None
        if "atgpo_proxy" in target_methods:
            if args.model is None:
                raise ValueError("atgpo_proxy with --input-candidates-jsonl requires --model.")
            scorer = MLXPolicyScorer(
                args.model,
                adapter_path=args.adapter_path,
                load_in_4bits=args.load_in_4bits,
            )
        for method in target_methods:
            method_records = cached_method_records(method, source_records, args, scorer=scorer)
            candidate_records.extend(method_records)
            sft_records = build_sft_records(
                method_records=method_records,
                sft_samples_per_example=args.sft_samples_per_example,
                rng=random.Random(args.seed + 907 + target_methods.index(method)),
            )
            out_dir = method_output_dir(Path(args.output_dir), method, target_methods)
            summarize_method(method, method_records, sft_records, out_dir, args)

        write_jsonl(Path(args.candidates_jsonl), candidate_records)
        print()
        print(f"candidate trajectories: {len(candidate_records)} -> {args.candidates_jsonl}")
        return

    if args.model is None:
        raise ValueError("--model is required unless --input-candidates-jsonl is provided.")

    scorer = MLXPolicyScorer(
        args.model,
        adapter_path=args.adapter_path,
        load_in_4bits=args.load_in_4bits,
    )

    candidate_records: list[dict] = []
    sft_records_by_method: dict[str, list[dict]] = {method: [] for method in target_methods}
    bucket_counts_by_method: dict[str, dict[str, int]] = {
        method: {} for method in target_methods
    }
    target_mass_by_method: dict[str, dict[str, float]] = {
        method: {} for method in target_methods
    }

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

        progress_parts = []
        for method in target_methods:
            target_diags = compute_target_diags(method, generated_samples, scorer, args)
            bucket_counts = bucket_counts_by_method[method]
            target_mass = target_mass_by_method[method]

            case_candidates: list[dict] = []
            for diag, sample in zip(target_diags, generated_samples):
                bucket = classify_generated_sample(sample, diag)
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
                target_mass[bucket] = target_mass.get(bucket, 0.0) + float(diag.target_weight)

                prediction = extract_answer(sample.completion) or extract_boxed(sample.completion) or ""
                record = {
                    "target_method": method,
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
                sft_records_by_method[method].append(
                    {
                        "prompt": chosen["prompt"],
                        "completion": chosen["completion"],
                        "case_id": chosen["case_id"],
                        "source_sample_id": chosen["sample_id"],
                        "draw_id": draw_idx,
                        "target_method": method,
                        "bucket": chosen["bucket"],
                        "target_weight": chosen["target_weight"],
                        "answer": chosen["answer"],
                    }
                )
            best_bucket = max(case_candidates, key=lambda item: item["target_weight"])["bucket"]
            progress_parts.append(f"{method}:best={best_bucket}")

        print(f"case {case_idx + 1:03d}/{args.num_examples}: " + " | ".join(progress_parts))

    write_jsonl(Path(args.candidates_jsonl), candidate_records)

    denom = max(args.num_examples, 1)
    print()
    print("Offline target build complete")
    print("=" * 44)
    print(f"candidate trajectories: {len(candidate_records)} -> {args.candidates_jsonl}")
    for method in target_methods:
        method_rng = random.Random(args.seed + 10_003 + target_methods.index(method))
        sft_records = sft_records_by_method[method]
        train, valid, test = split_records(
            sft_records,
            rng=method_rng,
            valid_fraction=args.valid_fraction,
            test_fraction=args.test_fraction,
        )
        out_dir = method_output_dir(Path(args.output_dir), method, target_methods)
        write_sft_dataset(out_dir, train=train, valid=valid, test=test)

        print()
        print(f"{method}")
        print("-" * len(method))
        print(f"sft records: {len(sft_records)} -> {out_dir}")
        print(f"split: train={len(train)} valid={len(valid)} test={len(test)}")
        print("Bucket counts")
        for bucket in sorted(bucket_counts_by_method[method]):
            print(f"- {bucket}: {bucket_counts_by_method[method][bucket]}")
        print("Average target mass per case")
        for bucket in sorted(target_mass_by_method[method]):
            print(f"- {bucket}: {target_mass_by_method[method][bucket] / denom:.3f}")


if __name__ == "__main__":
    main()
