#!/usr/bin/env python3
"""Run a small online PrefixIG-TPO anchor-beta sweep.

The script launches one online training run per beta, evaluates each resulting
adapter on single-hop, multi-hop, and noisy regimes, then writes a compact CSV
with OriginalPolicy metrics.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from summarize_original_policy_evals import original_policy_metrics, read_jsonl


REGIMES = ("singlehop", "multihop", "noisy")


def beta_tag(beta: float) -> str:
    return str(beta).replace(".", "p")


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    print()
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--resume-adapter-file",
        default="outputs/adapters/offline_prefixig_tpo_mixedhop_noisy_qwen_lora/adapters.safetensors",
    )
    parser.add_argument("--betas", default="0.02,0.05,0.10,0.20")
    parser.add_argument("--output-prefix", default="online_prefixig_tpo_anchor_sweep")
    parser.add_argument("--online-iters", type=int, default=5)
    parser.add_argument("--prompts-per-iter", type=int, default=2)
    parser.add_argument("--samples-per-prompt", type=int, default=3)
    parser.add_argument("--updates-per-iter", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--eval-num-examples", type=int, default=5)
    parser.add_argument("--eval-samples-per-example", type=int, default=2)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--action-max-tokens", type=int, default=32)
    parser.add_argument("--answer-max-tokens", type=int, default=24)
    parser.add_argument("--max-search-turns", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    betas = [float(item.strip()) for item in args.betas.split(",") if item.strip()]
    rows = []

    for beta in betas:
        tag = beta_tag(beta)
        adapter_dir = Path(f"outputs/adapters/{args.output_prefix}_beta_{tag}")
        rollout_jsonl = Path(f"outputs/{args.output_prefix}_beta_{tag}_rollouts.jsonl")
        adapter_file = adapter_dir / "adapters.safetensors"

        if not (args.skip_existing and adapter_file.exists()):
            train_cmd = [
                args.python,
                "scripts/online_prefixig_tpo_lora.py",
                "--model",
                args.model,
                "--load-in-4bits",
                "--adapter-path",
                str(adapter_dir),
                "--resume-adapter-file",
                args.resume_adapter_file,
                "--target-method",
                "prefixig_tpo",
                "--eval-regime",
                "mixedhop_noisy",
                "--online-iters",
                str(args.online_iters),
                "--prompts-per-iter",
                str(args.prompts_per_iter),
                "--samples-per-prompt",
                str(args.samples_per_prompt),
                "--updates-per-iter",
                str(args.updates_per_iter),
                "--learning-rate",
                str(args.learning_rate),
                "--anchor-beta",
                str(beta),
                "--max-tokens",
                str(args.max_tokens),
                "--action-max-tokens",
                str(args.action_max_tokens),
                "--answer-max-tokens",
                str(args.answer_max_tokens),
                "--max-search-turns",
                str(args.max_search_turns),
                "--seed",
                str(args.train_seed),
                "--save-every",
                "1",
                "--output-jsonl",
                str(rollout_jsonl),
            ]
            run_command(train_cmd, dry_run=args.dry_run)

        for regime in REGIMES:
            max_turns = "2" if regime == "singlehop" else str(args.max_search_turns)
            eval_jsonl = Path(
                f"outputs/eval_{args.output_prefix}_beta_{tag}_{regime}_"
                f"{args.eval_num_examples}x{args.eval_samples_per_example}_seed{args.seed}.jsonl"
            )
            if not (args.skip_existing and eval_jsonl.exists()):
                eval_cmd = [
                    args.python,
                    "scripts/mlx_generated_policy_diagnostics.py",
                    "--model",
                    args.model,
                    "--load-in-4bits",
                    "--num-examples",
                    str(args.eval_num_examples),
                    "--samples-per-example",
                    str(args.eval_samples_per_example),
                    "--seed",
                    str(args.seed),
                    "--rollout-mode",
                    "flexible",
                    "--eval-regime",
                    regime,
                    "--max-search-turns",
                    max_turns,
                    "--action-max-tokens",
                    "40",
                    "--answer-max-tokens",
                    "32",
                    "--lambda-ig",
                    "0.5",
                    "--tau",
                    "0.7",
                    "--adapter-path",
                    str(adapter_dir),
                    "--output-jsonl",
                    str(eval_jsonl),
                ]
                run_command(eval_cmd, dry_run=args.dry_run)

            if not args.dry_run:
                metrics = original_policy_metrics(read_jsonl(eval_jsonl))
                rows.append(
                    {
                        "beta": beta,
                        "regime": regime,
                        "correct": metrics["correct"],
                        "useful": metrics["useful_correct"],
                        "redundant": metrics["redundant_correct"],
                        "distractor": metrics["distractor_wrong"],
                        "useful_red": metrics["useful_red"],
                        "num_cases": int(metrics["num_cases"]),
                        "num_samples": int(metrics["num_samples"]),
                    }
                )

    if args.dry_run:
        return

    output_csv = Path(f"outputs/{args.output_prefix}_summary.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Anchor beta sweep summary")
    print("=" * 72)
    for row in rows:
        print(
            f"beta={row['beta']:.3f} {row['regime']:<9} "
            f"correct={row['correct']:.3f} useful={row['useful']:.3f} "
            f"red={row['redundant']:.3f} dist={row['distractor']:.3f} "
            f"useful-red={row['useful_red']:+.3f}"
        )
    print(f"\nWrote {output_csv}")


if __name__ == "__main__":
    main()
