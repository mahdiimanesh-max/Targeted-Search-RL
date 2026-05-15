#!/usr/bin/env python3
"""Synthetic PrefixIG-TPO diagnostics.

Runs the smoke-test logic over many controlled examples and reports aggregate
metrics for the first paper-style credit/target table.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prefix_ig_tpo_smoke import (  # noqa: E402
    MLXAnswerLikelihoodScorer,
    MockAnswerLikelihoodScorer,
    Sample,
    compute_diagnostics,
    mean_std,
    softmax,
)
from atgpo_components_smoke import compute_atgpo_segments  # noqa: E402


@dataclass(frozen=True)
class SyntheticCase:
    prompt: str
    answer: str
    samples: list[Sample]


@dataclass(frozen=True)
class ComponentDiagnostics:
    sample_id: str
    proxy_weight: float
    weighted_pg_loss: float
    clip_fraction: float
    sampled_kl: float
    token_advantage: float


def make_case(case_id: int, rng: random.Random) -> SyntheticCase:
    people = [
        ("Laleh Farzan", "Tehran"),
        ("Mira Okafor", "Lagos"),
        ("Jonas Meyer", "Zurich"),
        ("Nadia Karim", "Cairo"),
        ("Sofia Ionescu", "Bucharest"),
        ("Ravi Menon", "Kochi"),
        ("Elena Petrova", "Sofia"),
        ("Amara Diallo", "Dakar"),
    ]
    books = [
        "The River Map",
        "Silent Orchard",
        "Glass Harbor",
        "The Copper Door",
        "Winter Atlas",
        "Blue Meridian",
        "The Lantern Index",
        "Paper Observatory",
    ]
    distractor_cities = [
        "Paris",
        "Berlin",
        "Madrid",
        "Rome",
        "Vienna",
        "Lisbon",
        "Prague",
        "Athens",
    ]

    person, city = people[case_id % len(people)]
    book = books[case_id % len(books)]
    distractor = rng.choice([c for c in distractor_cities if c != city])
    publisher_city = rng.choice([c for c in distractor_cities if c not in {city, distractor}])

    prompt = (
        f"Question: Where was the author of {book} born?\n"
        "Use search if needed. Answer with <answer>\\boxed{...}</answer>."
    )

    old_logps = [-2.4, -2.1, -2.8, -2.6]
    rng.shuffle(old_logps)

    samples = [
        Sample(
            sample_id=f"case{case_id:03d}_useful_correct",
            prompt=prompt,
            answer=city,
            old_logp=old_logps[0],
            completion=(
                "<think>I need the author, then birthplace.</think>\n"
                f"<search>{book} author</search>\n"
                f"<result>Page 1: {book} was written by {person}.</result>\n"
                f"<search>{person} birthplace</search>\n"
                f"<result>Page 1: {person} was born in {city}.</result>\n"
                f"<answer>\\boxed{{{city}}}</answer>"
            ),
        ),
        Sample(
            sample_id=f"case{case_id:03d}_distractor_wrong",
            prompt=prompt,
            answer=city,
            old_logp=old_logps[1],
            completion=(
                "<think>I search but follow a distractor.</think>\n"
                f"<search>{book} publication city</search>\n"
                f"<result>Page 1: {book} was first published in {distractor}.</result>\n"
                f"<answer>\\boxed{{{distractor}}}</answer>"
            ),
        ),
        Sample(
            sample_id=f"case{case_id:03d}_no_search_correct",
            prompt=prompt,
            answer=city,
            old_logp=old_logps[2],
            completion=(
                "<think>I recall the answer directly.</think>\n"
                f"<answer>\\boxed{{{city}}}</answer>"
            ),
        ),
        Sample(
            sample_id=f"case{case_id:03d}_redundant_correct",
            prompt=prompt,
            answer=city,
            old_logp=old_logps[3],
            completion=(
                "<think>I search with an extra redundant turn.</think>\n"
                f"<search>{book} author</search>\n"
                f"<result>Page 1: {book} was written by {person}.</result>\n"
                f"<search>{person} birthplace</search>\n"
                f"<result>Page 1: {person} was born in {city}.</result>\n"
                f"<search>{book} publisher location</search>\n"
                f"<result>Page 1: The publisher of {book} has an office in {publisher_city}.</result>\n"
                f"<answer>\\boxed{{{city}}}</answer>"
            ),
        ),
    ]
    return SyntheticCase(prompt=prompt, answer=city, samples=samples)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def bucket_name(sample_id: str) -> str | None:
    buckets = [
        "useful_correct",
        "distractor_wrong",
        "no_search_correct",
        "redundant_correct",
    ]
    for bucket in buckets:
        if sample_id.endswith(bucket):
            return bucket
    return None


def summarize(all_diags):
    buckets = {
        "useful_correct": [],
        "distractor_wrong": [],
        "no_search_correct": [],
        "redundant_correct": [],
    }
    for diag in all_diags:
        bucket = bucket_name(diag.sample_id)
        if bucket:
            buckets[bucket].append(diag)

    print("Aggregate PrefixIG-TPO synthetic diagnostics")
    print("=" * 52)
    print(f"examples: {len(all_diags) // 4}")
    print(f"samples:  {len(all_diags)}")
    print()

    headers = ["bucket", "q_tpo", "IG", "zIG", "reward", "turns", "curv"]
    rows = []
    for bucket, diags in buckets.items():
        rows.append(
            [
                bucket,
                f"{mean([d.target_weight for d in diags]):.3f} +/- {stdev([d.target_weight for d in diags]):.3f}",
                f"{mean([d.prefix_ig for d in diags]):+.3f}",
                f"{mean([d.normalized_ig for d in diags]):+.3f}",
                f"{mean([d.final_reward for d in diags]):.3f}",
                f"{mean([d.num_turns for d in diags]):.2f}",
                f"{mean([d.curvature for d in diags]):.3f}",
            ]
        )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))

    target_mass_correct = sum(
        d.target_weight
        for d in all_diags
        if d.sample_id.endswith("useful_correct")
        or d.sample_id.endswith("no_search_correct")
        or d.sample_id.endswith("redundant_correct")
    ) / max(len(all_diags) // 4, 1)
    target_mass_search_correct = sum(
        d.target_weight
        for d in all_diags
        if d.sample_id.endswith("useful_correct")
        or d.sample_id.endswith("redundant_correct")
    ) / max(len(all_diags) // 4, 1)
    target_mass_distractor = sum(
        d.target_weight for d in all_diags if d.sample_id.endswith("distractor_wrong")
    ) / max(len(all_diags) // 4, 1)

    print()
    print("Paper-style metrics:")
    print(f"- target_mass_correct:        {target_mass_correct:.3f}")
    print(f"- target_mass_search_correct: {target_mass_search_correct:.3f}")
    print(f"- target_mass_distractor:     {target_mass_distractor:.3f}")
    print(
        "- useful_vs_redundant_gap:   "
        f"{mean([d.target_weight for d in buckets['useful_correct']]) - mean([d.target_weight for d in buckets['redundant_correct']]):+.3f}"
    )


def collect_metrics(all_diags, weight_field: str = "target_weight") -> dict[str, float]:
    buckets = {
        "useful_correct": [],
        "distractor_wrong": [],
        "no_search_correct": [],
        "redundant_correct": [],
    }
    for diag in all_diags:
        bucket = bucket_name(diag.sample_id)
        if bucket:
            buckets[bucket].append(diag)

    num_examples = max(len(all_diags) // 4, 1)
    def weight(diag) -> float:
        return float(getattr(diag, weight_field))

    useful_mass = sum(weight(d) for d in buckets["useful_correct"]) / num_examples
    redundant_mass = sum(weight(d) for d in buckets["redundant_correct"]) / num_examples
    no_search_mass = sum(weight(d) for d in buckets["no_search_correct"]) / num_examples
    distractor_mass = sum(weight(d) for d in buckets["distractor_wrong"]) / num_examples

    return {
        "q_useful": useful_mass,
        "q_redundant": redundant_mass,
        "q_no_search": no_search_mass,
        "q_distractor": distractor_mass,
        "target_mass_correct": useful_mass + redundant_mass + no_search_mass,
        "target_mass_search_correct": useful_mass + redundant_mass,
        "useful_vs_redundant_gap": useful_mass - redundant_mass,
        "target_mass_distractor": distractor_mass,
    }


def print_baseline_comparison(results: list[tuple[str, dict[str, float]]]) -> None:
    print("Baseline comparison")
    print("=" * 52)
    headers = [
        "method",
        "q_useful",
        "q_redundant",
        "q_no_search",
        "q_distractor",
        "correct",
        "search_correct",
        "useful-red",
    ]
    rows = []
    for name, metrics in results:
        rows.append(
            [
                name,
                f"{metrics['q_useful']:.3f}",
                f"{metrics['q_redundant']:.3f}",
                f"{metrics['q_no_search']:.3f}",
                f"{metrics['q_distractor']:.3f}",
                f"{metrics['target_mass_correct']:.3f}",
                f"{metrics['target_mass_search_correct']:.3f}",
                f"{metrics['useful_vs_redundant_gap']:+.3f}",
            ]
        )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))

    print()
    print("Reading guide:")
    print("- OriginalPolicy is the pre-update sampled policy mass, before constructing a TPO target.")
    print("- FinalReward-TPO rewards all correct answers similarly.")
    print("- PrefixIG-GRPO is an A-TGPO-style scalar-advantage one-step proxy, not a TPO target.")
    print("- A-TGPO-components uses token-level advantage broadcasting, clipping, and KL accounting.")
    print("- PrefixIG-TPO should move mass from no-search/redundant/distractor toward useful search.")
    print("- Curvature is a trust variant; it should sharpen or stabilize, not redefine the signal.")


def grpo_one_step_proxy(diags, step_scale: float):
    """Convert scalar utilities into a GRPO/A-TGPO-style one-step mass proxy.

    GRPO does not explicitly construct a target distribution. For diagnostics,
    we show the local direction induced by scalar advantages:

        A_i = normalize(U_i within prompt group)
        proxy_i ∝ p_old_i * exp(step_scale * A_i)

    This is only a comparable visualization of update pressure, not the exact
    clipped PPO/GRPO loss.
    """
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


def atgpo_component_one_step_proxy(
    diags,
    samples: list[Sample],
    scorer,
    step_scale: float,
    alpha: float,
    gamma: float,
    clip_low: float,
    clip_high: float,
    dynamic_clip: bool,
    sim_step: float,
    token_logprob_scorer=None,
) -> tuple[list, list[ComponentDiagnostics]]:
    """Run no-training A-TGPO components and make a comparable mass proxy.

    The component path produces token-level losses, not a TPO target
    distribution. For the side-by-side table, we convert clipped token update
    pressure into a local proxy:

        proxy_i proportional to p_old_i * exp(step_scale * normalize(-loss_i))

    This keeps the table comparable while preserving the actual component
    diagnostics separately.
    """
    sample_scores = []
    component_rows = []
    for sample in samples:
        _, _, segments = compute_atgpo_segments(
            sample=sample,
            scorer=scorer,
            alpha=alpha,
            gamma=gamma,
            clip_low=clip_low,
            clip_high=clip_high,
            dynamic_clip=dynamic_clip,
            sim_step=sim_step,
            token_logprob_scorer=token_logprob_scorer,
        )
        total_tokens = sum(segment.token_count for segment in segments)
        weighted_loss = sum(
            segment.pg_loss * segment.token_count for segment in segments
        ) / max(total_tokens, 1)
        clipfrac = sum(
            segment.clip_fraction * segment.token_count for segment in segments
        ) / max(total_tokens, 1)
        sampled_kl = sum(
            segment.sampled_kl * segment.token_count for segment in segments
        ) / max(total_tokens, 1)
        token_advantage = sum(
            segment.advantage * segment.token_count for segment in segments
        ) / max(total_tokens, 1)
        sample_scores.append(-weighted_loss)
        component_rows.append(
            ComponentDiagnostics(
                sample_id=sample.sample_id,
                proxy_weight=0.0,
                weighted_pg_loss=weighted_loss,
                clip_fraction=clipfrac,
                sampled_kl=sampled_kl,
                token_advantage=token_advantage,
            )
        )

    score_mean, score_std = mean_std(sample_scores)
    normalized_scores = [(score - score_mean) / score_std for score in sample_scores]
    proxy_logits = [
        math_log(max(diag.old_policy_prob, 1e-12)) + step_scale * score
        for diag, score in zip(diags, normalized_scores)
    ]
    proxy_weights = softmax(proxy_logits)
    proxy_diags = [
        replace(diag, target_weight=proxy_weight)
        for diag, proxy_weight in zip(diags, proxy_weights)
    ]
    component_rows = [
        replace(row, proxy_weight=proxy_weight)
        for row, proxy_weight in zip(component_rows, proxy_weights)
    ]
    return proxy_diags, component_rows


def print_component_diagnostics(rows: list[ComponentDiagnostics]) -> None:
    buckets = {
        "useful_correct": [],
        "distractor_wrong": [],
        "no_search_correct": [],
        "redundant_correct": [],
    }
    for row in rows:
        bucket = bucket_name(row.sample_id)
        if bucket:
            buckets[bucket].append(row)

    print()
    print("A-TGPO component diagnostics")
    print("=" * 52)
    headers = ["bucket", "proxy_q", "pg_loss", "clipfrac", "kl", "tok_adv"]
    table_rows = []
    for bucket, bucket_rows in buckets.items():
        table_rows.append(
            [
                bucket,
                f"{mean([row.proxy_weight for row in bucket_rows]):.3f}",
                f"{mean([row.weighted_pg_loss for row in bucket_rows]):+.3f}",
                f"{mean([row.clip_fraction for row in bucket_rows]):.3f}",
                f"{mean([row.sampled_kl for row in bucket_rows]):+.3f}",
                f"{mean([row.token_advantage for row in bucket_rows]):+.3f}",
            ]
        )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in table_rows))
        for col in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def math_log(value: float) -> float:
    import math

    return math.log(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default=None, help="Optional MLX model path.")
    parser.add_argument("--lambda-ig", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--curvature-eps", type=float, default=1e-3)
    parser.add_argument("--apply-curvature", action="store_true")
    parser.add_argument(
        "--grpo-step-scale",
        type=float,
        default=1.0,
        help="Diagnostic step scale for the PrefixIG-GRPO/A-TGPO-style one-step proxy.",
    )
    parser.add_argument("--atgpo-alpha", type=float, default=0.3)
    parser.add_argument("--atgpo-gamma", type=float, default=1.0)
    parser.add_argument("--atgpo-clip-low", type=float, default=3e-3)
    parser.add_argument("--atgpo-clip-high", type=float, default=4e-3)
    parser.add_argument("--atgpo-no-dynamic-clip", action="store_true")
    parser.add_argument(
        "--atgpo-sim-step",
        type=float,
        default=0.05,
        help="Virtual log-ratio step for no-training A-TGPO component diagnostics.",
    )
    parser.add_argument(
        "--compare-baselines",
        action="store_true",
        help="Report FinalReward-TPO, PrefixIG-TPO, and PrefixIG-TPO+curvature side by side.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = random.Random(args.seed)
    scorer = (
        MLXAnswerLikelihoodScorer(args.model)
        if args.model
        else MockAnswerLikelihoodScorer()
    )

    cases = [make_case(case_id, rng) for case_id in range(args.num_examples)]

    if args.compare_baselines:
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
            for case in cases:
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
                    )
                    all_component_rows.extend(component_rows)
                all_diags.extend(case_diags)
            results.append((name, collect_metrics(all_diags, weight_field=weight_field)))
        print_baseline_comparison(results)
        if all_component_rows:
            print_component_diagnostics(all_component_rows)
        return

    all_diags = []
    for case in cases:
        all_diags.extend(
            compute_diagnostics(
                samples=case.samples,
                scorer=scorer,
                lambda_ig=args.lambda_ig,
                tau=args.tau,
                curvature_eps=args.curvature_eps,
                apply_curvature=args.apply_curvature,
            )
        )
    summarize(all_diags)


if __name__ == "__main__":
    main()
