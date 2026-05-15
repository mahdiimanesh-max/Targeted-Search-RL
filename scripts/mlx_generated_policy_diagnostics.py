#!/usr/bin/env python3
"""Generated-policy PrefixIG/TPO diagnostics with MLX.

This is the first step beyond controlled hand-written trajectories:

1. Build synthetic QA cases with a small search index.
2. Ask a real MLX model to generate K trajectories per prompt.
3. Parse <search>/<result>/<answer> structure from generated text.
4. Score generated trajectories with real model log-probs.
5. Compare OriginalPolicy, reward-only TPO, PrefixIG-GRPO, PrefixIG-TPO, and
   curvature-aware PrefixIG-TPO over generated trajectory buckets.

No model weights are updated.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import replace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mlx_real_policy_diagnostics import (  # noqa: E402
    MLXPolicyScorer,
    grpo_one_step_proxy,
)
from prefix_ig_tpo_smoke import (  # noqa: E402
    Sample,
    compute_diagnostics,
    exact_match,
    extract_boxed,
    extract_answer,
    mean_std,
    parse_search_turns,
    softmax,
)
from prefix_ig_tpo_synthetic import (  # noqa: E402
    atgpo_component_one_step_proxy,
    make_case,
)


RESULT_RE = re.compile(r"<result>(?P<result>.*?)</result>", flags=re.DOTALL)


def unique_results(completions: list[str]) -> list[str]:
    seen = set()
    results = []
    for completion in completions:
        for match in RESULT_RE.finditer(completion):
            result = " ".join(match.group("result").split())
            if result not in seen:
                seen.add(result)
                results.append(result)
    return results


def build_generation_prompt(question_prompt: str, search_index: list[str]) -> str:
    docs = "\n".join(f"- {doc}" for doc in search_index)
    return (
        "Generate exactly one machine-readable trajectory for this QA task.\n"
        "Use only the facts in the Search Index.\n\n"
        "You MUST follow this grammar:\n"
        "<search>short query</search>\n"
        "<result>one copied fact from the Search Index</result>\n"
        "<search>short query</search>\n"
        "<result>one copied fact from the Search Index</result>\n"
        "<answer>\\boxed{final city only}</answer>\n\n"
        "Rules:\n"
        "- Use exactly two search turns.\n"
        "- First search for the book author.\n"
        "- Second search for that author's birthplace.\n"
        "- Copy result facts exactly from the Search Index.\n"
        "- The answer must be only the birthplace city.\n"
        "- Do not write bullets, explanations, labels, markdown, or text after </answer>.\n\n"
        f"{question_prompt}\n\n"
        "Search Index:\n"
        f"{docs}\n\n"
        "Now output the trajectory only:\n"
    )


def build_tool_loop_prompt(question_prompt: str) -> str:
    return (
        "Answer the QA task using search. At each step, output exactly one action.\n\n"
        "Allowed actions:\n"
        "<search>short query</search>\n"
        "<answer>\\boxed{final city only}</answer>\n\n"
        "Rules:\n"
        "- First search for the book author.\n"
        "- Then search for that author's birthplace.\n"
        "- After enough evidence, answer with only the birthplace city.\n"
        "- Do not write bullets, markdown, explanations, or text after </answer>.\n\n"
        f"{question_prompt}\n\n"
        "Begin:\n"
    )


def build_flexible_tool_loop_prompt(question_prompt: str, max_search_turns: int) -> str:
    return (
        "Answer the QA task using as few search actions as needed. At each step, "
        "output exactly one action.\n\n"
        "Allowed actions:\n"
        "<search>short query</search>\n"
        "<answer>\\boxed{final city only}</answer>\n\n"
        "Rules:\n"
        "- You may answer immediately if you already know the answer.\n"
        f"- Otherwise, use up to {max_search_turns} search actions.\n"
        "- Good search strategy: find the book author, then the author's birthplace.\n"
        "- Stop searching once you have enough evidence.\n"
        "- The answer must be only the birthplace city.\n"
        "- Do not write bullets, markdown, explanations, or text after </answer>.\n\n"
        f"{question_prompt}\n\n"
        "Begin:\n"
    )


def format_prompt(scorer: MLXPolicyScorer, user_prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return user_prompt
    tokenizer = scorer.tokenizer
    if not hasattr(tokenizer, "apply_chat_template"):
        return user_prompt
    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def generate_completion(
    scorer: MLXPolicyScorer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
) -> str:
    from mlx_lm.generate import generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(
        temp=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
    )
    text = generate(
        scorer.model,
        scorer.tokenizer,
        prompt,
        verbose=False,
        max_tokens=max_tokens,
        sampler=sampler,
    )
    if "</answer>" in text:
        text = text[: text.find("</answer>") + len("</answer>")]
    return text.strip()


def trim_at_first_marker(text: str, markers: list[str]) -> str:
    best = None
    for marker in markers:
        pos = text.find(marker)
        if pos >= 0:
            end = pos + len(marker)
            if best is None or end < best:
                best = end
    return text[:best].strip() if best is not None else text.strip()


def keep_from_first_action_tag(text: str) -> str:
    positions = [pos for pos in [text.find("<search>"), text.find("<answer>")] if pos >= 0]
    if not positions:
        return text.strip()
    return text[min(positions) :].strip()


def complete_forced_answer(raw_suffix: str) -> str:
    suffix = trim_at_first_marker(raw_suffix, ["</answer>", "}"])
    suffix = suffix.strip()
    if "</answer>" in suffix:
        return f"<answer>\\boxed{{{suffix}"
    if suffix.endswith("}"):
        return f"<answer>\\boxed{{{suffix}</answer>"
    city = re.split(r"[\n<]", suffix, maxsplit=1)[0].strip()
    city = city.strip(" .,:;")
    return f"<answer>\\boxed{{{city}}}</answer>"


def lexical_result(query: str, search_index: list[str]) -> str:
    query_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", query.lower())
        if term not in {"the", "a", "an", "of", "in", "was", "is", "where"}
    }
    best_doc = search_index[0] if search_index else "Page 1: No result found."
    best_score = -1
    for doc in search_index:
        doc_terms = set(re.findall(r"[a-z0-9]+", doc.lower()))
        score = len(query_terms & doc_terms)
        if "birth" in query.lower() and "born" in doc.lower():
            score += 3
        if "author" in query.lower() and "written by" in doc.lower():
            score += 3
        if score > best_score:
            best_score = score
            best_doc = doc
    return best_doc


def extract_search_query(text: str) -> str | None:
    match = re.search(r"<search>(.*?)</search>", text, flags=re.DOTALL)
    if not match:
        return None
    return " ".join(match.group(1).split())


def rollout_with_oracle_search(
    scorer: MLXPolicyScorer,
    prompt: str,
    search_index: list[str],
    max_turns: int,
    action_max_tokens: int,
    answer_max_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
) -> str:
    completion = ""
    for _turn in range(max_turns):
        action = generate_completion(
            scorer=scorer,
            prompt=prompt + completion,
            max_tokens=action_max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
        )
        action = keep_from_first_action_tag(action)
        action = trim_at_first_marker(action, ["</search>", "</answer>"])
        completion += action.strip() + "\n"
        if "</answer>" in action:
            return completion.strip()

        query = extract_search_query(action)
        if not query:
            completion += "<search>book author</search>\n"
            query = "book author"
        result = lexical_result(query, search_index)
        completion += f"<result>{result}</result>\n"

    answer_prompt = (
        prompt
        + completion
        + "Use the observed <result> facts. Complete the final city only:\n"
        "<answer>\\boxed{"
    )
    answer = generate_completion(
        scorer=scorer,
        prompt=answer_prompt,
        max_tokens=answer_max_tokens,
        temperature=0.0,
        top_p=1.0,
        min_p=0.0,
        top_k=0,
    )
    completion += complete_forced_answer(answer)
    return completion.strip()


def classify_generated_sample(sample, diag) -> str:
    prediction = extract_answer(sample.completion)
    if not prediction:
        prediction = extract_boxed(sample.completion) or ""
    correct = exact_match(prediction, [sample.answer]) > 0.0
    turns = parse_search_turns(sample.completion)

    if correct and not turns:
        return "no_search_correct"
    if correct and turns:
        if len(turns) > 2:
            return "redundant_correct"
        if diag.prefix_ig > 0.0:
            return "useful_correct"
        return "redundant_correct"
    if (not correct) and turns:
        return "distractor_wrong"
    return "other_wrong"


def normalized_query_terms(query: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "author",
        "birth",
        "birthplace",
        "born",
        "by",
        "for",
        "in",
        "is",
        "of",
        "the",
        "to",
        "was",
        "were",
        "what",
        "where",
        "who",
    }
    return {
        term
        for term in re.findall(r"[a-z0-9]+", query.lower())
        if term not in stopwords
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def redundancy_penalty(
    sample: Sample,
    diag,
    optimal_turns: int,
    repeat_threshold: float,
    min_turn_ig: float,
    lambda_extra_turn: float,
    lambda_repeat_query: float,
    lambda_low_ig: float,
) -> float:
    turns = parse_search_turns(sample.completion)
    extra_turns = max(len(turns) - optimal_turns, 0)

    repeated_queries = 0
    seen_queries: list[set[str]] = []
    for turn in turns:
        terms = normalized_query_terms(turn.query)
        if any(jaccard(terms, prev_terms) >= repeat_threshold for prev_terms in seen_queries):
            repeated_queries += 1
        seen_queries.append(terms)

    repeated_results = 0
    seen_results: set[str] = set()
    for turn in turns:
        result = " ".join(turn.result.lower().split())
        if result in seen_results:
            repeated_results += 1
        seen_results.add(result)

    low_gain_turns = sum(1 for ig in diag.turn_igs if ig <= min_turn_ig)
    return (
        lambda_extra_turn * extra_turns
        + lambda_repeat_query * max(repeated_queries, repeated_results)
        + lambda_low_ig * low_gain_turns
    )


def apply_efficiency_penalty(
    diags,
    samples: list[Sample],
    tau: float,
    apply_curvature: bool,
    reward_gated: bool,
    curvature_eps: float,
    optimal_turns: int,
    repeat_threshold: float,
    min_turn_ig: float,
    lambda_extra_turn: float,
    lambda_repeat_query: float,
    lambda_low_ig: float,
):
    penalties = [
        redundancy_penalty(
            sample=sample,
            diag=diag,
            optimal_turns=optimal_turns,
            repeat_threshold=repeat_threshold,
            min_turn_ig=min_turn_ig,
            lambda_extra_turn=lambda_extra_turn,
            lambda_repeat_query=lambda_repeat_query,
            lambda_low_ig=lambda_low_ig,
        )
        for sample, diag in zip(samples, diags)
    ]
    utilities = []
    if reward_gated:
        for diag, penalty in zip(diags, penalties):
            penalty_term = diag.final_reward * penalty
            if apply_curvature:
                penalty_term = penalty_term / math.sqrt(diag.curvature + curvature_eps)
            utilities.append(diag.utility - penalty_term)
    else:
        penalty_mean, penalty_std = mean_std(penalties)
        normalized_penalties = [
            (penalty - penalty_mean) / penalty_std for penalty in penalties
        ]
        for diag, normalized_penalty in zip(diags, normalized_penalties):
            penalty_term = normalized_penalty
            if apply_curvature:
                penalty_term = penalty_term / math.sqrt(diag.curvature + curvature_eps)
            utilities.append(diag.utility - penalty_term)

    target_weights = softmax(
        [
            sample.old_logp + utility / max(tau, 1e-8)
            for sample, utility in zip(samples, utilities)
        ]
    )
    return [
        replace(diag, utility=utility, target_weight=target_weight)
        for diag, utility, target_weight in zip(diags, utilities, target_weights)
    ]


def generated_bucket_metrics(all_rows, weight_field: str) -> dict[str, float]:
    buckets = [
        "useful_correct",
        "redundant_correct",
        "no_search_correct",
        "distractor_wrong",
        "other_wrong",
    ]
    totals = {bucket: 0.0 for bucket in buckets}
    case_ids = set()
    for case_id, bucket, diag, _sample in all_rows:
        case_ids.add(case_id)
        totals[bucket] += float(getattr(diag, weight_field))

    denom = max(len(case_ids), 1)
    averaged = {bucket: value / denom for bucket, value in totals.items()}
    averaged["target_mass_correct"] = (
        averaged["useful_correct"]
        + averaged["redundant_correct"]
        + averaged["no_search_correct"]
    )
    averaged["target_mass_search_correct"] = (
        averaged["useful_correct"] + averaged["redundant_correct"]
    )
    averaged["useful_vs_redundant_gap"] = (
        averaged["useful_correct"] - averaged["redundant_correct"]
    )
    return averaged


def print_generated_comparison(results: list[tuple[str, dict[str, float]]]) -> None:
    print("Generated-policy comparison")
    print("=" * 64)
    headers = [
        "method",
        "useful",
        "redundant",
        "no_search",
        "distractor",
        "other",
        "correct",
        "useful-red",
    ]
    rows = []
    for name, metrics in results:
        rows.append(
            [
                name,
                f"{metrics['useful_correct']:.3f}",
                f"{metrics['redundant_correct']:.3f}",
                f"{metrics['no_search_correct']:.3f}",
                f"{metrics['distractor_wrong']:.3f}",
                f"{metrics['other_wrong']:.3f}",
                f"{metrics['target_mass_correct']:.3f}",
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


def print_bucket_counts(rows) -> None:
    counts = {
        "useful_correct": 0,
        "redundant_correct": 0,
        "no_search_correct": 0,
        "distractor_wrong": 0,
        "other_wrong": 0,
    }
    for _case_id, bucket, _diag, _sample in rows:
        counts[bucket] += 1
    print()
    print("Generated sample counts")
    print("=" * 64)
    for bucket, count in counts.items():
        print(f"{bucket}: {count}")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


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
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--samples-per-example", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--action-max-tokens", type=int, default=64)
    parser.add_argument("--answer-max-tokens", type=int, default=48)
    parser.add_argument("--max-search-turns", type=int, default=2)
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
        default="guided",
        help="guided encourages exactly two searches; flexible lets the model stop or keep searching.",
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
    parser.add_argument(
        "--output-jsonl",
        default="outputs/generated_policy_diagnostics.jsonl",
    )
    parser.add_argument("--show-samples", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = random.Random(args.seed)
    scorer = MLXPolicyScorer(
        args.model,
        adapter_path=args.adapter_path,
        load_in_4bits=args.load_in_4bits,
    )

    case_rows = []
    jsonl_records = []
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
        generated_samples = []

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
                    sample_id=f"case{case_idx:03d}_gen{sample_idx:02d}",
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
        for diag, sample in zip(base_diags, generated_samples):
            bucket = classify_generated_sample(sample, diag)
            case_rows.append((case_idx, bucket, diag, sample))
            jsonl_records.append(
                {
                    "case_id": case_idx,
                    "sample_id": sample.sample_id,
                    "bucket": bucket,
                    "answer": case.answer,
                    "prediction": extract_answer(sample.completion)
                    or extract_boxed(sample.completion)
                    or "",
                    "old_logp": sample.old_logp,
                    "prefix_ig": diag.prefix_ig,
                    "turn_igs": list(diag.turn_igs),
                    "completion": sample.completion,
                }
            )

    configs = [
        ("OriginalPolicy", "old", 0.0, False, "old_policy_prob"),
        ("FinalReward-TPO", "tpo", 0.0, False, "target_weight"),
        ("PrefixIG-GRPO", "grpo", args.lambda_ig, False, "target_weight"),
        ("A-TGPO-components", "atgpo", args.lambda_ig, False, "target_weight"),
        ("PrefixIG-TPO", "tpo", args.lambda_ig, False, "target_weight"),
        ("PrefixIG-TPO+eff", "eff_tpo", args.lambda_ig, False, "target_weight"),
        ("PrefixIG-TPO+rg_eff", "rg_eff_tpo", args.lambda_ig, False, "target_weight"),
        ("PrefixIG-TPO+curv", "tpo", args.lambda_ig, True, "target_weight"),
        ("PrefixIG-TPO+curv+eff", "eff_tpo", args.lambda_ig, True, "target_weight"),
        ("PrefixIG-TPO+curv+rg_eff", "rg_eff_tpo", args.lambda_ig, True, "target_weight"),
    ]

    results = []
    for name, mode, lambda_ig, apply_curvature, weight_field in configs:
        method_rows = []
        for case_idx in range(args.num_examples):
            samples = [row[3] for row in case_rows if row[0] == case_idx]
            case_diags = compute_diagnostics(
                samples=samples,
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
                case_diags, _component_rows = atgpo_component_one_step_proxy(
                    diags=case_diags,
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
            if mode in {"eff_tpo", "rg_eff_tpo"}:
                case_diags = apply_efficiency_penalty(
                    diags=case_diags,
                    samples=samples,
                    tau=args.tau,
                    apply_curvature=apply_curvature,
                    reward_gated=mode == "rg_eff_tpo",
                    curvature_eps=args.curvature_eps,
                    optimal_turns=args.eff_optimal_turns,
                    repeat_threshold=args.eff_repeat_threshold,
                    min_turn_ig=args.eff_min_turn_ig,
                    lambda_extra_turn=args.eff_lambda_extra_turn,
                    lambda_repeat_query=args.eff_lambda_repeat_query,
                    lambda_low_ig=args.eff_lambda_low_ig,
                )
            for diag, base_row in zip(case_diags, [row for row in case_rows if row[0] == case_idx]):
                method_rows.append((case_idx, base_row[1], diag, base_row[3]))
        results.append((name, generated_bucket_metrics(method_rows, weight_field)))

    print_generated_comparison(results)
    print_bucket_counts(case_rows)
    write_jsonl(Path(args.output_jsonl), jsonl_records)
    print()
    print(f"Wrote generated samples to {args.output_jsonl}")

    if args.show_samples:
        print()
        print("Sample generated completions")
        print("=" * 64)
        for record in jsonl_records[: min(4, len(jsonl_records))]:
            print(f"\n{record['sample_id']} [{record['bucket']}]")
            print(record["completion"])


if __name__ == "__main__":
    main()
