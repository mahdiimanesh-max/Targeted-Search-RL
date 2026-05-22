#!/usr/bin/env python3
"""Evaluate MLX policies on HotpotQA-mini with a local lexical retriever.

This is the first "real-world mini retrieval QA" experiment in the repo. It
uses actual HotpotQA questions/passages, but keeps the retrieval and evaluation
small enough for a 16GB Apple Silicon laptop:

1. Load `outputs/hotpotqa_mini/eval.jsonl` and `corpus.jsonl`.
2. Let the model interact through <search> and <answer> actions.
3. Retrieve passages from the local corpus after each search.
4. Measure answer correctness, support-document coverage, distractor failures,
   and useful-vs-redundant search behavior.

No model weights are updated.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mlx_generated_policy_diagnostics import (  # noqa: E402
    apply_efficiency_penalty,
    format_prompt,
    generate_completion,
    grpo_one_step_proxy,
    keep_from_first_action_tag,
    trim_at_first_marker,
)
from mlx_real_policy_diagnostics import MLXPolicyScorer  # noqa: E402
from prefix_ig_tpo_smoke import (  # noqa: E402
    Sample,
    compute_diagnostics,
    exact_match,
    extract_answer,
    extract_boxed,
    mean_std,
    normalize_text,
    parse_search_turns,
    softmax,
    token_f1,
)
from prefix_ig_tpo_synthetic import atgpo_component_one_step_proxy  # noqa: E402


SEARCH_RE = re.compile(r"<search>(.*?)</search>", flags=re.DOTALL)
TITLE_RE = re.compile(r"^Title:\s*(.*?)\s*$", flags=re.MULTILINE)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "with",
}


@dataclass(frozen=True)
class RetrievedDoc:
    doc_id: str
    title: str
    text: str
    score: float


@dataclass(frozen=True)
class EvalSample:
    case_id: int
    sample_id: str
    question_id: str
    question: str
    references: tuple[str, ...]
    support_titles: tuple[str, ...]
    prompt: str
    completion: str
    prediction: str
    exact: float
    f1: float
    correct: bool
    bucket: str
    retrieved_titles: tuple[str, ...]
    support_hits: tuple[str, ...]
    old_logp: float
    prefix_ig: float
    turn_igs: tuple[float, ...]


class MiniBM25:
    def __init__(self, docs: list[dict[str, Any]], k1: float = 1.2, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_terms: list[Counter[str]] = []
        self.title_terms: list[set[str]] = []
        self.doc_lens: list[int] = []
        df: Counter[str] = Counter()

        for doc in docs:
            title_tokens = tokenize(doc.get("title", ""))
            text_tokens = tokenize(f"{doc.get('title', '')} {doc.get('text', '')}")
            counts = Counter(text_tokens)
            self.doc_terms.append(counts)
            self.title_terms.append(set(title_tokens))
            self.doc_lens.append(sum(counts.values()))
            df.update(counts.keys())

        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        n_docs = max(len(docs), 1)
        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int) -> list[RetrievedDoc]:
        query_terms = tokenize(query)
        if not query_terms:
            query_terms = tokenize(" ".join(query.split()[:8]))
        query_counts = Counter(query_terms)
        scored: list[RetrievedDoc] = []
        for idx, doc in enumerate(self.docs):
            score = 0.0
            counts = self.doc_terms[idx]
            doc_len = max(self.doc_lens[idx], 1)
            for term, qf in query_counts.items():
                tf = counts.get(term, 0)
                if tf <= 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / max(self.avgdl, 1e-8))
                score += qf * idf * (tf * (self.k1 + 1.0)) / denom
                if term in self.title_terms[idx]:
                    score += 0.35 * idf
            if score > 0.0:
                scored.append(
                    RetrievedDoc(
                        doc_id=str(doc.get("id", idx)),
                        title=str(doc.get("title", "")),
                        text=str(doc.get("text", "")),
                        score=score,
                    )
                )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def normalize_title(title: str) -> str:
    return " ".join(html.unescape(str(title)).lower().split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_prompt(
    record: dict[str, Any],
    max_search_turns: int,
    skill_text: str | None = None,
) -> str:
    skill_block = ""
    if skill_text:
        skill_block = (
            "Search strategy skills:\n"
            f"{skill_text.strip()}\n\n"
            "Apply the relevant skills silently while choosing search queries.\n\n"
        )
    return (
        "Answer the HotpotQA question using the local search tool. At each step, "
        "output exactly one action.\n\n"
        f"{skill_block}"
        "Allowed actions:\n"
        "<search>short query</search>\n"
        "<answer>\\boxed{final answer only}</answer>\n\n"
        "Rules:\n"
        "- Use search when evidence is needed.\n"
        f"- Use at most {max_search_turns} search actions.\n"
        "- Search for specific entities, titles, or relations from the question.\n"
        "- Stop searching once the retrieved evidence is sufficient.\n"
        "- The final answer must be short and inside <answer>\\boxed{...}</answer>.\n"
        "- Do not write bullets, markdown, explanations, or text after </answer>.\n\n"
        f"Question: {record['question']}\n\n"
        "Begin:\n"
    )


def extract_search_query(text: str) -> str | None:
    match = SEARCH_RE.search(text)
    if not match:
        return None
    query = " ".join(match.group(1).split())
    return query or None


def result_block(docs: list[RetrievedDoc], max_doc_chars: int) -> str:
    if not docs:
        return "<result>No local documents retrieved.</result>"
    parts = []
    for idx, doc in enumerate(docs, start=1):
        text = " ".join(doc.text.split())
        if len(text) > max_doc_chars:
            text = text[:max_doc_chars].rsplit(" ", 1)[0].strip()
        parts.append(
            f"[{idx}]\n"
            f"Title: {doc.title}\n"
            f"Text: {text}\n"
            f"Score: {doc.score:.3f}"
        )
    return "<result>\n" + "\n---\n".join(parts) + "\n</result>"


def complete_forced_answer(raw_suffix: str) -> str:
    suffix = trim_at_first_marker(raw_suffix, ["</answer>"]).strip()
    suffix = suffix.replace("<answer>", "").replace(r"\boxed{", "").strip()
    if "}" in suffix:
        suffix = suffix[: suffix.find("}")]
    suffix = re.split(r"[\n<]", suffix, maxsplit=1)[0].strip()
    suffix = suffix.strip(" .,:;\"'")
    return f"<answer>\\boxed{{{suffix}}}</answer>"


def clear_mlx_cache(scorer: MLXPolicyScorer) -> None:
    mx = getattr(scorer, "mx", None)
    if mx is not None and hasattr(mx, "clear_cache"):
        mx.clear_cache()


def has_complete_answer(action: str) -> bool:
    return "<answer>" in action and "</answer>" in action


def rollout_with_local_search(
    scorer: MLXPolicyScorer,
    prompt: str,
    retriever: MiniBM25,
    fallback_query: str,
    max_search_turns: int,
    top_k: int,
    max_doc_chars: int,
    action_max_tokens: int,
    answer_max_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k_sampling: int,
) -> str:
    completion = ""
    for _turn in range(max_search_turns):
        action = generate_completion(
            scorer=scorer,
            prompt=prompt + completion,
            max_tokens=action_max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k_sampling,
        )
        action = keep_from_first_action_tag(action)
        action = trim_at_first_marker(action, ["</search>", "</answer>"])
        if not action:
            action = f"<search>{fallback_query}</search>"
        if has_complete_answer(action):
            completion += action.strip() + "\n"
            return completion.strip()
        if "</answer>" in action and "<answer>" not in action:
            # A bare closing tag is a malformed stop, not a usable answer.
            break

        query = extract_search_query(action)
        if not query:
            query = fallback_query
            action = f"<search>{query}</search>"
        completion += action.strip() + "\n"
        docs = retriever.search(query, top_k=top_k)
        completion += result_block(docs, max_doc_chars=max_doc_chars) + "\n"

    answer_prompt = (
        prompt
        + completion
        + "Use the retrieved evidence. Complete the final answer only:\n"
        "<answer>\\boxed{"
    )
    answer_suffix = generate_completion(
        scorer=scorer,
        prompt=answer_prompt,
        max_tokens=answer_max_tokens,
        temperature=0.0,
        top_p=1.0,
        min_p=0.0,
        top_k=0,
    )
    completion += complete_forced_answer(answer_suffix)
    return completion.strip()


def retrieved_titles_from_completion(completion: str) -> tuple[str, ...]:
    titles = []
    for match in TITLE_RE.finditer(completion):
        title = " ".join(match.group(1).split())
        if title:
            titles.append(title)
    return tuple(titles)


def repeated_search_count(completion: str) -> int:
    turns = parse_search_turns(completion)
    seen: list[set[str]] = []
    repeats = 0
    for turn in turns:
        terms = set(tokenize(turn.query))
        if any(terms and len(terms & prev) / max(len(terms | prev), 1) >= 0.7 for prev in seen):
            repeats += 1
        seen.append(terms)
    return repeats


def classify_sample(
    completion: str,
    prediction: str,
    references: list[str],
    support_titles: list[str],
    correct_f1_threshold: float,
    useful_support_coverage: float,
    allow_reference_substring: bool,
) -> tuple[str, float, float, bool, tuple[str, ...], tuple[str, ...]]:
    exact = exact_match(prediction, references)
    f1 = token_f1(prediction, references)
    pred_norm = normalize_text(prediction)
    contains_reference = any(
        ref_norm and ref_norm in pred_norm
        for ref_norm in (normalize_text(reference) for reference in references)
    )
    correct = bool(
        exact > 0.0
        or f1 >= correct_f1_threshold
        or (allow_reference_substring and contains_reference)
    )
    turns = parse_search_turns(completion)
    retrieved_titles = retrieved_titles_from_completion(completion)

    support_set = {normalize_title(title) for title in support_titles if title}
    retrieved_set = {normalize_title(title) for title in retrieved_titles if title}
    support_hits = tuple(
        title for title in support_titles if normalize_title(title) in retrieved_set
    )
    required_support = max(len(support_set), 1)
    support_coverage = len({normalize_title(title) for title in support_hits}) / required_support
    has_repeats = repeated_search_count(completion) > 0

    if correct and not turns:
        return "no_search_correct", exact, f1, correct, retrieved_titles, support_hits
    if correct and turns:
        if support_coverage >= useful_support_coverage and not has_repeats:
            return "useful_correct", exact, f1, correct, retrieved_titles, support_hits
        return "redundant_correct", exact, f1, correct, retrieved_titles, support_hits
    if (not correct) and turns:
        return "distractor_wrong", exact, f1, correct, retrieved_titles, support_hits
    return "other_wrong", exact, f1, correct, retrieved_titles, support_hits


def empirical_metrics(samples: list[EvalSample]) -> dict[str, float]:
    buckets = Counter(sample.bucket for sample in samples)
    n = max(len(samples), 1)
    useful = buckets["useful_correct"] / n
    redundant = buckets["redundant_correct"] / n
    no_search = buckets["no_search_correct"] / n
    distractor = buckets["distractor_wrong"] / n
    other = buckets["other_wrong"] / n
    return {
        "useful_correct": useful,
        "redundant_correct": redundant,
        "no_search_correct": no_search,
        "distractor_wrong": distractor,
        "other_wrong": other,
        "target_mass_correct": sum(1 for sample in samples if sample.correct) / n,
        "target_mass_search_correct": useful + redundant,
        "useful_vs_redundant_gap": useful - redundant,
        "exact_match": sum(sample.exact for sample in samples) / n,
        "token_f1": sum(sample.f1 for sample in samples) / n,
        "support_coverage": (
            sum(len(set(map(normalize_title, sample.support_hits))) / max(len(sample.support_titles), 1) for sample in samples)
            / n
        ),
    }


def weighted_bucket_metrics(rows, weight_field: str) -> dict[str, float]:
    buckets = [
        "useful_correct",
        "redundant_correct",
        "no_search_correct",
        "distractor_wrong",
        "other_wrong",
    ]
    totals = {bucket: 0.0 for bucket in buckets}
    case_ids = set()
    for case_id, bucket, diag, _sample in rows:
        case_ids.add(case_id)
        totals[bucket] += float(getattr(diag, weight_field))

    denom = max(len(case_ids), 1)
    averaged = {bucket: value / denom for bucket, value in totals.items()}
    averaged["target_mass_correct"] = (
        averaged["useful_correct"]
        + averaged["redundant_correct"]
        + averaged["no_search_correct"]
    )
    averaged["useful_vs_redundant_gap"] = (
        averaged["useful_correct"] - averaged["redundant_correct"]
    )
    return averaged


def print_metrics_table(title: str, results: list[tuple[str, dict[str, float]]]) -> None:
    print(title)
    print("=" * 72)
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
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def target_policy_comparison(
    eval_samples: list[EvalSample],
    scorer: MLXPolicyScorer,
    tau: float,
    lambda_ig: float,
    curvature_eps: float,
    grpo_step_scale: float,
    atgpo_alpha: float,
    atgpo_gamma: float,
    atgpo_clip_low: float,
    atgpo_clip_high: float,
    atgpo_sim_step: float,
    eff_optimal_turns: int,
    eff_repeat_threshold: float,
    eff_min_turn_ig: float,
    eff_lambda_extra_turn: float,
    eff_lambda_repeat_query: float,
    eff_lambda_low_ig: float,
) -> list[tuple[str, dict[str, float]]]:
    by_case: dict[int, list[EvalSample]] = defaultdict(list)
    for sample in eval_samples:
        by_case[sample.case_id].append(sample)

    configs = [
        ("OriginalPolicy", "old", 0.0, False, "old_policy_prob"),
        ("FinalReward-TPO", "tpo", 0.0, False, "target_weight"),
        ("PrefixIG-GRPO", "grpo", lambda_ig, False, "target_weight"),
        ("A-TGPO-components", "atgpo", lambda_ig, False, "target_weight"),
        ("PrefixIG-TPO", "tpo", lambda_ig, False, "target_weight"),
        ("PrefixIG-TPO+rg_eff", "rg_eff_tpo", lambda_ig, False, "target_weight"),
        ("PrefixIG-TPO+curv", "tpo", lambda_ig, True, "target_weight"),
        ("PrefixIG-TPO+curv+rg_eff", "rg_eff_tpo", lambda_ig, True, "target_weight"),
    ]

    results = []
    for name, mode, method_lambda_ig, apply_curvature, weight_field in configs:
        method_rows = []
        for case_id, case_samples in by_case.items():
            samples = [
                Sample(
                    sample_id=item.sample_id,
                    prompt=item.prompt,
                    completion=item.completion,
                    answer=item.references[0],
                    old_logp=item.old_logp,
                )
                for item in case_samples
            ]
            diags = compute_diagnostics(
                samples=samples,
                scorer=scorer,
                lambda_ig=method_lambda_ig,
                tau=tau,
                curvature_eps=curvature_eps,
                apply_curvature=apply_curvature,
            )
            if mode == "grpo":
                diags = grpo_one_step_proxy(diags, step_scale=grpo_step_scale)
            if mode == "atgpo":
                diags, _component_rows = atgpo_component_one_step_proxy(
                    diags=diags,
                    samples=samples,
                    scorer=scorer,
                    step_scale=grpo_step_scale,
                    alpha=atgpo_alpha,
                    gamma=atgpo_gamma,
                    clip_low=atgpo_clip_low,
                    clip_high=atgpo_clip_high,
                    dynamic_clip=True,
                    sim_step=atgpo_sim_step,
                    token_logprob_scorer=scorer,
                )
            if mode == "rg_eff_tpo":
                diags = apply_efficiency_penalty(
                    diags=diags,
                    samples=samples,
                    tau=tau,
                    apply_curvature=apply_curvature,
                    reward_gated=True,
                    curvature_eps=curvature_eps,
                    optimal_turns=eff_optimal_turns,
                    repeat_threshold=eff_repeat_threshold,
                    min_turn_ig=eff_min_turn_ig,
                    lambda_extra_turn=eff_lambda_extra_turn,
                    lambda_repeat_query=eff_lambda_repeat_query,
                    lambda_low_ig=eff_lambda_low_ig,
                )
            for eval_sample, diag, sample in zip(case_samples, diags, samples):
                method_rows.append((case_id, eval_sample.bucket, diag, sample))
            clear_mlx_cache(scorer)
        results.append((name, weighted_bucket_metrics(method_rows, weight_field)))
        clear_mlx_cache(scorer)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--load-in-4bits", action="store_true")
    parser.add_argument("--eval-jsonl", default="outputs/hotpotqa_mini/eval.jsonl")
    parser.add_argument("--corpus-jsonl", default="outputs/hotpotqa_mini/corpus.jsonl")
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--samples-per-example", type=int, default=2)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-search-turns", type=int, default=3)
    parser.add_argument("--retrieval-top-k", type=int, default=2)
    parser.add_argument("--max-doc-chars", type=int, default=700)
    parser.add_argument("--action-max-tokens", type=int, default=48)
    parser.add_argument("--answer-max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument(
        "--skill-prompt-file",
        action="append",
        default=[],
        help=(
            "Optional SDAR-style skill markdown file to inject into the prompt. "
            "Can be passed multiple times. Useful for skill-conditioned "
            "diagnostics without running the full SDAR trainer."
        ),
    )
    parser.add_argument(
        "--skill-max-chars",
        type=int,
        default=1800,
        help="Maximum characters of concatenated skill text to inject.",
    )
    parser.add_argument("--correct-f1-threshold", type=float, default=0.8)
    parser.add_argument(
        "--allow-reference-substring",
        action="store_true",
        help=(
            "Count a prediction as correct when a normalized gold short answer "
            "appears inside a longer generated answer. Useful for diagnostic "
            "runs where the model answers in a sentence despite instructions."
        ),
    )
    parser.add_argument("--useful-support-coverage", type=float, default=1.0)
    parser.add_argument("--lambda-ig", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--curvature-eps", type=float, default=1e-3)
    parser.add_argument("--grpo-step-scale", type=float, default=1.0)
    parser.add_argument("--atgpo-alpha", type=float, default=0.3)
    parser.add_argument("--atgpo-gamma", type=float, default=1.0)
    parser.add_argument("--atgpo-clip-low", type=float, default=3e-3)
    parser.add_argument("--atgpo-clip-high", type=float, default=4e-3)
    parser.add_argument("--atgpo-sim-step", type=float, default=0.05)
    parser.add_argument("--eff-optimal-turns", type=int, default=2)
    parser.add_argument("--eff-repeat-threshold", type=float, default=0.55)
    parser.add_argument("--eff-min-turn-ig", type=float, default=0.05)
    parser.add_argument("--eff-lambda-extra-turn", type=float, default=0.6)
    parser.add_argument("--eff-lambda-repeat-query", type=float, default=0.4)
    parser.add_argument("--eff-lambda-low-ig", type=float, default=0.0)
    parser.add_argument("--target-comparison", action="store_true")
    parser.add_argument(
        "--no-sample-diagnostics",
        action="store_true",
        help=(
            "Skip per-sample old-logp/PrefixIG scoring during empirical evaluation. "
            "This is much lighter and is recommended for memory-constrained runs. "
            "Ignored when --target-comparison is used."
        ),
    )
    parser.add_argument("--output-jsonl", default="outputs/eval_hotpotqa_mini.jsonl")
    parser.add_argument("--show-samples", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = random.Random(args.seed)
    eval_records = read_jsonl(Path(args.eval_jsonl))
    corpus = read_jsonl(Path(args.corpus_jsonl))
    if args.num_examples >= 0:
        eval_records = rng.sample(eval_records, min(args.num_examples, len(eval_records)))

    skill_text = ""
    if args.skill_prompt_file:
        chunks = []
        for path in args.skill_prompt_file:
            chunks.append(Path(path).read_text(encoding="utf-8").strip())
        skill_text = "\n\n".join(chunks)
        if args.skill_max_chars > 0 and len(skill_text) > args.skill_max_chars:
            skill_text = skill_text[: args.skill_max_chars].rsplit("\n", 1)[0].strip()

    retriever = MiniBM25(corpus)
    scorer = MLXPolicyScorer(
        args.model,
        adapter_path=args.adapter_path,
        load_in_4bits=args.load_in_4bits,
    )

    eval_samples: list[EvalSample] = []
    output_records: list[dict[str, Any]] = []

    for case_id, record in enumerate(eval_records):
        user_prompt = build_prompt(
            record,
            max_search_turns=args.max_search_turns,
            skill_text=skill_text,
        )
        prompt = format_prompt(
            scorer=scorer,
            user_prompt=user_prompt,
            use_chat_template=not args.no_chat_template,
        )
        references = [str(item) for item in record.get("golden_answers") or [record["answer"]]]
        support_titles = [str(item) for item in record.get("supporting_titles", [])]

        for sample_idx in range(args.samples_per_example):
            completion = rollout_with_local_search(
                scorer=scorer,
                prompt=prompt,
                retriever=retriever,
                fallback_query=record["question"],
                max_search_turns=args.max_search_turns,
                top_k=args.retrieval_top_k,
                max_doc_chars=args.max_doc_chars,
                action_max_tokens=args.action_max_tokens,
                answer_max_tokens=args.answer_max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                min_p=args.min_p,
                top_k_sampling=args.top_k,
            )
            prediction = extract_answer(completion) or extract_boxed(completion) or ""
            bucket, exact, f1, correct, retrieved_titles, support_hits = classify_sample(
                completion=completion,
                prediction=prediction,
                references=references,
                support_titles=support_titles,
                correct_f1_threshold=args.correct_f1_threshold,
                useful_support_coverage=args.useful_support_coverage,
                allow_reference_substring=args.allow_reference_substring,
            )
            if args.no_sample_diagnostics and not args.target_comparison:
                old_logp = 0.0
                prefix_ig = 0.0
                turn_igs = tuple()
            else:
                old_logp = scorer.completion_mean_logp(prompt, completion)
                diag = compute_diagnostics(
                    samples=[
                        Sample(
                            sample_id=f"case{case_id:03d}_gen{sample_idx:02d}",
                            prompt=prompt,
                            completion=completion,
                            answer=references[0],
                            old_logp=old_logp,
                        )
                    ],
                    scorer=scorer,
                    lambda_ig=args.lambda_ig,
                    tau=args.tau,
                    curvature_eps=args.curvature_eps,
                    apply_curvature=False,
                )[0]
                prefix_ig = diag.prefix_ig
                turn_igs = tuple(diag.turn_igs)
            clear_mlx_cache(scorer)
            eval_sample = EvalSample(
                case_id=case_id,
                sample_id=f"case{case_id:03d}_gen{sample_idx:02d}",
                question_id=str(record.get("id", case_id)),
                question=record["question"],
                references=tuple(references),
                support_titles=tuple(support_titles),
                prompt=prompt,
                completion=completion,
                prediction=prediction,
                exact=exact,
                f1=f1,
                correct=correct,
                bucket=bucket,
                retrieved_titles=retrieved_titles,
                support_hits=support_hits,
                old_logp=old_logp,
                prefix_ig=prefix_ig,
                turn_igs=turn_igs,
            )
            eval_samples.append(eval_sample)
            output_records.append(
                {
                    "case_id": case_id,
                    "sample_id": eval_sample.sample_id,
                    "question_id": eval_sample.question_id,
                    "question": eval_sample.question,
                    "references": list(eval_sample.references),
                    "prediction": eval_sample.prediction,
                    "exact": eval_sample.exact,
                    "f1": eval_sample.f1,
                    "correct": eval_sample.correct,
                    "bucket": eval_sample.bucket,
                    "support_titles": list(eval_sample.support_titles),
                    "retrieved_titles": list(eval_sample.retrieved_titles),
                    "support_hits": list(eval_sample.support_hits),
                    "old_logp": eval_sample.old_logp,
                    "prefix_ig": eval_sample.prefix_ig,
                    "turn_igs": list(eval_sample.turn_igs),
                    "completion": eval_sample.completion,
                }
            )
        clear_mlx_cache(scorer)

    metrics = empirical_metrics(eval_samples)
    print_metrics_table("HotpotQA-mini empirical behavior", [("GeneratedPolicy", metrics)])
    print()
    print("Additional metrics")
    print("=" * 72)
    print(f"exact_match:      {metrics['exact_match']:.3f}")
    print(f"token_f1:         {metrics['token_f1']:.3f}")
    print(f"support_coverage: {metrics['support_coverage']:.3f}")
    print(f"samples:          {len(eval_samples)}")

    if args.target_comparison:
        print()
        target_results = target_policy_comparison(
            eval_samples=eval_samples,
            scorer=scorer,
            tau=args.tau,
            lambda_ig=args.lambda_ig,
            curvature_eps=args.curvature_eps,
            grpo_step_scale=args.grpo_step_scale,
            atgpo_alpha=args.atgpo_alpha,
            atgpo_gamma=args.atgpo_gamma,
            atgpo_clip_low=args.atgpo_clip_low,
            atgpo_clip_high=args.atgpo_clip_high,
            atgpo_sim_step=args.atgpo_sim_step,
            eff_optimal_turns=args.eff_optimal_turns,
            eff_repeat_threshold=args.eff_repeat_threshold,
            eff_min_turn_ig=args.eff_min_turn_ig,
            eff_lambda_extra_turn=args.eff_lambda_extra_turn,
            eff_lambda_repeat_query=args.eff_lambda_repeat_query,
            eff_lambda_low_ig=args.eff_lambda_low_ig,
        )
        print_metrics_table("HotpotQA-mini target-policy comparison", target_results)

    write_jsonl(Path(args.output_jsonl), output_records)
    print()
    print(f"Wrote generated samples to {args.output_jsonl}")

    if args.show_samples:
        print()
        print("Sample completions")
        print("=" * 72)
        for record in output_records[: min(3, len(output_records))]:
            print(f"\n{record['sample_id']} [{record['bucket']}]")
            print(f"Q: {record['question']}")
            print(f"Refs: {record['references']}")
            print(f"Pred: {record['prediction']}")
            print(record["completion"])


if __name__ == "__main__":
    main()
