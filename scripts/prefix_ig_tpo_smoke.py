#!/usr/bin/env python3
"""Smoke test for PrefixIG-TPO target construction.

This script validates the first research brick before any trainer changes:

1. Parse <search>/<result>/<answer> blocks from sampled completions.
2. Score final answers with exact match / token F1.
3. Compute prefix information gain for each search turn.
4. Aggregate PrefixIG into a trajectory utility.
5. Build TPO target weights and cheap curvature diagnostics.

By default it uses a deterministic mock answer-likelihood scorer so the smoke
test runs instantly. Pass --model to use an MLX model through mlx-lm.
"""

from __future__ import annotations

import argparse
import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol


SEARCH_RESULT_RE = re.compile(
    r"<search>(?P<query>.*?)</search>\s*<result>(?P<result>.*?)</result>",
    flags=re.DOTALL,
)
ANSWER_RE = re.compile(r"<answer>(?P<answer>.*?)</answer>", flags=re.DOTALL)


@dataclass(frozen=True)
class SearchTurn:
    query: str
    result: str
    start: int
    end: int


@dataclass(frozen=True)
class Sample:
    sample_id: str
    prompt: str
    completion: str
    answer: str
    old_logp: float = 0.0


@dataclass(frozen=True)
class SampleDiagnostics:
    sample_id: str
    final_reward: float
    exact_match: float
    token_f1: float
    prefix_ig: float
    normalized_ig: float
    utility: float
    old_policy_prob: float
    target_weight: float
    curvature: float
    num_turns: int
    predicted_answer: str
    turn_igs: tuple[float, ...]


class AnswerLikelihoodScorer(Protocol):
    def score(self, context: str, answer: str) -> float:
        """Return log-like score for answer under context."""


class MockAnswerLikelihoodScorer:
    """Cheap deterministic scorer for smoke tests.

    It is not a model. It simply makes contexts containing answer tokens score
    higher, which is enough to validate PrefixIG and TPO target plumbing.
    """

    def score(self, context: str, answer: str) -> float:
        context_norm = normalize_text(context)
        answer_norm = normalize_text(answer)
        answer_terms = answer_norm.split()
        if not answer_terms:
            return -10.0

        context_terms = set(context_norm.split())
        matched = sum(1 for term in answer_terms if term in context_terms)
        coverage = matched / max(len(answer_terms), 1)
        exact_bonus = 2.0 if answer_norm and answer_norm in context_norm else 0.0
        result_bonus = 0.15 * context_norm.count("page ")
        return -8.0 + 4.0 * coverage + exact_bonus + result_bonus


class MLXAnswerLikelihoodScorer:
    """MLX answer log-likelihood scorer.

    This is intentionally lightweight and imported lazily so the default smoke
    path does not require MLX.
    """

    def __init__(self, model_path: str):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load

        self.mx = mx
        self.nn = nn
        self.model, self.tokenizer = load(model_path)
        self.model.eval()

    def _encode(self, text: str) -> list[int]:
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            return self.tokenizer.encode(text)

    def score(self, context: str, answer: str) -> float:
        mx = self.mx
        nn = self.nn
        prefix = context.rstrip() + "\nAnswer: "
        prefix_ids = self._encode(prefix)
        answer_ids = self._encode(answer)
        if not prefix_ids or not answer_ids:
            return float("-inf")

        ids = prefix_ids + answer_ids
        if len(ids) < 2:
            return float("-inf")

        input_ids = mx.array([ids], dtype=mx.int32)
        logits = self.model(input_ids[:, :-1]).astype(mx.float32)
        targets = input_ids[:, 1:]
        log_probs = nn.log_softmax(logits, axis=-1)
        selected = mx.take_along_axis(
            log_probs, mx.expand_dims(targets, axis=-1), axis=-1
        ).squeeze(-1)

        start = max(len(prefix_ids) - 1, 0)
        end = start + len(answer_ids)
        answer_logps = selected[0, start:end]
        mean_logp = answer_logps.mean()
        mx.eval(mean_logp)
        return float(mean_logp.item())


def normalize_text(text: str) -> str:
    text = text.lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, references: Iterable[str]) -> float:
    pred_tokens = normalize_text(prediction).split()
    if not pred_tokens:
        return 0.0

    best = 0.0
    for reference in references:
        ref_tokens = normalize_text(reference).split()
        if not ref_tokens:
            continue
        overlap = Counter(pred_tokens) & Counter(ref_tokens)
        same = sum(overlap.values())
        if same == 0:
            continue
        precision = same / len(pred_tokens)
        recall = same / len(ref_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def exact_match(prediction: str, references: Iterable[str]) -> float:
    pred = normalize_text(prediction)
    return float(any(pred == normalize_text(ref) for ref in references))


def extract_boxed(text: str) -> str | None:
    idx = text.rfind(r"\boxed")
    if idx < 0:
        return None
    brace_start = text.find("{", idx)
    if brace_start < 0:
        return None
    depth = 0
    for pos in range(brace_start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : pos].strip()
    return None


def extract_answer(completion: str) -> str:
    matches = list(ANSWER_RE.finditer(completion))
    if not matches:
        return ""
    answer_text = matches[-1].group("answer").strip()
    boxed = extract_boxed(answer_text)
    return boxed if boxed is not None else answer_text


def parse_search_turns(completion: str) -> list[SearchTurn]:
    turns = []
    for match in SEARCH_RESULT_RE.finditer(completion):
        turns.append(
            SearchTurn(
                query=match.group("query").strip(),
                result=match.group("result").strip(),
                start=match.start(),
                end=match.end(),
            )
        )
    return turns


def final_reward(prediction: str, references: Iterable[str]) -> tuple[float, float, float]:
    refs = list(references)
    em = exact_match(prediction, refs)
    f1 = token_f1(prediction, refs)
    return em, em, f1


def prefix_information_gain(
    scorer: AnswerLikelihoodScorer,
    prompt: str,
    completion: str,
    answer: str,
) -> tuple[float, tuple[float, ...]]:
    turns = parse_search_turns(completion)
    previous_score = scorer.score(prompt, answer)
    turn_igs: list[float] = []

    for turn in turns:
        context = prompt + "\n" + completion[: turn.end]
        current_score = scorer.score(context, answer)
        turn_igs.append(current_score - previous_score)
        previous_score = current_score

    if not turn_igs:
        return 0.0, tuple()

    # A-TGPO-like v1d aggregate for the first turn: cumulative IG divided by
    # sqrt(number of terms). This is stable for trajectories with different
    # numbers of search turns.
    aggregate = sum(turn_igs) / math.sqrt(len(turn_igs))
    return aggregate, tuple(turn_igs)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(var)
    return mean, std if std > 1e-8 else 1.0


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    temperature = max(temperature, 1e-8)
    scaled = [value / temperature for value in values]
    max_value = max(scaled)
    exps = [math.exp(value - max_value) for value in scaled]
    denom = sum(exps)
    return [value / denom for value in exps]


def compute_diagnostics(
    samples: list[Sample],
    scorer: AnswerLikelihoodScorer,
    lambda_ig: float,
    tau: float,
    curvature_eps: float,
    apply_curvature: bool,
) -> list[SampleDiagnostics]:
    raw = []
    prefix_igs = []
    for sample in samples:
        predicted = extract_answer(sample.completion)
        reward, em, f1 = final_reward(predicted, [sample.answer])
        prefix_ig, turn_igs = prefix_information_gain(
            scorer=scorer,
            prompt=sample.prompt,
            completion=sample.completion,
            answer=sample.answer,
        )
        prefix_igs.append(prefix_ig)
        raw.append((sample, predicted, reward, em, f1, prefix_ig, turn_igs))

    ig_mean, ig_std = mean_std(prefix_igs)
    old_policy_probs = softmax([item[0].old_logp for item in raw])
    curvatures = [prob * (1.0 - prob) for prob in old_policy_probs]

    utilities = []
    enriched = []
    for idx, (sample, predicted, reward, em, f1, prefix_ig, turn_igs) in enumerate(raw):
        normalized_ig = (prefix_ig - ig_mean) / ig_std
        ig_term = lambda_ig * normalized_ig
        if apply_curvature:
            ig_term = ig_term / math.sqrt(curvatures[idx] + curvature_eps)
        utility = reward + ig_term
        utilities.append(utility)
        enriched.append((sample, predicted, reward, em, f1, prefix_ig, normalized_ig, utility, turn_igs))

    target_logits = [
        sample.old_logp + utility / max(tau, 1e-8)
        for (sample, *_), utility in zip(raw, utilities)
    ]
    target_weights = softmax(target_logits)

    diagnostics = []
    for idx, item in enumerate(enriched):
        sample, predicted, reward, em, f1, prefix_ig, normalized_ig, utility, turn_igs = item
        diagnostics.append(
            SampleDiagnostics(
                sample_id=sample.sample_id,
                final_reward=reward,
                exact_match=em,
                token_f1=f1,
                prefix_ig=prefix_ig,
                normalized_ig=normalized_ig,
                utility=utility,
                old_policy_prob=old_policy_probs[idx],
                target_weight=target_weights[idx],
                curvature=curvatures[idx],
                num_turns=len(turn_igs),
                predicted_answer=predicted,
                turn_igs=turn_igs,
            )
        )
    return diagnostics


def demo_samples() -> list[Sample]:
    prompt = (
        "Question: Where was the author of The River Map born?\n"
        "Use search if needed. Answer with <answer>\\boxed{...}</answer>."
    )
    answer = "Tehran"
    return [
        Sample(
            sample_id="useful_correct",
            prompt=prompt,
            answer=answer,
            old_logp=-2.4,
            completion=(
                "<think>I need the author, then birthplace.</think>\n"
                "<search>The River Map author</search>\n"
                "<result>Page 1: The River Map was written by Laleh Farzan.</result>\n"
                "<think>Now find Laleh Farzan birthplace.</think>\n"
                "<search>Laleh Farzan birthplace</search>\n"
                "<result>Page 1: Laleh Farzan was born in Tehran.</result>\n"
                "<answer>\\boxed{Tehran}</answer>"
            ),
        ),
        Sample(
            sample_id="distractor_wrong",
            prompt=prompt,
            answer=answer,
            old_logp=-2.1,
            completion=(
                "<think>I will search but follow the wrong entity.</think>\n"
                "<search>The River Map publication city</search>\n"
                "<result>Page 1: The River Map was first published in Paris.</result>\n"
                "<answer>\\boxed{Paris}</answer>"
            ),
        ),
        Sample(
            sample_id="no_search_correct",
            prompt=prompt,
            answer=answer,
            old_logp=-2.8,
            completion=(
                "<think>I recall the answer directly.</think>\n"
                "<answer>\\boxed{Tehran}</answer>"
            ),
        ),
        Sample(
            sample_id="redundant_correct",
            prompt=prompt,
            answer=answer,
            old_logp=-2.6,
            completion=(
                "<think>I search repeatedly.</think>\n"
                "<search>The River Map author</search>\n"
                "<result>Page 1: The River Map was written by Laleh Farzan.</result>\n"
                "<search>Laleh Farzan birthplace</search>\n"
                "<result>Page 1: Laleh Farzan was born in Tehran.</result>\n"
                "<search>Tehran country</search>\n"
                "<result>Page 1: Tehran is the capital of Iran.</result>\n"
                "<answer>\\boxed{Tehran}</answer>"
            ),
        ),
    ]


def format_turn_igs(values: tuple[float, ...]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(f"{value:+.3f}" for value in values) + "]"


def print_report(diagnostics: list[SampleDiagnostics]) -> None:
    headers = [
        "sample",
        "ans",
        "R",
        "IG",
        "zIG",
        "U",
        "p_old",
        "q_tpo",
        "curv",
        "turns",
        "turn_IGs",
    ]
    rows = []
    for diag in diagnostics:
        rows.append(
            [
                diag.sample_id,
                diag.predicted_answer or "<none>",
                f"{diag.final_reward:.2f}",
                f"{diag.prefix_ig:+.3f}",
                f"{diag.normalized_ig:+.3f}",
                f"{diag.utility:+.3f}",
                f"{diag.old_policy_prob:.3f}",
                f"{diag.target_weight:.3f}",
                f"{diag.curvature:.3f}",
                str(diag.num_turns),
                format_turn_igs(diag.turn_igs),
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

    print("\nSanity checks:")
    print(f"- Target weights sum: {sum(diag.target_weight for diag in diagnostics):.6f}")
    best = max(diagnostics, key=lambda item: item.target_weight)
    print(f"- Highest target mass: {best.sample_id} ({best.target_weight:.3f})")
    print("- PrefixIG is a training-time signal; inference cost is unchanged.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Optional MLX/MLX-LM model path. If omitted, use mock scorer.",
    )
    parser.add_argument(
        "--lambda-ig",
        type=float,
        default=0.3,
        help="Weight on normalized PrefixIG in utility.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Temperature for TPO target construction.",
    )
    parser.add_argument(
        "--curvature-eps",
        type=float,
        default=1e-3,
        help="Epsilon for curvature-aware trust diagnostics.",
    )
    parser.add_argument(
        "--apply-curvature",
        action="store_true",
        help="Apply curvature scaling to the IG term. Default only reports curvature.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    scorer: AnswerLikelihoodScorer
    if args.model:
        scorer = MLXAnswerLikelihoodScorer(args.model)
    else:
        scorer = MockAnswerLikelihoodScorer()

    diagnostics = compute_diagnostics(
        samples=demo_samples(),
        scorer=scorer,
        lambda_ig=args.lambda_ig,
        tau=args.tau,
        curvature_eps=args.curvature_eps,
        apply_curvature=args.apply_curvature,
    )
    print_report(diagnostics)


if __name__ == "__main__":
    main()
