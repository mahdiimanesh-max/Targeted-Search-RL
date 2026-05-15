#!/usr/bin/env python3
"""A-TGPO component smoke test without training.

This reproduces the accounting path of an A-TGPO-style update:

1. Parse search/result turn boundaries.
2. Compute PrefixIG per turn.
3. Build turn-level advantages from final reward + normalized PrefixIG.
4. Broadcast turn advantages to token-level records.
5. Compute old/current/reference token log-probs.
6. Compute adaptive clip scales.
7. Evaluate a PPO/GRPO-style clipped objective diagnostic.

No model is updated. Token log-probs are deterministic placeholders that mimic
the accounting interface; they can later be replaced with MLX/model scores.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prefix_ig_tpo_smoke import (  # noqa: E402
    MockAnswerLikelihoodScorer,
    Sample,
    demo_samples,
    exact_match,
    extract_answer,
    parse_search_turns,
    prefix_information_gain,
    token_f1,
)


TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class TokenRecord:
    index: int
    token: str
    old_logp: float
    current_logp: float
    ref_logp: float
    advantage: float
    ratio: float
    clipped_ratio: float
    pg_loss: float
    clip_fraction: float
    sampled_kl: float


@dataclass(frozen=True)
class TurnSegment:
    name: str
    text: str
    raw_ig: float
    norm_ig: float
    advantage: float
    clip_scale: float
    token_records: tuple[TokenRecord, ...]
    token_count: int
    old_logp: float
    current_logp: float
    ref_logp: float
    ratio: float
    clipped_ratio: float
    pg_loss: float
    clip_fraction: float
    sampled_kl: float


def whitespace_tokens(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    return tokens or ["<empty>"]


def stable_unit_interval(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return value / float(2**64 - 1)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(var)
    if std <= 1e-8:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def clipped_pg_loss(
    advantage: float,
    ratio: float,
    clip_low: float,
    clip_high: float,
) -> tuple[float, float, float]:
    lower = 1.0 - clip_low
    upper = 1.0 + clip_high
    clipped_ratio = min(max(ratio, lower), upper)
    loss_unclipped = -advantage * ratio
    loss_clipped = -advantage * clipped_ratio
    pg_loss = max(loss_unclipped, loss_clipped)
    clip_fraction = float(loss_clipped > loss_unclipped)
    return clipped_ratio, pg_loss, clip_fraction


def mock_token_logps(
    token: str,
    token_index: int,
    advantage: float,
    sim_step: float,
) -> tuple[float, float, float]:
    """Return deterministic old/current/reference log-probs for diagnostics."""
    unit = stable_unit_interval(f"{token}:{token_index}")
    length_penalty = 0.035 * min(len(token), 12)
    old_logp = -1.35 - length_penalty - 0.25 * unit
    ref_logp = old_logp - 0.015 + 0.03 * (unit - 0.5)

    # This is a virtual step, not a trained update. It only lets us exercise the
    # same ratio/clipping math that real A-TGPO would use after scoring a model.
    token_scale = 0.85 + 0.30 * unit
    log_ratio = sim_step * math.tanh(advantage) * token_scale
    current_logp = old_logp + max(min(log_ratio, 1.0), -1.0)
    return old_logp, current_logp, ref_logp


def build_token_records(
    text: str,
    start_index: int,
    advantage: float,
    clip_low: float,
    clip_high: float,
    sim_step: float,
    prefix_text: str | None = None,
    token_logprob_scorer=None,
) -> tuple[TokenRecord, ...]:
    records: list[TokenRecord] = []
    if token_logprob_scorer is not None and prefix_text is not None:
        old_logps, tokens = token_logprob_scorer.token_logprobs(prefix_text, text)
    else:
        old_logps = []
        tokens = []

    if not tokens:
        tokens = whitespace_tokens(text)
        old_logps = []

    for offset, token in enumerate(tokens):
        token_index = start_index + offset
        if offset < len(old_logps):
            old_logp = old_logps[offset]
            ref_logp = old_logp
            unit = stable_unit_interval(f"{token}:{token_index}")
            token_scale = 0.85 + 0.30 * unit
            log_ratio = sim_step * math.tanh(advantage) * token_scale
            current_logp = old_logp + max(min(log_ratio, 1.0), -1.0)
        else:
            old_logp, current_logp, ref_logp = mock_token_logps(
                token=token,
                token_index=token_index,
                advantage=advantage,
                sim_step=sim_step,
            )
        ratio = math.exp(current_logp - old_logp)
        clipped_ratio, pg_loss, clip_fraction = clipped_pg_loss(
            advantage=advantage,
            ratio=ratio,
            clip_low=clip_low,
            clip_high=clip_high,
        )
        records.append(
            TokenRecord(
                index=token_index,
                token=token,
                old_logp=old_logp,
                current_logp=current_logp,
                ref_logp=ref_logp,
                advantage=advantage,
                ratio=ratio,
                clipped_ratio=clipped_ratio,
                pg_loss=pg_loss,
                clip_fraction=clip_fraction,
                sampled_kl=current_logp - ref_logp,
            )
        )
    return tuple(records)


def summarize_segment(
    name: str,
    text: str,
    raw_ig: float,
    norm_ig: float,
    advantage: float,
    clip_scale: float,
    token_records: tuple[TokenRecord, ...],
) -> TurnSegment:
    return TurnSegment(
        name=name,
        text=text,
        raw_ig=raw_ig,
        norm_ig=norm_ig,
        advantage=advantage,
        clip_scale=clip_scale,
        token_records=token_records,
        token_count=len(token_records),
        old_logp=mean([record.old_logp for record in token_records]),
        current_logp=mean([record.current_logp for record in token_records]),
        ref_logp=mean([record.ref_logp for record in token_records]),
        ratio=mean([record.ratio for record in token_records]),
        clipped_ratio=mean([record.clipped_ratio for record in token_records]),
        pg_loss=mean([record.pg_loss for record in token_records]),
        clip_fraction=mean([record.clip_fraction for record in token_records]),
        sampled_kl=mean([record.sampled_kl for record in token_records]),
    )


def compute_atgpo_segments(
    sample: Sample,
    scorer,
    alpha: float,
    gamma: float,
    clip_low: float,
    clip_high: float,
    dynamic_clip: bool,
    sim_step: float,
    token_logprob_scorer=None,
) -> tuple[float, float, list[TurnSegment]]:
    prediction = extract_answer(sample.completion)
    em = exact_match(prediction, [sample.answer])
    f1 = token_f1(prediction, [sample.answer])
    final_reward = em

    _, raw_turn_igs_tuple = prefix_information_gain(
        scorer=scorer,
        prompt=sample.prompt,
        completion=sample.completion,
        answer=sample.answer,
    )
    raw_turn_igs = list(raw_turn_igs_tuple)
    norm_igs = normalize(raw_turn_igs)
    turns = parse_search_turns(sample.completion)

    segments: list[TurnSegment] = []
    prev_end = 0
    token_index = 0
    n_turns = len(turns)

    for idx, turn in enumerate(turns):
        segment_start = prev_end
        text = sample.completion[prev_end : turn.end]
        prev_end = turn.end

        discounted = 0.0
        n_terms = 0
        for j in range(idx, n_turns):
            discounted += (gamma ** (j - idx)) * norm_igs[j]
            n_terms += 1
        discounted = discounted / math.sqrt(max(n_terms, 1))

        advantage = final_reward + alpha * discounted
        norm_ig = norm_igs[idx]
        raw_ig = raw_turn_igs[idx]
        clip_scale = 1.0
        if dynamic_clip:
            clip_scale = 1.0 + 0.3 * (2.0 * sigmoid(norm_ig) - 1.0)

        token_records = build_token_records(
            text=text,
            start_index=token_index,
            advantage=advantage,
            clip_low=clip_low * clip_scale,
            clip_high=clip_high * clip_scale,
            sim_step=sim_step,
            prefix_text=sample.prompt + sample.completion[:segment_start],
            token_logprob_scorer=token_logprob_scorer,
        )
        token_index += len(token_records)
        segments.append(
            summarize_segment(
                name=f"search_turn_{idx + 1}",
                text=text,
                raw_ig=raw_ig,
                norm_ig=norm_ig,
                advantage=advantage,
                clip_scale=clip_scale,
                token_records=token_records,
            )
        )

    outcome_text = sample.completion[prev_end:]
    outcome_advantage = final_reward
    token_records = build_token_records(
        text=outcome_text,
        start_index=token_index,
        advantage=outcome_advantage,
        clip_low=clip_low,
        clip_high=clip_high,
        sim_step=sim_step,
        prefix_text=sample.prompt + sample.completion[:prev_end],
        token_logprob_scorer=token_logprob_scorer,
    )
    segments.append(
        summarize_segment(
            name="outcome",
            text=outcome_text,
            raw_ig=0.0,
            norm_ig=0.0,
            advantage=outcome_advantage,
            clip_scale=1.0,
            token_records=token_records,
        )
    )
    return em, f1, segments


def print_token_records(segment: TurnSegment, max_tokens: int) -> None:
    headers = ["i", "token", "old_lp", "cur_lp", "ref_lp", "ratio", "clamp", "loss", "kl"]
    rows = []
    for record in segment.token_records[:max_tokens]:
        token = record.token
        if len(token) > 18:
            token = token[:15] + "..."
        rows.append(
            [
                str(record.index),
                token,
                f"{record.old_logp:+.3f}",
                f"{record.current_logp:+.3f}",
                f"{record.ref_logp:+.3f}",
                f"{record.ratio:.3f}",
                f"{record.clipped_ratio:.3f}",
                f"{record.pg_loss:+.3f}",
                f"{record.sampled_kl:+.3f}",
            ]
        )
    if len(segment.token_records) > max_tokens:
        rows.append(["...", f"+{len(segment.token_records) - max_tokens} more", "", "", "", "", "", "", ""])
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    print(f"  tokens for {segment.name}:")
    print("  " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  " + "-+-".join("-" * width for width in widths))
    for row in rows:
        print("  " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def print_segments(
    sample: Sample,
    em: float,
    f1: float,
    segments: list[TurnSegment],
    show_tokens: bool,
    max_tokens: int,
) -> None:
    total_tokens = sum(segment.token_count for segment in segments)
    weighted_loss = sum(segment.pg_loss * segment.token_count for segment in segments) / max(
        total_tokens, 1
    )
    clipfrac = sum(
        segment.clip_fraction * segment.token_count for segment in segments
    ) / max(total_tokens, 1)

    print(f"\nSample: {sample.sample_id}")
    print(f"answer={extract_answer(sample.completion)!r} em={em:.1f} f1={f1:.3f}")
    print(f"token_weighted_pg_loss={weighted_loss:+.4f} clipfrac={clipfrac:.3f}")
    headers = [
        "segment",
        "tok",
        "rawIG",
        "normIG",
        "adv",
        "clip",
        "old_lp",
        "cur_lp",
        "ratio",
        "clamped",
        "loss",
        "kl",
    ]
    rows = []
    for segment in segments:
        rows.append(
            [
                segment.name,
                str(segment.token_count),
                f"{segment.raw_ig:+.3f}",
                f"{segment.norm_ig:+.3f}",
                f"{segment.advantage:+.3f}",
                f"{segment.clip_scale:.3f}",
                f"{segment.old_logp:+.3f}",
                f"{segment.current_logp:+.3f}",
                f"{segment.ratio:.3f}",
                f"{segment.clipped_ratio:.3f}",
                f"{segment.pg_loss:+.3f}",
                f"{segment.sampled_kl:+.3f}",
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
    if show_tokens:
        for segment in segments:
            print_token_records(segment, max_tokens=max_tokens)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--clip-low", type=float, default=3e-3)
    parser.add_argument("--clip-high", type=float, default=4e-3)
    parser.add_argument("--no-dynamic-clip", action="store_true")
    parser.add_argument("--show-tokens", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--sim-step",
        type=float,
        default=0.05,
        help="Deterministic virtual log-ratio step used only for no-training diagnostics.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    scorer = MockAnswerLikelihoodScorer()
    print("A-TGPO component smoke test (no training, mock likelihood scorer)")
    print("=" * 72)
    for sample in demo_samples():
        em, f1, segments = compute_atgpo_segments(
            sample=sample,
            scorer=scorer,
            alpha=args.alpha,
            gamma=args.gamma,
            clip_low=args.clip_low,
            clip_high=args.clip_high,
            dynamic_clip=not args.no_dynamic_clip,
            sim_step=args.sim_step,
        )
        print_segments(
            sample=sample,
            em=em,
            f1=f1,
            segments=segments,
            show_tokens=args.show_tokens,
            max_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()
