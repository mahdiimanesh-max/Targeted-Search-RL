#!/usr/bin/env python3
"""Online anchored PrefixIG-TPO LoRA training on HotpotQA-mini.

This is the real-retrieval counterpart to `online_prefixig_tpo_lora.py`.
Instead of synthetic oracle facts, each online iteration samples HotpotQA-mini
questions, rolls out trajectories against the local lexical retriever, builds a
PrefixIG-TPO target distribution over the sampled group, and updates a LoRA
adapter with the grouped TPO loss plus a small token/action behavior anchor.

For Qwen2.5-0.5B, the optional `--gold-anchor-count` is a pragmatic bootstrap:
it inserts canonical support-grounded trajectories into each group so the weak
base policy has at least one high-quality candidate to learn from.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_hotpotqa_mini import (  # noqa: E402
    MiniBM25,
    build_prompt,
    classify_sample,
    read_jsonl,
    result_block,
    rollout_with_local_search,
)
from mlx_generated_policy_diagnostics import (  # noqa: E402
    apply_efficiency_penalty,
    format_prompt,
    write_jsonl,
)
from online_prefixig_tpo_lora import (  # noqa: E402
    CurrentPolicyScorer,
    bucket_summary,
    online_tpo_anchor_loss,
    save_adapter,
)
from prefix_ig_tpo_smoke import (  # noqa: E402
    Sample,
    SampleDiagnostics,
    exact_match,
    extract_answer,
    extract_boxed,
    mean_std,
    parse_search_turns,
    prefix_information_gain,
    softmax,
    token_f1,
)
from train_offline_tpo_lora import (  # noqa: E402
    encode_group_records,
    make_group_batch,
)


def normalize_records(records: list[dict[str, Any]], rng: random.Random, n: int) -> list[dict[str, Any]]:
    if n < 0 or n >= len(records):
        selected = list(records)
        rng.shuffle(selected)
        return selected
    return rng.sample(records, n)


def build_gold_anchor_completion(
    record: dict[str, Any],
    max_search_turns: int,
    max_doc_chars: int,
) -> str:
    supports = [
        paragraph
        for paragraph in record.get("paragraphs", [])
        if paragraph.get("is_support")
    ]
    if not supports:
        supports = record.get("paragraphs", [])[: max_search_turns]
    supports = supports[:max_search_turns]

    parts = []
    for paragraph in supports:
        title = str(paragraph.get("title", "")).strip()
        text = " ".join(str(paragraph.get("text", "")).split())
        if len(text) > max_doc_chars:
            text = text[:max_doc_chars].rsplit(" ", 1)[0].strip()
        query = title or record["question"]
        parts.append(
            f"<search>{query}</search>\n"
            "<result>\n"
            f"[1]\nTitle: {title}\nText: {text}\nScore: 999.000\n"
            "</result>"
        )
    answer = str((record.get("golden_answers") or [record.get("answer", "")])[0])
    parts.append(f"<answer>\\boxed{{{answer}}}</answer>")
    return "\n".join(parts)


def compute_hotpot_diags(
    method: str,
    samples: list[Sample],
    references_by_sample: dict[str, list[str]],
    scorer,
    args,
) -> list[SampleDiagnostics]:
    apply_curvature = method in {"prefixig_tpo_curv", "prefixig_tpo_curv_rg_eff"}
    lambda_ig = args.lambda_ig if method != "reward_tpo" else 0.0

    raw = []
    prefix_igs = []
    for sample in samples:
        references = references_by_sample.get(sample.sample_id, [sample.answer])
        predicted = extract_answer(sample.completion) or extract_boxed(sample.completion) or ""
        em = exact_match(predicted, references)
        f1 = token_f1(predicted, references)
        reward = max(em, f1 if args.use_f1_reward else 0.0)
        prefix_ig, turn_igs = prefix_information_gain(
            scorer=scorer,
            prompt=sample.prompt,
            completion=sample.completion,
            answer=references[0],
        )
        prefix_igs.append(prefix_ig)
        raw.append((sample, predicted, reward, em, f1, prefix_ig, turn_igs))

    ig_mean, ig_std = mean_std(prefix_igs)
    old_policy_probs = softmax([item[0].old_logp for item in raw])
    curvatures = [prob * (1.0 - prob) for prob in old_policy_probs]

    diags = []
    utilities = []
    for idx, (sample, predicted, reward, em, f1, prefix_ig, turn_igs) in enumerate(raw):
        normalized_ig = (prefix_ig - ig_mean) / ig_std
        ig_term = lambda_ig * normalized_ig
        if apply_curvature:
            ig_term = ig_term / (curvatures[idx] + args.curvature_eps) ** 0.5
        utility = reward + ig_term
        utilities.append(utility)
        diags.append(
            SampleDiagnostics(
                sample_id=sample.sample_id,
                final_reward=reward,
                exact_match=em,
                token_f1=f1,
                prefix_ig=prefix_ig,
                normalized_ig=normalized_ig,
                utility=utility,
                old_policy_prob=old_policy_probs[idx],
                target_weight=0.0,
                curvature=curvatures[idx],
                num_turns=len(turn_igs),
                predicted_answer=predicted,
                turn_igs=turn_igs,
            )
        )

    if method in {"prefixig_tpo_rg_eff", "prefixig_tpo_curv_rg_eff"}:
        base_with_weights = [
            replace(diag, target_weight=weight)
            for diag, weight in zip(
                diags,
                softmax(
                    [
                        sample.old_logp + diag.utility / max(args.tau, 1e-8)
                        for sample, diag in zip(samples, diags)
                    ]
                ),
            )
        ]
        return apply_efficiency_penalty(
            diags=base_with_weights,
            samples=samples,
            tau=args.tau,
            apply_curvature=apply_curvature,
            reward_gated=True,
            curvature_eps=args.curvature_eps,
            optimal_turns=args.eff_optimal_turns,
            repeat_threshold=args.eff_repeat_threshold,
            min_turn_ig=args.eff_min_turn_ig,
            lambda_extra_turn=args.eff_lambda_extra_turn,
            lambda_repeat_query=args.eff_lambda_repeat_query,
            lambda_low_ig=args.eff_lambda_low_ig,
        )

    weights = softmax(
        [
            sample.old_logp + utility / max(args.tau, 1e-8)
            for sample, utility in zip(samples, utilities)
        ]
    )
    return [replace(diag, target_weight=weight) for diag, weight in zip(diags, weights)]


def sample_hotpot_groups(
    model,
    tokenizer,
    train_records: list[dict[str, Any]],
    retriever: MiniBM25,
    iter_idx: int,
    args,
) -> tuple[list[list[dict]], list[dict]]:
    scorer = CurrentPolicyScorer(model, tokenizer)
    rng = random.Random(args.seed + 1009 * iter_idx)
    selected_records = normalize_records(train_records, rng, args.prompts_per_iter)
    groups: list[list[dict]] = []
    records_out: list[dict] = []

    model.eval()
    for local_idx, record in enumerate(selected_records):
        case_id = (iter_idx - 1) * args.prompts_per_iter + local_idx
        user_prompt = build_prompt(
            record,
            max_search_turns=args.max_search_turns,
            skill_text=args.skill_text,
        )
        generation_prompt = format_prompt(
            scorer=scorer,
            user_prompt=user_prompt,
            use_chat_template=not args.no_chat_template,
        )
        references = [str(item) for item in record.get("golden_answers") or [record["answer"]]]
        support_titles = [str(item) for item in record.get("supporting_titles", [])]
        samples: list[Sample] = []
        references_by_sample: dict[str, list[str]] = {}

        for sample_idx in range(args.samples_per_prompt):
            completion = rollout_with_local_search(
                scorer=scorer,
                prompt=generation_prompt,
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
            old_logp = scorer.completion_mean_logp(generation_prompt, completion)
            sample = Sample(
                sample_id=f"hotpot{case_id:04d}_gen{sample_idx:02d}",
                prompt=generation_prompt,
                completion=completion,
                answer=references[0],
                old_logp=old_logp,
            )
            samples.append(sample)
            references_by_sample[sample.sample_id] = references

        for anchor_idx in range(args.gold_anchor_count):
            completion = build_gold_anchor_completion(
                record,
                max_search_turns=args.max_search_turns,
                max_doc_chars=args.max_doc_chars,
            )
            old_logp = scorer.completion_mean_logp(generation_prompt, completion)
            sample = Sample(
                sample_id=f"hotpot{case_id:04d}_gold{anchor_idx:02d}",
                prompt=generation_prompt,
                completion=completion,
                answer=references[0],
                old_logp=old_logp,
            )
            samples.append(sample)
            references_by_sample[sample.sample_id] = references

        diags = compute_hotpot_diags(
            method=args.target_method,
            samples=samples,
            references_by_sample=references_by_sample,
            scorer=scorer,
            args=args,
        )
        group_records = []
        for diag, sample in zip(diags, samples):
            prediction = extract_answer(sample.completion) or extract_boxed(sample.completion) or ""
            bucket, exact, f1, correct, retrieved_titles, support_hits = classify_sample(
                completion=sample.completion,
                prediction=prediction,
                references=references,
                support_titles=support_titles,
                correct_f1_threshold=args.correct_f1_threshold,
                useful_support_coverage=args.useful_support_coverage,
                allow_reference_substring=True,
            )
            record_out = {
                "target_method": args.target_method,
                "online_iter": iter_idx,
                "eval_regime": "hotpotqa_mini",
                "case_id": case_id,
                "question_id": record.get("id"),
                "sample_id": sample.sample_id,
                "prompt": user_prompt,
                "generation_prompt": generation_prompt,
                "completion": sample.completion,
                "answer": references[0],
                "references": references,
                "prediction": prediction,
                "bucket": bucket,
                "exact": exact,
                "f1": f1,
                "correct": correct,
                "support_titles": support_titles,
                "retrieved_titles": list(retrieved_titles),
                "support_hits": list(support_hits),
                "old_logp": sample.old_logp,
                "reward": diag.final_reward,
                "prefix_ig": diag.prefix_ig,
                "turn_igs": list(diag.turn_igs),
                "curvature": diag.curvature,
                "utility": diag.utility,
                "target_weight": diag.target_weight,
            }
            group_records.append(record_out)
            records_out.append(record_out)
        groups.append(group_records)

    mx.clear_cache()
    return groups, records_out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--resume-adapter-file", default=None)
    parser.add_argument("--load-in-4bits", action="store_true")
    parser.add_argument("--train-jsonl", default="outputs/hotpotqa_mini/train.jsonl")
    parser.add_argument("--corpus-jsonl", default="outputs/hotpotqa_mini/corpus.jsonl")
    parser.add_argument(
        "--skill-prompt-file",
        action="append",
        default=[],
        help=(
            "Optional SDAR-style skill markdown file to inject into training prompts. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--skill-max-chars",
        type=int,
        default=1800,
        help="Maximum characters of concatenated skill text to inject.",
    )
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--online-iters", type=int, default=5)
    parser.add_argument("--prompts-per-iter", type=int, default=2)
    parser.add_argument("--samples-per-prompt", type=int, default=3)
    parser.add_argument("--gold-anchor-count", type=int, default=0)
    parser.add_argument("--updates-per-iter", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--anchor-beta", type=float, default=0.1)
    parser.add_argument(
        "--target-method",
        choices=[
            "reward_tpo",
            "prefixig_tpo",
            "prefixig_tpo_curv",
            "prefixig_tpo_rg_eff",
            "prefixig_tpo_curv_rg_eff",
        ],
        default="prefixig_tpo_rg_eff",
    )
    parser.add_argument("--use-f1-reward", action="store_true")
    parser.add_argument("--correct-f1-threshold", type=float, default=0.65)
    parser.add_argument("--useful-support-coverage", type=float, default=1.0)
    parser.add_argument("--max-search-turns", type=int, default=2)
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--max-doc-chars", type=int, default=900)
    parser.add_argument("--action-max-tokens", type=int, default=48)
    parser.add_argument("--answer-max-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--lambda-ig", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--curvature-eps", type=float, default=1e-3)
    parser.add_argument("--eff-optimal-turns", type=int, default=2)
    parser.add_argument("--eff-repeat-threshold", type=float, default=0.55)
    parser.add_argument("--eff-min-turn-ig", type=float, default=0.05)
    parser.add_argument("--eff-lambda-extra-turn", type=float, default=0.6)
    parser.add_argument("--eff-lambda-repeat-query", type=float, default=0.4)
    parser.add_argument("--eff-lambda-low-ig", type=float, default=0.0)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--score-normalization", choices=["mean", "sum"], default="mean")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--show-samples", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    from mlx_lm.tuner.utils import print_trainable_parameters
    from mlx_lm.utils import save_config
    from mlx_lm_lora.utils import from_pretrained

    random.seed(args.seed)
    mx.random.seed(args.seed)

    train_records = read_jsonl(Path(args.train_jsonl))
    corpus = read_jsonl(Path(args.corpus_jsonl))
    retriever = MiniBM25(corpus)
    skill_text = ""
    if args.skill_prompt_file:
        chunks = []
        for path in args.skill_prompt_file:
            chunks.append(Path(path).read_text(encoding="utf-8").strip())
        skill_text = "\n\n".join(chunks)
        if args.skill_max_chars > 0 and len(skill_text) > args.skill_max_chars:
            skill_text = skill_text[: args.skill_max_chars].rsplit("\n", 1)[0].strip()
    args.skill_text = skill_text

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
    print("Online HotpotQA-mini PrefixIG-TPO")
    print("=" * 64)
    print(f"target_method: {args.target_method}")
    print(f"use_f1_reward: {args.use_f1_reward}")
    print(f"gold_anchor_count: {args.gold_anchor_count}")
    print(f"anchor_beta: {args.anchor_beta}")
    print(f"skill_prompt_files: {len(args.skill_prompt_file)}")
    print(
        f"schedule: {args.online_iters} iters x {args.prompts_per_iter} prompts "
        f"x {args.samples_per_prompt} samples, {args.updates_per_iter} updates/iter"
    )

    for online_iter in range(1, args.online_iters + 1):
        raw_groups, records = sample_hotpot_groups(
            model=model,
            tokenizer=tokenizer,
            train_records=train_records,
            retriever=retriever,
            iter_idx=online_iter,
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
            for record in records[: min(4, len(records))]:
                print(
                    f"- {record['sample_id']} {record['bucket']} "
                    f"q={float(record['target_weight']):.3f} "
                    f"f1={float(record['f1']):.3f}: "
                    f"{record['completion'][:220]!r}"
                )

        model.train()
        losses = []
        anchor_losses = []
        for update_idx in tqdm(range(args.updates_per_iter), desc=f"hotpot_update_{online_iter}"):
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
