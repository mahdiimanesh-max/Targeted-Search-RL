#!/usr/bin/env python3
"""Train a small MLX LoRA adapter with true offline TPO over trajectory groups.

This differs from the target-weighted SFT pilot: each training step sees all
candidate trajectories for one prompt/case, computes the current model
distribution over those trajectories, and matches the stored PrefixIG-TPO target
weights with a grouped cross-entropy loss.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from tqdm import tqdm


@dataclass
class EncodedTrajectory:
    sample_id: str
    bucket: str
    input_ids: list[int]
    completion_start: int
    target_weight: float


@dataclass
class EncodedGroup:
    case_id: int
    eval_regime: str
    trajectories: list[EncodedTrajectory]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encode_text(tokenizer, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def eos_token_id(tokenizer) -> int | None:
    token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(token_id, int):
        return token_id
    return None


def group_records(records: list[dict], target_method: str) -> list[list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for record in records:
        if str(record["target_method"]) != target_method:
            continue
        grouped.setdefault(int(record["case_id"]), []).append(record)
    return [grouped[key] for key in sorted(grouped)]


def split_groups(
    groups: list[EncodedGroup],
    seed: int,
    valid_fraction: float,
    test_fraction: float,
) -> tuple[list[EncodedGroup], list[EncodedGroup], list[EncodedGroup]]:
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_valid = max(1, int(round(n_total * valid_fraction))) if n_total >= 3 else 0
    n_test = max(1, int(round(n_total * test_fraction))) if n_total >= 3 else 0
    n_train = max(1, n_total - n_valid - n_test)
    train = shuffled[:n_train]
    valid = shuffled[n_train : n_train + n_valid]
    test = shuffled[n_train + n_valid :]
    return train, valid, test


def encode_group_records(
    groups: list[list[dict]],
    tokenizer,
    max_seq_length: int,
) -> list[EncodedGroup]:
    eos = eos_token_id(tokenizer)
    encoded_groups: list[EncodedGroup] = []

    for records in groups:
        trajectories: list[EncodedTrajectory] = []
        weight_sum = sum(max(float(record["target_weight"]), 0.0) for record in records)
        if weight_sum <= 0.0:
            continue

        for record in records:
            prompt_ids = encode_text(tokenizer, str(record["generation_prompt"]))
            completion_ids = encode_text(tokenizer, str(record["completion"]))
            if eos is not None and (not completion_ids or completion_ids[-1] != eos):
                completion_ids = completion_ids + [eos]

            if len(prompt_ids) >= max_seq_length - 2:
                prompt_ids = prompt_ids[-(max_seq_length - 2) :]

            available_completion = max_seq_length - len(prompt_ids)
            completion_ids = completion_ids[:available_completion]
            if not prompt_ids or not completion_ids:
                continue

            input_ids = prompt_ids + completion_ids
            trajectories.append(
                EncodedTrajectory(
                    sample_id=str(record["sample_id"]),
                    bucket=str(record["bucket"]),
                    input_ids=input_ids,
                    completion_start=len(prompt_ids),
                    target_weight=max(float(record["target_weight"]), 0.0) / weight_sum,
                )
            )

        if len(trajectories) < 2:
            continue
        encoded_groups.append(
            EncodedGroup(
                case_id=int(records[0]["case_id"]),
                eval_regime=str(records[0].get("eval_regime", "unknown")),
                trajectories=trajectories,
            )
        )

    return encoded_groups


def make_group_batch(group: EncodedGroup, score_normalization: str):
    max_len = max(len(item.input_ids) for item in group.trajectories)
    batch = []
    masks = []
    target_weights = []
    buckets = []

    for item in group.trajectories:
        pad_len = max_len - len(item.input_ids)
        ids = item.input_ids + [0] * pad_len
        target_len = max_len - 1
        # Targets are input_ids[1:]. Completion likelihood begins at the first
        # completion token, whose target position is completion_start - 1.
        start = max(item.completion_start - 1, 0)
        mask = [pos >= start and pos < len(item.input_ids) - 1 for pos in range(target_len)]
        batch.append(ids)
        masks.append(mask)
        target_weights.append(item.target_weight)
        buckets.append(item.bucket)

    q = mx.array(target_weights, dtype=mx.float32)
    q = q / q.sum()
    return (
        mx.array(batch, dtype=mx.int32),
        mx.array(masks, dtype=mx.bool_),
        q,
        buckets,
        score_normalization,
    )


def sequence_log_scores(model, inputs, completion_mask, score_normalization: str):
    logits = model(inputs[:, :-1])
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = logits.astype(mx.float32)
    targets = inputs[:, 1:]
    log_probs = nn.log_softmax(logits, axis=-1)
    selected = mx.take_along_axis(
        log_probs, mx.expand_dims(targets, axis=-1), axis=-1
    ).squeeze(-1)
    selected = mx.where(completion_mask, selected, mx.zeros_like(selected))
    score_sum = selected.sum(axis=1)
    lengths = completion_mask.sum(axis=1).astype(mx.float32)
    if score_normalization == "mean":
        return score_sum / mx.maximum(lengths, 1.0), lengths
    return score_sum, lengths


def offline_tpo_loss(model, inputs, completion_mask, q_target, score_normalization: str):
    log_scores, lengths = sequence_log_scores(
        model=model,
        inputs=inputs,
        completion_mask=completion_mask,
        score_normalization=score_normalization,
    )
    log_p = nn.log_softmax(log_scores, axis=0)
    p = nn.softmax(log_scores, axis=0)
    loss = -(mx.stop_gradient(q_target) * log_p).sum()
    policy_entropy = -(p * log_p).sum()
    target_entropy = -(q_target * mx.log(q_target + 1e-12)).sum()
    return loss, lengths.sum(), policy_entropy, target_entropy


def evaluate(model, groups: list[EncodedGroup], args) -> dict[str, float]:
    if not groups:
        return {"loss": float("nan"), "policy_entropy": float("nan"), "target_entropy": float("nan")}
    model.eval()
    losses = []
    policy_entropies = []
    target_entropies = []
    for group in groups:
        inputs, mask, q, _buckets, norm = make_group_batch(group, args.score_normalization)
        loss, _ntoks, policy_entropy, target_entropy = offline_tpo_loss(
            model, inputs, mask, q, norm
        )
        mx.eval(loss, policy_entropy, target_entropy)
        losses.append(float(loss.item()))
        policy_entropies.append(float(policy_entropy.item()))
        target_entropies.append(float(target_entropy.item()))
    model.train()
    return {
        "loss": sum(losses) / len(losses),
        "policy_entropy": sum(policy_entropies) / len(policy_entropies),
        "target_entropy": sum(target_entropies) / len(target_entropies),
    }


def bucket_mass(groups: list[EncodedGroup]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for group in groups:
        for traj in group.trajectories:
            totals[traj.bucket] = totals.get(traj.bucket, 0.0) + traj.target_weight
    denom = max(len(groups), 1)
    return {bucket: value / denom for bucket, value in sorted(totals.items())}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidates-jsonl", required=True)
    parser.add_argument("--target-method", default="prefixig_tpo")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument(
        "--resume-adapter-file",
        default=None,
        help="Optional adapter weights to load after creating LoRA layers.",
    )
    parser.add_argument("--load-in-4bits", action="store_true")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--steps-per-report", type=int, default=5)
    parser.add_argument("--steps-per-eval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=40)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--score-normalization", choices=["mean", "sum"], default="mean")
    parser.add_argument("--grad-checkpoint", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    from mlx_lm.tuner.utils import print_trainable_parameters
    from mlx_lm.utils import save_config
    from mlx_lm_lora.trainer.sft_trainer import grad_checkpoint
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
        print(f"Loading initial adapter weights from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)
    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    print_trainable_parameters(model)

    records = read_jsonl(Path(args.candidates_jsonl))
    raw_groups = group_records(records, args.target_method)
    groups = encode_group_records(raw_groups, tokenizer, args.max_seq_length)
    train_groups, valid_groups, test_groups = split_groups(
        groups,
        seed=args.seed,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
    )

    print()
    print("Offline TPO dataset")
    print("=" * 48)
    print(f"target_method: {args.target_method}")
    print(f"groups: train={len(train_groups)} valid={len(valid_groups)} test={len(test_groups)}")
    print(f"trajectories/group: {[len(group.trajectories) for group in groups[:5]]} ...")
    print("train target mass:")
    for bucket, mass in bucket_mass(train_groups).items():
        print(f"- {bucket}: {mass:.3f}")

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

    optimizer = optim.AdamW(learning_rate=args.learning_rate)
    loss_value_and_grad = nn.value_and_grad(model, offline_tpo_loss)

    def train_step(group: EncodedGroup, prev_grad, do_update: bool):
        inputs, mask, q, _buckets, norm = make_group_batch(group, args.score_normalization)
        (loss, ntoks, policy_entropy, target_entropy), grad = loss_value_and_grad(
            model, inputs, mask, q, norm
        )
        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)
        if do_update:
            if args.gradient_accumulation_steps > 1:
                grad = tree_map(lambda x: x / args.gradient_accumulation_steps, grad)
            optimizer.update(model, grad)
            grad = None
        return loss, ntoks, policy_entropy, target_entropy, grad

    model.train()
    grad_accum = None
    losses = []
    token_counts = []
    start = time.perf_counter()
    rng = random.Random(args.seed)
    pbar = tqdm(range(1, args.iters + 1), desc="Offline TPO")

    for iteration in pbar:
        if iteration == 1 or (iteration - 1) % len(train_groups) == 0:
            rng.shuffle(train_groups)
        group = train_groups[(iteration - 1) % len(train_groups)]
        do_update = iteration % args.gradient_accumulation_steps == 0
        loss, ntoks, policy_entropy, target_entropy, grad_accum = train_step(
            group, grad_accum, do_update
        )
        mx.eval(model.state, optimizer.state, loss, ntoks, policy_entropy, target_entropy)
        losses.append(float(loss.item()))
        token_counts.append(float(ntoks.item()))

        if iteration == 1 or iteration % args.steps_per_eval == 0 or iteration == args.iters:
            metrics = evaluate(model, valid_groups, args)
            print(
                f"Iter {iteration}: val_loss={metrics['loss']:.3f}, "
                f"val_policy_entropy={metrics['policy_entropy']:.3f}, "
                f"val_target_entropy={metrics['target_entropy']:.3f}"
            )

        if iteration % args.steps_per_report == 0 or iteration == args.iters:
            elapsed = max(time.perf_counter() - start, 1e-8)
            mean_loss = sum(losses) / len(losses)
            tokens = sum(token_counts)
            print(
                f"Iter {iteration}: loss={mean_loss:.3f}, "
                f"policy_entropy={float(policy_entropy.item()):.3f}, "
                f"target_entropy={float(target_entropy.item()):.3f}, "
                f"tok/s={tokens / elapsed:.1f}, peak_mem={mx.get_peak_memory() / 1e9:.3f}GB"
            )
            pbar.set_postfix({"loss": f"{mean_loss:.3f}"})
            losses = []
            token_counts = []
            start = time.perf_counter()

        if iteration % args.save_every == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(adapter_file), adapter_weights)
            checkpoint = adapter_path / f"{iteration:07d}_adapters.safetensors"
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(f"Iter {iteration}: saved {adapter_file} and {checkpoint}")

    if grad_accum is not None:
        if args.gradient_accumulation_steps > 1:
            grad_accum = tree_map(lambda x: x / args.gradient_accumulation_steps, grad_accum)
        optimizer.update(model, grad_accum)

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_file), adapter_weights)
    test_metrics = evaluate(model, test_groups, args)
    print()
    print("Final offline TPO metrics")
    print("=" * 48)
    print(f"train_loss_last: {mean_loss:.3f}")
    print(f"test_loss:       {test_metrics['loss']:.3f}")
    print(f"test_entropy:    {test_metrics['policy_entropy']:.3f}")
    print(f"saved_adapter:   {adapter_file}")


if __name__ == "__main__":
    main()
