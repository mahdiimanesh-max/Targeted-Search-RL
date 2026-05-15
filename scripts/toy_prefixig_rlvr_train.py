#!/usr/bin/env python3
"""Toy trainable RLVR experiment for PrefixIG-TPO.

This is a small controlled environment for paper-style learning curves. A tiny
causal transformer sees a target answer token, generates evidence/action tokens,
then emits an answer token. Final reward only checks the answer, while PrefixIG
credits useful evidence actions and the reward-gated efficiency penalty
discourages redundant evidence after a correct answer is already possible.

No LLM weights are used. This should run quickly on Apple Silicon with MLX.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm


TOK_USEFUL_BRIDGE = 0
TOK_USEFUL_ANSWER = 1
TOK_REDUNDANT = 2
TOK_DISTRACTOR = 3
TOK_NO_SEARCH = 4
ANSWER_OFFSET = 5


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, num_heads, bias=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_mult * d_model)
        self.fc2 = nn.Linear(ffn_mult * d_model, d_model)

    def __call__(self, x, mask):
        y = self.ln1(x)
        x = x + self.attn(y, y, y, mask=mask)
        y = self.ln2(x)
        return x + self.fc2(nn.gelu(self.fc1(y)))


class TinyCausalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_mult: int,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = mx.random.normal((max_seq_len, d_model), scale=0.02)
        self.layers = [
            TransformerBlock(d_model=d_model, num_heads=num_heads, ffn_mult=ffn_mult)
            for _ in range(num_layers)
        ]
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def __call__(self, tokens):
        single = tokens.ndim == 1
        if single:
            tokens = tokens[None, :]
        _, seq_len = tokens.shape
        x = self.token_embedding(tokens) + self.pos_embedding[:seq_len]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len, dtype=x.dtype)
        for layer in self.layers:
            x = layer(x, mask)
        logits = self.out(self.ln(x))
        log_probs = nn.log_softmax(logits, axis=-1)
        return log_probs[0] if single else log_probs


@dataclass(frozen=True)
class RolloutBatch:
    prompts: mx.array
    completions: mx.array
    old_token_logps: mx.array
    old_log_scores: mx.array
    utilities: mx.array
    token_advantages: mx.array
    rewards: mx.array
    prefix_ig: mx.array
    redundancy_penalty: mx.array


def sample_next(log_probs, temperature: float):
    if temperature <= 0:
        return mx.argmax(log_probs, axis=-1)
    return mx.random.categorical(log_probs / temperature, axis=-1)


def rollout_group(
    model: TinyCausalTransformer,
    *,
    batch_size: int,
    group_size: int,
    answer_count: int,
    completion_len: int,
    temperature: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    prompts = mx.random.randint(0, answer_count, shape=(batch_size,))
    full_len = 1 + completion_len
    all_completions = []
    all_log_scores = []

    was_training = model.training
    model.eval()
    try:
        for _ in range(group_size):
            tokens = mx.zeros((batch_size, full_len), dtype=mx.int32)
            tokens[:, 0] = ANSWER_OFFSET + prompts
            completion_tokens = []
            token_logps = []
            for step in range(completion_len):
                log_probs = model(tokens)
                step_log_probs = log_probs[:, step, :]
                sampled = sample_next(step_log_probs, temperature=temperature)
                sampled_logp = mx.take_along_axis(
                    step_log_probs, sampled[:, None], axis=-1
                ).squeeze(-1)
                tokens[:, step + 1] = sampled
                completion_tokens.append(sampled)
                token_logps.append(sampled_logp)
            all_completions.append(mx.stack(completion_tokens, axis=1))
            all_log_scores.append(mx.stack(token_logps, axis=1).sum(axis=1))
    finally:
        if was_training:
            model.train()

    old_token_logps = mx.stack(
        [
            mx.stack(
                [
                    mx.zeros((batch_size,), dtype=mx.float32)
                    for _ in range(completion_len)
                ],
                axis=1,
            )
        ],
        axis=1,
    )
    # Recompute as a stacked tensor for token-level A-TGPO accounting.
    # all_log_scores already contains the same values summed over time.
    all_token_logps = []
    for group_completions in all_completions:
        tokens = mx.concatenate([(ANSWER_OFFSET + prompts)[:, None], group_completions], axis=1)
        log_probs = model(tokens)
        selected = mx.take_along_axis(
            log_probs[:, :completion_len, :],
            group_completions[:, :, None],
            axis=-1,
        ).squeeze(-1)
        all_token_logps.append(selected)
    old_token_logps = mx.stack(all_token_logps, axis=1)
    return (
        prompts,
        mx.stack(all_completions, axis=1),
        old_token_logps,
        mx.stack(all_log_scores, axis=1),
    )


def sequence_log_scores(model, prompts, completions, answer_count: int, completion_len: int):
    return token_log_scores(model, prompts, completions, answer_count, completion_len).sum(axis=2)


def token_log_scores(model, prompts, completions, answer_count: int, completion_len: int):
    batch_size, group_size, _ = completions.shape
    flat_prompts = mx.repeat(prompts, group_size, axis=0)
    flat_completions = completions.reshape(batch_size * group_size, completion_len)
    full = mx.concatenate([(ANSWER_OFFSET + flat_prompts)[:, None], flat_completions], axis=1)
    log_probs = model(full)
    pred_log_probs = log_probs[:, :completion_len, :]
    selected = mx.take_along_axis(pred_log_probs, flat_completions[:, :, None], axis=-1)
    return selected.squeeze(-1).reshape(batch_size, group_size, completion_len)


def required_hops(prompts_np: np.ndarray, hop_mode: str) -> np.ndarray:
    if hop_mode == "single":
        return np.ones_like(prompts_np, dtype=np.int32)
    if hop_mode == "multi":
        return np.full_like(prompts_np, 2, dtype=np.int32)
    if hop_mode == "mixed":
        return 1 + (prompts_np % 2).astype(np.int32)
    raise ValueError(f"Unknown hop_mode: {hop_mode}")


def retrieval_noise_thresholds(retrieval_noise: str) -> tuple[int, int]:
    """Return deterministic distractor/stale corruption percentages."""
    if retrieval_noise == "clean":
        return 0, 0
    if retrieval_noise == "distractor25":
        return 25, 0
    if retrieval_noise == "distractor50":
        return 50, 0
    if retrieval_noise == "stale25":
        return 0, 25
    if retrieval_noise == "stale50":
        return 0, 50
    if retrieval_noise == "mixed25":
        return 15, 10
    if retrieval_noise == "mixed50":
        return 30, 20
    raise ValueError(f"Unknown retrieval_noise: {retrieval_noise}")


def analyze_trajectories(
    prompts_np: np.ndarray,
    completions_np: np.ndarray,
    *,
    lambda_ig: float,
    lambda_eff: float,
    reward_gated_eff: bool,
    ig_mode: str = "raw_vr",
    token_advantage_mode: str = "raw",
    hop_mode: str = "multi",
    retrieval_noise: str = "clean",
) -> dict[str, np.ndarray]:
    batch_size, group_size, completion_len = completions_np.shape
    target_answer = ANSWER_OFFSET + prompts_np[:, None]
    answer_token = completions_np[:, :, -1]
    rewards = (answer_token == target_answer).astype(np.float32)

    action_tokens = completions_np[:, :, :-1]
    required = required_hops(prompts_np, hop_mode)[:, None]
    bridge_seen = np.zeros((batch_size, group_size), dtype=bool)
    answer_evidence_seen = np.zeros((batch_size, group_size), dtype=bool)
    turn_igs = np.zeros_like(action_tokens, dtype=np.float32)
    redundant_turns = np.zeros((batch_size, group_size), dtype=np.float32)
    distractor_turns = np.zeros((batch_size, group_size), dtype=np.float32)
    distractor_pct, stale_pct = retrieval_noise_thresholds(retrieval_noise)
    group_ids = np.arange(group_size)[None, :]

    for step in range(completion_len - 1):
        token = action_tokens[:, :, step]
        bridge = token == TOK_USEFUL_BRIDGE
        answer_evidence = token == TOK_USEFUL_ANSWER
        distractor = token == TOK_DISTRACTOR
        single_hop = required == 1
        multi_hop = required == 2
        evidence_action = bridge | answer_evidence
        noise_score = (prompts_np[:, None] * 17 + group_ids * 31 + step * 13) % 100
        corrupted_distractor = evidence_action & (noise_score < distractor_pct)
        corrupted_stale = (
            evidence_action
            & (noise_score >= distractor_pct)
            & (noise_score < distractor_pct + stale_pct)
        )
        evidence_available = evidence_action & ~(corrupted_distractor | corrupted_stale)

        useful_bridge = multi_hop & bridge & evidence_available & ~bridge_seen
        useful_answer = answer_evidence & evidence_available & ~answer_evidence_seen
        useful_answer = np.where(multi_hop, useful_answer & bridge_seen, useful_answer)
        out_of_order_answer = multi_hop & answer_evidence & evidence_available & ~bridge_seen

        redundant = (
            (token == TOK_REDUNDANT)
            | (single_hop & bridge)
            | (bridge & bridge_seen)
            | (answer_evidence & answer_evidence_seen)
            | out_of_order_answer
            | corrupted_stale
        )
        distractor = distractor | corrupted_distractor

        turn_igs[:, :, step] = np.where(useful_bridge, 0.65, turn_igs[:, :, step])
        turn_igs[:, :, step] = np.where(useful_answer, 1.0, turn_igs[:, :, step])
        turn_igs[:, :, step] = np.where(out_of_order_answer, 0.05, turn_igs[:, :, step])
        turn_igs[:, :, step] = np.where(redundant, 0.05, turn_igs[:, :, step])
        turn_igs[:, :, step] = np.where(distractor, -0.25, turn_igs[:, :, step])

        redundant_turns += redundant.astype(np.float32)
        distractor_turns += distractor.astype(np.float32)
        bridge_seen |= bridge & evidence_available
        answer_evidence_seen |= answer_evidence & evidence_available

    non_null_turns = np.maximum((action_tokens != TOK_NO_SEARCH).sum(axis=2), 1)
    evidence_complete = np.where(
        required == 1,
        answer_evidence_seen,
        bridge_seen & answer_evidence_seen,
    )

    turn_mean = turn_igs.mean(axis=1, keepdims=True)
    turn_std = turn_igs.std(axis=1, keepdims=True) + 1e-6
    turn_group_igs = (turn_igs - turn_mean) / turn_std

    if ig_mode == "raw_sum":
        prefix_ig = turn_igs.sum(axis=2)
    elif ig_mode == "raw_vr":
        prefix_ig = turn_igs.sum(axis=2) / np.sqrt(non_null_turns)
    elif ig_mode == "turn_norm":
        prefix_ig = turn_group_igs.sum(axis=2)
    elif ig_mode == "turn_norm_vr":
        prefix_ig = turn_group_igs.sum(axis=2) / np.sqrt(non_null_turns)
    else:
        raise ValueError(f"Unknown ig_mode: {ig_mode}")

    raw_penalty = redundant_turns + np.maximum(non_null_turns - 1, 0) * 0.25
    if reward_gated_eff:
        eff_penalty = rewards * raw_penalty
    else:
        centered = raw_penalty - raw_penalty.mean(axis=1, keepdims=True)
        std = raw_penalty.std(axis=1, keepdims=True) + 1e-6
        eff_penalty = centered / std

    ig_centered = prefix_ig - prefix_ig.mean(axis=1, keepdims=True)
    ig_std = prefix_ig.std(axis=1, keepdims=True) + 1e-6
    normalized_ig = ig_centered / ig_std
    utilities = rewards + lambda_ig * normalized_ig - lambda_eff * eff_penalty

    token_advantages = np.zeros((batch_size, group_size, completion_len), dtype=np.float32)
    if token_advantage_mode == "raw":
        turn_credit = turn_igs
    elif token_advantage_mode == "turn_norm":
        turn_credit = turn_group_igs
    elif token_advantage_mode == "turn_norm_vr":
        accumulated = np.cumsum(turn_group_igs, axis=2)
        denom = np.sqrt(np.arange(1, completion_len, dtype=np.float32))[None, None, :]
        turn_credit = accumulated / denom
    else:
        raise ValueError(f"Unknown token_advantage_mode: {token_advantage_mode}")

    token_advantages[:, :, :-1] = rewards[:, :, None] + lambda_ig * turn_credit
    if reward_gated_eff:
        repeated_bridge = np.maximum(
            (action_tokens == TOK_USEFUL_BRIDGE).cumsum(axis=2) - 1, 0
        )
        repeated_answer = np.maximum(
            (action_tokens == TOK_USEFUL_ANSWER).cumsum(axis=2) - 1, 0
        )
        redundant_steps = (action_tokens == TOK_REDUNDANT).astype(np.float32)
        redundant_steps += repeated_bridge + repeated_answer
        token_advantages[:, :, :-1] -= lambda_eff * rewards[:, :, None] * redundant_steps
    token_advantages[:, :, -1] = rewards
    centered_token_adv = token_advantages - token_advantages.mean(axis=1, keepdims=True)
    std_token_adv = token_advantages.std(axis=1, keepdims=True) + 1e-6
    token_advantages = centered_token_adv / std_token_adv

    correct_useful = (rewards > 0) & evidence_complete & (redundant_turns <= 0)
    correct_redundant = (rewards > 0) & evidence_complete & (redundant_turns > 0)
    correct_no_search = (rewards > 0) & (~evidence_complete) & (non_null_turns <= 1)
    wrong_distractor = (rewards <= 0) & (distractor_turns > 0)

    return {
        "rewards": rewards.astype(np.float32),
        "prefix_ig": prefix_ig.astype(np.float32),
        "redundancy_penalty": raw_penalty.astype(np.float32),
        "utilities": utilities.astype(np.float32),
        "token_advantages": token_advantages.astype(np.float32),
        "useful_correct": correct_useful.astype(np.float32),
        "redundant_correct": correct_redundant.astype(np.float32),
        "no_search_correct": correct_no_search.astype(np.float32),
        "distractor_wrong": wrong_distractor.astype(np.float32),
    }


def make_rollout_batch(model, args, *, reward_gated_eff: bool, lambda_eff: float) -> RolloutBatch:
    prompts, completions, old_token_logps, old_log_scores = rollout_group(
        model,
        batch_size=args.batch_size,
        group_size=args.group_size,
        answer_count=args.answer_count,
        completion_len=args.completion_len,
        temperature=args.temperature,
    )
    mx.eval(prompts, completions, old_log_scores)
    stats = analyze_trajectories(
        np.asarray(prompts.tolist()),
        np.asarray(completions.tolist()),
        lambda_ig=args.lambda_ig,
        lambda_eff=lambda_eff,
        reward_gated_eff=reward_gated_eff,
        ig_mode=getattr(args, "ig_mode", "raw_vr"),
        token_advantage_mode=getattr(args, "token_advantage_mode", "raw"),
        hop_mode=getattr(args, "hop_mode", "multi"),
        retrieval_noise=getattr(args, "retrieval_noise", "clean"),
    )
    return RolloutBatch(
        prompts=prompts,
        completions=completions,
        old_token_logps=mx.stop_gradient(old_token_logps),
        old_log_scores=mx.stop_gradient(old_log_scores),
        utilities=mx.array(stats["utilities"]),
        token_advantages=mx.array(stats["token_advantages"]),
        rewards=mx.array(stats["rewards"]),
        prefix_ig=mx.array(stats["prefix_ig"]),
        redundancy_penalty=mx.array(stats["redundancy_penalty"]),
    )


def tpo_target(old_log_scores, utilities, tau: float):
    logits = nn.log_softmax(old_log_scores, axis=1) + utilities / max(tau, 1e-8)
    return nn.softmax(logits, axis=1)


def make_tpo_loss(model, args):
    def loss_fn(
        prompts,
        completions,
        old_log_scores,
        old_token_logps,
        utilities,
        token_advantages,
        redundancy_penalty,
    ):
        new_log_scores = sequence_log_scores(
            model,
            prompts=prompts,
            completions=completions,
            answer_count=args.answer_count,
            completion_len=args.completion_len,
        )
        q = mx.stop_gradient(tpo_target(mx.stop_gradient(old_log_scores), mx.stop_gradient(utilities), args.tau))
        return -(q * nn.log_softmax(new_log_scores, axis=1)).sum(axis=1).mean()

    return loss_fn


def make_grpo_loss(model, args):
    def loss_fn(
        prompts,
        completions,
        old_log_scores,
        old_token_logps,
        utilities,
        token_advantages,
        redundancy_penalty,
    ):
        new_log_scores = sequence_log_scores(
            model,
            prompts=prompts,
            completions=completions,
            answer_count=args.answer_count,
            completion_len=args.completion_len,
        )
        centered = utilities - utilities.mean(axis=1, keepdims=True)
        adv = mx.stop_gradient(centered / (utilities.std(axis=1, keepdims=True) + 1e-6))
        log_ratio = mx.clip(new_log_scores - mx.stop_gradient(old_log_scores), -20.0, 20.0)
        ratio = mx.exp(log_ratio)
        clipped = mx.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
        surrogate = mx.minimum(ratio * adv, clipped * adv)
        return -surrogate.mean()

    return loss_fn


def make_atgpo_loss(model, args):
    def loss_fn(
        prompts,
        completions,
        old_log_scores,
        old_token_logps,
        utilities,
        token_advantages,
        redundancy_penalty,
    ):
        new_token_logps = token_log_scores(
            model,
            prompts=prompts,
            completions=completions,
            answer_count=args.answer_count,
            completion_len=args.completion_len,
        )
        adv = mx.stop_gradient(token_advantages)
        log_ratio = mx.clip(new_token_logps - mx.stop_gradient(old_token_logps), -20.0, 20.0)
        ratio = mx.exp(log_ratio)

        # A-TGPO-style trust region: widen the update region for informative
        # token-level advantages, and shrink it for redundant trajectories.
        if getattr(args, "atgpo_adaptive_clip", False):
            informative_scale = mx.clip(
                1.0 + args.atgpo_adaptive_clip_scale * mx.maximum(adv, 0.0),
                args.atgpo_min_clip_mult,
                args.atgpo_max_clip_mult,
            )
            adaptive_scale = mx.where(
                adv > 0.0,
                informative_scale,
                args.atgpo_min_clip_mult,
            )
        else:
            adaptive_scale = 1.0
        redundant_scale = mx.clip(
            1.0 / (1.0 + args.atgpo_redundancy_clip * redundancy_penalty[:, :, None]),
            0.5,
            1.0,
        )
        clip_width = args.clip_eps * adaptive_scale * redundant_scale
        clipped = mx.minimum(
            mx.maximum(ratio, 1.0 - clip_width),
            1.0 + clip_width,
        )
        surrogate = mx.minimum(ratio * adv, clipped * adv)
        approx_kl = mx.exp(-log_ratio) - (-log_ratio) - 1.0
        return -(surrogate - args.atgpo_beta * approx_kl).mean()

    return loss_fn


def evaluate(model, args):
    batch = make_rollout_batch(
        model,
        args,
        reward_gated_eff=True,
        lambda_eff=args.lambda_eff,
    )
    mx.eval(batch.rewards, batch.prefix_ig, batch.redundancy_penalty)
    prompts_np = np.asarray(batch.prompts.tolist())
    completions_np = np.asarray(batch.completions.tolist())
    stats = analyze_trajectories(
        prompts_np,
        completions_np,
        lambda_ig=args.lambda_ig,
        lambda_eff=args.lambda_eff,
        reward_gated_eff=True,
        ig_mode=getattr(args, "ig_mode", "raw_vr"),
        token_advantage_mode=getattr(args, "token_advantage_mode", "raw"),
        hop_mode=getattr(args, "hop_mode", "multi"),
        retrieval_noise=getattr(args, "retrieval_noise", "clean"),
    )
    reward = float(stats["rewards"].mean())
    useful = float(stats["useful_correct"].mean())
    redundant = float(stats["redundant_correct"].mean())
    no_search = float(stats["no_search_correct"].mean())
    distractor = float(stats["distractor_wrong"].mean())
    return {
        "reward": reward,
        "useful_correct": useful,
        "redundant_correct": redundant,
        "no_search_correct": no_search,
        "distractor_wrong": distractor,
        "useful_minus_redundant": useful - redundant,
        "avg_prefix_ig": float(stats["prefix_ig"].mean()),
        "avg_redundancy_penalty": float(stats["redundancy_penalty"].mean()),
    }


def method_settings(method: str, args) -> dict[str, float | bool | str]:
    settings: dict[str, float | bool | str] = {
        "lambda_ig": args.lambda_ig,
        "lambda_eff": 0.0,
        "reward_gated_eff": False,
        "objective": "grpo",
        "ig_mode": "raw_vr",
        "token_advantage_mode": "raw",
        "atgpo_adaptive_clip": False,
    }
    if method == "reward_tpo":
        return {
            **settings,
            "lambda_ig": 0.0,
            "objective": "tpo",
        }
    if method == "prefixig_grpo":
        return settings
    if method == "prefixig_grpo_turn_norm":
        return {
            **settings,
            "ig_mode": "turn_norm",
        }
    if method == "prefixig_grpo_turn_norm_vr":
        return {
            **settings,
            "ig_mode": "turn_norm_vr",
        }
    if method == "prefixig_atgpo":
        return {
            **settings,
            "lambda_eff": args.lambda_eff,
            "reward_gated_eff": True,
            "objective": "atgpo",
            "ig_mode": "turn_norm_vr",
            "token_advantage_mode": "turn_norm_vr",
            "atgpo_adaptive_clip": True,
        }
    if method == "prefixig_tpo":
        return {
            **settings,
            "objective": "tpo",
        }
    if method == "prefixig_tpo_rg_eff":
        return {
            **settings,
            "lambda_eff": args.lambda_eff,
            "reward_gated_eff": True,
            "objective": "tpo",
        }
    raise ValueError(f"Unknown method: {method}")


def train_method(method: str, seed: int, args) -> list[dict[str, float | int | str]]:
    mx.random.seed(seed)
    np.random.seed(seed)
    model = TinyCausalTransformer(
        vocab_size=ANSWER_OFFSET + args.answer_count,
        max_seq_len=1 + args.completion_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_mult=args.ffn_mult,
    )
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    settings = method_settings(method, args)
    method_args = argparse.Namespace(**vars(args))
    for key, value in settings.items():
        setattr(method_args, key, value)

    if method_args.objective == "grpo":
        loss_fn = make_grpo_loss(model, method_args)
    elif method_args.objective == "atgpo":
        loss_fn = make_atgpo_loss(model, method_args)
    else:
        loss_fn = make_tpo_loss(model, method_args)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    rows = []
    iterator = tqdm(range(1, args.episodes + 1), desc=f"{method}/seed{seed}", disable=args.quiet)
    for episode in iterator:
        batch = make_rollout_batch(
            model,
            method_args,
            reward_gated_eff=bool(method_args.reward_gated_eff),
            lambda_eff=float(method_args.lambda_eff),
        )
        for _ in range(args.epochs_per_batch):
            loss, grads = loss_and_grad(
                batch.prompts,
                batch.completions,
                batch.old_log_scores,
                batch.old_token_logps,
                batch.utilities,
                batch.token_advantages,
                batch.redundancy_penalty,
            )
            optimizer.update(model, grads)
            mx.eval(model.state, optimizer.state, loss)

        if episode == 1 or episode % args.eval_every == 0 or episode == args.episodes:
            metrics = evaluate(model, method_args)
            row = {
                "method": method,
                "seed": seed,
                "episode": episode,
                "loss": float(loss.item()),
                **metrics,
            }
            rows.append(row)
            if not args.quiet:
                iterator.set_postfix(
                    {
                        "reward": f"{metrics['reward']:.3f}",
                        "useful-red": f"{metrics['useful_minus_redundant']:+.3f}",
                    }
                )
    return rows


def print_summary(rows: list[dict[str, float | int | str]]) -> None:
    final_rows = {}
    for row in rows:
        final_rows[(row["method"], row["seed"])] = row
    by_method: dict[str, list[dict[str, float | int | str]]] = {}
    for row in final_rows.values():
        by_method.setdefault(str(row["method"]), []).append(row)

    headers = ["method", "reward", "useful", "redundant", "distractor", "useful-red"]
    table = []
    for method, method_rows in by_method.items():
        table.append(
            [
                method,
                f"{np.mean([float(r['reward']) for r in method_rows]):.3f}",
                f"{np.mean([float(r['useful_correct']) for r in method_rows]):.3f}",
                f"{np.mean([float(r['redundant_correct']) for r in method_rows]):.3f}",
                f"{np.mean([float(r['distractor_wrong']) for r in method_rows]):.3f}",
                f"{np.mean([float(r['useful_minus_redundant']) for r in method_rows]):+.3f}",
            ]
        )
    widths = [max(len(headers[i]), *(len(row[i]) for row in table)) for i in range(len(headers))]
    print("Toy RLVR final metrics")
    print("=" * 72)
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in table:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        default=(
            "reward_tpo,prefixig_grpo,prefixig_grpo_turn_norm,"
            "prefixig_grpo_turn_norm_vr,prefixig_atgpo,"
            "prefixig_tpo,prefixig_tpo_rg_eff"
        ),
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--answer-count", type=int, default=4)
    parser.add_argument("--completion-len", type=int, default=4)
    parser.add_argument(
        "--hop-mode",
        choices=["single", "multi", "mixed"],
        default="multi",
        help="Evidence structure: one-hop, two-hop, or prompt-mixed.",
    )
    parser.add_argument(
        "--retrieval-noise",
        choices=[
            "clean",
            "distractor25",
            "distractor50",
            "stale25",
            "stale50",
            "mixed25",
            "mixed50",
        ],
        default="clean",
        help="Deterministic evidence corruption used for noisy-retriever tests.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs-per-batch", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--lambda-ig", type=float, default=0.5)
    parser.add_argument("--lambda-eff", type=float, default=0.4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--atgpo-beta", type=float, default=0.02)
    parser.add_argument("--atgpo-redundancy-clip", type=float, default=0.0)
    parser.add_argument("--atgpo-adaptive-clip-scale", type=float, default=0.5)
    parser.add_argument("--atgpo-min-clip-mult", type=float, default=0.5)
    parser.add_argument("--atgpo-max-clip-mult", type=float, default=2.0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--output-csv", default="outputs/toy_prefixig_rlvr_curves.csv")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    all_rows = []
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    for method in methods:
        for offset in range(args.num_seeds):
            all_rows.extend(train_method(method, seed=args.seed + offset, args=args))

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    print_summary(all_rows)
    print(f"\nWrote curves to {output_path}")


if __name__ == "__main__":
    main()
