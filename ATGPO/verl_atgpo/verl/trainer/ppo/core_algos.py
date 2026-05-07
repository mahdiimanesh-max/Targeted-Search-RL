# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert (
            kl_ctrl.horizon > 0
        ), f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(
            init_kl_coef=kl_ctrl.kl_coef,
            target_kl=kl_ctrl.target_kl,
            horizon=kl_ctrl.horizon,
        )
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
# def compute_grpo_outcome_advantage(
#     token_level_rewards: torch.Tensor,
#     response_mask: torch.Tensor,
#     index: np.ndarray,
#     epsilon: float = 1e-6,
#     norm_adv_by_std_in_grpo: str = True,
#     em_scores: torch.Tensor = None,
#     tp_scores: torch.Tensor = None,
# ):
#     """
#     Compute advantage for GRPO, operating only on Outcome reward
#     (with only one scalar reward for each response).

#     Args:
#         token_level_rewards: `(torch.Tensor)`
#             shape is (bs, response_length)
#         response_mask: `(torch.Tensor)`
#             shape is (bs, response_length)
#         norm_adv_by_std_in_grpo: (bool)
#             whether to scale the GRPO advantage.
#             If True, the advantage is scaled by the std, as in the original GRPO.
#             If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).
#         em_scores: `(torch.Tensor)`, optional
#             shape is (bs,). If provided together with tp_scores, em and tp are
#             normalized separately and their advantages are summed.
#         tp_scores: `(torch.Tensor)`, optional
#             shape is (bs,). See em_scores.

#     Returns:
#         advantages: `(torch.Tensor)`
#             shape is (bs, response_length)
#         Returns: `(torch.Tensor)`
#             shape is (bs, response_length)
#     """

#     def _normalize_scores(raw_scores: torch.Tensor) -> torch.Tensor:
#         """Group-normalize a (bs,) score tensor by index, return (bs, resp_len) advantages."""
#         id2score = defaultdict(list)
#         id2mean = {}
#         id2std = {}
#         bsz = raw_scores.shape[0]
#         for i in range(bsz):
#             id2score[index[i]].append(raw_scores[i])
#         for idx in id2score:
#             if len(id2score[idx]) == 1:
#                 id2mean[idx] = torch.tensor(0.0, device=raw_scores.device)
#                 id2std[idx] = torch.tensor(1.0, device=raw_scores.device)
#             elif len(id2score[idx]) > 1:
#                 stacked_scores = torch.stack(id2score[idx])
#                 id2mean[idx] = torch.mean(stacked_scores)
#                 # 使用 unbiased=False 对齐标准 GRPO 的总体方差计算
#                 id2std[idx] = torch.std(stacked_scores, unbiased=False)
#             else:
#                 raise ValueError(f"no score in prompt index: {idx}")
#         normed = raw_scores.clone()
#         for i in range(bsz):
#             if norm_adv_by_std_in_grpo:
#                 normed[i] = (raw_scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
#             else:
#                 normed[i] = raw_scores[i] - id2mean[index[i]]
#         #return normed.unsqueeze(-1) * response_mask
#         return normed

#     with torch.no_grad():
#         if em_scores is not None and tp_scores is not None:
#             # Normalize em and tp separately, then sum the two advantages
#             adv_em = _normalize_scores(em_scores.float())
#             adv_tp = _normalize_scores(tp_scores.float())
#             scores_1d = adv_em + adv_tp * 0.2

#             # Batch-level 兜底归一化 (防止 KL 散度爆炸)
#             if norm_adv_by_std_in_grpo:
#                 b_mean = scores_1d.mean()
#                 b_std = scores_1d.std(unbiased=False) + epsilon
#                 scores_1d = (scores_1d - b_mean) / b_std
#             # 映射回 Token 级别
#             scores = scores_1d.unsqueeze(-1) * response_mask
            
#         else:
#             # Original behavior: normalize the sum of token-level rewards
#             print("Using original GRPO behavior.")
#             raw_scores = token_level_rewards.sum(dim=-1)
#             scores_1d = _normalize_scores(raw_scores)
#             scores = scores_1d.unsqueeze(-1) * response_mask

#     return scores, scores
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (
                    id2std[index[i]] + epsilon
                )
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: if True, normalize advantage by std within group

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[
                    index[i]
                ] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
):
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.tensor(id2score[idx])
                len_tensor = torch.tensor(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor
):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (
            (token_level_rewards * response_mask)
            .flip(dims=[-1])
            .cumsum(dim=-1)
            .flip(dims=[-1])
        )
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(
            loss_mask, dim=-1
        )  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    advantage_noise_sigma: float = 0.0,
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        advantage_noise_sigma (float, optional):
            Standard deviation for random noise applied to advantages.
            Noise range will be [-sigma, +sigma]. Defaults to 0.0 (no noise).
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower

def compute_policy_loss_entropy_balanced_clipping(
    old_log_prob,
    log_prob,
    entropy,
    advantages,
    response_mask,
    enable_entropy_balanced_clipping,
    enable_entropy_balanced_advantage,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: (torch.Tensor)
            shape: (bs, response_length)
        log_prob: (torch.Tensor)
            shape: (bs, response_length)
        advantages: (torch.Tensor)
            shape: (bs, response_length)
        response_mask: (torch.Tensor)
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" /
            "seq-mean-token-sum" /
            "seq-mean-token-mean" /
            "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """

    # 对entropy矩阵进行标准化：(entropy - mean) / std
    if enable_entropy_balanced_advantage and entropy is not None:
        # 计算有效位置的entropy的均值和标准差
        valid_entropy = entropy * response_mask.float()
        valid_entropy_flat = valid_entropy[response_mask.bool()]

        if len(valid_entropy_flat) > 0:
            entropy_mean = valid_entropy_flat.mean()
            entropy_std = valid_entropy_flat.std()

            # 避免除零，如果标准差为0则设为1
            if entropy_std == 0:
                entropy_std = 1.0

            # 标准化entropy
            entropy_normalized = (entropy - entropy_mean) / entropy_std

            # 使用标准化后的entropy来调整advantages（detach版本，不参与梯度计算）
            advantages = advantages * (1 + 0.2 * entropy_normalized.detach())

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    # FIX: Convert the minimum bound to a tensor to match torch.clamp requirements
    if enable_entropy_balanced_clipping:
        min_bound = torch.full_like(ratio, 1 - cliprange_low)
        max_bound = (1 + cliprange_high) / ratio.detach() * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, min_bound, max_bound)
    else:
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1 - cliprange_low, 1 + cliprange_high
        )
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    loss_agg_mode: str = "seq-mean-token-mean",
):
    """
    Compute the clipped policy objective and related metrics for GSPO.
    See https://arxiv.org/pdf/2507.18071 for more details.
    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - cliprange_low, 1 + cliprange_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower




def compute_policy_loss_gspo_turn(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    loss_agg_mode: str = "seq-mean-token-mean",
    dynamic_clip: bool = False,
    ig_clip_scale: torch.Tensor = None,
):
    """
    Turn-level importance-sampling PPO loss (ATPO-style).

    Instead of token-level IS ratio, computes a turn-level IS ratio:
        r_turn = exp( mean_{tokens in turn}( log π_θ - log π_old ) )
    and applies PPO clipping per token using that turn-level ratio.

    When dynamic_clip=True and ig_clip_scale is provided, the clip range
    is scaled per-token by ig_clip_scale (from IG-adaptive clip):
        effective_clip_low  = cliprange_low  * ig_clip_scale
        effective_clip_high = cliprange_high * ig_clip_scale

    Args:
        old_log_prob: (bsz, response_length)
        log_prob:     (bsz, response_length)
        advantages:   (bsz, response_length) — per-token advantage (constant within turn)
        response_mask: (bsz, response_length) — 1 for valid response tokens
        cliprange:     base clip range (fallback for low/high)
        cliprange_low: lower clip range
        cliprange_high: upper clip range
        loss_agg_mode: aggregation mode
        dynamic_clip:  whether to apply IG-adaptive clip scaling
        ig_clip_scale: (bsz, response_length) per-token clip scale factor (from v1d)

    Returns:
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
    """
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob  # (bsz, seq_len)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # ── Turn-level IS ratio ──────────────────────────────────────────
    # Detect turn boundaries from advantage changes: a new turn starts
    # where the advantage value changes from the previous token.
    bsz, seq_len = advantages.shape

    # Compute per-token turn-mean of log-ratio, then exponentiate
    # For each token, the ratio uses the turn-level average log-ratio
    turn_log_ratio = torch.zeros_like(negative_approx_kl)

    for s in range(bsz):
        mask_s = response_mask[s]  # (seq_len,)
        adv_s = advantages[s]
        logr_s = negative_approx_kl[s]

        # Find valid positions
        valid_pos = mask_s.nonzero(as_tuple=True)[0]
        if len(valid_pos) == 0:
            continue

        # Group consecutive tokens by same advantage value (= same turn)
        turn_start = 0
        for i in range(1, len(valid_pos) + 1):
            # Turn boundary: advantage changes or we reach the end
            if i == len(valid_pos) or adv_s[valid_pos[i]] != adv_s[valid_pos[turn_start]]:
                # Compute mean log-ratio for this turn
                turn_positions = valid_pos[turn_start:i]
                turn_mean_lr = logr_s[turn_positions].mean()
                # Broadcast to all tokens in this turn
                turn_log_ratio[s, turn_positions] = turn_mean_lr
                turn_start = i

    # Turn-level ratio: for gradient, use log_prob - log_prob.detach() + turn_mean.detach()
    # This preserves per-token gradient while using turn-level IS ratio
    ratio_log = log_prob - log_prob.detach() + turn_log_ratio.detach()
    ratio_log = torch.clamp(ratio_log, max=10.0)
    ratio = torch.exp(ratio_log)

    # ── PPO clipping with optional IG-adaptive clip scale ────────────
    pg_losses1 = -advantages * ratio

    if dynamic_clip and ig_clip_scale is not None:
        # Per-token adaptive clip range
        effective_low = cliprange_low * ig_clip_scale
        effective_high = cliprange_high * ig_clip_scale
        clamped_ratio = torch.max(torch.min(ratio, 1.0 + effective_high), 1.0 - effective_low)
    else:
        clamped_ratio = torch.clamp(ratio, 1.0 - cliprange_low, 1.0 + cliprange_high)

    pg_losses2 = -advantages * clamped_ratio
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_losses.device)

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(
        vpreds, values - cliprange_value, values + cliprange_value
    )
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(
        loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )
    vf_clipfrac = verl_F.masked_mean(
        torch.gt(vf_losses2, vf_losses1).float(), response_mask
    )
    return vf_loss, vf_clipfrac


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(
        scores: torch.Tensor, reweight_method: str, weight_pow: float
    ) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where(
                (scores == max_score) | (scores == min_score), 1.0, 0.0
            )
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {
        key: tensor[sample_indices] for key, tensor in data.batch.items()
    }

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def _compute_turn_level_advantage(
    normalized_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    bsz: int,
    seq_len: int,
    device: torch.device,
    turn_boundary_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Turn-level discounted accumulation + broadcast.

    For each sample:
      1. Identify turn boundaries (positions where reward is non-zero,
         or from turn_boundary_mask if provided).
      2. Backward discounted accumulation: A_i = r_i + gamma * A_{i+1}
      3. Broadcast A_i to every response token in turn i.

    Args:
        normalized_rewards:  (bsz, seq_len) normalized reward tensor
        response_mask:       (bsz, seq_len) 1 for valid response tokens
        gamma:               discount factor
        bsz:                 batch size
        seq_len:             sequence length
        device:              torch device
        turn_boundary_mask:  optional (bsz, seq_len) bool mask marking turn-end
                             positions. When provided, used instead of the
                             `!= 0` heuristic so that turns whose normalised
                             reward happens to be exactly 0 are not missed.

    Returns:
        discounted_returns: (bsz, seq_len) advantage broadcast to all tokens
    """
    discounted_returns = torch.zeros(bsz, seq_len, device=device, dtype=normalized_rewards.dtype)

    for sample_idx in range(bsz):
        sample_rewards = normalized_rewards[sample_idx]   # (seq_len,)
        sample_mask    = response_mask[sample_idx]        # (seq_len,)

        # Step 1: locate turn-end positions
        if turn_boundary_mask is not None:
            reward_positions = turn_boundary_mask[sample_idx].nonzero(as_tuple=True)[0].tolist()
        else:
            reward_positions = (sample_rewards != 0).nonzero(as_tuple=True)[0].tolist()

        if len(reward_positions) == 0:
            continue

        # Step 2: backward discounted accumulation
        turn_data = []          # [(reward_pos, turn_advantage), ...]
        next_turn_adv = 0.0

        for pos in reversed(reward_positions):
            turn_reward = sample_rewards[pos].item()
            turn_adv    = turn_reward + gamma * next_turn_adv
            turn_data.append((pos, turn_adv))
            next_turn_adv = turn_adv

        turn_data.reverse()     # restore forward order

        # Step 3: broadcast to all tokens in each turn
        prev_end = 0
        for reward_pos, adv in turn_data:
            for t in range(prev_end, reward_pos + 1):
                if sample_mask[t] == 1:
                    discounted_returns[sample_idx, t] = adv
            prev_end = reward_pos + 1

    return discounted_returns


def compute_igpo_step_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    gamma: float = 1.0,
    info_gain_norm_mode: str = "joint",
    adv_rescale_alpha: float = 1.0,
) -> tuple:
    """
    IGPO step-level advantage estimation.

    Token-level reward layout expected (set by _compute_igpo_reward in ray_trainer):
        [0, ..., ig_1, 0, ..., ig_2, 0, ..., outcome_score]
                  ^                ^               ^
            turn-1 last tok   turn-2 last tok   final answer last tok
                                                (em_score in this project)

    Two masks are built internally:
      - outcome_mask : last valid response token per sequence  (carries em_score)
      - ig_mask      : any other non-zero reward position      (carries info-gain)

    Args:
        token_level_rewards: (bsz, response_length)
        response_mask:       (bsz, response_length)
        index:               (bsz,) prompt-group ids for within-group normalisation
        epsilon:             numerical stability constant
        norm_adv_by_std_in_grpo: whether to divide by within-group std
        gamma:               discount factor across turns (default 1.0)
        info_gain_norm_mode: "joint"       – normalise outcome and ig rewards together
                             "separate"    – normalise outcome and ig rewards independently
                             "turn-group"  – normalise outcome per-prompt, ig per-(prompt, turn_index)
                             "turn-group-v1d" – v1d advantage: A_t = α × D_t/√n + norm_outcome
                                               with fixed α (adv_rescale_alpha) and
                                               IG-adaptive clip scale via sigmoid
        adv_rescale_alpha:   fixed α coefficient for v1d advantage (default 1.0, typical 0.3)

    Returns:
        advantages: (bsz, response_length)
        returns:    same as advantages (for API compatibility)
    """
    bsz, seq_len = token_level_rewards.shape
    device = token_level_rewards.device

    with torch.no_grad():
        # ── Step 1: build outcome_mask and ig_mask ─────────────────────────────
        # outcome_mask : last valid token of each sequence (carries the em_score)
        # ig_mask      : any other non-zero reward position (carries info-gain)
        last_valid_pos = (seq_len - 1) - response_mask.flip(dims=[1]).to(torch.long).argmax(dim=1)
        position_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        outcome_mask = (position_indices == last_valid_pos.unsqueeze(1)) & (response_mask == 1)
        ig_mask = (response_mask == 1) & (~outcome_mask) & (token_level_rewards != 0)

        # ── Step 2: build consecutive group ids ────────────────────────────────
        unique_indices, inverse_indices = np.unique(index, return_inverse=True)
        group_ids = torch.tensor(inverse_indices, device=device, dtype=torch.long)   # (bsz,)
        num_groups = len(unique_indices)
        group_ids_expanded = group_ids.unsqueeze(1).expand(-1, seq_len)              # (bsz, seq_len)

        # ── Step 3: within-group statistics ────────────────────────────────────
        def compute_group_stats(mask: torch.Tensor):
            """Return (group_mean, group_std) tensors of shape (num_groups,)."""
            flat_mask    = mask.reshape(-1)
            flat_rewards = token_level_rewards.reshape(-1)
            flat_groups  = group_ids_expanded.reshape(-1)

            valid_idx = flat_mask.nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                return (torch.zeros(num_groups, device=device),
                        torch.ones(num_groups, device=device))

            valid_rewards = flat_rewards[valid_idx]
            valid_groups  = flat_groups[valid_idx]

            group_sum   = torch.zeros(num_groups, device=device).scatter_add_(
                0, valid_groups, valid_rewards)
            group_count = torch.zeros(num_groups, device=device).scatter_add_(
                0, valid_groups, torch.ones_like(valid_rewards))
            group_mean  = group_sum / group_count.clamp(min=1.0)

            sq_diff   = (valid_rewards - group_mean[valid_groups]) ** 2
            group_var = torch.zeros(num_groups, device=device).scatter_add_(
                0, valid_groups, sq_diff) / group_count.clamp(min=1.0)
            group_std = torch.sqrt(group_var + 1e-8)
            group_std = torch.where(group_count <= 1,
                                    torch.ones_like(group_std), group_std)
            return group_mean, group_std

        # ── Step 4: normalise rewards ───────────────────────────────────────────
        normalized_rewards = torch.zeros_like(token_level_rewards)

        if info_gain_norm_mode == "turn-group-v1d":
            # ── turn-group-v1d: v1d advantage formula ─────────────────────
            # A_t = α × D_t / √n + norm_outcome  (fixed α from config)
            # Also computes per-turn IG-adaptive clip scale:
            #   clip_scale = 1 + 0.3 * (2*sigmoid(normed_ig_t) - 1)
            # Outcome turn uses clip_scale = 1.0 (no adjustment).

            # Step 4a: normalise IG per-(prompt, turn_index) to get normed_ig_t
            ig_turn_index = torch.full((bsz, seq_len), -1, device=device, dtype=torch.long)
            max_turns = 0
            for i in range(bsz):
                ig_positions = ig_mask[i].nonzero(as_tuple=True)[0]
                for t_idx, pos in enumerate(ig_positions):
                    ig_turn_index[i, pos] = t_idx
                if len(ig_positions) > 0:
                    max_turns = max(max_turns, len(ig_positions))

            # normed_ig will hold z-score normalised IG at each IG position
            normed_ig_values = torch.zeros_like(token_level_rewards)

            if max_turns > 0:
                num_contrastive_groups = num_groups * max_turns

                flat_ig_mask = ig_mask.reshape(-1)
                flat_raw_ig = token_level_rewards.reshape(-1)
                flat_prompt_groups = group_ids_expanded.reshape(-1)
                flat_turn_index = ig_turn_index.reshape(-1)

                valid_ig_idx = flat_ig_mask.nonzero(as_tuple=True)[0]

                if valid_ig_idx.numel() > 0:
                    valid_ig_rewards = flat_raw_ig[valid_ig_idx]
                    valid_ig_prompt_groups = flat_prompt_groups[valid_ig_idx]
                    valid_ig_turn_indices = flat_turn_index[valid_ig_idx]

                    # Composite group: prompt_group * max_turns + turn_index
                    valid_ig_composite = valid_ig_prompt_groups * max_turns + valid_ig_turn_indices

                    # Per-composite-group mean and std
                    cg_sum = torch.zeros(num_contrastive_groups, device=device).scatter_add_(
                        0, valid_ig_composite, valid_ig_rewards)
                    cg_count = torch.zeros(num_contrastive_groups, device=device).scatter_add_(
                        0, valid_ig_composite, torch.ones_like(valid_ig_rewards))
                    cg_mean = cg_sum / cg_count.clamp(min=1.0)

                    cg_sq_diff = (valid_ig_rewards - cg_mean[valid_ig_composite]) ** 2
                    cg_var = torch.zeros(num_contrastive_groups, device=device).scatter_add_(
                        0, valid_ig_composite, cg_sq_diff) / cg_count.clamp(min=1.0)
                    cg_std = torch.sqrt(cg_var + 1e-8)
                    cg_std = torch.where(cg_count <= 1, torch.ones_like(cg_std), cg_std)

                    # Build full (bsz, seq_len) composite group id tensor
                    contrastive_group_ids = group_ids_expanded * max_turns + ig_turn_index
                    contrastive_group_ids = contrastive_group_ids.clamp(min=0, max=num_contrastive_groups - 1)

                    norm_ig = token_level_rewards - cg_mean[contrastive_group_ids]
                    if norm_adv_by_std_in_grpo:
                        norm_ig = norm_ig / (cg_std[contrastive_group_ids] + epsilon)
                    normed_ig_values = torch.where(ig_mask, norm_ig, normed_ig_values)

            # Step 4b: normalise outcome per-prompt
            outcome_mean, outcome_std = compute_group_stats(outcome_mask)
            norm_outcome = token_level_rewards - outcome_mean[group_ids_expanded]
            if norm_adv_by_std_in_grpo:
                norm_outcome = norm_outcome / (outcome_std[group_ids_expanded] + epsilon)

            # Extract per-sample outcome advantage scalar
            outcome_adv_per_sample = torch.zeros(bsz, device=device)
            for i in range(bsz):
                oc_pos = outcome_mask[i].nonzero(as_tuple=True)[0]
                if len(oc_pos) > 0:
                    outcome_adv_per_sample[i] = norm_outcome[i, oc_pos[0]].item()

            # Step 4c: compute v1d advantage per sample
            # A_t = α × D_t / √n + norm_outcome
            # D_t = Σ_{j=t}^{n_ig} γ^{j-t} × normed_ig_j  (discounted cumulative)
            # Also compute IG-adaptive clip scale:
            #   clip_scale_t = 1 + 0.3 * (2*sigmoid(normed_ig_t) - 1)
            #   bounded in (0.7, 1.3)
            final_adv = torch.zeros(bsz, seq_len, device=device)
            ig_clip_scale = torch.ones(bsz, seq_len, device=device)  # default = 1.0

            alpha = adv_rescale_alpha  # fixed α from config (e.g., 0.3)

            for s in range(bsz):
                ig_positions = ig_mask[s].nonzero(as_tuple=True)[0].tolist()
                outcome_pos = last_valid_pos[s].item()
                outcome_val = outcome_adv_per_sample[s].item()
                n_ig = len(ig_positions)

                # Collect normed IG values at each turn
                ig_values = [normed_ig_values[s, p].item() for p in ig_positions]

                # Compute discounted cumulative D_t
                ig_discounted = []
                for t in range(n_ig):
                    D_t = 0.0
                    for j in range(t, n_ig):
                        D_t += (gamma ** (j - t)) * ig_values[j]
                    n_terms = n_ig - t
                    D_t_normed = D_t / (n_terms ** 0.5) if n_terms > 0 else 0.0
                    ig_discounted.append(D_t_normed)

                # Compute IG-adaptive clip scale per IG turn (sigmoid option B)
                # clip_scale_t = 1 + 0.3 * (2*sigmoid(normed_ig_t) - 1)
                # normed_ig_t > 0 (good search) → enlarge clip, < 0 → shrink clip
                # Bounded in (0.7, 1.3) by construction
                for t_idx, pos in enumerate(ig_positions):
                    normed_ig_t = ig_values[t_idx]
                    sig = torch.sigmoid(torch.tensor(normed_ig_t, device=device)).item()
                    clip_s = 1.0 + 0.3 * (2.0 * sig - 1.0)
                    ig_clip_scale[s, pos] = clip_s
                # Outcome turn: clip_scale stays 1.0 (already initialized)

                # Assign advantages: broadcast to all tokens in each turn segment
                all_boundaries = sorted(ig_positions + [outcome_pos])
                prev_end = 0
                for bp in all_boundaries:
                    if bp == outcome_pos:
                        adv_val = outcome_val
                    else:
                        ig_idx = ig_positions.index(bp)
                        adv_val = alpha * ig_discounted[ig_idx] + outcome_val
                    for t in range(prev_end, bp + 1):
                        if response_mask[s, t] == 1:
                            final_adv[s, t] = adv_val
                            # Broadcast clip_scale from boundary to all tokens in this turn
                            ig_clip_scale[s, t] = ig_clip_scale[s, bp]
                    prev_end = bp + 1

            print(f"[IGPO v1d] alpha={alpha:.3f}, "
                  f"outcome_adv mean={outcome_adv_per_sample.mean().item():.4f}, "
                  f"std={outcome_adv_per_sample.std().item():.4f}, "
                  f"ig_clip_scale mean={ig_clip_scale[response_mask.bool()].mean().item():.4f}, "
                  f"std={ig_clip_scale[response_mask.bool()].std().item():.4f}")

            return final_adv, final_adv, ig_clip_scale

        elif info_gain_norm_mode == "turn-group":
            # ── Turn-group: discount first, then normalise ────────────────────
            # New pipeline:
            #   Step 4a-pre: auto-scale raw IG to match outcome's magnitude
            #   Step 4a: backward discounted accumulation on SCALED IG
            #   Step 4b: normalise discounted IG per-(prompt, turn_index)
            #   Step 4c: normalise outcome per-prompt
            # This avoids the "normalise then accumulate" problem where
            # summing z-scores inflates variance for early turns.

            # Step 4a-pre: compute batch-level σ_ig and σ_outcome, then scale IG
            # so that IG and outcome live in the same magnitude before accumulation.
            all_raw_ig = token_level_rewards[ig_mask]           # flat 1-D
            all_raw_oc = token_level_rewards[outcome_mask]      # flat 1-D

            if all_raw_ig.numel() > 1:
                sigma_ig = all_raw_ig.std(unbiased=False).clamp(min=epsilon)
            else:
                sigma_ig = torch.tensor(1.0, device=device)

            if all_raw_oc.numel() > 1:
                sigma_outcome = all_raw_oc.std(unbiased=False).clamp(min=epsilon)
            else:
                sigma_outcome = torch.tensor(1.0, device=device)

            ig_scale = sigma_outcome / sigma_ig
            # Apply the scale to a working copy so that raw token_level_rewards stays intact
            scaled_rewards = token_level_rewards.clone()
            scaled_rewards[ig_mask] = token_level_rewards[ig_mask] * ig_scale

            print(f"[IGPO turn-group] sigma_ig={sigma_ig.item():.4f}, "
                  f"sigma_outcome={sigma_outcome.item():.4f}, "
                  f"ig_scale={ig_scale.item():.4f}")

            # Step 4a: determine turn_index for every IG position per sample,
            #          then do backward discounted accumulation on SCALED IG values.
            ig_turn_index = torch.full((bsz, seq_len), -1, device=device, dtype=torch.long)
            max_turns = 0
            # discounted_ig stores the discounted cumulative IG at each IG position
            discounted_ig = torch.zeros_like(token_level_rewards)

            for i in range(bsz):
                ig_positions = ig_mask[i].nonzero(as_tuple=True)[0]  # sorted ascending
                for t_idx, pos in enumerate(ig_positions):
                    ig_turn_index[i, pos] = t_idx
                if len(ig_positions) > 0:
                    max_turns = max(max_turns, len(ig_positions))

                # Backward discounted accumulation on SCALED IG, seeded with outcome
                # so that outcome reward propagates back to search turns via discount.
                # The chain is: ig_0 <- ig_1 <- ... <- ig_K <- outcome
                # Now IG and outcome are in the same magnitude thanks to ig_scale.
                if len(ig_positions) > 0:
                    # Seed the accumulation with the discounted outcome reward
                    # (outcome is NOT scaled — it keeps its original magnitude;
                    #  IG has been scaled UP/DOWN to match outcome's magnitude)
                    outcome_pos = outcome_mask[i].nonzero(as_tuple=True)[0]
                    if len(outcome_pos) > 0:
                        next_val = gamma * scaled_rewards[i, outcome_pos[0]].item()
                    else:
                        next_val = 0.0
                    for pos in reversed(ig_positions):
                        scaled_ig = scaled_rewards[i, pos].item()
                        accumulated = scaled_ig + gamma * next_val
                        discounted_ig[i, pos] = accumulated
                        next_val = accumulated

            # Step 4b: normalise discounted IG per-(prompt, turn_index).
            if max_turns > 0:
                num_contrastive_groups = num_groups * max_turns

                flat_ig_mask = ig_mask.reshape(-1)
                flat_disc_ig = discounted_ig.reshape(-1)
                flat_prompt_groups = group_ids_expanded.reshape(-1)
                flat_turn_index = ig_turn_index.reshape(-1)

                valid_ig_idx = flat_ig_mask.nonzero(as_tuple=True)[0]

                if valid_ig_idx.numel() > 0:
                    valid_ig_rewards = flat_disc_ig[valid_ig_idx]
                    valid_ig_prompt_groups = flat_prompt_groups[valid_ig_idx]
                    valid_ig_turn_indices = flat_turn_index[valid_ig_idx]

                    # Composite group: prompt_group * max_turns + turn_index
                    valid_ig_composite = valid_ig_prompt_groups * max_turns + valid_ig_turn_indices

                    # Per-composite-group mean and std
                    cg_sum = torch.zeros(num_contrastive_groups, device=device).scatter_add_(
                        0, valid_ig_composite, valid_ig_rewards)
                    cg_count = torch.zeros(num_contrastive_groups, device=device).scatter_add_(
                        0, valid_ig_composite, torch.ones_like(valid_ig_rewards))
                    cg_mean = cg_sum / cg_count.clamp(min=1.0)

                    cg_sq_diff = (valid_ig_rewards - cg_mean[valid_ig_composite]) ** 2
                    cg_var = torch.zeros(num_contrastive_groups, device=device).scatter_add_(
                        0, valid_ig_composite, cg_sq_diff) / cg_count.clamp(min=1.0)
                    cg_std = torch.sqrt(cg_var + 1e-8)
                    cg_std = torch.where(cg_count <= 1, torch.ones_like(cg_std), cg_std)

                    # Build full (bsz, seq_len) composite group id tensor
                    contrastive_group_ids = group_ids_expanded * max_turns + ig_turn_index
                    contrastive_group_ids = contrastive_group_ids.clamp(min=0, max=num_contrastive_groups - 1)

                    norm_ig = discounted_ig - cg_mean[contrastive_group_ids]
                    if norm_adv_by_std_in_grpo:
                        norm_ig = norm_ig / (cg_std[contrastive_group_ids] + epsilon)
                    normalized_rewards = torch.where(ig_mask, norm_ig, normalized_rewards)

            # Step 4c: normalise outcome per-prompt → outcome advantage
            outcome_mean, outcome_std = compute_group_stats(outcome_mask)
            norm_outcome = token_level_rewards - outcome_mean[group_ids_expanded]
            if norm_adv_by_std_in_grpo:
                norm_outcome = norm_outcome / (outcome_std[group_ids_expanded] + epsilon)
            normalized_rewards = torch.where(outcome_mask, norm_outcome, normalized_rewards)

            # Step 4d: inject outcome advantage into every IG turn (1:1.5 uniform)
            outcome_adv_weight = 1.5  # outcome advantage injection weight
            # Extract per-sample outcome advantage scalar
            outcome_adv_per_sample = torch.zeros(bsz, device=device)
            for i in range(bsz):
                oc_pos = outcome_mask[i].nonzero(as_tuple=True)[0]
                if len(oc_pos) > 0:
                    outcome_adv_per_sample[i] = normalized_rewards[i, oc_pos[0]].item()

            # Add weighted outcome advantage to every IG position
            # outcome_adv_broadcast: (bsz,) -> (bsz, seq_len)
            outcome_adv_broadcast = outcome_adv_per_sample.unsqueeze(1).expand_as(normalized_rewards)
            normalized_rewards = torch.where(
                ig_mask,
                normalized_rewards + outcome_adv_weight * outcome_adv_broadcast,
                normalized_rewards,
            )

            print(f"[IGPO turn-group] sigma_ig={sigma_ig.item():.4f}, "
                  f"sigma_outcome={sigma_outcome.item():.4f}, "
                  f"ig_scale={ig_scale.item():.4f}, "
                  f"outcome_adv mean={outcome_adv_per_sample.mean().item():.4f}, "
                  f"std={outcome_adv_per_sample.std().item():.4f}")

        elif info_gain_norm_mode == "separate":
            # Normalise outcome (em_score) and ig rewards independently
            outcome_mean, outcome_std = compute_group_stats(outcome_mask)
            norm_outcome = token_level_rewards - outcome_mean[group_ids_expanded]
            if norm_adv_by_std_in_grpo:
                norm_outcome = norm_outcome / (outcome_std[group_ids_expanded] + epsilon)
            normalized_rewards = torch.where(outcome_mask, norm_outcome, normalized_rewards)

            ig_mean, ig_std = compute_group_stats(ig_mask)
            norm_ig = token_level_rewards - ig_mean[group_ids_expanded]
            if norm_adv_by_std_in_grpo:
                norm_ig = norm_ig / (ig_std[group_ids_expanded] + epsilon)
            normalized_rewards = torch.where(ig_mask, norm_ig, normalized_rewards)

        else:  # "joint" (default)
            joint_mask = outcome_mask | ig_mask
            g_mean, g_std = compute_group_stats(joint_mask)
            norm_val = token_level_rewards - g_mean[group_ids_expanded]
            if norm_adv_by_std_in_grpo:
                norm_val = norm_val / (g_std[group_ids_expanded] + epsilon)
            normalized_rewards = torch.where(joint_mask, norm_val, normalized_rewards)

        # ── Step 5: turn-level broadcast ──────────────────────────────────────
        # Build turn_boundary_mask to avoid missing turns whose normalised
        # reward happens to be exactly 0 after group normalisation.
        turn_boundary_mask = outcome_mask | ig_mask  # (bsz, seq_len)

        if info_gain_norm_mode == "turn-group":
            # For turn-group mode, discounted accumulation was already done
            # in Step 4a on raw IG values (excluding outcome). Here we only
            # need to broadcast each turn's normalised advantage to its tokens.
            # Pass gamma=0.0 to _compute_turn_level_advantage so it performs
            # pure broadcast without any further accumulation.
            discounted_returns = _compute_turn_level_advantage(
                normalized_rewards=normalized_rewards,
                response_mask=response_mask,
                gamma=0.0,
                bsz=bsz,
                seq_len=seq_len,
                device=device,
                turn_boundary_mask=turn_boundary_mask,
            )
        else:
            # For "joint" and "separate" modes, use the original formulation:
            # backward discounted accumulation across all turns (IG + outcome).
            discounted_returns = _compute_turn_level_advantage(
                normalized_rewards=normalized_rewards,
                response_mask=response_mask,
                gamma=gamma,
                bsz=bsz,
                seq_len=seq_len,
                device=device,
                turn_boundary_mask=turn_boundary_mask,
            )

        # No safety clamp — match IGPO original implementation

    return discounted_returns, discounted_returns, None
