# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    IGPO = "igpo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def find_search_segments_by_text(resp_ids_list: list, tokenizer) -> list:
    """
    Decode token-id sequence character by character, then use text matching to
    find all <search>...</search> intervals.

    Returns:
        list of (seg_start, seg_end) tuples where
          seg_start: index of the first token of <search>
          seg_end:   index of the last  token of </search>
    """
    SEARCH_START_STR = "<search>"
    SEARCH_END_STR   = "</search>"

    pad_id    = tokenizer.pad_token_id
    valid_len = len(resp_ids_list)
    while valid_len > 0 and resp_ids_list[valid_len - 1] == pad_id:
        valid_len -= 1
    if valid_len == 0:
        return []

    # Build per-token character start positions
    token_char_starts = []
    chars = []
    for tok_id in resp_ids_list[:valid_len]:
        tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
        token_char_starts.append(len(chars))
        chars.extend(list(tok_str))
    full_text = "".join(chars)
    token_char_starts.append(len(chars))  # sentinel

    import re
    s_starts_char = [m.start() for m in re.finditer(re.escape(SEARCH_START_STR), full_text)]
    e_starts_char = [m.start() for m in re.finditer(re.escape(SEARCH_END_STR),   full_text)]

    def char_to_token(char_pos):
        for t_idx in range(valid_len):
            if token_char_starts[t_idx] <= char_pos < token_char_starts[t_idx + 1]:
                return t_idx
        return valid_len - 1

    s_token_positions = [char_to_token(c) for c in s_starts_char]
    e_token_positions = [char_to_token(c) for c in e_starts_char]

    se_text_len = len(SEARCH_END_STR)

    def end_last_token(e_char_start):
        e_char_end = e_char_start + se_text_len - 1
        return char_to_token(e_char_end)

    segments = []
    used_end = set()
    for s_tok in s_token_positions:
        matched_e_tok  = None
        matched_e_char = None
        for e_char, e_tok in zip(e_starts_char, e_token_positions):
            if e_tok > s_tok and e_char not in used_end:
                matched_e_tok  = end_last_token(e_char)
                matched_e_char = e_char
                break
        if matched_e_tok is not None:
            used_end.add(matched_e_char)
            segments.append((s_tok, matched_e_tok))
    return segments


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # If em_score and tp_score are both available, normalize them separately
        #em_scores_tensor = None
        #tp_scores_tensor = None
        #if ("em_score" in data.non_tensor_batch and "tp_score" in data.non_tensor_batch):
            #em_scores_tensor = torch.tensor(data.non_tensor_batch["em_score"], dtype=torch.float32)
            #tp_scores_tensor = torch.tensor(data.non_tensor_batch["tp_score"], dtype=torch.float32)
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            #em_scores=em_scores_tensor,
            #tp_scores=tp_scores_tensor,
        )
        advantages = torch.clamp(advantages, min=-5.0, max=5.0)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.OPO:
        advantages, returns = core_algos.compute_opo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.IGPO:
        # IGPO step-level advantage: per-turn discounted accumulation + broadcast.
        # token_level_rewards must already contain info-gain rewards at turn boundaries
        # (written by _compute_igpo_reward before compute_advantage is called).
        igpo_gamma = kwargs.get("igpo_gamma", 1.0)
        igpo_norm_mode = kwargs.get("igpo_norm_mode", "joint")
        igpo_adv_rescale_alpha = kwargs.get("igpo_adv_rescale_alpha", 1.0)
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        advantages, returns, ig_clip_scale = core_algos.compute_igpo_step_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            gamma=igpo_gamma,
            info_gain_norm_mode=igpo_norm_mode,
            adv_rescale_alpha=igpo_adv_rescale_alpha,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if ig_clip_scale is not None:
            data.batch["ig_clip_scale"] = ig_clip_scale
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.IGPO,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]

            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True, is_validate=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source = test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            data_source_lst.append(data_source)
            reward_extra_infos_dict["data_source"].extend(data_source) 

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        # Calculate TC (average tool calls) and TP (correct samples / total tool calls)
        import re
        tool_call_counts = []
        for output_text in sample_outputs:
            # Count <search> tags in each output
            count = len(re.findall(r'<search>', output_text, re.IGNORECASE))
            tool_call_counts.append(count)
        
        total_tool_calls = sum(tool_call_counts)
        num_samples = len(sample_outputs)
        
        # TC: average tool calls per sample
        tc_metric = total_tool_calls / num_samples if num_samples > 0 else 0.0
        
        # TP: correct samples / total tool calls
        # Get em_score from reward_extra_infos_dict if available
        correct_samples = 0
        if "em_score" in reward_extra_infos_dict and len(reward_extra_infos_dict["em_score"]) > 0:
            em_scores = reward_extra_infos_dict["em_score"]
            correct_samples = sum([1 for score in em_scores if score == 1.0])
        elif "acc" in reward_extra_infos_dict and len(reward_extra_infos_dict["acc"]) > 0:
            # Fallback to acc if em_score is not available
            acc_scores = reward_extra_infos_dict["acc"]
            correct_samples = sum([1 for score in acc_scores if score == 1.0])
        
        tp_metric = correct_samples / total_tool_calls if total_tool_calls > 0 else 0.0

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val
        
        # Add TC and TP metrics to metric_dict
        metric_dict["val-core/TC"] = tc_metric
        metric_dict["val-core/TP"] = tp_metric
        
        # Also add detailed statistics for debugging
        metric_dict["val-aux/total_tool_calls"] = total_tool_calls
        metric_dict["val-aux/correct_samples"] = correct_samples
        metric_dict["val-aux/num_samples"] = num_samples

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _pad_batch_to_match(self, batch1, batch2):
        """
        对齐两个DataProto batch的序列长度，使它们可以安全地concat
        
        Args:
            batch1: 第一个DataProto batch (可能是None)
            batch2: 第二个DataProto batch
            
        Returns:
            对齐后的 (batch1, batch2)
        """
        if batch1 is None:
            return batch1, batch2
        
        # 检测需要对齐的所有tensor及其目标长度
        alignment_specs = {}
        
        for key in batch1.batch.keys():
            if key in batch2.batch:
                tensor1 = batch1.batch[key]
                tensor2 = batch2.batch[key]
                
                if hasattr(tensor1, 'shape') and hasattr(tensor2, 'shape') and len(tensor1.shape) >= 2:
                    if tensor1.shape[1] != tensor2.shape[1]:
                        max_len = max(tensor1.shape[1], tensor2.shape[1])
                        alignment_specs[key] = max_len
        
        # 如果需要对齐，则执行padding
        if alignment_specs:
            #print(f"Aligning tensors: {list(alignment_specs.keys())}")
            batch1 = self._pad_batch(batch1, alignment_specs)
            batch2 = self._pad_batch(batch2, alignment_specs)
    
        return batch1, batch2


    def _pad_batch(self, batch, alignment_specs):
        """
        对DataProto batch中的tensor进行padding

        Args:
            batch: DataProto对象
            alignment_specs: dict，key为tensor名称，value为目标长度
            
        Returns:
            padding后的DataProto对象
        """
        #padded_batch = batch.clone()
        padded_batch = batch

        for key, target_len in alignment_specs.items():
            if key not in padded_batch.batch:
                continue
                
            tensor = padded_batch.batch[key]
            if not hasattr(tensor, 'shape') or len(tensor.shape) < 2:
                continue
                
            current_len = tensor.shape[1]
            if current_len >= target_len:
                continue
            
            pad_size = target_len - current_len
            
            # 根据tensor名称确定padding策略
            if key in ['input_ids', 'prompts']:
                # input_ids用0 (pad_token_id) padding
                pad_value = 0
                pad_shape = [tensor.shape[0], pad_size] + list(tensor.shape[2:])
                padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                padded_batch.batch[key] = torch.cat([tensor, padding], dim=1)
                
            elif key in ['attention_mask', 'loss_mask']:
                # attention_mask用0 padding (表示padding位置)
                pad_value = 0
                pad_shape = [tensor.shape[0], pad_size] + list(tensor.shape[2:])
                padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                padded_batch.batch[key] = torch.cat([tensor, padding], dim=1)
                
            elif key == 'position_ids':
                # position_ids需要连续递增
                last_positions = tensor[:, -1:]  # [batch_size, 1]
                # 从last_position+1开始递增
                increments = torch.arange(1, pad_size + 1, device=tensor.device).unsqueeze(0)  # [1, pad_size]
                padding = last_positions + increments  # [batch_size, pad_size]
                padded_batch.batch[key] = torch.cat([tensor, padding], dim=1)
                
            elif key in ['responses', 'step_mask']:
                # responses和step_mask用0 padding
                pad_value = 0
                pad_shape = [tensor.shape[0], pad_size] + list(tensor.shape[2:])
                padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                padded_batch.batch[key] = torch.cat([tensor, padding], dim=1)
                
            elif key in ['token_level_scores', 'token_level_rewards']:
                # reward相关的tensor用0.0 padding
                pad_value = 0.0
                pad_shape = [tensor.shape[0], pad_size] + list(tensor.shape[2:])
                padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                padded_batch.batch[key] = torch.cat([tensor, padding], dim=1)
                
            elif key == 'tool_use_scores':
                # tool_use_scores维度是[batch_size, 2]，不需要在序列维度padding
                continue
                
            else:
                # 其他tensor使用默认padding值
                pad_value = 0.0 if tensor.dtype.is_floating_point else 0
                pad_shape = [tensor.shape[0], pad_size] + list(tensor.shape[2:])
                padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                padded_batch.batch[key] = torch.cat([tensor, padding], dim=1)

        return padded_batch

    def _debug_inspect_batch_log_probs(self, batch, batch_name="Unknown"):
        """
        深度诊断 LogProbs 的状态，用于确认 Reference Model 是否失效。
        """
        print(f"\n{'='*20} [DEBUG: {batch_name}] {'='*20}")
        
        # 1. 检查 Key 是否存在
        if batch is None or batch.batch is None:
            print(f"  [ERROR] Batch object is None!")
            return

        keys = list(batch.batch.keys())
        if 'old_log_probs' not in keys:
            print(f"  [CRITICAL ALERT] 'old_log_probs' NOT FOUND! Available keys: {keys}")
            # 尝试查找其他可能的 key
            if 'log_probs' in keys:
                print(f"  [INFO] Found 'log_probs' instead. Using that.")
                lp = batch.batch['log_probs']
            else:
                print(f"  [STOP] No log probability field found.")
                return
        else:
            lp = batch.batch['old_log_probs']

        # 2. 检查数值状态
        # 确保转为 float 进行统计，防止溢出
        lp_flat = lp.flatten().float()
        total_count = lp_flat.numel()
        zero_count = (lp_flat == 0).sum().item()
        non_zero_count = total_count - zero_count
        
        print(f"  Shape: {lp.shape}")
        print(f"  Total Tokens: {total_count}")
        print(f"  Zero Tokens:  {zero_count} ({zero_count/total_count*100:.1f}%)")
        
        # 3. 核心判断
        if non_zero_count == 0:
            print(f"  [CRITICAL FAILURE] ALL log_probs are ZERO! Reference Model is ineffective.")
        else:
            # 统计非零值的分布
            valid_values = lp_flat[lp_flat != 0]
            mean_val = valid_values.mean().item()
            min_val = valid_values.min().item()
            max_val = valid_values.max().item()
            
            print(f"  [STATUS: OK] Found {non_zero_count} non-zero values.")
            print(f"  Statistics (Non-Zero): Mean={mean_val:.4f}, Min={min_val:.4f}, Max={max_val:.4f}")
            print(f"  Sample values: {valid_values[:5].tolist()}")
        
        print(f"{'='*50}\n")

    def _compute_igpo_reward(self, batch: DataProto) -> None:
        """
        Compute per-turn information-gain rewards and write them into
        batch.batch["token_level_scores"] **in-place**.

        For each trajectory in the batch:
          1. Locate all <search>...</search> segments via find_search_segments_by_text.
          2. For each turn t (0-indexed), build a pseudo-sequence:
               [context_t + PREFIX + ground_truth + SUFFIX]
             where context_t = prompt + response tokens up to (but not including)
             the (t+1)-th <search> tag, i.e. it includes the t-th retrieval result.
          3. Call actor_rollout_wg.compute_log_prob on the pseudo-sequence to get
             P(GT | context_t).
          4. ig_t = P(GT | context_t) - P(GT | context_{t-1})   (prob_diff mode)
          5. Write ig_t to token_level_scores at the last token of the t-th
             retrieval result (= seg_start_{t+1} - 1), which is the turn boundary
             following IGPO's convention.

        The outcome reward (em_score) already sits at valid_response_length - 1
        and is NOT touched by this function.

        Args:
            batch: DataProto with keys:
                - batch["input_ids"]       (bsz, prompt_len + resp_len)
                - batch["attention_mask"]  (bsz, prompt_len + resp_len)
                - batch["responses"]       (bsz, resp_len)
                - batch["token_level_scores"] (bsz, resp_len)  ← modified in-place
        """
        import math

        PREFIX = "\nNow there's enough information to answer\n</think>\n<answer>\n"
        SUFFIX  = "\n</answer><|im_end|>"

        responses          = batch.batch["responses"]           # (bsz, resp_len)
        input_ids          = batch.batch["input_ids"]           # (bsz, full_len)
        attention_mask     = batch.batch["attention_mask"]      # (bsz, full_len)
        token_level_scores = batch.batch["token_level_scores"]  # (bsz, resp_len)

        bsz      = responses.size(0)
        resp_len = responses.size(1)
        full_len = input_ids.size(1)
        prompt_len = full_len - resp_len

        # ── Build GT pseudo-token-ids once per unique ground truth ──────────────
        ground_truths = batch.non_tensor_batch.get("reward_model", None)
        if ground_truths is None:
            print("[IGPO] _compute_igpo_reward: 'reward_model' not found in non_tensor_batch, skipping.")
            return

        # ground_truths is a list of dicts, e.g.:
        #   {'ground_truth': {'target': ['81427']}, 'style': 'rule'}
        # We need to extract the actual answer text from the nested structure.
        import ast as _ast

        def _extract_gt_text(raw):
            """Extract plain-text ground truth from various nested formats."""
            # Step 0: unwrap numpy wrapper if present
            if hasattr(raw, 'item'):
                raw = raw.item()

            # Step 1: if raw is a string that looks like a dict/list, parse it
            if isinstance(raw, str):
                raw_stripped = raw.strip()
                if raw_stripped and raw_stripped[0] in ('{', '['):
                    try:
                        raw = _ast.literal_eval(raw_stripped)
                    except (ValueError, SyntaxError):
                        try:
                            import json as _json
                            raw = _json.loads(raw_stripped)
                        except (ValueError, TypeError):
                            return raw  # give up parsing, use as-is

            # Step 2: if raw is a dict, drill into it
            if isinstance(raw, dict):
                # Try 'ground_truth' key first
                gt_val = raw.get("ground_truth", raw)
                # gt_val might itself be a dict like {'target': ['81427']}
                if isinstance(gt_val, dict):
                    # Try common keys: 'target', 'answer', 'ground_truth'
                    for key in ("target", "answer", "ground_truth"):
                        if key in gt_val:
                            gt_val = gt_val[key]
                            break
                    else:
                        # Fall back to first value
                        gt_val = next(iter(gt_val.values()))
                # gt_val might be a list like ['81427']
                if isinstance(gt_val, (list, tuple)) and len(gt_val) > 0:
                    gt_val = gt_val[0]
                return str(gt_val) if gt_val is not None else ""

            # Step 3: if raw is a list, take the first element
            if isinstance(raw, (list, tuple)) and len(raw) > 0:
                return str(raw[0])

            return str(raw) if raw is not None else ""

        gt_texts = []
        for i, gt in enumerate(ground_truths):
            gt_text = _extract_gt_text(gt)
            gt_texts.append(gt_text)
            # Print diagnostic for first 3 samples
            if i < 3:
                print(
                    f"[IGPO][GT_EXTRACT] i={i}, raw_type={type(gt).__name__}, "
                    f"raw_repr={gt}, extracted_gt='{gt_text}'"
                )

        # Pre-tokenize each GT pseudo-response and locate GT token range
        pseudo_resp_ids_list = []   # list of list[int]
        gt_token_ranges      = []   # list of (start, end) in pseudo_resp token space

        for gt_text in gt_texts:
            full_text = f"{PREFIX}{gt_text}{SUFFIX}"
            encoding  = self.tokenizer(
                full_text,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            token_ids      = encoding["input_ids"][0].tolist()
            offset_mapping = encoding["offset_mapping"][0].tolist()

            gt_char_start = len(PREFIX)
            gt_char_end   = len(PREFIX) + len(gt_text)

            gt_token_start, gt_token_end = None, None
            for tok_idx, (char_start, char_end) in enumerate(offset_mapping):
                if gt_token_start is None and char_end > gt_char_start:
                    gt_token_start = tok_idx
                if char_start < gt_char_end and char_end > 0:
                    gt_token_end = tok_idx + 1

            if gt_token_start is None:
                gt_token_start = len(token_ids)
            if gt_token_end is None:
                gt_token_end = len(token_ids)

            pseudo_resp_ids_list.append(token_ids)
            gt_token_ranges.append((gt_token_start, gt_token_end))

        # ── Pass 1: collect all pseudo-sequences across all samples ─────────────
        # Each entry: (sample_idx, turn_idx, turn_boundary, gt_start, gt_end, pseudo_resp_len)
        meta_list        = []   # (sample_idx, turn_idx, boundary, gt_start, gt_end, pseudo_len)
        all_full_ids     = []   # list of list[int], one per (sample, turn)
        all_full_masks   = []

        pad_id = self.tokenizer.pad_token_id

        # Diagnostic counters for Pass 1
        skip_p1_gt_empty      = 0  # gt_start >= gt_end
        skip_p1_no_search     = 0  # no search segments found

        for sample_idx in range(bsz):
            resp_ids = responses[sample_idx].tolist()
            gt_start, gt_end = gt_token_ranges[sample_idx]

            if gt_start >= gt_end:
                skip_p1_gt_empty += 1
                continue

            search_segments = find_search_segments_by_text(resp_ids, self.tokenizer)
            if not search_segments:
                skip_p1_no_search += 1
                continue

            num_turns = len(search_segments)

            # Compute turn boundaries.
            # For turn t (0-indexed):
            #   boundary = seg_start[t+1] - 1  (last token before next <search>)
            # For the last turn:
            #   boundary = seg_end of the last <search>...</search> segment
            #   (i.e. the last token of </search>), so the ig reward covers
            #   the retrieval result block without conflicting with the outcome reward.
            turn_boundaries = []
            for t in range(num_turns - 1):
                turn_boundaries.append(search_segments[t + 1][0] - 1)
            # Last turn: use the end of the last </search> tag
            last_seg_end = search_segments[-1][1]
            turn_boundaries.append(last_seg_end)

            prompt_ids  = input_ids[sample_idx, :prompt_len].tolist()
            pseudo_resp = pseudo_resp_ids_list[sample_idx]
            pseudo_len  = len(pseudo_resp)

            for t in range(num_turns):
                boundary  = turn_boundaries[t]
                context_t = resp_ids[:boundary + 1]
                while context_t and context_t[-1] == pad_id:
                    context_t = context_t[:-1]

                full_ids  = prompt_ids + context_t + pseudo_resp
                full_mask = [1] * len(full_ids)
                all_full_ids.append(full_ids)
                all_full_masks.append(full_mask)
                meta_list.append((sample_idx, t, boundary, gt_start, gt_end, pseudo_len))

        if not meta_list:
            print(f"[IGPO] _compute_igpo_reward: step={self.global_steps}, no search segments found, skipping.")
            return

        # ── Pass 2: pad all sequences to the same length ────────────────────────
        max_pseudo_len = max(len(ids) for ids in all_full_ids)
        # Determine the max pseudo_resp length (for GT token index alignment)
        max_pseudo_resp_len = max(m[5] for m in meta_list)

        padded_input_ids  = []
        padded_attn_masks = []
        for ids, mask in zip(all_full_ids, all_full_masks):
            pad_count = max_pseudo_len - len(ids)
            padded_input_ids.append([pad_id] * pad_count + ids)
            padded_attn_masks.append([0]    * pad_count + mask)

        # ── Pass 3: pad total rows to be divisible by num_workers ───────────────
        num_workers = self.actor_rollout_wg.world_size
        total_rows  = len(padded_input_ids)
        remainder   = total_rows % num_workers
        num_dummy   = (num_workers - remainder) % num_workers
        dummy_ids   = [pad_id] * max_pseudo_len
        dummy_mask  = [0]      * max_pseudo_len
        for _ in range(num_dummy):
            padded_input_ids.append(dummy_ids)
            padded_attn_masks.append(dummy_mask)

        pseudo_input_tensor = torch.tensor(padded_input_ids,  dtype=torch.long)
        pseudo_attn_tensor  = torch.tensor(padded_attn_masks, dtype=torch.long)
        pseudo_pos_tensor   = (pseudo_attn_tensor.cumsum(dim=-1) - 1).clamp(min=0)
        # responses = last max_pseudo_resp_len tokens
        pseudo_resp_tensor  = pseudo_input_tensor[:, -max_pseudo_resp_len:]

        pseudo_data = DataProto.from_dict({
            "input_ids":      pseudo_input_tensor,
            "attention_mask": pseudo_attn_tensor,
            "position_ids":   pseudo_pos_tensor,
        })
        pseudo_data.batch["responses"] = pseudo_resp_tensor

        # ── Pass 4: single compute_log_prob call for the whole batch ─────────────
        log_prob_output = self.actor_rollout_wg.compute_log_prob(pseudo_data)
        old_log_probs   = log_prob_output.batch["old_log_probs"]  # (total_rows+dummy, *)
        lp_seq_len      = old_log_probs.size(1)

        # ── Pass 5: write ig rewards back to token_level_scores ─────────────────
        # Group rows by sample_idx to compute ig_t = P(GT|ctx_t) - P(GT|ctx_{t-1})
        from collections import defaultdict
        sample_rows = defaultdict(list)  # sample_idx -> list of (row_idx, turn_idx, boundary, gt_start, gt_end, pseudo_len)
        for row_idx, meta in enumerate(meta_list):
            sample_idx = meta[0]
            sample_rows[sample_idx].append((row_idx,) + meta[1:])

        total_ig_applied = 0
        # Diagnostic counters for Pass 5
        skip_p5_align    = 0  # lp_gt_start >= lp_gt_end after clamp
        skip_p5_nan      = 0  # mean_lp is nan or inf
        skip_p5_boundary = 0  # boundary out of [0, resp_len)
        ig_positive      = 0  # ig_t > 0
        ig_negative      = 0  # ig_t < 0
        ig_zero          = 0  # ig_t == 0

        for sample_idx, rows in sample_rows.items():
            # Sort by turn_idx to ensure correct ig_t computation
            rows.sort(key=lambda x: x[1])

            prev_p = None  # turn 0 only stores baseline, no ig_reward produced
            for row_idx, turn_idx, boundary, gt_start, gt_end, pseudo_len in rows:
                # Align GT token positions to old_log_probs columns.
                # old_log_probs covers the last max_pseudo_resp_len tokens of each row.
                # Within that window, the GT tokens are at offset:
                #   (max_pseudo_resp_len - pseudo_len) + gt_start
                # because each pseudo_resp is right-aligned (padded on the left).
                offset = max_pseudo_resp_len - pseudo_len
                lp_gt_start = offset + gt_start
                lp_gt_end   = offset + gt_end

                lp_gt_start = max(0, lp_gt_start)
                lp_gt_end   = min(lp_seq_len, lp_gt_end)

                if lp_gt_start >= lp_gt_end:
                    # do NOT reset prev_p, keep previous valid baseline
                    skip_p5_align += 1
                    continue

                lp_gt   = old_log_probs[row_idx, lp_gt_start:lp_gt_end]
                mean_lp = lp_gt.mean().item()
                if math.isnan(mean_lp) or math.isinf(mean_lp):
                    # do NOT reset prev_p, keep previous valid baseline
                    skip_p5_nan += 1
                    continue

                # Use log_prob_diff mode (same as IGPO original train.sh):
                # ig = mean_log_P(GT|ctx_t) - mean_log_P(GT|ctx_{t-1})
                # This produces IG values in ~0.01-0.1 range instead of ~1e-6 with prob_diff
                p_gt = mean_lp  # use log-prob directly,

                # Alignment diagnostic: print first 5 samples x first 2 turns
                if sample_idx < 5 and turn_idx < 2:
                    print(
                        f"[IGPO][DIAG] sample={sample_idx}, turn={turn_idx}, "
                        f"offset={offset}, gt_start={gt_start}, gt_end={gt_end}, "
                        f"lp_gt_start={lp_gt_start}, lp_gt_end={lp_gt_end}, "
                        f"lp_seq_len={lp_seq_len}, pseudo_len={pseudo_len}, "
                        f"mean_lp={mean_lp:.4f}, p_gt={p_gt:.6f}, prev_p={prev_p}"
                    )

                if prev_p is None:
                    # turn 0: only store baseline P(GT | context_0), no ig_reward
                    prev_p = p_gt
                else:
                    ig_t = p_gt - prev_p
                    prev_p = p_gt

                    # Scale IG to control its magnitude relative to outcome reward.
                    #ig_t = ig_t * 0.2

                    if 0 <= boundary < resp_len:
                        token_level_scores[sample_idx, boundary] += ig_t
                        total_ig_applied += 1
                        if ig_t > 0:
                            ig_positive += 1
                        elif ig_t < 0:
                            ig_negative += 1
                        else:
                            ig_zero += 1
                    else:
                        skip_p5_boundary += 1

        # total rows entering Pass 5 = total_rows - num_dummy
        valid_rows = total_rows - num_dummy

        # Compute ig_t mean and std from token_level_scores at written positions
        ig_values = []
        for sample_idx, rows in sample_rows.items():
            rows_sorted = sorted(rows, key=lambda x: x[1])
            prev_p_stat = None
            for row_idx, turn_idx, boundary, gt_start, gt_end, pseudo_len in rows_sorted:
                offset = max_pseudo_resp_len - pseudo_len
                lp_gt_start = max(0, offset + gt_start)
                lp_gt_end   = min(lp_seq_len, offset + gt_end)
                if lp_gt_start >= lp_gt_end:
                    continue
                lp_gt   = old_log_probs[row_idx, lp_gt_start:lp_gt_end]
                mean_lp = lp_gt.mean().item()
                if math.isnan(mean_lp) or math.isinf(mean_lp):
                    continue
                # Use log_prob_diff mode (consistent with actual IG computation above)
                p_gt = mean_lp
                if prev_p_stat is None:
                    prev_p_stat = p_gt
                else:
                    raw_ig = p_gt - prev_p_stat
                    # Apply same ig_scale as in actual IG writing
                    ig_values.append(raw_ig * 0.5)
                    prev_p_stat = p_gt

        if ig_values:
            ig_tensor = torch.tensor(ig_values, dtype=torch.float32)
            ig_mean = ig_tensor.mean().item()
            ig_std  = ig_tensor.std().item()
            ig_min  = ig_tensor.min().item()
            ig_max  = ig_tensor.max().item()
        else:
            ig_mean = ig_std = ig_min = ig_max = 0.0

        # 对齐诊断打印（只打印前5个样本的前2个turn）
        if len(ig_values) > 0:
            print(f"[IGPO] Alignment Diagnostics (first 5 samples, first 2 turns):")
            for sample_idx, rows in list(sample_rows.items())[:5]:
                rows_sorted = sorted(rows, key=lambda x: x[1])
                for row_idx, turn_idx, boundary, gt_start, gt_end, pseudo_len in rows_sorted[:2]:
                    offset = max_pseudo_resp_len - pseudo_len
                    lp_gt_start = max(0, offset + gt_start)
                    lp_gt_end   = min(lp_seq_len, offset + gt_end)
                    if lp_gt_start >= lp_gt_end:
                        continue
                    lp_gt   = old_log_probs[row_idx, lp_gt_start:lp_gt_end]
                    mean_lp = lp_gt.mean().item()
                    if math.isnan(mean_lp) or math.isinf(mean_lp):
                        continue
                    # Print log-prob directly (consistent with log_prob_diff mode)
                    print(f"Sample {sample_idx}, Turn {turn_idx}: mean_lp={mean_lp:.4f}")

        print(
            f"[IGPO] _compute_igpo_reward: step={self.global_steps}, "
            f"bsz={bsz}, total_pseudo_rows={total_rows}, dummy_rows={num_dummy}, "
            f"ig_rewards_written={total_ig_applied}\n"
            f"[IGPO] Pass1 skips: gt_empty={skip_p1_gt_empty}, no_search={skip_p1_no_search} "
            f"(samples skipped={skip_p1_gt_empty + skip_p1_no_search}/{bsz})\n"
            f"[IGPO] Pass5 skips: align={skip_p5_align}, nan={skip_p5_nan}, boundary={skip_p5_boundary} "
            f"(rows skipped={skip_p5_align + skip_p5_nan + skip_p5_boundary}/{valid_rows})\n"
            f"[IGPO] ig_t distribution: positive={ig_positive}, negative={ig_negative}, zero={ig_zero}\n"
            f"[IGPO] ig_t stats: mean={ig_mean:.6f}, std={ig_std:.6f}, min={ig_min:.6f}, max={ig_max:.6f}"
        )

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.gen_steps += 1
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        filtered_out_batches = []

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = new_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                
                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                #gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)


                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch (with-tool rollout, 64*12)
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

                        if gen_batch_output.meta_info and "metrics" in gen_batch_output.meta_info:
                            metrics.update(gen_batch_output.meta_info["metrics"])

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=new_batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)
                        new_batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]


                    # Dynamic Sampling
                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:   # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (new_batch.batch["token_level_rewards"].sum(dim=-1).numpy())
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (new_batch.batch["token_level_scores"].sum(dim=-1).numpy())

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]

                        filtered_out_prompt_uids = [
                            uid for uid in prompt_uid2metric_vals.keys() 
                            if uid not in kept_prompt_uids
                        ]

                        if filtered_out_prompt_uids:
                            print(f"\n=== Step {self.global_steps}: Filtered out {len(filtered_out_prompt_uids)} prompt groups ===")
                            for uid in filtered_out_prompt_uids:
                                rewards = prompt_uid2metric_vals[uid]
                                std_val = prompt_uid2metric_std[uid]
                                print(f"Prompt UID {uid}: rewards={rewards}, std={std_val:.6f}")
                            print("=" * 60)

                        #current_batch_prompt_count = len(kept_prompt_uids)
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        filtered_out_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)
                            else:
                                filtered_out_traj_idxs.append(idx)

                        if filtered_out_traj_idxs:
                            filtered_out_batch = new_batch[filtered_out_traj_idxs]

                            # Print detailed trajectory-level rewards for filtered out samples
                            print(f"\n--- Detailed trajectory rewards for filtered out samples ---")
                            filtered_sequence_rewards = filtered_out_batch.batch["token_level_rewards"].sum(-1).numpy()
                            filtered_uids = filtered_out_batch.non_tensor_batch["uid"]
                            
                            for traj_idx, (uid, seq_reward) in enumerate(zip(filtered_uids, filtered_sequence_rewards)):
                                print(f"Trajectory {traj_idx} (UID {uid}): sequence_reward={seq_reward:.6f}")
                            print("-" * 60)

                            filtered_out_batches.append(filtered_out_batch)
                        
                        new_batch = new_batch[kept_traj_idxs]
                        batch, new_batch = self._pad_batch_to_match(batch, new_batch)

                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")                       
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                #progress_bar.update(1)
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                print(f"{num_gen_batches=} >= {max_num_gen_batches=}. Trying to fill from backup samples...")
                                
                                if filtered_out_batches:
                                    # 计算还需要多少个prompt
                                    needed_prompts = prompt_bsz - num_prompt_in_batch
                                    print(f"Need {needed_prompts} more prompts. Searching in {len(filtered_out_batches)} backup batches...")
                                    
                                    # 从备用样本中选择轨迹
                                    backup_batch = None
                                    backup_prompt_count = 0
                                    
                                    for backup_batch_candidate in filtered_out_batches:
                                        if backup_batch is None:
                                            backup_batch = backup_batch_candidate
                                        else:
                                            backup_batch, backup_batch_candidate = self._pad_batch_to_match(backup_batch, backup_batch_candidate)
                                            backup_batch = DataProto.concat([backup_batch, backup_batch_candidate])
                                    
                                    if backup_batch is not None:
                                        # 从备用样本中获取unique prompt UIDs
                                        backup_prompt_uids = list(set(backup_batch.non_tensor_batch["uid"]))
                                        
                                        # 选择需要的prompt数量
                                        selected_prompt_uids = backup_prompt_uids[:needed_prompts]
                                        
                                        # 筛选对应的轨迹
                                        backup_kept_traj_idxs = []
                                        for idx, traj_uid in enumerate(backup_batch.non_tensor_batch["uid"]):
                                            if traj_uid in selected_prompt_uids:
                                                backup_kept_traj_idxs.append(idx)
                                        
                                        if backup_kept_traj_idxs:
                                            backup_selected_batch = backup_batch[backup_kept_traj_idxs]
                                            backup_selected_batch, _ = self._pad_batch_to_match(backup_selected_batch, batch)
                                            
                                            # 合并到主batch中
                                            batch = DataProto.concat([batch, backup_selected_batch])
                                            backup_prompt_count = len(selected_prompt_uids)
                                            num_prompt_in_batch += backup_prompt_count
                                            
                                            print(f"Added {backup_prompt_count} prompts from backup samples. Total prompts: {num_prompt_in_batch}")
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]             

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()


                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        if "loss_mask" in batch.batch.keys():
                            loss_mask = batch.batch["loss_mask"]
                        else:
                            loss_mask = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)


                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        # reward_extra_infos_dict: dict[str, list]
                        # if self.config.reward_model.launch_reward_fn_async:
                        #     reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        # batch.batch["token_level_scores"] = reward_tensor

                        # print(f"{list(reward_extra_infos_dict.keys())=}")
                        # if reward_extra_infos_dict:
                        #     batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # # compute rewards. apply_kl_penalty if available
                        # if self.config.algorithm.use_kl_in_reward:
                        #     batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        # ---- IGPO warmup: use GRPO for first N steps, then switch to IGPO ----
                        igpo_warmup_steps = self.config.algorithm.get("igpo_warmup_steps", 0)
                        use_igpo_this_step = (
                            self.config.algorithm.adv_estimator == AdvantageEstimator.IGPO
                            and (igpo_warmup_steps == 0 or self.global_steps > igpo_warmup_steps)
                        )

                        if use_igpo_this_step:
                            effective_adv_estimator = AdvantageEstimator.IGPO
                        elif (self.config.algorithm.adv_estimator == AdvantageEstimator.IGPO
                              and igpo_warmup_steps > 0
                              and self.global_steps <= igpo_warmup_steps):
                            effective_adv_estimator = AdvantageEstimator.GRPO
                            if self.global_steps == 1 or self.global_steps == igpo_warmup_steps:
                                print(f"[IGPO Warmup] step {self.global_steps}/{igpo_warmup_steps}: "
                                      f"using GRPO advantage (switch to IGPO at step {igpo_warmup_steps + 1})")
                        else:
                            effective_adv_estimator = self.config.algorithm.adv_estimator

                        # ---- IGPO: compute per-turn info-gain rewards ----
                        if use_igpo_this_step:
                            try:
                                self._compute_igpo_reward(batch)
                            except Exception as e:
                                print(f"[IGPO] ERROR in _compute_igpo_reward: {e}")
                                import traceback
                                traceback.print_exc()

                        batch = compute_advantage(
                            batch,
                            adv_estimator=effective_adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            igpo_gamma=self.config.algorithm.igpo_gamma,
                            igpo_norm_mode=self.config.algorithm.igpo_norm_mode,
                            igpo_adv_rescale_alpha=getattr(self.config.algorithm, 'igpo_adv_rescale_alpha', 1.0),
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable

                            # ---- Dynamic clip switching for IGPO warmup ----
                            # Stage 1 (warmup): use tight clip from config (e.g., 3e-4/4e-4)
                            # Stage 2 (post-warmup): switch to wider clip for IGPO (e.g., 3e-3/4e-3)
                            if (igpo_warmup_steps > 0
                                and self.config.algorithm.adv_estimator == AdvantageEstimator.IGPO
                                and use_igpo_this_step):
                                igpo_clip_ratio_low = self.config.algorithm.get("igpo_clip_ratio_low", None)
                                igpo_clip_ratio_high = self.config.algorithm.get("igpo_clip_ratio_high", None)
                                if igpo_clip_ratio_low is not None and igpo_clip_ratio_high is not None:
                                    batch.meta_info["clip_ratio_low_override"] = igpo_clip_ratio_low
                                    batch.meta_info["clip_ratio_high_override"] = igpo_clip_ratio_high
                                    if self.global_steps == igpo_warmup_steps + 1:
                                        print(f"[IGPO Dynamic Clip] step {self.global_steps}: "
                                              f"switching clip from {self.config.actor_rollout_ref.actor.clip_ratio_low}/"
                                              f"{self.config.actor_rollout_ref.actor.clip_ratio_high} to "
                                              f"{igpo_clip_ratio_low}/{igpo_clip_ratio_high}")

                            actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                num_prompt_in_batch = 0
                num_gen_batches = 0
                batch = None
                filtered_out_batches = []
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
