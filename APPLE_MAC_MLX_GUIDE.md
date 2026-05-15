# Running A-TGPO Ideas on an Apple Silicon Laptop

This note explains how to think about running this repository on an Apple Silicon Mac, especially a 16 GB M4 MacBook Air with MLX/Metal available.

## Short Answer

The current A-TGPO training stack is **not runnable as-is** on an Apple Silicon laptop.

The algorithm can be adapted to a laptop-sized MLX experiment, but the existing implementation is built around NVIDIA/CUDA infrastructure: CUDA PyTorch, vLLM, Ray workers, `flash-attn`, `faiss-gpu`, and multi-GPU training.

For a Mac, the practical route is:

1. Use this repository as the reference implementation of A-TGPO.
2. Use an MLX-based trainer, such as `mlx-lm-lora`, as the Apple Metal runtime.
3. Port the A-TGPO algorithmic pieces into an MLX GRPO/PPO-style training loop.

## What Your Laptop Can Do

An M4 MacBook Air with 16 GB unified memory and a 10-core Apple GPU can run small local MLX fine-tuning experiments.

Good targets:

- Qwen 0.5B to 1.5B models.
- LoRA or QLoRA-style adapter training.
- Small GRPO-style groups, such as `group_size=2` or `group_size=4`.
- Shorter completions, around 512 to 1024 tokens.
- Small HotpotQA/NQ slices for debugging the reward and turn-credit logic.
- Local retrieval experiments if the retriever is made CPU/MPS/MLX friendly.

Bad targets for this laptop:

- Full A-TGPO training with Qwen3-4B as configured here.
- `rollout.n=16` with long 6192-token responses.
- vLLM-based rollout.
- Ray/FSDP multi-GPU training.
- CUDA-only `flash-attn`, `faiss-gpu`, and NVIDIA memory monitoring.

## Why This Repo Does Not Run As-Is on Mac

The current setup assumes CUDA in several places:

- The README installs CUDA PyTorch, `pytorch-cuda`, `faiss-gpu`, and `flash-attn`.
- `ATGPO/scripts/config/ppo_trainer_dr.yaml` sets `trainer.device: cuda`.
- `ATGPO/scripts/ATGPO_multihop_qwen3_4B.sh` is configured for 8 GPUs.
- The retriever server calls `.cuda()` directly.
- vLLM/SGLang integration in veRL expects CUDA-style GPU execution.

Apple Silicon uses Metal/MLX, not CUDA. PyTorch MPS support exists, but this codebase is not written as a simple single-process PyTorch script; it is a distributed veRL training stack with CUDA-specific inference engines.

## What Should Be Ported to MLX

The most valuable part to port is not the infrastructure. It is the A-TGPO credit assignment logic.

The key pieces are:

1. **Final answer reward**
   - Use the logic from `ATGPO/verl_atgpo/verl/utils/reward_score/deep_research_em.py`.
   - Validate `<think>`, `<search>`, `<result>`, and `<answer>` formatting.
   - Extract the final `\boxed{...}` answer.
   - Score with F1/EM against the ground truth.

2. **Per-turn information gain**
   - Locate each search turn in the generated response.
   - For each turn, compute how much the turn improved the model's log-probability of the ground-truth answer.
   - Conceptually:

```text
IG_t = log P(correct answer | context after turn t)
     - log P(correct answer | context before turn t)
```

3. **Turn-group normalization**
   - Generate multiple completions per prompt.
   - Compare turn 1 only against other turn 1 samples for the same prompt.
   - Compare turn 2 only against other turn 2 samples for the same prompt.
   - This avoids mixing early and late search turns, which have different difficulty and information content.

4. **Variance-rescaled discounted accumulation**
   - Later information gains should influence earlier turns.
   - The implementation uses a discounted future sum and divides by the square root of the number of accumulated terms so earlier turns do not get inflated just because they have more future turns.

```text
D_t = sum_j>=t gamma^(j-t) * normalized_IG_j
A_t = normalized_outcome_reward + alpha * D_t / sqrt(number_of_terms)
```

5. **Adaptive turn-level clipping**
   - Good/informative turns get a wider PPO/GRPO clip range.
   - Uninformative turns get a narrower clip range.
   - In this repo's `turn-group-v1d` mode, the scale is approximately bounded between `0.7` and `1.3`.

```text
clip_scale_t = 1 + 0.3 * (2 * sigmoid(normalized_IG_t) - 1)
effective_clip = base_clip * clip_scale_t
```

## Recommended Laptop Experiment

Start with a small controlled experiment instead of trying to reproduce the paper-scale run.

Suggested settings:

```text
model: Qwen2.5-0.5B-Instruct or Qwen2.5/Qwen3 1.5B
training type: LoRA
train mode: GRPO-like, then A-TGPO variant
group size: 2 or 4
max completion length: 512 initially, then 1024
batch size: 1 prompt per update, gradient accumulation if needed
dataset: 100-1000 examples from NQ or HotpotQA
retrieval: simple local retriever or cached search snippets
```

The first goal should be to verify that the algorithmic signal works:

- Are final-answer rewards computed correctly?
- Are search turns detected correctly?
- Are per-turn information gains nonzero and sensible?
- Do positive IG turns get larger clip scales?
- Does the model learn to search less randomly and answer more accurately?

## Suggested MLX Implementation Plan

If using `mlx-lm-lora`, the natural starting point is the existing GRPO trainer.

Implementation sketch:

1. Add an `atgpo` or `igpo` training option next to GRPO.
2. Reuse GRPO generation: multiple completions per prompt.
3. Add a reward function for the final boxed answer.
4. Add a parser for `<search>...</search>` / `<result>...</result>` turns.
5. Add an MLX function that computes ground-truth answer log-probability under different truncated contexts.
6. Write a function equivalent to `compute_igpo_step_advantage`.
7. Modify the PPO/GRPO loss to accept:
   - per-token or per-turn advantages;
   - optional per-token/per-turn clip scale.
8. Run a tiny smoke test on 5-10 prompts before training.

## Minimal Validation Checklist

Before doing real training, confirm:

- MLX reports `Device(gpu, 0)`.
- A tiny LoRA run works on your machine.
- A sample completion can be parsed into turns.
- A sample ground-truth answer log-prob can be computed.
- `IG_t` changes when search results are added to context.
- Turn advantages have reasonable mean/std.
- Clip scales stay in the intended range.

## Practical Recommendation

Do not try to make veRL/vLLM/FSDP run on the Mac. That is the hard path and mostly fights the framework.

The productive path is to make a **small MLX A-TGPO prototype**:

- keep this repo as the paper/reference code;
- keep the Mac runtime in MLX;
- port only the reward, information-gain, advantage, and clipping logic;
- use small models and short rollouts.

Once the MLX prototype is working, the same logic can be scaled back up on a CUDA machine using the original A-TGPO stack.
