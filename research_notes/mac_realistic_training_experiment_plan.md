# Mac-Realistic Training Experiment Plan

Goal: produce a paper-facing training result for PrefixIG-TPO / evidence-efficient
search credit using experiments that are feasible on a 16GB Apple Silicon laptop.

## Repository Read

### A-TGPO

- Full training stack is CUDA/verl/vLLM/Ray oriented.
- Good as conceptual reference for token-level PPO/GRPO accounting, clipping,
  turn segmentation, and rollout structure.
- Not the primary Mac training path.

### mlx-lm-lora

- Best practical Mac path.
- Already has MLX LoRA training for `grpo`, `tpo`, `es_tpo`, `zo_adam`, `fl`,
  and `cezo_fl`.
- Existing successful Qwen2.5-0.5B runs show long but feasible Apple Metal
  training.
- This is the repo to adapt for actual model training.

### HyperscaleES

- Useful for ES/low-memory optimization ideas.
- `mlx_experiments/digits_eggroll.py` shows Mac/MLX ES-style experiments.
- The JAX LLM experiments are heavier and less directly useful for a MacBook.

### zo_ldsd

- CUDA/HF/OPT oriented.
- Useful as a conceptual ZO baseline source, but not a direct Mac training path.

## Recommended Experiment Ladder

### Experiment 0: Accounting and Diagnostics

Status: mostly done.

- Synthetic PrefixIG-TPO comparison.
- Real MLX old-policy scoring on controlled trajectories.
- Generated oracle-search rollouts.
- Flexible generated rollouts exposing the redundant-search failure mode.

Purpose: establish the mechanism before training.

### Experiment 1: Small Trainable Toy RLVR

Use the existing MLX transformer RLVR setup in `mlx-lm-lora/mlx_lm_lora/tpo_transformer_rlvr.py`.

Add a search-credit variant of the toy task:

- baseline reward-only GRPO/TPO,
- PrefixIG-TPO,
- PrefixIG-TPO + efficiency penalty,
- optional curvature.

Why this matters:

- Fast enough for many seeds.
- Gives clean learning curves.
- Provides a paper figure without expensive LLM training.

Expected paper role: controlled evidence that the objective learns useful
intermediate actions instead of merely final reward.

### Experiment 2: Qwen2.5-0.5B Offline Preference/TPO Training

Train on generated trajectory groups from our oracle-search setup.

Dataset per prompt:

- useful correct trajectory,
- redundant correct trajectory,
- distractor/wrong trajectory,
- sometimes no-search/direct trajectory if available.

Training variants:

- Original/base evaluation only,
- reward-only TPO,
- PrefixIG-GRPO or A-TGPO-style scalar advantage,
- PrefixIG-TPO,
- PrefixIG-TPO + reward-gated efficiency,
- optional curvature version.

Why offline first:

- Avoids expensive online generation during every training step.
- Fits Mac memory better.
- Lets us directly test the target distribution idea.

Expected paper role: first real model update showing improved evidence-efficient
rollout preference.

### Experiment 3: Tiny Online GRPO/TPO Fine-Tune

Use `mlx-lm-lora` GRPO/TPO infrastructure on a small synthetic search QA dataset.

Keep it small:

- Qwen2.5-0.5B 4-bit LoRA.
- 50-200 prompts.
- group size 2-4.
- short completions.
- 50-200 iterations for smoke/pilot, then 500 if stable.

Metrics:

- exact-match answer accuracy,
- useful search rate,
- redundant search rate,
- distractor rate,
- average search turns,
- evidence efficiency: correct answer per search turn,
- generated-policy table before and after training.

Expected paper role: actual LLM training result on Apple Silicon.

### Experiment 4: ZO/ES Baseline

Only after Experiments 1-3 are stable.

Use `mlx-lm-lora` existing `es`, `es_tpo`, or `zo_adam` modes instead of porting
`zo_ldsd` directly.

Hypothesis:

- PrefixIG-TPO targets can guide low-memory ES/ZO updates.
- This connects to HyperscaleES / ZO-LDSD style training without requiring CUDA.

Expected paper role: optional ablation, not the core contribution.

## Main Objective Variant To Test Next

Current flexible rollout results show that raw PrefixIG still overweights
redundant correct trajectories. The next training objective should be:

```text
utility = reward
        + lambda_ig * normalized_prefix_ig
        - reward * lambda_eff * normalized_redundancy_penalty
```

This is reward-gated efficiency. It penalizes waste only when the answer is
correct, preventing short wrong trajectories from benefiting.

Potential curvature version:

```text
utility = reward
        + curvature_scale * lambda_ig * normalized_prefix_ig
        - reward * lambda_eff * normalized_redundancy_penalty
```

## Concrete Next Step

Implement reward-gated efficiency in the diagnostic scripts first, then create
an offline training dataset from generated trajectory groups. If the diagnostic
table improves, port the same utility into an MLX offline TPO training script.

Recommended immediate sequence:

1. Add `PrefixIG-TPO+reward_gated_eff` diagnostic row.
2. Generate 50-100 offline trajectory groups with oracle search.
3. Save them as JSONL with target weights.
4. Add a small MLX offline TPO trainer or adapt existing `tpo_trainer.py`.
5. Train Qwen2.5-0.5B LoRA for a short pilot.
6. Evaluate before/after with `mlx_generated_policy_diagnostics.py`.

