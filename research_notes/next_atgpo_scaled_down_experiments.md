# Next Experiments: Small A-TGPO-Style Ablations

This note lists the next experiments we should run to connect our PrefixIG-TPO
direction more directly to the original A-TGPO paper, while keeping everything
small enough for Apple Silicon / MLX.

The original A-TGPO repo targets Qwen3-4B with veRL, vLLM, Ray, CUDA, and a
retriever stack. We will not reproduce that full setup on the Mac. Instead, we
will make small versions of the same scientific questions:

- Does turn-level evidence credit help?
- Do A-TGPO components help beyond scalar GRPO?
- Does target-policy construction help beyond A-TGPO-style clipping?
- Does the method behave differently on single-hop vs multi-hop search?
- Does it stay robust when retrieval includes distractors?

## Experiment 1: A-TGPO Component Ablation in Toy RLVR

### Goal

Replicate the original A-TGPO component story in our small trainable RLVR setup.
The original paper emphasizes:

```text
turn-group normalization
variance-rescaled discounted accumulation
adaptive turn-level clipping
```

We should test each component in isolation and compare it against our
PrefixIG-TPO target approach.

### Variants

```text
FinalReward-TPO
PrefixIG-GRPO
PrefixIG-GRPO + turn-group normalization
PrefixIG-GRPO + turn-group normalization + variance-rescaled accumulation
A-TGPO full: + adaptive turn-level clipping
PrefixIG-TPO
PrefixIG-TPO + reward-gated efficiency
```

### Metrics

```text
final reward
useful_correct
redundant_correct
distractor_wrong
useful_minus_redundant
average PrefixIG
average redundancy penalty
learning curves over episodes
multi-seed mean/std
```

### Expected Paper Role

This is the cleanest bridge to A-TGPO. It can support a table like:

```text
A-TGPO components improve scalar advantage training,
but TPO target construction better redistributes mass toward useful evidence.
```

### Completed Pilot Result

We implemented the explicit component ablations in:

```text
scripts/toy_prefixig_rlvr_train.py
```

and ran a 5-seed, 200-episode MLX toy RLVR experiment:

```text
outputs/toy_prefixig_rlvr_atgpo_component_ablation_5seed.csv
outputs/figures_atgpo_component_ablation/
```

Final metrics:

```text
method                     | reward | useful | redundant | distractor | useful-red
---------------------------+--------+--------+-----------+------------+-----------
reward_tpo                 | 0.921  | 0.088  | 0.091     | 0.007      | -0.004
prefixig_grpo              | 0.876  | 0.876  | 0.000     | 0.000      | +0.876
prefixig_grpo_turn_norm    | 0.596  | 0.000  | 0.596     | 0.001      | -0.596
prefixig_grpo_turn_norm_vr | 0.595  | 0.000  | 0.595     | 0.001      | -0.595
prefixig_atgpo             | 0.797  | 0.000  | 0.797     | 0.001      | -0.797
prefixig_tpo               | 0.844  | 0.826  | 0.013     | 0.009      | +0.813
prefixig_tpo_rg_eff        | 0.846  | 0.838  | 0.004     | 0.011      | +0.833
```

Interpretation:

- Reward-only TPO learns answer correctness but does not separate useful from
  redundant correct trajectories.
- Raw PrefixIG-GRPO is very strong in this toy setting.
- The naive turn-group normalization and variance-rescaled accumulation
  variants collapse toward redundant-correct behavior in this implementation.
- The full A-TGPO-style token objective with adaptive clipping also overweights
  redundant correct trajectories here, likely because reward is broadcast across
  turns in correct trajectories and the toy environment makes redundant evidence
  highly correlated with correctness.
- PrefixIG-TPO and PrefixIG-TPO+reward-gated efficiency remain strong and stable;
  reward-gated efficiency gives the lowest redundant-correct rate among the TPO
  target methods.

This result suggests that the A-TGPO component story is sensitive to how
turn-level credit is consumed. Our next ablation should make the environment
harder for raw PrefixIG-GRPO by adding noisy retrieval and redundant evidence
lures, then test whether target construction remains more robust.

### Implementation Notes

Most of this belongs in:

```text
scripts/toy_prefixig_rlvr_train.py
```

We already have:

```text
reward_tpo
prefixig_atgpo
prefixig_tpo
prefixig_tpo_rg_eff
```

Next we should split `prefixig_atgpo` into explicit ablations:

```text
prefixig_grpo
prefixig_grpo_turn_norm
prefixig_grpo_turn_norm_vr
prefixig_atgpo
```

## Experiment 2: Single-Hop vs Multi-Hop Mini Search

### Goal

The original repo has separate single-hop and multi-hop QA settings. We should
build a small version to test whether evidence-credit methods help more when
the task requires multiple search turns.

### Dataset Conditions

```text
single-hop: answer can be found with one search result
multi-hop: answer requires bridge evidence plus answer evidence
mixed-hop: examples randomly require one or two search turns
```

### Variants

```text
Original/base policy
FinalReward-TPO
A-TGPO-style PrefixIG scalar objective
PrefixIG-TPO
PrefixIG-TPO + reward-gated efficiency
```

### Metrics

```text
answer accuracy
useful search rate
redundant search rate
distractor rate
average search turns
correct answer per search turn
useful_minus_redundant
```

### Expected Paper Role

This shows that our method is specifically useful for multi-turn agentic search,
not just for generic answer accuracy. We expect:

```text
small gains on single-hop
larger gains on multi-hop
best efficiency on mixed-hop
```

## Experiment 3: Adaptive Clipping Ablation

### Goal

The original A-TGPO contribution includes adaptive turn-level clipping. We should
compare this directly against fixed clipping and target-policy construction.

### Variants

```text
PrefixIG-GRPO + fixed clip
PrefixIG-GRPO + adaptive A-TGPO clip
PrefixIG-TPO
PrefixIG-TPO + reward-gated efficiency
```

### Metrics

```text
reward
useful_correct
redundant_correct
distractor_wrong
clip fraction
mean effective clip range
turn-level update magnitude
training stability across seeds
```

### Expected Paper Role

This answers a precise question:

```text
Is it better to modulate the PPO/GRPO update region,
or to define a better target distribution over trajectories?
```

Our hypothesis:

```text
adaptive clipping helps scalar updates,
but PrefixIG-TPO gives stronger useful-vs-redundant separation.
```

## Experiment 4: Noisy Retriever / Distractor Robustness

### Goal

The original A-TGPO setting depends on search/retrieval. A realistic smaller
experiment should test whether methods remain robust when the retriever returns
distractors or repeated stale results.

### Retrieval Conditions

```text
clean retrieval
25% distractor result
50% distractor result
repeated stale result
ambiguous bridge result
```

### Variants

```text
FinalReward-TPO
A-TGPO-style PrefixIG scalar objective
PrefixIG-TPO
PrefixIG-TPO + reward-gated efficiency
PrefixIG-TPO + curvature trust, optional
```

### Metrics

```text
answer accuracy
distractor search rate
useful search rate
redundant search rate
average search turns
PrefixIG under noisy evidence
target mass on distractor trajectories
```

### Expected Paper Role

This tests the mechanism more deeply. PrefixIG should penalize evidence that
does not increase answer likelihood, and reward-gated efficiency should avoid
rewarding short but wrong trajectories.

## Experiment 5: Offline Qwen LoRA Single-Hop vs Multi-Hop

### Goal

Now that the offline rg_eff Qwen2.5-0.5B LoRA pilot works, build small real-model
versions of the single-hop and multi-hop experiments.

### Datasets

```text
outputs/offline_rg_eff_qwen_singlehop_sft
outputs/offline_rg_eff_qwen_multihop_sft
outputs/offline_rg_eff_qwen_mixedhop_sft
```

Each dataset should include generated trajectory groups and full candidate JSONL
with:

```text
target_weight
reward
PrefixIG
turn_IGs
redundancy_penalty
bucket label
```

### Training Variants

Start with SFT distillation because it already works:

```text
base Qwen evaluation only
rg_eff single-hop LoRA
rg_eff multi-hop LoRA
rg_eff mixed-hop LoRA
```

Then move to true offline TPO when implemented:

```text
reward-only offline TPO
PrefixIG offline TPO
PrefixIG + rg_eff offline TPO
```

### Metrics

```text
generated-policy OriginalPolicy row
useful_correct
redundant_correct
distractor_wrong
answer correctness
average search turns
correct per search turn
held-out SFT loss/perplexity as secondary metric
```

### Expected Paper Role

This is the real-model bridge from toy RLVR to language-model behavior. The
first pilot already showed:

```text
Base Qwen OriginalPolicy useful: 0.114, distractor: 0.886, correct: 0.114
rg_eff-LoRA OriginalPolicy useful: 0.880, distractor: 0.120, correct: 0.880
```

The next version should increase sample size and separate single-hop from
multi-hop.

## Recommended Order

### Step 1

Run Experiment 1: A-TGPO component ablation in toy RLVR.

Why first:

- Fast.
- Multi-seed feasible.
- Directly connects to original A-TGPO.
- Gives a clean ablation table.

### Step 2

Run Experiment 2: single-hop vs multi-hop mini search in the toy/generated
diagnostic setup.

Why second:

- Directly mirrors the original repo's single-hop and multi-hop settings.
- Helps frame where the method is most useful.

### Step 3

Run Experiment 4: noisy retriever robustness.

Why third:

- Strong mechanism test.
- Helps show PrefixIG is not just memorizing answer patterns.

### Step 4

Run Experiment 5: Qwen LoRA single-hop vs multi-hop.

Why fourth:

- More expensive.
- Builds on the stable toy and diagnostic results.

### Step 5

Implement true offline TPO training over grouped Qwen trajectories.

Why last:

- Most engineering work.
- Best done after we know which variants matter.

## Current Best Next Action

Implement Experiment 1 by splitting the toy RLVR `prefixig_atgpo` method into
explicit A-TGPO component ablations:

```text
prefixig_grpo
prefixig_grpo_turn_norm
prefixig_grpo_turn_norm_vr
prefixig_atgpo
prefixig_tpo
prefixig_tpo_rg_eff
```

Then run a 5-seed, 200-episode table similar to the existing toy RLVR result.
