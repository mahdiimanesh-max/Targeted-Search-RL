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

### Completed Pilot Result

We added `--hop-mode {single,multi,mixed}` to:

```text
scripts/toy_prefixig_rlvr_train.py
```

The toy action vocabulary now separates one-hop answer evidence from two-hop
bridge-plus-answer evidence. We ran 5-seed, 200-episode single-hop and multi-hop
experiments with:

```text
reward_tpo
prefixig_grpo
prefixig_tpo
prefixig_tpo_rg_eff
```

Outputs:

```text
outputs/toy_prefixig_rlvr_singlehop_5seed.csv
outputs/toy_prefixig_rlvr_multihop_5seed.csv
outputs/figures_singlehop_5seed/
outputs/figures_multihop_5seed/
```

Single-hop final metrics:

```text
method              | reward | useful | redundant | distractor | useful-red
--------------------+--------+--------+-----------+------------+-----------
reward_tpo          | 0.911  | 0.086  | 0.027     | 0.015      | +0.059
prefixig_grpo       | 0.594  | 0.594  | 0.001     | 0.000      | +0.593
prefixig_tpo        | 0.810  | 0.779  | 0.017     | 0.015      | +0.762
prefixig_tpo_rg_eff | 0.811  | 0.793  | 0.010     | 0.017      | +0.783
```

Multi-hop final metrics:

```text
method              | reward | useful | redundant | distractor | useful-red
--------------------+--------+--------+-----------+------------+-----------
reward_tpo          | 0.911  | 0.003  | 0.009     | 0.015      | -0.006
prefixig_grpo       | 0.989  | 0.989  | 0.000     | 0.000      | +0.989
prefixig_tpo        | 0.796  | 0.760  | 0.016     | 0.016      | +0.744
prefixig_tpo_rg_eff | 0.769  | 0.741  | 0.008     | 0.022      | +0.732
```

Interpretation:

- Reward-only TPO gets high answer reward in both settings, but it does not
  learn useful evidence behavior, especially in multi-hop where useful evidence
  is almost zero.
- PrefixIG methods strongly increase useful evidence behavior.
- In single-hop, PrefixIG-TPO+rg_eff gives the best useful-minus-redundant gap.
- In multi-hop, raw PrefixIG-GRPO is strongest in this toy setup, while
  PrefixIG-TPO and rg_eff still substantially outperform reward-only on process
  quality.

This is useful for the paper because it shows that answer reward and process
quality can diverge sharply. The multi-hop reward-only model answers correctly
without learning the required evidence process, which motivates explicit
evidence-credit objectives.

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

### Completed Pilot Result

We added a deterministic noisy-retriever mode to:

```text
scripts/toy_prefixig_rlvr_train.py
```

using:

```text
--retrieval-noise clean|distractor25|distractor50|stale25|stale50|mixed25|mixed50
```

The noisy setting corrupts some useful evidence actions into either distractor
evidence or stale/redundant evidence. We ran mixed-hop tasks under moderate and
strong mixed corruption.

Outputs:

```text
outputs/toy_prefixig_rlvr_mixedhop_noisy25_5seed.csv
outputs/toy_prefixig_rlvr_mixedhop_noisy50_5seed.csv
outputs/figures_mixedhop_noisy25_5seed/
outputs/figures_mixedhop_noisy50_5seed/
```

Moderate noise, `mixed25`:

```text
method              | reward | useful | redundant | distractor | useful-red
--------------------+--------+--------+-----------+------------+-----------
reward_tpo          | 0.911  | 0.040  | 0.018     | 0.019      | +0.022
prefixig_grpo       | 0.994  | 0.576  | 0.224     | 0.000      | +0.352
prefixig_tpo        | 0.804  | 0.544  | 0.028     | 0.041      | +0.517
prefixig_tpo_rg_eff | 0.756  | 0.536  | 0.007     | 0.061      | +0.529
```

Strong noise, `mixed50`:

```text
method              | reward | useful | redundant | distractor | useful-red
--------------------+--------+--------+-----------+------------+-----------
reward_tpo          | 0.911  | 0.025  | 0.009     | 0.023      | +0.016
prefixig_grpo       | 0.994  | 0.097  | 0.602     | 0.002      | -0.505
prefixig_tpo        | 0.792  | 0.210  | 0.148     | 0.073      | +0.061
prefixig_tpo_rg_eff | 0.759  | 0.268  | 0.015     | 0.084      | +0.253
```

Interpretation:

- This is our strongest toy evidence so far for the target-policy approach.
- Reward-only keeps high answer reward but does not learn useful evidence.
- PrefixIG-GRPO gets very high reward, but under strong noise it collapses into
  redundant-correct behavior and has negative useful-minus-redundant.
- PrefixIG-TPO is more robust than PrefixIG-GRPO under noisy retrieval.
- PrefixIG-TPO+rg_eff is best on useful-minus-redundant and sharply suppresses
  redundant behavior under both moderate and strong noise.

This supports the main hypothesis better than the clean single-hop/multi-hop
experiment: target construction and reward-gated efficiency matter most when
retrieval contains stale or distracting evidence.

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

### Updated Qwen LoRA Pilot: 5x2 Held-Out Seed

We trained five Qwen2.5-0.5B conditions: base Qwen, reward-only TPO LoRA,
PrefixIG-TPO LoRA, PrefixIG-TPO with reward-gated efficiency, and an A-TGPO
component proxy LoRA. We then evaluated them on 5 held-out examples with 2
samples per example, seed 101, using only the generated-policy `OriginalPolicy`
row for model comparison.

```text
model                    correct | useful | redundant | distractor | useful-red
Base Qwen                0.174   | 0.174  | 0.000     | 0.826      | +0.174
reward_tpo LoRA          1.000   | 0.600  | 0.400     | 0.000      | +0.200
prefixig_tpo LoRA        0.914   | 0.914  | 0.000     | 0.086      | +0.914
prefixig_tpo_rg_eff LoRA 0.698   | 0.391  | 0.307     | 0.302      | +0.084
atgpo_proxy LoRA         0.801   | 0.801  | 0.000     | 0.199      | +0.801
```

Interpretation:

- Reward-only TPO improves correctness but induces redundant correct search.
- PrefixIG-TPO is the strongest current Qwen LoRA result on useful-red and
  correctness/usefulness balance.
- A-TGPO proxy is a strong baseline and should stay in the main comparison.
- The current reward-gated efficiency variant is unstable in this real-model
  pilot and should be treated as an ablation.

Next: repeat this comparison over additional seeds before making a paper-scale
claim.

### Qwen LoRA Noisy/Distractor Stress Test

We added an MLX generated-policy `--eval-regime noisy` setting. It places a
wrong birthplace result before the correct birthplace result, testing whether a
model can avoid following a plausible but bad retrieved fact.

```text
model                    correct | useful | redundant | distractor | useful-red
Base Qwen                0.000   | 0.000  | 0.000     | 1.000      | +0.000
reward_tpo LoRA          0.000   | 0.000  | 0.000     | 1.000      | +0.000
prefixig_tpo LoRA        0.111   | 0.111  | 0.000     | 0.889      | +0.111
prefixig_tpo_rg_eff LoRA 0.000   | 0.000  | 0.000     | 1.000      | +0.000
atgpo_proxy LoRA         0.090   | 0.090  | 0.000     | 0.910      | +0.090
```

Interpretation:

- This noisy ordering is too hard for the current small offline-distilled
  adapters; all methods mostly follow the distractor.
- PrefixIG-TPO is the only method that clearly rises above base and reward-only.
- A-TGPO proxy is close but slightly below PrefixIG-TPO on this seed.
- This belongs in the paper as a stress-test table, not the headline result.

Next: run single-hop clean eval, then repeat the multihop/noisy tables over at
least one additional seed.

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
