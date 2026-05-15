# Counterfactual Target Policy Optimization for Multi-Turn Search Reasoning

## One-Sentence Idea

Multi-turn search RL should not only ask whether a search turn increased answer likelihood at that point in the trajectory; it should ask whether the retrieved evidence was actually necessary for the final answer, and then use that counterfactual contribution to construct a target policy over search actions.

## Motivation

Search-augmented reasoning has a credit assignment problem. A final answer reward tells us whether the whole trajectory succeeded, but it does not tell us which search turn mattered. A prefix-based information gain signal is better, but it is still order-dependent:

```text
IG_t = log P(a* | e_1, ..., e_t) - log P(a* | e_1, ..., e_{t-1})
```

This can misattribute credit when:

- an early result is necessary but only becomes useful after a later result,
- two results are redundant,
- a late result receives credit mostly because it appears after useful context,
- a result looks useful locally but is not needed for the final answer.

A²TGPO already uses prefix-based information gain, turn-group normalization, variance-rescaled accumulation, and adaptive turn-level clipping. Our novelty should therefore not be "turn-level IG" itself. The gap is that prefix IG measures local change along one observed order, not counterfactual contribution to the final reasoning state.

## Main Claim

Prefix information gain is myopic and order-dependent. We propose counterfactual evidence credit, which estimates each search turn's marginal contribution to final answer likelihood under removal, replacement, or permutation interventions. We then use this credit to build a counterfactual target policy, making TPO central rather than an optimizer add-on.

## Method Overview

For a question, the model generates a multi-turn search trajectory:

```text
q_1 -> e_1
q_2 -> e_2
...
q_T -> e_T
final answer y
```

where `q_t` is a generated search query/action and `e_t` is the retrieved result. Let:

```text
s(E) = log P_model(a* | question, evidence set/order E)
```

where `a*` is the ground-truth answer.

### 1. Counterfactual Evidence Credit

The simplest credit is leave-one-out:

```text
CEC_t = s({e_1, ..., e_T}) - s({e_1, ..., e_T} \ {e_t})
```

If removing evidence `e_t` lowers the model's likelihood of the correct answer, then the search turn that produced `e_t` mattered.

### 2. Replacement Credit

Removal can change prompt length and format. A stricter intervention replaces the evidence with a distractor while preserving the structure:

```text
CEC_t = s(E_all) - s(E_all with e_t replaced by d_t)
```

The distractor can be:

- a retrieval result from another question,
- a lower-ranked result for the same query,
- a result from a deliberately corrupted query,
- a randomly sampled irrelevant passage.

### 3. Permutation / Shapley-Style Credit

Leave-one-out can miss synergy. A more principled version estimates marginal contribution across sampled evidence contexts:

```text
CEC_t = average over sampled subsets S:
        s(S union {e_t}) - s(S)
```

This approximates Shapley-style credit without requiring all subsets. In practice, 2-4 sampled subsets or permutations per trajectory may be enough for a useful signal.

## Why TPO Is Central

If we only attach `CEC_t` as a scalar advantage, then the method becomes "better reward shaping for GRPO/PPO." That may work, but it makes TPO secondary.

Our stronger formulation is:

```text
Counterfactual credit defines the target policy.
TPO is the mechanism that projects the model toward that target.
```

For each prompt, we sample multiple trajectories and search turns. We compute a utility:

```text
U_i = final_reward_i + lambda * normalized_CEC_i - rho * redundancy_i
```

Then construct a target distribution over sampled trajectories or turns:

```text
q_i proportional to pi_old_i * exp(U_i / tau)
```

The model is trained to match this target distribution. This is different from simply increasing log-probability proportional to an advantage. The method explicitly redistributes probability mass toward search actions whose evidence is counterfactually necessary.

## Divergence-Controlled Projection

Divergence choice should be a stability mechanism, not the headline novelty.

Potential projection geometries:

- Forward KL: conservative, mode-covering, preserves more behavior.
- Reverse KL: sharper, mode-seeking, may collapse onto a few search styles.
- JS: bounded and often more stable when counterfactual credit is noisy.
- Alpha divergence: interpolates between mode-covering and mode-seeking behavior.

Paper framing:

```text
Counterfactual evidence credit decides what deserves probability mass.
Divergence-controlled projection decides how aggressively to move that mass.
```

## Optional Curvature-Aware Trust

A curvature or Fisher-style proxy can control update strength:

```text
step_scale_t = CEC_t / sqrt(curvature_t + eps)
```

Possible cheap proxies:

- token log-prob variance within the search action,
- entropy of the action/query tokens,
- KL between current and reference policy on the turn,
- gradient norm or Fisher diagonal approximation if available.

This is optional. It can become a strong extension if experiments show instability from high-credit but fragile turns.

## Novelty Boundary

We should not claim:

- turn-level information gain,
- prefix before/after answer-likelihood gain,
- turn-group normalization,
- adaptive turn-level clipping,
- target policy optimization in general.

Those are already covered by nearby work such as A²TGPO and TPO.

We can claim:

- counterfactual evidence contribution for multi-turn search RL,
- removal/replacement/permutation credit beyond prefix IG,
- counterfactual target policy construction,
- showing that counterfactual credit works best when used to build a target distribution rather than only as scalar reward shaping,
- divergence-controlled projection as a stability mechanism for noisy counterfactual credit.

## Candidate Paper Claim

Existing multi-turn search RL methods either optimize sparse trajectory rewards or use prefix-based information gain to shape turn-level updates. We show that prefix information gain is order-dependent and can miscredit redundant or synergistic evidence. We introduce Counterfactual Target Policy Optimization, which estimates each retrieved result's marginal contribution to the final answer under evidence interventions and uses this contribution to construct a target policy over search actions. This improves answer accuracy, search efficiency, and credit alignment in multi-turn retrieval reasoning.

## Candidate Titles

- Counterfactual Target Policy Optimization for Multi-Turn Search Reasoning
- Which Search Evidence Mattered? Counterfactual Credit for Tool-Augmented RL
- Beyond Prefix Information Gain: Counterfactual Target Policies for Search-Augmented Reasoning

