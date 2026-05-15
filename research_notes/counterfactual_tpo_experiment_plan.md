# Experiment Plan: Counterfactual Target Policy Optimization

## Goal

Test whether counterfactual evidence credit improves multi-turn search RL beyond final-reward-only training and prefix-information-gain methods.

The paper should frame small models as a controlled setting for studying credit assignment, not as a laptop constraint.

## Main Hypotheses

1. Prefix information gain over-rewards late, redundant, or order-lucky search turns.
2. Counterfactual evidence credit better identifies which retrieved results were necessary for the final answer.
3. Counterfactual credit works better when used to construct a TPO-style target distribution than when used only as scalar GRPO/PPO advantage shaping.
4. JS or alpha-divergence target projection is more stable than plain forward KL when counterfactual credit is noisy.

## Phase 0: Reproducible Mini Stack

Create a self-contained training script in the MLX repo, not inside the full A²TGPO stack.

Required components:

- prompt format with `<think>`, `<search>`, `<result>`, and `<answer>` tags,
- parser for generated search turns,
- cached or oracle retrieval results,
- final answer EM/F1 reward,
- answer-likelihood scorer `s(E) = log P(a* | question, evidence E)`,
- prefix IG scorer,
- counterfactual evidence credit scorer,
- GRPO/PPO-style scalar advantage baseline,
- TPO-style target distribution update.

Success criterion:

- One small run completes end-to-end.
- Logs include final reward, prefix IG, CEC, target weights, answer accuracy, and search-turn count.

## Phase 1: Controlled Synthetic Search Task

Purpose:

Show the mechanism clearly before using messy QA benchmarks.

Task design:

- Each question requires 2 pieces of evidence.
- Retrieval returns useful, redundant, and distractor snippets.
- Some examples require evidence synergy:

```text
e1: The author of Book X is Person Y.
e2: Person Y was born in City Z.
Question: Where was the author of Book X born?
```

Controlled failure cases:

- redundant evidence,
- late evidence that only looks useful because earlier evidence set it up,
- misleading distractor evidence,
- two-hop synergy where the first hop has low prefix IG.

Baselines:

- final-reward GRPO,
- trajectory-level TPO,
- prefix-IG GRPO/A²TGPO-style advantage,
- prefix-IG TPO,
- CEC-GRPO,
- CEC-TPO.

Metrics:

- answer accuracy,
- evidence selection accuracy,
- useful-search rate,
- redundant-search rate,
- correlation between assigned credit and ground-truth useful evidence,
- training stability.

Success criterion:

- CEC assigns higher credit to necessary early evidence than prefix IG.
- CEC-TPO beats CEC-GRPO or reaches the same accuracy with fewer/redundant searches.

## Phase 2: Bamboogle / NQ-Small

Purpose:

Move from synthetic tasks to real QA while keeping fast iteration.

Recommended datasets:

- Bamboogle for compact multi-hop QA,
- NQ-small for single-hop sanity,
- optionally a small HotpotQA slice for bridge/comparison questions.

Retrieval setup:

- Start with cached/oracle snippets.
- Add distractor snippets from other examples.
- Later replace oracle retrieval with a local retrieval index if needed.

Training variants:

```text
FinalReward-GRPO
FinalReward-TPO
PrefixIG-GRPO
PrefixIG-TPO
CEC-GRPO
CEC-TPO
```

CEC variants:

```text
LOO-CEC:         s(E_all) - s(E_without_t)
Replace-CEC:     s(E_all) - s(E_with_distractor_t)
Sampled-Shapley: average_s [s(S union e_t) - s(S)]
```

Metrics:

- exact match,
- token F1,
- format validity,
- average number of search turns,
- useful vs redundant search rate,
- CEC/prefix-IG agreement,
- target entropy,
- target mass assigned to successful turns.

Success criterion:

- CEC-TPO improves answer accuracy or search efficiency over prefix-IG TPO.
- At minimum, CEC produces better credit-alignment analysis even before large accuracy gains.

## Phase 3: Multi-Hop Real Benchmarks

Purpose:

Show the method matters when order and synergy are real.

Datasets:

- HotpotQA-small,
- 2WikiMultiHopQA-small,
- MuSiQue/MiniMuSiQue if feasible.

Key experiment:

Compare prefix IG against CEC on bridge questions where early evidence is necessary but not immediately answer-revealing.

Metrics:

- answer EM/F1,
- supporting-evidence recall if labels are available,
- search efficiency,
- early-turn credit quality,
- failure mode categories.

Success criterion:

- CEC improves early-turn credit and reduces redundant late searches.
- CEC-TPO has stronger or more stable gains than CEC-GRPO.

## Phase 4: TPO and Divergence Ablations

Purpose:

Show TPO is not decorative.

Ablations:

```text
Counterfactual credit as scalar advantage only
Counterfactual credit as target distribution
Forward-KL target projection
Reverse-KL target projection
JS target projection
Alpha divergence target projection
No counterfactual credit, target from final reward only
No final reward, counterfactual credit only
```

Expected result:

- Scalar CEC helps.
- CEC target policy helps more or is more stable.
- JS/alpha helps when CEC is noisy.
- Reverse KL may be strong but less stable.

Plots:

- validation accuracy over updates,
- target entropy over updates,
- KL to reference/current policy,
- distribution of CEC values,
- probability mass shift toward high-CEC turns,
- clip/projection strength vs CEC confidence.

## Phase 5: Optional Curvature-Aware Trust

Only add this if earlier phases show instability.

Variants:

- entropy-scaled CEC,
- log-prob-variance-scaled CEC,
- KL-scaled target temperature,
- Fisher-diagonal proxy if cheap.

Example:

```text
U_t = final_reward + lambda * CEC_t / sqrt(var_logprob_t + eps)
```

Success criterion:

- fewer collapsed search behaviors,
- smoother target entropy,
- improved performance under larger learning rates or sharper target temperatures.

## Minimal First Implementation Order

1. Implement parser for search/result/answer tags.
2. Implement final EM/F1 reward.
3. Implement answer-likelihood scorer.
4. Implement prefix IG.
5. Implement leave-one-out CEC.
6. Add CEC as scalar advantage to verify signal.
7. Add TPO target construction from CEC utility.
8. Run controlled synthetic task.
9. Run Bamboogle/NQ-small.
10. Add replacement CEC.
11. Add sampled-Shapley CEC only if leave-one-out is promising.
12. Add JS/alpha divergence ablations.

## First Paper-Quality Table

Rows:

```text
FinalReward-GRPO
FinalReward-TPO
PrefixIG-GRPO
PrefixIG-TPO
LOO-CEC-GRPO
LOO-CEC-TPO
Replace-CEC-TPO
Sampled-Shapley-CEC-TPO
```

Columns:

```text
Answer EM
Answer F1
Search turns
Useful-search rate
Redundant-search rate
Credit alignment
Target entropy
```

## First Paper-Quality Figure

Use a single example with 3 search turns:

```text
turn 1: necessary bridge evidence
turn 2: answer evidence
turn 3: redundant evidence
```

Plot:

```text
prefix IG credit vs CEC credit vs target probability mass
```

This figure should visually show why prefix IG miscredits and why counterfactual target construction is better.

