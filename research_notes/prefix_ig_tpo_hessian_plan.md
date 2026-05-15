# PrefixIG-TPO with Curvature-Aware Trust

## Goal

Build the first publishable implementation around a cheaper signal than counterfactual evidence credit:

```text
prefix information gain -> target policy construction
```

Instead of asking whether evidence is counterfactually necessary, we reuse the A²TGPO-style prefix information-gain signal and test whether it works better when used to construct a TPO target distribution rather than only as scalar GRPO/PPO advantage shaping.

## Core Claim

A²TGPO uses prefix information gain as turn-level advantage shaping. We test a different use of the same signal:

```text
Information gain should define the target distribution over sampled trajectories/search turns.
TPO then projects the policy toward that target.
```

This is cheaper than CEC because it avoids leave-one-out or replacement rescoring.

## Method Sketch

For each prompt, sample `K` trajectories. Each trajectory may contain multiple search/result turns.

For trajectory `i`, compute:

```text
R_i = final answer reward
IG_i = normalized aggregate prefix information gain
U_i = R_i + lambda * IG_i - rho * redundant_search_penalty
```

Then construct a TPO-style target distribution:

```text
q_i proportional to stopgrad(pi_old_i) * exp(U_i / tau)
```

Train the model to match `q_i` over the sampled completions.

## Prefix Information Gain

For each search turn `t`:

```text
IG_t = log P(a* | context up to turn t)
     - log P(a* | context up to turn t-1)
```

where `a*` is the ground-truth answer.

Trajectory-level utility can use one of:

```text
IG_sum     = sum_t normalized_IG_t
IG_mean    = mean_t normalized_IG_t
IG_max     = max_t normalized_IG_t
IG_disc    = sum_t gamma^(T-t) * normalized_IG_t
IG_v1d     = discounted sum divided by sqrt(number of accumulated terms)
```

Start with `IG_v1d` because it is closest to the A²TGPO implementation.

## Curvature / Hessian Information

Hessian information should be used as a trust or stability mechanism, not as the headline novelty.

Full Hessians are too expensive for laptop-scale MLX experiments. Start with cheap curvature proxies.

### Option A: TPO Group Curvature

The TPO loss over sampled completions uses a softmax over sequence log-scores. The local curvature of the softmax projection is related to:

```text
H_i approx p_i * (1 - p_i)
```

where `p_i` is the current policy probability over sampled completions within the group.

This gives a cheap group-level confidence proxy:

```text
curv_i = p_i * (1 - p_i)
U_i_curv = U_i / sqrt(curv_i + eps)
```

Interpretation:

- high curvature means the target is sensitive to this sample;
- low curvature may indicate saturation or low local sensitivity;
- clipping this scale is important.

### Option B: Log-Prob Variance Proxy

For a trajectory or search turn:

```text
curv_i = variance(token_log_probs_i)
U_i_curv = U_i / sqrt(curv_i + eps)
```

This is simple and cheap. It treats unstable/high-variance generations more conservatively.

### Option C: Entropy-Based Trust

Use token or sequence entropy as a confidence proxy:

```text
tau_i = tau_base * (1 + beta * entropy_i)
```

Higher entropy makes the target softer. Lower entropy permits sharper updates.

## Initial Baselines

Run these before adding curvature:

```text
FinalReward-GRPO
FinalReward-TPO
PrefixIG-GRPO
PrefixIG-TPO
```

Then add:

```text
PrefixIG-TPO + group-curvature trust
PrefixIG-TPO + logprob-variance trust
PrefixIG-TPO + entropy-temperature trust
```

## Minimal First Implementation

Implement a smoke-test credit harness before changing the optimizer:

1. Parse `<search>`, `<result>`, and `<answer>` blocks.
2. Compute final answer EM/F1 reward.
3. Compute answer log-likelihood for a given context.
4. Compute prefix IG for each search turn.
5. Aggregate prefix IG into trajectory utility.
6. Build a TPO target distribution from final reward plus prefix IG.
7. Print target weights and diagnostics for one prompt with multiple sampled trajectories.

Success criterion:

```text
Given multiple sampled completions for one prompt, the target distribution should put more mass on completions with correct answers and useful search turns.
```

## First Experiment

Use a controlled synthetic multi-hop search task.

Each example should include:

- one necessary bridge snippet,
- one answer snippet,
- one redundant or distractor snippet.

Metrics:

```text
answer accuracy
format validity
average search turns
prefix IG distribution
target entropy
target mass on correct trajectories
target mass on useful-search trajectories
```

## Paper Framing

Main paper claim:

```text
Turn-level information gain is more effective when used to construct target policies than when used only as scalar advantage shaping.
```

Possible title:

```text
Targeting Informative Search: Information-Gain Target Policies for Retrieval-Augmented Reasoning
```

CEC remains a future extension:

```text
Counterfactual evidence credit can replace prefix IG when additional training-time compute is available.
```

## First Code Target

The first code target should be an MLX-side module, not a veRL change:

```text
mlx_lm_lora/trainer/prefix_ig_tpo.py
scripts/prefix_ig_tpo_smoke.py
```

The smoke script should run before any full training job.
