# Offline Qwen LoRA Baseline Training Summary

Date: 2026-05-15

Base model: `Qwen/Qwen2.5-0.5B-Instruct`

Training setup:

- 4-bit loading
- LoRA rank 8, scale 20.0, dropout 0.0
- 8 adapted layers
- batch size 1
- gradient accumulation 8
- 80 training iterations
- learning rate `5e-5`
- max sequence length 1024
- trainable parameters: 1.494M / 494.033M, or 0.302%

The first three datasets were built from the same cached 64 generated Qwen
candidate trajectories in `outputs/offline_rg_eff_qwen_candidates.jsonl`. The
A-TGPO proxy dataset reuses the same cached trajectories, but recomputes
token-level A-TGPO component proxy weights with the Qwen scorer. Each target
method produced 128 SFT records, split into 102 train, 13 validation, and 13
test records.

## Target Masses

Average target mass per case during dataset construction:

| target method | useful_correct | redundant_correct | distractor_wrong |
| --- | ---: | ---: | ---: |
| reward_tpo | 0.296 | 0.357 | 0.347 |
| prefixig_tpo | 0.414 | 0.342 | 0.244 |
| prefixig_tpo_rg_eff | 0.471 | 0.228 | 0.302 |
| atgpo_proxy | 0.361 | 0.385 | 0.253 |

Interpretation: reward-only TPO puts similar mass on redundant-correct and
distractor trajectories. PrefixIG-TPO shifts mass toward useful-correct
trajectories. Reward-gated efficiency further reduces redundant-correct mass
while keeping useful-correct mass highest. The A-TGPO proxy assigns slightly
more mass to redundant-correct than useful-correct on this cached batch, which
is a useful contrast for testing whether scalar/token-level credit can overvalue
redundant evidence.

## Training Results

| adapter | final train loss | final validation loss | peak memory |
| --- | ---: | ---: | ---: |
| `outputs/adapters/offline_reward_tpo_qwen_lora` | 0.100 | 0.332 | 1.308GB |
| `outputs/adapters/offline_prefixig_tpo_qwen_lora` | 0.082 | 0.251 | 1.308GB |
| `outputs/adapters/offline_prefixig_tpo_rg_eff_qwen_lora` | 0.100 | 0.108 | 1.308GB |
| `outputs/adapters/offline_atgpo_proxy_qwen_lora` | 0.207 | 0.398 | 1.308GB |

These losses only show that the target distributions are learnable by small
LoRA adapters. The next step is generated-policy evaluation on held-out prompts,
where we compare useful, redundant, distractor, and correctness behavior.

## Small Generated-Policy Evaluation

Each model was evaluated with 4 examples and 2 samples per example. The table
uses only the `OriginalPolicy` row from each diagnostic run, because that row is
the actual generated behavior of the loaded model or adapter.

| loaded model | useful | redundant | distractor | correct | useful-red |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen | 0.375 | 0.137 | 0.488 | 0.512 | +0.239 |
| reward_tpo LoRA | 0.500 | 0.374 | 0.126 | 0.874 | +0.126 |
| prefixig_tpo LoRA | 0.652 | 0.000 | 0.348 | 0.652 | +0.652 |
| prefixig_tpo_rg_eff LoRA | 0.631 | 0.369 | 0.000 | 1.000 | +0.261 |
| atgpo_proxy LoRA | 1.000 | 0.000 | 0.000 | 1.000 | +1.000 |

This is a pilot result, not a final paper-scale result. The A-TGPO proxy adapter
is very strong on this tiny evaluation, but it should be rerun with more
examples and seeds before making a main claim.

## Held-Out 5x2 Evaluation

We reran the generated-policy comparison with 5 examples, 2 samples per example,
seed 101, flexible rollouts, at most 2 search turns, `action-max-tokens=40`, and
`answer-max-tokens=32`. The table again uses only `OriginalPolicy`, i.e. the
actual generated behavior of each loaded model.

| loaded model | useful | redundant | distractor | correct | useful-red |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen | 0.174 | 0.000 | 0.826 | 0.174 | +0.174 |
| reward_tpo LoRA | 0.600 | 0.400 | 0.000 | 1.000 | +0.200 |
| prefixig_tpo LoRA | 0.914 | 0.000 | 0.086 | 0.914 | +0.914 |
| prefixig_tpo_rg_eff LoRA | 0.391 | 0.307 | 0.302 | 0.698 | +0.084 |
| atgpo_proxy LoRA | 0.801 | 0.000 | 0.199 | 0.801 | +0.801 |

This is the cleaner Qwen LoRA pilot result. Reward-only training improves
correctness but creates redundant correct search. PrefixIG-TPO gives the best
useful-minus-redundant gap and the best balance of correctness and useful
search. A-TGPO proxy remains a strong baseline. The current reward-gated
efficiency variant is not reliable on this held-out seed and should be treated
as an ablation rather than the main method.

## Noisy/Distractor 5x2 Evaluation

We also evaluated the same five loaded models in a noisy/distractor regime where
the search index presents a wrong birthplace result before the correct one. This
is a harsh robustness test for a small model.

| loaded model | useful | redundant | distractor | correct | useful-red |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen | 0.000 | 0.000 | 1.000 | 0.000 | +0.000 |
| reward_tpo LoRA | 0.000 | 0.000 | 1.000 | 0.000 | +0.000 |
| prefixig_tpo LoRA | 0.111 | 0.000 | 0.889 | 0.111 | +0.111 |
| prefixig_tpo_rg_eff LoRA | 0.000 | 0.000 | 1.000 | 0.000 | +0.000 |
| atgpo_proxy LoRA | 0.090 | 0.000 | 0.910 | 0.090 | +0.090 |

All methods mostly fail under this adversarial retrieval ordering. PrefixIG-TPO
is the only method with a clear improvement over base and reward-only, and it is
slightly ahead of the A-TGPO proxy on this seed. This should be reported as a
stress-test result, not as the main Qwen LoRA performance claim.

## Mixed-Hop Qwen LoRA Follow-Up

The single-hop evaluation exposed an important failure mode: the original
PrefixIG-TPO LoRA was trained on multi-hop-style trajectories and over-searched
when only one useful search was needed. To test whether this is a data-regime
problem rather than a method problem, we built a mixed-hop offline dataset that
alternates single-hop and multi-hop cases during target construction.

The mixed-hop dataset used 16 prompts, 4 generated candidates per prompt, and 8
SFT draws per prompt. Each target method produced 128 SFT records, split into
102 train, 13 validation, and 13 test records. Average target mass during
dataset construction was:

| target method | useful_correct | distractor_wrong |
| --- | ---: | ---: |
| prefixig_tpo_mixedhop | 0.760 | 0.240 |
| atgpo_proxy_mixedhop | 0.790 | 0.210 |

Both adapters used the same Mac-feasible Qwen2.5-0.5B LoRA setup as the earlier
baselines: 4-bit loading, rank-8 LoRA over 8 layers, batch size 1, gradient
accumulation 8, 80 iterations, and masked-prompt SFT. The PrefixIG-TPO mixed-hop
adapter reached final train loss `0.106` and validation loss `0.149`. The
A-TGPO-proxy mixed-hop adapter reached final train loss `0.209` and validation
loss `0.364`.

We then evaluated all five earlier model conditions plus the two mixed-hop
adapters on held-out single-hop, multi-hop, and noisy regimes. Each result uses
5 examples, 2 samples per example, seed 101, and reports only the
`OriginalPolicy` row, i.e. actual generated behavior from the loaded model.

Single-hop:

| loaded model | useful | redundant | distractor | correct | useful-red |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen | 0.724 | 0.200 | 0.076 | 0.924 | +0.524 |
| reward_tpo LoRA | 0.800 | 0.200 | 0.000 | 1.000 | +0.600 |
| prefixig_tpo LoRA | 0.382 | 0.618 | 0.000 | 1.000 | -0.236 |
| prefixig_tpo_rg_eff LoRA | 0.102 | 0.898 | 0.000 | 1.000 | -0.796 |
| atgpo_proxy LoRA | 0.902 | 0.098 | 0.000 | 1.000 | +0.804 |
| prefixig_tpo_mixedhop LoRA | 1.000 | 0.000 | 0.000 | 1.000 | +1.000 |
| atgpo_proxy_mixedhop LoRA | 0.215 | 0.607 | 0.178 | 0.822 | -0.392 |

Multi-hop:

| loaded model | useful | redundant | distractor | correct | useful-red |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen | 0.174 | 0.000 | 0.826 | 0.174 | +0.174 |
| reward_tpo LoRA | 0.600 | 0.400 | 0.000 | 1.000 | +0.200 |
| prefixig_tpo LoRA | 0.914 | 0.000 | 0.086 | 0.914 | +0.914 |
| prefixig_tpo_rg_eff LoRA | 0.391 | 0.307 | 0.302 | 0.698 | +0.084 |
| atgpo_proxy LoRA | 0.801 | 0.000 | 0.199 | 0.801 | +0.801 |
| prefixig_tpo_mixedhop LoRA | 0.305 | 0.000 | 0.695 | 0.305 | +0.305 |
| atgpo_proxy_mixedhop LoRA | 0.910 | 0.000 | 0.090 | 0.910 | +0.910 |

Noisy/distractor:

| loaded model | useful | redundant | distractor | correct | useful-red |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen | 0.000 | 0.000 | 1.000 | 0.000 | +0.000 |
| reward_tpo LoRA | 0.000 | 0.000 | 1.000 | 0.000 | +0.000 |
| prefixig_tpo LoRA | 0.111 | 0.000 | 0.889 | 0.111 | +0.111 |
| prefixig_tpo_rg_eff LoRA | 0.000 | 0.000 | 1.000 | 0.000 | +0.000 |
| atgpo_proxy LoRA | 0.090 | 0.000 | 0.910 | 0.090 | +0.090 |
| prefixig_tpo_mixedhop LoRA | 0.000 | 0.000 | 1.000 | 0.000 | +0.000 |
| atgpo_proxy_mixedhop LoRA | 0.000 | 0.105 | 0.895 | 0.105 | -0.105 |

The mixed-hop PrefixIG-TPO result is useful because it fixes the earlier
single-hop over-search failure: useful-red improves from `-0.236` to `+1.000`
while preserving perfect correctness on this tiny single-hop evaluation.
However, it loses the strong multi-hop behavior of the original PrefixIG-TPO
adapter. The mixed-hop A-TGPO proxy shows the opposite pattern: it is strong on
multi-hop but poor on single-hop. Neither mixed-hop adapter solves the noisy
distractor regime. The next real-model step should therefore be a larger,
balanced mixed-hop target set and a dedicated noisy-retrieval training split
rather than relying on hop mixing alone.

## Mixed-Hop + Noisy Qwen LoRA Follow-Up

We then added an explicit mixed-hop + noisy training regime. This regime cycles
single-hop, multi-hop, and noisy/distractor cases during offline target
construction. The first version was not trainable: because the base model rarely
sampled useful noisy trajectories, more than half of the target mass remained on
`distractor_wrong`. We therefore made two targeted changes:

- The noisy retriever now returns tied birthplace evidence together, preserving
  the intended setting of misleading evidence before corrective evidence rather
  than an impossible one-result retrieval failure.
- The offline target builder can add useful-correct oracle anchors to each
  candidate group with `--oracle-anchor-count`, ensuring the target policy has
  positive trajectories to move mass toward.

With 18 prompts, 4 generated samples per prompt, 2 useful-correct oracle anchors
per prompt, and 8 SFT draws per prompt, the corrected target distributions were:

| target method | useful_correct | redundant_correct | distractor_wrong |
| --- | ---: | ---: | ---: |
| prefixig_tpo_mixedhop_noisy | 0.804 | 0.083 | 0.113 |
| atgpo_proxy_mixedhop_noisy | 0.776 | 0.096 | 0.128 |

Both adapters used the same Qwen2.5-0.5B 4-bit LoRA setup as before. Training
results:

| adapter | final train loss | final validation loss | peak memory |
| --- | ---: | ---: | ---: |
| `outputs/adapters/offline_prefixig_tpo_mixedhop_noisy_qwen_lora` | 0.038 | 0.247 | 1.291GB |
| `outputs/adapters/offline_atgpo_proxy_mixedhop_noisy_qwen_lora` | 0.089 | 0.081 | 1.275GB |

Held-out single-hop:

| loaded model | useful | redundant | distractor | correct | useful-red |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen | 0.724 | 0.200 | 0.076 | 0.924 | +0.524 |
| reward_tpo LoRA | 0.800 | 0.200 | 0.000 | 1.000 | +0.600 |
| prefixig_tpo LoRA | 0.382 | 0.618 | 0.000 | 1.000 | -0.236 |
| atgpo_proxy LoRA | 0.902 | 0.098 | 0.000 | 1.000 | +0.804 |
| prefixig_tpo_mixedhop LoRA | 1.000 | 0.000 | 0.000 | 1.000 | +1.000 |
| prefixig_tpo_mixedhop_noisy LoRA | 0.800 | 0.200 | 0.000 | 1.000 | +0.600 |
| atgpo_proxy_mixedhop_noisy LoRA | 0.526 | 0.000 | 0.474 | 0.526 | +0.526 |

Held-out multi-hop:

| loaded model | useful | redundant | distractor | correct | useful-red |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen | 0.174 | 0.000 | 0.826 | 0.174 | +0.174 |
| reward_tpo LoRA | 0.600 | 0.400 | 0.000 | 1.000 | +0.200 |
| prefixig_tpo LoRA | 0.914 | 0.000 | 0.086 | 0.914 | +0.914 |
| atgpo_proxy LoRA | 0.801 | 0.000 | 0.199 | 0.801 | +0.801 |
| prefixig_tpo_mixedhop LoRA | 0.305 | 0.000 | 0.695 | 0.305 | +0.305 |
| prefixig_tpo_mixedhop_noisy LoRA | 1.000 | 0.000 | 0.000 | 1.000 | +1.000 |
| atgpo_proxy_mixedhop_noisy LoRA | 0.800 | 0.000 | 0.200 | 0.800 | +0.800 |

Corrected noisy/distractor:

| loaded model | useful | redundant | distractor | correct | useful-red |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen | 0.093 | 0.000 | 0.907 | 0.093 | +0.093 |
| reward_tpo LoRA | 0.200 | 0.400 | 0.400 | 0.600 | -0.200 |
| prefixig_tpo LoRA | 0.505 | 0.200 | 0.295 | 0.705 | +0.305 |
| atgpo_proxy LoRA | 0.400 | 0.196 | 0.404 | 0.596 | +0.204 |
| prefixig_tpo_mixedhop_noisy LoRA | 0.800 | 0.000 | 0.200 | 0.800 | +0.800 |
| atgpo_proxy_mixedhop_noisy LoRA | 0.800 | 0.000 | 0.200 | 0.800 | +0.800 |

Focused comparison:

| evaluation regime | method | correct | useful | redundant | distractor | useful-red |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| single-hop | PrefixIG-TPO mixedhop+noisy | 1.000 | 0.800 | 0.200 | 0.000 | +0.600 |
| single-hop | A-TGPO proxy mixedhop+noisy | 0.526 | 0.526 | 0.000 | 0.474 | +0.526 |
| multi-hop | PrefixIG-TPO mixedhop+noisy | 1.000 | 1.000 | 0.000 | 0.000 | +1.000 |
| multi-hop | A-TGPO proxy mixedhop+noisy | 0.800 | 0.800 | 0.000 | 0.200 | +0.800 |
| corrected noisy/distractor | PrefixIG-TPO mixedhop+noisy | 0.800 | 0.800 | 0.000 | 0.200 | +0.800 |
| corrected noisy/distractor | A-TGPO proxy mixedhop+noisy | 0.800 | 0.800 | 0.000 | 0.200 | +0.800 |

This is the strongest Qwen LoRA result so far. The PrefixIG-TPO mixed-hop+noisy
adapter preserves perfect correctness on single-hop and multi-hop while sharply
improving noisy retrieval behavior. Unlike reward-only training, it does not
shift mass into redundant correct search under noisy retrieval. The A-TGPO proxy
matches the noisy score, but is weaker on single-hop and multi-hop in this seed.
Because the experiment uses only 5 held-out prompts per regime, this is still a
pilot result; the next step is to repeat it across more seeds and a larger
held-out set.

## True Offline TPO Pilot

We implemented a standalone true offline TPO trainer in
`scripts/train_offline_tpo_lora.py`. Unlike the earlier target-weighted SFT
distillation, this trainer loads all candidate trajectories for a case, scores
each completion under the current model, forms a softmax distribution over the
group, and minimizes cross-entropy to the stored PrefixIG-TPO target weights.

The trainer is feasible on the Mac setup with Qwen2.5-0.5B 4-bit LoRA. A full
80-iteration scratch run peaked at about `10.05GB` memory and finished in a few
minutes. It reached validation loss around `1.30` and test loss `1.238`. We also
ran a 40-iteration low-learning-rate refinement initialized from the strong
weighted-SFT adapter.

Held-out comparison:

| regime | model | correct | useful | redundant | distractor | useful-red |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| single-hop | PrefixIG-TPO weighted-SFT | 1.000 | 0.800 | 0.200 | 0.000 | +0.600 |
| single-hop | PrefixIG-TPO true-TPO scratch | 0.485 | 0.485 | 0.000 | 0.283 | +0.485 |
| single-hop | PrefixIG-TPO SFT-warm true-TPO | 1.000 | 1.000 | 0.000 | 0.000 | +1.000 |
| multi-hop | PrefixIG-TPO weighted-SFT | 1.000 | 1.000 | 0.000 | 0.000 | +1.000 |
| multi-hop | PrefixIG-TPO true-TPO scratch | 0.190 | 0.190 | 0.000 | 0.703 | +0.190 |
| multi-hop | PrefixIG-TPO SFT-warm true-TPO | 0.333 | 0.333 | 0.000 | 0.667 | +0.333 |
| corrected noisy | PrefixIG-TPO weighted-SFT | 0.800 | 0.800 | 0.000 | 0.200 | +0.800 |
| corrected noisy | PrefixIG-TPO true-TPO scratch | 0.000 | 0.000 | 0.000 | 0.585 | +0.000 |
| corrected noisy | PrefixIG-TPO SFT-warm true-TPO | 0.419 | 0.337 | 0.082 | 0.581 | +0.256 |

Interpretation: true offline TPO is now implemented, but the first pure grouped
loss is under-constrained for generation quality. It optimizes relative
probability mass within candidate groups, but it does not directly preserve
formatting and trajectory imitation the way SFT does. Warm-starting from the
weighted-SFT adapter fixes single-hop and removes redundancy there, but still
hurts multi-hop and noisy behavior. The next version should use a hybrid
objective, for example `offline_TPO_loss + beta * SFT_loss` on high-target
trajectories, or a KL/format regularizer, before making true offline TPO the
paper headline.

## Online PrefixIG-TPO + Token/Action Anchor Pilot

We implemented `scripts/online_prefixig_tpo_lora.py` as the first online
target-policy optimization pilot. The run warm-starts from
`outputs/adapters/offline_prefixig_tpo_mixedhop_noisy_qwen_lora`, samples fresh
rollouts from the current policy each online round, builds PrefixIG-TPO target
weights over each sampled group, and applies a grouped TPO update with a small
target-weighted token/action NLL anchor.

Configuration:

| setting | value |
| --- | --- |
| model | Qwen2.5-0.5B-Instruct |
| quantization | 4-bit |
| online iterations | 5 |
| prompts per iteration | 2 |
| samples per prompt | 3 |
| update steps per iteration | 2 |
| learning rate | `1e-5` |
| anchor weight | `0.05` |
| peak memory | about `7.5GB` |

Held-out comparison, 5 examples x 2 samples, seed 101:

| regime | model | correct | useful | redundant | distractor | useful-red |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| single-hop | PrefixIG-TPO weighted-SFT | 1.000 | 0.800 | 0.200 | 0.000 | +0.600 |
| single-hop | A-TGPO proxy | 0.526 | 0.526 | 0.000 | 0.474 | +0.526 |
| single-hop | online PrefixIG-TPO+anchor | 0.898 | 0.698 | 0.200 | 0.102 | +0.498 |
| multi-hop | PrefixIG-TPO weighted-SFT | 1.000 | 1.000 | 0.000 | 0.000 | +1.000 |
| multi-hop | A-TGPO proxy | 0.800 | 0.800 | 0.000 | 0.200 | +0.800 |
| multi-hop | online PrefixIG-TPO+anchor | 0.804 | 0.804 | 0.000 | 0.196 | +0.804 |
| noisy | PrefixIG-TPO weighted-SFT | 0.800 | 0.800 | 0.000 | 0.200 | +0.800 |
| noisy | A-TGPO proxy | 0.800 | 0.800 | 0.000 | 0.200 | +0.800 |
| noisy | online PrefixIG-TPO+anchor | 0.902 | 0.902 | 0.000 | 0.098 | +0.902 |

Interpretation: online PrefixIG-TPO with an action anchor is much more stable
than pure offline TPO and improves noisy retrieval. It is also the first
non-SFT-main-objective version of our method that is competitive with A-TGPO
proxy across all three regimes: it is much better on single-hop, slightly
better on multi-hop, and better on noisy retrieval. This matters because the
true-offline-TPO and KL-anchored offline-TPO variants were usually behind
A-TGPO on multi-hop and noisy retrieval.

The limitation is that this first online setting hurts the very strong
weighted-SFT warm start on single-hop and multi-hop. The next tuning pass should
reduce update strength or increase anchoring, then repeat this same three-regime
comparison.

### Anchor-Beta Sweep

We ran a small sweep over the token/action anchor weight while keeping the
online schedule fixed. The weak-anchor setting (`beta=0.02`) was unstable on
multi-hop and noisy prompts. Increasing the anchor to `beta=0.10` preserved the
clean-regime behavior and improved noisy retrieval.

| anchor beta | regime | correct | useful | redundant | distractor | useful-red |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.02 | single-hop | 0.896 | 0.696 | 0.200 | 0.104 | +0.496 |
| 0.02 | multi-hop | 0.402 | 0.402 | 0.000 | 0.598 | +0.402 |
| 0.02 | noisy | 0.702 | 0.702 | 0.000 | 0.298 | +0.702 |
| 0.10 | single-hop | 1.000 | 0.800 | 0.200 | 0.000 | +0.600 |
| 0.10 | multi-hop | 1.000 | 1.000 | 0.000 | 0.000 | +1.000 |
| 0.10 | noisy | 1.000 | 1.000 | 0.000 | 0.000 | +1.000 |

Interpretation: the anchor is not just a stabilizer in principle; its strength
controls whether the online update preserves the action language and clean
multi-hop behavior. The `beta=0.10` pilot is the strongest online result so far,
but it should be repeated with more held-out prompts and seeds before being used
as a final headline.
