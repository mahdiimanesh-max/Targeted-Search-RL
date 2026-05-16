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
