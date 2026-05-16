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

The three datasets were built from the same cached 64 generated Qwen candidate
trajectories in `outputs/offline_rg_eff_qwen_candidates.jsonl`. Each target
method produced 128 SFT records, split into 102 train, 13 validation, and 13
test records.

## Target Masses

Average target mass per case during dataset construction:

| target method | useful_correct | redundant_correct | distractor_wrong |
| --- | ---: | ---: | ---: |
| reward_tpo | 0.296 | 0.357 | 0.347 |
| prefixig_tpo | 0.414 | 0.342 | 0.244 |
| prefixig_tpo_rg_eff | 0.471 | 0.228 | 0.302 |

Interpretation: reward-only TPO puts similar mass on redundant-correct and
distractor trajectories. PrefixIG-TPO shifts mass toward useful-correct
trajectories. Reward-gated efficiency further reduces redundant-correct mass
while keeping useful-correct mass highest.

## Training Results

| adapter | final train loss | final validation loss | peak memory |
| --- | ---: | ---: | ---: |
| `outputs/adapters/offline_reward_tpo_qwen_lora` | 0.100 | 0.332 | 1.308GB |
| `outputs/adapters/offline_prefixig_tpo_qwen_lora` | 0.082 | 0.251 | 1.308GB |
| `outputs/adapters/offline_prefixig_tpo_rg_eff_qwen_lora` | 0.100 | 0.108 | 1.308GB |

These losses only show that the target distributions are learnable by small
LoRA adapters. The next step is generated-policy evaluation on held-out prompts,
where we compare useful, redundant, distractor, and correctness behavior.
