# Methodology and Experiments Draft

## Methodology

We study reinforcement learning for multi-turn retrieval-augmented reasoning, where a model must decide not only the final answer but also which intermediate search actions are useful. In this setting, final-answer reward is often too sparse. A trajectory that uses two necessary evidence searches and a trajectory that performs several redundant searches may both receive the same terminal correctness reward, even though the first trajectory is more efficient and better aligned with the intended reasoning process. This creates a credit-assignment problem: the optimizer can learn to produce correct answers without learning which search actions actually improved the answer.

Our method starts from the information-gain view used in A-TGPO-style multi-turn reasoning. For a prompt `x`, a generated trajectory `y_i` contains a sequence of search/result turns followed by an answer. Let the ground-truth answer be `a*`. For each search turn `t`, we estimate the incremental value of the newly retrieved evidence by measuring the change in answer log-likelihood:

```text
IG_{i,t} = log p(a* | x, prefix up to turn t)
         - log p(a* | x, prefix before turn t).
```

Positive information gain means that the retrieved evidence made the correct answer more likely under the model. Low or negative gain indicates that the turn was redundant, distracting, or harmful. We aggregate turn-level gains into a trajectory-level evidence score. In the current implementation, we use a normalized aggregate PrefixIG score so that it can be combined with terminal reward across groups:

```text
U_i = R_i + lambda_ig * z(IG_i),
```

where `R_i` is the final-answer reward and `z(IG_i)` is the group-normalized prefix information gain. This signal is computed only during training or offline target construction; inference cost is unchanged.

The key methodological choice is to use PrefixIG to construct a target distribution rather than only as a scalar policy-gradient advantage. In a GRPO/PPO-style update, a scalar advantage increases or decreases the probability of each sampled trajectory independently. This can help, but it does not explicitly redistribute probability mass among competing trajectories for the same prompt. Target Policy Optimization (TPO) gives us a more direct mechanism. For each prompt, we sample a group of `K` trajectories and define a target distribution over the group:

```text
q_i proportional to stopgrad(pi_old(y_i | x)) * exp(U_i / tau).
```

The old-policy probability keeps the target close to the sampled policy support, while the exponentiated utility shifts mass toward trajectories that are correct and have useful evidence. The model is then trained to match this target distribution over the sampled completions. In sequence form, if `s_i` is the current model sequence log-score for completion `y_i`, the group probability is

```text
p_i = softmax_i(s_i),
```

and the TPO loss is the cross-entropy from the target distribution to the current grouped policy:

```text
L_TPO = - sum_i q_i log p_i.
```

This differs from simply adding PrefixIG as reward shaping. The target distribution makes the comparison within each prompt explicit: a useful correct search trajectory can receive more target mass than a redundant correct trajectory, even when both have the same final answer reward.

During early generated-policy experiments, we found that PrefixIG alone can still overvalue redundant correct trajectories. A redundant trajectory can accumulate positive evidence gain if it repeats or rephrases helpful evidence, even though it uses more search turns than necessary. To address this, we add a reward-gated efficiency penalty. Let `C_i` denote a redundancy cost based on extra turns beyond the optimal number, repeated queries/results, and low-information-gain turns. We define:

```text
U_i = R_i
    + lambda_ig * z(IG_i)
    - R_i * lambda_eff * C_i.
```

The reward gate is important. Without it, a short but wrong trajectory could be rewarded for being efficient. By multiplying the efficiency penalty by `R_i`, we penalize waste primarily among correct trajectories. This encourages the model to prefer trajectories that are both correct and evidence-efficient.

We also implemented a curvature-aware diagnostic variant. The goal is not to make curvature the main contribution, but to test whether cheap second-order proxies can stabilize or sharpen target construction. For a grouped softmax over completions, a simple local curvature proxy is:

```text
curv_i = p_i * (1 - p_i).
```

We use this as a trust or scaling signal in diagnostic target construction:

```text
U_i_curv = U_i / sqrt(curv_i + eps).
```

This is not a full Hessian. It is a lightweight group-curvature approximation that is feasible on a laptop and can later be replaced by richer Hessian or variance-based trust measures.

Our experimental methodology therefore separates three ideas that are often conflated:

1. Evidence credit: PrefixIG estimates which search turns improved the answer likelihood.
2. Target construction: TPO converts evidence-aware utilities into a group distribution.
3. Efficiency control: reward-gated redundancy penalties distinguish useful search from unnecessary search.

We compare these components against final-reward-only TPO, PrefixIG-GRPO/A-TGPO-style scalar advantage updates, A-TGPO component accounting with token-level log probabilities and clipping, PrefixIG-TPO, and PrefixIG-TPO with reward-gated efficiency.

## Experiments

We designed the experiments as a ladder, from deterministic diagnostics to trainable model updates. This was necessary because the original A-TGPO codebase is built around CUDA, veRL, vLLM, and Ray, while our development machine is an Apple Silicon laptop. The repo therefore uses the original A-TGPO implementation as a conceptual reference for turn segmentation, token-level log-probability accounting, clipping, and KL/trust-region structure, while using MLX and `mlx-lm-lora` for Mac-feasible experiments.

### Synthetic Target-Construction Diagnostics

The first experiment uses controlled synthetic multi-hop QA cases. Each prompt has candidate trajectories corresponding to useful correct search, redundant correct search, no-search correct answering, and distractor/wrong search. Because the candidates are controlled, this experiment isolates whether the target construction assigns probability mass to the intended trajectory type.

The main metric is `useful-red`, defined as target mass on useful correct trajectories minus target mass on redundant correct trajectories. Positive values mean the method prefers concise useful evidence over redundant evidence. In the early synthetic comparison, final-reward-only TPO barely separated useful from redundant correct trajectories, while PrefixIG-based methods created a much larger gap:

```text
method                useful-red
OriginalPolicy        +0.002
FinalReward-TPO       +0.004
PrefixIG-GRPO         +0.093
A-TGPO-components     +0.108
PrefixIG-TPO          +0.112
PrefixIG-TPO+curv     +0.288
```

This diagnostic supports the central claim that final reward alone cannot distinguish useful and redundant correct trajectories. PrefixIG helps, and using it inside the TPO target distribution is more direct than using it only as scalar advantage shaping.

### Real MLX Policy Scoring and Generated Rollouts

We next moved from hand-written candidates to trajectories generated and scored by a real MLX language model. We used Qwen2.5-0.5B-Instruct through MLX to generate search trajectories under an oracle-search loop: the model emits `<search>` actions, a lexical search function returns `<result>` facts, and the model eventually emits `<answer>`. We then score completions with the model's own log probabilities and compute PrefixIG from answer likelihood changes across prefixes.

This setting exposed an important failure mode. Flexible generation sometimes produced correct but redundant search behavior. Raw PrefixIG can assign nontrivial value to these trajectories because repeated useful evidence can still increase answer likelihood. In generated-policy diagnostics, reward-only methods improved correctness but did not reliably prefer efficient search. PrefixIG-GRPO and PrefixIG-TPO improved the useful-vs-redundant gap, while reward-gated efficiency further reduced redundant target mass.

A representative generated-policy comparison showed the following useful-vs-redundant target-mass gaps:

```text
method                   useful-red
OriginalPolicy           -0.015
FinalReward-TPO          -0.014
PrefixIG-GRPO            +0.133
A-TGPO-components        -0.008
PrefixIG-TPO             +0.166
PrefixIG-TPO+rg_eff      +0.273
PrefixIG-TPO+curv        +0.337
PrefixIG-TPO+curv+rg_eff +0.406
```

These are target-construction diagnostics, not yet model-training results. Their purpose is to validate that the offline objective points the update in the desired direction before spending memory and time on LoRA fine-tuning.

### Trainable Toy RLVR

The next experiment tests whether the target-construction advantage survives actual training. We implemented a small MLX transformer RLVR task where the model chooses discrete search/action tokens and then emits an answer token. The task is deliberately small enough to run with multiple seeds on a laptop, while preserving the core ambiguity of the real problem: a final answer can be correct with useful search, redundant search, or poor evidence behavior.

We compare four training objectives:

- `reward_tpo`: final-reward-only target policy optimization.
- `prefixig_atgpo`: an A-TGPO-style scalar PrefixIG advantage path with clipping/trust accounting.
- `prefixig_tpo`: PrefixIG utility used to construct TPO targets.
- `prefixig_tpo_rg_eff`: PrefixIG-TPO with reward-gated efficiency.

Across five seeds and 200 episodes, the final metrics were:

```text
method              reward  useful  redundant  distractor  useful-red
reward_tpo          0.921   0.088   0.091      0.007       -0.004
prefixig_atgpo      0.799   0.694   0.105      0.000       +0.590
prefixig_tpo        0.844   0.826   0.013      0.009       +0.813
prefixig_tpo_rg_eff 0.847   0.838   0.004      0.011       +0.834
```

The reward-only baseline achieved the highest final-answer reward, but it did not learn evidence-efficient behavior: useful and redundant correct behavior were nearly tied. The A-TGPO-style PrefixIG scalar objective strongly improved useful search behavior, showing that intermediate evidence credit matters. PrefixIG-TPO performed better still, producing high useful-search rates with very low redundant-search rates. Reward-gated efficiency gave the best useful-vs-redundant gap, reducing redundant correct behavior from `0.013` under PrefixIG-TPO to `0.004`.

This result is important because it separates answer accuracy from search quality. A method can be accurate while still using poor or unnecessary evidence. Our method improves the process metric without collapsing reward, which is the behavior we want for retrieval-augmented reasoning systems.

### A-TGPO Component Ablation

To connect more directly to the original A-TGPO methodology, we also ran a small component ablation in the toy RLVR environment. The original A-TGPO design combines three ideas: turn-group normalization of information gain, variance-rescaled discounted accumulation, and adaptive turn-level clipping. We implemented small analogues of these components in the toy trainer and compared them against reward-only TPO, raw PrefixIG-GRPO, PrefixIG-TPO, and PrefixIG-TPO with reward-gated efficiency.

Across five seeds and 200 episodes, the final metrics were:

```text
method                     reward  useful  redundant  distractor  useful-red
reward_tpo                 0.921   0.088   0.091      0.007       -0.004
prefixig_grpo              0.876   0.876   0.000      0.000       +0.876
prefixig_grpo_turn_norm    0.596   0.000   0.596      0.001       -0.596
prefixig_grpo_turn_norm_vr 0.595   0.000   0.595      0.001       -0.595
prefixig_atgpo             0.797   0.000   0.797      0.001       -0.797
prefixig_tpo               0.844   0.826   0.013      0.009       +0.813
prefixig_tpo_rg_eff        0.846   0.838   0.004      0.011       +0.833
```

This ablation should be interpreted carefully. It does not show that A-TGPO fails in general; it shows that, in this simplified toy environment, scalar turn-level credit is sensitive to how normalization, accumulation, and clipping consume the information-gain signal. The naive turn-normalized variants and our simplified A-TGPO-style token objective over-concentrate on redundant-correct trajectories. A likely reason is that terminal reward is broadcast across action tokens, and in this toy environment redundant evidence remains highly correlated with correctness. This can cause token-level scalar updates to reinforce redundant actions even when the trajectory-level evidence objective would prefer concise useful search.

The target-policy methods are more stable in this ablation. PrefixIG-TPO keeps high useful-correct behavior with low redundant-correct behavior, and reward-gated efficiency further reduces redundancy. This supports the broader methodological point: rather than only shaping token-level scalar advantages, constructing an explicit target distribution over competing trajectories gives a more direct way to move probability mass from redundant or distractor trajectories toward useful evidence trajectories.

### Noisy Retriever Robustness

The clean toy settings above are useful for controlled credit assignment, but real retrieval-augmented reasoning often contains stale or distracting evidence. We therefore added a noisy-retriever variant to the toy RLVR environment. In this setting, some evidence actions that would otherwise be useful are deterministically corrupted into either distractor evidence or stale/redundant evidence. This lets us test whether each objective learns evidence-efficient behavior when retrieval quality is imperfect.

We ran mixed-hop tasks under two corruption levels. `mixed25` corrupts a moderate fraction of evidence actions, while `mixed50` creates a stronger noisy-retrieval condition. Each result is averaged over five seeds and 200 episodes.

Under moderate retrieval noise, PrefixIG-TPO and PrefixIG-TPO with reward-gated efficiency outperform scalar PrefixIG-GRPO on the useful-minus-redundant process metric:

```text
mixed25
method              reward  useful  redundant  distractor  useful-red
reward_tpo          0.911   0.040   0.018      0.019       +0.022
prefixig_grpo       0.994   0.576   0.224      0.000       +0.352
prefixig_tpo        0.804   0.544   0.028      0.041       +0.517
prefixig_tpo_rg_eff 0.756   0.536   0.007      0.061       +0.529
```

Under stronger retrieval noise, the difference becomes more pronounced. PrefixIG-GRPO maintains high answer reward, but it collapses toward redundant-correct behavior: redundant correct trajectories rise to `0.602`, and useful-minus-redundant becomes negative. PrefixIG-TPO remains positive, and reward-gated efficiency gives the best useful-minus-redundant gap while sharply suppressing redundancy:

```text
mixed50
method              reward  useful  redundant  distractor  useful-red
reward_tpo          0.911   0.025   0.009      0.023       +0.016
prefixig_grpo       0.994   0.097   0.602      0.002       -0.505
prefixig_tpo        0.792   0.210   0.148      0.073       +0.061
prefixig_tpo_rg_eff 0.759   0.268   0.015      0.084       +0.253
```

This is the strongest controlled evidence for our target-policy construction so far. In clean settings, scalar PrefixIG-GRPO can be very competitive because the information-gain signal is nearly unambiguous. Under noisy retrieval, however, scalar updates can still reinforce trajectories that get the answer right while relying on stale or redundant evidence. PrefixIG-TPO is more robust because it constructs a group-level target distribution over competing trajectories, and reward-gated efficiency explicitly shifts mass away from redundant correct search. This experiment supports the hypothesis that target construction matters most when evidence quality is imperfect, which is the realistic regime for retrieval-augmented reasoning.

### Offline Qwen2.5-0.5B LoRA Pilot

The next step is a real language-model update. We implemented an offline target builder that generates Qwen2.5-0.5B trajectories, computes PrefixIG-TPO plus reward-gated efficiency target weights, writes a full candidate JSONL with `target_weight`, and creates an SFT-style dataset by sampling completions according to the `rg_eff` target distribution. This is a practical first Mac experiment because it avoids online generation during training and uses the existing MLX LoRA trainer.

The current pilot uses Qwen2.5-0.5B-Instruct with 4-bit loading, LoRA adapters, masked-prompt SFT, short sequence length, and gradient checkpointing. This is not yet a true offline TPO trainer; it is a distillation step from the target distribution into an adapter. The full candidate JSONL preserves grouped target weights so that the next implementation can train directly with the offline TPO cross-entropy objective.

In the first pilot run, the LoRA adapter trained successfully on a 16GB Apple Silicon laptop. The run used rank-8 LoRA over 8 layers, with 1.494M trainable parameters out of 494.033M total parameters, or 0.302% of the model. Training ran for 80 iterations and saved the final adapter to `outputs/adapters/offline_rg_eff_qwen_lora/adapters.safetensors`. The final training loss was `0.091`. On the held-out SFT-style offline target set, the adapter reached test loss `0.281` and perplexity `1.325`. These SFT metrics are not the main paper claim, but they verify that the rg_eff-distilled target distribution is learnable by a small Qwen2.5-0.5B LoRA adapter under the Mac-feasible setup.

The intended evaluation is before/after generated-policy diagnostics on the same search QA distribution:

```text
exact-match answer accuracy
useful correct search rate
redundant correct search rate
distractor/wrong search rate
average search turns
useful-red gap
correct answer per search turn
```

We then ran memory-light generated-policy behavior diagnostics. The key evaluation row is `OriginalPolicy`, which reflects the actual generated behavior of the loaded model before any target reweighting. The other diagnostic rows in the script ask how different objectives would reweight the same generated batch; they are useful for debugging target construction, but they are not trained-model comparisons.

The first small pilot compared base Qwen and an rg_eff-distilled LoRA over 8 generated trajectories. The base model produced mostly distractor/wrong search trajectories, while the rg_eff-LoRA adapter shifted strongly toward useful correct search. This established that the offline target distribution can transfer into generated behavior, but the sample size was too small and did not elicit redundant correct trajectories reliably.

We therefore ran a cleaner held-out comparison using 5 examples, 2 samples per example, seed 101, flexible oracle-search rollouts, at most 2 search turns, and shorter action/answer budgets. We compared five loaded model conditions: base Qwen, reward-only TPO distillation, PrefixIG-TPO distillation, PrefixIG-TPO with reward-gated efficiency, and an A-TGPO component proxy distillation baseline. The table reports only the `OriginalPolicy` row for each loaded model:

```text
model                    useful  redundant  no_search  distractor  other  correct  useful-red
Base Qwen                0.174   0.000      0.000      0.826       0.000  0.174    +0.174
reward_tpo LoRA          0.600   0.400      0.000      0.000       0.000  1.000    +0.200
prefixig_tpo LoRA        0.914   0.000      0.000      0.086       0.000  0.914    +0.914
prefixig_tpo_rg_eff LoRA 0.391   0.307      0.000      0.302       0.000  0.698    +0.084
atgpo_proxy LoRA         0.801   0.000      0.000      0.199       0.000  0.801    +0.801
```

This held-out pilot gives a more informative pattern. Reward-only TPO distillation reaches perfect correctness on this small sample, but 40% of its probability mass is redundant correct search, giving a weak useful-minus-redundant gap. This is the expected failure mode of final-answer reward: it teaches the model to get the answer right, but it does not distinguish concise useful evidence from extra search.

PrefixIG-TPO gives the strongest behavior in this comparison: high correctness, almost all correct mass assigned to useful search, no redundant correct mass, and the largest useful-minus-redundant gap. The A-TGPO proxy is also strong and is an important baseline, but it trails PrefixIG-TPO on both correctness and useful evidence mass in this held-out seed. The reward-gated efficiency variant underperforms here, suggesting that the current efficiency penalty is not yet robust in the real-model LoRA setting. We therefore treat `rg_eff` as an ablation rather than the main method for the current paper story.

We also ran a noisy/distractor stress test with the same 5-example, 2-sample, seed-101 setup. In this regime, the search index places a wrong birthplace result before the correct one, so the model must avoid following the first plausible retrieved fact. This is intentionally harsh for a small model and should be interpreted as a robustness diagnostic rather than the main performance table:

```text
model                    useful  redundant  no_search  distractor  other  correct  useful-red
Base Qwen                0.000   0.000      0.000      1.000       0.000  0.000    +0.000
reward_tpo LoRA          0.000   0.000      0.000      1.000       0.000  0.000    +0.000
prefixig_tpo LoRA        0.111   0.000      0.000      0.889       0.000  0.111    +0.111
prefixig_tpo_rg_eff LoRA 0.000   0.000      0.000      1.000       0.000  0.000    +0.000
atgpo_proxy LoRA         0.090   0.000      0.000      0.910       0.000  0.090    +0.090
```

The noisy result shows that all current small LoRA models mostly follow the distractor evidence. However, PrefixIG-TPO is the only method that clearly improves over base and reward-only in this stress test, and it slightly exceeds the A-TGPO proxy. This supports keeping noisy retrieval as a robustness benchmark, but it also shows that the present offline SFT distillation setup is not yet sufficient for strong distractor resistance.

Overall, the experiments so far support a clear story. Final reward teaches correctness but not evidence efficiency. PrefixIG provides useful intermediate credit. TPO target construction uses that signal directly by redistributing probability mass among competing trajectories. A-TGPO-style token credit is a strong baseline, and the current evidence does not support claiming that we beat it in all settings. Instead, the strongest current claim is that PrefixIG-TPO is a simple, Mac-feasible target-policy method that improves generated evidence behavior over reward-only training and is competitive with an A-TGPO proxy baseline. The next empirical milestone is to repeat the Qwen LoRA evaluation across more seeds and then replace SFT distillation with a true offline TPO trainer over grouped trajectory targets.

### Mixed-Hop Qwen LoRA Follow-Up

The first Qwen LoRA results suggested a useful diagnosis. PrefixIG-TPO performed very well on the held-out multi-hop regime, but its adapter had been trained mostly on multi-hop-style trajectories. On single-hop prompts, this produced an over-search failure: the model remained correct, but it shifted probability toward redundant correct trajectories. To test whether this was a data-regime issue, we constructed a mixed-hop offline training set that alternates single-hop and multi-hop cases during target construction.

We trained two additional Qwen2.5-0.5B LoRA adapters with the same 4-bit, rank-8, 80-iteration MLX setup: `prefixig_tpo_mixedhop` and `atgpo_proxy_mixedhop`. The mixed-hop target builder used 16 prompts, 4 generated candidates per prompt, and 8 SFT draws per prompt, yielding 128 SFT records per target method. The average target mass was concentrated mostly on useful correct trajectories: `0.760` useful correct for PrefixIG-TPO mixed-hop and `0.790` useful correct for the A-TGPO proxy mixed-hop. The PrefixIG-TPO mixed-hop adapter reached validation loss `0.149`, while the A-TGPO proxy mixed-hop adapter reached validation loss `0.364`.

We then evaluated the earlier five model conditions plus the two mixed-hop adapters on held-out single-hop, multi-hop, and noisy/distractor regimes. The table reports only the `OriginalPolicy` row, i.e. actual generated behavior from the loaded model.

```text
Single-hop, 5 examples x 2 samples, seed 101
model                      useful  redundant  distractor  correct  useful-red
Base Qwen                  0.724   0.200      0.076       0.924    +0.524
reward_tpo LoRA            0.800   0.200      0.000       1.000    +0.600
prefixig_tpo LoRA          0.382   0.618      0.000       1.000    -0.236
prefixig_tpo_rg_eff LoRA   0.102   0.898      0.000       1.000    -0.796
atgpo_proxy LoRA           0.902   0.098      0.000       1.000    +0.804
prefixig_tpo_mixedhop      1.000   0.000      0.000       1.000    +1.000
atgpo_proxy_mixedhop       0.215   0.607      0.178       0.822    -0.392

Multi-hop, 5 examples x 2 samples, seed 101
model                      useful  redundant  distractor  correct  useful-red
Base Qwen                  0.174   0.000      0.826       0.174    +0.174
reward_tpo LoRA            0.600   0.400      0.000       1.000    +0.200
prefixig_tpo LoRA          0.914   0.000      0.086       0.914    +0.914
prefixig_tpo_rg_eff LoRA   0.391   0.307      0.302       0.698    +0.084
atgpo_proxy LoRA           0.801   0.000      0.199       0.801    +0.801
prefixig_tpo_mixedhop      0.305   0.000      0.695       0.305    +0.305
atgpo_proxy_mixedhop       0.910   0.000      0.090       0.910    +0.910

Noisy/distractor, 5 examples x 2 samples, seed 101
model                      useful  redundant  distractor  correct  useful-red
Base Qwen                  0.000   0.000      1.000       0.000    +0.000
reward_tpo LoRA            0.000   0.000      1.000       0.000    +0.000
prefixig_tpo LoRA          0.111   0.000      0.889       0.111    +0.111
prefixig_tpo_rg_eff LoRA   0.000   0.000      1.000       0.000    +0.000
atgpo_proxy LoRA           0.090   0.000      0.910       0.090    +0.090
prefixig_tpo_mixedhop      0.000   0.000      1.000       0.000    +0.000
atgpo_proxy_mixedhop       0.000   0.105      0.895       0.105    -0.105
```

The mixed-hop experiment is encouraging, but in a specific way. PrefixIG-TPO mixed-hop fixes the single-hop over-search failure completely in this small evaluation: useful-minus-redundant moves from `-0.236` to `+1.000` while correctness remains perfect. This supports the idea that PrefixIG-TPO can learn concise useful search when the training target distribution includes the relevant regime. However, the same mixed-hop adapter loses the strong multi-hop behavior of the original PrefixIG-TPO adapter. The A-TGPO proxy mixed-hop adapter shows the opposite tradeoff: it is strong on multi-hop but poor on single-hop.

The noisy/distractor result remains a negative stress test for all current small LoRA adapters. Mixing single-hop and multi-hop trajectories is not enough to teach distractor resistance. For the next real-model experiment, we should build a larger balanced target set with an explicit noisy-retrieval split, then train one PrefixIG-TPO adapter and one A-TGPO proxy adapter under the same data budget. This would directly test whether the target-policy construction can learn to reject misleading evidence rather than merely adapting to hop count.

### Mixed-Hop + Noisy Qwen LoRA

We then ran the direct follow-up: a mixed-hop + noisy training regime that cycles single-hop, multi-hop, and noisy/distractor cases. This required one correction to the noisy environment. The previous noisy retriever returned only the first high-ranking wrong birthplace fact, which made many cases closer to impossible retrieval than distractor robustness. We changed the retriever so tied birthplace matches return misleading and corrective evidence together, preserving the intended challenge: the model sees a wrong fact before a correct fact and must still answer from the useful evidence.

The first target build still had too much distractor mass, because the base model rarely sampled useful noisy trajectories. We therefore added useful-correct oracle anchors to each candidate group during offline target construction. With 18 prompts, 4 generated samples per prompt, 2 oracle anchors per prompt, and 8 SFT draws per prompt, the resulting target distributions were strongly useful-correct:

```text
target method                  useful  redundant  distractor
prefixig_tpo_mixedhop_noisy    0.804   0.083      0.113
atgpo_proxy_mixedhop_noisy     0.776   0.096      0.128
```

Both adapters trained successfully on the same Mac-feasible Qwen2.5-0.5B 4-bit LoRA setup. PrefixIG-TPO mixed-hop+noisy reached final training loss `0.038` and validation loss `0.247`. A-TGPO proxy mixed-hop+noisy reached final training loss `0.089` and validation loss `0.081`.

The held-out generated-policy results are the strongest real-model results so far:

```text
Single-hop, 5 examples x 2 samples, seed 101
model                            useful  redundant  distractor  correct  useful-red
Base Qwen                        0.724   0.200      0.076       0.924    +0.524
reward_tpo LoRA                  0.800   0.200      0.000       1.000    +0.600
prefixig_tpo LoRA                0.382   0.618      0.000       1.000    -0.236
atgpo_proxy LoRA                 0.902   0.098      0.000       1.000    +0.804
prefixig_tpo_mixedhop            1.000   0.000      0.000       1.000    +1.000
prefixig_tpo_mixedhop_noisy      0.800   0.200      0.000       1.000    +0.600
atgpo_proxy_mixedhop_noisy       0.526   0.000      0.474       0.526    +0.526

Multi-hop, 5 examples x 2 samples, seed 101
model                            useful  redundant  distractor  correct  useful-red
Base Qwen                        0.174   0.000      0.826       0.174    +0.174
reward_tpo LoRA                  0.600   0.400      0.000       1.000    +0.200
prefixig_tpo LoRA                0.914   0.000      0.086       0.914    +0.914
atgpo_proxy LoRA                 0.801   0.000      0.199       0.801    +0.801
prefixig_tpo_mixedhop            0.305   0.000      0.695       0.305    +0.305
prefixig_tpo_mixedhop_noisy      1.000   0.000      0.000       1.000    +1.000
atgpo_proxy_mixedhop_noisy       0.800   0.000      0.200       0.800    +0.800

Corrected noisy/distractor, 5 examples x 2 samples, seed 101
model                            useful  redundant  distractor  correct  useful-red
Base Qwen                        0.093   0.000      0.907       0.093    +0.093
reward_tpo LoRA                  0.200   0.400      0.400       0.600    -0.200
prefixig_tpo LoRA                0.505   0.200      0.295       0.705    +0.305
atgpo_proxy LoRA                 0.400   0.196      0.404       0.596    +0.204
prefixig_tpo_mixedhop_noisy      0.800   0.000      0.200       0.800    +0.800
atgpo_proxy_mixedhop_noisy       0.800   0.000      0.200       0.800    +0.800
```

This result changes the paper story in a good way. The earlier PrefixIG-TPO adapter was strong on multi-hop but brittle across regimes. The mixed-hop+noisy PrefixIG-TPO adapter is balanced: it preserves perfect correctness on single-hop and multi-hop while substantially improving noisy retrieval. It also avoids the reward-only failure mode where noisy correctness comes with redundant search. The A-TGPO proxy matches PrefixIG-TPO on the corrected noisy split, but is weaker on single-hop and multi-hop in this seed. Because this is still a 5-prompt-per-regime pilot, the right next step is multi-seed evaluation and a larger held-out set, not a stronger claim yet.
