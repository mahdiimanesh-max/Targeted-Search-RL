# Methodology and Experiments Draft

## Paper Thesis

We study reinforcement learning for retrieval-augmented reasoning systems that must decide both **what answer to give** and **which intermediate search actions are worth taking**. Final-answer reward alone is too coarse for this setting. A trajectory that uses exactly the necessary evidence and a trajectory that performs redundant or distracting searches can receive the same terminal reward, even though their reasoning processes differ substantially.

Our central claim is:

```text
Prefix-level information gain can be used to construct target policies that move probability mass toward useful evidence-seeking trajectories, not merely correct final answers.
```

The paper should be framed as a method and mechanism paper, with small-model real-world pilots. The strongest evidence is that PrefixIG-TPO consistently improves **evidence-use behavior**; answer-accuracy gains are promising but not yet robust enough to oversell.

## Method

### Prefix Information Gain

For a prompt `x`, generated trajectory `y_i`, and ground-truth answer `a*`, each search/result turn can be scored by how much it changes the model's likelihood of the correct answer:

```text
IG_{i,t} = log p(a* | x, prefix through turn t)
         - log p(a* | x, prefix before turn t).
```

Positive information gain means the newly retrieved evidence made the correct answer more likely. Low or negative gain suggests redundant, distracting, or unhelpful evidence. We aggregate turn-level gains into a trajectory-level PrefixIG score and combine it with terminal reward:

```text
U_i = R_i + lambda_ig * z(IG_i),
```

where `R_i` is final-answer reward and `z(IG_i)` is a group-normalized PrefixIG score.

### PrefixIG-TPO

Instead of using PrefixIG only as a scalar advantage, we use it to construct a target distribution over candidate trajectories for the same prompt. For each prompt, we sample a group of `K` trajectories and define:

```text
q_i proportional to stopgrad(pi_old(y_i | x)) * exp(U_i / tau).
```

The old-policy factor keeps the target near the sampled policy support, while the exponentiated utility shifts probability mass toward trajectories that are both correct and evidence-useful. If `s_i` is the current model sequence log-score for completion `y_i`, then

```text
p_i = softmax_i(s_i)
L_TPO = - sum_i q_i log p_i.
```

This makes the within-prompt comparison explicit: a useful correct trajectory can receive more target mass than a redundant correct trajectory even when both have the same terminal answer reward.

### Online Anchor Variant

The most successful trainable variant uses online rollout refresh plus a token/action anchor:

```text
loss = grouped PrefixIG-TPO loss + beta * weighted token/action NLL.
```

The grouped TPO term adapts the policy toward the PrefixIG target distribution. The token/action anchor prevents the model from drifting away from the search/action language while training on very small online batches.

### Baselines

The main comparisons should be:

- `reward_tpo`: final-reward-only target policy optimization.
- `A-TGPO proxy`: our Mac-feasible proxy for A-TGPO-style turn/token credit, clipping, and trust accounting. This should not be described as a full reproduction of the original A-TGPO system.
- `PrefixIG-TPO`: our main target-policy method.
- `online PrefixIG-TPO+anchor`: online rollout refresh with token/action anchoring.

Secondary variants such as `rg_eff`, `curv`, true offline TPO, and inference-only target reweighting belong in the appendix unless needed for a specific ablation.

## Main Experimental Story

The experiments are organized as a ladder:

1. **Controlled mechanism:** show that final reward does not distinguish useful and redundant search.
2. **Trainable toy RLVR:** show that the target-policy idea survives actual optimization.
3. **Small-model Qwen LoRA:** show that a real model can absorb the target behavior.
4. **Online training:** show that PrefixIG-TPO can work without weighted-SFT as the main objective.
5. **HotpotQA-mini pilot:** test the method in real retrieval QA under an SDAR search-skill prior.

## Main Result 1: Trainable Toy RLVR

We implemented a small MLX transformer RLVR task where the model chooses discrete search/action tokens and then emits an answer token. The task is small enough to run across seeds on a laptop, but preserves the key ambiguity: final answers can be correct with useful search, redundant search, or poor evidence behavior.

Across five seeds and 200 episodes:

```text
method              reward  useful  redundant  distractor  useful-red
reward_tpo          0.921   0.088   0.091      0.007       -0.004
prefixig_atgpo      0.799   0.694   0.105      0.000       +0.590
prefixig_tpo        0.844   0.826   0.013      0.009       +0.813
prefixig_tpo_rg_eff 0.847   0.838   0.004      0.011       +0.834
```

Interpretation: reward-only optimization achieves high answer reward but does not learn evidence-efficient behavior. PrefixIG-based objectives learn useful search. PrefixIG-TPO gives the strongest main-method result, substantially increasing useful behavior while keeping redundant search low.

This is the cleanest mechanism result in the paper.

## Main Result 2: Qwen2.5-0.5B LoRA Generated Behavior

We trained Qwen2.5-0.5B LoRA adapters from offline target distributions and evaluated only the `OriginalPolicy` row from each generated-policy diagnostic table. This row measures actual generated behavior from the loaded model, not post-hoc reweighting.

### Held-Out Multi-Hop Pilot

```text
model                    useful  redundant  no_search  distractor  other  correct  useful-red
Base Qwen                0.174   0.000      0.000      0.826       0.000  0.174    +0.174
reward_tpo LoRA          0.600   0.400      0.000      0.000       0.000  1.000    +0.200
prefixig_tpo LoRA        0.914   0.000      0.000      0.086       0.000  0.914    +0.914
prefixig_tpo_rg_eff LoRA 0.391   0.307      0.000      0.302       0.000  0.698    +0.084
atgpo_proxy LoRA         0.801   0.000      0.000      0.199       0.000  0.801    +0.801
```

Interpretation: reward-only TPO reaches correctness but allocates substantial mass to redundant correct search. PrefixIG-TPO gives the strongest evidence behavior, with high correctness and no redundant correct mass. The A-TGPO proxy is a strong baseline but trails PrefixIG-TPO on this seed.

### Mixed-Hop + Noisy Training

We then trained PrefixIG-TPO and A-TGPO proxy adapters on a mixed-hop+noisy target set with useful-correct anchors. This is the most balanced small-model offline result.

```text
Evaluation regime          method                         correct  useful  redundant  distractor  useful-red
single-hop                 PrefixIG-TPO mixedhop+noisy    1.000    0.800   0.200      0.000       +0.600
single-hop                 A-TGPO proxy mixedhop+noisy    0.526    0.526   0.000      0.474       +0.526

multi-hop                  PrefixIG-TPO mixedhop+noisy    1.000    1.000   0.000      0.000       +1.000
multi-hop                  A-TGPO proxy mixedhop+noisy    0.800    0.800   0.000      0.200       +0.800

corrected noisy/distractor PrefixIG-TPO mixedhop+noisy    0.800    0.800   0.000      0.200       +0.800
corrected noisy/distractor A-TGPO proxy mixedhop+noisy    0.800    0.800   0.000      0.200       +0.800
```

Interpretation: PrefixIG-TPO is balanced across single-hop, multi-hop, and corrected noisy retrieval. A-TGPO proxy matches on corrected noisy retrieval but is weaker on single-hop and multi-hop in this seed.

## Main Result 3: Online PrefixIG-TPO With Token/Action Anchor

To test whether PrefixIG-TPO can work without weighted-SFT as the main training objective, we implemented online rollout refresh:

```text
q(y | x) proportional to pi_old(y | x) exp(U(y) / tau).
```

The model samples fresh trajectories, builds grouped PrefixIG-TPO targets, and updates a LoRA adapter with grouped TPO loss plus token/action anchoring. The best anchor setting was `beta=0.10`.

```text
anchor beta  regime      correct  useful  redundant  distractor  useful-red
0.02         single-hop  0.896    0.696   0.200      0.104       +0.496
0.02         multi-hop   0.402    0.402   0.000      0.598       +0.402
0.02         noisy       0.702    0.702   0.000      0.298       +0.702

0.10         single-hop  1.000    0.800   0.200      0.000       +0.600
0.10         multi-hop   1.000    1.000   0.000      0.000       +1.000
0.10         noisy       1.000    1.000   0.000      0.000       +1.000
```

We repeated the `beta=0.10` setting in a fresh directory and reproduced the same held-out behavior:

```text
run                          regime      correct  useful  redundant  distractor  useful-red
online PrefixIG-TPO beta=.10 single-hop  1.000    0.800   0.200      0.000       +0.600
online PrefixIG-TPO beta=.10 multi-hop   1.000    1.000   0.000      0.000       +1.000
online PrefixIG-TPO beta=.10 noisy       1.000    1.000   0.000      0.000       +1.000
```

Interpretation: online PrefixIG-TPO is viable when the token/action anchor is strong enough. This result supports the online direction and distinguishes the method from pure weighted-SFT distillation.

## Main Result 4: HotpotQA-Mini Real-World Pilot

We built `HotpotQA-mini` from the `hotpot_qa/distractor` split: 80 training questions, 40 evaluation questions, and a local corpus of 951 deduplicated passages. Each example preserves the question, short gold answer, supporting document titles, support passages, and distractor passages. The evaluator uses a local BM25-style retriever and lets the model interact through `<search>`, `<result>`, and `<answer>` actions.

Early unrestricted HotpotQA-mini runs showed that synthetic success did not transfer automatically. Qwen2.5-0.5B often retrieved evidence but failed answer synthesis. We therefore used Qwen2.5-1.5B and added SDAR search-skill prompts as a Mac-feasible search prior. We did not run the full SDAR CUDA/vLLM/Ray training stack; we only injected SDAR search skills into the prompt.

### SDAR Prompting Improves the Candidate Pool

```text
HotpotQA-mini, Qwen2.5-1.5B-Instruct 4-bit, 8 examples x 2 samples, seed 101
condition          correct  useful  redundant  no_search  distractor  other  useful-red  exact  token_f1  support_cov
no skills          0.125    0.062   0.062      0.000      0.812       0.062  +0.000      0.000  0.051     0.531
SDAR search skills 0.438    0.312   0.062      0.062      0.500       0.062  +0.250      0.188  0.279     0.656
```

Interpretation: SDAR-style search skills improve the candidate pool and make HotpotQA-mini feasible enough to test our training objective.

### SDAR + Method Comparison

All conditions below use Qwen2.5-1.5B 4-bit and the same SDAR skill prompts. The base condition is inference only. The trained conditions use the same SDAR prompts during online rollout generation and evaluation, then update separate LoRA adapters with either PrefixIG-TPO or A-TGPO proxy targets.

```text
HotpotQA-mini + SDAR skills, Qwen2.5-1.5B-Instruct 4-bit, 8 examples x 2 samples, seed 101
condition             correct  useful  redundant  no_search  distractor  other  useful-red  exact  token_f1  support_cov
Base + SDAR           0.250    0.188   0.062      0.000      0.562       0.188  +0.125      0.000  0.155     0.531
PrefixIG-TPO + SDAR   0.500    0.375   0.125      0.000      0.500       0.000  +0.250      0.188  0.357     0.750
A-TGPO proxy + SDAR   0.375    0.250   0.125      0.000      0.500       0.125  +0.125      0.250  0.344     0.531
```

Seed 101 is a positive pilot: PrefixIG-TPO improves correctness, useful evidence use, token F1, and support coverage over the SDAR-only base, and outperforms the A-TGPO proxy on the main process metrics.

We then ran a scale-up/replication with 12 held-out examples, 2 samples per example, seed 202, and 8 online iterations:

```text
HotpotQA-mini + SDAR skills, Qwen2.5-1.5B-Instruct 4-bit, 12 examples x 2 samples, seed 202
condition             correct  useful  redundant  no_search  distractor  other  useful-red  exact  token_f1  support_cov
Base + SDAR           0.208    0.000   0.208      0.000      0.750       0.042  -0.208      0.208  0.238     0.438
PrefixIG-TPO + SDAR   0.208    0.167   0.042      0.000      0.750       0.042  +0.125      0.167  0.223     0.500
A-TGPO proxy + SDAR   0.000    0.000   0.000      0.000      0.958       0.042  +0.000      0.000  0.021     0.438
```

The replication is mixed. PrefixIG-TPO does not improve correctness over Base+SDAR on seed 202. However, it shifts correct behavior from redundant evidence to useful evidence and improves support coverage. A-TGPO proxy is unstable in this run, collapsing to mostly distractor-wrong trajectories.

### Real-World Claim

The current real-world claim should be cautious:

```text
Under the same SDAR search prior, PrefixIG-TPO shows promising evidence-use improvements and better stability than the A-TGPO proxy in small HotpotQA-mini pilots. Robust answer-accuracy gains require larger seeds and data.
```

This is enough to motivate the method, but not enough to claim a solved real-world QA benchmark.

## Recommended Main Paper Structure

1. Introduction: final reward misses evidence quality.
2. Method: PrefixIG and target-policy construction.
3. Controlled experiments: toy RLVR and Qwen LoRA.
4. Online training: PrefixIG-TPO + action anchor.
5. Real-world pilot: HotpotQA-mini with SDAR skills.
6. Limitations: small samples, A-TGPO proxy rather than full A-TGPO, Hotpot accuracy not yet robust.

## Appendix A: Target-Construction Diagnostics

### Synthetic Target Diagnostics

Controlled synthetic cases isolate whether target construction assigns mass to useful correct trajectories rather than redundant correct trajectories. The main metric is `useful-red`: useful-correct target mass minus redundant-correct target mass.

```text
method                useful-red
OriginalPolicy        +0.002
FinalReward-TPO       +0.004
PrefixIG-GRPO         +0.093
A-TGPO-components     +0.108
PrefixIG-TPO          +0.112
PrefixIG-TPO+curv     +0.288
```

### Real MLX Generated Rollout Diagnostics

Generated-policy diagnostics with Qwen2.5-0.5B showed that reward-only objectives improve correctness but do not reliably prefer efficient search. PrefixIG-based target construction improves the useful-vs-redundant gap.

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

These diagnostics are useful for debugging target construction, but they should not be presented as trained-model results.

## Appendix B: A-TGPO Component Ablation

To connect with the original A-TGPO methodology, we implemented small analogues of turn-group normalization, variance-rescaled accumulation, and adaptive turn-level clipping in the toy trainer.

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

This ablation should be interpreted carefully. It does not show that A-TGPO fails in general; it shows that simplified token-level scalar credit can over-concentrate on redundant-correct trajectories in our toy setting.

## Appendix C: Additional Qwen LoRA Ablations

### Single-Hop vs Multi-Hop Mixed Training

The first mixed-hop adapter fixed single-hop over-search but lost some multi-hop behavior. This motivated the mixed-hop+noisy training regime used in the main results.

### Noisy/Distractor Stress Tests

The early noisy/distractor setup was too harsh because the retriever often returned only misleading evidence. We corrected it so misleading and corrective evidence can appear together. This produced the mixed-hop+noisy result reported in the main text.

### True Offline TPO Loss Pilot

We implemented a true grouped offline TPO trainer, but the first pure grouped-loss result underperformed weighted-SFT-style distillation. Qwen2.5-0.5B 4-bit LoRA with 6 trajectories per group peaked around `10GB` memory and was feasible, but the training signal was unstable. This motivated the online rollout-refresh plus token/action anchor variant.

### Candidate-Selection Inference

We also tested no-training candidate selection: generate candidates with Base+SDAR, then reweight/select with PrefixIG-TPO or A-TGPO scoring. This did not clearly help PrefixIG-TPO. The result is useful as an internal diagnostic, but it should not be in the main paper unless we want to emphasize that the benefit comes from policy adaptation rather than cheap reranking.

## Current Bottom Line

The strongest defensible claim is:

```text
PrefixIG-TPO is a simple target-policy method for evidence-seeking reasoning. It improves useful evidence behavior in controlled RLVR and small Qwen LoRA experiments, is competitive with an A-TGPO-style proxy, and shows promising process-quality improvements in HotpotQA-mini when combined with an SDAR search prior.
```

The weakest claim would be:

```text
PrefixIG-TPO robustly improves real-world HotpotQA answer accuracy.
```

We should not claim that yet.
