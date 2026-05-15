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

We then ran a memory-light generated-policy behavior diagnostic comparing the base model and the rg_eff-LoRA model. Each model generated 8 trajectories total, from 4 examples with 2 samples per example, using flexible oracle-search rollouts with at most 2 search turns. This is a small pilot due to laptop memory constraints, but it tests the key behavioral question: whether the adapter changes what the model actually generates, not just its offline SFT loss.

The main row is `OriginalPolicy`, which reflects the actual generated behavior of each model before any target reweighting. The base Qwen2.5-0.5B model produced mostly distractor/wrong search trajectories, while the rg_eff-LoRA adapter produced mostly useful correct search trajectories:

```text
model                 useful  redundant  no_search  distractor  other  correct  useful-red
Base Qwen             0.114   0.000      0.000      0.886       0.000  0.114    +0.114
rg_eff-LoRA           0.880   0.000      0.000      0.120       0.000  0.880    +0.880
```

In count terms, the base model generated 1 useful correct trajectory and 7 distractor/wrong trajectories. The rg_eff-LoRA model shifted this distribution sharply toward useful correct search. Relative to the base model, the LoRA adapter improved useful correct behavior by `+0.766`, improved correctness by `+0.766`, and reduced distractor behavior by `-0.766`.

This result is the first real-model behavioral evidence that the offline rg_eff target distribution transfers into generated search behavior. However, it should be interpreted as a pilot, not a final paper-scale result. The sample size is small, and neither model generated redundant correct trajectories in this memory-light setting, so this particular comparison mainly tests useful correctness rather than redundancy reduction. The redundancy claim is currently supported more directly by the controlled synthetic diagnostics and toy RLVR training results. The next step is to repeat this base-vs-LoRA behavioral evaluation with more examples, more samples per example, and settings that elicit redundant correct behavior without exceeding memory limits.

Overall, the experiments so far support a clear story. Final reward teaches correctness but not evidence efficiency. PrefixIG provides useful intermediate credit. A-TGPO-style scalar advantage improves evidence use in some clean settings, but scalar turn-level credit can be brittle when redundant or noisy evidence is correlated with correctness. TPO target construction uses the same evidence signal more directly by redistributing probability mass among competing trajectories. Reward-gated efficiency fixes a real failure mode of raw PrefixIG by penalizing redundant search only when the answer is correct. The noisy-retriever results show where this matters most: under strong evidence corruption, PrefixIG-GRPO overweights redundant correct trajectories, while PrefixIG-TPO with reward-gated efficiency preserves a positive useful-vs-redundant gap. The Qwen2.5-0.5B LoRA pilot adds an important behavioral step: a small adapter distilled from rg_eff target weights substantially improved useful correct generated trajectories in a memory-limited real-model setting. The next empirical milestone is to scale this evaluation and replace SFT distillation with a true offline TPO trainer over grouped trajectory targets.
