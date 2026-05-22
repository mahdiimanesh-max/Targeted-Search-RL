#!/usr/bin/env bash
set -euo pipefail

cd /Users/mehdiiranmanesh/Desktop/A-TGPO

export PYTHONPYCACHEPREFIX=/private/tmp/atgpo_pycache

PY=/Users/mehdiiranmanesh/Desktop/mlx-lm-lora/venv/bin/python3
MODEL=Qwen/Qwen2.5-1.5B-Instruct
TAG=hotpot_sdar_qwen25_1p5b_8x2_seed101

mkdir -p outputs/logs outputs/adapters

SKILLS=(
  --skill-prompt-file /Users/mehdiiranmanesh/Desktop/SDAR/skills/search/general_skills.md
  --skill-prompt-file /Users/mehdiiranmanesh/Desktop/SDAR/skills/search/multi_hop_reasoning.md
  --skill-prompt-file /Users/mehdiiranmanesh/Desktop/SDAR/skills/search/compare.md
  --skill-max-chars 1200
)

COMMON_EVAL=(
  --model "$MODEL"
  --load-in-4bits
  --num-examples 8
  --samples-per-example 2
  --seed 101
  --max-search-turns 2
  --retrieval-top-k 2
  --action-max-tokens 36
  --answer-max-tokens 16
  --max-doc-chars 500
  --correct-f1-threshold 0.65
  --allow-reference-substring
  --no-sample-diagnostics
  "${SKILLS[@]}"
)

COMMON_TRAIN=(
  --model "$MODEL"
  --load-in-4bits
  --online-iters 5
  --prompts-per-iter 1
  --samples-per-prompt 2
  --updates-per-iter 1
  --learning-rate 5e-6
  --anchor-beta 0.1
  --max-search-turns 2
  --retrieval-top-k 2
  --action-max-tokens 32
  --answer-max-tokens 16
  --max-doc-chars 500
  --max-seq-length 1024
  --correct-f1-threshold 0.65
  --use-f1-reward
  --seed 7
  "${SKILLS[@]}"
)

echo "============================================================"
echo "1/5 Base + SDAR skill-prompt evaluation"
echo "============================================================"
"$PY" scripts/eval_hotpotqa_mini.py \
  "${COMMON_EVAL[@]}" \
  --output-jsonl "outputs/eval_${TAG}_base_sdar.jsonl" \
  2>&1 | tee "outputs/logs/${TAG}_base_sdar_eval.log"

echo "============================================================"
echo "2/5 Train SDAR + PrefixIG-TPO online LoRA"
echo "============================================================"
"$PY" scripts/online_hotpotqa_prefixig_tpo_lora.py \
  "${COMMON_TRAIN[@]}" \
  --adapter-path "outputs/adapters/${TAG}_prefixig_tpo" \
  --target-method prefixig_tpo \
  --output-jsonl "outputs/${TAG}_prefixig_tpo_online_rollouts.jsonl" \
  2>&1 | tee "outputs/logs/${TAG}_prefixig_tpo_train.log"

echo "============================================================"
echo "3/5 Evaluate SDAR + PrefixIG-TPO online LoRA"
echo "============================================================"
"$PY" scripts/eval_hotpotqa_mini.py \
  "${COMMON_EVAL[@]}" \
  --adapter-path "outputs/adapters/${TAG}_prefixig_tpo" \
  --output-jsonl "outputs/eval_${TAG}_prefixig_tpo.jsonl" \
  2>&1 | tee "outputs/logs/${TAG}_prefixig_tpo_eval.log"

echo "============================================================"
echo "4/5 Train SDAR + A-TGPO proxy online LoRA"
echo "============================================================"
"$PY" scripts/online_hotpotqa_prefixig_tpo_lora.py \
  "${COMMON_TRAIN[@]}" \
  --adapter-path "outputs/adapters/${TAG}_atgpo_proxy" \
  --target-method atgpo_proxy \
  --output-jsonl "outputs/${TAG}_atgpo_proxy_online_rollouts.jsonl" \
  2>&1 | tee "outputs/logs/${TAG}_atgpo_proxy_train.log"

echo "============================================================"
echo "5/5 Evaluate SDAR + A-TGPO proxy online LoRA"
echo "============================================================"
"$PY" scripts/eval_hotpotqa_mini.py \
  "${COMMON_EVAL[@]}" \
  --adapter-path "outputs/adapters/${TAG}_atgpo_proxy" \
  --output-jsonl "outputs/eval_${TAG}_atgpo_proxy.jsonl" \
  2>&1 | tee "outputs/logs/${TAG}_atgpo_proxy_eval.log"

echo "Done. Key outputs:"
echo "  outputs/eval_${TAG}_base_sdar.jsonl"
echo "  outputs/eval_${TAG}_prefixig_tpo.jsonl"
echo "  outputs/eval_${TAG}_atgpo_proxy.jsonl"
echo "  outputs/logs/${TAG}_*.log"
