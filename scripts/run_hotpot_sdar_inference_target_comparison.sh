#!/usr/bin/env bash
set -euo pipefail

cd /Users/mehdiiranmanesh/Desktop/A-TGPO

export PYTHONPYCACHEPREFIX=/private/tmp/atgpo_pycache

PY=/Users/mehdiiranmanesh/Desktop/mlx-lm-lora/venv/bin/python3
MODEL=Qwen/Qwen2.5-1.5B-Instruct
TAG=hotpot_sdar_inference_targets_qwen25_1p5b_8x2_seed101

mkdir -p outputs/logs

"$PY" scripts/eval_hotpotqa_mini.py \
  --model "$MODEL" \
  --load-in-4bits \
  --num-examples 8 \
  --samples-per-example 2 \
  --seed 101 \
  --max-search-turns 2 \
  --retrieval-top-k 2 \
  --action-max-tokens 36 \
  --answer-max-tokens 16 \
  --max-doc-chars 500 \
  --correct-f1-threshold 0.65 \
  --allow-reference-substring \
  --skill-prompt-file /Users/mehdiiranmanesh/Desktop/SDAR/skills/search/general_skills.md \
  --skill-prompt-file /Users/mehdiiranmanesh/Desktop/SDAR/skills/search/multi_hop_reasoning.md \
  --skill-prompt-file /Users/mehdiiranmanesh/Desktop/SDAR/skills/search/compare.md \
  --skill-max-chars 1200 \
  --target-comparison \
  --lambda-ig 0.5 \
  --tau 0.7 \
  --output-jsonl "outputs/eval_${TAG}.jsonl" \
  2>&1 | tee "outputs/logs/${TAG}.log"

echo "Done. This was inference-time target-policy comparison only; no LoRA training."
echo "Output JSONL: outputs/eval_${TAG}.jsonl"
echo "Log:          outputs/logs/${TAG}.log"
