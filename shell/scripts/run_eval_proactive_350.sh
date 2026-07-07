#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/quewenjun/workspace/proactive_vlm/LiveStar"
cd "${REPO_ROOT}"

FRAMES_PER_INTERVAL="${FRAMES_PER_INTERVAL:-4}"
CTX_INTERVALS="${CTX_INTERVALS:-20}"
MAX_HISTORY_TURNS="${MAX_HISTORY_TURNS:-${CTX_INTERVALS}}"
MAX_CONTEXT_FRAMES=$((FRAMES_PER_INTERVAL * CTX_INTERVALS))
GPUS="${GPUS:-4,5,6,7}"

TAG="fpi${FRAMES_PER_INTERVAL}_ctxi${CTX_INTERVALS}_f${MAX_CONTEXT_FRAMES}_hist${MAX_HISTORY_TURNS}"
RESULT="evaluate/output/egoproactive_sved_sample350_${TAG}_majority1.jsonl"
SHARD_DIR="evaluate/output/egoproactive_sved_sample350_${TAG}_majority1_shards"
VIZ_DIR="evaluate/output/egoproactive_sved_sample350_${TAG}_majority1_viz"
RUN_LOG="evaluate/output/egoproactive_sved_sample350_${TAG}_majority1_run.log"

LEGACY_RESULT="evaluate/output/egoproactive_sved_sample350_fpi4_majority1.jsonl"
LEGACY_SHARD_DIR="evaluate/output/egoproactive_sved_sample350_fpi4_majority1_shards"
LEGACY_VIZ_DIR="evaluate/output/egoproactive_sved_sample350_fpi4_majority1_viz"
LEGACY_RUN_LOG="evaluate/output/egoproactive_sved_sample350_fpi4_majority1_run.log"
LEGACY_CTX32_RESULT="evaluate/output/egoproactive_sved_sample350_fpi4_ctx32_hist8_majority1.jsonl"
LEGACY_CTX32_SHARD_DIR="evaluate/output/egoproactive_sved_sample350_fpi4_ctx32_hist8_majority1_shards"
LEGACY_CTX32_VIZ_DIR="evaluate/output/egoproactive_sved_sample350_fpi4_ctx32_hist8_majority1_viz"
LEGACY_CTX32_RUN_LOG="evaluate/output/egoproactive_sved_sample350_fpi4_ctx32_hist8_majority1_run.log"

mkdir -p evaluate/output
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[run] Cleaning stale sample350 outputs"
rm -f "${RESULT}" "${RUN_LOG}" "${LEGACY_RESULT}" "${LEGACY_RUN_LOG}" "${LEGACY_CTX32_RESULT}" "${LEGACY_CTX32_RUN_LOG}"
rm -rf "${SHARD_DIR}" "${VIZ_DIR}" "${LEGACY_SHARD_DIR}" "${LEGACY_VIZ_DIR}" "${LEGACY_CTX32_SHARD_DIR}" "${LEGACY_CTX32_VIZ_DIR}"

echo "[run] Starting 350-sample multi-GPU evaluation on GPUs ${GPUS}"
echo "[run] frames_per_interval=${FRAMES_PER_INTERVAL} ctx_intervals=${CTX_INTERVALS} max_context_frames=${MAX_CONTEXT_FRAMES} max_history_turns=${MAX_HISTORY_TURNS}"
conda run -n LiveStar python evaluate/eval_proactive.py \
  --num-samples 350 \
  --gpus "${GPUS}" \
  --frames-per-interval "${FRAMES_PER_INTERVAL}" \
  --max-context-intervals "${CTX_INTERVALS}" \
  --max-context-frames "${MAX_CONTEXT_FRAMES}" \
  --max-history-turns "${MAX_HISTORY_TURNS}" \
  --clear-cache-per-video \
  --output "${RESULT}" \
  2>&1 | tee "${RUN_LOG}"

echo "[run] Building visualization for worst 10 per-video macro-F1 samples"
python evaluate/visualize_proactive_results.py \
  --input "${RESULT}" \
  --output-dir "${VIZ_DIR}" \
  --top-k-worst 10 \
  2>&1 | tee -a "${RUN_LOG}"

echo "[done] Result: ${RESULT}"
echo "[done] Visualization: ${VIZ_DIR}/index.html"
echo "[done] Log: ${RUN_LOG}"
