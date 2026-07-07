#!/usr/bin/env bash
set -euo pipefail

NUM_SAMPLES="${NUM_SAMPLES:-350}"
INTERVAL_SAMPLE_FPS="${INTERVAL_SAMPLE_FPS:-2}"
L_MAX="${L_MAX:-160}"
ALPHA="${ALPHA:-1.06}"
SIGMA="${SIGMA:-0.75}"
BETA="${BETA:-0.3}"
BEAM_K="${BEAM_K:-3}"
MAX_RECALL="${MAX_RECALL:-2}"
RECALL_MIN_GAP="${RECALL_MIN_GAP:-8}"
NUM_RUNS="${NUM_RUNS:-3}"
GPUS="${GPUS:-4,5,6,7}"

FPS_TAG="${INTERVAL_SAMPLE_FPS//./p}"
TAG="sample${NUM_SAMPLES}_fps${FPS_TAG}_lmax${L_MAX}_majority1"
RESULT="evaluate/output/egoproactive_livestarpro_${TAG}.jsonl"
SHARD_DIR="evaluate/output/egoproactive_livestarpro_${TAG}_shards"
VIZ_DIR="evaluate/output/egoproactive_livestarpro_${TAG}_viz"
RUN_LOG="evaluate/output/egoproactive_livestarpro_${TAG}_run.log"

mkdir -p evaluate/output
rm -f "${RESULT}" "${RUN_LOG}"
rm -rf "${SHARD_DIR}" "${VIZ_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[run] Starting LiveStarPro ${NUM_SAMPLES}-sample evaluation on GPUs ${GPUS}"
echo "[run] Result: ${RESULT}"
echo "[run] Visualization: ${VIZ_DIR}"

conda run -n LiveStar python evaluate/eval_proactive_pro.py \
  --num-samples "${NUM_SAMPLES}" \
  --gpus "${GPUS}" \
  --interval-sample-fps "${INTERVAL_SAMPLE_FPS}" \
  --l-max "${L_MAX}" \
  --alpha "${ALPHA}" \
  --sigma "${SIGMA}" \
  --beta "${BETA}" \
  --beam-k "${BEAM_K}" \
  --max-recall "${MAX_RECALL}" \
  --recall-min-gap "${RECALL_MIN_GAP}" \
  --num-runs "${NUM_RUNS}" \
  --kv \
  --clear-cache-per-video \
  --output "${RESULT}" \
  2>&1 | tee "${RUN_LOG}"

python evaluate/visualize_proactive_results.py \
  --input "${RESULT}" \
  --output-dir "${VIZ_DIR}" \
  --top-k-worst 10 \
  2>&1 | tee -a "${RUN_LOG}"

echo "[done] LiveStarPro results saved to ${RESULT}"
echo "[done] HTML report: ${VIZ_DIR}/index.html"
