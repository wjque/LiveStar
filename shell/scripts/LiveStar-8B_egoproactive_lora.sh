#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=${GRADIENT_ACC:-$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))}
if [ "${GRADIENT_ACC}" -lt 1 ]; then
  GRADIENT_ACC=1
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-34229}
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

ANNOTATIONS=${ANNOTATIONS:-/data1/wearable_ai_challenge_data/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl}
VIDEO_FOLDER=${VIDEO_FOLDER:-/data1/wearable_ai_challenge_data/egoproactive/val}
DATA_OUTPUT_DIR=${DATA_OUTPUT_DIR:-/data1/finetune/data/wearableai_val}
MAX_SESSIONS=${MAX_SESSIONS:-}
FRAMES_PER_INTERVAL=${FRAMES_PER_INTERVAL:-2}
FRAME_HISTORY_CHUNKS=${FRAME_HISTORY_CHUNKS:-4}
MAX_HISTORY_TURNS=${MAX_HISTORY_TURNS:-4}
TRAIN_RATIO=${TRAIN_RATIO:-0.8}
DEV_RATIO=${DEV_RATIO:-0.1}
SEED=${SEED:-42}
FORCE_PREPARE=${FORCE_PREPARE:-0}

MODEL_CODE_DIR=${MODEL_CODE_DIR:-${PROJECT_ROOT}/inference}
WEIGHTS_DIR=${WEIGHTS_DIR:-/data1/LiveStar_8B}
RUNTIME_MODEL_DIR=${RUNTIME_MODEL_DIR:-${PROJECT_ROOT}/work_dirs/runtime/LiveStar_8B}
OUTPUT_DIR=${OUTPUT_DIR:-/data1/finetune/model/lora_adapter}
META_PATH=${META_PATH:-${DATA_OUTPUT_DIR}/meta/egoproactive_train_meta.json}

USE_LLM_LORA=${USE_LLM_LORA:-16}
USE_BACKBONE_LORA=${USE_BACKBONE_LORA:-0}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-True}
FREEZE_MLP=${FREEZE_MLP:-True}
FREEZE_LLM=${FREEZE_LLM:-True}
UNFREEZE_VIT_LAYERS=${UNFREEZE_VIT_LAYERS:-0}
UNFREEZE_LM_HEAD=${UNFREEZE_LM_HEAD:-False}
LEARNING_RATE=${LEARNING_RATE:-4e-5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-10}
MAX_STEPS=${MAX_STEPS:--1}
SAVE_STRATEGY=${SAVE_STRATEGY:-no}
SAVE_STEPS=${SAVE_STEPS:-200}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/zero_stage1_config.json}
REPORT_TO=${REPORT_TO:-tensorboard}

mkdir -p "${OUTPUT_DIR}" "${RUNTIME_MODEL_DIR}"

if [ "${FORCE_PREPARE}" = "1" ] || [ ! -f "${META_PATH}" ]; then
  PREPARE_ARGS=(
    python livestar/train/prepare_egoproactive_sft.py
    --annotations "${ANNOTATIONS}"
    --video-folder "${VIDEO_FOLDER}"
    --output-dir "${DATA_OUTPUT_DIR}"
    --frames-per-interval "${FRAMES_PER_INTERVAL}"
    --frame-history-chunks "${FRAME_HISTORY_CHUNKS}"
    --max-history-turns "${MAX_HISTORY_TURNS}"
    --train-ratio "${TRAIN_RATIO}"
    --dev-ratio "${DEV_RATIO}"
    --seed "${SEED}"
  )
  if [ -n "${MAX_SESSIONS}" ]; then
    PREPARE_ARGS+=(--max-sessions "${MAX_SESSIONS}")
  fi
  "${PREPARE_ARGS[@]}"
fi

if [ ! -f "${RUNTIME_MODEL_DIR}/config.json" ] || \
   [ ! -f "${RUNTIME_MODEL_DIR}/model.safetensors.index.json" ] || \
   ! compgen -G "${RUNTIME_MODEL_DIR}/model-*.safetensors" > /dev/null; then
  find "${RUNTIME_MODEL_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  for item in "${MODEL_CODE_DIR}"/*; do
    name=$(basename "${item}")
    if [[ "${name}" == model-*.safetensors ]]; then
      continue
    fi
    ln -s "${item}" "${RUNTIME_MODEL_DIR}/${name}"
  done
  for item in "${WEIGHTS_DIR}"/model-*.safetensors; do
    ln -s "${item}" "${RUNTIME_MODEL_DIR}/$(basename "${item}")"
  done
fi

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  livestar/train/livestar_chat_finetune.py \
  --model_name_or_path "${RUNTIME_MODEL_DIR}" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "${META_PATH}" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 1 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm "${FREEZE_LLM}" \
  --freeze_mlp "${FREEZE_MLP}" \
  --freeze_backbone "${FREEZE_BACKBONE}" \
  --use_llm_lora "${USE_LLM_LORA}" \
  --use_backbone_lora "${USE_BACKBONE_LORA}" \
  --unfreeze_vit_layers "${UNFREEZE_VIT_LAYERS}" \
  --unfreeze_lm_head "${UNFREEZE_LM_HEAD}" \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACC}" \
  --evaluation_strategy "no" \
  --save_strategy "${SAVE_STRATEGY}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 1 \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size False \
  --use_thumbnail False \
  --ps_version "v2" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --report_to "${REPORT_TO}" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
