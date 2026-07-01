# LiveStar-8B EgoProactive LoRA 微调说明

本文档说明 `shell/scripts/LiveStar-8B_egoproactive_lora.sh` 的训练 pipeline、常用可调参数、训练运行方式、训练指标查看方式，以及验证/测试运行方式。

项目目录：

```bash
cd /home/quewenjun/workspace/proactive_vlm/LiveStar
```

默认 Conda 环境：

```bash
conda activate LiveStar
```

## 1. 训练 Pipeline

训练脚本：

```text
shell/scripts/LiveStar-8B_egoproactive_lora.sh
```

整体流程如下。

### 1.1 初始化分布式训练环境

脚本默认使用 2 张 GPU：

```bash
GPUS=${GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-8}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=${GRADIENT_ACC:-$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))}
```

默认有效 batch size 为：

```text
GPUS * PER_DEVICE_BATCH_SIZE * GRADIENT_ACC = 2 * 1 * 4 = 8
```

脚本还会设置：

```bash
CUDA_VISIBLE_DEVICES=0,1
PYTHONPATH=${PROJECT_ROOT}
MASTER_ADDR=127.0.0.1
MASTER_PORT=34229
LAUNCHER=pytorch
```

### 1.2 准备 EgoProactive SFT 数据

当 `FORCE_PREPARE=1`，或者默认 meta 文件不存在时，脚本会运行：

```bash
python livestar/train/prepare_egoproactive_sft.py
```

默认输入数据：

```text
ANNOTATIONS=/data1/wearable_ai_challenge_data/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl
VIDEO_FOLDER=/data1/wearable_ai_challenge_data/egoproactive/val
DATA_OUTPUT_DIR=/data1/finetune/data/wearableai_val
```

数据准备逻辑：

- 先按 session/video 行级别随机划分，而不是按 chunk/sample 划分。
- 默认 `TRAIN_RATIO=0.8`，`DEV_RATIO=0.1`，剩余为 test。
- 默认 `SEED=42`，保证划分可复现。
- 如果设置 `MAX_SESSIONS`，会先截取前 N 个 session，再做随机划分。
- 每个 video interval 默认抽 1 帧，单帧时取 interval 中点帧。
- 每个 chunk 生成一个 SFT sample，答案规范化为 `$silent$` 或 `$interrupt$...`。
- 每个 sample 默认包含当前 chunk 加前 4 个 chunk 的抽帧历史。
- 每个 prompt 默认包含最近 4 轮对话历史。

输出文件：

```text
${DATA_OUTPUT_DIR}/frames/
${DATA_OUTPUT_DIR}/annotations/egoproactive_train.jsonl
${DATA_OUTPUT_DIR}/annotations/egoproactive_dev.jsonl
${DATA_OUTPUT_DIR}/annotations/egoproactive_test.jsonl
${DATA_OUTPUT_DIR}/meta/egoproactive_train_meta.json
${DATA_OUTPUT_DIR}/meta/egoproactive_dev_meta.json
${DATA_OUTPUT_DIR}/meta/egoproactive_test_meta.json
```

训练默认只读取：

```text
META_PATH=${DATA_OUTPUT_DIR}/meta/egoproactive_train_meta.json
```

也就是说，默认训练脚本不会自动使用 dev/test split 做评估。

### 1.3 组装运行时模型目录

脚本默认将模型代码目录和权重目录组装到运行时目录：

```text
MODEL_CODE_DIR=${PROJECT_ROOT}/inference
WEIGHTS_DIR=/data1/LiveStar_8B
RUNTIME_MODEL_DIR=${PROJECT_ROOT}/work_dirs/runtime/LiveStar_8B
```

如果运行时目录缺少 `config.json`、`model.safetensors.index.json` 或 `model-*.safetensors`，脚本会清空运行时目录，然后：

- 从 `MODEL_CODE_DIR` 软链接 tokenizer/config/model code 等文件。
- 从 `WEIGHTS_DIR` 软链接 `model-*.safetensors` 权重分片。

### 1.4 启动 LoRA 微调

训练入口：

```text
livestar/train/livestar_chat_finetune.py
```

启动方式：

```bash
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  livestar/train/livestar_chat_finetune.py \
  ...
```

默认 LoRA 配置：

```text
USE_LLM_LORA=16
USE_BACKBONE_LORA=0
FREEZE_LLM=True
FREEZE_BACKBONE=True
FREEZE_MLP=True
```

因此默认会冻结 LLM、视觉 backbone、MLP projector，然后只在 LLM 上挂 LoRA adapter，rank 为 16，`lora_alpha=32`。

训练默认参数：

```text
bf16=True
num_train_epochs=1
learning_rate=4e-5
weight_decay=0.05
warmup_ratio=0.03
lr_scheduler_type=cosine
max_seq_length=8192
grad_checkpoint=True
group_by_length=True
evaluation_strategy=no
deepspeed=zero_stage1_config.json
report_to=tensorboard
```

### 1.5 保存结果

默认输出目录：

```text
OUTPUT_DIR=/data1/finetune/model/lora_adapter
```

默认 `SAVE_STRATEGY=no`，训练过程中不按 step 保存 checkpoint。训练结束后：

- 如果使用 LoRA，rank 0 会保存 LoRA adapter。
- 默认只启用 LLM LoRA，因此 adapter 直接保存在 `${OUTPUT_DIR}`。
- 如果同时启用 LLM LoRA 和 vision LoRA，则分别保存到 `${OUTPUT_DIR}/llm` 和 `${OUTPUT_DIR}/vision`。

常见输出：

```text
${OUTPUT_DIR}/adapter_config.json
${OUTPUT_DIR}/adapter_model.safetensors
${OUTPUT_DIR}/training_log.txt
${OUTPUT_DIR}/train_results.json
${OUTPUT_DIR}/trainer_state.json
${OUTPUT_DIR}/events.out.tfevents.*
```

## 2. 参数说明

所有参数都可以通过环境变量覆盖。

### 2.1 数据准备参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `ANNOTATIONS` | `/data1/wearable_ai_challenge_data/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl` | EgoProactive 原始 JSONL 标注。 |
| `VIDEO_FOLDER` | `/data1/wearable_ai_challenge_data/egoproactive/val` | 视频目录。 |
| `DATA_OUTPUT_DIR` | `/data1/finetune/data/wearableai_val` | 抽帧、SFT JSONL、meta 输出目录。 |
| `MAX_SESSIONS` | 空 | 只使用前 N 个 session，适合 smoke test。 |
| `FRAMES_PER_INTERVAL` | `1` | 每个 interval 抽帧数。 |
| `FRAME_HISTORY_CHUNKS` | `4` | 每个 sample 包含当前 chunk 以及多少个历史 chunk 的帧。 |
| `MAX_HISTORY_TURNS` | `4` | prompt 中保留的历史对话轮数。`0` 表示不保留，负数表示保留全部。 |
| `TRAIN_RATIO` | `0.8` | session 级 train split 比例。 |
| `DEV_RATIO` | `0.1` | session 级 dev split 比例。剩余为 test。 |
| `SEED` | `42` | 数据划分随机种子。 |
| `FORCE_PREPARE` | `0` | 设为 `1` 时强制重新抽帧和生成 SFT 数据。 |

### 2.2 LoRA 与冻结参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `USE_LLM_LORA` | `16` | LLM LoRA rank。设为 `0` 关闭 LLM LoRA。 |
| `USE_BACKBONE_LORA` | `0` | 视觉 backbone LoRA rank。设为非 0 开启。 |
| `FREEZE_LLM` | `True` | 是否冻结 LLM 原始参数。 |
| `FREEZE_BACKBONE` | `True` | 是否冻结视觉 backbone 原始参数。 |
| `FREEZE_MLP` | `True` | 是否冻结 MLP projector。 |
| `UNFREEZE_VIT_LAYERS` | `0` | 额外解冻 ViT encoder 的后续层。 |
| `UNFREEZE_LM_HEAD` | `False` | 是否解冻 LLM lm_head。 |

默认推荐保持：

```bash
USE_LLM_LORA=16 USE_BACKBONE_LORA=0 FREEZE_LLM=True FREEZE_BACKBONE=True FREEZE_MLP=True
```

### 2.3 优化与保存参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `LEARNING_RATE` | `4e-5` | 学习率。 |
| `NUM_TRAIN_EPOCHS` | `1` | 训练 epoch 数。 |
| `MAX_STEPS` | `-1` | 最大训练 step。设为正数可用于控制更新次数。 |
| `SAVE_STRATEGY` | `no` | checkpoint 保存策略。可设为 `steps`。 |
| `SAVE_STEPS` | `200` | `SAVE_STRATEGY=steps` 时的保存间隔。 |
| `MAX_SEQ_LENGTH` | `8192` | tokenizer 最大序列长度。 |
| `DEEPSPEED_CONFIG` | `${PROJECT_ROOT}/zero_stage1_config.json` | DeepSpeed 配置文件。 |
| `REPORT_TO` | `tensorboard` | Trainer 日志后端。 |

## 3. 运行训练

### 3.1 默认 LoRA 训练

```bash
cd /home/quewenjun/workspace/proactive_vlm/LiveStar
conda activate LiveStar

bash shell/scripts/LiveStar-8B_egoproactive_lora.sh
```

### 3.2 指定 GPU 和输出目录

```bash
cd /home/quewenjun/workspace/proactive_vlm/LiveStar
conda activate LiveStar

GPUS=2 \
CUDA_VISIBLE_DEVICES=0,1 \
OUTPUT_DIR=/data1/finetune/model/lora_adapter_exp01 \
bash shell/scripts/LiveStar-8B_egoproactive_lora.sh
```

### 3.3 快速 smoke test

只用少量 session 和少量 step 验证 pipeline：

```bash
cd /home/quewenjun/workspace/proactive_vlm/LiveStar
conda activate LiveStar

MAX_SESSIONS=8 \
FORCE_PREPARE=1 \
MAX_STEPS=5 \
OUTPUT_DIR=/data1/finetune/model/lora_adapter_smoke \
DATA_OUTPUT_DIR=/data1/finetune/data/wearableai_val_smoke \
MASTER_PORT=34230 \
bash shell/scripts/LiveStar-8B_egoproactive_lora.sh
```

### 3.4 重新生成数据再训练

如果修改了抽帧、history 或划分参数，需要强制重新准备数据：

```bash
cd /home/quewenjun/workspace/proactive_vlm/LiveStar
conda activate LiveStar

FORCE_PREPARE=1 \
FRAMES_PER_INTERVAL=2 \
FRAME_HISTORY_CHUNKS=6 \
MAX_HISTORY_TURNS=6 \
OUTPUT_DIR=/data1/finetune/model/lora_adapter_hist6 \
DATA_OUTPUT_DIR=/data1/finetune/data/wearableai_val_hist6 \
bash shell/scripts/LiveStar-8B_egoproactive_lora.sh
```

### 3.5 保存中间 checkpoint

默认只保存最终 LoRA adapter。如果需要每隔固定 step 保存 checkpoint：

```bash
cd /home/quewenjun/workspace/proactive_vlm/LiveStar
conda activate LiveStar

SAVE_STRATEGY=steps \
SAVE_STEPS=200 \
OUTPUT_DIR=/data1/finetune/model/lora_adapter_ckpt \
bash shell/scripts/LiveStar-8B_egoproactive_lora.sh
```

## 4. 查看训练指标

### 4.1 查看训练日志

默认训练日志：

```bash
tail -n 100 /data1/finetune/model/lora_adapter/training_log.txt
```

日志中通常包含：

- `TrainingArguments` 参数摘要。
- rank、device、GPU、分布式训练信息。
- tokenizer 和模型加载信息。
- 数据集名称和长度，例如 `Add dataset: egoproactive_train with length: ...`。
- 可训练参数名，默认主要是 LoRA 参数。
- Trainer 每 `logging_steps=1` 输出的 `loss`、`learning_rate`、`epoch` 等指标。
- 失败时的 traceback。

注意：数据准备阶段的 `print` 默认不会写入 `training_log.txt`，因为 `tee` 只包住了后面的 `torchrun` 训练命令。

### 4.2 查看最终训练指标

```bash
cat /data1/finetune/model/lora_adapter/train_results.json
cat /data1/finetune/model/lora_adapter/all_results.json
```

常见字段包括：

- `train_loss`
- `train_runtime`
- `train_samples_per_second`
- `train_steps_per_second`
- `train_samples`

### 4.3 查看 Trainer 状态和逐步日志历史

```bash
cat /data1/finetune/model/lora_adapter/trainer_state.json
```

其中 `log_history` 会记录训练过程中的 step 级日志。

也可以用 `jq` 提取 loss：

```bash
jq '.log_history[] | select(has("loss"))' /data1/finetune/model/lora_adapter/trainer_state.json
```

### 4.4 使用 TensorBoard

```bash
tensorboard \
  --logdir /data1/finetune/model/lora_adapter \
  --host 0.0.0.0 \
  --port 6006
```

浏览器打开：

```text
http://<服务器IP>:6006
```

如果使用 SSH 端口转发：

```bash
ssh -L 6006:127.0.0.1:6006 <user>@<server>
```

本地浏览器打开：

```text
http://localhost:6006
```

## 5. 运行验证和测试

训练脚本默认不会自动运行验证或测试。任务级评测需要使用：

```text
evaluate/eval_proactive.py
```

该脚本会：

- 加载 LiveStar-8B base 权重。
- 可选加载 LoRA adapter。
- 在 EgoProactive annotation 上生成预测。
- 调用 starter kit 的 `run_evaluation.py` 计算 proactive 指标。
- 写出 prediction JSONL、result JSON 和 run log。

默认评测输入：

```text
ANNOTATIONS=/data1/wearable_ai_challenge_data/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl
VIDEO_FOLDER=/data1/wearable_ai_challenge_data/egoproactive/val
WEIGHTS_DIR=/data1/LiveStar_8B
MODEL_CODE_DIR=${PROJECT_ROOT}/inference
STARTER_KIT=/data1/wearable_ai_challenge_data/starter_kit
```

### 5.1 快速验证 smoke run

只跑少量 session，确认训练后的 LoRA 能正常加载、生成和打分：

```bash
cd /home/quewenjun/workspace/proactive_vlm/LiveStar
conda activate LiveStar

python evaluate/eval_proactive.py \
  --lora-adapter /data1/finetune/model/lora_adapter \
  --max-samples 8 \
  --output evaluate/output/egoproactive_lora_smoke_predictions.jsonl \
  --eval-output evaluate/output/egoproactive_lora_smoke_results.json
```

查看结果：

```bash
cat evaluate/output/egoproactive_lora_smoke_results.json
tail -n 5 evaluate/output/log.jsonl
```

### 5.2 全量验证/测试

在默认 EgoProactive validation annotation 上跑全量评测：

```bash
cd /home/quewenjun/workspace/proactive_vlm/LiveStar
conda activate LiveStar

python evaluate/eval_proactive.py \
  --lora-adapter /data1/finetune/model/lora_adapter \
  --output evaluate/output/egoproactive_lora_predictions.jsonl \
  --eval-output evaluate/output/egoproactive_lora_results.json
```

查看结果：

```bash
cat evaluate/output/egoproactive_lora_results.json
tail -n 5 evaluate/output/log.jsonl
```

### 5.3 多 GPU 并行生成预测

`eval_proactive.py` 支持按 sample 做多进程并行，每个 worker 使用一个 GPU：

```bash
cd /home/quewenjun/workspace/proactive_vlm/LiveStar
conda activate LiveStar

python evaluate/eval_proactive.py \
  --lora-adapter /data1/finetune/model/lora_adapter \
  --gpu-ids 0,1,2,3 \
  --output evaluate/output/egoproactive_lora_predictions.jsonl \
  --eval-output evaluate/output/egoproactive_lora_results.json
```

### 5.4 只生成预测，不打分

```bash
cd /home/quewenjun/workspace/proactive_vlm/LiveStar
conda activate LiveStar

python evaluate/eval_proactive.py \
  --lora-adapter /data1/finetune/model/lora_adapter \
  --output evaluate/output/egoproactive_lora_predictions.jsonl \
  --generate-only
```

### 5.5 已有预测，只重新打分

```bash
cd /home/quewenjun/workspace/proactive_vlm/LiveStar
conda activate LiveStar

python evaluate/eval_proactive.py \
  --output evaluate/output/egoproactive_lora_predictions.jsonl \
  --eval-output evaluate/output/egoproactive_lora_results.json \
  --eval-only
```

### 5.6 断点续跑评测

如果预测生成中断，可以用 `--resume` 跳过已写出的预测行：

```bash
cd /home/quewenjun/workspace/proactive_vlm/LiveStar
conda activate LiveStar

python evaluate/eval_proactive.py \
  --lora-adapter /data1/finetune/model/lora_adapter \
  --resume \
  --output evaluate/output/egoproactive_lora_predictions.jsonl \
  --eval-output evaluate/output/egoproactive_lora_results.json
```

### 5.7 starter-kit 使用单独 Python 环境

如果 LiveStar 推理环境和 starter-kit 评测环境不同，可以通过 `--eval-python` 指定评测环境：

```bash
cd /home/quewenjun/workspace/proactive_vlm/LiveStar
conda activate LiveStar

python evaluate/eval_proactive.py \
  --lora-adapter /data1/finetune/model/lora_adapter \
  --output evaluate/output/egoproactive_lora_predictions.jsonl \
  --eval-output evaluate/output/egoproactive_lora_results.json \
  --eval-python "conda run -n wearable_eval python"
```

### 5.8 关于 train/dev/test split 的注意事项

`prepare_egoproactive_sft.py` 生成的：

```text
${DATA_OUTPUT_DIR}/annotations/egoproactive_dev.jsonl
${DATA_OUTPUT_DIR}/annotations/egoproactive_test.jsonl
```

是 LiveStar SFT sample 格式，字段包括 `image`、`conversations`、`label` 等；而 `evaluate/eval_proactive.py` 需要的是 EgoProactive 原始 session annotation 格式，字段包括 `video_path`、`video_intervals`、`answers` 等。

因此，当前仓库中可直接运行的验证/测试方式是使用 `evaluate/eval_proactive.py` 对原始 EgoProactive annotation 做 smoke 或全量评测。若要严格按训练时的 session 级 `dev/test` split 做 held-out 验证，需要额外导出对应的原始 session JSONL，再作为 `--annotations` 传给评测脚本。

示例：

```bash
python evaluate/eval_proactive.py \
  --annotations /path/to/egoproactive_dev_sessions.jsonl \
  --video-folder /data1/wearable_ai_challenge_data/egoproactive/val \
  --lora-adapter /data1/finetune/model/lora_adapter \
  --output evaluate/output/egoproactive_lora_dev_predictions.jsonl \
  --eval-output evaluate/output/egoproactive_lora_dev_results.json
```

```bash
python evaluate/eval_proactive.py \
  --annotations /path/to/egoproactive_test_sessions.jsonl \
  --video-folder /data1/wearable_ai_challenge_data/egoproactive/val \
  --lora-adapter /data1/finetune/model/lora_adapter \
  --output evaluate/output/egoproactive_lora_test_predictions.jsonl \
  --eval-output evaluate/output/egoproactive_lora_test_results.json
```

## 6. 推荐实验记录

每次训练建议记录：

```text
OUTPUT_DIR
DATA_OUTPUT_DIR
ANNOTATIONS
VIDEO_FOLDER
GPUS
BATCH_SIZE
PER_DEVICE_BATCH_SIZE
GRADIENT_ACC
USE_LLM_LORA
USE_BACKBONE_LORA
FREEZE_LLM
FREEZE_BACKBONE
FREEZE_MLP
LEARNING_RATE
NUM_TRAIN_EPOCHS
MAX_STEPS
FRAMES_PER_INTERVAL
FRAME_HISTORY_CHUNKS
MAX_HISTORY_TURNS
TRAIN_RATIO
DEV_RATIO
SEED
```

每次评测建议记录：

```text
lora_adapter
annotations
video_folder
output
eval_output
max_samples
gpu_ids
frames_per_interval
max_frames
max_history_turns
decode_factor
ppl_runs
```
