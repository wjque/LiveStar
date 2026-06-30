# Phase 1 技术报告：基于 LiveStar 的 EgoProactive Baseline 验证

日期：2026-06-30

## 1. 阶段目标

本阶段目标是将 LiveStar 作为 Proactive VLM 赛道的 backbone，完成从环境、权重、推理、预测格式到 Wearable AI Workshop starter kit 评测的第一版闭环。目标不是追求最高指标，而是建立一个可以复现实验、定位问题、支撑后续优化的 baseline pipeline。

本阶段完成内容：

- 配置并验证 LiveStar 推理环境。
- 下载并验证 LiveStar-8B 权重。
- 清理项目 `inference/` 目录中的权重残留，改为从外部权重目录加载。
- 新增通用自定义权重目录推理脚本。
- 基于 ECCV 2026 Wearable AI Workshop 的 EgoProactive 验证集和 starter kit，实现第一版 LiveStar baseline 评测脚本。
- 修正 LiveStar 两种推理模式的使用方式：区分“生成输出”和“困惑度内部计算”。
- 记录 smoke test 实验结果，分析 baseline 的有效性和主要问题。

## 2. 代码与数据位置

项目目录：

```text
/home/quewenjun/workspace/proactive_vlm/LiveStar
```

LiveStar 权重目录：

```text
/data1/LiveStar_8B
```

EgoProactive 验证数据：

```text
/data1/wearable_ai_challenge_data/egoproactive
```

starter kit：

```text
/data1/wearable_ai_challenge_data/starter_kit
```

本阶段新增/修改的主要文件：

```text
inference/infer_custom.py
evaluate/eval_proactive.py
docs/phase1_livestar_proactive_report.md
```

实验输出目录：

```text
evaluate/output/
```

## 3. 环境与依赖

使用 Conda 环境：

```text
LiveStar
```

关键版本：

```text
python=3.9.21
torch=2.5.1+cu124
torchvision=0.20.1+cu124
transformers=4.37.2
opencv=4.11.0
safetensors=0.7.0
cuda_available=True
cuda_version=12.4
gpu_count=8
gpu0=NVIDIA A800-SXM4-80GB
```

环境安装过程中遇到的问题与处理：

- `flash_attn==2.7.4.post1` 不能在未安装 torch 的情况下直接从 requirements 构建。
  - 处理：先安装 `torch==2.5.1` 和 `torchvision==0.20.1`，再安装其余 requirements，最后安装匹配 `cu12 + torch2.5 + cp39` 的 flash-attn wheel。
- Hugging Face / GitHub 网络下载不稳定。
  - 处理：按允许的方式使用本地代理：

```bash
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
```

- pip/conda 下载缓存过大。
  - 处理：执行 pip cache purge 和 conda clean，清理缓存与临时 wheel。
- `uv` 已从 base Conda 环境卸载，避免后续环境工具混杂。

## 4. 模型选择与权重

本阶段选择 LiveStar-8B 作为 backbone。

选择理由：

- LiveStar 原生面向 online / streaming video understanding。
- README 中提供了 streaming inference demo。
- 模型包含 SVeD 风格的 response-silence decoding，可用于 Proactive VLM 的时机判断。

权重来源：

```text
yzy666/LiveStar_8B
```

本地权重文件：

```text
/data1/LiveStar_8B/model-00001-of-00004.safetensors 4940001680
/data1/LiveStar_8B/model-00002-of-00004.safetensors 4915914584
/data1/LiveStar_8B/model-00003-of-00004.safetensors 4915914592
/data1/LiveStar_8B/model-00004-of-00004.safetensors 1379084264
```

目录大小：

```text
16G /data1/LiveStar_8B
```

权重验证：

- 4 个 safetensors shard 均可通过 `safe_open` 打开。
- 模型可以通过 `AutoModel.from_pretrained(..., trust_remote_code=True)` 加载。
- 单帧推理可正常生成文本。
- 单帧推理显存峰值约 `16.25GB`。

## 5. 数据集结构与统计

EgoProactive 验证集文件：

```text
/data1/wearable_ai_challenge_data/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl
```

视频目录：

```text
/data1/wearable_ai_challenge_data/egoproactive/val/*.mp4
```

每条样本包含字段：

```text
video_path
duration_in_sec
video_intervals
query
domain
task
answers
dialog
```

示例：

```text
video_path: 0028bee556249cd3.mp4
query: How do I decorate a notebook cover with stickers?
task: Decorating a notebook cover with stickers
video_intervals: 10 chunks
answers: ["$interrupt$...", "$interrupt$...", "$silent$", ...]
```

验证集统计：

```text
sessions: 700
videos: 700
total chunks: 9935
chunks per session: min=4, max=30, mean=14.19, median=14
duration seconds: min=16.0, max=587.7, mean=162.10, median=146.7
interrupt labels: 5352, ratio=0.5387
silent labels: 4583, ratio=0.4613
```

该标签分布接近平衡，因此 all-interrupt 或 all-silent baseline 都会明显暴露类别偏置。

## 6. Starter Kit 评测规则

starter kit 中 Proactive 任务定义：

- 输入：一个 session，包含 query、视频 chunk 序列、历史 dialog。
- 每个 chunk 输出一个 answer。
- answer 必须以如下标签之一开头：

```text
$interrupt$<utterance>
$silent$
```

评测逻辑：

- 仅评估二分类标签：`interrupt` vs `silent`。
- `parse_tag()` 只检查字符串开头。
- 任意不以 `$interrupt$` 开头的输出都会被视为 `silent`。
- 指标包括：
  - interrupt precision / recall / F1
  - silent precision / recall / F1
  - macro F1
  - g-mean F1

本阶段的评测命令最终调用：

```bash
python /data1/wearable_ai_challenge_data/starter_kit/run_evaluation.py \
  --task proactive \
  --eval-only \
  --golden <golden_jsonl> \
  --predictions <prediction_jsonl> \
  --eval-output <result_json>
```

## 7. Pipeline 设计

Phase 1 pipeline 分为两部分：LiveStar 推理生成预测、starter kit 评测打分。

### 7.1 推理环境与评测环境解耦

由于 LiveStar 依赖较旧的 `transformers==4.37.2`，而 starter kit README 建议 Python 3.12、较新的 transformers 和可能的 vLLM 环境，因此脚本设计为允许两套环境分离：

- LiveStar 环境：负责模型加载、视频推理、生成预测 JSONL。
- Eval 环境：只负责运行 starter kit 的 `eval-only`。

`evaluate/eval_proactive.py` 提供参数：

```bash
--eval-python "conda run -n wearable_eval python"
```

如果当前 LiveStar 环境可以运行 starter kit 的纯评测路径，也可以使用默认 `sys.executable`。

### 7.2 外部权重目录加载

README 原始流程要求：

```bash
mv LiveStar_8B/*.safetensors inference/
```

这会污染项目目录并占用空间。本阶段改为运行时构造临时模型目录：

1. 从 `inference/` 软链接模型代码、tokenizer、config、index 文件。
2. 从 `/data1/LiveStar_8B` 软链接 `model-*.safetensors`。
3. 使用临时目录调用 `AutoTokenizer.from_pretrained()` 和 `AutoModel.from_pretrained()`。
4. 进程结束后临时目录自动清理。

对应实现：

```text
inference/infer_custom.py::make_runtime_model_dir
evaluate/eval_proactive.py::make_runtime_model_dir
```

这样 `inference/` 不再包含权重文件，便于版本管理。

### 7.3 视频帧采样

在 `evaluate/eval_proactive.py` 中：

1. 使用 OpenCV 打开视频。
2. 根据 `video_intervals` 对每个 chunk 采样。
3. 默认每个 chunk 采样：

```text
--frames-per-interval 2
```

smoke test 中为加速使用：

```text
--frames-per-interval 1
```

4. 图像预处理：

```text
Resize(448, 448)
ToTensor
Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
dtype=bfloat16
device=cuda
```

### 7.4 LiveStar 两种推理模式

LiveStar 的 `model.chat()` 有两类关键调用方式：

#### A. 生成分支

调用条件：

```python
check_answer is None
```

行为：

- 调用 `self.generate(...)`
- 返回自然语言 response
- 如果 `return_history=True`，返回：

```python
response, history, new_past_key_values
```

该分支才可以作为 `$interrupt$` 的文本来源。

#### B. 困惑度 / 内部检查分支

调用条件：

```python
check_answer is not None
```

行为：

- 构造 labels。
- 调用 `streaming_infer(...)`。
- 返回：

```python
perplexity, None
```

该分支仅用于判断当前 chunk 是否需要重新生成，不应把返回值当作输出文本。

### 7.5 SVeD 时机判断流程

修正后的 `eval_proactive.py` 采用官方 `demo.py` 的 SVeD 状态机：

1. 第一个 chunk：
   - 使用生成分支生成 `output_last`。
   - 使用 `self_check=True` 的困惑度分支计算 `decode_threshold`。
   - 输出 `$interrupt$<output_last>`。

2. 后续 chunk：
   - 使用 `check_answer=output_last[:check_len]` 调用困惑度分支。
   - 得到 `output_perplexity`。
   - 若：

```python
output_perplexity > decode_threshold * decode_factor
```

则认为需要说话：

- 调用生成分支得到新的 `output_last`。
- 再用 `self_check=True` 更新 `decode_threshold`。
- 输出 `$interrupt$<output_last>`。

否则：

- 输出 `$silent$`。
- 将当前 frame marker 合并到 `chat_history`，保持官方 demo 的上下文行为。

默认参数：

```text
--decode-factor 1.04
--check-len 1000
--ppl-runs 1
```

### 7.6 为什么不静默下采样历史帧

LiveStar 的 `chat_history` 会保留历史里的 `<image>` 占位符。若对累计帧做下采样但不同步修改历史文本中的 image placeholder，会导致：

```text
sum(num_patches_list) != prompt 中 image token 数量
```

或者更隐蔽地造成图文对齐错误。

因此当前 baseline 中：

- 默认 `--max-frames 512`。
- 如果累计帧数超过 `--max-frames`，直接 fail fast。
- 后续若要加速，需要设计显式的历史压缩或 KV/cache 策略，而不是简单丢帧。

## 8. 使用方式

### 8.1 通用自定义权重推理

脚本：

```text
inference/infer_custom.py
```

命令：

```bash
cd /home/quewenjun/workspace/proactive_vlm/LiveStar

conda run -n LiveStar python inference/infer_custom.py \
  --weights-dir /data1/LiveStar_8B \
  --video assets/videos/HPtIGhOsViM.mp4 \
  --num-frames 1 \
  --max-new-tokens 64
```

验证结果：

```text
OUTPUT_BEGIN
A woman in a blue top is sitting at a wooden table, holding a bottle of red nail polish.
OUTPUT_END
max_memory_allocated_gb: 16.25
```

### 8.2 EgoProactive baseline 生成 + 评测

脚本：

```text
evaluate/eval_proactive.py
```

smoke test 命令：

```bash
cd /home/quewenjun/workspace/proactive_vlm/LiveStar

conda run -n LiveStar python evaluate/eval_proactive.py \
  --max-samples 1 \
  --frames-per-interval 1 \
  --max-new-tokens 48 \
  --ppl-runs 1 \
  --output evaluate/output/smoke_proactive_sved_predictions.jsonl \
  --eval-output evaluate/output/smoke_proactive_sved_results.json
```

只生成预测：

```bash
conda run -n LiveStar python evaluate/eval_proactive.py \
  --generate-only \
  --output evaluate/output/egoproactive_livestar_predictions.jsonl
```

只评测已有预测：

```bash
conda run -n LiveStar python evaluate/eval_proactive.py \
  --eval-only \
  --output evaluate/output/egoproactive_livestar_predictions.jsonl
```

使用独立评测环境：

```bash
conda run -n LiveStar python evaluate/eval_proactive.py \
  --eval-python "conda run -n wearable_eval python"
```

## 9. 实验记录

本阶段只进行了小规模 smoke test，目的是验证 pipeline 与推理模式是否正确，不代表最终模型性能。

### 9.1 单帧模型加载验证

输入：

```text
assets/videos/HPtIGhOsViM.mp4
```

设置：

```text
num_frames=1
max_new_tokens=64
weights_dir=/data1/LiveStar_8B
```

结果：

```text
模型加载成功
CUDA 可用
输出正常
max_memory_allocated_gb=16.25
```

输出示例：

```text
A woman in a blue top is sitting at a wooden table, holding a bottle of red nail polish.
```

结论：

- 权重、custom remote code、tokenizer、视觉 encoder、语言生成链路均可运行。

### 9.2 初版 naive baseline

初版实现方式：

- 每个 chunk 都直接调用生成分支。
- 未使用 LiveStar 的困惑度分支。
- 将普通文本 fallback 为 `$interrupt$...`。

结果文件：

```text
evaluate/output/smoke_proactive_results_full_summary.json
```

1 条样本，10 个 chunk：

```text
macro_f1=0.3333
gmean_f1=0.0000
interrupt_precision=0.5000
interrupt_recall=1.0000
interrupt_f1=0.6667
silent_precision=0.0000
silent_recall=0.0000
silent_f1=0.0000
tp=5
fp=5
tn=0
fn=0
support=10
```

分析：

- 模型输出全部被转成 `$interrupt$`。
- 由于该样本 gold 中有 5 个 interrupt、5 个 silent，因此 all-interrupt 得到 interrupt recall=1.0，但 silent F1=0。
- g-mean F1=0，说明二分类时机判断完全失衡。
- 根因不是模型一定总想说话，而是推理逻辑错误：没有使用 SVeD 的 silence decision；同时 fallback 策略过于激进。

### 9.3 修正为 SVeD 后的 baseline

修正内容：

- 首 chunk 生成。
- 后续 chunk 先计算 perplexity。
- 只有超过阈值才生成 `$interrupt$`。
- 否则输出 `$silent$`。

结果文件：

```text
evaluate/output/smoke_proactive_sved_results_summary.json
```

1 条样本，10 个 chunk：

```text
macro_f1=0.6970
gmean_f1=0.6963
interrupt_precision=0.7500
interrupt_recall=0.6000
interrupt_f1=0.6667
silent_precision=0.6667
silent_recall=0.8000
silent_f1=0.7273
tp=3
fp=1
tn=4
fn=2
support=10
```

预测示例：

```text
0 $interrupt$The person is holding a brown cardboard.
1 $interrupt$The person sticks the cardboard onto the notebook cover.
2 $interrupt$The person takes out a sticker and sticks it onto the cardboard.
3 $silent$
4 $silent$
5 $silent$
6 $silent$
7 $silent$
8 $interrupt$The person takes out another sticker and sticks it onto the cardboard.
9 $silent$
```

分析：

- 与 naive all-interrupt 相比，macro F1 从 0.3333 提升到 0.6970。
- g-mean F1 从 0 提升到 0.6963，说明类别平衡明显改善。
- SVeD 的时机判断对 Proactive 任务是有用的。
- 但输出内容仍偏视频描述，例如 `The person ...`，不是真正的用户指导。

### 9.4 加强指导性 prompt 后的结果

修改 prompt：

- 要求 second person / imperative。
- 明确禁止描述视频。
- 给出 good / bad style 示例。

结果文件：

```text
evaluate/output/smoke_proactive_sved_prompt_results_summary.json
```

1 条样本，10 个 chunk：

```text
macro_f1=0.5833
gmean_f1=0.5774
interrupt_precision=0.6667
interrupt_recall=0.4000
interrupt_f1=0.5000
silent_precision=0.5714
silent_recall=0.8000
silent_f1=0.6667
tp=2
fp=1
tn=4
fn=3
support=10
```

预测示例：

```text
0 $interrupt$The person is not decorating the notebook cover.
1 $interrupt$The person is not decorating the notebook cover.
2 $interrupt$The person is not decorating the notebook cover.
3 $silent$
4 $silent$
5 $silent$
6 $silent$
7 $silent$
8 $silent$
9 $silent$
```

分析：

- 更强 prompt 降低了 interrupt 频率，silent recall 保持 0.8。
- 但 interrupt recall 从 0.6 降到 0.4。
- 内容仍然没有变成指导，而是描述/否定描述。
- 说明仅靠 prompt 很难将 LiveStar 的 streaming description 行为稳定改造成 procedural proactive guidance。

## 10. 主要问题与解决方案

### 10.1 权重放在 inference 目录不利于维护

问题：

- README 建议将 `.safetensors` 移入 `inference/`。
- 权重体积 16G，不适合进入代码目录，也不适合 git 管理。

解决：

- 清理 `inference/model-*.safetensors`。
- 使用临时目录软链接模型代码和外部权重。
- 新增 `--weights-dir` 参数。

结果：

- `inference/` 保持轻量。
- `/data1/LiveStar_8B` 作为唯一权重目录。

### 10.2 decord 退出时崩溃

问题：

- 使用 decord 读取视频时，模型已经输出结果，但进程退出出现：

```text
pure virtual method called
terminate called without an active exception
exit code 134
```

解决：

- baseline 脚本改用 OpenCV 读取视频帧。

结果：

- 推理可以正常退出。
- 与 starter kit 的默认 frame extraction 方式更一致。

### 10.3 初版推理误用 LiveStar chat 模式

问题：

- 初版每个 chunk 都直接生成文本。
- 没有使用 `check_answer` 分支做 internal perplexity。
- 将无标签输出强行转换为 `$interrupt$`，导致 all-interrupt。

解决：

- 对照 `inference/demo.py` 与 `modeling_livestar_chat.py` 修正：
  - `check_answer is None` 才生成文本。
  - `check_answer is not None` 只用于 perplexity。
  - 使用 `decode_threshold * decode_factor` 决定 silence / interrupt。

结果：

- 1 条样本 smoke test 中不再 all-interrupt。
- macro F1 从 0.3333 提升到 0.6970。

### 10.4 LiveStar 输出偏描述，不偏指导

问题：

- 即便 prompt 要求指导，模型仍输出：

```text
The person is ...
```

而不是：

```text
Place the sticker ...
Press it flat ...
```

原因分析：

- LiveStar 原始 demo 目标是 streaming video description。
- Wearable AI Proactive 任务需要 procedural guidance。
- 两者在输出语义上存在 domain/task gap。

当前处理：

- prompt 中加入 second-person / imperative 约束。
- 但 smoke test 显示 prompt 不能根本解决。

后续方向：

- 使用 EgoProactive 的 train/val 风格数据做 SFT 或 LoRA。
- 增加一个 lightweight rewrite/guidance adapter，将 description 转成 action guidance。
- 使用 task/query/dialog 对输出进行约束，而不是只依赖 video caption 模型。

### 10.5 环境版本可能冲突

问题：

- LiveStar 需要旧版 transformers 和 remote code。
- starter kit 推荐较新的 Python/transformers/vLLM 组合。

解决：

- 推理脚本支持 `--eval-python`。
- 生成和评测通过 JSONL 文件交互。

结果：

- 可以在 `LiveStar` 环境生成预测。
- 可以在另一个 `wearable_eval` 环境评测。

## 11. 当前结论

1. LiveStar-8B 可以成功作为 Proactive VLM baseline backbone 加载和运行。
2. 权重外置到 `/data1/LiveStar_8B` 的方案可行，避免污染 `inference/`。
3. EgoProactive 评测 pipeline 已打通：
   - JSONL annotations
   - OpenCV frame sampling
   - LiveStar SVeD 推理
   - starter kit eval-only
   - results JSON 输出
4. LiveStar 的 SVeD 机制对 interrupt/silent timing 有实际帮助。
5. 当前最大问题不是 timing，而是 utterance 内容：
   - 模型偏描述。
   - Proactive 任务需要指导。
   - prompt engineering 效果有限。

## 12. 下一阶段建议

Phase 2 建议按以下顺序推进：

1. 扩大评测规模。
   - 先跑 `--max-samples 10/50/100`，记录平均耗时、显存、F1。
   - 再决定是否跑完整 700 validation。

2. 做 decode-factor sweep。
   - 尝试：

```text
decode_factor = 1.00, 1.02, 1.04, 1.06, 1.08, 1.10
```

   - 观察 interrupt precision / recall trade-off。

3. 做 prompt sweep。
   - 分离 timing prompt 和 guidance generation prompt。
   - 测试是否需要在生成时显式加入 query/task/dialog，而 perplexity 判断时只用 frame marker。

4. 内容侧优化。
   - 使用 EgoProactive answers 做 LoRA/SFT。
   - 或增加 post-processing/rewrite model，把 LiveStar description 改写为 task-specific guidance。

5. 性能优化。
   - 当前 baseline 每个 chunk 都重新传累计帧，简单但不高效。
   - 后续可以研究 `past_key_values` / `use_kvcache`。
   - 需要谨慎处理 image placeholder 与 `num_patches_list` 对齐。

6. 输出文件管理。
   - 建议将 `evaluate/output/` 加入 `.gitignore`。
   - 保留关键 summary JSON 到 `docs/experiments/` 或实验追踪系统。

## 13. Phase 1 可复现命令汇总

验证模型单帧推理：

```bash
cd /home/quewenjun/workspace/proactive_vlm/LiveStar

conda run -n LiveStar python inference/infer_custom.py \
  --weights-dir /data1/LiveStar_8B \
  --video assets/videos/HPtIGhOsViM.mp4 \
  --num-frames 1 \
  --max-new-tokens 64
```

运行 1 条 EgoProactive smoke test：

```bash
conda run -n LiveStar python evaluate/eval_proactive.py \
  --max-samples 1 \
  --frames-per-interval 1 \
  --max-new-tokens 48 \
  --ppl-runs 1 \
  --output evaluate/output/smoke_proactive_sved_predictions.jsonl \
  --eval-output evaluate/output/smoke_proactive_sved_results.json
```

只评测已有预测：

```bash
conda run -n LiveStar python evaluate/eval_proactive.py \
  --eval-only \
  --max-samples 1 \
  --output evaluate/output/smoke_proactive_sved_predictions.jsonl \
  --eval-output evaluate/output/smoke_proactive_sved_results.json
```

使用独立评测环境：

```bash
conda run -n LiveStar python evaluate/eval_proactive.py \
  --eval-python "conda run -n wearable_eval python"
```

