# LiveStar Gate Finetune 与 Chat Finetune 对比

本文档对比以下两个训练入口：

- `livestar/train/livestar_chat_finetune.py`
- `livestar/train/livestar_gate_finetune.py`

结论：`livestar_gate_finetune.py` 是在常规 chat SFT 流程上增加 silent/interrupt 二分类 gate 训练的版本。它仍然保留 LiveStar 的视觉编码、LLM、LoRA、冻结策略和 `Trainer` 主流程，但数据预处理、模型类、loss 构造、日志和保存逻辑都围绕 gate head 做了扩展。

## 1. 主要修改

### 1.1 模型入口从 chat 切到 gate

`livestar_gate_finetune.py` 不再从 `livestar.model.livestar_chat` 导入模型，而是从 `livestar.model.livestar_gate` 导入：

```python
from livestar.model.livestar_gate import (
    InternVisionConfig,
    InternVisionModel,
    InternVLChatConfig,
    InternVLChatModel,
)
```

对应的 gate 模型在 `livestar/model/livestar_gate/modeling_livestar_chat.py` 中新增了：

- `gate_head`：`LayerNorm -> Linear -> GELU -> Dropout -> Linear(1)`。
- `gate_labels`、`gate_positions`、`gate_loss_weight`、`lm_loss_weight`、`gate_pos_weight` 等 `forward` 入参。
- `GateCausalLMOutputWithPast`，额外返回 `gate_loss`、`lm_loss`、`gate_logits`。

配置文件 `livestar/model/livestar_gate/configuration_livestar_chat.py` 也新增了 gate 相关配置：

```text
gate_hidden_size
gate_dropout
gate_loss_weight
lm_loss_weight
gate_pos_weight
```

### 1.2 训练脚本新增 gate head 保存逻辑

`livestar_gate_finetune.py` 新增 `save_gate_head()`：

- 保存 `model.gate_head.state_dict()`。
- 优先保存为 `gate_head.safetensors`，如果没有 `safetensors` 则保存为 `gate_head.pt`。
- 额外保存 `gate_config.json`，记录 gate hidden size、dropout、loss 权重、`conv_style` 和 `max_seq_length` 等信息。

chat 版本只保存 LoRA adapter 或完整模型，不会单独保存 gate head。

### 1.3 新增 EgoProactive prompt 清洗

gate 版本新增以下清洗函数：

- `strip_egoproactive_prefix()`
- `sanitize_prompt_text()`
- `sanitize_gate_conversations()`

作用是把原始 EgoProactive SFT 数据中的控制符从语言建模目标里剥离：

- 当前 assistant 答案以 `$interrupt$` 开头时，去掉 `$interrupt$` 前缀，只保留自然语言指令。
- 当前 assistant 答案为 `$silent$` 时，转成空字符串。
- prompt 历史里的 `$silent$` 行会被丢弃。
- prompt 历史里的 `$interrupt$` 前缀会被移除。
- 原本要求模型输出 `$silent$` 或 `$interrupt$` 的指令文本被改写成更自然的决策指令。

因此 gate 训练的目标不是让 LLM 生成 `$silent$/$interrupt$` 字符串，而是：

- gate head 学习是否应该说话。
- LLM 只在 interrupt 样本上学习要说的自然语言内容。

### 1.4 预处理函数改为 gate 专用

chat 版本会根据 `conv_style` 选择多个预处理函数：

- `preprocess`
- `preprocess_mpt`
- `preprocess_internlm`
- `preprocess_phi3`
- `preprocess_internvl2_5`

gate 版本只支持 `conv_style=internvl2_5`：

```python
if self.template_name == 'internvl2_5':
    return preprocess_internvl2_5_gate
raise ValueError(...)
```

`preprocess_internvl2_5_gate()` 做了几件关键事情：

1. 清洗 conversation。
2. 把 `<image>` 替换为 `<img><IMG_CONTEXT>...</img>` 图像 token 序列。
3. 按 ChatML 形式拼接 system/user/assistant turn。
4. 生成常规 `input_ids`、`labels`、`attention_mask`。
5. 额外生成 `gate_labels` 和 `gate_positions`。

其中 `gate_position` 被设为 assistant turn 开始前的最后一个 token 位置：

```python
gate_position = max(0, cursor - 1)
```

这个位置表示模型在开始回答之前，基于当前视频 chunk 和上下文判断是否应该说话。

### 1.5 labels 构造从全量 assistant SFT 改为 interrupt-only LM

chat 版本会按普通 SFT 方式监督 assistant 输出。

gate 版本的 `labels` 构造更特殊：

- system/user token 全部置为 `IGNORE_INDEX=-100`。
- assistant token 默认也全部置为 `-100`。
- 只有当 `gate_label == 1`，也就是 interrupt 样本时，才把 assistant 内容作为 LM 目标。
- assistant header，例如 `<|im_start|>assistant\n`，被置为 `-100`。
- assistant 末尾 token 也被置为 `-100`。

结果是：

- silent 样本只贡献 gate BCE loss，不贡献 LM loss。
- interrupt 样本同时贡献 gate BCE loss 和自然语言指令的 LM loss。
- `$interrupt$` 前缀不会作为 LM 目标。
- `$silent$` 不会作为 LM 目标。

### 1.6 Dataset 返回新增字段

gate 版本在图像、多图、视频、纯文本路径中都会额外返回：

```python
gate_labels
gate_positions
```

`gate_label` 的来源：

1. 如果样本顶层字段 `label` 是 `silent`，则为 `0`。
2. 如果 `label` 是 `interrupt`，则为 `1`。
3. 如果没有显式 `label`，则根据最后一轮 assistant 答案推断：
   - 以 `$silent$` 开头或为空：`0`
   - 否则：`1`

这些字段不需要单独 collator。`concat_pad_data_collator()` 会把普通 tensor 字段 `torch.stack` 成 batch；`pixel_values` 和 `image_flags` 仍走原来的 concat 逻辑。

### 1.7 packed dataset 被禁用

chat 版本仍保留 packed dataset 分支。

gate 版本在参数解析后直接禁止：

```python
if data_args.use_packed_ds:
    raise ValueError('Gate fine-tuning does not support --use_packed_ds yet.')
```

原因是 gate 训练额外依赖每个样本的 `gate_position`，packed 后需要重新定义样本边界、gate 位置和 loss 聚合，目前脚本没有实现。

### 1.8 新增 gate loss 权重参数

gate 版本在 `DataTrainingArguments` 中新增：

```text
gate_loss_weight: silent/interrupt BCE loss 权重，默认 1.0
lm_loss_weight: interrupt-only LM loss 权重，默认 1.0
gate_pos_weight: BCE 正类权重，默认 0.0，表示不启用
```

这些参数会写入 `InternVLChatConfig`，再由模型 `forward` 读取。

### 1.9 gate head 强制初始化、float 计算并参与训练

加载或构建模型后，gate 版本会执行：

```python
model._init_gate_head()
model.gate_head.float()
```

随后无论是否冻结 LLM、ViT、MLP，都会确保 gate head 可训练：

```python
for param in model.gate_head.parameters():
    param.requires_grad = True
```

这意味着默认 LoRA/冻结策略下，训练参数通常是：

- LLM LoRA adapter，如果启用 `use_llm_lora`
- Vision LoRA adapter，如果启用 `use_backbone_lora`
- `gate_head`

### 1.10 Trainer 日志被扩展

gate 版本新增：

- `GateTrainer`
- `GateProgressCallback`

`GateTrainer.log()` 会从模型上读取：

```text
_last_lm_loss
_last_gate_loss
_last_gate_prob
```

并把它们加入训练日志：

```text
lm_loss
gate_loss
gate_prob
```

`GateProgressCallback` 会在 tqdm postfix 中显示 `train_loss` 和 `gate_prob`。

## 2. 训练流程

`livestar_gate_finetune.py` 的整体训练流程如下。

### 2.1 初始化

1. 调用 `patch_accelerate_dataloader_args()`，兼容 Transformers 4.37 和 Accelerate 1.x 的 dataloader 参数差异。
2. 应用 RMSNorm、sampler、dataloader patch。
3. 根据 `LAUNCHER` 初始化分布式训练，默认 backend 是 `nccl`。
4. 用 `HfArgumentParser` 解析 `ModelArguments`、`DataTrainingArguments`、`TrainingArguments`。
5. 禁止 `use_packed_ds`。
6. 设置 `training_args.logging_nan_inf_filter = False`，保留非有限 loss 的显式暴露。

### 2.2 Tokenizer 和特殊 token

1. 从 `model_name_or_path` 或 `llm_path` 加载 tokenizer。
2. 设置 `tokenizer.model_max_length = data_args.max_seq_length`。
3. 添加视觉和定位相关特殊 token：

```text
<img>
</img>
<IMG_CONTEXT>
<quad>
</quad>
<ref>
</ref>
<box>
</box>
```

4. 记录 `<IMG_CONTEXT>` 的 token id，用于之后把这些 token 的 embedding 替换为视觉特征。

### 2.3 加载或构建模型

有两种路径：

1. 如果提供 `model_name_or_path`，从已有 checkpoint 加载 `InternVLGateModel`。
2. 否则分别从 `vision_path`、`llm_path` 加载 ViT 和 LLM，再组装 `InternVLGateModel`。

无论哪条路径，都会把以下运行参数写入 config：

```text
template
select_layer
dynamic_image_size
use_thumbnail
ps_version
min_dynamic_patch
max_dynamic_patch
gate_loss_weight
lm_loss_weight
gate_pos_weight
```

随后：

1. 设置 `model.img_context_token_id`。
2. 初始化并转 float `gate_head`。
3. 根据 `force_image_size` 调整视觉位置编码。
4. 固定 `model.num_image_token = 16`。
5. 如果 tokenizer 新增了 token，则 resize LLM embedding，并用旧 embedding 均值初始化新增 token。
6. 关闭 LLM cache，开启视觉和 LLM gradient checkpoint。

### 2.4 构建训练数据集

`build_datasets()` 读取 `data_args.meta_path` 指向的 meta json。每个数据集会创建一个 `LazySupervisedDataset`。

`LazySupervisedDataset.__getitem__()` 根据样本字段走不同路径：

- `image` 是字符串：`multi_modal_get_item()`
- `image` 是列表：`multi_modal_multi_image_get_item()`
- `video` 存在：`video_get_item()`
- 否则：`pure_text_get_item()`

每条样本的核心处理顺序是：

1. 加载图像、视频帧或构造纯文本白图占位。
2. 根据 `dynamic_image_size` 做动态切图，否则使用单图。
3. 使用 `build_transform()` 转成 `pixel_values`。
4. 推断 `gate_label`。
5. 调用 `preprocess_internvl2_5_gate()` 生成 token、LM labels、gate label、gate position。
6. 计算 `position_ids`。
7. 返回 Trainer 所需字段：

```text
input_ids
labels
attention_mask
position_ids
pixel_values
image_flags
gate_labels
gate_positions
```

### 2.5 冻结、LoRA 和可训练参数

模型加载后按参数执行：

- `freeze_backbone`：冻结 ViT。
- `freeze_llm`：冻结 LLM。
- `freeze_mlp`：冻结视觉到语言的 MLP projector。
- `use_backbone_lora`：给 ViT 包 LoRA。
- `use_llm_lora`：给 LLM 包 LoRA。
- `unfreeze_lm_head`：单独解冻 LLM head。
- `unfreeze_vit_layers`：解冻部分 ViT layer。

最后强制设置 `gate_head.requires_grad = True`。

### 2.6 Collator 和 Trainer

因为 gate 版本禁止 packed dataset，实际使用的是：

```python
collator = concat_pad_data_collator
```

它会：

- pad `input_ids`、`labels`、`position_ids` 到当前 batch 最大长度。
- 重新计算 `attention_mask`。
- stack `gate_labels`、`gate_positions` 等普通 tensor 字段。
- concat `pixel_values` 和 `image_flags`。

然后构造：

```python
trainer = GateTrainer(...)
```

训练时调用：

```python
trainer.train(resume_from_checkpoint=checkpoint)
```

### 2.7 保存

训练完成后：

- 如果使用 LoRA，只保存 LoRA adapter，并额外保存 gate head。
- 如果不使用 LoRA，保存完整模型，并额外保存 gate head。

gate 版本新增的输出至少包括：

```text
gate_head.safetensors 或 gate_head.pt
gate_config.json
```

## 3. 损失函数构建

gate 版本的损失由两部分组成：

```text
total_loss = lm_loss_weight * lm_loss + gate_loss_weight * gate_loss
```

### 3.1 LM loss

模型 `forward()` 先完成常规多模态前向：

1. 用 LLM embedding 层把 `input_ids` 转为 `input_embeds`。
2. 用 ViT 提取 `pixel_values` 的视觉特征。
3. 找到 `input_ids == img_context_token_id` 的位置。
4. 用视觉特征替换这些 `<IMG_CONTEXT>` token 的 embedding。
5. 调用 LLM 得到 `logits` 和最后层 hidden states。

LM loss 的计算方式：

```python
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
lm_loss = CrossEntropyLoss()(shift_logits, shift_labels)
```

`CrossEntropyLoss` 默认忽略 `ignore_index=-100`。脚本还会先判断当前 batch 是否存在非 `-100` label：

```python
if (shift_labels != -100).any():
    ...
```

所以：

- 全 silent batch 不会计算 `lm_loss`。
- 混合 batch 中，silent 样本的 token label 全是 `-100`，不会贡献 LM loss。
- interrupt 样本只在 assistant 自然语言内容 token 上贡献 LM loss。

### 3.2 Gate loss

每个样本会带一个 `gate_position`。模型从最后一层 hidden states 中取出该位置：

```python
hidden_states = outputs.hidden_states[-1]
gate_states = hidden_states[batch_indices, gate_positions]
```

然后用 gate head 生成一个 logit：

```python
gate_logits = self.gate_head(gate_states.float()).squeeze(-1)
```

gate label 是二值：

```text
0 = silent
1 = interrupt
```

gate loss 是二分类 BCE with logits：

```python
gate_loss = BCEWithLogitsLoss(pos_weight=pos_weight)(
    gate_logits.float(),
    gate_labels,
)
```

如果 `gate_pos_weight > 0`，则作为 BCE 的正类权重，用于提高 interrupt 正样本的损失权重；否则不启用 `pos_weight`。

数学形式可以写为：

```text
p_i = sigmoid(z_i)

gate_loss_i =
  - [ pos_weight * y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
```

其中：

- `z_i` 是第 `i` 个样本的 gate logit。
- `y_i` 是 gate label。
- `pos_weight` 只作用在 `y_i=1` 的 interrupt 样本上。

### 3.3 总 loss

模型把可用 loss 加权求和：

```python
losses = []
if lm_loss is not None:
    losses.append(lm_weight * lm_loss)
if gate_loss is not None:
    losses.append(gate_weight * gate_loss)
loss = sum(losses)
```

其中：

- `lm_weight = config.lm_loss_weight`，除非 `forward()` 显式传入 `lm_loss_weight`。
- `gate_weight = config.gate_loss_weight`，除非 `forward()` 显式传入 `gate_loss_weight`。

常见情况：

| 样本类型 | gate_loss | lm_loss | 训练含义 |
| --- | --- | --- | --- |
| silent | 有 | 无 | 只训练 gate 判断不要说话 |
| interrupt | 有 | 有 | 同时训练 gate 判断要说话，以及 LLM 生成具体指令 |
| silent + interrupt 混合 batch | 有 | 有 | BCE 覆盖全 batch，CE 只覆盖 interrupt token |

## 4. 与 chat finetune 的本质区别

chat finetune 的训练目标是：

```text
给定图像/视频和 prompt，直接学习 assistant 文本输出。
```

gate finetune 的训练目标拆成两层：

```text
第一层：gate head 判断当前时刻是否需要说话。
第二层：如果需要说话，LLM 生成短的、及时的、可执行指令。
```

因此，`livestar_gate_finetune.py` 相比 `livestar_chat_finetune.py` 的核心变化不是简单多了一个分类头，而是把原来的 `$silent$/$interrupt$` 文本监督改成了：

- `$silent$/$interrupt$` 由 gate BCE loss 学习。
- interrupt 后面的自然语言内容由 LM CE loss 学习。
- silent 样本不再污染语言模型，让模型学习生成 `$silent$` 字符串。

这更符合推理时的两阶段决策方式：先判断是否打断，再决定要说什么。

## 5. 关键代码位置

- `livestar/train/livestar_gate_finetune.py`
  - `save_gate_head()`：保存 gate head 和 gate config。
  - `sanitize_gate_conversations()`：清理 `$silent$/$interrupt$` 控制符。
  - `preprocess_internvl2_5_gate()`：构造 LM labels、gate labels、gate positions。
  - `LazySupervisedDataset.get_gate_label()`：推断二分类标签。
  - `GateTrainer`：训练日志中记录 `lm_loss`、`gate_loss`、`gate_prob`。
  - `main()`：加载 gate 模型、禁用 packed dataset、初始化 gate head、设置 gate 参数可训练、保存 gate head。

- `livestar/model/livestar_gate/modeling_livestar_chat.py`
  - `gate_head`：新增二分类头。
  - `forward()`：构造 LM loss、gate loss 和总 loss。
  - `GateCausalLMOutputWithPast`：额外返回 gate 相关输出。

- `livestar/model/livestar_gate/configuration_livestar_chat.py`
  - gate 相关配置项：`gate_hidden_size`、`gate_dropout`、`gate_loss_weight`、`lm_loss_weight`、`gate_pos_weight`。

