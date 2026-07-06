# ===================================================================
# LiveStar - Live Streaming Assistant 
# for Real-World Online Video Understanding
# ===================================================================
# Modified from: InternVL (Original Copyright (c) 2024 OpenGVLab)
# Licensed under The MIT License [see LICENSE for details]
# ===================================================================

import logging
import math
import os
import random
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Literal, Optional
import time
import numpy as np
import torch.nn.functional as F

try:
    import orjson as json
except:
    import json

import torch
import torch.distributed as dist
import transformers
from livestar.dist_utils import init_dist
from livestar.model.livestarllm.modeling_livestar_llm import InternLM2ForCausalLM
from livestar.conversation import get_conv_template
from livestar.model.livestar_gate import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
from livestar.patch import (concat_pad_data_collator,
                            replace_internlm2_attention_class,
                            replace_llama_attention_class,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_phi3_attention_class,
                            replace_qwen2_attention_class,
                            replace_train_dataloader, replace_train_sampler)
from livestar.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from livestar.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    check_conversations_repetition,
                                    dynamic_preprocess)
from livestar.train.dataset_packed import PackedDataset, packed_collate_fn
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          set_seed)
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)


def patch_accelerate_dataloader_args():
    """Keep Transformers 4.37 Trainer compatible with newer Accelerate.

    Transformers 4.37 passes dataloader kwargs such as ``dispatch_batches``
    directly to ``Accelerator``. Accelerate 1.x moved them under
    ``DataLoaderConfiguration``. This shim preserves the old Trainer call site
    without changing training behavior.
    """
    try:
        import inspect
        from accelerate import Accelerator
        from accelerate.utils import DataLoaderConfiguration
    except Exception:
        return

    signature = inspect.signature(Accelerator.__init__)
    if "dispatch_batches" in signature.parameters:
        return

    original_init = Accelerator.__init__

    def compatible_init(self, *args, **kwargs):
        dataloader_keys = {
            "split_batches",
            "dispatch_batches",
            "even_batches",
            "use_seedable_sampler",
            "data_seed",
        }
        dataloader_kwargs = {
            key: kwargs.pop(key) for key in list(kwargs.keys()) if key in dataloader_keys
        }
        if dataloader_kwargs and kwargs.get("dataloader_config") is None:
            kwargs["dataloader_config"] = DataLoaderConfiguration(**dataloader_kwargs)
        return original_init(self, *args, **kwargs)

    Accelerator.__init__ = compatible_init


def set_peft_base_model_path(module, base_model_name_or_path: str) -> None:
    if not hasattr(module, "peft_config"):
        return
    for peft_config in module.peft_config.values():
        peft_config.base_model_name_or_path = base_model_name_or_path


def save_gate_head(model, output_dir: str, data_args=None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    gate_state = {
        key: value.detach().cpu()
        for key, value in model.gate_head.state_dict().items()
    }
    try:
        from safetensors.torch import save_file
        save_file(gate_state, os.path.join(output_dir, 'gate_head.safetensors'))
    except ImportError:
        torch.save(gate_state, os.path.join(output_dir, 'gate_head.pt'))

    gate_config = {
        'gate_hidden_size': model.config.gate_hidden_size,
        'gate_dropout': model.config.gate_dropout,
        'gate_loss_weight': model.config.gate_loss_weight,
        'lm_loss_weight': model.config.lm_loss_weight,
        'gate_pos_weight': model.config.gate_pos_weight,
        'base_model_name_or_path': getattr(model.config, '_name_or_path', ''),
    }
    if data_args is not None:
        gate_config.update(
            {
                'conv_style': data_args.conv_style,
                'max_seq_length': data_args.max_seq_length,
            }
        )
    with open(os.path.join(output_dir, 'gate_config.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(gate_config, ensure_ascii=False, indent=2) + '\n')


# Try to import petrel_client for image loading, fallback to PIL if unavailable
try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

EGOPROACTIVE_DECISION_INSTRUCTION = (
    "Decide whether it is useful to speak at the current moment. "
    "If speaking is useful, provide one short, timely, actionable instruction. "
    "If no timely help is needed, do not produce an instruction."
)


def strip_egoproactive_prefix(text: str) -> str:
    text = (text or "").strip()
    lowered = text.lower()
    if lowered.startswith("$interrupt$"):
        return text[len("$interrupt$") :].strip()
    if lowered.startswith("$silent$"):
        return ""
    return text


def sanitize_prompt_text(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        lowered = stripped.lower()
        if lowered.startswith("$silent$"):
            continue
        if lowered.startswith("$interrupt$"):
            lines.append(indent + strip_egoproactive_prefix(stripped))
            continue

        cleaned = None
        for role in ("assistant:", "user:"):
            if lowered.startswith(role):
                content = stripped[len(role) :].lstrip()
                content_lowered = content.lower()
                if content_lowered.startswith("$silent$"):
                    cleaned = None
                    break
                if content_lowered.startswith("$interrupt$"):
                    cleaned = stripped[: len(role)] + " " + strip_egoproactive_prefix(content)
                    break
        if cleaned is not None:
            lines.append(indent + cleaned.rstrip())
        elif not any(lowered.startswith(role) and stripped[len(role) :].lstrip().lower().startswith("$silent$")
                     for role in ("assistant:", "user:")):
            lines.append(line)
    return "\n".join(lines)


def sanitize_gate_conversations(conversations):
    conversations = deepcopy(conversations)
    if not conversations:
        return conversations

    if conversations[-1].get("from") == "gpt":
        conversations[-1]["value"] = strip_egoproactive_prefix(conversations[-1].get("value", ""))

    for turn in conversations:
        if turn.get("from") == "human":
            value = sanitize_prompt_text(turn.get("value", ""))
            marker = "You are now at the end of the current video chunk."
            if marker in value:
                value = value[: value.find(marker)].rstrip()
                value = value + "\n\n" + EGOPROACTIVE_DECISION_INSTRUCTION
            value = value.replace("Output exactly `$silent$`", "Do not output an instruction")
            value = value.replace("start with `$interrupt$` and then give", "give")
            turn["value"] = value
    return conversations


def preprocess_internvl2_5_gate(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        gate_label: int,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1,
) -> Dict:
    assert len(sources) == 1, 'process only the first conversations'
    conversations = sanitize_gate_conversations(sources[0])

    if conversations[0]['from'] == 'system':
        system_prompt = conversations[0]['value']
        conversations = conversations[1:]
    else:
        conv = get_conv_template(template_name)
        system_prompt = conv.system_message

    if not text_only:
        new_conversations = []
        current_image_idx = 0
        for conversation in conversations:
            if conversation['from'] == 'human':
                image_cnt = conversation['value'].count('<image>')
                for _ in range(image_cnt):
                    if current_image_idx == num_image:
                        break
                    image_tokens = (
                        f'{IMG_START_TOKEN}'
                        f'{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}'
                        f'{IMG_END_TOKEN}'
                    )
                    conversation['value'] = conversation['value'].replace('<image>', image_tokens, 1)
                    current_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
        assert current_image_idx == num_image, f'{current_image_idx} != {num_image}'

    batches, roles = [], []
    if system_prompt is not None:
        batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
        roles.append('system')
    for conversation in conversations:
        if conversation['from'] == 'human':
            batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
            roles.append('human')
        elif conversation['from'] == 'gpt':
            batches.append(f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n')
            roles.append('gpt')
        else:
            raise NotImplementedError

    add_bos_token = getattr(tokenizer, 'add_bos_token', False)
    if add_bos_token:
        batches[0] = tokenizer.bos_token + batches[0]

    input_ids = tokenizer(
        batches,
        return_tensors='np',
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=False,
    ).input_ids

    if add_bos_token:
        input_ids = [item[1:] for item in input_ids]

    final_input_ids, final_targets = [], []
    gate_position = 0
    ignore_ids = tokenizer('<|im_start|>assistant\n', return_tensors='np').input_ids[0]
    ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
    cursor = 0
    for role, input_id in zip(roles, input_ids):
        final_input_ids.append(input_id)
        if role == 'system' or role == 'human':
            final_targets.append(np.full(input_id.shape, IGNORE_INDEX))
        elif role == 'gpt':
            gate_position = max(0, cursor - 1)
            target = np.full(input_id.shape, IGNORE_INDEX)
            if gate_label == 1:
                target = input_id.copy()
                target[:ignore_len] = IGNORE_INDEX
                target[-1:] = IGNORE_INDEX
            final_targets.append(target)
        else:
            raise NotImplementedError
        cursor += input_id.shape[0]

    input_ids = torch.tensor(np.concatenate(final_input_ids))[:tokenizer.model_max_length]
    targets = torch.tensor(np.concatenate(final_targets))[:tokenizer.model_max_length]
    if gate_position >= input_ids.size(0):
        raise ValueError(
            f'Gate position {gate_position} was truncated by max_seq_length={tokenizer.model_max_length}. '
            'Increase MAX_SEQ_LENGTH or reduce frame/history length.'
        )

    padding = False if group_by_length or use_packed_ds else True
    if padding:
        current_length = input_ids.size(0)
        padding_length = tokenizer.model_max_length - current_length
        input_ids = F.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
        targets = F.pad(targets, (0, padding_length), value=IGNORE_INDEX)

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        gate_labels=torch.tensor([gate_label], dtype=torch.float),
        gate_positions=torch.tensor([gate_position], dtype=torch.long),
    )


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the ViT. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP. Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of LLM. Default is False.'},
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},
    )
    ps_version: Literal['v1', 'v2'] = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is v2.'}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'}
    )
    use_liger: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the liger kernel.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True. Default is False.'},
    )
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling. Default is False.'},
    )
    dynamic_image_size: bool = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic high resolution strategy. Default is False.'},
    )
    use_thumbnail: bool = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image. Default is False.'},
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video data. Default is 8.'},
    )
    max_num_frame: int = field(
        default=32,
        metadata={'help': 'The maximum number of frames for video data. Default is 32.'},
    )
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(
        default='imagenet',
        metadata={'help': 'The normalization type for the image. Default is imagenet.'},
    )
    use_packed_ds: bool = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for efficient training. Default is False.'},
    )
    num_images_expected: int = field(
        default=40,
        metadata={'help': 'The maximum number of images per packed sample. Default is 40.'},
    )
    max_packed_tokens: int = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: int = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: int = field(
        default=1000,
        metadata={'help': 'The log frequency of the packed dataset. Default is 1000.'},
    )
    strict_mode: bool = field(
        default=True,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: bool = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: bool = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    loss_reduction: str = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is token.'},
    )
    loss_reduction_all_gather: bool = field(
        default=False,
        metadata={'help': 'Whether to gather all during loss reduction. Default is False.'},
    )
    gate_loss_weight: float = field(
        default=1.0,
        metadata={'help': 'Weight for the silent/interrupt gate BCE loss.'},
    )
    lm_loss_weight: float = field(
        default=1.0,
        metadata={'help': 'Weight for the interrupt-only language modeling loss.'},
    )
    gate_pos_weight: float = field(
        default=0.0,
        metadata={'help': 'Optional positive class weight for gate BCE loss. 0 disables weighting.'},
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        min_num_frame=8,  # for video data
        max_num_frame=32,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        # hyperparameters for packed training
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        # TODO: quick resume
        self._state_dict = {}

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'internvl2_5':
            return preprocess_internvl2_5_gate
        raise ValueError(
            f'Gate fine-tuning currently supports conv_style=internvl2_5 only, '
            f'but got {self.template_name}.'
        )

    def get_gate_label(self, data_item):
        label = str(data_item.get('label', '')).strip().lower()
        if label == 'silent':
            return 0
        if label == 'interrupt':
            return 1
        conversations = data_item.get('conversations', [])
        answer = str(conversations[-1].get('value', '') if conversations else '').strip().lower()
        return 0 if answer.startswith('$silent$') or not answer else 1

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        gate_label = self.get_gate_label(data_item)
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name,
                                  gate_label=gate_label)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            gate_labels=ret['gate_labels'][0],
            gate_positions=ret['gate_positions'][0],
        )
        return ret

    # def multi_modal_multi_image_get_item(self, data_item):
    #     # Build transformation function
    #     transform = self.get_transform()

    #     images, num_tiles = [], []
    #     num_image = len(data_item['image'])
    #     for image_path in data_item['image']:
    #         # Merge the image path
    #         image_path = self.get_image_path(image_path)
    #         # Load the image using tcs_loader if available, otherwise use PIL
    #         image = self.load_image(image_path)
    #         if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
    #             image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
    #                                        max_num=max(1, self.max_dynamic_patch // num_image),
    #                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
    #             images += image
    #             num_tiles.append(len(image))
    #         else:  # Otherwise, use the original image as a single patch
    #             images.append(image)
    #             num_tiles.append(1)
    #     pixel_values = [transform(image) for image in images]
    #     pixel_values = torch.stack(pixel_values)
    #     num_patches = pixel_values.size(0)

    #     # Select the appropriate preprocessing function based on the template name
    #     preprocess_function = self.get_preprocess_function()

    #     # Preprocess the conversations and generate the return dictionary
    #     num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
    #     ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
    #                               self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
    #                               use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_image)

    #     # Calculate position_ids for packed dataset
    #     position_ids = ret['attention_mask'].long().cumsum(-1) - 1
    #     position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
    #     image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    #     assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f"image tokens are truncated, this dataset is {self.ds_name}. {(ret['input_ids'][0] == image_end_token_id).sum()} != {num_image}, {data_item['image']}"

    #     # Create the final return dictionary
    #     ret = dict(
    #         input_ids=ret['input_ids'][0],
    #         labels=ret['labels'][0],
    #         attention_mask=ret['attention_mask'][0],
    #         position_ids=position_ids[0],
    #         pixel_values=pixel_values,
    #         image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
    #     )
    #     return ret
    
    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=max(1, self.max_dynamic_patch // num_image),
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        gate_label = self.get_gate_label(data_item)
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name,
                                  num_image=num_image, gate_label=gate_label)
         
        # Calculate position_ids for packed dataset
        # for i in range(len(ret['input_ids'])):
        #     seq = ret['input_ids'][i]
        #     start_indices = []
        #     end_indices = []
        #     # Collect all indices
        #     stack = []
        #     for idx, token in enumerate(seq):
        #         if token == 92544:
        #             if stack:
        #                 start_indices.append(idx)
        #                 end_indices.append(stack.pop())
                        
        #         elif token == 92545 and idx + 2 < len(seq) and seq[idx + 2] == 92542:
        #             stack.append(idx + 1)

        #     # Process each end index to find the next start index and mask the interval            
        #     for start, end in zip(start_indices, end_indices):
        #         for j in range(end + 1, start - 8):
        #             if j < len(ret['attention_mask'][i]): 
        #                 ret['attention_mask'][i][j] = False
        # mask = ret['labels'] != -100
        # ret['attention_mask'][mask] = False   

        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        
        # position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        # position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f"image tokens are truncated, this dataset is {self.ds_name}. {(ret['input_ids'][0] == image_end_token_id).sum()} != {num_image}, {data_item['image']}"
        
        # with open('/mnt/workspace/xxx/xxx/ret_2bsz.txt', 'a') as f:
        #     f.write(f"--------------- ret ---------------\n")
        #     f.write(f"input_ids shape: {ret['input_ids'].shape}\n")
        #     f.write(f"attention_mask shape: {ret['attention_mask'].shape}\n")
        #     f.write(f"position_ids shape: {position_ids.shape}\n")
        #     f.write(f"input_ids: {ret['input_ids'].tolist()}\n")
        #     f.write(f"attention_mask: {ret['attention_mask'].tolist()}\n")
        #     f.write(f"position_ids: {position_ids}\n")

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            gate_labels=ret['gate_labels'][0],
            gate_positions=ret['gate_positions'][0],
        )
        return ret

    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains a video placeholder
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        # Get the video file path
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)

        # Load the video frames using tcs_loader
        # TODO: Load videos without using tcsloader.
        image_list = self.tcs_loader(
            video_path,
            image_type='video',
            max_num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=data_item.get('clip', None))

        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens + '\n')

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        gate_label = self.get_gate_label(data_item)
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name,
                                  num_image=num_patches, gate_label=gate_label)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            gate_labels=ret['gate_labels'][0],
            gate_positions=ret['gate_positions'][0],
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        gate_label = self.get_gate_label(data_item)
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name, gate_label=gate_label)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
            gate_labels=ret['gate_labels'][0],
            gate_positions=ret['gate_positions'][0],
        )
        return ret

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                # conversations = data_item['conversations']
                # check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10)
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name, flush=True)
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=}'
            )

        for i in range(start_idx, len(self)):
            yield self[i]


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    min_num_frame=8,
    max_num_frame=32,
    normalize_type='imagenet',
):
    datasets = []
    lengths = []
    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style, ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length and not data_args.use_packed_ds,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            min_num_frame=min_num_frame,
            max_num_frame=max_num_frame,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            # hyperparameters for packed training
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
        )
        # tmp = dataset[0]
        # import pdb
        # pdb.set_trace()
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
        )
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)


class GateTrainer(Trainer):
    @staticmethod
    def _finite_scalar(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().float()
            if not torch.isfinite(value).all():
                return None
            return value.mean().item()
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def log(self, logs, *args, **kwargs):
        is_train_step_log = 'loss' in logs and 'train_runtime' not in logs
        if is_train_step_log:
            model = self.model
            if hasattr(model, 'module'):
                model = model.module
            lm_loss = self._finite_scalar(getattr(model, '_last_lm_loss', None))
            gate_loss = self._finite_scalar(getattr(model, '_last_gate_loss', None))
            gate_prob = self._finite_scalar(getattr(model, '_last_gate_prob', None))
            if lm_loss is not None:
                logs['lm_loss'] = round(lm_loss, 6)
            if gate_loss is not None:
                logs['gate_loss'] = round(gate_loss, 6)
            if gate_prob is not None:
                logs['gate_prob'] = round(gate_prob, 6)
        super().log(logs, *args, **kwargs)


class GateProgressCallback(ProgressCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or self.training_bar is None or logs is None:
            return

        postfix = {}
        if 'loss' in logs:
            postfix['train_loss'] = logs['loss']
        if 'gate_prob' in logs:
            postfix['gate_prob'] = logs['gate_prob']
        if postfix:
            self.training_bar.set_postfix(postfix, refresh=False)


def main():
    patch_accelerate_dataloader_args()

    # Apply necessary patches for the transformers library
    replace_llama_rmsnorm_with_fused_rmsnorm()
    replace_train_sampler()
    replace_train_dataloader()

    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.use_packed_ds:
        raise ValueError('Gate fine-tuning does not support --use_packed_ds yet.')

    training_args.use_packed_ds = data_args.use_packed_ds
    training_args.logging_nan_inf_filter = False

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=model_args.use_fast_tokenizer)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = TCSLoader('~/petreloss.conf') if has_tcs_loader else None

    if data_args.use_packed_ds:
        replace_internlm2_attention_class()
        replace_qwen2_attention_class()
        replace_phi3_attention_class()
        replace_llama_attention_class()

    if model_args.use_liger:
        from livestar.patch import apply_liger_kernel_to_internvit
        from liger_kernel.transformers import (apply_liger_kernel_to_llama,
                                               apply_liger_kernel_to_qwen2)
        apply_liger_kernel_to_llama()
        apply_liger_kernel_to_qwen2()
        # apply_liger_kernel_to_internvit()

    if model_args.model_name_or_path is not None:
        logger.info('Loading InternVLGateModel...')
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
            # config.llm_config.attn_implementation = 'eager'  # for InternLM
            # logger.info('Using attention for InternLM')
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        config.gate_loss_weight = data_args.gate_loss_weight
        config.lm_loss_weight = data_args.lm_loss_weight
        config.gate_pos_weight = data_args.gate_pos_weight
        model = InternVLChatModel.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config, use_flash_attn=False)
    else:
        logger.info('Loading ViT-6B...')
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        vision_model = InternVisionModel.from_pretrained(
            model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
        logger.info('Loading LLaMA...')
        llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
        if llm_config.model_type == 'internlm2':
            model_type = InternLM2ForCausalLM
            llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        llm = model_type.from_pretrained(
            model_args.llm_path, torch_dtype=torch.bfloat16,
            config=llm_config, trust_remote_code=True)
        logger.info('Building InternVLGateConfig...')
        internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(), llm_config.to_dict(), downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square, template=data_args.conv_style,
            select_layer=model_args.vision_select_layer, dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail, ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch,
            gate_loss_weight=data_args.gate_loss_weight, lm_loss_weight=data_args.lm_loss_weight,
            gate_pos_weight=data_args.gate_pos_weight)
        internvl_chat_config.force_image_size = data_args.force_image_size
        logger.info('Building InternVLGateModel...')
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)
    model.img_context_token_id = img_context_token_id
    model._init_gate_head()
    model.gate_head.float()

    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info('Finished')

    patch_size = model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size,
                                                 patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    # model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))
    model.num_image_token = 16
    
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    train_dataset = build_datasets(
        data_args, tokenizer, tcs_loader, model, group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type, min_num_frame=data_args.min_num_frame,
        max_num_frame=data_args.max_num_frame)

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        set_peft_base_model_path(model.vision_model, model_args.model_name_or_path)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        set_peft_base_model_path(model.language_model, model_args.model_name_or_path)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    for param in model.gate_head.parameters():
        param.requires_grad = True

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    if data_args.use_packed_ds:
        collator = partial(
            packed_collate_fn,
            data_collator=concat_pad_data_collator,
            max_item_length=data_args.max_packed_tokens if data_args.strict_mode else 0,
            micro_num=training_args.train_batch_size,
            len2weight=partial(len2weight, loss_reduction=data_args.loss_reduction),
            loss_reduction_all_gather=data_args.loss_reduction_all_gather,
        )
    else:
        collator = concat_pad_data_collator

    trainer = GateTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(PrinterCallback)
    if not training_args.disable_tqdm:
        trainer.add_callback(GateProgressCallback)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        # import pdb
        # pdb.set_trace()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        use_lora = bool(model_args.use_llm_lora or model_args.use_backbone_lora)
        if use_lora:
            if dist.get_rank() == 0:
                os.makedirs(training_args.output_dir, exist_ok=True)
                if model_args.use_llm_lora and hasattr(model.language_model, 'save_pretrained'):
                    if model_args.use_backbone_lora:
                        model.language_model.save_pretrained(os.path.join(training_args.output_dir, 'llm'))
                    else:
                        model.language_model.save_pretrained(training_args.output_dir)
                if model_args.use_backbone_lora and hasattr(model.vision_model, 'save_pretrained'):
                    if model_args.use_llm_lora:
                        model.vision_model.save_pretrained(os.path.join(training_args.output_dir, 'vision'))
                    else:
                        model.vision_model.save_pretrained(training_args.output_dir)
                save_gate_head(model, training_args.output_dir, data_args)
        else:
            trainer.save_model()  # Saves the tokenizer too for easy upload
            if dist.get_rank() == 0:
                save_gate_head(model, training_args.output_dir, data_args)

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


if __name__ == '__main__':
    main()
