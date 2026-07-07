import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/data1/wearable_ai_challenge_data/egoproactive")
DEFAULT_ANN_FILE = DEFAULT_DATA_ROOT / "wearable_ai_2026_egoproactive_val_700.jsonl"
DEFAULT_VIDEO_DIR = DEFAULT_DATA_ROOT / "val"
DEFAULT_MODEL_PATH = REPO_ROOT / "inference"
DEFAULT_WEIGHTS_DIR = Path("/data1/LiveStar_8B")
DEFAULT_FRAMES_PER_INTERVAL = 4
DEFAULT_MAX_CONTEXT_INTERVALS = 20
DEFAULT_MAX_CONTEXT_FRAMES = 0
DEFAULT_MAX_HISTORY_TURNS = 20
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "evaluate"
    / "output"
    / (
        "egoproactive_sved_sample350_"
        f"fpi{DEFAULT_FRAMES_PER_INTERVAL}_"
        f"ctxi{DEFAULT_MAX_CONTEXT_INTERVALS}_"
        f"f{DEFAULT_FRAMES_PER_INTERVAL * DEFAULT_MAX_CONTEXT_INTERVALS}_"
        f"hist{DEFAULT_MAX_HISTORY_TURNS}_majority1.jsonl"
    )
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PROACTIVE_TASK_PROMPT = (
    "You are a proactive wearable AI assistant. The user asks: {query}\n"
    "I will provide first-person video frames sequentially. Watch the user's "
    "current progress and respond only when timely guidance is useful. When "
    "you respond, give one concise instruction for the next useful action. "
    "Avoid repeating guidance that was already given.\n"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample egoproactive videos and evaluate LiveStar SVeD interrupt/silent decisions."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ann-file", type=Path, default=DEFAULT_ANN_FILE)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.06)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--check-len", type=int, default=1000)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--max-num", type=int, default=1)
    parser.add_argument("--frames-per-interval", type=int, default=DEFAULT_FRAMES_PER_INTERVAL)
    parser.add_argument("--max-context-intervals", type=int, default=DEFAULT_MAX_CONTEXT_INTERVALS)
    parser.add_argument("--max-context-frames", type=int, default=DEFAULT_MAX_CONTEXT_FRAMES)
    parser.add_argument("--max-history-turns", type=int, default=DEFAULT_MAX_HISTORY_TURNS)
    parser.add_argument("--clear-cache-per-video", dest="clear_cache_per_video", action="store_true", default=True)
    parser.add_argument("--no-clear-cache-per-video", dest="clear_cache_per_video", action="store_false")
    parser.add_argument("--max-intervals-per-video", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--worker-input", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=-1)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(path, base=None):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (base or Path.cwd()) / path


def label_from_answer(answer):
    if answer.startswith("$interrupt$"):
        return "interrupt"
    if answer.startswith("$silent$"):
        return "silent"
    return "unknown"


def load_records(ann_file, video_dir, max_intervals_per_video=0):
    records = []
    skipped = Counter()
    with ann_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            data = json.loads(line)
            intervals = data.get("video_intervals", [])
            answers = data.get("answers", [])
            dialogs = data.get("dialog", [])
            if not (len(intervals) == len(answers) == len(dialogs)):
                skipped["length_mismatch"] += 1
                continue

            labels = [label_from_answer(answer) for answer in answers]
            if any(label == "unknown" for label in labels):
                skipped["unknown_label"] += 1
                continue

            video_path = video_dir / data["video_path"]
            if not video_path.exists():
                skipped["missing_video"] += 1
                continue

            if max_intervals_per_video > 0:
                intervals = intervals[:max_intervals_per_video]
                answers = answers[:max_intervals_per_video]
                labels = labels[:max_intervals_per_video]

            records.append(
                {
                    "line_no": line_no,
                    "video_path": data["video_path"],
                    "video_file": str(video_path),
                    "duration_in_sec": data.get("duration_in_sec"),
                    "video_intervals": intervals,
                    "query": data.get("query", ""),
                    "domain": data.get("domain", ""),
                    "task": data.get("task", ""),
                    "gt_answers": answers,
                    "gt_labels": labels,
                }
            )

    if skipped:
        print("Skipped records:", dict(skipped))
    return records


def sample_records(records, num_samples, seed):
    if num_samples <= 0 or num_samples >= len(records):
        return list(records)
    rng = random.Random(seed)
    return rng.sample(records, num_samples)


def attach_sample_order(records):
    for idx, record in enumerate(records):
        record["sample_order"] = idx
    return records


def parse_gpus(gpus):
    return [gpu.strip() for gpu in str(gpus).split(",") if gpu.strip()]


def estimate_work(record, frames_per_interval):
    return len(record["video_intervals"]) * max(1, int(frames_per_interval))


def shard_records(records, num_shards, frames_per_interval):
    shards = [{"records": [], "work": 0} for _ in range(num_shards)]
    ordered = sorted(
        records,
        key=lambda item: estimate_work(item, frames_per_interval),
        reverse=True,
    )
    for record in ordered:
        shard = min(shards, key=lambda item: item["work"])
        shard["records"].append(record)
        shard["work"] += estimate_work(record, frames_per_interval)
    for shard in shards:
        shard["records"].sort(key=lambda item: item.get("sample_order", 0))
    return shards


def write_records_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def read_records_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def has_model_weights(model_dir):
    model_dir = Path(model_dir)
    patterns = ("*.safetensors", "pytorch_model*.bin", "*.ckpt")
    return any(any(model_dir.glob(pattern)) for pattern in patterns)


def symlink_file(src, dst):
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src.resolve())


def prepare_model_dir(model_path, weights_dir):
    model_path = Path(model_path).resolve()
    weights_dir = Path(weights_dir).resolve()
    if has_model_weights(model_path):
        return str(model_path), None

    if not weights_dir.exists() or not has_model_weights(weights_dir):
        raise FileNotFoundError(
            "No model weights found in --model-path or --weights-dir. "
            f"Checked model_path={model_path} weights_dir={weights_dir}"
        )

    tmp_ctx = tempfile.TemporaryDirectory(prefix="livestar_eval_model_")
    tmp_dir = Path(tmp_ctx.name)

    for src in model_path.iterdir():
        if src.is_file():
            symlink_file(src, tmp_dir / src.name)
    for src in weights_dir.iterdir():
        if src.is_file() and (
            src.suffix == ".safetensors"
            or src.name.startswith("pytorch_model")
            or src.name.endswith(".bin")
        ):
            symlink_file(src, tmp_dir / src.name)

    return str(tmp_dir), tmp_ctx


def build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        ),
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))

    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def interval_midpoint(interval):
    start, end = interval
    return max(0.0, (float(start) + float(end)) / 2.0)


def interval_sample_times(interval, frames_per_interval):
    start, end = interval
    start = max(0.0, float(start))
    end = max(start, float(end))
    frames_per_interval = max(1, int(frames_per_interval))
    if frames_per_interval == 1:
        return [interval_midpoint((start, end))]
    if end == start:
        return [start for _ in range(frames_per_interval)]
    span = end - start
    return [
        start + span * (sample_idx + 0.5) / frames_per_interval
        for sample_idx in range(frames_per_interval)
    ]


def load_interval_frames(video_path, intervals, input_size=448, max_num=1, frames_per_interval=1):
    import torch
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps())
    if fps <= 0:
        raise ValueError(f"Invalid FPS for {video_path}: {fps}")
    max_frame = len(vr) - 1
    transform = build_transform(input_size)

    pixel_values_list = []
    num_patches_list = []
    interval_samples = []
    for interval in intervals:
        sampled_times = interval_sample_times(interval, frames_per_interval)
        frame_indices = []
        frame_times = []
        for timestamp in sampled_times:
            frame_idx = min(max_frame, max(0, int(round(timestamp * fps))))
            image = Image.fromarray(vr[frame_idx].asnumpy()).convert("RGB")
            tiles = dynamic_preprocess(
                image, image_size=input_size, use_thumbnail=True, max_num=max_num
            )
            pixel_values = torch.stack([transform(tile) for tile in tiles])
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            frame_indices.append(frame_idx)
            frame_times.append(frame_idx / fps)
        interval_samples.append(
            {
                "target_times_sec": sampled_times,
                "frame_indices": frame_indices,
                "frame_times_sec": frame_times,
            }
        )

    return torch.cat(pixel_values_list), num_patches_list, interval_samples


def pixel_prefix(pixel_values, num_patches_list, end_sample):
    patch_count = sum(num_patches_list[:end_sample])
    return pixel_values[:patch_count]


def sample_patch_bounds(num_patches_list, sample_id):
    if sample_id < 1 or sample_id > len(num_patches_list):
        raise IndexError(
            f"sample_id={sample_id} is outside 1..{len(num_patches_list)}"
        )
    start = sum(num_patches_list[: sample_id - 1])
    end = start + num_patches_list[sample_id - 1]
    return start, end


def pixel_select(pixel_values, num_patches_list, sample_ids):
    import torch

    if not sample_ids:
        return pixel_values[:0], []

    chunks = []
    selected_num_patches = []
    for sample_id in sample_ids:
        start, end = sample_patch_bounds(num_patches_list, sample_id)
        chunks.append(pixel_values[start:end])
        selected_num_patches.append(num_patches_list[sample_id - 1])
    return torch.cat(chunks, dim=0), selected_num_patches


def make_frame_prompt(start_frame_number, count):
    return "".join(
        f"Frame-{frame_number}: <image>\n"
        for frame_number in range(start_frame_number, start_frame_number + count)
    )


def make_frame_prompt_for_ids(frame_numbers):
    return "".join(f"Frame-{frame_number}: <image>\n" for frame_number in frame_numbers)


def build_window_history(history_entries, current_sample_id, args, query):
    max_context_frames = int(getattr(args, "max_context_frames", 0) or 0)
    min_sample = 1
    if max_context_frames > 0:
        min_sample = max(1, current_sample_id - max_context_frames + 1)

    entries = []
    for entry in history_entries:
        frame_ids = [
            sample_id
            for sample_id in entry["frame_ids"]
            if min_sample <= sample_id <= current_sample_id
        ]
        if frame_ids:
            entries.append((frame_ids, entry["answer"]))

    max_history_turns = int(getattr(args, "max_history_turns", 0) or 0)
    if max_history_turns > 0:
        entries = entries[-max_history_turns:]

    history = []
    retained_sample_ids = []
    for entry_idx, (frame_ids, answer) in enumerate(entries):
        question = make_frame_prompt_for_ids(frame_ids)
        if entry_idx == 0:
            question = PROACTIVE_TASK_PROMPT.format(query=query) + question
        history.append((question, answer))
        retained_sample_ids.extend(frame_ids)

    return history, retained_sample_ids


def build_chat_inputs(
    pixel_values,
    num_patches_list,
    history_entries,
    current_sample_id,
    args,
    query,
    self_check=False,
):
    history, history_sample_ids = build_window_history(
        history_entries,
        current_sample_id,
        args,
        query,
    )

    if self_check:
        if not history_sample_ids or history_sample_ids[-1] != current_sample_id:
            raise RuntimeError(
                "self_check requires the current sampled frame to be present "
                "as the last history frame"
            )
        input_sample_ids = history_sample_ids
    else:
        input_sample_ids = history_sample_ids + [current_sample_id]

    selected_pixels, selected_num_patches = pixel_select(
        pixel_values,
        num_patches_list,
        input_sample_ids,
    )
    context_info = {
        "history_turns": len(history),
        "history_frames": len(history_sample_ids),
        "input_frames": len(input_sample_ids),
        "first_input_sample": input_sample_ids[0] if input_sample_ids else None,
        "last_input_sample": input_sample_ids[-1] if input_sample_ids else None,
    }
    return selected_pixels, selected_num_patches, history, context_info


def check_answer_text(text, max_chars):
    if not text:
        return " "
    return text[: min(max_chars, len(text))]


def average_perplexity(
    model,
    tokenizer,
    pixel_values,
    question,
    generation_config,
    num_patches_list,
    history,
    check_answer,
    self_check,
    num_runs,
):
    total = 0.0
    for _ in range(num_runs):
        ppl, _ = model.chat(
            tokenizer,
            pixel_values,
            question,
            dict(generation_config),
            num_patches_list=list(num_patches_list),
            history=list(history) if history is not None else None,
            return_history=False,
            check_answer=check_answer,
            self_check=self_check,
        )
        total += float(ppl)
    return total / max(num_runs, 1)


def generate_response(
    model,
    tokenizer,
    pixel_values,
    question,
    generation_config,
    num_patches_list,
    history,
):
    response, new_history, _ = model.chat(
        tokenizer,
        pixel_values,
        question,
        dict(generation_config),
        num_patches_list=list(num_patches_list),
        history=history,
        return_history=True,
    )
    return response, new_history


def is_oom_error(exc):
    message = str(exc).lower()
    return "out of memory" in message or exc.__class__.__name__ == "OutOfMemoryError"


def clear_cuda_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def make_error_result(record, args, error_type, exc):
    return {
        "sample_order": record.get("sample_order"),
        "shard_index": getattr(args, "shard_index", -1),
        "video_path": record["video_path"],
        "query": record["query"],
        "domain": record["domain"],
        "task": record["task"],
        "duration_in_sec": record["duration_in_sec"],
        "interval_accuracy": 0.0,
        "num_intervals": 0,
        "expected_num_intervals": len(record.get("video_intervals", [])),
        "interval_results": [],
        "error": True,
        "error_type": error_type,
        "error_message": str(exc)[:1000],
    }


def evaluate_record(record, model, tokenizer, generation_config, args):
    import torch

    pixel_values, num_patches_list, interval_samples = load_interval_frames(
        record["video_file"],
        record["video_intervals"],
        input_size=args.input_size,
        max_num=args.max_num,
        frames_per_interval=args.frames_per_interval,
    )
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)

    history_entries = []
    last_response = None
    threshold = None
    interval_results = []
    sample_cursor = 0

    for interval_idx, interval in enumerate(record["video_intervals"]):
        samples = interval_samples[interval_idx]
        sample_count = len(samples["frame_indices"])
        sample_results = []

        for local_sample_idx in range(sample_count):
            end_sample = sample_cursor + local_sample_idx + 1
            frame_prompt = make_frame_prompt(end_sample, 1)

            if threshold is None:
                cur_pixel_values, cur_num_patches = pixel_select(
                    pixel_values,
                    num_patches_list,
                    [end_sample],
                )
                question = PROACTIVE_TASK_PROMPT.format(query=record["query"]) + frame_prompt
                response, _ = generate_response(
                    model,
                    tokenizer,
                    cur_pixel_values,
                    question,
                    generation_config,
                    cur_num_patches,
                    history=None,
                )
                history_entries.append({"frame_ids": [end_sample], "answer": response})
                (
                    check_pixel_values,
                    check_num_patches,
                    check_history,
                    context_info,
                ) = build_chat_inputs(
                    pixel_values,
                    num_patches_list,
                    history_entries,
                    end_sample,
                    args,
                    record["query"],
                    self_check=True,
                )
                threshold = average_perplexity(
                    model,
                    tokenizer,
                    check_pixel_values,
                    frame_prompt,
                    generation_config,
                    check_num_patches,
                    check_history,
                    check_answer_text(response, args.check_len),
                    self_check=True,
                    num_runs=args.num_runs,
                )
                last_response = response
                sample_pred_label = "interrupt"
                ppl = threshold
                decision_threshold = threshold
            else:
                (
                    cur_pixel_values,
                    cur_num_patches,
                    cur_history,
                    context_info,
                ) = build_chat_inputs(
                    pixel_values,
                    num_patches_list,
                    history_entries,
                    end_sample,
                    args,
                    record["query"],
                    self_check=False,
                )
                ppl = average_perplexity(
                    model,
                    tokenizer,
                    cur_pixel_values,
                    frame_prompt,
                    generation_config,
                    cur_num_patches,
                    cur_history,
                    check_answer_text(last_response, args.check_len),
                    self_check=False,
                    num_runs=args.num_runs,
                )
                decision_threshold = threshold * args.alpha
                if ppl > decision_threshold:
                    response, _ = generate_response(
                        model,
                        tokenizer,
                        cur_pixel_values,
                        frame_prompt,
                        generation_config,
                        cur_num_patches,
                        cur_history,
                    )
                    history_entries.append({"frame_ids": [end_sample], "answer": response})
                    (
                        check_pixel_values,
                        check_num_patches,
                        check_history,
                        context_info,
                    ) = build_chat_inputs(
                        pixel_values,
                        num_patches_list,
                        history_entries,
                        end_sample,
                        args,
                        record["query"],
                        self_check=True,
                    )
                    threshold = average_perplexity(
                        model,
                        tokenizer,
                        check_pixel_values,
                        frame_prompt,
                        generation_config,
                        check_num_patches,
                        check_history,
                        check_answer_text(response, args.check_len),
                        self_check=True,
                        num_runs=args.num_runs,
                    )
                    last_response = response
                    sample_pred_label = "interrupt"
                else:
                    response = ""
                    if history_entries:
                        history_entries[-1]["frame_ids"].append(end_sample)
                    else:
                        history_entries.append({"frame_ids": [end_sample], "answer": ""})
                    sample_pred_label = "silent"

            sample_results.append(
                {
                    "sample_index": end_sample,
                    "local_sample_index": local_sample_idx,
                    "frame_index": samples["frame_indices"][local_sample_idx],
                    "time_sec": samples["frame_times_sec"][local_sample_idx],
                    "target_time_sec": samples["target_times_sec"][local_sample_idx],
                    "pred_label": sample_pred_label,
                    "ppl": ppl,
                    "decision_threshold": decision_threshold,
                    "active_threshold": threshold,
                    "generated_text": response,
                    "context_history_turns": context_info["history_turns"],
                    "context_history_frames": context_info["history_frames"],
                    "context_input_frames": context_info["input_frames"],
                    "context_first_sample_index": context_info["first_input_sample"],
                    "context_last_sample_index": context_info["last_input_sample"],
                }
            )

        gt_label = record["gt_labels"][interval_idx]
        interrupt_votes = sum(
            item["pred_label"] == "interrupt" for item in sample_results
        )
        pred_label = (
            "interrupt"
            if interrupt_votes >= 1
            else "silent"
        )
        generated_texts = [
            item["generated_text"]
            for item in sample_results
            if item["pred_label"] == "interrupt" and item["generated_text"]
        ]
        last_sample = sample_results[-1]
        frame_prompt_start = sample_cursor + 1
        sample_cursor += sample_count
        interval_results.append(
            {
                "interval_index": interval_idx,
                "interval": interval,
                "num_sampled_frames": sample_count,
                "sampled_frame_indices": samples["frame_indices"],
                "sampled_times_sec": samples["frame_times_sec"],
                "target_sample_times_sec": samples["target_times_sec"],
                "representative_frame_index": samples["frame_indices"][-1],
                "representative_time_sec": samples["frame_times_sec"][-1],
                "frame_prompt_start": frame_prompt_start,
                "frame_prompt_end": sample_cursor,
                "sample_results": sample_results,
                "interrupt_votes": interrupt_votes,
                "silent_votes": len(sample_results) - interrupt_votes,
                "gt_label": gt_label,
                "pred_label": pred_label,
                "correct": pred_label == gt_label,
                "ppl": last_sample["ppl"],
                "decision_threshold": last_sample["decision_threshold"],
                "active_threshold": last_sample["active_threshold"],
                "generated_text": " | ".join(generated_texts),
                "gt_answer": record["gt_answers"][interval_idx],
            }
        )

    correct = sum(item["correct"] for item in interval_results)
    return {
        "sample_order": record.get("sample_order"),
        "shard_index": getattr(args, "shard_index", -1),
        "video_path": record["video_path"],
        "query": record["query"],
        "domain": record["domain"],
        "task": record["task"],
        "duration_in_sec": record["duration_in_sec"],
        "interval_accuracy": correct / len(interval_results) if interval_results else 0.0,
        "num_intervals": len(interval_results),
        "interval_results": interval_results,
        "error": False,
    }


def load_model_and_tokenizer(model_dir, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device is cuda, but CUDA is not available.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    model = model.half().to(device).to(torch.bfloat16).eval()
    return model, tokenizer


def update_counts(counts, result):
    if result.get("error"):
        return
    for item in result["interval_results"]:
        gt = item["gt_label"]
        pred = item["pred_label"]
        counts[(gt, pred)] += 1


def safe_div(num, den):
    return num / den if den else 0.0


def compute_metrics(counts):
    tp = counts[("interrupt", "interrupt")]
    fn = counts[("interrupt", "silent")]
    fp = counts[("silent", "interrupt")]
    tn = counts[("silent", "silent")]
    total = tp + fn + fp + tn

    interrupt_precision = safe_div(tp, tp + fp)
    interrupt_recall = safe_div(tp, tp + fn)
    silent_precision = safe_div(tn, tn + fn)
    silent_recall = safe_div(tn, tn + fp)
    return {
        "total": total,
        "accuracy": safe_div(tp + tn, total),
        "confusion": {
            "gt_interrupt_pred_interrupt": tp,
            "gt_interrupt_pred_silent": fn,
            "gt_silent_pred_interrupt": fp,
            "gt_silent_pred_silent": tn,
        },
        "interrupt": {
            "precision": interrupt_precision,
            "recall": interrupt_recall,
            "f1": safe_div(2 * interrupt_precision * interrupt_recall, interrupt_precision + interrupt_recall),
        },
        "silent": {
            "precision": silent_precision,
            "recall": silent_recall,
            "f1": safe_div(2 * silent_precision * silent_recall, silent_precision + silent_recall),
        },
    }


def print_metrics(metrics):
    print("\n== Overall Metrics ==")
    print(f"intervals: {metrics['total']}")
    print(f"accuracy : {metrics['accuracy']:.4f}")
    print("confusion:", metrics["confusion"])
    print(
        "interrupt: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}".format(
            **metrics["interrupt"]
        )
    )
    print(
        "silent   : precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}".format(
            **metrics["silent"]
        )
    )


def build_experiment_config(args, generation_config=None, gpus=None, shard_plan=None):
    return {
        "protocol": "generated-history SVeD interval classification",
        "sampling": {
            "frames_per_interval": args.frames_per_interval,
            "strategy": "uniform sub-interval centers",
            "decision_step": "one SVeD decision per sampled frame",
            "interval_aggregation": "majority=1; interrupt if at least 1 sampled frame interrupts, otherwise silent",
            "self_check_frame": "current sampled frame",
        },
        "context": {
            "max_context_intervals": args.max_context_intervals,
            "max_context_frames": args.max_context_frames,
            "max_history_turns": args.max_history_turns,
            "frame_selection": "retain the most recent sampled frames and rebuild matching history/image patches",
            "clear_cache_per_video": args.clear_cache_per_video,
        },
        "data": {
            "ann_file": str(args.ann_file),
            "video_dir": str(args.video_dir),
            "num_samples": args.num_samples,
            "seed": args.seed,
            "max_intervals_per_video": args.max_intervals_per_video,
        },
        "model": {
            "model_path": str(args.model_path),
            "weights_dir": str(args.weights_dir),
            "device": args.device,
            "gpus": gpus if gpus is not None else parse_gpus(args.gpus),
        },
        "execution": {
            "mode": "multi_gpu" if (gpus if gpus is not None else parse_gpus(args.gpus)) else "single_process",
            "shard_index": args.shard_index,
            "total_shards": args.total_shards,
            "shard_plan": shard_plan or [],
        },
        "sved": {
            "alpha": args.alpha,
            "num_runs": args.num_runs,
            "check_len": args.check_len,
        },
        "preprocess": {
            "input_size": args.input_size,
            "max_num": args.max_num,
        },
        "generation": generation_config or {},
    }


def dry_run(records, selected, args):
    label_counts = Counter(label for record in selected for label in record["gt_labels"])
    total_intervals = sum(len(record["video_intervals"]) for record in selected)
    gpus = parse_gpus(args.gpus)
    print("== Dry Run ==")
    print(f"usable records : {len(records)}")
    print(f"selected       : {len(selected)}")
    print(f"intervals      : {total_intervals}")
    print(f"frames/interval: {args.frames_per_interval}")
    print(f"max context intervals: {args.max_context_intervals}")
    print(f"max context frames: {args.max_context_frames}")
    print(f"max history turns : {args.max_history_turns}")
    print(f"sampled frames : {total_intervals * max(1, args.frames_per_interval)}")
    print(f"labels         : {dict(label_counts)}")
    if gpus:
        shards = shard_records(selected, len(gpus), args.frames_per_interval)
        print("shards:")
        for idx, shard in enumerate(shards):
            print(
                f"  shard {idx} gpu={gpus[idx]} records={len(shard['records'])} "
                f"estimated_sampled_frames={shard['work']}"
            )
    for idx, record in enumerate(selected, 1):
        print(
            f"{idx:02d}. {record['video_path']} | intervals={len(record['video_intervals'])} "
            f"| domain={record['domain']} | task={record['task']}"
        )


def make_generation_config(args):
    return {
        "temperature": 0.0,
        "max_new_tokens": args.max_new_tokens,
        "top_p": 0.1,
        "num_beams": 1,
        "repetition_penalty": 1.05,
    }


def run_worker(args, selected, experiment_config):
    generation_config = {
        "temperature": 0.0,
        "max_new_tokens": args.max_new_tokens,
        "top_p": 0.1,
        "num_beams": 1,
        "repetition_penalty": 1.05,
    }

    model_dir, tmp_ctx = prepare_model_dir(args.model_path, args.weights_dir)
    print(f"Model dir: {model_dir}")
    model, tokenizer = load_model_and_tokenizer(model_dir, args.device)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kwargs: x

    import torch

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    failed = Counter()
    with args.output.open("w", encoding="utf-8") as f_out, torch.no_grad():
        for record in tqdm(selected, desc="Evaluating egoproactive"):
            try:
                result = evaluate_record(record, model, tokenizer, generation_config, args)
            except Exception as exc:
                if not is_oom_error(exc):
                    raise
                clear_cuda_cache()
                result = make_error_result(record, args, "oom", exc)
                failed["oom"] += 1
                print(
                    f"{record['video_path']}: OOM skipped; "
                    f"{result['error_message'][:180]}"
                )
            finally:
                if args.clear_cache_per_video:
                    clear_cuda_cache()
            result["experiment_config"] = experiment_config
            update_counts(counts, result)
            json.dump(result, f_out, ensure_ascii=False)
            f_out.write("\n")
            f_out.flush()
            if not result.get("error"):
                print(
                    f"{record['video_path']}: interval_acc={result['interval_accuracy']:.4f} "
                    f"({result['num_intervals']} intervals)"
                )

    metrics = compute_metrics(counts)
    print_metrics(metrics)
    if failed:
        print(f"failed records: {dict(failed)}")
    print(f"\nSaved predictions to {args.output}")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()


def shard_plan_summary(shards, gpus):
    return [
        {
            "shard_index": idx,
            "gpu": gpus[idx],
            "records": len(shard["records"]),
            "estimated_sampled_frames": shard["work"],
        }
        for idx, shard in enumerate(shards)
    ]


def build_worker_cmd(args, shard_input, shard_output, shard_index, total_shards):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data-root",
        str(args.data_root),
        "--ann-file",
        str(args.ann_file),
        "--video-dir",
        str(args.video_dir),
        "--model-path",
        str(args.model_path),
        "--weights-dir",
        str(args.weights_dir),
        "--output",
        str(shard_output),
        "--num-samples",
        str(args.num_samples),
        "--seed",
        str(args.seed),
        "--alpha",
        str(args.alpha),
        "--num-runs",
        str(args.num_runs),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--check-len",
        str(args.check_len),
        "--input-size",
        str(args.input_size),
        "--max-num",
        str(args.max_num),
        "--frames-per-interval",
        str(args.frames_per_interval),
        "--max-context-intervals",
        str(args.max_context_intervals),
        "--max-context-frames",
        str(args.max_context_frames),
        "--max-history-turns",
        str(args.max_history_turns),
        "--max-intervals-per-video",
        str(args.max_intervals_per_video),
        "--device",
        args.device,
        "--gpus",
        "",
        "--worker-input",
        str(shard_input),
        "--shard-index",
        str(shard_index),
        "--total-shards",
        str(total_shards),
    ]
    if args.clear_cache_per_video:
        cmd.append("--clear-cache-per-video")
    else:
        cmd.append("--no-clear-cache-per-video")
    return cmd


def launch_multi_gpu(args, selected, gpus, generation_config):
    shards = shard_records(selected, len(gpus), args.frames_per_interval)
    plan = shard_plan_summary(shards, gpus)
    experiment_config = build_experiment_config(
        args, generation_config, gpus=gpus, shard_plan=plan
    )
    shard_dir = args.output.parent / f"{args.output.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    log_handles = []
    for idx, shard in enumerate(shards):
        shard_input = shard_dir / f"shard_{idx:02d}_input.jsonl"
        shard_output = shard_dir / f"shard_{idx:02d}_output.jsonl"
        shard_log = shard_dir / f"shard_{idx:02d}.log"
        write_records_jsonl(shard_input, shard["records"])
        cmd = build_worker_cmd(args, shard_input, shard_output, idx, len(shards))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus[idx]
        log_f = shard_log.open("w", encoding="utf-8")
        log_handles.append(log_f)
        print(
            f"Launching shard {idx} on GPU {gpus[idx]}: "
            f"{len(shard['records'])} records, {shard['work']} sampled frames"
        )
        processes.append(
            {
                "index": idx,
                "output": shard_output,
                "log": shard_log,
                "process": subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(REPO_ROOT),
                ),
            }
        )

    failed = []
    for item in processes:
        returncode = item["process"].wait()
        if returncode != 0:
            failed.append((item["index"], returncode, item["log"]))
    for handle in log_handles:
        handle.close()

    if failed:
        for shard_index, returncode, log_path in failed:
            print(f"Shard {shard_index} failed with code {returncode}. Log: {log_path}")
        raise RuntimeError("One or more evaluation shards failed.")

    merged = []
    for item in processes:
        for result in read_records_jsonl(item["output"]):
            result["experiment_config"] = experiment_config
            merged.append(result)
    merged.sort(key=lambda item: item.get("sample_order", 0))

    counts = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f_out:
        for result in merged:
            update_counts(counts, result)
            json.dump(result, f_out, ensure_ascii=False)
            f_out.write("\n")

    metrics = compute_metrics(counts)
    print_metrics(metrics)
    print(f"\nSaved merged predictions to {args.output}")
    print(f"Shard logs and outputs are in {shard_dir}")


def resolve_args(args):
    args.data_root = resolve_path(args.data_root)
    args.ann_file = resolve_path(args.ann_file, args.data_root)
    args.video_dir = resolve_path(args.video_dir, args.data_root)
    args.model_path = resolve_path(args.model_path, REPO_ROOT)
    args.weights_dir = resolve_path(args.weights_dir)
    args.output = resolve_path(args.output, REPO_ROOT)
    if args.max_context_frames <= 0 and args.max_context_intervals > 0:
        args.max_context_frames = (
            max(1, int(args.frames_per_interval)) * int(args.max_context_intervals)
        )
    if args.worker_input is not None:
        args.worker_input = resolve_path(args.worker_input, REPO_ROOT)
    return args


def main():
    args = resolve_args(parse_args())

    if args.worker_input is not None:
        selected = read_records_jsonl(args.worker_input)
    else:
        records = load_records(
            args.ann_file,
            args.video_dir,
            max_intervals_per_video=args.max_intervals_per_video,
        )
        selected = attach_sample_order(sample_records(records, args.num_samples, args.seed))

    if args.dry_run:
        dry_run(records if args.worker_input is None else selected, selected, args)
        return

    generation_config = make_generation_config(args)
    gpus = parse_gpus(args.gpus)
    if gpus and args.worker_input is None:
        launch_multi_gpu(args, selected, gpus, generation_config)
        return

    experiment_config = build_experiment_config(args, generation_config, gpus=gpus)
    run_worker(args, selected, experiment_config)


if __name__ == "__main__":
    main()
