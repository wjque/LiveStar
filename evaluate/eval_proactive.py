import argparse
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/data1/wearable_ai_challenge_data/egoproactive")
DEFAULT_ANN_FILE = DEFAULT_DATA_ROOT / "wearable_ai_2026_egoproactive_val_700.jsonl"
DEFAULT_VIDEO_DIR = DEFAULT_DATA_ROOT / "val"
DEFAULT_MODEL_PATH = REPO_ROOT / "inference"
DEFAULT_WEIGHTS_DIR = Path("/data1/LiveStar_8B")
DEFAULT_OUTPUT = REPO_ROOT / "evaluate" / "output" / "egoproactive_sved_sample10.jsonl"

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
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.06)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--check-len", type=int, default=1000)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--max-num", type=int, default=1)
    parser.add_argument("--max-intervals-per-video", type=int, default=0)
    parser.add_argument("--device", default="cuda")
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


def load_interval_frames(video_path, intervals, input_size=448, max_num=1):
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
    frame_indices = []
    frame_times = []
    for interval in intervals:
        timestamp = interval_midpoint(interval)
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

    return torch.cat(pixel_values_list), num_patches_list, frame_indices, frame_times


def pixel_prefix(pixel_values, num_patches_list, end_frame):
    patch_count = sum(num_patches_list[:end_frame])
    return pixel_values[:patch_count]


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


def evaluate_record(record, model, tokenizer, generation_config, args):
    import torch

    pixel_values, num_patches_list, frame_indices, frame_times = load_interval_frames(
        record["video_file"],
        record["video_intervals"],
        input_size=args.input_size,
        max_num=args.max_num,
    )
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)

    history = None
    last_response = None
    threshold = None
    interval_results = []

    for interval_idx, interval in enumerate(record["video_intervals"]):
        end_frame = interval_idx + 1
        cur_pixel_values = pixel_prefix(pixel_values, num_patches_list, end_frame)
        cur_num_patches = num_patches_list[:end_frame]
        frame_prompt = f"Frame-{end_frame}: <image>\n"

        if interval_idx == 0:
            question = PROACTIVE_TASK_PROMPT.format(query=record["query"]) + frame_prompt
            response, history = generate_response(
                model,
                tokenizer,
                cur_pixel_values,
                question,
                generation_config,
                cur_num_patches,
                history=None,
            )
            threshold = average_perplexity(
                model,
                tokenizer,
                cur_pixel_values,
                frame_prompt,
                generation_config,
                cur_num_patches,
                history,
                check_answer_text(response, args.check_len),
                self_check=True,
                num_runs=args.num_runs,
            )
            last_response = response
            pred_label = "interrupt"
            ppl = threshold
            decision_threshold = threshold
        else:
            ppl = average_perplexity(
                model,
                tokenizer,
                cur_pixel_values,
                frame_prompt,
                generation_config,
                cur_num_patches,
                history,
                check_answer_text(last_response, args.check_len),
                self_check=False,
                num_runs=args.num_runs,
            )
            decision_threshold = threshold * args.alpha
            if ppl > decision_threshold:
                response, history = generate_response(
                    model,
                    tokenizer,
                    cur_pixel_values,
                    frame_prompt,
                    generation_config,
                    cur_num_patches,
                    history,
                )
                threshold = average_perplexity(
                    model,
                    tokenizer,
                    cur_pixel_values,
                    frame_prompt,
                    generation_config,
                    cur_num_patches,
                    history,
                    check_answer_text(response, args.check_len),
                    self_check=True,
                    num_runs=args.num_runs,
                )
                last_response = response
                pred_label = "interrupt"
            else:
                response = ""
                history[-1] = (history[-1][0] + frame_prompt, history[-1][1])
                pred_label = "silent"

        gt_label = record["gt_labels"][interval_idx]
        interval_results.append(
            {
                "interval_index": interval_idx,
                "interval": interval,
                "representative_frame_index": frame_indices[interval_idx],
                "representative_time_sec": frame_times[interval_idx],
                "gt_label": gt_label,
                "pred_label": pred_label,
                "correct": pred_label == gt_label,
                "ppl": ppl,
                "decision_threshold": decision_threshold,
                "active_threshold": threshold,
                "generated_text": response,
                "gt_answer": record["gt_answers"][interval_idx],
            }
        )

    correct = sum(item["correct"] for item in interval_results)
    return {
        "video_path": record["video_path"],
        "query": record["query"],
        "domain": record["domain"],
        "task": record["task"],
        "duration_in_sec": record["duration_in_sec"],
        "interval_accuracy": correct / len(interval_results) if interval_results else 0.0,
        "num_intervals": len(interval_results),
        "interval_results": interval_results,
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


def dry_run(records, selected):
    label_counts = Counter(label for record in selected for label in record["gt_labels"])
    print("== Dry Run ==")
    print(f"usable records : {len(records)}")
    print(f"selected       : {len(selected)}")
    print(f"labels         : {dict(label_counts)}")
    for idx, record in enumerate(selected, 1):
        print(
            f"{idx:02d}. {record['video_path']} | intervals={len(record['video_intervals'])} "
            f"| domain={record['domain']} | task={record['task']}"
        )


def main():
    args = parse_args()
    args.ann_file = resolve_path(args.ann_file, args.data_root)
    args.video_dir = resolve_path(args.video_dir, args.data_root)
    args.model_path = resolve_path(args.model_path, REPO_ROOT)
    args.weights_dir = resolve_path(args.weights_dir)
    args.output = resolve_path(args.output, REPO_ROOT)

    records = load_records(
        args.ann_file,
        args.video_dir,
        max_intervals_per_video=args.max_intervals_per_video,
    )
    selected = sample_records(records, args.num_samples, args.seed)

    if args.dry_run:
        dry_run(records, selected)
        return

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
    with args.output.open("w", encoding="utf-8") as f_out, torch.no_grad():
        for record in tqdm(selected, desc="Evaluating egoproactive"):
            result = evaluate_record(record, model, tokenizer, generation_config, args)
            update_counts(counts, result)
            json.dump(result, f_out, ensure_ascii=False)
            f_out.write("\n")
            f_out.flush()
            print(
                f"{record['video_path']}: interval_acc={result['interval_accuracy']:.4f} "
                f"({result['num_intervals']} intervals)"
            )

    metrics = compute_metrics(counts)
    print_metrics(metrics)
    print(f"\nSaved predictions to {args.output}")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
