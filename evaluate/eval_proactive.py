#!/usr/bin/env python3
"""Evaluate LiveStar on the ECCV 2026 Wearable AI EgoProactive split.

This script intentionally separates prediction generation from scoring:
LiveStar inference can run in the `LiveStar` environment, while scoring can be
delegated to the starter-kit environment through --eval-python.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/data1/wearable_ai_challenge_data")
DEFAULT_ANNOTATIONS = (
    DEFAULT_DATA_ROOT
    / "egoproactive"
    / "wearable_ai_2026_egoproactive_val_700.jsonl"
)
DEFAULT_VIDEO_FOLDER = DEFAULT_DATA_ROOT / "egoproactive" / "val"
DEFAULT_STARTER_KIT = DEFAULT_DATA_ROOT / "starter_kit"
DEFAULT_MODEL_CODE_DIR = PROJECT_ROOT / "inference"
DEFAULT_WEIGHTS_DIR = Path("/data1/LiveStar_8B")
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluate" / "output" / "egoproactive_livestar_predictions.jsonl"
DEFAULT_EVAL_OUTPUT = PROJECT_ROOT / "evaluate" / "output" / "egoproactive_livestar_results.json"

SYSTEM_PROMPT = (
    "You are a proactive AI assistant watching a first-person video of the user "
    "performing a procedural task. The user has issued a high-level query. "
    "When it is useful to speak, generate one short, timely, actionable instruction "
    "for the user. Speak directly to the user in second person or imperative form. "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS), help="EgoProactive JSONL annotations.")
    parser.add_argument("--video-folder", default=str(DEFAULT_VIDEO_FOLDER), help="Folder containing validation videos.")
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR), help="Directory containing model-*.safetensors.")
    parser.add_argument(
        "--model-code-dir",
        default=str(DEFAULT_MODEL_CODE_DIR),
        help="Directory containing LiveStar model code/tokenizer/index files.",
    )
    parser.add_argument("--starter-kit", default=str(DEFAULT_STARTER_KIT), help="Starter-kit directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Prediction JSONL output path.")
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL_OUTPUT), help="Starter-kit results JSON path.")
    parser.add_argument(
        "--lora-adapter",
        default="",
        help=(
            "Optional PEFT LoRA adapter directory. A directory with adapter_config.json "
            "is loaded as the LLM adapter. A directory containing llm/ and/or vision/ "
            "subdirectories loads those adapters respectively."
        ),
    )
    parser.add_argument(
        "--llm-lora-adapter",
        default="",
        help="Optional explicit PEFT LoRA adapter directory for model.language_model.",
    )
    parser.add_argument(
        "--vision-lora-adapter",
        default="",
        help="Optional explicit PEFT LoRA adapter directory for model.vision_model.",
    )
    parser.add_argument(
        "--eval-python",
        default=sys.executable,
        help=(
            "Python command used for starter-kit eval. Examples: "
            "`python`, `/path/to/python`, or `conda run -n wearable_eval python`."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Only process the first N sessions.")
    parser.add_argument("--resume", action="store_true", help="Skip already written prediction rows.")
    parser.add_argument(
        "--gpu-ids",
        default="",
        help=(
            "Comma-separated GPU IDs for multi-process sample parallelism, e.g. "
            "`0,1,2,3`. Each worker sees one GPU through CUDA_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument("--generate-only", action="store_true", help="Write predictions but skip starter-kit scoring.")
    parser.add_argument("--eval-only", action="store_true", help="Skip inference and score an existing prediction file.")
    parser.add_argument("--frames-per-interval", type=int, default=2, help="Frames sampled per chunk interval.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=512,
        help=(
            "Max cumulative frames per session. SVeD keeps image placeholders in "
            "chat history, so this script fails fast instead of silently "
            "downsampling history. Use lower --frames-per-interval for memory."
        ),
    )
    parser.add_argument("--max-history-turns", type=int, default=4, help="Prior dialog turns to include. -1 keeps all.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max generated tokens per chunk.")
    parser.add_argument("--input-size", type=int, default=448, help="Visual encoder input size.")
    parser.add_argument("--decode-factor", type=float, default=1.04, help="SVeD threshold multiplier.")
    parser.add_argument("--check-len", type=int, default=1000, help="Max chars from the last output used for PPL checks.")
    parser.add_argument("--ppl-runs", type=int, default=1, help="Repeat PPL checks and average the result.")
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required path does not exist: {path}")


def has_adapter_config(path: Path) -> bool:
    return (path / "adapter_config.json").is_file()


def resolve_lora_adapters(args: argparse.Namespace) -> dict[str, Path]:
    adapters: dict[str, Path] = {}
    if args.lora_adapter:
        root = Path(args.lora_adapter).expanduser()
        llm_dir = root / "llm"
        vision_dir = root / "vision"
        if has_adapter_config(llm_dir):
            adapters["llm"] = llm_dir
        elif has_adapter_config(root):
            adapters["llm"] = root
        if has_adapter_config(vision_dir):
            adapters["vision"] = vision_dir
        if not adapters:
            raise FileNotFoundError(
                f"No adapter_config.json found in {root}, {llm_dir}, or {vision_dir}."
            )
    if args.llm_lora_adapter:
        adapters["llm"] = Path(args.llm_lora_adapter).expanduser()
    if args.vision_lora_adapter:
        adapters["vision"] = Path(args.vision_lora_adapter).expanduser()
    for name, path in adapters.items():
        require_file(path / "adapter_config.json")
    return adapters


def load_lora_adapters(model, args: argparse.Namespace) -> dict[str, str]:
    adapters = resolve_lora_adapters(args)
    if not adapters:
        return {}
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Loading LoRA adapters requires peft in the active environment.") from exc

    loaded: dict[str, str] = {}
    if "llm" in adapters:
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            str(adapters["llm"]),
            is_trainable=False,
        )
        loaded["llm"] = str(adapters["llm"].resolve())
    if "vision" in adapters:
        model.vision_model = PeftModel.from_pretrained(
            model.vision_model,
            str(adapters["vision"]),
            is_trainable=False,
        )
        loaded["vision"] = str(adapters["vision"].resolve())
    return loaded


def make_runtime_model_dir(model_code_dir: Path, weights_dir: Path) -> tuple[tempfile.TemporaryDirectory, Path]:
    """Create a temporary HF model directory without copying large weights."""
    model_code_dir = model_code_dir.resolve()
    weights_dir = weights_dir.resolve()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.model"):
        require_file(model_code_dir / name)

    weight_files = sorted(weights_dir.glob("model-*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No model-*.safetensors files found in {weights_dir}")

    tmp = tempfile.TemporaryDirectory(prefix="livestar_proactive_")
    runtime_dir = Path(tmp.name)
    for item in model_code_dir.iterdir():
        if item.name.startswith("model-") and item.suffix == ".safetensors":
            continue
        target = runtime_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, symlinks=True)
        else:
            target.symlink_to(item)
    for item in weight_files:
        (runtime_dir / item.name).symlink_to(item)
    return tmp, runtime_dir


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def sample_frame_indices(
    fps: float,
    total_frames: int,
    interval: tuple[float, float],
    frames_per_interval: int,
) -> list[int]:
    start, end = interval
    start_frame = max(0, int(start * fps))
    end_frame = min(max(total_frames - 1, 0), int(end * fps))
    if end_frame < start_frame:
        return []
    n = min(frames_per_interval, end_frame - start_frame + 1)
    if n <= 0:
        return []
    if n == 1:
        return [(start_frame + end_frame) // 2]
    step = (end_frame - start_frame) / n
    return [min(end_frame, int(start_frame + i * step)) for i in range(n)]


def load_interval_tensors(
    video_path: Path,
    intervals: list[list[float]],
    transform: T.Compose,
    frames_per_interval: int,
) -> list[list[torch.Tensor]]:
    require_file(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            raise RuntimeError(f"Invalid video metadata: {video_path}")

        tensors_by_interval: list[list[torch.Tensor]] = []
        for raw_interval in intervals:
            interval = (float(raw_interval[0]), float(raw_interval[1]))
            frame_tensors: list[torch.Tensor] = []
            for idx in sample_frame_indices(fps, total_frames, interval, frames_per_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensors.append(transform(Image.fromarray(frame).convert("RGB")))
            tensors_by_interval.append(frame_tensors)
        return tensors_by_interval
    finally:
        cap.release()


def normalize_history(dialog_at_chunk: list[dict[str, object]], max_history_turns: int) -> list[str]:
    turns = dialog_at_chunk[1:] if dialog_at_chunk else []
    if max_history_turns == 0:
        turns = []
    elif max_history_turns > 0:
        turns = turns[-max_history_turns:]

    rendered: list[str] = []
    for turn in turns:
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        role = str(turn.get("role", "user")).strip().lower()
        rendered.append(f"{role}: {text}")
    return rendered


def build_initial_question(row: dict[str, object], frame_count: int, max_history_turns: int) -> str:
    query = str(row.get("query", "")).strip()
    task = str(row.get("task", "")).strip()
    domain = str(row.get("domain", "")).strip()
    dialog = row.get("dialog", [])
    history = normalize_history(dialog[0] if dialog else [], max_history_turns)
    frame_prompt = "".join(f"Frame-{i + 1}: <image>\n" for i in range(frame_count))

    parts = [
        SYSTEM_PROMPT,
        f"User query: {query}",
    ]
    if task:
        parts.append(f"Task: {task}")
    if domain:
        parts.append(f"Domain: {domain}")
    if history:
        parts.append("Recent dialog:\n" + "\n".join(history))
    parts.append(
        "Observed video frames up to the current chunk:\n"
        f"{frame_prompt}"
        "Guidance:"
    )
    return "\n\n".join(parts)


def build_frame_question(start_idx: int, frame_count: int) -> str:
    return "".join(f"Frame-{start_idx + i + 1}: <image>\n" for i in range(frame_count))


def normalize_interrupt(raw: str) -> str:
    text = (raw or "").strip()
    lowered = text.lower()
    if lowered.startswith("$interrupt$"):
        return "$interrupt$" + text[len("$interrupt$") :].strip()
    if lowered.startswith("$silent$"):
        return "$silent$"
    if not text:
        return "$silent$"
    if "$interrupt$" in lowered[:80]:
        after = text[lowered.find("$interrupt$") + len("$interrupt$") :].strip()
        return "$interrupt$" + after
    return "$interrupt$" + text


def count_chunks(rows: list[dict[str, object]]) -> int:
    return sum(len(row.get("answers", [])) for row in rows)


def compute_avg_perplexity(
    model,
    tokenizer,
    pixel_values: torch.Tensor,
    question: str,
    generation_config: dict[str, object],
    num_patches_list: list[int],
    history: list[tuple[str, str]],
    check_answer: str,
    self_check: bool,
    num_runs: int,
) -> float:
    total = 0.0
    for _ in range(max(num_runs, 1)):
        perplexity, _ = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=list(num_patches_list),
            history=history,
            return_history=False,
            check_answer=check_answer,
            self_check=self_check,
        )
        total += float(perplexity)
    return total / max(num_runs, 1)


def already_completed_rows(output_path: Path, rows: list[dict[str, object]]) -> int:
    if not output_path.exists():
        return 0
    preds = load_jsonl(output_path)
    count = 0
    for pred, row in zip(preds, rows):
        if pred.get("video_path") != row.get("video_path"):
            break
        count += 1
    return count


def parse_gpu_ids(gpu_ids: str) -> list[str]:
    text = gpu_ids.strip()
    if not text:
        return []
    if text.lower() == "all":
        if not torch.cuda.is_available():
            return []
        return [str(i) for i in range(torch.cuda.device_count())]
    return [part.strip() for part in text.split(",") if part.strip()]


def build_parallel_worker_command(
    args: argparse.Namespace,
    annotations: Path,
    output_path: Path,
    shard_index: int,
    num_shards: int,
) -> list[str]:
    eval_output = output_path.with_suffix(".results.json")
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--annotations",
        str(annotations),
        "--video-folder",
        str(args.video_folder),
        "--weights-dir",
        str(args.weights_dir),
        "--model-code-dir",
        str(args.model_code_dir),
        "--starter-kit",
        str(args.starter_kit),
        "--output",
        str(output_path),
        "--eval-output",
        str(eval_output),
        "--lora-adapter",
        str(args.lora_adapter),
        "--llm-lora-adapter",
        str(args.llm_lora_adapter),
        "--vision-lora-adapter",
        str(args.vision_lora_adapter),
        "--eval-python",
        str(args.eval_python),
        "--frames-per-interval",
        str(args.frames_per_interval),
        "--max-frames",
        str(args.max_frames),
        "--max-history-turns",
        str(args.max_history_turns),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--input-size",
        str(args.input_size),
        "--decode-factor",
        str(args.decode_factor),
        "--check-len",
        str(args.check_len),
        "--ppl-runs",
        str(args.ppl_runs),
        "--generate-only",
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(num_shards),
    ]


def generate_predictions_parallel(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        generate_predictions(args, rows)
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_idx = already_completed_rows(output_path, rows) if args.resume else 0
    completed_predictions = load_jsonl(output_path)[:start_idx] if start_idx else []
    pending_rows = rows[start_idx:]
    if not pending_rows:
        print("All requested rows already exist in the prediction file; skipping generation.")
        return

    worker_gpu_ids = gpu_ids[: min(len(gpu_ids), len(pending_rows))]
    with tempfile.TemporaryDirectory(prefix="livestar_parallel_") as tmp_name:
        tmp_dir = Path(tmp_name)
        worker_annotations = tmp_dir / "rows.jsonl"
        dump_jsonl(worker_annotations, pending_rows)

        processes: list[tuple[int, str, Path, subprocess.Popen]] = []
        for rank, gpu_id in enumerate(worker_gpu_ids):
            worker_output = tmp_dir / f"predictions_rank{rank}.jsonl"
            cmd = build_parallel_worker_command(
                args,
                worker_annotations,
                worker_output,
                shard_index=rank,
                num_shards=len(worker_gpu_ids),
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            print(f"[worker {rank}] GPU {gpu_id}: {' '.join(shlex.quote(part) for part in cmd)}")
            process = subprocess.Popen(cmd, cwd=str(Path.cwd()), env=env)
            processes.append((rank, gpu_id, worker_output, process))

        failures: list[str] = []
        for rank, gpu_id, _, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append(f"worker {rank} on GPU {gpu_id} exited with code {return_code}")
        if failures:
            raise RuntimeError("; ".join(failures))

        predictions_by_rank: dict[int, list[dict[str, object]]] = {}
        for rank, _, worker_output, _ in processes:
            require_file(worker_output)
            predictions = load_jsonl(worker_output)
            expected_rows = pending_rows[rank :: len(worker_gpu_ids)]
            if len(predictions) != len(expected_rows):
                raise RuntimeError(
                    f"Worker {rank} wrote {len(predictions)} rows, expected {len(expected_rows)}."
                )
            for pred, row in zip(predictions, expected_rows):
                if pred.get("video_path") != row.get("video_path"):
                    raise RuntimeError(
                        f"Worker {rank} output order mismatch: expected {row.get('video_path')}, "
                        f"got {pred.get('video_path')}"
                    )
            predictions_by_rank[rank] = predictions

        merged_pending: list[dict[str, object]] = []
        for local_idx, _ in enumerate(pending_rows):
            rank = local_idx % len(worker_gpu_ids)
            rank_pos = local_idx // len(worker_gpu_ids)
            merged_pending.append(predictions_by_rank[rank][rank_pos])

        merged = completed_predictions + merged_pending
        dump_jsonl(output_path, merged)
        print(f"Merged {len(merged)} prediction rows into {output_path}")


def generate_predictions(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_idx = already_completed_rows(output_path, rows) if args.resume else 0
    mode = "a" if start_idx else "w"

    transform = build_transform(args.input_size)
    tmp, runtime_model_dir = make_runtime_model_dir(Path(args.model_code_dir), Path(args.weights_dir))

    with tmp:
        print(f"Loading LiveStar from temporary model dir: {runtime_model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(runtime_model_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(runtime_model_dir, trust_remote_code=True).half().cuda().to(torch.bfloat16)
        loaded_adapters = load_lora_adapters(model, args)
        if loaded_adapters:
            print(f"Loaded LoRA adapters: {loaded_adapters}")
        model.eval()

        generation_config = {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "num_beams": 1,
            "repetition_penalty": 1.05,
        }

        with output_path.open(mode, encoding="utf-8") as out_f, torch.no_grad():
            iterator = tqdm(list(enumerate(rows[start_idx:], start=start_idx)), desc="EgoProactive")
            for row_idx, row in iterator:
                video_path = Path(args.video_folder) / str(row["video_path"])
                intervals = row.get("video_intervals", [])
                tensors_by_interval = load_interval_tensors(
                    video_path,
                    intervals,
                    transform,
                    frames_per_interval=args.frames_per_interval,
                )

                answers: list[str] = []
                cumulative: list[torch.Tensor] = []
                seen_frame_count = 0
                output_last = ""
                chat_history = None
                decode_threshold = None
                for chunk_idx, chunk_tensors in enumerate(tensors_by_interval):
                    new_frame_count = len(chunk_tensors)
                    cumulative.extend(chunk_tensors)
                    if args.max_frames > 0 and len(cumulative) > args.max_frames:
                        raise RuntimeError(
                            f"{row['video_path']} exceeded --max-frames={args.max_frames}. "
                            "SVeD history requires all prior image placeholders to stay aligned; "
                            "lower --frames-per-interval or raise --max-frames."
                        )
                    if not cumulative:
                        answers.append("$silent$")
                        continue

                    pixel_values = torch.stack(cumulative).to(torch.bfloat16).to(model.device)
                    num_patches_list = [1] * len(cumulative)
                    if chat_history is None:
                        question = build_initial_question(row, len(cumulative), args.max_history_turns)
                    else:
                        question = build_frame_question(seen_frame_count, max(new_frame_count, 1))
                    try:
                        if chat_history is None:
                            output_last, chat_history, _ = model.chat(
                                tokenizer,
                                pixel_values,
                                question,
                                generation_config,
                                num_patches_list=num_patches_list,
                                history=None,
                                return_history=True,
                            )
                            decode_threshold = compute_avg_perplexity(
                                model,
                                tokenizer,
                                pixel_values,
                                build_frame_question(0, len(cumulative)),
                                generation_config,
                                num_patches_list,
                                chat_history,
                                check_answer=output_last[: min(args.check_len, len(output_last))],
                                self_check=True,
                                num_runs=args.ppl_runs,
                            )
                            answers.append(normalize_interrupt(output_last))
                            seen_frame_count += new_frame_count
                            continue

                        output_perplexity = compute_avg_perplexity(
                            model,
                            tokenizer,
                            pixel_values,
                            question,
                            generation_config,
                            num_patches_list,
                            chat_history,
                            check_answer=output_last[: min(args.check_len, len(output_last))],
                            self_check=False,
                            num_runs=args.ppl_runs,
                        )

                        if decode_threshold is not None and output_perplexity > decode_threshold * args.decode_factor:
                            output_last, chat_history, _ = model.chat(
                                tokenizer,
                                pixel_values,
                                question,
                                generation_config,
                                num_patches_list=num_patches_list,
                                history=chat_history,
                                return_history=True,
                            )
                            decode_threshold = compute_avg_perplexity(
                                model,
                                tokenizer,
                                pixel_values,
                                question,
                                generation_config,
                                num_patches_list,
                                chat_history,
                                check_answer=output_last[: min(args.check_len, len(output_last))],
                                self_check=True,
                                num_runs=args.ppl_runs,
                            )
                            answers.append(normalize_interrupt(output_last))
                        else:
                            chat_history[-1] = (chat_history[-1][0] + question, chat_history[-1][1])
                            answers.append("$silent$")
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        raise RuntimeError(
                            "CUDA OOM during generation. Try smaller --max-frames "
                            "or --frames-per-interval."
                        )
                    seen_frame_count += new_frame_count

                pred = {"video_path": row["video_path"], "answers": answers}
                out_f.write(json.dumps(pred, ensure_ascii=False) + "\n")
                out_f.flush()
                iterator.set_postfix(row=row_idx + 1, chunks=len(answers))


def maybe_write_subset_annotations(args: argparse.Namespace, rows: list[dict[str, object]]) -> Path:
    annotations = Path(args.annotations)
    if args.max_samples is None:
        return annotations.resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    subset_path = output_path.with_suffix(f".golden_first_{len(rows)}.jsonl")
    dump_jsonl(subset_path, rows)
    return subset_path.resolve()


def run_starter_kit_eval(args: argparse.Namespace, rows: list[dict[str, object]]) -> dict[str, object]:
    starter_kit = Path(args.starter_kit).resolve()
    eval_script = starter_kit / "run_evaluation.py"
    require_file(eval_script)
    require_file(Path(args.output))
    golden_path = maybe_write_subset_annotations(args, rows)
    eval_output = Path(args.eval_output).resolve()

    cmd = [
        *shlex.split(args.eval_python),
        str(eval_script),
        "--task",
        "proactive",
        "--eval-only",
        "--golden",
        str(golden_path),
        "--predictions",
        str(Path(args.output).resolve()),
        "--eval-output",
        str(eval_output),
    ]
    print("Running starter-kit evaluation:")
    print(" ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=str(starter_kit), check=True)
    require_file(eval_output)
    return load_json(eval_output)


def write_run_log(
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    started_at: float,
    generation_seconds: float | None,
    scoring_seconds: float | None,
    results: dict[str, object] | None,
) -> None:
    output_path = Path(args.output).resolve()
    eval_output_path = Path(args.eval_output).resolve()
    log_path = output_path.parent / "log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "samples": len(rows),
        "chunks": count_chunks(rows),
        "elapsed_seconds": round(now - started_at, 3),
        "generation_seconds": round(generation_seconds, 3) if generation_seconds is not None else None,
        "scoring_seconds": round(scoring_seconds, 3) if scoring_seconds is not None else None,
        "overall_metrics": results.get("overall") if results else None,
        "predictions_path": str(output_path),
        "eval_output_path": str(eval_output_path) if eval_output_path.exists() else None,
        "weights_dir": str(Path(args.weights_dir).resolve()),
        "model_code_dir": str(Path(args.model_code_dir).resolve()),
        "lora_adapter": str(Path(args.lora_adapter).resolve()) if args.lora_adapter else "",
        "llm_lora_adapter": str(Path(args.llm_lora_adapter).resolve()) if args.llm_lora_adapter else "",
        "vision_lora_adapter": str(Path(args.vision_lora_adapter).resolve()) if args.vision_lora_adapter else "",
        "max_samples": args.max_samples,
        "frames_per_interval": args.frames_per_interval,
        "max_frames": args.max_frames,
        "max_history_turns": args.max_history_turns,
        "max_new_tokens": args.max_new_tokens,
        "decode_factor": args.decode_factor,
        "ppl_runs": args.ppl_runs,
        "generate_only": args.generate_only,
        "eval_only": args.eval_only,
        "gpu_ids": args.gpu_ids,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    print(f"Run log written to {log_path}")


def main() -> None:
    started_at = time.time()
    args = parse_args()
    annotations = Path(args.annotations)
    require_file(annotations)
    rows = load_jsonl(annotations)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        rows = rows[args.shard_index :: args.num_shards]
    if not rows:
        raise RuntimeError("No evaluation rows loaded.")

    if args.eval_only and args.generate_only:
        raise ValueError("--eval-only and --generate-only cannot be used together.")

    generation_seconds: float | None = None
    scoring_seconds: float | None = None
    results: dict[str, object] | None = None

    if not args.eval_only:
        generation_started = time.time()
        generate_predictions_parallel(args, rows)
        generation_seconds = time.time() - generation_started

    if not args.generate_only:
        scoring_started = time.time()
        results = run_starter_kit_eval(args, rows)
        scoring_seconds = time.time() - scoring_started

    write_run_log(args, rows, started_at, generation_seconds, scoring_seconds, results)


if __name__ == "__main__":
    main()
