#!/usr/bin/env python3
"""Prepare EgoProactive chunk-level SFT data for LiveStar.

The LiveStar training entry point already supports interleaved multi-image
samples with a ``conversations`` field. This script converts each
EgoProactive video interval into one such sample and caches the sampled frames
as JPEG files.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2


DEFAULT_DATA_ROOT = Path("/data1/wearable_ai_challenge_data")
DEFAULT_ANNOTATIONS = (
    DEFAULT_DATA_ROOT
    / "egoproactive"
    / "wearable_ai_2026_egoproactive_val_700.jsonl"
)
DEFAULT_VIDEO_FOLDER = DEFAULT_DATA_ROOT / "egoproactive" / "val"
DEFAULT_OUTPUT_DIR = Path("/data1/finetune/data/wearableai_val")

SYSTEM_PROMPT = (
    "You are a proactive AI assistant watching a first-person video of the user "
    "performing a procedural task. Decide whether it is useful to speak at the "
    "current moment. If speaking is useful, provide one short, timely, actionable "
    "instruction. If not, stay silent."
)

DECISION_INSTRUCTION = (
    "You are now at the end of the current video chunk. Output exactly `$silent$` "
    "if no timely help is needed. If help is needed, start with `$interrupt$` and "
    "then give one short, actionable instruction for the user."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--video-folder", default=str(DEFAULT_VIDEO_FOLDER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--frames-per-interval", type=int, default=1)
    parser.add_argument(
        "--frame-history-chunks",
        type=int,
        default=4,
        help="Include the current chunk plus this many previous chunks of sampled frames.",
    )
    parser.add_argument("--max-history-turns", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_frame_indices(
    fps: float,
    total_frames: int,
    interval: list[float],
    frames_per_interval: int,
) -> list[int]:
    start, end = float(interval[0]), float(interval[1])
    start_frame = max(0, int(start * fps))
    end_frame = min(max(total_frames - 1, 0), int(end * fps))
    if end_frame < start_frame:
        return []
    count = min(frames_per_interval, end_frame - start_frame + 1)
    if count <= 0:
        return []
    if count == 1:
        return [(start_frame + end_frame) // 2]
    step = (end_frame - start_frame) / count
    return [min(end_frame, int(start_frame + i * step)) for i in range(count)]


def extract_frames_for_session(
    row: dict[str, Any],
    video_folder: Path,
    output_dir: Path,
    frames_per_interval: int,
    jpeg_quality: int,
    force_extract: bool,
) -> list[list[str]]:
    video_path = video_folder / str(row["video_path"])
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            raise RuntimeError(f"Invalid video metadata: {video_path}")

        video_stem = video_path.stem
        frame_dir = output_dir / "frames" / video_stem
        frame_dir.mkdir(parents=True, exist_ok=True)

        interval_paths: list[list[str]] = []
        for chunk_idx, interval in enumerate(row.get("video_intervals", [])):
            paths: list[str] = []
            frame_indices = sample_frame_indices(
                fps=fps,
                total_frames=total_frames,
                interval=interval,
                frames_per_interval=frames_per_interval,
            )
            for local_idx, frame_idx in enumerate(frame_indices):
                frame_path = frame_dir / f"chunk_{chunk_idx:04d}_frame_{local_idx:02d}_{frame_idx:08d}.jpg"
                if force_extract or not frame_path.exists():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    cv2.imwrite(
                        str(frame_path),
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                    )
                paths.append(frame_path.relative_to(output_dir).as_posix())
            interval_paths.append(paths)
        return interval_paths
    finally:
        cap.release()


def normalize_history(dialog_at_chunk: list[dict[str, Any]], max_history_turns: int) -> list[str]:
    turns = dialog_at_chunk[1:] if dialog_at_chunk else []
    if max_history_turns == 0:
        turns = []
    elif max_history_turns > 0:
        turns = turns[-max_history_turns:]

    rendered: list[str] = []
    for turn in turns:
        role = str(turn.get("role", "assistant")).strip().lower()
        text = str(turn.get("text", "")).strip()
        if text:
            rendered.append(f"{role}: {text}")
    return rendered


def normalize_answer(answer: str) -> str:
    text = (answer or "").strip()
    lowered = text.lower()
    if lowered.startswith("$silent$"):
        return "$silent$"
    if lowered.startswith("$interrupt$"):
        return "$interrupt$" + text[len("$interrupt$") :].strip()
    if not text:
        return "$silent$"
    return "$interrupt$" + text


def build_user_prompt(
    row: dict[str, Any],
    chunk_idx: int,
    image_count: int,
    max_history_turns: int,
) -> str:
    query = str(row.get("query", "")).strip()
    task = str(row.get("task", "")).strip()
    domain = str(row.get("domain", "")).strip()
    dialogs = row.get("dialog", [])
    dialog_at_chunk = dialogs[chunk_idx] if chunk_idx < len(dialogs) else []
    history = normalize_history(dialog_at_chunk, max_history_turns)
    frame_prompt = "".join(f"Frame-{i + 1}: <image>\n" for i in range(image_count))

    parts = [
        f"User query: {query}",
    ]
    if task:
        parts.append(f"Task: {task}")
    if domain:
        parts.append(f"Domain: {domain}")
    if history:
        parts.append("Recent dialog:\n" + "\n".join(history))
    parts.append("Observed recent video frames up to the current chunk:\n" + frame_prompt.rstrip())
    parts.append(DECISION_INSTRUCTION)
    return "\n\n".join(parts)


def build_samples_for_session(
    row: dict[str, Any],
    frame_paths_by_interval: list[list[str]],
    frame_history_chunks: int,
    max_history_turns: int,
) -> list[dict[str, Any]]:
    answers = row.get("answers", [])
    video_stem = Path(str(row["video_path"])).stem
    samples: list[dict[str, Any]] = []
    for chunk_idx, raw_answer in enumerate(answers):
        start = max(0, chunk_idx - frame_history_chunks)
        image_paths = [
            path
            for paths in frame_paths_by_interval[start : chunk_idx + 1]
            for path in paths
        ]
        if not image_paths:
            continue
        answer = normalize_answer(str(raw_answer))
        label = "silent" if answer == "$silent$" else "interrupt"
        samples.append(
            {
                "id": f"{video_stem}_chunk_{chunk_idx:04d}",
                "image": image_paths,
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {
                        "from": "human",
                        "value": build_user_prompt(
                            row,
                            chunk_idx=chunk_idx,
                            image_count=len(image_paths),
                            max_history_turns=max_history_turns,
                        ),
                    },
                    {"from": "gpt", "value": answer},
                ],
                "video_path": row["video_path"],
                "chunk_index": chunk_idx,
                "label": label,
                "task": row.get("task", ""),
                "domain": row.get("domain", ""),
            }
        )
    return samples


def split_sessions(
    rows: list[dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if not 0 < train_ratio <= 1:
        raise ValueError("--train-ratio must be in (0, 1].")
    if not 0 <= dev_ratio < 1:
        raise ValueError("--dev-ratio must be in [0, 1).")
    if train_ratio + dev_ratio > 1:
        raise ValueError("--train-ratio + --dev-ratio must be <= 1.")

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    train_end = int(len(indices) * train_ratio)
    dev_end = train_end + int(len(indices) * dev_ratio)
    splits = {
        "train": [rows[i] for i in indices[:train_end]],
        "dev": [rows[i] for i in indices[train_end:dev_end]],
        "test": [rows[i] for i in indices[dev_end:]],
    }
    return {name: split_rows for name, split_rows in splits.items() if split_rows}


def write_meta(output_dir: Path, split_counts: dict[str, int]) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for split, count in split_counts.items():
        meta = {
            f"egoproactive_{split}": {
                "root": str(output_dir.resolve()),
                "annotation": str((output_dir / "annotations" / f"egoproactive_{split}.jsonl").resolve()),
                "data_augment": False,
                "repeat_time": 1,
                "length": count,
            }
        }
        meta_path = meta_dir / f"egoproactive_{split}_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    annotations = Path(args.annotations)
    video_folder = Path(args.video_folder)
    output_dir = Path(args.output_dir)
    rows = load_jsonl(annotations)
    if args.max_sessions is not None:
        rows = rows[: args.max_sessions]
    if not rows:
        raise RuntimeError("No EgoProactive sessions loaded.")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = split_sessions(rows, args.train_ratio, args.dev_ratio, args.seed)

    split_samples: dict[str, list[dict[str, Any]]] = {}
    for split, sessions in split_rows.items():
        samples: list[dict[str, Any]] = []
        for row_idx, row in enumerate(sessions, start=1):
            frame_paths = extract_frames_for_session(
                row=row,
                video_folder=video_folder,
                output_dir=output_dir,
                frames_per_interval=args.frames_per_interval,
                jpeg_quality=args.jpeg_quality,
                force_extract=args.force_extract,
            )
            samples.extend(
                build_samples_for_session(
                    row,
                    frame_paths_by_interval=frame_paths,
                    frame_history_chunks=args.frame_history_chunks,
                    max_history_turns=args.max_history_turns,
                )
            )
            if row_idx % 25 == 0:
                print(f"[{split}] processed {row_idx}/{len(sessions)} sessions")
        split_samples[split] = samples

    split_counts: dict[str, int] = {}
    for split, samples in split_samples.items():
        annotation_path = output_dir / "annotations" / f"egoproactive_{split}.jsonl"
        dump_jsonl(annotation_path, samples)
        split_counts[split] = len(samples)
        label_counts: dict[str, int] = {}
        for sample in samples:
            label_counts[sample["label"]] = label_counts.get(sample["label"], 0) + 1
        print(f"{split}: {len(samples)} samples, labels={label_counts}")

    write_meta(output_dir, split_counts)
    print(f"Wrote EgoProactive LiveStar SFT data to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
