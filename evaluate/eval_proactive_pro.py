import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_DIR = REPO_ROOT / "evaluate"
INFERENCE_DIR = REPO_ROOT / "inference"
for path in (str(EVALUATE_DIR), str(INFERENCE_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import eval_proactive as base  # noqa: E402
from tshm import RecursiveEventTree, ShortTermMemory, embed_text  # noqa: E402


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "evaluate"
    / "output"
    / "egoproactive_livestarpro_sample350_fps2_lmax160_majority1.jsonl"
)
NUM_IMAGE_TOKEN = 16
DEFAULT_INTERVAL_SAMPLE_FPS = 2.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LiveStarPro SVeD/TSHM interrupt-silent decisions on egoproactive."
    )
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument("--ann-file", type=Path, default=base.DEFAULT_ANN_FILE)
    parser.add_argument("--video-dir", type=Path, default=base.DEFAULT_VIDEO_DIR)
    parser.add_argument("--model-path", type=Path, default=base.DEFAULT_MODEL_PATH)
    parser.add_argument("--weights-dir", type=Path, default=base.DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.06)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--check-len", type=int, default=1000)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--max-num", type=int, default=1)
    parser.add_argument("--interval-sample-fps", type=float, default=DEFAULT_INTERVAL_SAMPLE_FPS)
    parser.add_argument("--max-intervals-per-video", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--worker-input", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=-1)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clear-cache-per-video", dest="clear_cache_per_video", action="store_true", default=True)
    parser.add_argument("--no-clear-cache-per-video", dest="clear_cache_per_video", action="store_false")

    parser.add_argument("--l-max", type=int, default=160)
    parser.add_argument("--sigma", type=float, default=0.75)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--beam-k", type=int, default=3)
    parser.add_argument("--max-recall", type=int, default=2)
    parser.add_argument("--recall-min-gap", type=int, default=8)
    parser.add_argument("--kv", dest="kv", action="store_true", default=True)
    parser.add_argument("--no-kv", dest="kv", action="store_false")
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--no-viscache", action="store_true")
    parser.add_argument("--recall-verbatim", action="store_true")
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def resolve_args(args):
    args.data_root = base.resolve_path(args.data_root)
    args.ann_file = base.resolve_path(args.ann_file, args.data_root)
    args.video_dir = base.resolve_path(args.video_dir, args.data_root)
    args.model_path = base.resolve_path(args.model_path, REPO_ROOT)
    args.weights_dir = base.resolve_path(args.weights_dir)
    args.output = base.resolve_path(args.output, REPO_ROOT)
    if args.worker_input is not None:
        args.worker_input = base.resolve_path(args.worker_input, REPO_ROOT)
    if args.max_num != 1 and not args.no_viscache:
        print(
            "LiveStarPro visual feature cache assumes one patch per sampled frame; "
            "--no-viscache is enabled because --max-num != 1."
        )
        args.no_viscache = True
    return args


def frame_prompt(sample_id):
    return f"Frame-{sample_id}: <image>\n"


def frame_prompt_for_ids(sample_ids):
    return "".join(frame_prompt(sample_id) for sample_id in sample_ids)


def interval_sample_count(interval, sample_fps):
    start, end = interval
    duration = max(0.0, float(end) - float(start))
    return max(1, int(math.ceil(duration * float(sample_fps))))


def interval_sample_times(interval, sample_fps):
    start, end = interval
    start = max(0.0, float(start))
    end = max(start, float(end))
    sample_count = interval_sample_count((start, end), sample_fps)
    if sample_count == 1:
        return [base.interval_midpoint((start, end))]
    if end == start:
        return [start for _ in range(sample_count)]
    span = end - start
    return [
        start + span * (sample_idx + 0.5) / sample_count
        for sample_idx in range(sample_count)
    ]


def load_interval_frames_by_time(video_path, intervals, input_size=448, max_num=1, sample_fps=2.0):
    import torch
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps())
    if fps <= 0:
        raise ValueError(f"Invalid FPS for {video_path}: {fps}")
    max_frame = len(vr) - 1
    transform = base.build_transform(input_size)

    pixel_values_list = []
    num_patches_list = []
    interval_samples = []
    for interval in intervals:
        sampled_times = interval_sample_times(interval, sample_fps)
        frame_indices = []
        frame_times = []
        for timestamp in sampled_times:
            frame_idx = min(max_frame, max(0, int(round(timestamp * fps))))
            image = Image.fromarray(vr[frame_idx].asnumpy()).convert("RGB")
            tiles = base.dynamic_preprocess(
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


def estimate_work(record, sample_fps):
    return sum(interval_sample_count(interval, sample_fps) for interval in record["video_intervals"])


def shard_records(records, num_shards, sample_fps):
    shards = [{"records": [], "work": 0} for _ in range(num_shards)]
    ordered = sorted(records, key=lambda item: estimate_work(item, sample_fps), reverse=True)
    for record in ordered:
        shard = min(shards, key=lambda item: item["work"])
        shard["records"].append(record)
        shard["work"] += estimate_work(record, sample_fps)
    for shard in shards:
        shard["records"].sort(key=lambda item: item.get("sample_order", 0))
    return shards


class ProactiveLiveStarProSession:
    def __init__(self, model, tokenizer, generation_config, pixel_values, num_patches_list, query, args):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.pixel_values = pixel_values
        self.num_patches_list = num_patches_list
        self.query = query
        self.args = args
        self.stm = ShortTermMemory(l_max=args.l_max)
        self.tree = RecursiveEventTree(sigma=args.sigma, beta=args.beta, beam_k=args.beam_k)
        self.vit_cache = {}
        self.verify_kv = None
        self.verify_ids = None
        self.threshold = None
        self.last_response = None
        self.recall_by_clip = {}
        self.n_decode = 0
        self.n_silent = 0
        self.n_retrieval = 0
        self.n_compressed = 0

    def _trace(self, text):
        if self.args.trace:
            print(text, flush=True)

    def precompute_visual(self):
        if self.args.no_viscache:
            return
        import torch

        with torch.no_grad():
            for sample_id in range(1, len(self.num_patches_list) + 1):
                start, end = base.sample_patch_bounds(self.num_patches_list, sample_id)
                self.vit_cache[sample_id] = self.model.extract_feature(self.pixel_values[start:end])

    def _pixels(self, sample_ids):
        return base.pixel_select(self.pixel_values, self.num_patches_list, sample_ids)

    def _vis(self, sample_ids):
        if self.args.no_viscache or not self.vit_cache:
            return None
        import torch

        return torch.cat([self.vit_cache[sample_id] for sample_id in sample_ids], dim=0)

    def _captioned_clips(self):
        return [clip for clip in self.stm.clips if clip.caption]

    def _committed_sample_ids(self):
        return [frame.t for clip in self._captioned_clips() for frame in clip.frames]

    def _recall_text(self, recall):
        if not recall:
            return ""
        if self.args.recall_verbatim:
            return f"[Recall relevant earlier events: {recall}] "
        return (
            "[For continuity only - earlier you already described: "
            f"{recall}. Do NOT restate it; describe only what is NEW in the current frame.] "
        )

    def _derive_history(self):
        history = []
        for idx, clip in enumerate(self._captioned_clips()):
            question = ""
            if idx == 0:
                question += base.PROACTIVE_TASK_PROMPT.format(query=self.query)
            question += self._recall_text(self.recall_by_clip.get(clip.clip_id))
            question += frame_prompt_for_ids([frame.t for frame in clip.frames])
            history.append((question, clip.caption))
        return history

    def _context_info(self, history_sample_ids, input_sample_ids, history):
        return {
            "history_turns": len(history),
            "history_frames": len(history_sample_ids),
            "input_frames": len(input_sample_ids),
            "first_input_sample": input_sample_ids[0] if input_sample_ids else None,
            "last_input_sample": input_sample_ids[-1] if input_sample_ids else None,
        }

    def _cap_tokens(self, text):
        return len(self.tokenizer(text or " ").input_ids)

    def _emb(self, text):
        return embed_text(self.model, self.tokenizer, text)

    def _ppl(self, sample_id, check_answer, self_check):
        history_sample_ids = self._committed_sample_ids()
        input_sample_ids = list(history_sample_ids)
        if sample_id not in input_sample_ids:
            input_sample_ids.append(sample_id)
        pixels, patches = self._pixels(input_sample_ids)
        history = self._derive_history()
        context_info = self._context_info(history_sample_ids, input_sample_ids, history)
        total = 0.0
        kv_last = None
        for _ in range(max(1, self.args.num_runs)):
            ppl, kv_out = self.model.chat(
                self.tokenizer,
                pixels,
                frame_prompt(sample_id),
                dict(self.generation_config),
                num_patches_list=list(patches),
                history=list(history),
                return_history=False,
                check_answer=base.check_answer_text(check_answer, self.args.check_len),
                self_check=self_check,
                visual_features=self._vis(input_sample_ids),
                use_kvcache=self.args.kv,
                past_key_values=self.verify_kv,
                past_input_ids=self.verify_ids,
            )
            total += float(ppl)
            kv_last = kv_out
        if self.args.kv and kv_last is not None:
            self.verify_kv, self.verify_ids = kv_last
        return total / max(1, self.args.num_runs), context_info

    def _generate(self, sample_id, recall):
        history_sample_ids = self._committed_sample_ids()
        input_sample_ids = list(history_sample_ids)
        if sample_id not in input_sample_ids:
            input_sample_ids.append(sample_id)
        pixels, patches = self._pixels(input_sample_ids)
        history = self._derive_history()
        question = self._recall_text(recall) + frame_prompt(sample_id)
        if not history:
            question = base.PROACTIVE_TASK_PROMPT.format(query=self.query) + question
        response, _, _ = self.model.chat(
            self.tokenizer,
            pixels,
            question,
            dict(self.generation_config),
            num_patches_list=list(patches),
            history=list(history),
            return_history=True,
            visual_features=self._vis(input_sample_ids),
        )
        return response, self._context_info(history_sample_ids, input_sample_ids, history)

    def _retrieve(self, query_emb, sample_id):
        info = {
            "retrieval_used": False,
            "retrieved_text": "",
            "retrieval_evaluations": 0,
            "retrieval_candidates": 0,
        }
        if self.args.no_retrieval or len(self.tree) == 0:
            return None, info

        hits, n_eval = self.tree.retrieve(query_emb, k=self.args.beam_k)
        if self.args.recall_min_gap > 0:
            hits = [hit for hit in hits if (sample_id - hit.t_end) >= self.args.recall_min_gap]
        captions = [hit.caption for hit in hits if hit.caption][: self.args.max_recall]
        info["retrieval_evaluations"] = n_eval
        info["retrieval_candidates"] = len(hits)
        if not captions:
            return None, info

        recall = " | ".join(captions)
        self.n_retrieval += 1
        info["retrieval_used"] = True
        info["retrieved_text"] = recall
        return recall, info

    def _compress(self):
        if self.stm.token_count() <= self.stm.l_max:
            return []
        evicted, _ = self.stm.compress()
        inserted = []
        for unit in evicted:
            if self.args.no_retrieval:
                continue
            if unit.embedding is None or unit.embedding.numel() <= 1:
                unit.embedding = self._emb(unit.caption)
            tag = self.tree.insert(unit)
            inserted.append(tag)
        self.n_compressed += len(evicted)
        return inserted

    def _memory_stats(self):
        tree_stats = self.tree.stats()
        return {
            "memory_active_clips": len(self.stm.clips),
            "memory_active_tokens": self.stm.token_count(),
            "memory_compressed_clips": self.n_compressed,
            "tree_size": tree_stats["size"],
            "tree_roots": tree_stats["roots"],
            "tree_height": tree_stats["height"],
            "decode_count": self.n_decode,
            "silent_count": self.n_silent,
            "retrieval_count": self.n_retrieval,
        }

    def process_sample(self, sample_id):
        if self.threshold is None:
            return self._init_sample(sample_id)
        return self._step_sample(sample_id)

    def _init_sample(self, sample_id):
        self.stm.start_clip()
        self.stm.add_frame(t=sample_id, score=0.0, n_tokens=NUM_IMAGE_TOKEN)
        response, _ = self._generate(sample_id, recall=None)
        self.stm.clips[-1].caption = response
        ref_ppl, context_info = self._ppl(sample_id, response, self_check=True)
        self.stm.clips[-1].frames[-1].score = ref_ppl
        self.threshold = ref_ppl
        self.last_response = response
        self.n_decode += 1
        compressed = self._compress()
        self._trace(f"sample={sample_id} init decode ref_ppl={ref_ppl:.3f}")
        return {
            "pred_label": "interrupt",
            "ppl": ref_ppl,
            "decision_threshold": ref_ppl,
            "active_threshold": self.threshold,
            "self_check_ppl": ref_ppl,
            "generated_text": response,
            "context_info": context_info,
            "compressed_events": compressed,
            **self._memory_stats(),
            "retrieval_used": False,
            "retrieved_text": "",
            "retrieval_evaluations": 0,
            "retrieval_candidates": 0,
        }

    def _step_sample(self, sample_id):
        current_clip = self.stm.clips[-1]
        active_caption = current_clip.caption
        ppl, context_info = self._ppl(sample_id, active_caption, self_check=False)
        decision_threshold = self.threshold * self.args.alpha

        if ppl > decision_threshold:
            summary_emb = self._emb(active_caption)
            self.stm.finalize_clip(
                active_caption,
                self._cap_tokens(active_caption),
                summary_emb=summary_emb,
            )
            recall, retrieval_info = self._retrieve(summary_emb, sample_id)
            response, _ = self._generate(sample_id, recall)
            self.stm.start_clip()
            self.stm.add_frame(t=sample_id, score=0.0, n_tokens=NUM_IMAGE_TOKEN)
            self.stm.clips[-1].caption = response
            if recall:
                self.recall_by_clip[self.stm.clips[-1].clip_id] = recall
            ref_ppl, context_info = self._ppl(sample_id, response, self_check=True)
            self.stm.clips[-1].frames[-1].score = ref_ppl
            self.threshold = ref_ppl
            self.last_response = response
            self.n_decode += 1
            pred_label = "interrupt"
            generated_text = response
            self_check_ppl = ref_ppl
            self._trace(
                f"sample={sample_id} decode ppl={ppl:.3f} threshold={decision_threshold:.3f}"
            )
        else:
            self.stm.add_frame(t=sample_id, score=ppl, n_tokens=NUM_IMAGE_TOKEN)
            self.n_silent += 1
            pred_label = "silent"
            generated_text = ""
            self_check_ppl = None
            retrieval_info = {
                "retrieval_used": False,
                "retrieved_text": "",
                "retrieval_evaluations": 0,
                "retrieval_candidates": 0,
            }
            self._trace(
                f"sample={sample_id} silent ppl={ppl:.3f} threshold={decision_threshold:.3f}"
            )

        compressed = self._compress()
        return {
            "pred_label": pred_label,
            "ppl": ppl,
            "decision_threshold": decision_threshold,
            "active_threshold": self.threshold,
            "self_check_ppl": self_check_ppl,
            "generated_text": generated_text,
            "context_info": context_info,
            "compressed_events": compressed,
            **self._memory_stats(),
            **retrieval_info,
        }

    def summary(self):
        return {
            "decode": self.n_decode,
            "silent": self.n_silent,
            "retrieval": self.n_retrieval,
            "compressed_clips": self.n_compressed,
            "active_clips": len(self.stm.clips),
            "active_tokens": self.stm.token_count(),
            "tree": self.tree.stats(),
            "kv_enabled": self.args.kv,
            "visual_cache_enabled": not self.args.no_viscache,
        }


def evaluate_record(record, model, tokenizer, generation_config, args):
    import torch

    pixel_values, num_patches_list, interval_samples = load_interval_frames_by_time(
        record["video_file"],
        record["video_intervals"],
        input_size=args.input_size,
        max_num=args.max_num,
        sample_fps=args.interval_sample_fps,
    )
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)

    session = ProactiveLiveStarProSession(
        model,
        tokenizer,
        generation_config,
        pixel_values,
        num_patches_list,
        record["query"],
        args,
    )
    session.precompute_visual()

    interval_results = []
    sample_cursor = 0
    for interval_idx, interval in enumerate(record["video_intervals"]):
        samples = interval_samples[interval_idx]
        sample_count = len(samples["frame_indices"])
        sample_results = []

        for local_sample_idx in range(sample_count):
            sample_id = sample_cursor + local_sample_idx + 1
            decision = session.process_sample(sample_id)
            context_info = decision["context_info"]
            sample_results.append(
                {
                    "sample_index": sample_id,
                    "local_sample_index": local_sample_idx,
                    "frame_index": samples["frame_indices"][local_sample_idx],
                    "time_sec": samples["frame_times_sec"][local_sample_idx],
                    "target_time_sec": samples["target_times_sec"][local_sample_idx],
                    "pred_label": decision["pred_label"],
                    "ppl": decision["ppl"],
                    "decision_threshold": decision["decision_threshold"],
                    "active_threshold": decision["active_threshold"],
                    "self_check_ppl": decision["self_check_ppl"],
                    "generated_text": decision["generated_text"],
                    "context_history_turns": context_info["history_turns"],
                    "context_history_frames": context_info["history_frames"],
                    "context_input_frames": context_info["input_frames"],
                    "context_first_sample_index": context_info["first_input_sample"],
                    "context_last_sample_index": context_info["last_input_sample"],
                    "retrieval_used": decision["retrieval_used"],
                    "retrieved_text": decision["retrieved_text"],
                    "retrieval_evaluations": decision["retrieval_evaluations"],
                    "retrieval_candidates": decision["retrieval_candidates"],
                    "compressed_events": decision["compressed_events"],
                    "memory_active_clips": decision["memory_active_clips"],
                    "memory_active_tokens": decision["memory_active_tokens"],
                    "memory_compressed_clips": decision["memory_compressed_clips"],
                    "tree_size": decision["tree_size"],
                    "tree_roots": decision["tree_roots"],
                    "tree_height": decision["tree_height"],
                    "decode_count": decision["decode_count"],
                    "silent_count": decision["silent_count"],
                    "retrieval_count": decision["retrieval_count"],
                }
            )

        gt_label = record["gt_labels"][interval_idx]
        interrupt_votes = sum(item["pred_label"] == "interrupt" for item in sample_results)
        pred_label = "interrupt" if interrupt_votes >= 1 else "silent"
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
        "livestarpro_summary": session.summary(),
        "error": False,
    }


def make_generation_config(args):
    return {
        "temperature": 0.0,
        "max_new_tokens": args.max_new_tokens,
        "top_p": 0.1,
        "num_beams": 1,
        "repetition_penalty": 1.05,
    }


def build_experiment_config(args, generation_config=None, gpus=None, shard_plan=None):
    return {
        "protocol": "LiveStarPro TSHM SVeD interval classification",
        "sampling": {
            "interval_sample_fps": args.interval_sample_fps,
            "strategy": "uniform sub-interval centers; sample count = ceil(interval duration * interval_sample_fps)",
            "decision_step": "one LiveStarPro SVeD decision per sampled frame",
            "interval_aggregation": "majority=1; interrupt if at least 1 sampled frame interrupts, otherwise silent",
            "self_check_frame": "current sampled frame",
        },
        "context": {
            "memory": "Tree-Structured Hierarchical Memory with Peak-End short-term compression",
            "l_max": args.l_max,
            "clear_cache_per_video": args.clear_cache_per_video,
        },
        "livestarpro": {
            "enabled": True,
            "tshm": True,
            "retrieval": not args.no_retrieval,
            "kv": args.kv,
            "visual_cache": not args.no_viscache,
            "l_max": args.l_max,
            "sigma": args.sigma,
            "beta": args.beta,
            "beam_k": args.beam_k,
            "max_recall": args.max_recall,
            "recall_min_gap": args.recall_min_gap,
            "recall_verbatim": args.recall_verbatim,
        },
        "data": {
            "ann_file": str(args.ann_file),
            "video_dir": str(args.video_dir),
            "num_samples": args.num_samples,
            "seed": args.seed,
            "max_intervals_per_video": args.max_intervals_per_video,
        },
        "model": {
            "variant": "LiveStarPro",
            "model_path": str(args.model_path),
            "weights_dir": str(args.weights_dir),
            "device": args.device,
            "gpus": gpus if gpus is not None else base.parse_gpus(args.gpus),
        },
        "execution": {
            "mode": "multi_gpu" if (gpus if gpus is not None else base.parse_gpus(args.gpus)) else "single_process",
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
    total_sampled_frames = sum(estimate_work(record, args.interval_sample_fps) for record in selected)
    gpus = base.parse_gpus(args.gpus)
    print("== LiveStarPro Dry Run ==")
    print(f"usable records : {len(records)}")
    print(f"selected       : {len(selected)}")
    print(f"intervals      : {total_intervals}")
    print(f"interval fps   : {args.interval_sample_fps}")
    print(f"sampled frames : {total_sampled_frames}")
    print(f"labels         : {dict(label_counts)}")
    print(
        "LiveStarPro    : "
        f"l_max={args.l_max} sigma={args.sigma} beta={args.beta} "
        f"beam_k={args.beam_k} max_recall={args.max_recall} "
        f"recall_min_gap={args.recall_min_gap} kv={args.kv} "
        f"visual_cache={not args.no_viscache}"
    )
    if gpus:
        shards = shard_records(selected, len(gpus), args.interval_sample_fps)
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


def run_worker(args, selected, experiment_config):
    import torch

    generation_config = make_generation_config(args)
    model_dir, tmp_ctx = base.prepare_model_dir(args.model_path, args.weights_dir)
    print(f"Model dir: {model_dir}")
    model, tokenizer = base.load_model_and_tokenizer(model_dir, args.device)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kwargs: x

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    failed = Counter()
    with args.output.open("w", encoding="utf-8") as f_out, torch.no_grad():
        for record in tqdm(selected, desc="Evaluating egoproactive with LiveStarPro"):
            try:
                result = evaluate_record(record, model, tokenizer, generation_config, args)
            except Exception as exc:
                if not base.is_oom_error(exc):
                    raise
                base.clear_cuda_cache()
                result = base.make_error_result(record, args, "oom", exc)
                failed["oom"] += 1
                print(
                    f"{record['video_path']}: OOM skipped; "
                    f"{result['error_message'][:180]}"
                )
            finally:
                if args.clear_cache_per_video:
                    base.clear_cuda_cache()
            result["experiment_config"] = experiment_config
            base.update_counts(counts, result)
            json.dump(result, f_out, ensure_ascii=False)
            f_out.write("\n")
            f_out.flush()
            if not result.get("error"):
                print(
                    f"{record['video_path']}: interval_acc={result['interval_accuracy']:.4f} "
                    f"({result['num_intervals']} intervals)"
                )

    metrics = base.compute_metrics(counts)
    base.print_metrics(metrics)
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
        "--interval-sample-fps",
        str(args.interval_sample_fps),
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
        "--l-max",
        str(args.l_max),
        "--sigma",
        str(args.sigma),
        "--beta",
        str(args.beta),
        "--beam-k",
        str(args.beam_k),
        "--max-recall",
        str(args.max_recall),
        "--recall-min-gap",
        str(args.recall_min_gap),
    ]
    cmd.append("--kv" if args.kv else "--no-kv")
    if args.clear_cache_per_video:
        cmd.append("--clear-cache-per-video")
    else:
        cmd.append("--no-clear-cache-per-video")
    if args.no_retrieval:
        cmd.append("--no-retrieval")
    if args.no_viscache:
        cmd.append("--no-viscache")
    if args.recall_verbatim:
        cmd.append("--recall-verbatim")
    if args.trace:
        cmd.append("--trace")
    return cmd


def launch_multi_gpu(args, selected, gpus, generation_config):
    shards = shard_records(selected, len(gpus), args.interval_sample_fps)
    plan = shard_plan_summary(shards, gpus)
    experiment_config = build_experiment_config(args, generation_config, gpus=gpus, shard_plan=plan)
    shard_dir = args.output.parent / f"{args.output.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    log_handles = []
    for idx, shard in enumerate(shards):
        shard_input = shard_dir / f"shard_{idx:02d}_input.jsonl"
        shard_output = shard_dir / f"shard_{idx:02d}_output.jsonl"
        shard_log = shard_dir / f"shard_{idx:02d}.log"
        base.write_records_jsonl(shard_input, shard["records"])
        cmd = build_worker_cmd(args, shard_input, shard_output, idx, len(shards))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus[idx]
        log_f = shard_log.open("w", encoding="utf-8")
        log_handles.append(log_f)
        print(
            f"Launching LiveStarPro shard {idx} on GPU {gpus[idx]}: "
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
        raise RuntimeError("One or more LiveStarPro evaluation shards failed.")

    merged = []
    for item in processes:
        for result in base.read_records_jsonl(item["output"]):
            result["experiment_config"] = experiment_config
            merged.append(result)
    merged.sort(key=lambda item: item.get("sample_order", 0))

    counts = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f_out:
        for result in merged:
            base.update_counts(counts, result)
            json.dump(result, f_out, ensure_ascii=False)
            f_out.write("\n")

    metrics = base.compute_metrics(counts)
    base.print_metrics(metrics)
    print(f"\nSaved merged predictions to {args.output}")
    print(f"Shard logs and outputs are in {shard_dir}")


def main():
    args = resolve_args(parse_args())

    if args.worker_input is not None:
        selected = base.read_records_jsonl(args.worker_input)
        records = selected
    else:
        records = base.load_records(
            args.ann_file,
            args.video_dir,
            max_intervals_per_video=args.max_intervals_per_video,
        )
        selected = base.attach_sample_order(
            base.sample_records(records, args.num_samples, args.seed)
        )

    if args.dry_run:
        dry_run(records, selected, args)
        return

    generation_config = make_generation_config(args)
    gpus = base.parse_gpus(args.gpus)
    if gpus and args.worker_input is None:
        launch_multi_gpu(args, selected, gpus, generation_config)
        return

    experiment_config = build_experiment_config(args, generation_config, gpus=gpus)
    run_worker(args, selected, experiment_config)


if __name__ == "__main__":
    main()
