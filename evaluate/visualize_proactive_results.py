import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "evaluate" / "output" / "egoproactive_sved_sample350_fpi4_ctxi20_f80_hist20_majority1.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluate" / "output" / "egoproactive_sved_sample350_fpi4_ctxi20_f80_hist20_majority1_viz"
DEFAULT_VIDEO_DIR = Path("/data1/wearable_ai_challenge_data/egoproactive/val")

LABEL_COLORS = {
    "interrupt": "#d95f02",
    "silent": "#1b9e77",
    "unknown": "#7570b3",
}
ERROR_COLOR = "#e7298a"
GRID_COLOR = "#d8dee9"
TEXT_COLOR = "#1f2933"
MUTED_COLOR = "#667085"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an HTML/SVG visualization report for egoproactive SVeD results."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--top-k-worst", type=int, default=10)
    parser.add_argument("--max-text-chars", type=int, default=180)
    return parser.parse_args()


def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    return records


def safe_div(num, den):
    return num / den if den else 0.0


def fmt_pct(value):
    return f"{value * 100:.1f}%"


def fmt_float(value, digits=3):
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def clean_answer(answer):
    if answer.startswith("$interrupt$"):
        return answer[len("$interrupt$") :]
    if answer.startswith("$silent$"):
        return ""
    return answer


def truncate(text, max_chars):
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def binary_metrics(tp, fn, fp):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
    }


def per_record_metrics(record):
    counts = Counter()
    intervals = record.get("interval_results", [])
    correct = 0
    for item in intervals:
        gt = item.get("gt_label", "unknown")
        pred = item.get("pred_label", "unknown")
        counts[(gt, pred)] += 1
        correct += int(gt == pred)

    tp = counts[("interrupt", "interrupt")]
    fn = counts[("interrupt", "silent")]
    fp = counts[("silent", "interrupt")]
    tn = counts[("silent", "silent")]
    interrupt = binary_metrics(tp, fn, fp)
    silent = binary_metrics(tn, fp, fn)
    macro_f1 = (interrupt["f1"] + silent["f1"]) / 2
    total = len(intervals)
    return {
        "video_path": record.get("video_path", ""),
        "domain": record.get("domain", ""),
        "task": record.get("task", ""),
        "accuracy": safe_div(correct, total),
        "correct": correct,
        "total": total,
        "interrupt_f1": interrupt["f1"],
        "silent_f1": silent["f1"],
        "macro_f1": macro_f1,
    }


def summarize(records):
    input_records = len(records)
    failed_records = [record for record in records if record.get("error")]
    records = [record for record in records if not record.get("error")]
    counts = Counter()
    per_domain = defaultdict(lambda: [0, 0])
    per_video = []
    label_counts = Counter()
    pred_counts = Counter()

    for record in records:
        intervals = record.get("interval_results", [])
        correct = 0
        for item in intervals:
            gt = item.get("gt_label", "unknown")
            pred = item.get("pred_label", "unknown")
            counts[(gt, pred)] += 1
            label_counts[gt] += 1
            pred_counts[pred] += 1
            if gt == pred:
                correct += 1
        total = len(intervals)
        domain = record.get("domain", "unknown")
        per_domain[domain][0] += correct
        per_domain[domain][1] += total
        per_video.append(per_record_metrics(record))

    tp = counts[("interrupt", "interrupt")]
    fn = counts[("interrupt", "silent")]
    fp = counts[("silent", "interrupt")]
    tn = counts[("silent", "silent")]
    total = tp + fn + fp + tn

    interrupt_precision = safe_div(tp, tp + fp)
    interrupt_recall = safe_div(tp, tp + fn)
    silent_precision = safe_div(tn, tn + fn)
    silent_recall = safe_div(tn, tn + fp)

    metrics = {
        "input_records": input_records,
        "records": len(records),
        "failed_records": len(failed_records),
        "failed_videos": [
            {
                "video_path": record.get("video_path", ""),
                "domain": record.get("domain", ""),
                "task": record.get("task", ""),
                "error_type": record.get("error_type", ""),
                "error_message": record.get("error_message", ""),
            }
            for record in failed_records
        ],
        "intervals": total,
        "accuracy": safe_div(tp + tn, total),
        "counts": counts,
        "label_counts": label_counts,
        "pred_counts": pred_counts,
        "per_video": per_video,
        "per_domain": {
            domain: {
                "correct": values[0],
                "total": values[1],
                "accuracy": safe_div(values[0], values[1]),
            }
            for domain, values in sorted(per_domain.items())
        },
        "interrupt": {
            "precision": interrupt_precision,
            "recall": interrupt_recall,
            "f1": safe_div(
                2 * interrupt_precision * interrupt_recall,
                interrupt_precision + interrupt_recall,
            ),
        },
        "silent": {
            "precision": silent_precision,
            "recall": silent_recall,
            "f1": safe_div(2 * silent_precision * silent_recall, silent_precision + silent_recall),
        },
    }
    return metrics


def extract_experiment_config(records):
    for record in records:
        config = record.get("experiment_config")
        if config:
            return config
    return {}


def get_nested(mapping, path, default=""):
    cur = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def config_table(config):
    if not config:
        return '<div class="empty-note">No experiment_config found in this result file.</div>'
    rows = [
        ("Protocol", get_nested(config, ["protocol"])),
        ("Frames per interval", get_nested(config, ["sampling", "frames_per_interval"])),
        ("Sampling strategy", get_nested(config, ["sampling", "strategy"])),
        ("Decision step", get_nested(config, ["sampling", "decision_step"])),
        ("Interval aggregation", get_nested(config, ["sampling", "interval_aggregation"])),
        ("Self-check frame", get_nested(config, ["sampling", "self_check_frame"])),
        ("Max context intervals", get_nested(config, ["context", "max_context_intervals"])),
        ("Max context frames", get_nested(config, ["context", "max_context_frames"])),
        ("Max history turns", get_nested(config, ["context", "max_history_turns"])),
        ("Context frame selection", get_nested(config, ["context", "frame_selection"])),
        ("Clear cache per video", get_nested(config, ["context", "clear_cache_per_video"])),
        ("Num samples", get_nested(config, ["data", "num_samples"])),
        ("Seed", get_nested(config, ["data", "seed"])),
        ("Alpha", get_nested(config, ["sved", "alpha"])),
        ("Num runs", get_nested(config, ["sved", "num_runs"])),
        ("Max new tokens", get_nested(config, ["generation", "max_new_tokens"])),
        ("Input size", get_nested(config, ["preprocess", "input_size"])),
        ("Model path", get_nested(config, ["model", "model_path"])),
        ("Weights dir", get_nested(config, ["model", "weights_dir"])),
        ("Annotation file", get_nested(config, ["data", "ann_file"])),
        ("Video dir", get_nested(config, ["data", "video_dir"])),
    ]
    return "\n".join(
        f"<tr><th>{esc(key)}</th><td>{esc(value)}</td></tr>"
        for key, value in rows
        if value != ""
    )


def select_worst_records(records, metrics, top_k):
    by_video = {item["video_path"]: item for item in metrics["per_video"]}
    records = [record for record in records if not record.get("error")]
    ranked = sorted(
        records,
        key=lambda record: (
            by_video.get(record.get("video_path", ""), {}).get("macro_f1", 0.0),
            by_video.get(record.get("video_path", ""), {}).get("accuracy", 0.0),
            record.get("video_path", ""),
        ),
    )
    return ranked[: max(0, top_k)]


def confusion_svg(metrics):
    counts = metrics["counts"]
    cells = [
        ("GT interrupt / Pred interrupt", counts[("interrupt", "interrupt")], 0, 0),
        ("GT interrupt / Pred silent", counts[("interrupt", "silent")], 1, 0),
        ("GT silent / Pred interrupt", counts[("silent", "interrupt")], 0, 1),
        ("GT silent / Pred silent", counts[("silent", "silent")], 1, 1),
    ]
    max_count = max([value for _, value, _, _ in cells] or [1])
    width, height = 620, 360
    left, top = 190, 70
    cell_w, cell_h = 180, 105
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Confusion matrix">',
        '<text x="350" y="28" text-anchor="middle" class="svg-title">Confusion Matrix</text>',
        '<text x="350" y="55" text-anchor="middle" class="svg-muted">Predicted label</text>',
        '<text x="80" y="185" text-anchor="middle" class="svg-muted" transform="rotate(-90 80 185)">Ground truth</text>',
        f'<text x="{left + cell_w * 0.5}" y="92" text-anchor="middle" class="svg-axis">interrupt</text>',
        f'<text x="{left + cell_w * 1.5}" y="92" text-anchor="middle" class="svg-axis">silent</text>',
        f'<text x="{left - 20}" y="{top + cell_h * 0.6}" text-anchor="end" class="svg-axis">interrupt</text>',
        f'<text x="{left - 20}" y="{top + cell_h * 1.6}" text-anchor="end" class="svg-axis">silent</text>',
    ]
    for label, value, col, row in cells:
        intensity = 0.12 + 0.72 * safe_div(value, max_count)
        color = f"rgba(37, 99, 235, {intensity:.3f})"
        x = left + col * cell_w
        y = top + row * cell_h + 35
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="{cell_w - 8}" height="{cell_h - 8}" rx="6" fill="{color}" stroke="#cbd5e1"/>',
                f'<title>{esc(label)}: {value}</title>',
                f'<text x="{x + (cell_w - 8) / 2}" y="{y + 44}" text-anchor="middle" class="svg-count">{value}</text>',
                f'<text x="{x + (cell_w - 8) / 2}" y="{y + 72}" text-anchor="middle" class="svg-muted">{esc(label.split(" / ")[1])}</text>',
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts)


def metric_bars_svg(metrics):
    series = [
        ("Overall Acc", metrics["accuracy"], "#2563eb"),
        ("Int Precision", metrics["interrupt"]["precision"], LABEL_COLORS["interrupt"]),
        ("Int Recall", metrics["interrupt"]["recall"], LABEL_COLORS["interrupt"]),
        ("Int F1", metrics["interrupt"]["f1"], LABEL_COLORS["interrupt"]),
        ("Sil Precision", metrics["silent"]["precision"], LABEL_COLORS["silent"]),
        ("Sil Recall", metrics["silent"]["recall"], LABEL_COLORS["silent"]),
        ("Sil F1", metrics["silent"]["f1"], LABEL_COLORS["silent"]),
    ]
    width, height = 820, 330
    left, top = 170, 48
    bar_h, gap = 24, 12
    chart_w = 560
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Metric bars">',
        '<text x="410" y="26" text-anchor="middle" class="svg-title">Aggregate Metrics</text>',
    ]
    for tick in range(0, 101, 25):
        x = left + chart_w * tick / 100
        parts.append(f'<line x1="{x}" y1="{top - 8}" x2="{x}" y2="{top + 252}" stroke="{GRID_COLOR}"/>')
        parts.append(f'<text x="{x}" y="{top + 276}" text-anchor="middle" class="svg-muted">{tick}%</text>')
    for idx, (name, value, color) in enumerate(series):
        y = top + idx * (bar_h + gap)
        parts.append(f'<text x="{left - 12}" y="{y + 17}" text-anchor="end" class="svg-axis">{esc(name)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{chart_w}" height="{bar_h}" rx="5" fill="#eef2f7"/>')
        parts.append(
            f'<rect x="{left}" y="{y}" width="{chart_w * max(0.0, min(1.0, value))}" height="{bar_h}" rx="5" fill="{color}"/>'
        )
        parts.append(f'<text x="{left + chart_w + 12}" y="{y + 17}" class="svg-axis">{fmt_pct(value)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def label_distribution_svg(metrics):
    gt = metrics["label_counts"]
    pred = metrics["pred_counts"]
    labels = ["interrupt", "silent"]
    width, height = 620, 260
    left, top = 150, 56
    chart_w = 360
    bar_h = 34
    max_count = max([gt[label] for label in labels] + [pred[label] for label in labels] + [1])
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Label distribution">',
        '<text x="310" y="26" text-anchor="middle" class="svg-title">Label Distribution</text>',
    ]
    rows = [("GT", gt), ("Pred", pred)]
    for row_idx, (row_name, counter) in enumerate(rows):
        y_base = top + row_idx * 86
        parts.append(f'<text x="70" y="{y_base + 24}" text-anchor="middle" class="svg-axis">{row_name}</text>')
        x = left
        for label in labels:
            value = counter[label]
            width_part = chart_w * safe_div(value, max_count)
            y = y_base + (0 if label == "interrupt" else 42)
            parts.append(f'<text x="{left - 12}" y="{y + 23}" text-anchor="end" class="svg-axis">{label}</text>')
            parts.append(f'<rect x="{left}" y="{y}" width="{chart_w}" height="{bar_h}" rx="5" fill="#eef2f7"/>')
            parts.append(
                f'<rect x="{x}" y="{y}" width="{width_part}" height="{bar_h}" rx="5" fill="{LABEL_COLORS[label]}"/>'
            )
            parts.append(f'<text x="{left + width_part + 8}" y="{y + 23}" class="svg-axis">{value}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def per_video_svg(rows, title="Worst Per-video Macro-F1"):
    rows = sorted(rows, key=lambda item: (item.get("macro_f1", 0.0), item.get("accuracy", 0.0)))
    width = 980
    row_h = 30
    height = 72 + max(1, len(rows)) * row_h
    left, top = 260, 42
    chart_w = 560
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Per-video accuracy">',
        f'<text x="490" y="24" text-anchor="middle" class="svg-title">{esc(title)}</text>',
    ]
    for tick in range(0, 101, 25):
        x = left + chart_w * tick / 100
        parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{height - 24}" stroke="{GRID_COLOR}"/>')
        parts.append(f'<text x="{x}" y="{height - 6}" text-anchor="middle" class="svg-muted">{tick}%</text>')
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        name = Path(row["video_path"]).stem
        macro_f1 = row.get("macro_f1", 0.0)
        accuracy = row.get("accuracy", 0.0)
        color = "#16a34a" if macro_f1 >= 0.75 else "#ca8a04" if macro_f1 >= 0.5 else "#dc2626"
        parts.append(f'<text x="{left - 10}" y="{y + 19}" text-anchor="end" class="svg-axis">{esc(name)}</text>')
        parts.append(f'<rect x="{left}" y="{y + 5}" width="{chart_w}" height="18" rx="4" fill="#eef2f7"/>')
        parts.append(
            f'<rect x="{left}" y="{y + 5}" width="{chart_w * macro_f1}" height="18" rx="4" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{left + chart_w + 10}" y="{y + 19}" class="svg-axis">macro-F1 {fmt_pct(macro_f1)} | acc {fmt_pct(accuracy)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def timeline_svg(record):
    intervals = record.get("interval_results", [])
    if not intervals:
        return ""
    max_time = max(
        [float(item.get("interval", [0, 0])[1]) for item in intervals]
        + [float(record.get("duration_in_sec") or 0)]
        + [1.0]
    )
    width, height = 1040, 156
    left, right = 86, 24
    chart_w = width - left - right
    gt_y, pred_y = 48, 92
    lane_h = 25
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="timeline" role="img" aria-label="Timeline for {esc(record.get("video_path", ""))}">',
        f'<text x="{left}" y="24" class="svg-axis">{esc(record.get("video_path", ""))}</text>',
        f'<text x="18" y="{gt_y + 17}" class="svg-axis">GT</text>',
        f'<text x="18" y="{pred_y + 17}" class="svg-axis">Pred</text>',
    ]
    tick_count = 5
    for i in range(tick_count + 1):
        t = max_time * i / tick_count
        x = left + chart_w * t / max_time
        parts.append(f'<line x1="{x}" y1="36" x2="{x}" y2="124" stroke="{GRID_COLOR}"/>')
        parts.append(f'<text x="{x}" y="146" text-anchor="middle" class="svg-muted">{t:.0f}s</text>')

    for lane, y, label_key in (("GT", gt_y, "gt_label"), ("Pred", pred_y, "pred_label")):
        for item in intervals:
            start, end = item.get("interval", [0, 0])
            start = float(start)
            end = float(end)
            x = left + chart_w * start / max_time
            w = max(2.0, chart_w * max(0.0, end - start) / max_time)
            label = item.get(label_key, "unknown")
            color = LABEL_COLORS.get(label, LABEL_COLORS["unknown"])
            opacity = "0.95" if lane == "Pred" else "0.72"
            stroke = ERROR_COLOR if lane == "Pred" and not item.get("correct") else "#ffffff"
            stroke_w = 2 if lane == "Pred" and not item.get("correct") else 1
            title = (
                f'{lane} interval {item.get("interval_index")}: {label}; '
                f'GT={item.get("gt_label")}; Pred={item.get("pred_label")}; '
                f'PPL={fmt_float(item.get("ppl"))}'
            )
            parts.append(
                f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="{lane_h}" rx="4" '
                f'fill="{color}" opacity="{opacity}" stroke="{stroke}" stroke-width="{stroke_w}">'
                f"<title>{esc(title)}</title></rect>"
            )
            if lane == "Pred" and not item.get("correct"):
                cx = x + w / 2
                cy = y + lane_h / 2
                parts.append(
                    f'<line x1="{cx - 5:.2f}" y1="{cy - 5:.2f}" x2="{cx + 5:.2f}" y2="{cy + 5:.2f}" stroke="#ffffff" stroke-width="2"/>'
                )
                parts.append(
                    f'<line x1="{cx + 5:.2f}" y1="{cy - 5:.2f}" x2="{cx - 5:.2f}" y2="{cy + 5:.2f}" stroke="#ffffff" stroke-width="2"/>'
                )
    parts.append("</svg>")
    return "\n".join(parts)


def domain_table(metrics):
    rows = []
    for domain, data in metrics["per_domain"].items():
        rows.append(
            "<tr>"
            f"<td>{esc(domain)}</td>"
            f"<td>{data['correct']}</td>"
            f"<td>{data['total']}</td>"
            f"<td>{fmt_pct(data['accuracy'])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def interval_table(record, max_text_chars):
    rows = []
    for item in record.get("interval_results", []):
        correct = bool(item.get("correct"))
        row_class = "ok" if correct else "bad"
        start, end = item.get("interval", ["", ""])
        gt_text = truncate(clean_answer(item.get("gt_answer", "")), max_text_chars)
        pred_text = truncate(item.get("generated_text", ""), max_text_chars)
        sampled_frames = item.get("sampled_frame_indices")
        sampled_times = item.get("sampled_times_sec")
        if sampled_frames is None:
            sampled_frames = [item.get("representative_frame_index", "")]
        if sampled_times is None:
            sampled_times = [item.get("representative_time_sec", "")]
        sampled_frames_text = ", ".join(str(frame) for frame in sampled_frames)
        sampled_times_text = ", ".join(fmt_float(time_value, 2) for time_value in sampled_times)
        sample_preds = [
            sample.get("pred_label", "")
            for sample in item.get("sample_results", [])
        ]
        sample_preds_text = ", ".join(sample_preds)
        votes_text = ""
        if "interrupt_votes" in item and "silent_votes" in item:
            votes_text = f'{item["interrupt_votes"]} int / {item["silent_votes"]} sil'
        rows.append(
            f'<tr class="{row_class}">'
            f'<td>{item.get("interval_index", "")}</td>'
            f'<td>{fmt_float(start, 1)}-{fmt_float(end, 1)}s</td>'
            f"<td>{esc(sampled_frames_text)}</td>"
            f"<td>{esc(sampled_times_text)}</td>"
            f"<td>{esc(sample_preds_text)}</td>"
            f"<td>{esc(votes_text)}</td>"
            f'<td><span class="pill {esc(item.get("gt_label", ""))}">{esc(item.get("gt_label", ""))}</span></td>'
            f'<td><span class="pill {esc(item.get("pred_label", ""))}">{esc(item.get("pred_label", ""))}</span></td>'
            f'<td>{"yes" if correct else "no"}</td>'
            f'<td>{fmt_float(item.get("ppl"))}</td>'
            f'<td>{fmt_float(item.get("decision_threshold"))}</td>'
            f'<td>{esc(gt_text)}</td>'
            f'<td>{esc(pred_text)}</td>'
            "</tr>"
        )
    return "\n".join(rows)


def video_index_html(records, metrics, video_links):
    by_video = {item["video_path"]: item for item in metrics["per_video"]}
    rows = []
    for idx, record in enumerate(records, 1):
        video_path = record.get("video_path", "")
        item = by_video.get(video_path, {})
        anchor = f"video-{idx}"
        link = video_links.get(video_path, {})
        rows.append(
            "<tr>"
            f'<td><a href="#{anchor}">{idx}</a></td>'
            f"<td>{esc(video_path)}</td>"
            f"<td>{fmt_pct(item.get('macro_f1', 0.0))}</td>"
            f"<td>{fmt_pct(item.get('accuracy', 0.0))}</td>"
            f"<td>{esc(record.get('domain', ''))}</td>"
            f'<td><a href="{esc(link.get("relative", ""))}">video</a></td>'
            f"<td><code>{esc(link.get('source', ''))}</code></td>"
            "</tr>"
        )
    return "\n".join(rows)


def failed_records_html(records, video_links):
    if not records:
        return '<div class="empty-note">No failed samples.</div>'
    rows = []
    for idx, record in enumerate(records, 1):
        video_path = record.get("video_path", "")
        link = video_links.get(video_path, {})
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{esc(video_path)}</td>"
            f"<td>{esc(record.get('error_type', ''))}</td>"
            f"<td>{esc(truncate(record.get('error_message', ''), 240))}</td>"
            f'<td><a href="{esc(link.get("relative", ""))}">video</a></td>'
            f"<td><code>{esc(link.get('source', ''))}</code></td>"
            "</tr>"
        )
    return (
        '<table class="domain-table">'
        "<thead><tr><th>#</th><th>Video</th><th>Error</th><th>Message</th><th>Link</th><th>Source</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def build_html(records, detail_records, failed_records, metrics, input_path, max_text_chars, experiment_config, video_links):
    frames_per_interval = get_nested(experiment_config, ["sampling", "frames_per_interval"], "n/a")
    cards = [
        ("Input Videos", metrics["input_records"]),
        ("Evaluated", metrics["records"]),
        ("Failed", metrics["failed_records"]),
        ("Intervals", metrics["intervals"]),
        ("Accuracy", fmt_pct(metrics["accuracy"])),
        ("Interrupt F1", fmt_pct(metrics["interrupt"]["f1"])),
        ("Silent F1", fmt_pct(metrics["silent"]["f1"])),
        ("Frames / Interval", frames_per_interval),
    ]
    card_html = "\n".join(
        f'<div class="card"><div class="card-label">{esc(label)}</div><div class="card-value">{esc(value)}</div></div>'
        for label, value in cards
    )

    record_sections = []
    by_video = {item["video_path"]: item for item in metrics["per_video"]}
    for idx, record in enumerate(detail_records, 1):
        acc = record.get("interval_accuracy", 0.0)
        correct = sum(1 for item in record.get("interval_results", []) if item.get("correct"))
        total = len(record.get("interval_results", []))
        video_path = record.get("video_path", "")
        video_link = video_links.get(video_path, {})
        metric = by_video.get(video_path, {})
        anchor = f"video-{idx}"
        record_sections.append(
            f"""
            <section class="video-section" id="{anchor}">
              <details open>
                <summary>
                  <span>{idx}. {esc(video_path)}</span>
                  <span class="summary-meta">macro-F1 {fmt_pct(metric.get("macro_f1", 0.0))} | acc {fmt_pct(acc)} ({correct}/{total}) | {esc(record.get("domain", ""))}</span>
                </summary>
                <div class="video-meta">
                  <div><strong>Task:</strong> {esc(record.get("task", ""))}</div>
                  <div><strong>Query:</strong> {esc(record.get("query", ""))}</div>
                  <div><strong>Video:</strong> <a href="{esc(video_link.get("relative", ""))}">{esc(video_link.get("relative", ""))}</a></div>
                  <div><code>{esc(video_link.get("source", ""))}</code></div>
                </div>
                <video class="video-player" controls preload="metadata" src="{esc(video_link.get("relative", ""))}"></video>
                {timeline_svg(record)}
                <div class="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>#</th><th>Interval</th><th>Frames</th><th>Times</th><th>Sample preds</th><th>Votes</th>
                        <th>GT</th><th>Pred</th><th>Correct</th><th>PPL</th><th>Gate</th>
                        <th>GT answer</th><th>Generated text</th>
                      </tr>
                    </thead>
                    <tbody>
                      {interval_table(record, max_text_chars)}
                    </tbody>
                  </table>
                </div>
              </details>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Egoproactive SVeD Visualization</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --text: {TEXT_COLOR};
      --muted: {MUTED_COLOR};
      --border: #d9e2ec;
      --interrupt: {LABEL_COLORS["interrupt"]};
      --silent: {LABEL_COLORS["silent"]};
      --error: {ERROR_COLOR};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 52px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    h2 {{ margin: 24px 0 12px; font-size: 18px; }}
    .source {{ color: var(--muted); margin-bottom: 18px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
    }}
    .card-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .card-value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      overflow: auto;
    }}
    svg {{ width: 100%; height: auto; }}
    .svg-title {{ font-size: 18px; font-weight: 700; fill: var(--text); }}
    .svg-axis {{ font-size: 13px; fill: var(--text); }}
    .svg-muted {{ font-size: 12px; fill: var(--muted); }}
    .svg-count {{ font-size: 30px; font-weight: 800; fill: #0f172a; }}
    .legend {{ display: flex; gap: 16px; color: var(--muted); margin: 8px 0 16px; }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 3px;
      margin-right: 6px;
      vertical-align: -1px;
    }}
    .legend .interrupt::before {{ background: var(--interrupt); }}
    .legend .silent::before {{ background: var(--silent); }}
    .legend .wrong::before {{ background: var(--error); }}
    .domain-table, .config-table {{ width: 100%; border-collapse: collapse; }}
    .domain-table td, .domain-table th, .config-table td, .config-table th {{
      border-bottom: 1px solid var(--border);
      padding: 8px;
      text-align: left;
    }}
    .config-table th {{ width: 220px; color: var(--muted); }}
    .empty-note {{ color: var(--muted); }}
    .video-section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      margin: 14px 0;
      overflow: hidden;
    }}
    summary {{
      cursor: pointer;
      padding: 12px 14px;
      font-weight: 700;
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }}
    .summary-meta {{ color: var(--muted); font-weight: 500; }}
    .video-meta {{ padding: 0 14px 10px; color: var(--muted); }}
    .video-player {{ display: block; width: min(720px, calc(100% - 28px)); max-height: 420px; margin: 0 14px 12px; background: #000; border-radius: 8px; }}
    .timeline {{ padding: 0 12px; }}
    .table-wrap {{ overflow-x: auto; padding: 0 14px 14px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 1120px; }}
    th, td {{ border-top: 1px solid var(--border); padding: 7px 8px; vertical-align: top; text-align: left; }}
    th {{ background: #f8fafc; font-size: 12px; color: var(--muted); }}
    tr.bad td {{ background: #fff5f8; }}
    .pill {{
      display: inline-block;
      min-width: 66px;
      text-align: center;
      padding: 2px 7px;
      border-radius: 999px;
      color: #fff;
      font-size: 12px;
      font-weight: 700;
    }}
    .pill.interrupt {{ background: var(--interrupt); }}
    .pill.silent {{ background: var(--silent); }}
    @media (max-width: 900px) {{
      .cards {{ grid-template-columns: repeat(2, 1fr); }}
      .grid {{ grid-template-columns: 1fr; }}
      summary {{ display: block; }}
      .summary-meta {{ display: block; margin-top: 4px; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>Egoproactive SVeD Visualization</h1>
  <div class="source">Source: {esc(input_path)}</div>
  <div class="cards">{card_html}</div>

  <h2>Experiment Config</h2>
  <div class="panel">
    <table class="config-table">
      <tbody>{config_table(experiment_config)}</tbody>
    </table>
  </div>

  <h2>Summary</h2>
  <div class="legend">
    <span class="interrupt">interrupt</span>
    <span class="silent">silent</span>
    <span class="wrong">wrong prediction outline / row</span>
  </div>
  <div class="grid">
    <div class="panel">{confusion_svg(metrics)}</div>
    <div class="panel">{metric_bars_svg(metrics)}</div>
    <div class="panel">{label_distribution_svg(metrics)}</div>
    <div class="panel">
      <h2 style="margin-top:0">Domain Accuracy</h2>
      <table class="domain-table">
        <thead><tr><th>Domain</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr></thead>
        <tbody>{domain_table(metrics)}</tbody>
      </table>
    </div>
  </div>

  <h2>Per-video Accuracy</h2>
  <div class="panel">{per_video_svg([by_video.get(record.get("video_path", ""), {}) for record in detail_records])}</div>

  <h2>Failed Samples</h2>
  <div class="panel">{failed_records_html(failed_records, video_links)}</div>

  <h2>Worst Video Index</h2>
  <div class="panel">
    <table class="domain-table">
      <thead><tr><th>#</th><th>Video</th><th>Macro-F1</th><th>Accuracy</th><th>Domain</th><th>Link</th><th>Source</th></tr></thead>
      <tbody>{video_index_html(detail_records, metrics, video_links)}</tbody>
    </table>
  </div>

  <h2>Worst Video Timelines And Intervals</h2>
  {''.join(record_sections)}
</main>
</body>
</html>
"""


def prepare_video_links(records, output_dir, video_dir):
    link_dir = output_dir / "videos"
    link_dir.mkdir(parents=True, exist_ok=True)
    links = {}
    for record in records:
        video_name = record.get("video_path", "")
        source = (video_dir / video_name).resolve()
        dest = link_dir / video_name
        if dest.exists() or dest.is_symlink():
            try:
                if dest.resolve() != source:
                    dest.unlink()
            except FileNotFoundError:
                dest.unlink()
        if not dest.exists() and not dest.is_symlink() and source.exists():
            dest.symlink_to(source)
        links[video_name] = {
            "source": str(source),
            "relative": f"videos/{video_name}",
        }
    return links


def write_report(records, metrics, input_path, output_dir, max_text_chars, experiment_config, top_k_worst, video_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_records = select_worst_records(records, metrics, top_k_worst)
    failed_records = [record for record in records if record.get("error")]
    video_links = prepare_video_links(detail_records + failed_records, output_dir, video_dir)
    html_text = build_html(
        records,
        detail_records,
        failed_records,
        metrics,
        input_path,
        max_text_chars,
        experiment_config,
        video_links,
    )
    report_path = output_dir / "index.html"
    report_path.write_text(html_text, encoding="utf-8")

    summary_path = output_dir / "summary.json"
    summary = {
        "experiment_config": experiment_config,
        "input_records": metrics["input_records"],
        "records": metrics["records"],
        "failed_records": metrics["failed_records"],
        "failed_videos": metrics["failed_videos"],
        "intervals": metrics["intervals"],
        "accuracy": metrics["accuracy"],
        "interrupt": metrics["interrupt"],
        "silent": metrics["silent"],
        "confusion": {
            f"{gt}_as_{pred}": value for (gt, pred), value in sorted(metrics["counts"].items())
        },
        "per_domain": metrics["per_domain"],
        "per_video": metrics["per_video"],
        "detail_videos": [record.get("video_path", "") for record in detail_records],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path, summary_path


def main():
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    video_dir = args.video_dir.expanduser().resolve()
    records = load_jsonl(input_path)
    metrics = summarize(records)
    experiment_config = extract_experiment_config(records)
    report_path, summary_path = write_report(
        records,
        metrics,
        input_path,
        output_dir,
        args.max_text_chars,
        experiment_config,
        args.top_k_worst,
        video_dir,
    )
    print(
        f"Loaded {metrics['input_records']} videos; "
        f"evaluated {metrics['records']}, failed {metrics['failed_records']}; "
        f"{metrics['intervals']} intervals"
    )
    print(f"Accuracy: {fmt_pct(metrics['accuracy'])}")
    print(f"HTML report: {report_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
