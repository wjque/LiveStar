import argparse
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "evaluate" / "output" / "egoproactive_sved_sample10_fpi4.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "evaluate" / "output" / "egoproactive_sved_sample10_fpi4_majority1.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-aggregate saved multi-frame proactive SVeD results without rerunning the model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--interrupt-threshold", type=int, default=1)
    return parser.parse_args()


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def aggregation_label(sample_results, interrupt_threshold):
    interrupt_votes = sum(
        sample.get("pred_label") == "interrupt" for sample in sample_results
    )
    return (
        "interrupt" if interrupt_votes >= interrupt_threshold else "silent",
        interrupt_votes,
        len(sample_results) - interrupt_votes,
    )


def aggregation_description(interrupt_threshold):
    return (
        f"majority={interrupt_threshold}; interrupt if at least "
        f"{interrupt_threshold} sampled frame(s) interrupt, otherwise silent"
    )


def reaggregate(rows, interrupt_threshold):
    counts = Counter()
    for row in rows:
        config = row.get("experiment_config")
        if config:
            config.setdefault("sampling", {})[
                "interval_aggregation"
            ] = aggregation_description(interrupt_threshold)

        correct = 0
        intervals = row.get("interval_results", [])
        for item in intervals:
            sample_results = item.get("sample_results", [])
            if not sample_results:
                pred_label = item.get("pred_label", "silent")
                interrupt_votes = int(pred_label == "interrupt")
                silent_votes = int(pred_label == "silent")
            else:
                pred_label, interrupt_votes, silent_votes = aggregation_label(
                    sample_results, interrupt_threshold
                )

            item["pred_label"] = pred_label
            item["interrupt_votes"] = interrupt_votes
            item["silent_votes"] = silent_votes
            item["correct"] = pred_label == item.get("gt_label")
            correct += int(item["correct"])
            counts[(item.get("gt_label"), pred_label)] += 1

        row["interval_accuracy"] = correct / len(intervals) if intervals else 0.0

    return rows, counts


def safe_div(num, den):
    return num / den if den else 0.0


def print_metrics(counts):
    tp = counts[("interrupt", "interrupt")]
    fn = counts[("interrupt", "silent")]
    fp = counts[("silent", "interrupt")]
    tn = counts[("silent", "silent")]
    total = tp + fn + fp + tn
    print(f"intervals: {total}")
    print(f"accuracy : {safe_div(tp + tn, total):.4f}")
    print(
        "confusion:",
        {
            "gt_interrupt_pred_interrupt": tp,
            "gt_interrupt_pred_silent": fn,
            "gt_silent_pred_interrupt": fp,
            "gt_silent_pred_silent": tn,
        },
    )


def main():
    args = parse_args()
    rows = load_jsonl(args.input)
    rows, counts = reaggregate(rows, args.interrupt_threshold)
    write_jsonl(args.output, rows)
    print_metrics(counts)
    print(f"Saved re-aggregated results to {args.output}")


if __name__ == "__main__":
    main()
