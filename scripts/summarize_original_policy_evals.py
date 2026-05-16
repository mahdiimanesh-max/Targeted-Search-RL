#!/usr/bin/env python3
"""Summarize only OriginalPolicy behavior across generated-policy eval JSONLs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


BUCKETS = [
    "useful_correct",
    "redundant_correct",
    "no_search_correct",
    "distractor_wrong",
    "other_wrong",
]


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    denom = sum(exps)
    return [value / denom for value in exps]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def original_policy_metrics(records: list[dict]) -> dict[str, float]:
    by_case: dict[int, list[dict]] = {}
    for record in records:
        by_case.setdefault(int(record["case_id"]), []).append(record)

    totals = {bucket: 0.0 for bucket in BUCKETS}
    for case_records in by_case.values():
        weights = softmax([float(record["old_logp"]) for record in case_records])
        for record, weight in zip(case_records, weights):
            bucket = str(record["bucket"])
            if bucket not in totals:
                bucket = "other_wrong"
            totals[bucket] += weight

    denom = max(len(by_case), 1)
    metrics = {bucket: totals[bucket] / denom for bucket in BUCKETS}
    metrics["correct"] = (
        metrics["useful_correct"]
        + metrics["redundant_correct"]
        + metrics["no_search_correct"]
    )
    metrics["useful_red"] = metrics["useful_correct"] - metrics["redundant_correct"]
    metrics["num_cases"] = float(len(by_case))
    metrics["num_samples"] = float(len(records))
    return metrics


def parse_run(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Runs must use LABEL=PATH format.")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Run label cannot be empty.")
    return label, Path(path)


def print_table(rows: list[dict[str, str]]) -> None:
    headers = [
        "model",
        "useful",
        "redundant",
        "no_search",
        "distractor",
        "other",
        "correct",
        "useful-red",
        "samples",
    ]
    widths = [
        max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    ]
    print("OriginalPolicy model comparison")
    print("=" * 76)
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[header].ljust(widths[idx]) for idx, header in enumerate(headers)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="Evaluation JSONL in LABEL=PATH format. Can be repeated.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV path for the summarized OriginalPolicy metrics.",
    )
    args = parser.parse_args()

    rows = []
    csv_rows = []
    for label, path in args.run:
        metrics = original_policy_metrics(read_jsonl(path))
        row = {
            "model": label,
            "useful": f"{metrics['useful_correct']:.3f}",
            "redundant": f"{metrics['redundant_correct']:.3f}",
            "no_search": f"{metrics['no_search_correct']:.3f}",
            "distractor": f"{metrics['distractor_wrong']:.3f}",
            "other": f"{metrics['other_wrong']:.3f}",
            "correct": f"{metrics['correct']:.3f}",
            "useful-red": f"{metrics['useful_red']:+.3f}",
            "samples": f"{int(metrics['num_samples'])}",
        }
        rows.append(row)
        csv_rows.append(
            {
                "model": label,
                "useful": metrics["useful_correct"],
                "redundant": metrics["redundant_correct"],
                "no_search": metrics["no_search_correct"],
                "distractor": metrics["distractor_wrong"],
                "other": metrics["other_wrong"],
                "correct": metrics["correct"],
                "useful_red": metrics["useful_red"],
                "num_cases": int(metrics["num_cases"]),
                "num_samples": int(metrics["num_samples"]),
            }
        )

    print_table(rows)

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print()
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
