"""
Append execution time from a benchmark JSON to a metrics CSV.

Usage:
  python src/add_execution_time_to_csv.py \
    --csv results/v1/rag/gpt.41/benchmark_v2_improved_metrics_20250722_231047.csv \
    --json results/v1/rag/gpt.41/benchmark_v2_improved_results_20250722_231047.json \
    --out results/v1/rag/gpt.41/benchmark_v2_improved_metrics_20250722_231047_with_time.csv

Notes:
  - Joins on the `task_id` column in the CSV against entries in JSON["results"].
  - Adds a new column `execution_time_seconds` (float). Missing entries become NaN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import csv


def load_execution_times(json_path: Path) -> Dict[str, float]:
    with json_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    results = data.get("results", [])
    mapping: Dict[str, float] = {}

    for item in results:
        task_id = item.get("task_id")
        exec_time = item.get("execution_time_seconds")
        if task_id is not None and exec_time is not None:
            mapping[task_id] = float(exec_time)

    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Append execution_time_seconds to metrics CSV.")
    parser.add_argument("--csv", required=True, type=Path, help="Path to metrics CSV file")
    parser.add_argument("--json", required=True, type=Path, help="Path to benchmark results JSON file")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <csv_basename>_with_time.csv in same directory)",
    )

    args = parser.parse_args()

    csv_path: Path = args.csv
    json_path: Path = args.json
    out_path: Path | None = args.out

    if out_path is None:
        out_path = csv_path.with_name(csv_path.stem + "_with_time" + csv_path.suffix)

    # Load CSV with stdlib csv
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "task_id" not in fieldnames:
        raise ValueError("CSV must contain a 'task_id' column to join on.")

    exec_times = load_execution_times(json_path)

    matched = 0
    for row in rows:
        task_id = row.get("task_id")
        exec_time = exec_times.get(task_id)
        if exec_time is not None:
            matched += 1
            row["execution_time_seconds"] = exec_time
        else:
            row["execution_time_seconds"] = ""

    if "execution_time_seconds" not in fieldnames:
        fieldnames.append("execution_time_seconds")

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path} — matched {matched}/{len(rows)} rows with execution_time_seconds.")


if __name__ == "__main__":
    main()
