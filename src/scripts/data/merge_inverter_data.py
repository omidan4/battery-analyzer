#!/usr/bin/env python3
"""Create a per-second merged dataset from the June 3 inverter logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

from src import aggregation, data_loading


DATASETS = [
    (
        "phase",
        Path("data/Inverter Phase Currents-data-as-joinbyfield-2025-10-03 17_38_30.csv"),
    ),
    (
        "temps",
        Path("data/Inverter Temps-data-as-joinbyfield-2025-10-03 17_39_12.csv"),
    ),
    (
        "dc",
        Path(
            "data/Inverter Voltage and Current-data-as-joinbyfield-2025-10-03 17_38_50.csv"
        ),
    ),
]


def aggregate_with_prefix(prefix: str, path: Path, rules) -> Dict[str, Dict[str, float]]:
    rows = data_loading.load_sensor_csv(path, rules)
    aggregates = aggregation.aggregate_rows_to_seconds(rows)
    keyed: Dict[str, Dict[str, float]] = {}
    for entry in aggregates:
        ts = entry["timestamp"]
        prefixed = {}
        for key, value in entry.items():
            if key == "timestamp":
                continue
            if key == "raw_sample_count":
                new_key = f"{prefix}_raw_sample_count"
            else:
                new_key = f"{prefix}_{key}" if key.startswith("raw_") else key
            prefixed[new_key] = value
        keyed[ts] = prefixed
    return keyed


def merge_aggregates(
    datasets: Iterable[Dict[str, Dict[str, float]]],
) -> List[Dict[str, float]]:
    timestamps = set()
    for data in datasets:
        timestamps.update(data.keys())
    merged_rows: List[Dict[str, float]] = []
    for ts in sorted(timestamps):
        row: Dict[str, float] = {"timestamp": ts}
        for data in datasets:
            if ts in data:
                row.update(data[ts])
        merged_rows.append(row)
    return merged_rows


def write_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    if not rows:
        print("No rows to write.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="clean/inverter_merged_1hz.csv",
        help="Output CSV path (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rules = data_loading.default_column_rules()
    aggregates = [
        aggregate_with_prefix(prefix, path, rules) for prefix, path in DATASETS
    ]
    merged = merge_aggregates(aggregates)
    write_csv(merged, Path(args.output))


if __name__ == "__main__":
    main()
