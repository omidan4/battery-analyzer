#!/usr/bin/env python3
"""Run basic QA checks against the merged 1 Hz dataset."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="clean/inverter_merged_1hz.csv",
        help="Path to merged CSV (default: %(default)s)",
    )
    return parser.parse_args()


def try_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def run_checks(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    with path.open() as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []
        numeric_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"missing": 0, "non_numeric": 0, "min": math.inf, "max": -math.inf}
        )
        row_count = 0
        timestamps = []
        for row in reader:
            row_count += 1
            ts = row.get("timestamp")
            if ts:
                timestamps.append(ts)
            for key, value in row.items():
                if key == "timestamp":
                    continue
                stats = numeric_stats[key]
                if value is None or not value.strip():
                    stats["missing"] += 1
                    continue
                parsed = try_float(value)
                if parsed is None:
                    stats["non_numeric"] += 1
                    continue
                stats["min"] = min(stats["min"], parsed)
                stats["max"] = max(stats["max"], parsed)

    unique_timestamps = len(set(timestamps))
    print(f"Rows: {row_count}")
    print(f"Unique timestamps: {unique_timestamps}")
    print(f"Duplicate timestamps: {row_count - unique_timestamps}")

    sorted_timestamps = sorted(timestamps)
    out_of_order = sum(
        1
        for observed, expected in zip(timestamps, sorted_timestamps)
        if observed != expected
    )
    print(f"Timestamps out of order: {out_of_order}")

    print("\nColumn stats:")
    for column in columns:
        if column == "timestamp":
            continue
        stats = numeric_stats[column]
        missing = int(stats["missing"])
        non_numeric = int(stats["non_numeric"])
        min_val = None if stats["min"] is math.inf else stats["min"]
        max_val = None if stats["max"] == -math.inf else stats["max"]
        print(
            f"  {column}: missing={missing}, non_numeric={non_numeric}, min={min_val}, max={max_val}"
        )


def main() -> None:
    args = parse_args()
    run_checks(Path(args.input))


if __name__ == "__main__":
    main()
