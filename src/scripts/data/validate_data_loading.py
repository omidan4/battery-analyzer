#!/usr/bin/env python3
"""
Quick validation script for the data loading and aggregation utilities.

Usage:
    python scripts/validate_data_loading.py
"""
from pathlib import Path

from src import aggregation, data_loading


def main() -> None:
    data_dir = Path("data")
    rules = data_loading.default_column_rules()
    for csv_path in sorted(data_dir.glob("*.csv")):
        rows = data_loading.load_sensor_csv(csv_path, rules)
        aggregates = aggregation.aggregate_rows_to_seconds(rows)
        print(
            f"{csv_path.name}: raw_rows={len(rows):5d}, aggregates={len(aggregates):4d}"
        )


if __name__ == "__main__":
    main()
