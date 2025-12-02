"""
Timestamp alignment and per-second aggregation helpers.

These utilities take the normalized rows from `data_loading.load_sensor_csv`
and collapse the rapid-fire samples (multiple per second) into a stable 1 Hz
time base with summary statistics per signal.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


class TimestampParseError(ValueError):
    """Raised when a timestamp string cannot be parsed with known formats."""


def parse_timestamp(value: str) -> datetime:
    """Parse timestamp strings that may or may not include milliseconds."""
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise TimestampParseError(f"Unsupported timestamp format: {value!r}")


@dataclass
class AggregatedStat:
    sum: float = 0.0
    count: int = 0
    min: Optional[float] = None
    max: Optional[float] = None

    def update(self, value: float) -> None:
        if self.min is None or value < self.min:
            self.min = value
        if self.max is None or value > self.max:
            self.max = value
        self.sum += value
        self.count += 1

    def to_dict(self, prefix: str) -> Dict[str, float]:
        if self.count == 0 or self.min is None or self.max is None:
            return {}
        mean = self.sum / self.count
        return {
            f"{prefix}_mean": mean,
            f"{prefix}_min": self.min,
            f"{prefix}_max": self.max,
            f"{prefix}_count": float(self.count),
        }


def aggregate_rows_to_seconds(
    rows: Sequence[Mapping[str, Optional[float]]],
    *,
    timestamp_key: str = "timestamp",
    numeric_columns: Optional[Iterable[str]] = None,
) -> List[Dict[str, float]]:
    """
    Collapse raw rows into 1 Hz aggregates keyed by floored timestamps.

    Args:
        rows: normalized dictionaries (output of the loader).
        timestamp_key: name of the timestamp column.
        numeric_columns: optional whitelist of columns to aggregate; if omitted,
                         all numeric-looking keys except timestamp are used.

    Returns:
        Sorted list of dictionaries, each representing one-second aggregate.
    """
    aggregates: Dict[datetime, Dict[str, AggregatedStat]] = defaultdict(dict)
    counts: Dict[datetime, int] = defaultdict(int)
    predefined_columns = list(numeric_columns) if numeric_columns is not None else None

    for row in rows:
        raw_ts = row.get(timestamp_key)
        if raw_ts is None or not isinstance(raw_ts, str):
            continue
        ts = parse_timestamp(raw_ts)
        ts_floor = ts.replace(microsecond=0)
        counts[ts_floor] += 1

        row_columns = predefined_columns or [key for key in row.keys() if key != timestamp_key]
        for col in row_columns:
            value = row.get(col)
            if value is None:
                continue
            bucket = aggregates.setdefault(ts_floor, {})
            stats = bucket.setdefault(col, AggregatedStat())
            stats.update(float(value))

    output: List[Dict[str, float]] = []
    for ts in sorted(counts.keys()):
        entry: Dict[str, float] = {
            "timestamp": ts.isoformat(sep=" "),
            "raw_sample_count": float(counts[ts]),
        }
        for col, stats in aggregates.get(ts, {}).items():
            entry.update(stats.to_dict(col))
        output.append(entry)

    return output


__all__ = [
    "aggregate_rows_to_seconds",
    "parse_timestamp",
    "TimestampParseError",
]
