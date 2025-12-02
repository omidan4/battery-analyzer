"""
Utilities for reading the inverter CSV exports with consistent normalization.

The logger produces Windows-formatted CSV files that include:
* A leading 'sep=,' line
* UTF-8 BOM
* Numeric columns suffixed with unit text (e.g., '0.04 Volts')
* Verbose column names that we want to standardize

This module centralizes the parsing rules so downstream scripts can rely on a
clean schema independent of the raw export quirks.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


# Precompiled regex that captures the first float-like token in a cell.
FLOAT_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class ColumnRule:
    """Describes how a raw CSV column should be handled."""

    canonical_name: str
    strip_units: Sequence[str] = field(default_factory=tuple)
    required: bool = False
    numeric: bool = True


def _detect_header(reader) -> Sequence[str]:
    """Return header row taking into account an optional 'sep=,' preface."""
    first_line = reader.readline()
    if not first_line:
        return []
    lower = first_line.strip().lower()
    if lower.startswith("sep="):
        raw_header = reader.readline()
        return next(csv.reader([raw_header]))
    reader.seek(0)
    return next(csv.reader([reader.readline()]))


def _open_csv(path: Path):
    """Open CSV in a context manager with BOM-safe encoding."""
    return path.open("r", encoding="utf-8-sig", newline="")


def _normalize_value(
    raw_value: Optional[str],
    strip_units: Sequence[str],
) -> Optional[str]:
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value:
        return None
    for unit in strip_units:
        if value.endswith(unit):
            value = value[: -len(unit)].strip()
    return value or None


def _coerce_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if FLOAT_PATTERN.fullmatch(value):
        try:
            return float(value)
        except ValueError:
            return None
    match = FLOAT_PATTERN.search(value)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def load_sensor_csv(
    path: Path,
    column_map: Dict[str, ColumnRule],
    allow_extra_columns: bool = True,
) -> List[Dict[str, Optional[float]]]:
    """
    Load a CSV and normalize column names plus numeric values.

    Args:
        path: CSV file path.
        column_map: Mapping from raw column names to ColumnRule definitions.
        allow_extra_columns: When True, unknown columns are kept with their
            original names; otherwise they raise a ValueError.

    Returns:
        List of dictionaries holding normalized numeric values (floats) or None.
    """
    rows: List[Dict[str, Optional[float]]] = []
    with _open_csv(path) as fh:
        header = _detect_header(fh)
        reader = csv.DictReader(fh, fieldnames=header)
        for raw_row in reader:
            normalized_row: Dict[str, Optional[float]] = {}
            for raw_name, raw_value in raw_row.items():
                rule = column_map.get(raw_name)
                if rule is None:
                    if not allow_extra_columns:
                        raise ValueError(f"Unexpected column '{raw_name}' in {path}")
                    canonical_name = raw_name.strip()
                    strip_units: Sequence[str] = ()
                    numeric = True
                else:
                    canonical_name = rule.canonical_name
                    strip_units = rule.strip_units
                    numeric = rule.numeric
                normalized_str = _normalize_value(raw_value, strip_units)
                normalized_row[canonical_name] = (
                    _coerce_float(normalized_str) if numeric else normalized_str
                )
            rows.append(normalized_row)

    _validate_required_columns(path, rows, column_map.values())
    return rows


def _validate_required_columns(
    path: Path,
    rows: Sequence[Dict[str, Optional[float]]],
    rules: Iterable[ColumnRule],
) -> None:
    if not rows:
        return
    missing_columns = []
    for rule in rules:
        if not rule.required:
            continue
        if all(rule.canonical_name not in row or row[rule.canonical_name] is None for row in rows):
            missing_columns.append(rule.canonical_name)
    if missing_columns:
        raise ValueError(
            f"{path} missing required columns with data: {', '.join(sorted(missing_columns))}"
        )


def default_column_rules() -> Dict[str, ColumnRule]:
    """Factory for the standard schema used across inverter CSVs."""
    return {
        "Time": ColumnRule("timestamp", required=True, numeric=False),
        "INV_DC_Bus_Voltage": ColumnRule("inv_dc_bus_voltage", strip_units=("Volts",)),
        "INV_DC_Bus_Current": ColumnRule("inv_dc_bus_current", strip_units=("Amps",)),
        "INV_Phase_A_Current": ColumnRule("inv_phase_a_current"),
        "INV_Phase_B_Current": ColumnRule("inv_phase_b_current"),
        "INV_Phase_C_Current": ColumnRule("inv_phase_c_current"),
        "INV_Control_Board_Temp": ColumnRule("inv_control_board_temp"),
        "INV_Coolant_Temp": ColumnRule("inv_coolant_temp"),
        "INV_Gate_Driver_Board_Temp": ColumnRule("inv_gate_driver_board_temp"),
        "INV_Hot_Spot_Temp": ColumnRule("inv_hot_spot_temp"),
        "INV_Module_A_Temp": ColumnRule("inv_module_a_temp"),
        "INV_Module_B_Temp": ColumnRule("inv_module_b_temp"),
        "INV_Module_C_Temp": ColumnRule("inv_module_c_temp"),
        'sensorReading {messageName="M165_Motor_Position_Info", rawCAN="165", signalName="INV_Motor_Speed"}': ColumnRule(
            "inv_motor_speed"
        ),
    }


__all__ = [
    "ColumnRule",
    "default_column_rules",
    "load_sensor_csv",
]
