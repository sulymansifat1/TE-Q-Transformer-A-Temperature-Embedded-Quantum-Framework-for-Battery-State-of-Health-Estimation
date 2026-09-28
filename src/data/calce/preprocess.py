"""Build NASA-contract CALCE CS2 discharge windows from official Arbin Excel logs.

Each accepted window is one complete 1C discharge, linearly resampled to 512
phase points, with feature order [Voltage_V, Current_A, Temperature_C, Time_norm].
Temperature is a constant 23 C ambient fill because the official CS2 Excel files
have no measured temperature column. No scaler is fitted here.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import pandas as pd


CELL_IDS = ("CS2_35", "CS2_36", "CS2_37", "CS2_38")
FEATURE_ORDER = ("Voltage_V", "Current_A", "Temperature_C", "Time_norm")
DEFAULT_SEQUENCE_LENGTH = 512
AMBIENT_TEMPERATURE_C = 23.0
DISCHARGE_CURRENT_THRESHOLD_A = -0.4
MIN_DISCHARGE_POINTS = 16
MIN_CAPACITY_AH = 0.15
MAX_CAPACITY_AH = 1.30
MIN_START_VOLTAGE_V = 3.6
MAX_END_VOLTAGE_V = 3.2
FILE_NAME_RE = re.compile(r"^CS2_\d+_(\d+)_(\d+)_(\d+)\.xlsx$", re.IGNORECASE)
NEEDED_COLUMNS = (
    "Test_Time(s)",
    "Step_Index",
    "Cycle_Index",
    "Current(A)",
    "Voltage(V)",
    "Discharge_Capacity(Ah)",
)


def _parse_file_date(path: Path) -> datetime:
    match = FILE_NAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected CALCE filename: {path.name}")
    month, day, year = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    year = 2000 + year if year < 100 else year
    return datetime(year, month, day)


def _cell_excel_files(raw_root: Path, cell_id: str) -> list[Path]:
    cell_dir = raw_root / cell_id / cell_id
    if not cell_dir.is_dir():
        raise FileNotFoundError(cell_dir)
    files = [path for path in cell_dir.glob("*.xlsx") if not path.name.startswith("~$")]
    if not files:
        raise FileNotFoundError(f"No Excel logs in {cell_dir}")
    return sorted(files, key=_parse_file_date)


def _xlsx_sheet_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("xl/workbook.xml")
    root = ElementTree.fromstring(xml)
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    names = [el.get("name") for el in root.findall(".//m:sheet", ns)]
    return [name for name in names if name]


def _read_excel(path: Path, sheet: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for engine in ("calamine", "openpyxl"):
        try:
            return pd.read_excel(path, sheet_name=sheet, engine=engine)
        except Exception as exc:  # noqa: BLE001 - engine fallback is intentional
            last_error = exc
    raise RuntimeError(f"Failed to read {path}") from last_error


def _channel_sheet_name(path: Path) -> str:
    for name in _xlsx_sheet_names(path):
        if str(name).startswith("Channel"):
            return str(name)
    raise ValueError(f"No Channel sheet in {path}")


def _read_channel(path: Path) -> pd.DataFrame:
    sheet = _channel_sheet_name(path)
    frame = _read_excel(path, sheet)
    missing = [name for name in NEEDED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns {missing}; have {list(frame.columns)}")
    return frame.loc[:, list(NEEDED_COLUMNS)].copy()


def _cycle_capacity_ah(capacity: np.ndarray) -> float:
    """Per-discharge Ah. Arbin Discharge_Capacity may reset or accumulate."""
    return float(abs(capacity[-1] - capacity[0])) if capacity.size else 0.0


def _validate_discharge(
    time_s: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    capacity_ah: float,
) -> list[str]:
    reasons: list[str] = []
    if time_s.size < MIN_DISCHARGE_POINTS:
        reasons.append(f"fewer than {MIN_DISCHARGE_POINTS} discharge samples")
    arrays = (time_s, voltage, current)
    if any(not np.all(np.isfinite(values)) for values in arrays) or not np.isfinite(capacity_ah):
        reasons.append("non-finite raw value")
    if time_s.size:
        diffs = np.diff(time_s)
        if diffs.size and np.any(diffs < 0):
            reasons.append("decreasing Test_Time(s) inside discharge")
    if float(np.mean(current)) >= DISCHARGE_CURRENT_THRESHOLD_A:
        reasons.append("mean current is not a 1C-scale discharge")
    if capacity_ah < MIN_CAPACITY_AH:
        reasons.append("discharge capacity below minimum")
    if capacity_ah > MAX_CAPACITY_AH:
        reasons.append("discharge capacity exceeds CS2 physical range")
    if float(voltage[0]) < MIN_START_VOLTAGE_V:
        reasons.append("discharge does not start near the charged voltage")
    if float(voltage[-1]) > MAX_END_VOLTAGE_V:
        reasons.append("discharge does not reach a low enough cutoff voltage")
    return reasons


def _resample_discharge(
    time_s: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    duration = float(time_s[-1] - time_s[0])
    if duration <= 0:
        raise ValueError("Cannot resample a discharge with non-positive duration")
    source_phase = (time_s - time_s[0]) / duration
    # Guard duplicate timestamps by averaging via interpolation on unique times.
    unique_phase, unique_idx = np.unique(source_phase, return_index=True)
    unique_voltage = voltage[unique_idx]
    unique_current = current[unique_idx]
    target_phase = np.linspace(0.0, 1.0, sequence_length, dtype=np.float64)
    window = np.column_stack(
        [
            np.interp(target_phase, unique_phase, unique_voltage),
            np.interp(target_phase, unique_phase, unique_current),
            np.full(sequence_length, AMBIENT_TEMPERATURE_C, dtype=np.float64),
            target_phase,
        ]
    )
    return window.astype(np.float32)


def _discharge_mask(cycle: pd.DataFrame) -> np.ndarray:
    current = cycle["Current(A)"].to_numpy(dtype=np.float64)
    step = cycle["Step_Index"].to_numpy(dtype=np.float64)
    by_current = current < DISCHARGE_CURRENT_THRESHOLD_A
    discharge_steps = []
    for step_id in np.unique(step[by_current]):
        step_rows = current[step == step_id]
        if float(np.mean(step_rows)) < DISCHARGE_CURRENT_THRESHOLD_A:
            discharge_steps.append(step_id)
    if discharge_steps:
        return np.isin(step, np.asarray(discharge_steps)) & by_current
    return by_current


def extract_cell_discharges(
    raw_root: Path,
    cell_id: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    windows: list[np.ndarray] = []
    capacities: list[float] = []
    soh_values: list[float] = []
    cycles: list[int] = []
    durations: list[float] = []
    n_points: list[int] = []
    sources: list[str] = []
    excluded: list[dict[str, Any]] = []
    seen_windows: set[bytes] = set()
    global_cycle = 0

    for path in _cell_excel_files(raw_root, cell_id):
        print(f"  reading {cell_id} {path.name}", flush=True)
        frame = _read_channel(path)
        for cycle_index, cycle in frame.groupby("Cycle_Index", sort=True):
            mask = _discharge_mask(cycle)
            if not np.any(mask):
                excluded.append(
                    {
                        "cell_id": cell_id,
                        "file": path.name,
                        "file_cycle_index": int(cycle_index),
                        "reasons": ["no discharge current samples"],
                    }
                )
                continue
            discharge = cycle.loc[mask]
            time_s = discharge["Test_Time(s)"].to_numpy(dtype=np.float64)
            voltage = discharge["Voltage(V)"].to_numpy(dtype=np.float64)
            current = discharge["Current(A)"].to_numpy(dtype=np.float64)
            capacity_raw = discharge["Discharge_Capacity(Ah)"].to_numpy(dtype=np.float64)
            capacity_ah = _cycle_capacity_ah(capacity_raw)
            reasons = _validate_discharge(time_s, voltage, current, capacity_ah)
            if reasons:
                excluded.append(
                    {
                        "cell_id": cell_id,
                        "file": path.name,
                        "file_cycle_index": int(cycle_index),
                        "reasons": reasons,
                    }
                )
                continue
            window = _resample_discharge(time_s, voltage, current, sequence_length)
            digest = np.round(window, 6).tobytes()
            if digest in seen_windows:
                excluded.append(
                    {
                        "cell_id": cell_id,
                        "file": path.name,
                        "file_cycle_index": int(cycle_index),
                        "reasons": ["duplicate resampled window"],
                    }
                )
                continue
            seen_windows.add(digest)
            global_cycle += 1
            windows.append(window)
            capacities.append(capacity_ah)
            cycles.append(global_cycle)
            durations.append(float(time_s[-1] - time_s[0]))
            n_points.append(int(time_s.size))
            sources.append(path.name)

    if not windows:
        raise ValueError(f"{cell_id} produced no valid discharge windows")
    capacity_arr = np.asarray(capacities, dtype=np.float32)
    c0 = float(capacity_arr[0])
    soh_values = (capacity_arr / c0).astype(np.float32)
    data = {
        "X_unscaled": np.stack(windows).astype(np.float32),
        "soh": soh_values,
        "capacity_ah": capacity_arr,
        "cycle": np.asarray(cycles, dtype=np.int32),
        "duration_seconds": np.asarray(durations, dtype=np.float32),
        "n_raw_points": np.asarray(n_points, dtype=np.int32),
        "source_file": np.asarray(sources),
        "c0_ah": np.asarray([c0], dtype=np.float32),
    }
    return data, excluded


def build_calce_dataset(
    raw_root: Path,
    *,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    cells: tuple[str, ...] = CELL_IDS,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    excluded_all: list[dict[str, Any]] = []
    for cell_id in cells:
        data, excluded = extract_cell_discharges(raw_root, cell_id, sequence_length)
        output[cell_id] = data
        excluded_all.extend(excluded)

    metadata: dict[str, Any] = {
        "dataset": "CALCE CS2",
        "source": "https://web.calce.umd.edu/batteries/data/",
        "raw_root": str(raw_root),
        "cells": list(cells),
        "feature_order": list(FEATURE_ORDER),
        "shape_contract": f"[N, {sequence_length}, 4]",
        "sequence_length": sequence_length,
        "window_semantics": (
            "one complete 1C discharge resampled by linear interpolation onto "
            "512 normalized phase points; windows never cross a file or Cycle_Index"
        ),
        "current_provenance": "measured Arbin Current(A); discharge negative",
        "temperature_policy": (
            f"constant {AMBIENT_TEMPERATURE_C} C ambient fill; Excel logs have no temperature column"
        ),
        "time_policy": "within-discharge Test_Time(s) converted to unitless phase 0..1",
        "soh_formula": "per-discharge Ah increment / first valid discharge Ah of that cell",
        "capacity_definition": "abs(last-first Discharge_Capacity(Ah) within the discharge step)",
        "scaling_policy": "no scaler fitted; output is unscaled physical units plus Time_norm",
        "discharge_current_threshold_A": DISCHARGE_CURRENT_THRESHOLD_A,
        "min_capacity_ah": MIN_CAPACITY_AH,
        "max_capacity_ah": MAX_CAPACITY_AH,
        "excluded_count": len(excluded_all),
        "excluded_segments": excluded_all,
        "cell_window_counts": {
            cell: int(values["X_unscaled"].shape[0]) for cell, values in output.items()
        },
        "initial_capacity_ah": {
            cell: float(values["c0_ah"][0]) for cell, values in output.items()
        },
    }
    return output, metadata


def save_calce_dataset(
    data: dict[str, dict[str, np.ndarray]],
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for cell_id, values in data.items():
        np.save(output_dir / f"{cell_id}_X_unscaled.npy", values["X_unscaled"], allow_pickle=False)
        np.save(output_dir / f"{cell_id}_soh.npy", values["soh"], allow_pickle=False)
        np.save(output_dir / f"{cell_id}_capacity_ah.npy", values["capacity_ah"], allow_pickle=False)
        np.save(output_dir / f"{cell_id}_cycle.npy", values["cycle"], allow_pickle=False)
        np.save(output_dir / f"{cell_id}_duration_seconds.npy", values["duration_seconds"], allow_pickle=False)
        np.save(output_dir / f"{cell_id}_n_raw_points.npy", values["n_raw_points"], allow_pickle=False)
        np.save(output_dir / f"{cell_id}_source_file.npy", values["source_file"], allow_pickle=True)
    excluded = metadata.get("excluded_segments", [])
    slim = {key: value for key, value in metadata.items() if key != "excluded_segments"}
    slim["excluded_count"] = len(excluded)
    (output_dir / "preprocess_metadata.json").write_text(
        json.dumps(slim, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "excluded_segments.jsonl").open("w", encoding="utf-8") as handle:
        for row in excluded:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("Dataset/calce/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Dataset/calce/processed_nasa_contract"),
    )
    args = parser.parse_args()
    data, metadata = build_calce_dataset(args.raw_root)
    save_calce_dataset(data, metadata, args.output_dir)
    print(json.dumps({k: metadata[k] for k in ("cells", "cell_window_counts", "initial_capacity_ah", "excluded_count")}, indent=2))


if __name__ == "__main__":
    main()
