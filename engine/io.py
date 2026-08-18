"""
engine/io.py — Data ingestion layer.

Reads the four raw CSV files produced by the fab/telemetry system and
returns them as a dict of DataFrames, keyed by table name.

Expected files
--------------
- interconnect_measurements.csv  : per-via resistance / continuity readings
- process_log.csv                : fab process parameters per lot/wafer
- design_manifest.csv            : design metadata (layer, pitch, via type)
- environmental_log.csv          : cleanroom temp/humidity timeseries

Usage
-----
    from engine.io import load_raw
    tables = load_raw("data/")
"""

from __future__ import annotations

import pathlib
from typing import Dict

import pandas as pd


_EXPECTED_FILES = {
    "measurements": "interconnect_measurements.csv",
    "process":      "process_log.csv",
    "design":       "design_manifest.csv",
    "environment":  "environmental_log.csv",
}


def load_raw(data_dir: str | pathlib.Path) -> Dict[str, pd.DataFrame]:
    """Load all four source CSVs from *data_dir*.

    Parameters
    ----------
    data_dir:
        Path to the folder that contains the four CSV files.

    Returns
    -------
    dict mapping table name → DataFrame (columns/dtypes not yet validated).
    """
    data_dir = pathlib.Path(data_dir)
    tables: Dict[str, pd.DataFrame] = {}

    for name, filename in _EXPECTED_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected data file not found: {path}")
        tables[name] = pd.read_csv(path)

    return tables
