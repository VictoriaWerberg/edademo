"""
engine/validate.py — Schema validation layer (Milestone 2).

Uses pandera to define expected schemas for each table and flag rows
that violate them. Invalid rows are quarantined, not silently dropped.

Usage
-----
    from engine.validate import validate_all
    clean, quarantine = validate_all(tables)
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
import pandera as pa
from pandera.pandas import Column, DataFrameSchema, Check


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

measurements_schema = DataFrameSchema(
    columns={
        "meas_id":           Column(int,   Check.greater_than_or_equal_to(0), nullable=False),
        "lot_id":            Column(str,   Check.str_matches(r"^LOT\d{3}$"),  nullable=False),
        "wafer_id":          Column(str,   Check.str_matches(r"^LOT\d{3}-W\d{2}$"), nullable=False),
        "die_id":            Column(int,   Check.between(1, 200),             nullable=False),
        "via_id":            Column(int,   Check.greater_than_or_equal_to(1), nullable=False),
        "via_type":          Column(str,   Check.isin(["copper", "tungsten"]), nullable=False),
        "layer":             Column(str,   Check.isin(["M1", "M2", "M3", "M4"]), nullable=False),
        "resistance_ohm":    Column(float, Check.between(0.0, 10.0),          nullable=False),
        "continuity_pass":   Column(bool,                                      nullable=False),
        "mean_humidity_pct": Column(float, Check.between(0.0, 100.0),         nullable=False),
        "anneal_temp_c":     Column(float, Check.between(300.0, 500.0),       nullable=False),
        "meas_timestamp":    Column(str,                                       nullable=False),
    },
    coerce=True,
)

process_schema = DataFrameSchema(
    columns={
        "lot_id":             Column(str,   Check.str_matches(r"^LOT\d{3}$"), nullable=False),
        "wafer_id":           Column(str,   Check.str_matches(r"^LOT\d{3}-W\d{2}$"), nullable=False),
        "deposition_temp_c":  Column(float, Check.between(250.0, 400.0),      nullable=False),
        "etch_time_s":        Column(float, Check.between(30.0, 200.0),        nullable=False),
        "anneal_temp_c":      Column(float, Check.between(300.0, 500.0),       nullable=False),
        "pressure_torr":      Column(float, Check.between(0.001, 0.05),        nullable=False),
        "operator_id":        Column(str,   Check.str_matches(r"^OP\d{2}$"),   nullable=False),
        "process_timestamp":  Column(str,                                       nullable=False),
    },
    coerce=True,
)

design_schema = DataFrameSchema(
    columns={
        "via_type":             Column(str,   Check.isin(["copper", "tungsten"]),    nullable=False),
        "layer":                Column(str,   Check.isin(["M1", "M2", "M3", "M4"]), nullable=False),
        "pitch_um":             Column(float, Check.between(0.05, 5.0),              nullable=False),
        "aspect_ratio":         Column(float, Check.between(1.0, 20.0),              nullable=False),
        "metal_stack":          Column(str,                                           nullable=False),
        "nominal_resistance_ohm": Column(float, Check.between(0.01, 50.0),           nullable=False),
    },
    coerce=True,
)

environment_schema = DataFrameSchema(
    columns={
        "env_timestamp":      Column(str,   nullable=False),
        "temperature_c":      Column(float, Check.between(15.0, 30.0),  nullable=False),
        "humidity_pct":       Column(float, Check.between(0.0, 100.0),  nullable=False),
        "particle_count_m3":  Column(int,   Check.greater_than_or_equal_to(0), nullable=False),
        "station_id":         Column(str,   Check.isin(["STA1", "STA2", "STA3"]), nullable=False),
    },
    coerce=True,
)

_SCHEMAS = {
    "measurements": measurements_schema,
    "process":      process_schema,
    "design":       design_schema,
    "environment":  environment_schema,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_all(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Validate each table against its pandera schema.

    Rows that fail any check are moved to the quarantine dict; the clean
    dict contains only rows that passed every check.

    Parameters
    ----------
    tables:
        Output of ``engine.io.load_raw``.

    Returns
    -------
    (clean, quarantine)
        Both are dicts keyed by table name.
    """
    clean: Dict[str, pd.DataFrame] = {}
    quarantine: Dict[str, pd.DataFrame] = {}

    for name, df in tables.items():
        schema = _SCHEMAS.get(name)
        if schema is None:
            # Unknown table — pass through unchanged, no quarantine
            clean[name] = df.copy()
            quarantine[name] = pd.DataFrame()
            continue

        try:
            # lazy=True collects ALL failures before raising, so we can
            # extract the bad row indices rather than stopping at the first.
            schema.validate(df, lazy=True)
            clean[name] = df.copy()
            quarantine[name] = pd.DataFrame()

        except pa.errors.SchemaErrors as exc:
            bad_idx = set(exc.failure_cases["index"].dropna().astype(int).tolist())
            quarantine[name] = df.loc[df.index.isin(bad_idx)].copy()
            clean[name]      = df.loc[~df.index.isin(bad_idx)].copy()

    return clean, quarantine
