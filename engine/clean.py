"""
engine/clean.py — Cleaning and join logic (Milestone 3).

Takes validated tables and produces a single analysis-ready DataFrame:

  measurements
    ├── LEFT JOIN process_log      on [lot_id, wafer_id]
    ├── LEFT JOIN design_manifest  on [via_type, layer]
    └── (env data already denormalised onto measurements at generation time)

Derived columns added here
--------------------------
  resistance_ratio   : resistance_ohm / nominal_resistance_ohm
  anneal_group       : "high" (>420 °C) | "low" (≤420 °C)   — ANOVA factor 1
  humidity_group     : "high" (>47 %)   | "low" (≤47 %)     — ANOVA factor 2
  continuity_int     : 1 / 0 integer version of continuity_pass

Usage
-----
    from engine.clean import build_analysis_frame
    df = build_analysis_frame(clean_tables)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

# Thresholds that define the two hidden-pattern boundaries
ANNEAL_THRESHOLD  = 420.0   # °C
HUMIDITY_THRESHOLD = 47.0   # %


def build_analysis_frame(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join and clean the four validated tables into one analysis frame.

    Parameters
    ----------
    tables:
        Output of ``engine.validate.validate_all`` (clean side).

    Returns
    -------
    Single merged DataFrame ready for ``engine.stats``.
    """
    meas   = tables["measurements"].copy()
    proc   = tables["process"].copy()
    design = tables["design"].copy()

    # ------------------------------------------------------------------
    # 1. Parse timestamps
    # ------------------------------------------------------------------
    meas["meas_timestamp"]    = pd.to_datetime(meas["meas_timestamp"])
    proc["process_timestamp"] = pd.to_datetime(proc["process_timestamp"])

    # ------------------------------------------------------------------
    # 2. Join: measurements ← process_log
    #    Brings in deposition_temp_c, etch_time_s, pressure_torr, operator_id
    #    (anneal_temp_c is already on measurements, keep process copy for check)
    # ------------------------------------------------------------------
    proc_cols = [
        "wafer_id",
        "deposition_temp_c",
        "etch_time_s",
        "pressure_torr",
        "operator_id",
    ]
    df = meas.merge(proc[proc_cols], on="wafer_id", how="left")

    # ------------------------------------------------------------------
    # 3. Join: measurements ← design_manifest
    #    Brings in pitch_um, aspect_ratio, nominal_resistance_ohm, metal_stack
    # ------------------------------------------------------------------
    design_cols = [
        "via_type",
        "layer",
        "pitch_um",
        "aspect_ratio",
        "nominal_resistance_ohm",
        "metal_stack",
    ]
    df = df.merge(design[design_cols], on=["via_type", "layer"], how="left")

    # ------------------------------------------------------------------
    # 4. Derived columns
    # ------------------------------------------------------------------
    # Resistance relative to the nominal for that via_type × layer cell
    df["resistance_ratio"] = (
        df["resistance_ohm"] / df["nominal_resistance_ohm"]
    ).round(4)

    # ANOVA factors — categorical splits at the pattern thresholds
    df["anneal_group"]   = np.where(df["anneal_temp_c"] > ANNEAL_THRESHOLD,  "high", "low")
    df["humidity_group"] = np.where(df["mean_humidity_pct"] > HUMIDITY_THRESHOLD, "high", "low")

    # Numeric version of the boolean for logistic / correlation use
    df["continuity_int"] = df["continuity_pass"].astype(int)

    # ------------------------------------------------------------------
    # 5. Column ordering — analytical columns up front
    # ------------------------------------------------------------------
    leading = [
        "meas_id", "lot_id", "wafer_id", "die_id", "via_id",
        "via_type", "layer", "metal_stack",
        "resistance_ohm", "nominal_resistance_ohm", "resistance_ratio",
        "continuity_pass", "continuity_int",
        "anneal_temp_c", "anneal_group",
        "mean_humidity_pct", "humidity_group",
        "deposition_temp_c", "etch_time_s", "pressure_torr",
        "pitch_um", "aspect_ratio",
        "operator_id", "meas_timestamp",
    ]
    remaining = [c for c in df.columns if c not in leading]
    df = df[leading + remaining]

    # ------------------------------------------------------------------
    # 6. Final dtype coercions
    # ------------------------------------------------------------------
    for cat_col in ("via_type", "layer", "metal_stack", "anneal_group",
                    "humidity_group", "operator_id", "lot_id"):
        df[cat_col] = df[cat_col].astype("category")

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Quick summary helper (used by the API)
# ---------------------------------------------------------------------------

def frame_summary(df: pd.DataFrame) -> dict:
    """Return a plain-dict summary of the analysis frame for JSON serialisation."""
    return {
        "n_rows":             int(len(df)),
        "n_wafers":           int(df["wafer_id"].nunique()),
        "n_lots":             int(df["lot_id"].nunique()),
        "via_types":          sorted(df["via_type"].cat.categories.tolist()),
        "layers":             sorted(df["layer"].cat.categories.tolist()),
        "resistance_mean":    round(float(df["resistance_ohm"].mean()), 4),
        "resistance_std":     round(float(df["resistance_ohm"].std()),  4),
        "continuity_pass_pct": round(float(df["continuity_pass"].mean() * 100), 2),
        "anneal_high_pct":    round(float((df["anneal_group"] == "high").mean() * 100), 1),
        "humidity_high_pct":  round(float((df["humidity_group"] == "high").mean() * 100), 1),
    }
