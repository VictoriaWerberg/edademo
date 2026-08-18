"""
engine/stats.py — Statistical analysis layer (Milestone 4).

Surfaces the two hidden patterns in the synthetic data:

  Pattern 1 — via_type × anneal_group interaction on resistance_ratio
              Two-way ANOVA with eta-squared effect sizes.

  Pattern 2 — humidity_group effect on continuity yield
              Chi-square test + odds ratio + phi coefficient.

Additional outputs
------------------
  - Group means table (resistance by via_type × anneal_group)
  - Pearson correlation matrix for continuous predictors
  - Plain-English ``findings`` dict consumed by the API / site

Usage
-----
    from engine.stats import run_analysis
    results = run_analysis(df)          # df from engine.clean.build_analysis_frame
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eta_squared(anova_table: pd.DataFrame) -> pd.Series:
    """Partial eta-squared: SS_effect / SS_total for each term."""
    ss_total = anova_table["sum_sq"].sum()
    return (anova_table["sum_sq"] / ss_total).rename("eta_sq")


def _odds_ratio(ct: pd.DataFrame) -> float:
    """Odds ratio from a 2×2 contingency table (pass vs fail × group)."""
    # ct rows: humidity_group (high/low), cols: continuity_int (0/1)
    a = ct.loc["high", 1]   # high humidity, pass
    b = ct.loc["high", 0]   # high humidity, fail
    c = ct.loc["low",  1]   # low humidity,  pass
    d = ct.loc["low",  0]   # low humidity,  fail
    if b == 0 or c == 0:
        return float("nan")
    return float((a / b) / (c / d))


def _phi(ct: pd.DataFrame) -> float:
    """Phi coefficient (effect size for 2×2 chi-square)."""
    chi2, *_ = stats.chi2_contingency(ct, correction=False)
    n = ct.values.sum()
    return float(np.sqrt(chi2 / n))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Run the full statistics suite on the cleaned analysis frame.

    Parameters
    ----------
    df:
        Output of ``engine.clean.build_analysis_frame``.

    Returns
    -------
    dict with keys:
        anova           DataFrame  — two-way ANOVA table (Type II SS)
        group_means     DataFrame  — mean resistance_ratio per via_type × anneal_group
        chi2_yield      dict       — chi-square result for humidity → yield
        correlations    DataFrame  — Pearson r matrix (continuous vars)
        findings        dict       — plain-English key results (JSON-safe)
    """
    results: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # 1. Two-way ANOVA: via_type × anneal_group → resistance_ratio        #
    # ------------------------------------------------------------------ #
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ols(
            "resistance_ratio ~ C(via_type) * C(anneal_group)",
            data=df,
        ).fit()

    anova_table = anova_lm(model, typ=2)
    anova_table["eta_sq"] = _eta_squared(anova_table)
    anova_table = anova_table.round(6)
    results["anova"] = anova_table

    # Group means (the pattern in numbers)
    group_means = (
        df.groupby(["via_type", "anneal_group"], observed=True)["resistance_ratio"]
        .agg(mean="mean", std="std", n="count")
        .round(4)
        .reset_index()
    )
    results["group_means"] = group_means

    # ------------------------------------------------------------------ #
    # 2. Chi-square: humidity_group × continuity_pass                     #
    # ------------------------------------------------------------------ #
    ct = pd.crosstab(df["humidity_group"], df["continuity_int"])
    # Ensure both columns (0 and 1) exist even if all rows pass
    for col in [0, 1]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[[0, 1]]   # fail | pass

    chi2_val, p_val, dof, *_ = stats.chi2_contingency(ct, correction=False)

    yield_by_group = (
        df.groupby("humidity_group", observed=True)["continuity_pass"]
        .mean()
        .mul(100)
        .round(2)
        .to_dict()
    )

    results["chi2_yield"] = {
        "chi2":             round(float(chi2_val), 4),
        "p_value":          round(float(p_val), 6),
        "dof":              int(dof),
        "odds_ratio":       round(_odds_ratio(ct), 4),
        "phi":              round(_phi(ct), 4),
        "yield_pct_by_group": yield_by_group,
    }

    # ------------------------------------------------------------------ #
    # 3. Pearson correlation matrix (continuous predictors)               #
    # ------------------------------------------------------------------ #
    corr_cols = [
        "resistance_ratio",
        "anneal_temp_c",
        "mean_humidity_pct",
        "deposition_temp_c",
        "etch_time_s",
        "pressure_torr",
        "aspect_ratio",
        "pitch_um",
    ]
    results["correlations"] = df[corr_cols].corr(method="pearson").round(4)

    # ------------------------------------------------------------------ #
    # 4. Plain-English findings (JSON-safe scalars only)                  #
    # ------------------------------------------------------------------ #
    interaction_row = anova_table.loc["C(via_type):C(anneal_group)"]
    main_via_row    = anova_table.loc["C(via_type)"]

    # Tungsten high vs low anneal uplift
    gm = group_means.set_index(["via_type", "anneal_group"])["mean"]
    w_high = float(gm.get(("tungsten", "high"), float("nan")))
    w_low  = float(gm.get(("tungsten", "low"),  float("nan")))
    c_high = float(gm.get(("copper",   "high"), float("nan")))
    c_low  = float(gm.get(("copper",   "low"),  float("nan")))
    tungsten_uplift_pct = round((w_high - w_low) / w_low * 100, 1) if w_low else float("nan")
    copper_delta_pct    = round((c_high - c_low) / c_low * 100, 1) if c_low else float("nan")

    results["findings"] = {
        # Pattern 1
        "pattern1_title": "Tungsten via resistance elevates at high anneal temperature",
        "pattern1_interaction_F":  round(float(interaction_row["F"]), 2),
        "pattern1_interaction_p":  round(float(interaction_row["PR(>F)"]), 6),
        "pattern1_interaction_eta_sq": round(float(interaction_row["eta_sq"]), 4),
        "pattern1_tungsten_uplift_pct": tungsten_uplift_pct,
        "pattern1_copper_delta_pct":    copper_delta_pct,
        "pattern1_significant":  bool(float(interaction_row["PR(>F)"]) < 0.05),

        # Pattern 2
        "pattern2_title": "Cleanroom humidity excursion reduces interconnect yield",
        "pattern2_chi2":      results["chi2_yield"]["chi2"],
        "pattern2_p":         results["chi2_yield"]["p_value"],
        "pattern2_odds_ratio":results["chi2_yield"]["odds_ratio"],
        "pattern2_phi":       results["chi2_yield"]["phi"],
        "pattern2_yield_high_humidity": yield_by_group.get("high"),
        "pattern2_yield_low_humidity":  yield_by_group.get("low"),
        "pattern2_significant": bool(results["chi2_yield"]["p_value"] < 0.05),
    }

    return results


# ---------------------------------------------------------------------------
# Serialisation helper (DataFrames → dicts for JSON)
# ---------------------------------------------------------------------------

def results_to_json(results: dict[str, Any]) -> dict[str, Any]:
    """Convert DataFrames inside the results dict to JSON-serialisable dicts."""
    out = {}
    for key, val in results.items():
        if isinstance(val, pd.DataFrame):
            out[key] = val.reset_index().to_dict(orient="records")
        else:
            out[key] = val
    return out
