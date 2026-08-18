"""
engine/generic.py — Data profiling for any CSV upload.

Returns a thorough JSON-safe profile with no assumptions about column names:
  - shape, duplicates, constant columns
  - per-column type, cardinality, missing count
  - numeric: describe, skewness, kurtosis, IQR outlier count, sample histogram
  - categorical: top values, rare values (<1 %), unique-ID flag
  - date columns: detected and summarised (min, max, span)
  - high-correlation pairs (|r| > 0.7)
  - sample rows (first 5)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_generic_eda(df: pd.DataFrame) -> dict:
    """Profile any DataFrame and return a JSON-safe exploration report."""
    df = _coerce_dates(df.copy())
    n_rows, n_cols = df.shape

    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    date_cols        = df.select_dtypes(include="datetime").columns.tolist()

    return {
        "mode":            "generic",
        "shape":           {"rows": n_rows, "cols": n_cols},
        "duplicate_rows":  int(df.duplicated().sum()),
        "constant_cols":   [c for c in df.columns if df[c].nunique() <= 1],
        "col_types": {
            "numeric":     numeric_cols,
            "categorical": categorical_cols,
            "datetime":    date_cols,
        },
        "columns":         _profile_columns(df, numeric_cols, date_cols),
        "correlations":    _correlations(df, numeric_cols),
        "sample_rows":     _sample_rows(df),
    }


# ---------------------------------------------------------------------------
# Column profiling
# ---------------------------------------------------------------------------

def _profile_columns(df, numeric_cols, date_cols):
    profiles = []

    for col in df.columns:
        series   = df[col]
        n_miss   = int(series.isnull().sum())
        n_unique = int(series.nunique())
        base = {
            "name":        col,
            "dtype":       str(series.dtype),
            "n_unique":    n_unique,
            "missing":     n_miss,
            "missing_pct": round(n_miss / len(df) * 100, 1),
        }

        if col in numeric_cols:
            base["kind"]  = "numeric"
            base.update(_numeric_profile(series))

        elif col in date_cols:
            base["kind"]  = "datetime"
            base.update(_date_profile(series))

        else:
            base["kind"]  = "categorical"
            base.update(_categorical_profile(series, len(df)))

        profiles.append(base)

    return profiles


def _numeric_profile(s: pd.Series) -> dict:
    clean = s.dropna()
    if clean.empty:
        return {}

    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr    = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out  = int(((clean < lo) | (clean > hi)).sum())

    # 10-bin histogram (counts only — no numpy array in JSON)
    counts, edges = np.histogram(clean, bins=10)
    hist = [
        {"bin_start": round(float(edges[i]), 4), "count": int(counts[i])}
        for i in range(len(counts))
    ]

    return {
        "min":      round(float(clean.min()),  4),
        "max":      round(float(clean.max()),  4),
        "mean":     round(float(clean.mean()), 4),
        "median":   round(float(clean.median()), 4),
        "std":      round(float(clean.std()),  4),
        "skewness": round(float(clean.skew()), 4),
        "kurtosis": round(float(clean.kurt()), 4),
        "n_outliers_iqr": n_out,
        "histogram": hist,
    }


def _categorical_profile(s: pd.Series, n_rows: int) -> dict:
    vc      = s.value_counts(dropna=False)
    top10   = [{"value": str(k), "count": int(v), "pct": round(v / n_rows * 100, 1)}
               for k, v in vc.head(10).items()]
    rare    = int((vc < max(1, n_rows * 0.01)).sum())   # values appearing in <1 % of rows
    is_id   = s.nunique() >= 0.95 * n_rows              # looks like a unique ID column

    return {
        "top_values":  top10,
        "rare_values": rare,
        "looks_like_id": bool(is_id),
    }


def _date_profile(s: pd.Series) -> dict:
    clean = s.dropna()
    if clean.empty:
        return {}
    span = clean.max() - clean.min()
    return {
        "min":       str(clean.min()),
        "max":       str(clean.max()),
        "span_days": span.days,
    }


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def _correlations(df: pd.DataFrame, numeric_cols: list) -> dict:
    if len(numeric_cols) < 2:
        return {"matrix": [], "high_pairs": []}

    corr = df[numeric_cols].corr(method="pearson").round(4)

    # High-correlation pairs (|r| > 0.7, excluding self)
    pairs = []
    cols  = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                pairs.append({"col_a": cols[i], "col_b": cols[j], "r": float(r)})
    pairs.sort(key=lambda x: abs(x["r"]), reverse=True)

    matrix = corr.reset_index().rename(columns={"index": "column"}).to_dict(orient="records")

    return {"matrix": matrix, "high_pairs": pairs, "numeric_cols": numeric_cols}


# ---------------------------------------------------------------------------
# Sample rows
# ---------------------------------------------------------------------------

def _sample_rows(df: pd.DataFrame) -> dict:
    sample = df.head(5).copy()
    # Convert everything to strings so JSON serialises cleanly
    for col in sample.select_dtypes(include="datetime").columns:
        sample[col] = sample[col].astype(str)
    return {
        "columns": sample.columns.tolist(),
        "rows":    sample.fillna("").astype(str).values.tolist(),
    }


# ---------------------------------------------------------------------------
# Date coercion (best-effort)
# ---------------------------------------------------------------------------

def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Try to parse string columns that look like dates."""
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(50)
        # Only attempt if values look date-like
        if sample.str.match(
            r"^\d{4}[-/]\d{2}[-/]\d{2}|^\d{2}[-/]\d{2}[-/]\d{4}"
        ).mean() > 0.8:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df
