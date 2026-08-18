"""
engine/generic.py — Generic EDA for arbitrary CSV uploads.

Runs on any DataFrame with no assumptions about column names.
Returns a JSON-safe dict with:
  - shape / dtypes
  - missing value report
  - descriptive statistics (numeric columns)
  - top value counts (categorical columns)
  - Pearson correlation matrix (numeric columns)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_generic_eda(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape

    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # --- missing values ---
    missing = (
        df.isnull()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_pct=lambda x: (x["missing_count"] / n_rows * 100).round(2))
        .query("missing_count > 0")
        .reset_index()
        .rename(columns={"index": "column"})
        .to_dict(orient="records")
    )

    # --- descriptive stats (numeric) ---
    if numeric_cols:
        desc = (
            df[numeric_cols]
            .describe()
            .round(4)
            .reset_index()
            .rename(columns={"index": "stat"})
            .to_dict(orient="records")
        )
    else:
        desc = []

    # --- top value counts (categorical, up to 5 cols, top 10 each) ---
    value_counts = {}
    for col in categorical_cols[:5]:
        vc = df[col].value_counts().head(10)
        value_counts[col] = [
            {"value": str(k), "count": int(v)} for k, v in vc.items()
        ]

    # --- correlation matrix (numeric) ---
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(method="pearson").round(4)
        correlations = corr.reset_index().rename(columns={"index": "column"}).to_dict(orient="records")
    else:
        correlations = []

    # --- column type summary ---
    col_summary = [
        {
            "column":   col,
            "dtype":    str(df[col].dtype),
            "n_unique": int(df[col].nunique()),
            "missing":  int(df[col].isnull().sum()),
        }
        for col in df.columns
    ]

    return {
        "mode":             "generic",
        "n_rows":           n_rows,
        "n_cols":           n_cols,
        "numeric_cols":     numeric_cols,
        "categorical_cols": categorical_cols,
        "col_summary":      col_summary,
        "missing":          missing,
        "describe":         desc,
        "value_counts":     value_counts,
        "correlations":     correlations,
    }
