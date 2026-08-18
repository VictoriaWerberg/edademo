"""
engine/clean.py — Cleaning and join logic (Milestone 3).

Takes validated tables and produces a single analysis-ready DataFrame
by merging on shared keys, handling missing values, and normalising units.

Usage
-----
    from engine.clean import build_analysis_frame
    df = build_analysis_frame(clean_tables)
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


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
    # Milestone 3: implement join logic, unit normalisation, outlier handling.
    raise NotImplementedError("Cleaning logic is implemented in Milestone 3.")
