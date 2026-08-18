"""
engine/stats.py — Statistical analysis layer (Milestone 4).

Runs ANOVA, effect-size estimation (eta-squared / Cohen's d), and
correlation analysis on the cleaned analysis frame to surface the two
hidden patterns baked into the synthetic data.

Usage
-----
    from engine.stats import run_analysis
    results = run_analysis(df)
"""

from __future__ import annotations

import pandas as pd


def run_analysis(df: pd.DataFrame) -> dict:
    """Run the full statistics suite on the cleaned analysis frame.

    Parameters
    ----------
    df:
        Output of ``engine.clean.build_analysis_frame``.

    Returns
    -------
    dict with keys:
        - ``anova``      : ANOVA table(s) as DataFrames
        - ``effect_size``: effect-size estimates per factor
        - ``correlations``: pairwise correlation matrix
        - ``summary``    : plain-English findings dict (consumed by the API)
    """
    # Milestone 4: implement ANOVA and effect-size logic.
    raise NotImplementedError("Stats layer is implemented in Milestone 4.")
