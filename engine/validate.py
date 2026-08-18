"""
engine/validate.py — Schema validation layer (Milestone 2).

Uses pandera to define expected schemas for each table and flag rows
that violate them. Invalid rows are quarantined, not silently dropped.

Schemas defined here
--------------------
- measurements_schema
- process_schema
- design_schema
- environment_schema

Usage
-----
    from engine.validate import validate_all
    clean, quarantine = validate_all(tables)
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

# pandera imported lazily so the module loads even if not installed yet
try:
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check
    _PA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Schemas (to be expanded in Milestone 2)
# ---------------------------------------------------------------------------

measurements_schema: "pa.DataFrameSchema | None" = None  # placeholder
process_schema:      "pa.DataFrameSchema | None" = None
design_schema:       "pa.DataFrameSchema | None" = None
environment_schema:  "pa.DataFrameSchema | None" = None


def validate_all(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Validate each table against its schema.

    Parameters
    ----------
    tables:
        Output of ``engine.io.load_raw``.

    Returns
    -------
    (clean, quarantine)
        *clean* contains only rows that passed all checks.
        *quarantine* contains rows that failed at least one check.
    """
    if not _PA_AVAILABLE:
        raise RuntimeError("pandera is required for validation. Run: pip install pandera")

    # Schemas are filled in during Milestone 2.
    # Until then, return tables unchanged with an empty quarantine.
    clean = {name: df.copy() for name, df in tables.items()}
    quarantine: Dict[str, pd.DataFrame] = {name: pd.DataFrame() for name in tables}
    return clean, quarantine
