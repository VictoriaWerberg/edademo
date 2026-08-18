"""
engine/executor.py — Safe execution of user-supplied pandas code.

Runs code in a restricted namespace containing only:
  df       — the session DataFrame (read-only copy)
  pd       — pandas
  np       — numpy
  result   — the variable the user must assign their output to

Returns a JSON-serialisable dict:
  {"type": "dataframe", "columns": [...], "rows": [[...], ...], "shape": [r, c]}
  {"type": "scalar",    "value": ...}
  {"type": "series",    "index": [...], "values": [...]}
  {"type": "error",     "message": "..."}
"""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np
import pandas as pd


# Builtins that are safe in a data exploration context
_SAFE_BUILTINS = {
    "abs": abs, "len": len, "max": max, "min": min,
    "round": round, "sum": sum, "sorted": sorted,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "str": str, "int": int, "float": float, "bool": bool,
    "print": print, "range": range, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter, "any": any, "all": all,
    "isinstance": isinstance, "type": type,
}


def run_code(code: str, df: pd.DataFrame) -> dict:
    """Execute *code* with *df* in scope; return a JSON-safe result dict."""
    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "df": df.copy(),   # copy so user can't mutate the stored df
        "result": None,
    }

    try:
        exec(compile(code, "<user-code>", "exec"), namespace)  # noqa: S102
    except Exception:
        return {"type": "error", "message": traceback.format_exc(limit=5)}

    raw = namespace.get("result")
    return _to_json(raw)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _to_json(obj: Any) -> dict:
    if obj is None:
        return {"type": "scalar", "value": None}

    if isinstance(obj, pd.DataFrame):
        # Limit to 200 rows for display
        sample = obj.head(200)
        for col in sample.select_dtypes(include="datetime").columns:
            sample = sample.copy()
            sample[col] = sample[col].astype(str)
        return {
            "type":    "dataframe",
            "columns": sample.columns.tolist(),
            "rows":    sample.fillna("").astype(str).values.tolist(),
            "shape":   list(obj.shape),
        }

    if isinstance(obj, pd.Series):
        s = obj.head(200)
        return {
            "type":   "series",
            "name":   str(s.name) if s.name is not None else "",
            "index":  [str(i) for i in s.index.tolist()],
            "values": [_safe_scalar(v) for v in s.tolist()],
        }

    if isinstance(obj, (int, float, np.integer, np.floating)):
        return {"type": "scalar", "value": _safe_scalar(obj)}

    if isinstance(obj, str):
        return {"type": "scalar", "value": obj}

    # Fallback: coerce to string
    return {"type": "scalar", "value": str(obj)}


def _safe_scalar(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, float) and (v != v):   # NaN
        return None
    return v
