"""
engine/executor.py — Safe execution of user-supplied pandas code.

Namespace available to user code:
  df       — session DataFrame (copy)
  pd       — pandas
  np       — numpy
  io       — io module (for df.info(buf=...) etc.)
  plt      — matplotlib.pyplot (if available)
  result   — assign final output here

Returns a JSON-serialisable dict:
  {"type": "dataframe", ...}
  {"type": "series",    ...}
  {"type": "scalar",    ...}
  {"type": "image",     "data": "<base64 PNG>"}   # when plt figure produced
  {"type": "error",     "message": "..."}
"""

from __future__ import annotations

import base64
import io
import traceback
from typing import Any

import numpy as np
import pandas as pd

# ── Optional matplotlib ───────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Dark theme to match the UI
    mpl.rcParams.update({
        'figure.facecolor':  '#1a1d27',
        'axes.facecolor':    '#22263a',
        'text.color':        '#e2e8f0',
        'axes.labelcolor':   '#e2e8f0',
        'xtick.color':       '#8892b0',
        'ytick.color':       '#8892b0',
        'axes.edgecolor':    '#2e3450',
        'grid.color':        '#2e3450',
        'legend.facecolor':  '#22263a',
        'legend.edgecolor':  '#2e3450',
        'figure.dpi':        110,
    })
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Optional scipy ────────────────────────────────────────────────
try:
    import scipy.stats as _scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    _scipy_stats = None


# ── Safe builtins ─────────────────────────────────────────────────
_SAFE_BUILTINS = {
    "abs": abs, "len": len, "max": max, "min": min,
    "round": round, "sum": sum, "sorted": sorted,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "str": str, "int": int, "float": float, "bool": bool,
    "print": print, "range": range, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter,
    "any": any, "all": all,
    "isinstance": isinstance, "type": type,
    "hasattr": hasattr, "getattr": getattr,
}


# ── Public entry point ────────────────────────────────────────────
def run_code(code: str, df: pd.DataFrame) -> dict:
    """Execute *code* with *df* in scope; return a JSON-safe result dict."""

    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "pd":     pd,
        "np":     np,
        "io":     io,
        "df":     df.copy(),
        "result": None,
        "stats":  _scipy_stats,   # scipy.stats (may be None if not installed)
    }

    if HAS_MPL:
        plt.close('all')           # clear any leftover figures
        namespace["plt"] = plt
        namespace["mpl"] = mpl

    try:
        exec(compile(code, "<user-code>", "exec"), namespace)   # noqa: S102
    except Exception:
        return {"type": "error", "message": traceback.format_exc(limit=6)}

    # If a matplotlib figure was produced, return it as a PNG
    if HAS_MPL and plt.get_fignums():
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight',
                    facecolor='#1a1d27', dpi=110)
        plt.close('all')
        buf.seek(0)
        return {"type": "image", "data": base64.b64encode(buf.read()).decode()}

    return _to_json(namespace.get("result"))


# ── Serialisation helpers ─────────────────────────────────────────
def _to_json(obj: Any) -> dict:
    if obj is None:
        return {"type": "scalar", "value": None}

    if isinstance(obj, pd.DataFrame):
        sample = obj.head(200).copy()
        for col in sample.select_dtypes(include="datetime").columns:
            sample[col] = sample[col].astype(str)
        # Always include the index, exactly like Jupyter notebook does.
        sample = sample.reset_index()
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

    return {"type": "scalar", "value": str(obj)}


def _safe_scalar(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, float) and v != v:   # NaN
        return None
    return v
