"""
api/app.py — Flask API wrapping the analysis engine.

Endpoints
---------
GET  /           — health check
POST /analyze    — accepts any CSV (or ZIP of four CSVs), returns JSON

Two modes
---------
  interconnect  — uploaded CSV has the expected measurement columns;
                  runs the full validate → clean → ANOVA pipeline.
  generic       — any other CSV; runs describe / correlations / missing-value EDA.

ZIP uploads always run in interconnect mode (all four tables present).
"""

from __future__ import annotations

import io
import sys
import pathlib
import tempfile
import zipfile

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from engine.io import load_raw
from engine.validate import validate_all
from engine.clean import build_analysis_frame, frame_summary
from engine.stats import run_analysis, results_to_json
from engine.generic import run_generic_eda

app = Flask(__name__)
CORS(app)

MAX_BYTES = 20 * 1024 * 1024  # 20 MB

# Columns that identify a measurements CSV
_INTERCONNECT_COLS = {"meas_id", "via_type", "layer", "resistance_ohm", "continuity_pass"}


@app.route("/", methods=["GET"])
def health():
    """Health check."""
    return jsonify({"status": "ok", "message": "Interconnect Analysis API"})


@app.route("/analyze", methods=["POST"])
def analyze():
    """Accept a CSV or ZIP, auto-detect mode, return JSON analysis."""

    if "file" not in request.files:
        return jsonify(
            {"error": "No file uploaded. POST multipart/form-data with key 'file'."}
        ), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "Empty filename."}), 400

    raw = uploaded.read()
    if len(raw) > MAX_BYTES:
        return jsonify({"error": f"File too large (max {MAX_BYTES // 1_048_576} MB)."}), 413

    try:
        filename = uploaded.filename.lower()

        if filename.endswith(".zip"):
            return _run_interconnect_zip(raw)
        else:
            df_uploaded = pd.read_csv(io.BytesIO(raw))
            if _INTERCONNECT_COLS.issubset(set(df_uploaded.columns)):
                return _run_interconnect_csv(raw)
            else:
                return jsonify(run_generic_eda(df_uploaded))

    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Interconnect pipeline
# ---------------------------------------------------------------------------

def _run_interconnect_csv(raw: bytes):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        (tmp / "interconnect_measurements.csv").write_bytes(raw)
        _copy_demo_files(tmp)
        return _run_pipeline(tmp)


def _run_interconnect_zip(raw: bytes):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            zf.extractall(tmp)
        return _run_pipeline(tmp)


def _run_pipeline(data_dir: pathlib.Path):
    tables = load_raw(data_dir)
    clean, quarantine = validate_all(tables)

    quarantine_counts = {k: len(v) for k, v in quarantine.items()}

    df = build_analysis_frame(clean)
    results = run_analysis(df)

    payload = results_to_json(results)
    payload["mode"]               = "interconnect"
    payload["summary"]            = frame_summary(df)
    payload["quarantine_counts"]  = quarantine_counts
    payload["total_rows_analyzed"] = int(len(df))
    payload["total_quarantined"]   = sum(quarantine_counts.values())

    return jsonify(payload)


def _copy_demo_files(dest: pathlib.Path) -> None:
    data_dir = pathlib.Path(__file__).parent.parent / "data"
    for name in ("process_log", "design_manifest", "environmental_log"):
        src = data_dir / f"{name}.csv"
        if not src.exists():
            raise FileNotFoundError(f"Demo data file missing: {src}")
        (dest / f"{name}.csv").write_bytes(src.read_bytes())


if __name__ == "__main__":
    app.run(debug=True)
