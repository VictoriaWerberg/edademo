"""
api/app.py — Flask API wrapping the analysis engine.

Endpoints
---------
GET  /           — health check
POST /analyze    — accepts a CSV upload, runs the pipeline, returns JSON

Deployed to Heroku (see Procfile).
"""

from __future__ import annotations

import io
import sys
import pathlib
import tempfile
import zipfile

from flask import Flask, jsonify, request
from flask_cors import CORS

# Make engine/ importable regardless of working directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from engine.io import load_raw
from engine.validate import validate_all
from engine.clean import build_analysis_frame, frame_summary
from engine.stats import run_analysis, results_to_json

app = Flask(__name__)
CORS(app)

MAX_BYTES = 20 * 1024 * 1024  # 20 MB upload cap


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Interconnect Analysis API"})


@app.route("/analyze", methods=["POST"])
def analyze():
    """Accept a CSV or ZIP bundle, run the pipeline, return JSON findings."""

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
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)

            if uploaded.filename.lower().endswith(".zip"):
                # Expect a ZIP containing the four CSVs
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    zf.extractall(tmp)
                data_dir = tmp
            else:
                # Single CSV — treat as interconnect_measurements; use bundled
                # synthetic data for the other three tables (demo mode)
                (tmp / "interconnect_measurements.csv").write_bytes(raw)
                _copy_demo_files(tmp)
                data_dir = tmp

            # --- pipeline ---
            tables = load_raw(data_dir)
            clean, quarantine = validate_all(tables)

            quarantine_counts = {k: len(v) for k, v in quarantine.items()}
            total_quarantined = sum(quarantine_counts.values())

            df = build_analysis_frame(clean)
            results = run_analysis(df)

            payload = results_to_json(results)
            payload["summary"]            = frame_summary(df)
            payload["quarantine_counts"]  = quarantine_counts
            payload["total_rows_analyzed"] = int(len(df))
            payload["total_quarantined"]   = total_quarantined

            return jsonify(payload)

    except FileNotFoundError as exc:
        return jsonify({"error": f"Missing expected file: {exc}"}), 422
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


def _copy_demo_files(dest: pathlib.Path) -> None:
    """Copy bundled synthetic CSVs into dest for single-file demo mode."""
    data_dir = pathlib.Path(__file__).parent.parent / "data"
    for name in ("process_log", "design_manifest", "environmental_log"):
        src = data_dir / f"{name}.csv"
        if src.exists():
            (dest / f"{name}.csv").write_bytes(src.read_bytes())
        else:
            raise FileNotFoundError(
                f"Demo data file not found: {src}. "
                "Run data/generate_synthetic.py first."
            )


if __name__ == "__main__":
    app.run(debug=True)
