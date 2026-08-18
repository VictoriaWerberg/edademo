"""
api/app.py — Flask API wrapping the analysis engine.

Exposes one endpoint:
    POST /analyze   — accepts a CSV upload, runs the pipeline, returns JSON

Deployed to Heroku (see Procfile).
"""

from __future__ import annotations

import io
import sys
import pathlib

from flask import Flask, jsonify, request
from flask_cors import CORS

# Make engine/ importable when running from the api/ directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

app = Flask(__name__)
CORS(app)  # allow the Netlify site to call this API


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Interconnect Analysis API"})


@app.route("/analyze", methods=["POST"])
def analyze():
    """Accept a zipped CSV bundle or individual CSV, run the pipeline."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a CSV as multipart/form-data with key 'file'."}), 400

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # --- pipeline (implemented milestone by milestone) ---
    # from engine.io import load_raw
    # from engine.validate import validate_all
    # from engine.clean import build_analysis_frame
    # from engine.stats import run_analysis
    # tables = load_raw(...)
    # clean, quarantine = validate_all(tables)
    # df = build_analysis_frame(clean)
    # results = run_analysis(df)
    # return jsonify(results)

    # Milestone 1 placeholder:
    return jsonify({
        "status": "received",
        "filename": uploaded.filename,
        "message": "Pipeline not yet wired — analysis coming in Milestones 2-4.",
    })


if __name__ == "__main__":
    app.run(debug=True)
