"""
api/app.py — Flask API wrapping the analysis engine.

Endpoints
---------
GET  /           — health check
POST /analyze    — accepts any CSV (or ZIP of four CSVs), returns JSON
POST /upload     — upload any CSV; returns session_id + preview + column info
POST /execute    — run pandas code against a stored session DataFrame
POST /ai-assist  — ask Claude to write/fix pandas code for a given schema

Two modes for /analyze (legacy)
---------
  interconnect  — uploaded CSV has the expected measurement columns;
                  runs the full validate → clean → ANOVA pipeline.
  generic       — any other CSV; runs describe / correlations / missing-value EDA.
"""

from __future__ import annotations

import io
import os
import sys
import pathlib
import tempfile
import zipfile

import anthropic
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from engine.io import load_raw
from engine.validate import validate_all
from engine.clean import build_analysis_frame, frame_summary
from engine.stats import run_analysis, results_to_json
from engine.generic import run_generic_eda
from engine.session_store import create_session, get_df
from engine.executor import run_code

app = Flask(__name__)
CORS(app)

MAX_BYTES = 20 * 1024 * 1024  # 20 MB

# Columns that identify a measurements CSV
_INTERCONNECT_COLS = {"meas_id", "via_type", "layer", "resistance_ohm", "continuity_pass"}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Interconnect Analysis API"})


# ---------------------------------------------------------------------------
# Legacy full-analysis endpoint
# ---------------------------------------------------------------------------

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
# Upload — stores session, returns preview
# ---------------------------------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload():
    """Upload any CSV and store it for interactive exploration.

    Returns:
        session_id  — UUID to reference this upload in /execute and /ai-assist
        columns     — list of {name, dtype, n_unique, missing, missing_pct}
        preview     — first 20 rows as {columns: [...], rows: [[...], ...]}
        shape       — {rows, cols}
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "Empty filename."}), 400

    raw = uploaded.read()
    if len(raw) > MAX_BYTES:
        return jsonify({"error": f"File too large (max {MAX_BYTES // 1_048_576} MB)."}), 413

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        return jsonify({"error": f"Could not parse CSV: {exc}"}), 400

    session_id = create_session(df)

    # Column metadata
    columns_meta = []
    for col in df.columns:
        s = df[col]
        n_miss = int(s.isnull().sum())
        columns_meta.append({
            "name":        col,
            "dtype":       str(s.dtype),
            "n_unique":    int(s.nunique()),
            "missing":     n_miss,
            "missing_pct": round(n_miss / len(df) * 100, 1),
        })

    # Preview (first 20 rows)
    preview_df = df.head(20).copy()
    for col in preview_df.select_dtypes(include="datetime").columns:
        preview_df[col] = preview_df[col].astype(str)

    preview = {
        "columns": preview_df.columns.tolist(),
        "rows":    preview_df.fillna("").astype(str).values.tolist(),
    }

    return jsonify({
        "session_id": session_id,
        "shape":      {"rows": len(df), "cols": len(df.columns)},
        "columns":    columns_meta,
        "preview":    preview,
    })


# ---------------------------------------------------------------------------
# Execute — run pandas code against stored session df
# ---------------------------------------------------------------------------

@app.route("/execute", methods=["POST"])
def execute():
    """Run user-supplied pandas code against a stored session DataFrame.

    Body JSON:
        session_id  — from /upload
        code        — Python code; must assign final result to `result`

    Returns the result as a JSON-safe dict.
    """
    body = request.get_json(force=True, silent=True) or {}
    session_id = body.get("session_id", "")
    code       = body.get("code", "")

    if not session_id:
        return jsonify({"type": "error", "message": "Missing session_id"}), 400
    if not code.strip():
        return jsonify({"type": "error", "message": "No code provided"}), 400

    df = get_df(session_id)
    if df is None:
        return jsonify({"type": "error", "message": "Session not found or expired. Please re-upload your file."}), 404

    result = run_code(code, df)
    return jsonify(result)


# ---------------------------------------------------------------------------
# AI assist — ask Claude to write pandas code
# ---------------------------------------------------------------------------

@app.route("/ai-assist", methods=["POST"])
def ai_assist():
    """Ask Claude to write or fix pandas code for the user's question.

    Body JSON:
        session_id  — from /upload (used to get column schema)
        question    — what the user wants to do
        current_code (optional) — code to fix/improve

    Returns:
        {"code": "...", "explanation": "..."}
    """
    body = request.get_json(force=True, silent=True) or {}
    session_id   = body.get("session_id", "")
    question     = body.get("question", "")
    current_code = body.get("current_code", "")

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    if not question.strip():
        return jsonify({"error": "No question provided"}), 400

    df = get_df(session_id)
    if df is None:
        return jsonify({"error": "Session not found or expired. Please re-upload your file."}), 404

    # Build schema description
    schema_lines = []
    for col in df.columns:
        s = df[col]
        n_miss = int(s.isnull().sum())
        schema_lines.append(f"  - {col} ({s.dtype}), {int(s.nunique())} unique, {n_miss} missing")
    schema_str = "\n".join(schema_lines)

    sample_vals = {}
    for col in df.columns[:8]:
        vals = df[col].dropna().head(3).tolist()
        sample_vals[col] = [str(v) for v in vals]
    sample_str = "\n".join(f"  {k}: {v}" for k, v in sample_vals.items())

    system_prompt = f"""You are a concise pandas data assistant. The user has a DataFrame called `df` with the following columns:

{schema_str}

Sample values (up to 3 per column, first 8 columns shown):
{sample_str}

Rules:
- Write Python / pandas code that operates on `df`.
- Always assign the final result to a variable named `result`.
- Keep code short and readable — 1 to 10 lines.
- Do NOT import anything; pd and np are already available.
- Return ONLY a JSON object with two keys:
    "code": the Python code string
    "explanation": one or two plain-English sentences describing what the code does
- No markdown fences, no extra text — just the raw JSON object."""

    user_content = question
    if current_code.strip():
        user_content += f"\n\nMy current code (please fix or improve):\n{current_code}"

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured on the server."}), 503

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        raw_text = message.content[0].text.strip()

        # Parse JSON from Claude's response
        import json
        # Strip markdown fences if Claude added them despite the instructions
        if raw_text.startswith("```"):
            raw_text = "\n".join(raw_text.split("\n")[1:])
            raw_text = raw_text.rsplit("```", 1)[0].strip()

        parsed = json.loads(raw_text)
        return jsonify(parsed)

    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"AI assist failed: {exc}"}), 500


# ---------------------------------------------------------------------------
# Interconnect pipeline helpers
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
