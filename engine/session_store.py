"""
engine/session_store.py — Session storage with disk persistence.

Sessions are kept in memory for speed and also written to /tmp as parquet
files so they survive a dyno restart (within the same dyno's /tmp lifetime).
TTL is 4 hours.
"""

from __future__ import annotations

import pathlib
import time
import uuid

import pandas as pd

_store: dict = {}
_TTL = 14400  # 4 hours
_DISK_DIR = pathlib.Path("/tmp/eda_sessions")
_DISK_DIR.mkdir(exist_ok=True)


def create_session(df: pd.DataFrame) -> str:
    """Store a DataFrame and return a new session_id."""
    session_id = str(uuid.uuid4())
    _store[session_id] = {"df": df, "created_at": time.time()}
    # Persist to disk as parquet
    try:
        df.to_parquet(_DISK_DIR / f"{session_id}.parquet", index=True)
    except Exception:
        pass  # parquet write failure is non-fatal
    _purge_expired()
    return session_id


def get_df(session_id: str) -> pd.DataFrame | None:
    """Return the stored DataFrame or None if expired/missing."""
    entry = _store.get(session_id)
    if entry is not None:
        if time.time() - entry["created_at"] > _TTL:
            _evict(session_id)
            return None
        return entry["df"]

    # Not in memory — try disk (dyno restarted)
    path = _DISK_DIR / f"{session_id}.parquet"
    if path.exists():
        try:
            df = pd.read_parquet(path)
            _store[session_id] = {"df": df, "created_at": time.time()}
            return df
        except Exception:
            path.unlink(missing_ok=True)

    return None


def _evict(session_id: str) -> None:
    _store.pop(session_id, None)
    (_DISK_DIR / f"{session_id}.parquet").unlink(missing_ok=True)


def _purge_expired() -> None:
    now = time.time()
    expired = [sid for sid, e in _store.items() if now - e["created_at"] > _TTL]
    for sid in expired:
        _evict(sid)
    # Also clean up orphaned disk files older than TTL
    for p in _DISK_DIR.glob("*.parquet"):
        if now - p.stat().st_mtime > _TTL:
            p.unlink(missing_ok=True)
