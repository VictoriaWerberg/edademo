"""
engine/session_store.py — In-memory session storage with TTL.

Sessions expire after 1 hour of inactivity.
"""

from __future__ import annotations

import time
import uuid

import pandas as pd

_store: dict = {}
_TTL = 3600  # seconds


def create_session(df: pd.DataFrame) -> str:
    """Store a DataFrame and return a new session_id."""
    session_id = str(uuid.uuid4())
    _store[session_id] = {
        "df": df,
        "created_at": time.time(),
    }
    _purge_expired()
    return session_id


def get_df(session_id: str) -> pd.DataFrame | None:
    """Return the stored DataFrame or None if expired/missing."""
    entry = _store.get(session_id)
    if entry is None:
        return None
    if time.time() - entry["created_at"] > _TTL:
        del _store[session_id]
        return None
    return entry["df"]


def _purge_expired() -> None:
    now = time.time()
    expired = [sid for sid, e in _store.items() if now - e["created_at"] > _TTL]
    for sid in expired:
        del _store[sid]
