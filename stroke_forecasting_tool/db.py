"""
Dashboard run history -- Supabase (Postgres).

Deliberately does NOT store the source data or the full donor-month master
panel (that's what made an earlier version of this file slow and heavy).
Instead, after each pipeline run finishes, only the DISPLAY-LEVEL numbers
every page actually renders -- aggregated summaries, chart series, table
rows, scalar metrics, and the full per-donor LTV table (small enough
compressed to include in full, restoring the "download all donor
predictions" feature for past runs too) -- are extracted into one JSON
object, gzip'd, and inserted as a new row here. A handful of small summary
columns (donor_count, total_income, the MAPEs) sit alongside the blob
un-gzipped, so the Run History page's list/trend-over-time view is cheap --
it never has to decompress every run's full blob just to build a list.

This is a real history: every completed run gets its own row, and any past
run can be reloaded in full via load_dashboard_run(). The most recent run
is what auto-loads on a fresh app session (see app.py's startup check).

See app.py's build_dashboard_state() / restore_dashboard_state() for what
goes into a run's JSON blob and how it's mapped back to session_state.

Fails soft everywhere: no secrets configured, or a connection error, just
means state isn't saved/available -- the app falls back to the normal
upload flow. Same pattern as auth.py's google_auth_configured().
"""

import gzip
import json
import traceback
from datetime import datetime

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

SCHEMA = """
CREATE TABLE IF NOT EXISTS dashboard_runs (
    run_id         TEXT PRIMARY KEY,
    run_at         TIMESTAMP NOT NULL,
    run_by         TEXT,
    donor_count    INTEGER,
    total_income   DOUBLE PRECISION,
    mape_ml        DOUBLE PRECISION,
    mape_linear    DOUBLE PRECISION,
    mape_ltv       DOUBLE PRECISION,
    mape_stockflow DOUBLE PRECISION,
    mape_sbg       DOUBLE PRECISION,
    mape_bgnbd     DOUBLE PRECISION,
    mape_gw        DOUBLE PRECISION,
    duration_seconds DOUBLE PRECISION,
    data           BYTEA NOT NULL
);
ALTER TABLE dashboard_runs ADD COLUMN IF NOT EXISTS mape_ltv DOUBLE PRECISION;
ALTER TABLE dashboard_runs ADD COLUMN IF NOT EXISTS mape_gw DOUBLE PRECISION;
ALTER TABLE dashboard_runs ADD COLUMN IF NOT EXISTS duration_seconds DOUBLE PRECISION;
"""


def db_configured() -> bool:
    """True once secrets.toml's [database] section has a real connection
    string (not the placeholder scaffold)."""
    try:
        db = st.secrets.get('database')
        if not db:
            return False
        url = db.get('url', '')
        return bool(url) and not url.startswith('REPLACE_WITH_')
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def get_engine():
    """One pooled connection, reused across reruns -- st.cache_resource is
    the right tool here (not cache_data), since an engine is a live
    connector, not serializable data."""
    if not db_configured():
        return None
    try:
        engine = create_engine(st.secrets['database']['url'], pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text(SCHEMA))
            conn.commit()
        return engine
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return None


def save_dashboard_run(state: dict, summary: dict) -> str | None:
    """Inserts a new run. Returns the run_id on success, None if saving
    isn't available (not configured, or a connection error) -- the caller
    should treat that as "this run's dashboard state wasn't saved", not as
    a reason to fail the pipeline run itself.

    summary: dict with any of donor_count, total_income, mape_ml,
        mape_linear, mape_ltv, mape_stockflow, mape_sbg, mape_bgnbd, mape_gw,
        duration_seconds -- the small columns the Run History list/trend
        view reads without needing to decompress every run's full blob.
    """
    engine = get_engine()
    if engine is None:
        return None
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        payload = gzip.compress(json.dumps(state).encode(), compresslevel=9)
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO dashboard_runs
                    (run_id, run_at, run_by, donor_count, total_income,
                     mape_ml, mape_linear, mape_ltv, mape_stockflow, mape_sbg, mape_bgnbd, mape_gw,
                     duration_seconds, data)
                VALUES
                    (:run_id, :run_at, :run_by, :donor_count, :total_income,
                     :mape_ml, :mape_linear, :mape_ltv, :mape_stockflow, :mape_sbg, :mape_bgnbd, :mape_gw,
                     :duration_seconds, :data)
            """), {
                'run_id': run_id, 'run_at': datetime.now(), 'run_by': summary.get('run_by'),
                'donor_count': summary.get('donor_count'), 'total_income': summary.get('total_income'),
                'mape_ml': summary.get('mape_ml'), 'mape_linear': summary.get('mape_linear'),
                'mape_ltv': summary.get('mape_ltv'), 'mape_stockflow': summary.get('mape_stockflow'),
                'mape_sbg': summary.get('mape_sbg'), 'mape_bgnbd': summary.get('mape_bgnbd'),
                'mape_gw': summary.get('mape_gw'), 'duration_seconds': summary.get('duration_seconds'),
                'data': payload,
            })
        return run_id
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return None


def list_dashboard_runs() -> pd.DataFrame:
    """All past runs, most recent first, WITHOUT their full blobs -- cheap,
    for the Run History list/trend-over-time view. Empty DataFrame if
    unavailable."""
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(text("""
                SELECT run_id, run_at, run_by, donor_count, total_income,
                       mape_ml, mape_linear, mape_ltv, mape_stockflow, mape_sbg, mape_bgnbd, mape_gw,
                       duration_seconds
                FROM dashboard_runs ORDER BY run_at DESC
            """), conn)
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return pd.DataFrame()


def load_dashboard_run(run_id: str):
    """Returns (state_dict, run_at) for one specific run, or (None, None)
    if it doesn't exist or the DB isn't reachable."""
    engine = get_engine()
    if engine is None:
        return None, None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text('SELECT data, run_at FROM dashboard_runs WHERE run_id = :run_id'),
                {'run_id': run_id},
            ).fetchone()
        if row is None:
            return None, None
        raw = bytes(row[0])
        try:
            raw = gzip.decompress(raw)
        except gzip.BadGzipFile:
            pass
        return json.loads(raw), row[1]
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return None, None


def delete_dashboard_run(run_id: str) -> bool:
    """Deletes one run. Returns True on success."""
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            conn.execute(text('DELETE FROM dashboard_runs WHERE run_id = :run_id'), {'run_id': run_id})
        return True
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return False
