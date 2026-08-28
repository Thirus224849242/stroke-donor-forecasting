"""
Pipeline run history — Supabase (Postgres) persistence.

Deliberately NOT caching the source data (it changes too often for that to
be worth anything). What's saved here is a timestamped snapshot of what
each pipeline run PRODUCED: summary KPIs/MAPEs, each model's monthly
forecast, and the actuals series as it looked at that moment (so an old
run's chart redraws exactly as it did then, even if later source data
revises history). Three small, cheap tables -- see build_production_forecast
and friends for where the numbers actually come from; this module only
persists and reads them back.

Fails soft everywhere: no secrets configured, or a connection error, just
means history isn't saved/available this run -- never breaks the pipeline
itself. Same pattern as auth.py's google_auth_configured().
"""

import traceback
from datetime import datetime

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    run_at         TIMESTAMP NOT NULL,
    run_by         TEXT,
    master_rows    INTEGER,
    donor_count    INTEGER,
    total_income   DOUBLE PRECISION,
    mape_linear    DOUBLE PRECISION,
    mape_ml        DOUBLE PRECISION,
    mape_ltv       DOUBLE PRECISION,
    mape_stockflow DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS forecasts (
    run_id           TEXT NOT NULL REFERENCES runs(run_id),
    model            TEXT NOT NULL,
    calendar_month   INTEGER NOT NULL,
    predicted_income DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS actuals_snapshot (
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    donor_month   DATE NOT NULL,
    total_income  DOUBLE PRECISION,
    active_donors INTEGER
);
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
            for statement in SCHEMA.strip().split(';'):
                if statement.strip():
                    conn.execute(text(statement))
            conn.commit()
        return engine
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return None


def save_run(run_by, master_rows, donor_count, total_income, mapes, forecasts, actuals_df):
    """
    Persist one pipeline run. Returns the run_id on success, None if history
    saving isn't available (not configured, or a connection error) -- the
    caller should treat that as "history wasn't saved this time", not as a
    reason to fail the pipeline run itself.

    mapes: dict with any of 'linear', 'ml', 'ltv', 'stockflow' -> float
    forecasts: dict of model_name -> DataFrame with calendar_month,
        predicted_income columns (e.g. {'ml': forecast_df_ml, ...})
    actuals_df: the monthly actuals DataFrame (donor_month, total_income,
        active_donors columns)
    """
    engine = get_engine()
    if engine is None:
        return None

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    def _as_float(x):
        # A caller passing a pandas/numpy scalar (e.g. straight off a
        # Series.mean()) instead of a plain Python float breaks psycopg2's
        # parameter binding outright -- guard against that here, once, so
        # every caller doesn't need to remember to cast.
        return None if x is None else float(x)

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO runs (run_id, run_at, run_by, master_rows, donor_count,
                                   total_income, mape_linear, mape_ml, mape_ltv, mape_stockflow)
                VALUES (:run_id, :run_at, :run_by, :master_rows, :donor_count,
                        :total_income, :mape_linear, :mape_ml, :mape_ltv, :mape_stockflow)
            """), {
                'run_id': run_id, 'run_at': datetime.now(), 'run_by': run_by,
                'master_rows': int(master_rows), 'donor_count': int(donor_count),
                'total_income': _as_float(total_income),
                'mape_linear': _as_float(mapes.get('linear')), 'mape_ml': _as_float(mapes.get('ml')),
                'mape_ltv': _as_float(mapes.get('ltv')), 'mape_stockflow': _as_float(mapes.get('stockflow')),
            })

            for model_name, df in forecasts.items():
                if df is None or df.empty:
                    continue
                rows = [
                    {'run_id': run_id, 'model': model_name,
                     'calendar_month': int(r['calendar_month']),
                     'predicted_income': float(r['predicted_income'])}
                    for _, r in df.iterrows()
                ]
                conn.execute(text("""
                    INSERT INTO forecasts (run_id, model, calendar_month, predicted_income)
                    VALUES (:run_id, :model, :calendar_month, :predicted_income)
                """), rows)

            if actuals_df is not None and not actuals_df.empty:
                a = actuals_df.copy()
                # donor_month is a pandas Period column throughout this app's
                # pipeline (built via .dt.to_period('M')) -- pd.to_datetime()
                # can't be called on Period data directly and raises, which
                # was silently rolling back the whole transaction (including
                # the already-succeeded runs insert) every single time.
                if isinstance(a['donor_month'].dtype, pd.PeriodDtype):
                    a['donor_month'] = a['donor_month'].dt.to_timestamp()
                a['donor_month'] = pd.to_datetime(a['donor_month']).dt.date
                rows = [
                    {'run_id': run_id, 'donor_month': r['donor_month'],
                     'total_income': float(r['total_income']),
                     'active_donors': int(r['active_donors'])}
                    for _, r in a.iterrows()
                ]
                conn.execute(text("""
                    INSERT INTO actuals_snapshot (run_id, donor_month, total_income, active_donors)
                    VALUES (:run_id, :donor_month, :total_income, :active_donors)
                """), rows)
        return run_id
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return None


def get_runs():
    """All past runs, most recent first. Empty DataFrame if unavailable."""
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(text('SELECT * FROM runs ORDER BY run_at DESC'), conn)
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return pd.DataFrame()


def get_run_forecasts(run_id):
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(
                text('SELECT * FROM forecasts WHERE run_id = :run_id ORDER BY model, calendar_month'),
                conn, params={'run_id': run_id},
            )
    except Exception:
        return pd.DataFrame()


def get_run_actuals(run_id):
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(
                text('SELECT * FROM actuals_snapshot WHERE run_id = :run_id ORDER BY donor_month'),
                conn, params={'run_id': run_id},
            )
    except Exception:
        return pd.DataFrame()
