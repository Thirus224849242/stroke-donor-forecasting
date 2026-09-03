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

-- Google sign-in access control -- being on an allowed domain (see
-- auth.py's ALLOWED_GOOGLE_DOMAINS) is necessary but no longer
-- sufficient on its own for a non-admin: they also need a row here with
-- status='approved'. status is one of 'pending' (just requested, not
-- yet reviewed), 'approved', or 'denied' (can re-request -- see
-- request_access() below). Admins skip the request-access flow entirely
-- (handle_google_redirect() lets GOOGLE_ADMIN_EMAILS straight in) but
-- still get upserted here via upsert_approved_user(), so this table
-- stays a complete record of everyone with access, not just the
-- analysts who went through the request queue.
CREATE TABLE IF NOT EXISTS users (
    email        TEXT PRIMARY KEY,
    name         TEXT,
    role         TEXT NOT NULL DEFAULT 'Analyst',
    status       TEXT NOT NULL DEFAULT 'pending',
    requested_at TIMESTAMP NOT NULL,
    decided_at   TIMESTAMP,
    decided_by   TEXT
);
-- When this user last opened Run History -- backs the sidebar's "new run"
-- notification badge (count_new_runs() below). NULL until their first
-- ever visit; count_new_runs() falls back to requested_at in that case
-- (their signup date) rather than to the epoch, so a long-registered user
-- who's simply never opened the page sees everything since they joined,
-- not the app's entire run history.
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_seen_runs_at TIMESTAMP;
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
        count_new_runs.clear()  # every other user's sidebar badge should see this run right away, not after 30s
        return run_id
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return None


@st.cache_data(show_spinner=False, ttl=30)
def count_new_runs(email: str) -> int:
    """Runs saved since this user last opened Run History -- backs the
    sidebar's notification badge (render_sidebar() in ui.py). Cached with
    a short ttl (unlike every other read in this file) specifically
    because, unlike those, this one runs on EVERY page's sidebar on EVERY
    rerun, not just while actually on Run History/Access Requests --
    an uncached query there would mean two extra DB round trips per
    rerun, app-wide. 0 (no badge) if the user has no row at all (DB
    unreachable, or a locally hardcoded dev/demo login that never went
    through the Google request/approve flow and so was never upserted
    into `users`) -- same fail-soft convention as the rest of this file."""
    engine = get_engine()
    if engine is None:
        return 0
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT COUNT(*) FROM dashboard_runs
                WHERE run_at > (
                    SELECT COALESCE(last_seen_runs_at, requested_at) FROM users WHERE email = :email
                )
            """), {'email': email}).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        return 0


def mark_runs_seen(email: str) -> None:
    """Call once when Run History actually renders (see app.py) -- resets
    count_new_runs()'s baseline to now. A no-op (0 rows affected, no
    error) for a user with no `users` row, same case count_new_runs()
    already handles."""
    engine = get_engine()
    if engine is None:
        return
    try:
        with engine.begin() as conn:
            conn.execute(
                text('UPDATE users SET last_seen_runs_at = :now WHERE email = :email'),
                {'now': datetime.now(), 'email': email},
            )
        count_new_runs.clear()  # so the sidebar badge clears immediately, not after its 30s ttl
    except Exception:
        print('db.py error:', traceback.format_exc())  # shows up in server logs


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
        count_new_runs.clear()  # a deleted run shouldn't linger in anyone's "new" badge count
        return True
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return False


# ── Access control (Google sign-in request/approve queue) ──────────────────

def get_user(email: str) -> dict | None:
    """One user's access record, or None if they've never requested (or
    been granted) access at all. `status` is 'pending', 'approved', or
    'denied' -- see handle_google_redirect() in auth.py for how each is
    handled."""
    engine = get_engine()
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT email, name, role, status, requested_at, decided_at, decided_by
                FROM users WHERE email = :email
            """), {'email': email}).mappings().fetchone()
        return dict(row) if row else None
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return None


def request_access(email: str, name: str) -> bool:
    """Creates a pending access request for a brand-new email, or resets
    an existing DENIED one back to pending (a re-request) -- the WHERE
    clause on the conflict update means this is a no-op for anyone
    already 'pending' or 'approved', so it's safe to call unconditionally
    from the request-access screen's button without checking get_user()
    again first."""
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO users (email, name, role, status, requested_at)
                VALUES (:email, :name, 'Analyst', 'pending', :requested_at)
                ON CONFLICT (email) DO UPDATE SET
                    status = 'pending', requested_at = EXCLUDED.requested_at,
                    decided_at = NULL, decided_by = NULL
                WHERE users.status = 'denied'
            """), {'email': email, 'name': name, 'requested_at': datetime.now()})
        count_pending_requests.clear()  # so the sidebar badge picks this up immediately, not after its 30s ttl
        return True
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return False


@st.cache_data(show_spinner=False, ttl=30)
def count_pending_requests() -> int:
    """Count of requests awaiting a decision -- backs the sidebar's
    notification badge (render_sidebar() in ui.py). No "seen" tracking
    the way count_new_runs() has: unlike a run, a pending request is
    inherently an open item needing action, so the badge is just the
    live pending count, and naturally drops as decide_access_request()
    resolves each one -- no separate mark-seen step. Cached (short ttl)
    for the same reason count_new_runs() is: this runs on every page's
    sidebar, not just Access Requests itself."""
    engine = get_engine()
    if engine is None:
        return 0
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT COUNT(*) FROM users WHERE status = 'pending'")).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        return 0


def list_pending_requests() -> pd.DataFrame:
    """All pending access requests, oldest first -- for the Administrator-
    only Access Requests page. Empty DataFrame if unavailable."""
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(text("""
                SELECT email, name, requested_at FROM users
                WHERE status = 'pending' ORDER BY requested_at
            """), conn)
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return pd.DataFrame()


def decide_access_request(email: str, approve: bool, decided_by: str) -> bool:
    """Approves or denies one pending request. New accounts are always
    approved as Analyst -- promoting someone to Administrator is a code
    change (GOOGLE_ADMIN_EMAILS in auth.py), not something this queue
    grants, so `role` is untouched here."""
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE users SET status = :status, decided_at = :decided_at, decided_by = :decided_by
                WHERE email = :email
            """), {
                'status': 'approved' if approve else 'denied',
                'decided_at': datetime.now(), 'decided_by': decided_by, 'email': email,
            })
        count_pending_requests.clear()  # so the sidebar badge drops immediately, not after its 30s ttl
        return True
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return False


def upsert_approved_user(email: str, name: str, role: str, decided_by: str) -> bool:
    """Used for admin logins, which skip the request-access queue
    entirely -- keeps `users` a complete record of everyone with access,
    not just the analysts who went through it. Safe to call on every
    admin login (idempotent update, not a fresh insert each time)."""
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO users (email, name, role, status, requested_at, decided_at, decided_by)
                VALUES (:email, :name, :role, 'approved', :now, :now, :decided_by)
                ON CONFLICT (email) DO UPDATE SET
                    name = EXCLUDED.name, role = EXCLUDED.role, status = 'approved'
            """), {'email': email, 'name': name, 'role': role, 'now': datetime.now(), 'decided_by': decided_by})
        return True
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return False
