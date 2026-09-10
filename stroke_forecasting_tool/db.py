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
import hashlib
import hmac
import json
import secrets
import traceback
from datetime import datetime

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# ── Local (email+password) account credentials ──────────────────────────────
# Used only by auth.py's local-login form -- Google sign-in never touches
# this. Two accounts (admin@/analyst@strokefoundation.org.au) originally
# lived as plaintext passwords directly in auth.py's source (the Critical
# finding from the first security review). A later pass moved them into
# this table, hashed -- but the ACTUAL password value used to seed that
# hash was still a hardcoded literal in this file's own source (the same
# finding, one file over): storing a hash of a publicly-known, committed
# password is no safer than storing the password itself, since anyone who
# can read the source already knows what to type in. Fixed properly now:
# the real password values live ONLY in secrets.toml (gitignored, same
# pattern as [auth] and [database] below), read at connect time, never
# committed anywhere. No secrets.toml entry for an account means that
# account simply isn't seeded -- opt-in, not on by default.
PBKDF2_ITERATIONS = 260_000  # current OWASP-recommended floor for PBKDF2-SHA256


def hash_password(password: str) -> str:
    """Returns 'salt_hex$hash_hex' -- both parts fit in one TEXT column, and
    the salt travels with its own hash rather than needing a second column."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, PBKDF2_ITERATIONS)
    return f'{salt.hex()}${digest.hex()}'


def verify_password(password: str, stored: str) -> bool:
    """Recomputes the hash using the STORED salt and compares digests with
    hmac.compare_digest (constant-time, avoids leaking timing info about how
    much of the hash matched)."""
    try:
        salt_hex, digest_hex = stored.split('$', 1)
    except (ValueError, AttributeError):
        return False
    candidate = hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS)
    return hmac.compare_digest(candidate.hex(), digest_hex)


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

-- Local (email+password) login accounts, alongside Google sign-in -- see
-- hash_password()/verify_password() above and get_local_account() below.
-- Seeded/rotated from secrets.toml's [local_accounts] section by
-- _seed_local_accounts() right after this schema runs, on every connect
-- (needs Python to compute each password's hash, which plain DDL can't do
-- inline) -- opt-in per account, and re-syncing the hash on every connect
-- is what makes a secrets.toml password change an actual rotation.
CREATE TABLE IF NOT EXISTS local_accounts (
    email         TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'Analyst',
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMP NOT NULL DEFAULT now()
);
-- Two-factor auth (TOTP, e.g. Google Authenticator/Authy) for local
-- accounts -- NULL means 2FA is off for that account (the default; opt-in
-- per user via the account menu, not forced). Set once a user scans the
-- QR code and confirms it with a real code from their app (see
-- set_totp_secret() below) -- never written from an unverified value, so
-- a row with this column populated is a real, working authenticator, not
-- just "the user started setup". Google sign-in accounts don't use this
-- at all -- their 2FA (if any) is Google's own, managed on their account,
-- not this app's.
ALTER TABLE local_accounts ADD COLUMN IF NOT EXISTS totp_secret TEXT;
"""

# Static metadata (email/name/role) for the two accounts -- NOT the
# password, which now comes only from secrets.toml's [local_accounts]
# section at seed time (see _seed_local_accounts()). Splitting it this way
# means this list can stay in source safely: an email address and a role
# name are not secrets. admin@ seeds as Super Admin, not just
# Administrator -- the local-login bootstrap path needs to land someone
# with enough power to actually set everyone else up (Local Accounts +
# Access Requests are Super-Admin-only, see app.py), same reasoning as
# GOOGLE_ADMIN_EMAILS bootstrapping Super Admin in auth.py.
_DEMO_ACCOUNTS = [
    {'email': 'admin@strokefoundation.org.au', 'secret_key': 'admin_password',
     'name': 'Admin User', 'role': 'Super Admin'},
    {'email': 'analyst@strokefoundation.org.au', 'secret_key': 'analyst_password',
     'name': 'Data Analyst', 'role': 'Analyst'},
]


def _seed_local_accounts(conn) -> None:
    """Seeds/rotates local_accounts from secrets.toml's [local_accounts]
    section -- e.g. admin_password = "..." for admin@strokefoundation.org.au.
    An account with no matching key in secrets.toml is skipped entirely
    (opt-in, not seeded by default). ON CONFLICT DO UPDATE (not DO NOTHING)
    deliberately: this makes changing a password in secrets.toml and
    restarting the app an actual, working rotation path -- including for
    an account that was already seeded with an old value (e.g. this fixes,
    on next connect, any database that was already seeded by the earlier,
    hardcoded-password version of this function).

    role is deliberately NOT in that UPDATE SET list -- same reasoning as
    the GOOGLE_ADMIN_EMAILS fix in auth.py's handle_google_redirect():
    this list is only the role's source of truth for the very first
    INSERT (a brand-new email, which the VALUES clause below still
    handles normally). Re-syncing role on every reconnect would silently
    re-promote/re-demote an account back to whatever's hardcoded here
    every time the app restarts, undoing anything a Super Admin set
    through the Local Accounts page -- the exact same standing-override
    bug, just for local accounts instead of Google ones. Once a row
    exists, its role is the database's alone to manage."""
    configured = st.secrets.get('local_accounts', {})
    for acct in _DEMO_ACCOUNTS:
        password = configured.get(acct['secret_key'])
        if not password:
            continue  # no secrets.toml entry for this account -- don't seed it
        conn.execute(text("""
            INSERT INTO local_accounts (email, name, role, password_hash)
            VALUES (:email, :name, :role, :password_hash)
            ON CONFLICT (email) DO UPDATE SET
                password_hash = EXCLUDED.password_hash,
                name = EXCLUDED.name
        """), {
            'email': acct['email'], 'name': acct['name'], 'role': acct['role'],
            'password_hash': hash_password(password),
        })


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
        # engine.begin() (auto-commits on a clean exit, rolls back on an
        # exception), not engine.connect() + a manual conn.commit() --
        # confirmed live, the manual-commit form can silently fail to
        # persist a write against this database's pooled connection (the
        # secrets.toml setup instructions specifically call for Supabase's
        # "Transaction pooler" mode, which doesn't guarantee one logical
        # connection stays pinned to one backend Postgres connection the
        # way session-mode pooling does) -- an UPDATE ran with no error and
        # a normal rowcount, then simply wasn't there on the next read.
        # engine.begin() is what every other write in this file already
        # uses, and it doesn't reproduce the problem.
        with engine.begin() as conn:
            conn.execute(text(SCHEMA))
            _seed_local_accounts(conn)
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
        list_dashboard_runs.clear()  # so Overview/Run History see the new run immediately, not after its own ttl
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


@st.cache_data(show_spinner=False, ttl=30)
def list_dashboard_runs() -> pd.DataFrame:
    """All past runs, most recent first, WITHOUT their full blobs -- cheap,
    for the Run History list/trend-over-time view. Empty DataFrame if
    unavailable. Cached (short ttl, cleared immediately by save/delete
    below) -- this was an uncached query hit on every single Overview
    load (the default landing page) plus twice more per rerun on Run
    History, same "runs on every page, not just its own" reasoning as
    count_new_runs()/count_pending_requests() above."""
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


@st.cache_data(show_spinner=False, max_entries=20)
def load_dashboard_run(run_id: str):
    """Returns (state_dict, run_at) for one specific run, or (None, None)
    if it doesn't exist or the DB isn't reachable.

    Cached, no ttl -- a saved run's blob is write-once (nothing ever
    updates a dashboard_runs row in place, only inserts or deletes one),
    so unlike the ttl'd caches elsewhere in this file there's nothing for
    a short expiry to protect against; delete_dashboard_run() below
    clears this the same way it clears list_dashboard_runs(), for the
    one case that actually invalidates an entry. Was uncached until this
    was flagged as a source of "pages feel slow after a restart": the
    app's every-page auto-restore-latest-run step (app.py) called this
    on every fresh session with no cache at all, meaning a full DB round
    trip plus gzip-decompressing the whole dashboard-state JSON blob
    every single time someone signed in, even seconds apart, on top of
    the connection-latency cost get_engine() already documents.
    max_entries=20, same reasoning as cached_ml_forecast/friends in
    app.py -- this is keyed on run_id, which is unbounded in principle
    (every run anyone has ever saved), so it needs a cap to avoid an
    unbounded memory leak on a host this memory-constrained."""
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
        list_dashboard_runs.clear()  # so it disappears from Overview/Run History immediately
        load_dashboard_run.clear()  # so a stale blob can't be reloaded via a lingering run_id reference
        return True
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return False


# ── Access control (Google sign-in request/approve queue) ──────────────────

def get_local_account(email: str) -> dict | None:
    """One local (email+password) account's row -- email/name/role/
    password_hash/totp_secret -- or None if that email has no local
    account at all. auth.py's render_login() calls this then checks the
    password against password_hash with verify_password(); DB unreachable
    or no matching row both just mean the login attempt fails, same as a
    wrong password would (no separate "account doesn't exist" message, so
    this can't be used to enumerate which emails have accounts).
    totp_secret is None for every account that hasn't turned 2FA on --
    render_login() only prompts for a code when it's actually set.
    created_at is here for the Profile page's "member since" line --
    every other caller of this function just ignores the extra key."""
    engine = get_engine()
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT email, name, role, password_hash, totp_secret, created_at
                FROM local_accounts WHERE email = :email
            """), {'email': email}).mappings().fetchone()
        return dict(row) if row else None
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return None


@st.cache_data(show_spinner=False, ttl=15)
def list_local_accounts() -> pd.DataFrame:
    """Every local account, most recently created first -- for the
    Administrator-only Local Accounts page. Never includes password_hash
    or the raw totp_secret (only whether one is set, as totp_enabled) --
    password_hash is only ever read by get_local_account(), for verifying
    an actual login attempt, and a TOTP secret should never leave the
    database once written. Empty DataFrame if unavailable. Cached (short
    ttl, cleared immediately by create/update-role/delete/set_totp_secret/
    clear_totp_secret below) -- this ran uncached on every render of the
    Users page, and again on every single admin action taken there
    (approve/deny/revoke/restore/role-change/delete all trigger a full
    rerun)."""
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(text("""
                SELECT email, name, role, created_at, (totp_secret IS NOT NULL) AS totp_enabled
                FROM local_accounts ORDER BY created_at DESC
            """), conn)
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return pd.DataFrame()


def create_local_account(email: str, name: str, role: str, password: str) -> bool:
    """Creates a new local account. Fails (returns False, no exception) if
    that email already has one -- INSERT with no ON CONFLICT clause simply
    errors on a duplicate key, which the except below turns into a clean
    False rather than a raised exception, so the Local Accounts page can
    show 'that email already has an account' instead of a stack trace."""
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO local_accounts (email, name, role, password_hash)
                VALUES (:email, :name, :role, :password_hash)
            """), {
                'email': email, 'name': name, 'role': role,
                'password_hash': hash_password(password),
            })
        list_local_accounts.clear()  # so the new account shows up on the Users page immediately
        return True
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return False


def update_local_account_password(email: str, new_password: str) -> bool:
    """Sets a new password for an existing local account -- used by BOTH
    the Administrator's reset-password action on the Local Accounts page
    and a signed-in user's own self-service change-password form (the
    Profile page, app.py). Same hashing as every other password write in
    this file; the caller is responsible for having already verified
    whatever it needs to (the admin's own role, or the user's current
    password) before calling this."""
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                UPDATE local_accounts SET password_hash = :password_hash WHERE email = :email
            """), {'email': email, 'password_hash': hash_password(new_password)})
        return result.rowcount > 0
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return False


def update_local_account_profile(email: str, new_name: str, new_email: str) -> bool:
    """Super-Admin-only self-service: changes a local account's own name
    and/or email (email is this table's primary key, so this covers both
    a name-only edit and an actual email change in one statement).
    Analysts and Administrators can't reach this at all -- app.py's
    Profile page only renders this form for auth_method == 'password'
    AND role == 'Super Admin', same "only touch what the caller already
    checked" split every other write in this file follows. Returns False
    (not a raised exception) if new_email is already used by a different
    account -- same duplicate-key handling as create_local_account()."""
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                UPDATE local_accounts SET email = :new_email, name = :new_name WHERE email = :email
            """), {'email': email, 'new_email': new_email, 'new_name': new_name})
        list_local_accounts.clear()  # so a changed name/email shows up on the Users page immediately
        return result.rowcount > 0
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return False


def set_totp_secret(email: str, secret: str) -> bool:
    """Turns two-factor auth ON for a local account -- called only after
    auth.py's setup flow has already verified a real code from the user's
    authenticator app against this exact secret, so a row only ever ends
    up with a secret that's proven to work, never one from an abandoned
    or failed setup attempt."""
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                UPDATE local_accounts SET totp_secret = :secret WHERE email = :email
            """), {'email': email, 'secret': secret})
        list_local_accounts.clear()
        return result.rowcount > 0
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return False


def clear_totp_secret(email: str) -> bool:
    """Turns two-factor auth OFF -- the user's own self-service "disable"
    action, or a Super Admin's recovery action for someone locked out
    after losing their authenticator device (there's no other way back in
    for them otherwise, since a TOTP secret is never displayed or
    recoverable once set)."""
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                UPDATE local_accounts SET totp_secret = NULL WHERE email = :email
            """), {'email': email})
        list_local_accounts.clear()
        return result.rowcount > 0
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return False


def update_local_account_role(email: str, role: str) -> bool:
    """Admin-only: change an existing local account's role. Deliberately
    separate from update_local_account_password -- a self-service caller
    should never be able to reach this one."""
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                UPDATE local_accounts SET role = :role WHERE email = :email
            """), {'email': email, 'role': role})
        list_local_accounts.clear()  # so the new role shows up on the Users page immediately
        return result.rowcount > 0
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return False


def delete_local_account(email: str) -> bool:
    """Removes a local account entirely -- the Local Accounts page's
    'Delete' action. Does not touch that email's Google-sign-in access
    (the separate `users` table, if they also have a row there) -- the
    two login paths are independent, deleting one doesn't affect the
    other."""
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            conn.execute(text('DELETE FROM local_accounts WHERE email = :email'), {'email': email})
        list_local_accounts.clear()  # so the deleted account disappears from the Users page immediately
        return True
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return False


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
        list_pending_requests.clear()  # so the new request shows up on the Users page immediately
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


@st.cache_data(show_spinner=False, ttl=15)
def list_pending_requests() -> pd.DataFrame:
    """All pending access requests, oldest first -- for the Administrator-
    only Access Requests page. Empty DataFrame if unavailable. Cached
    (short ttl, cleared immediately by request_access()/
    decide_access_request() below) -- same reasoning as
    list_local_accounts() above: uncached, this ran on every render of
    the Users page and every admin action taken there."""
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
        # Approve moves the row from pending to approved; deny moves it from
        # pending to denied -- clearing all three unconditionally is simpler
        # and just as cheap as branching on `approve` to clear only the two
        # actually affected.
        list_pending_requests.clear()
        list_approved_users.clear()
        list_denied_users.clear()
        return True
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return False


@st.cache_data(show_spinner=False, ttl=15)
def list_approved_users() -> pd.DataFrame:
    """Every currently-approved Google-sign-in user (both people who went
    through the request queue and the GOOGLE_ADMIN_EMAILS bootstrap seed --
    see handle_google_redirect() in auth.py) -- for the Administrator-only
    user-management view, where their role can be changed or their access
    revoked. Local (email+password) accounts are a completely separate
    table/page (local_accounts / Local Accounts) -- this is Google
    sign-in access only. Cached (short ttl, cleared immediately by
    decide_access_request()/update_user_role()/revoke_user_access() --
    same reasoning as list_pending_requests() above."""
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(text("""
                SELECT email, name, role, decided_at FROM users
                WHERE status = 'approved' ORDER BY decided_at DESC NULLS LAST
            """), conn)
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=15)
def list_denied_users() -> pd.DataFrame:
    """Every Google sign-in user whose access is currently revoked/denied --
    the counterpart to list_approved_users(). Without this, a revoked user
    simply vanished from every admin view once decided (list_pending_requests
    only shows 'pending', list_approved_users only shows 'approved') -- a
    Super Admin could revoke someone but had no way to see or reverse that
    decision afterward; only the affected user, from their own "Access
    denied" screen, could submit a fresh request to get back on the pending
    queue. This lets a Super Admin restore access directly instead, with no
    dependency on the revoked person doing anything first."""
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(text("""
                SELECT email, name, role, decided_at FROM users
                WHERE status = 'denied' ORDER BY decided_at DESC NULLS LAST
            """), conn)
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return pd.DataFrame()


def update_user_role(email: str, role: str) -> bool:
    """Promotes/demotes an approved Google-sign-in user -- this is what
    makes GOOGLE_ADMIN_EMAILS a one-time bootstrap seed rather than a
    standing override (see handle_google_redirect()'s docstring in
    auth.py): once a `users` row exists, this is the only thing that can
    change its role, and it takes effect on that person's very next
    login regardless of what GOOGLE_ADMIN_EMAILS still says."""
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                UPDATE users SET role = :role WHERE email = :email AND status = 'approved'
            """), {'email': email, 'role': role})
        list_approved_users.clear()  # so the new role shows up on the Users page immediately
        return result.rowcount > 0
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return False


def revoke_user_access(email: str, decided_by: str) -> bool:
    """Revokes an approved user's Google-sign-in access. Sets status back
    to 'denied' -- deliberately NOT a hard delete: handle_google_redirect()
    only ever consults GOOGLE_ADMIN_EMAILS when a `users` row is entirely
    absent (record is None), so deleting a bootstrap admin's row would
    make them look brand-new again and silently RE-GRANT them Administrator
    on their next login if their email is still in that list -- keeping
    the row present with status='denied' is what actually, permanently
    blocks them (same mechanism decide_access_request()'s deny path
    already relies on for the ordinary request-queue case)."""
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                UPDATE users SET status = 'denied', decided_at = :now, decided_by = :decided_by
                WHERE email = :email AND status = 'approved'
            """), {'email': email, 'now': datetime.now(), 'decided_by': decided_by})
        list_approved_users.clear()  # so the revoked user disappears from the approved list immediately
        list_denied_users.clear()  # ...and appears on the revoked list immediately
        return result.rowcount > 0
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
        list_approved_users.clear()  # in case this changed an existing row's name/role
        return True
    except Exception as exc:
        print('db.py error:', traceback.format_exc())  # shows up in server logs
        st.session_state.db_error = str(exc)
        return False
