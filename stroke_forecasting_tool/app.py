import os
import tempfile
from pathlib import Path


def _bootstrap_secrets_from_env():
    """Streamlit Cloud and local dev provide secrets via a .streamlit/
    secrets.toml file; hosts like Hugging Face Spaces provide them as plain
    environment variables (Repository secrets) instead, and st.login()/
    st.secrets only ever read from the TOML file -- there's no built-in way
    to point them at env vars directly. If a real secrets.toml already
    exists, this does nothing. Otherwise, if the expected env vars are
    present, it writes one -- regenerated fresh on every container start,
    never committed to git, so real secrets never touch the repo either way.
    """
    secrets_path = Path(__file__).parent / '.streamlit' / 'secrets.toml'
    if secrets_path.exists():
        return
    client_id = os.environ.get('GOOGLE_CLIENT_ID', '')
    database_url = os.environ.get('DATABASE_URL', '')
    if not client_id and not database_url:
        return  # nothing to bootstrap; app runs with those features disabled
    secrets_path.parent.mkdir(parents=True, exist_ok=True)
    secrets_path.write_text(
        '[auth]\n'
        f'redirect_uri = "{os.environ.get("GOOGLE_REDIRECT_URI", "")}"\n'
        f'cookie_secret = "{os.environ.get("GOOGLE_COOKIE_SECRET", "")}"\n'
        f'client_id = "{client_id}"\n'
        f'client_secret = "{os.environ.get("GOOGLE_CLIENT_SECRET", "")}"\n'
        'server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"\n'
        '\n'
        '[database]\n'
        f'url = "{database_url}"\n'
    )


_bootstrap_secrets_from_env()

import contextlib
import gzip
import io
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from auth import (
    complete_sign_out, handle_google_redirect, init_session_state, initials, render_login,
    render_transition_spinner,
)
from branding import TITLE_LOGO_PATH
from db import (
    clear_totp_secret, create_local_account, db_configured, decide_access_request, delete_dashboard_run,
    delete_local_account, get_local_account, list_approved_users, list_dashboard_runs, list_denied_users,
    list_local_accounts, list_pending_requests, load_dashboard_run, mark_runs_seen,
    revoke_user_access, save_dashboard_run, set_totp_secret, update_local_account_password,
    update_local_account_profile, update_local_account_role, update_user_role, verify_password,
)
# pipeline.* modules are deliberately NOT imported here at module level --
# measured directly, importing them (they pull in scikit-learn, statsmodels,
# and lifetimes) costs ~1.3s of the ~1.8s this file's imports take in total,
# and every one of those seconds was being paid on EVERY app start just to
# reach the login page, which calls none of this. Each is imported lazily
# instead, right inside the one or two functions that actually call it
# (cached_ml_forecast, cached_linear_forecast, cached_ltv_fit,
# run_pipeline_models, and the Data Pipeline run handler below) -- Python
# caches an import in sys.modules after the first real one, so this only
# defers WHEN that ~1.3s is paid (once real work is happening, not before
# the login page even renders), it doesn't pay it more than once.
# `import ui` (a module reference), NOT `from ui import AMBER, TEXT, ...` --
# confirmed live in dark mode: every colour constant below is reassigned by
# ui.py's own _apply_palette() on every rerun (see its docstring in ui.py),
# but `from ui import TEXT` copies whatever TEXT was AT IMPORT TIME into a
# brand-new name in this module's own namespace -- it does not track ui.py's
# name being reassigned later, so every f-string in this file that used a
# bare {TEXT}/{TEAL}/etc. was permanently stuck on the LIGHT-mode value
# forever, regardless of the dark-mode toggle (this is exactly why KPI icons,
# chart lines, and section dividers looked washed out / low-contrast in dark
# mode -- they were light-mode colours rendered on a dark background). Every
# call site below now does ui.TEXT/ui.TEAL/etc., which re-reads ui.py's
# current module attribute at the moment it's actually used.
import ui
from ui import (
    CACHED_RUN_BANNER_PAGES, card, chart, clear_nav_overlay, empty_state, inject_global_css,
    kpi, new_execution_log, overall_progress, page_header, pill, render_action_button_css,
    render_cached_run_banner, render_nav_transition_overlay, render_sidebar, render_startup_progress,
    render_topbar, render_upload_progress_tracker, stage_row, upload_slot,
)


# ── Cached forecast fits ────────────────────────────────────────────────────
# fit_ml_forecast in particular refits a 300-estimator, direct-multi-step
# gradient-boosting model several pages call it just to populate a
# comparison table, and Streamlit reruns the *entire* script on every widget
# interaction, so without caching that refit happened on every click, even
# ones unrelated to the forecast (this was the actual cause of the page
# feeling slow to load/interact with, not just a missing spinner). monthly
# only changes after a new pipeline run, which naturally produces a
# different cache key, so no manual invalidation is needed.
#
# max_entries=20 on all three -- this cache is process-wide (shared across
# every session, not per-user), and its key is the actual `monthly`/`master`
# DataFrame content, which changes every time ANYONE loads a different past
# run from Run History (unbounded in principle -- the run-history table
# itself has no row cap). Without a bound, every distinct historical run
# anyone ever opens adds a permanent entry that's never evicted -- a slow
# memory leak, not an immediate crash, but a real one on a host this memory-
# constrained (see hosting-options.md). 20 comfortably covers "the handful
# of runs someone's actually flipping between in one sitting" while
# guaranteeing old entries eventually fall off.
@st.cache_data(show_spinner=False, max_entries=20)
def cached_ml_forecast(monthly, n_forecast=24):
    from pipeline.ml_forecast import fit_ml_forecast
    return fit_ml_forecast(monthly, n_forecast=n_forecast)


@st.cache_data(show_spinner=False, max_entries=20)
def cached_linear_forecast(monthly, n_train=24, n_forecast=24):
    from pipeline.forecast import fit_linear_forecast
    return fit_linear_forecast(monthly, n_train=n_train, n_forecast=n_forecast)


# Donor LTV (Pareto/NBD + Gamma-Gamma) is by far the slowest fit in the app
# (~5 min) and nothing on the main pipeline path reads its output only the
# separate Donor Lifetime Value page does. It's deliberately NOT part of
# run_pipeline_models() any more; it's fit lazily, right here, the first
# time that page is opened, keyed on `master` so revisiting the page or
# switching pages elsewhere never refits it.
@st.cache_data(show_spinner=False, max_entries=20)
def cached_ltv_fit(master):
    from pipeline.ltv_model import fit_ltv_model
    return fit_ltv_model(master)


def run_pipeline_models(master, stages=None, progress=None, log=None):
    """Fits the linear, ML and stock-flow models against `master`,
    computes every page-level breakdown that's derived from it (income by
    supplier/campaign type, retention curves, ...), and populates the
    session_state keys the rest of the app reads from. Donor LTV
    (Pareto/NBD + Gamma-Gamma) is deliberately NOT fit here any more -- it
    was the slowest stage (~5 min) and only the separate Donor Lifetime
    Value page reads its output, so it's fit lazily there instead, on
    first visit (see cached_ltv_fit() near the top of this file). Only
    called for a LIVE run, on a freshly-built master -- restoring a cached
    dashboard state populates the same session_state keys directly
    instead (see restore_dashboard_state() below), it doesn't call this or
    need `master` at all. That split is what keeps the cache lightweight:
    `master` itself (1M+ rows on real data) never gets persisted or
    re-read, only these much smaller derived tables are.

    stages: optional {stage_number: st.empty()} for the Data Pipeline
    page's numbered stage list (stages 3-5 -- 1/2/6 are updated by the
    caller around file-build and dashboard-save, which live outside this
    function). None outside a live pipeline run (e.g. tests) -- every
    stage_row() call below is guarded on it.
    progress: optional (placeholder, start_count) for the "OVERALL
    PROGRESS" bar -- start_count is how many of the 6 total stages are
    already done by the time this is called (always 2: file ingestion +
    master build, both handled by the caller before this runs). Updated
    live, in real fractions (done stages / 6), as each of stages 3-5
    reaches a real end-state (done/warning/error) -- never incremented on
    'running', so the bar never claims progress that hasn't happened.
    log: optional log(msg) callable (see ui.new_execution_log) for the
    Execution Log panel -- a no-op when not supplied (e.g. tests).
    """
    from pipeline.forecast import (
        get_campaign_roi, get_monthly_actuals, get_retention_by_segment, get_supplier_breakdown,
    )
    from pipeline.stock_flow_forecast import build_production_forecast

    N_STAGES = 6
    log = log or (lambda msg: None)
    prog_placeholder, done = progress or (None, 0)

    def stage(n, name, cat, state, detail=''):
        nonlocal done
        if stages:
            stage_row(stages[n], n, name, cat, state, detail)
        if prog_placeholder and state in ('done', 'warning', 'error'):
            done += 1
            overall_progress(prog_placeholder, done, N_STAGES)

    stage(3, 'Linear Trend Baseline', 'Trend', 'running')
    with st.spinner('Fitting the linear-trend baseline (trained from 2019, 12-month holdout)…'):
        monthly = get_monthly_actuals(master)
        forecast_df_linear, mape_linear, linear_slope, _ = cached_linear_forecast(monthly, n_train=24, n_forecast=24)
        log(f':material/check_circle: Linear baseline fitted, {mape_linear:.1f}% MAPE')
        stage(3, 'Linear Trend Baseline', 'Trend', 'done', f'{mape_linear:.1f}% MAPE')

    stage(4, 'ML Forecast Model', 'Gradient Boosting', 'running')
    with st.spinner('Training the gradient-boosting forecast model (direct multi-step, trained from 2019, 12-month holdout)…'):
        try:
            forecast_df_ml, mape_ml, importances, trend_slope = cached_ml_forecast(monthly, n_forecast=24)
            log(f':material/check_circle: ML forecast model trained, {mape_ml:.1f}% MAPE')
            stage(4, 'ML Forecast Model', 'Gradient Boosting', 'done', f'{mape_ml:.1f}% MAPE')
        except ValueError:
            log(':material/warning: Not enough monthly history for the ML model, using the linear baseline instead.')
            forecast_df_ml, mape_ml, importances, trend_slope = forecast_df_linear, mape_linear, {}, linear_slope
            stage(4, 'ML Forecast Model', 'Gradient Boosting', 'warning', 'Fell back to linear')

    # Donor LTV (Pareto/NBD + Gamma-Gamma) is no longer fit here -- it was
    # the slowest stage in the pipeline (~5 min) and nothing downstream of
    # this function reads its output, only the separate Donor Lifetime
    # Value page does. It's fit lazily there instead, on first visit (see
    # cached_ltv_fit() above) -- these session_state keys stay set to None
    # from a live run so every page's existing `if x_state is None:` checks
    # and the Run History schema keep working unchanged.
    ltv_results = ltv_tuning = ltv_metrics = ltv_error = ltv_monthly = None

    stage(5, 'Stock-Flow Model', 'SARIMA + Cohort Survival', 'running')
    with st.spinner('Fitting the stock-flow model (recruits × retention × gift, SARIMA + cohort survival)…'):
        def on_sf_progress(msg):
            stage(5, 'Stock-Flow Model', 'SARIMA + Cohort Survival', 'running', msg)

        try:
            sf_result = build_production_forecast(master, progress_callback=on_sf_progress)
            sf_error = None
            sf_mape = sf_result['walkforward_summary']['income_mape'].mean()
            log(f':material/check_circle: Stock-flow model fitted, '
                     f'{sf_mape:.1f}% avg MAPE across 3 walk-forward windows')
            stage(5, 'Stock-Flow Model', 'SARIMA + Cohort Survival', 'done', f'{sf_mape:.1f}% avg MAPE')
        except Exception as exc:
            sf_result = None
            sf_error = str(exc)
            log(f':material/warning: Stock-flow model skipped, {sf_error}')
            log(traceback.format_exc())
            stage(5, 'Stock-Flow Model', 'SARIMA + Cohort Survival', 'warning', 'Skipped')

    # sBG/BG-NBD and the gift-waterfall bridge are no longer fitted -- the
    # Income Forecast page only offers ML/Linear/Stock-flow now. These
    # session_state keys stay set to None (rather than removed) so
    # build_dashboard_state()/restore_dashboard_state() and the Run History
    # schema keep working unchanged; they'll just always be empty going
    # forward.
    sbg_results = sbg_monthly = bgnbd_monthly = sbg_metrics = sbg_error = None
    gw_monthly = mape_gw = gw_metrics = gw_error = None

    with st.spinner('Finalising and updating dashboard views…'):
        st.session_state.master             = master
        st.session_state.monthly            = monthly
        st.session_state.forecast_df        = forecast_df_ml
        st.session_state.mape               = mape_ml
        st.session_state.ml_importances     = importances
        st.session_state.ml_trend_slope     = trend_slope
        st.session_state.forecast_df_linear = forecast_df_linear
        st.session_state.mape_linear        = mape_linear
        st.session_state.ltv_results        = ltv_results
        st.session_state.ltv_tuning         = ltv_tuning
        st.session_state.ltv_metrics        = ltv_metrics
        st.session_state.ltv_error          = ltv_error
        st.session_state.ltv_monthly        = ltv_monthly
        if ltv_results is not None and not ltv_results.empty:
            counts, edges = np.histogram(ltv_results['predicted_ltv_24m'].dropna(), bins=30)
            st.session_state.ltv_histogram = {'bin_edges': edges.tolist(), 'counts': counts.tolist()}
        else:
            st.session_state.ltv_histogram = None
        if sf_result is not None:
            st.session_state.sf_walkforward = sf_result['walkforward_summary']
            st.session_state.sf_components  = sf_result['components_df']
            st.session_state.sf_zone12      = sf_result['zone12_forecast']
            st.session_state.sf_zone3       = sf_result['zone3_scenarios']
            st.session_state.sf_assumptions = sf_result['assumptions']
            st.session_state.mape_stockflow = float(sf_result['walkforward_summary']['income_mape'].mean())
        else:
            st.session_state.mape_stockflow = None
        st.session_state.sf_error           = sf_error
        st.session_state.sbg_results        = sbg_results
        st.session_state.sbg_monthly        = sbg_monthly
        st.session_state.bgnbd_monthly      = bgnbd_monthly
        st.session_state.sbg_metrics        = sbg_metrics
        st.session_state.mape_sbg           = sbg_metrics.get('sbg_holdout_mape') if sbg_metrics else None
        st.session_state.mape_bgnbd         = sbg_metrics.get('bgnbd_holdout_mape') if sbg_metrics else None
        st.session_state.gw_monthly         = gw_monthly
        st.session_state.mape_gw            = mape_gw
        st.session_state.gw_metrics         = gw_metrics
        st.session_state.gw_error           = gw_error
        st.session_state.sbg_error          = sbg_error
        st.session_state.pipeline_run       = True

        # ── Page-level breakdowns derived from `master` -- computed once
        # here (not on every page rerun, which is what the pages used to do
        # inline), and this session_state shape is exactly what
        # build_dashboard_state() below reads from to save the cached
        # dashboard state -- so a loaded cached state and a live run always
        # populate pages identically. donor_month is always normalised to a
        # 'YYYY-MM' string here, matching what JSON round-tripping produces
        # on the way back out of the cache (a Period column can't survive
        # JSON serialization as anything but a string anyway).
        paid = master[master['paid_flag'] == True]
        st.session_state.master_rows         = len(master)
        st.session_state.donor_count         = int(master['recurring_payment_id'].nunique())
        st.session_state.contact_count       = int(master['contact_id'].nunique())
        st.session_state.supplier_count      = int(master['supplier'].nunique())
        st.session_state.campaign_type_count = int(master['campaign_type'].nunique())
        st.session_state.total_income        = float(paid['success_amt'].sum())
        st.session_state.supplier_summary = (
            paid.groupby('supplier')
                .agg(total_income=('success_amt', 'sum'),
                     active_donors=('recurring_payment_id', 'nunique'),
                     avg_gift=('success_amt', 'mean'))
                .reset_index().sort_values('total_income', ascending=False)
        )
        st.session_state.campaign_summary = get_campaign_roi(master)
        supplier_monthly = get_supplier_breakdown(master)
        supplier_monthly['donor_month'] = supplier_monthly['donor_month'].astype(str)
        st.session_state.supplier_monthly = supplier_monthly
        campaign_monthly = (
            paid.groupby(['campaign_type', 'donor_month'])['success_amt']
                .sum().reset_index().rename(columns={'success_amt': 'total_income'})
        )
        campaign_monthly['donor_month'] = campaign_monthly['donor_month'].astype(str)
        st.session_state.campaign_monthly = campaign_monthly
        st.session_state.retention_by_segment = {
            seg: get_retention_by_segment(master, seg)
            for seg in ('supplier', 'campaign_type', 'recruit_year')
        }
        log(':material/check_circle: Dashboard views updated')


def _df_records(df):
    """DataFrame -> list-of-dicts, JSON-ready. Empty/None -> []."""
    return df.to_dict('records') if df is not None and not df.empty else []


def _df_or_none(records):
    """list-of-dicts (JSON-loaded) -> DataFrame, or None if empty -- mirrors
    what a live pipeline run leaves in session_state when a model wasn't
    available (None, not an empty DataFrame), so every page's existing
    `if x_state is None:` checks work unchanged either way."""
    return pd.DataFrame(records) if records else None


def build_dashboard_state():
    """Extracts the display-level data every page renders -- the
    aggregated summaries, chart series, table rows, and scalar metrics
    already sitting in session_state after run_pipeline_models() -- into
    one JSON-ready dict, one key per dashboard page's worth of data. No raw
    master panel (that's the one thing genuinely too large and not needed
    by any page -- see run_pipeline_models()'s own docstring), but the full
    per-donor LTV table IS included: at real scale it compresses to ~2MB,
    comfortably inside the 5MB per-run budget, and keeping it is what lets
    a past run's "download all donor predictions" button keep working, not
    just its top-15/top-50 views.
    """
    ss = st.session_state

    return {
        'scalars': {
            'master_rows': ss.master_rows, 'donor_count': ss.donor_count,
            'contact_count': ss.contact_count, 'supplier_count': ss.supplier_count,
            'campaign_type_count': ss.campaign_type_count, 'total_income': ss.total_income,
            'mape': ss.mape, 'mape_linear': ss.mape_linear, 'mape_stockflow': ss.mape_stockflow,
            'mape_sbg': ss.mape_sbg, 'mape_bgnbd': ss.mape_bgnbd, 'mape_gw': ss.mape_gw,
            'ml_importances': ss.ml_importances, 'ltv_metrics': ss.ltv_metrics,
            'sf_assumptions': ss.sf_assumptions, 'ltv_histogram': ss.ltv_histogram,
            'gw_metrics': ss.gw_metrics,
        },
        'monthly': _df_records(ss.monthly),
        'supplier_summary': _df_records(ss.supplier_summary),
        'campaign_summary': _df_records(ss.campaign_summary),
        'supplier_monthly': _df_records(ss.supplier_monthly),
        'campaign_monthly': _df_records(ss.campaign_monthly),
        'retention_by_segment': {k: _df_records(v) for k, v in (ss.retention_by_segment or {}).items()},
        'forecasts': {
            'ml': _df_records(ss.forecast_df), 'linear': _df_records(ss.forecast_df_linear),
            'ltv': _df_records(ss.ltv_monthly), 'stockflow': _df_records(ss.sf_zone12),
            'sbg': _df_records(ss.sbg_monthly), 'bgnbd': _df_records(ss.bgnbd_monthly),
            'gift_waterfall': _df_records(ss.gw_monthly),
        },
        'sf_zone3': {k: _df_records(v) for k, v in (ss.sf_zone3 or {}).items()},
        'ltv_tuning': _df_records(ss.ltv_tuning),
        'ltv_donors': _df_records(ss.ltv_results),
        'sbg_results': _df_records(ss.sbg_results),
        'sbg_metrics': ss.sbg_metrics,
    }


def restore_dashboard_state(state):
    """The inverse of build_dashboard_state() -- populates every
    session_state key the pages read from, from a previously-saved cache.
    Deliberately mirrors run_pipeline_models()'s own assignments key for
    key, so every page behaves identically whether its data came from a
    live run or a cached one."""
    ss = st.session_state
    scalars = state.get('scalars', {})
    for key in ('master_rows', 'donor_count', 'contact_count', 'supplier_count',
                'campaign_type_count', 'total_income', 'mape', 'mape_linear', 'mape_stockflow',
                'mape_sbg', 'mape_bgnbd', 'mape_gw', 'ml_importances', 'ltv_metrics', 'sf_assumptions',
                'ltv_histogram', 'gw_metrics'):
        ss[key] = scalars.get(key)

    ss.monthly           = pd.DataFrame(state.get('monthly', []))
    ss.supplier_summary  = pd.DataFrame(state.get('supplier_summary', []))
    ss.campaign_summary  = pd.DataFrame(state.get('campaign_summary', []))
    ss.supplier_monthly  = pd.DataFrame(state.get('supplier_monthly', []))
    ss.campaign_monthly  = pd.DataFrame(state.get('campaign_monthly', []))
    ss.retention_by_segment = {k: pd.DataFrame(v) for k, v in state.get('retention_by_segment', {}).items()}

    forecasts = state.get('forecasts', {})
    ss.forecast_df        = _df_or_none(forecasts.get('ml'))
    ss.forecast_df_linear = _df_or_none(forecasts.get('linear'))
    ss.ltv_monthly        = _df_or_none(forecasts.get('ltv'))
    ss.sf_zone12          = _df_or_none(forecasts.get('stockflow'))
    ss.sbg_monthly        = _df_or_none(forecasts.get('sbg'))
    ss.bgnbd_monthly      = _df_or_none(forecasts.get('bgnbd'))
    ss.gw_monthly         = _df_or_none(forecasts.get('gift_waterfall'))

    ss.sf_zone3    = {k: pd.DataFrame(v) for k, v in state.get('sf_zone3', {}).items()}
    ss.ltv_tuning  = _df_or_none(state.get('ltv_tuning'))
    ss.ltv_results = _df_or_none(state.get('ltv_donors'))
    ss.sbg_results = _df_or_none(state.get('sbg_results'))
    ss.sbg_metrics = state.get('sbg_metrics')

    # Not restorable from a lightweight cache (no equivalent stored -- these
    # are large intermediate/raw objects, not display data): ltv_error,
    # sf_error, sbg_error, gw_error, ml_trend_slope, sf_walkforward, sf_components.
    # Every page already falls back to a generic message when these are
    # None, so leaving them unset is safe.
    ss.pipeline_run = True


def build_page_export_csv(page_name):
    """Bundles every backing table for `page_name` into one CSV, one table
    per section (a '## <name>' header line, then that table's own header +
    rows, then a blank line before the next section) -- a plain-text
    convention that stays readable in a text editor and re-splits cleanly
    in Excel/Sheets. Reads the same session_state DataFrames each page's
    own charts/tables already read from (set by run_pipeline_models() or
    restore_dashboard_state() -- both populate the same keys, so this
    works identically for a live run or one reloaded from history) --
    which is often MORE complete than what any single on-page table shows
    (e.g. the Donor Lifetime Value page's own table caps at the top 50
    rows; this exports the full donor table it's drawn from). Returns None
    if the page has nothing tabular to export, or no data is loaded yet.
    """
    ss = st.session_state
    if not ss.pipeline_run:
        return None

    tables = {}  # section title -> DataFrame
    if page_name == 'Overview':
        tables['Monthly actuals'] = ss.monthly
        tables['Supplier summary'] = ss.supplier_summary
        tables['Campaign summary'] = ss.campaign_summary
    elif page_name == 'Income Forecast':
        tables['Monthly actuals'] = ss.monthly
        tables['ML forecast (24m)'] = ss.forecast_df
        tables['Linear forecast (24m)'] = ss.forecast_df_linear
        tables['Stock-flow forecast (months 1-18)'] = ss.sf_zone12
        for name, df in (ss.sf_zone3 or {}).items():
            tables[f'Stock-flow scenario: {name} (months 19-36)'] = df
    elif page_name == 'Retention Analysis':
        for seg, df in (ss.retention_by_segment or {}).items():
            tables[f'Retention by {seg}'] = df
    elif page_name == 'Donor Lifetime Value':
        tables['Donor LTV predictions (full table)'] = ss.ltv_results
        tables['Model validation (penalizer grid search)'] = ss.ltv_tuning
    elif page_name == 'Supplier Insights':
        tables['Supplier summary'] = ss.supplier_summary
        tables['Supplier monthly income'] = ss.supplier_monthly
    elif page_name == 'Campaign ROI':
        tables['Campaign summary'] = ss.campaign_summary
        tables['Campaign monthly income'] = ss.campaign_monthly
    elif page_name == 'Run History' and db_configured():
        tables['Pipeline runs'] = list_dashboard_runs()

    tables = {name: df for name, df in tables.items() if df is not None and not df.empty}
    if not tables:
        return None
    return _tables_to_csv(tables)


# Split out of build_page_export_csv() above specifically so the EXPENSIVE
# part (df.to_csv() over every table -- on the Donor Lifetime Value page in
# particular, the full per-donor LTV table, potentially tens of thousands
# of rows) is cached, while the CHEAP part (deciding which session_state
# DataFrames belong to which page) still runs every time. build_page_export_
# csv() itself can't be cached directly -- it reads a different, page-
# dependent set of session_state keys internally that @st.cache_data has no
# way to see, so caching on page_name alone would silently serve a stale
# export after the underlying data changes. Caching on `tables` instead
# sidesteps that entirely: the cache key IS the actual DataFrame content,
# so a cache hit is only ever returned when the data is genuinely unchanged
# (Streamlit's hasher supports dict-of-DataFrame arguments natively).
@st.cache_data(show_spinner=False, max_entries=20)
def _tables_to_csv(tables: dict) -> str:
    buf = io.StringIO()
    for name, df in tables.items():
        buf.write(f'## {name}\n')
        df.to_csv(buf, index=False)
        buf.write('\n')
    return buf.getvalue()


@st.fragment
def _render_2fa_card(_email):
    """The Profile page's "Two-factor authentication" card, as its own
    fragment -- every interaction inside it (Set up 2FA, Cancel, Verify
    and enable, Disable 2FA) reruns only this card, not the whole script.
    Without this, Streamlit's normal full-script rerun behaviour fades
    every OTHER element on the page while it re-executes -- reported live
    as "the whole screen turning translucent white" on both Set up 2FA
    and Cancel, exactly the full-page loading effect the local QR
    spinner below was meant to avoid in the first place; a spinner inside
    a fragment stays genuinely local since nothing outside the fragment
    ever re-renders at all.

    _email is the only thing threaded in from the Profile page's own
    render -- everything else (the account row, whether 2FA is already
    on) is read fresh from the database on every fragment run rather than
    passed in as a snapshot, so a fragment-scoped rerun after enabling/
    disabling always reflects the real current state, not a stale value
    captured before this fragment last ran. st.rerun(scope='fragment')
    (not the plain, whole-app st.rerun() used everywhere else in this
    file) is what keeps those follow-up reruns local too.
    """
    _account = get_local_account(_email)
    with card(title='Two-factor authentication'):
        _totp_enabled = bool(_account and _account.get('totp_secret'))
        if _totp_enabled:
            st.success('Two-factor authentication is enabled.', icon=':material/check_circle:')
            st.caption('Every sign-in will ask for a code from your authenticator app.')
            # A second confirming step (current password, inside its own
            # form) before actually turning it off -- same "prove you're
            # still you" reasoning as Change password re-checking the
            # current password above, for a change that lowers this
            # account's security.
            with st.form('totp_disable_form', border=False):
                totp_disable_pw = st.text_input(
                    'Enter your password to disable', type='password', key='totp_disable_pw',
                )
                disable_submitted = st.form_submit_button(
                    'Disable 2FA', icon=':material/remove_moderator:', width='stretch',
                )
            if disable_submitted:
                if not verify_password(totp_disable_pw, _account['password_hash']):
                    st.error('Incorrect password.', icon=':material/error:')
                elif clear_totp_secret(_email):
                    st.success('Two-factor authentication disabled.', icon=':material/check_circle:')
                    st.rerun(scope='fragment')
                else:
                    st.error('Could not disable two-factor authentication right now. '
                              'Try again shortly.', icon=':material/error:')
        else:
            st.caption('Add an extra step at sign-in using an authenticator app '
                       '(Google Authenticator, Authy, 1Password, etc).')
            # The secret lives in session_state ONLY from here until a
            # real code confirms it below -- set_totp_secret() (db.py)
            # is never called until that verification succeeds, so a
            # user who starts setup and never finishes it leaves no
            # half-configured 2FA behind in the database, just an
            # abandoned value in this session that a rerun/new session
            # never sees again.
            if not st.session_state.get('totp_setup_secret'):
                # A placeholder, not a bare st.button() call, specifically so
                # a successful click can clear the button back out again
                # below -- st.button() always renders its widget regardless
                # of the True/False it returns, so without this the button
                # would stay sitting on screen, now doing nothing, right
                # alongside the QR code that falls through and renders
                # underneath it in this same pass.
                _setup_btn_ph = st.empty()
                if _setup_btn_ph.button('Set up 2FA', key='totp_setup_start',
                                          icon=':material/qr_code_2:', width='stretch'):
                    # Wrapped in its own spinner too, not just the QR
                    # generation step below -- reported live as a
                    # multi-second blank card on the very first "Set up
                    # 2FA" click of a freshly started server, with nothing
                    # shown the whole time: `import pyotp` right here is
                    # itself part of that one-time cold-import cost (see
                    # the QR generation spinner's own comment below for the
                    # detail), and it happens BEFORE that later spinner
                    # even exists. Belt-and-suspenders: covers the actual
                    # slow step regardless of exactly which import turns
                    # out to be the slow one.
                    with st.spinner('Setting up two-factor authentication…'):
                        try:
                            import pyotp
                            st.session_state.totp_setup_secret = pyotp.random_base32()
                            _setup_btn_ph.empty()
                            # No st.rerun() -- falls straight through to the
                            # generation block below in this SAME fragment pass
                            # (fragments rerun automatically on their own widget
                            # clicks regardless, same as the whole app normally
                            # would, but scoped to just this card).
                        except Exception:
                            st.error('Two-factor setup is unavailable right now. Try again shortly, '
                                      'or contact an administrator.', icon=':material/error:')

            if st.session_state.get('totp_setup_secret'):
                # Everything from here on can fail (a missing/broken pyotp or
                # qrcode install, an unpicklable secret, etc) -- caught so a
                # setup-time error shows as a normal in-card message instead of
                # an uncaught traceback replacing the whole page. Cancel is
                # rendered whether generation succeeded or not, so a failure
                # here never leaves the user stuck without a way back.
                _qr_error = False
                _qr_ph = st.empty()

                # The QR is generated ONCE per setup attempt and cached in
                # session_state (raw PNG bytes), not regenerated on every
                # fragment rerun of this card -- confirmed live: a fragment
                # reruns its WHOLE body on any interaction inside it, so
                # without this, clicking Cancel (or submitting the verify
                # form with a wrong code) re-ran this entire generation
                # block again first, from the top, before ever reaching the
                # Cancel/verify handling below -- visibly re-showing
                # "Generating QR code..." for another full 0.6s pause before
                # actually collapsing, instead of collapsing immediately.
                # Caching means a rerun triggered by something OTHER than
                # freshly clicking "Set up 2FA" just reuses the
                # already-generated bytes, near-instantly, with no repeat
                # spinner/pause. Cleared alongside totp_setup_secret itself
                # everywhere that's cleared, so a later "Set up 2FA" click
                # always generates (and shows the loading state for) a
                # genuinely fresh QR rather than a stale cached one.
                if not st.session_state.get('totp_qr_cache'):
                    # Genuinely confirmed live (a two-pass render-then-
                    # st.rerun() attempt was tried here first and STILL
                    # never showed anything, even with a 2-second pause
                    # inserted before the rerun to rule out timing): a
                    # script pass that ends via st.rerun() never flushes
                    # its own in-progress content to the browser at all --
                    # the frontend only ever receives the result of the
                    # pass that actually completes normally. A custom
                    # placeholder rendered mid-pass, no matter how long a
                    # sleep() follows it, can't work here for that reason.
                    # st.spinner() is the one primitive Streamlit itself
                    # guarantees renders DURING blocking code inside a
                    # single pass (it's what every other long-running step
                    # in this app already relies on, e.g.
                    # run_pipeline_models()'s "Fitting the linear-trend
                    # baseline…") -- so that's what actually shows here,
                    # even though it costs the custom QR-sized placeholder
                    # box (a smaller layout jump when the image swaps in,
                    # accepted as the trade-off for a spinner that's
                    # actually visible at all).
                    with st.spinner('Generating your QR code…'):
                        try:
                            import io
                            import time

                            # pyotp/qrcode are both imported HERE, inside the
                            # spinner, not once unconditionally above it --
                            # reported live as the card just sitting blank
                            # (no spinner, no error, nothing) for several
                            # seconds on a freshly started server. Root
                            # cause: pyotp pulls in `cryptography`, and
                            # importing that for the very first time in the
                            # process is genuinely slow (a few real seconds,
                            # confirmed live); that unconditional import used
                            # to run BEFORE this spinner even started, so the
                            # whole delay happened with nothing on screen at
                            # all. Every later "Set up 2FA" click in this
                            # same server process is instant, same as
                            # always, once the module's actually warm.
                            import pyotp
                            import qrcode

                            _secret = st.session_state.totp_setup_secret
                            _uri = pyotp.TOTP(_secret).provisioning_uri(
                                name=_email, issuer_name='Stroke Foundation Donor Forecasting',
                            )
                            _buf = io.BytesIO()
                            qrcode.make(_uri).save(_buf, format='PNG')
                            time.sleep(0.4)
                            st.session_state.totp_qr_cache = _buf.getvalue()
                        except Exception:
                            _qr_error = True
                            with _qr_ph.container():
                                st.error('Could not generate a setup code right now. Try again shortly, '
                                          'or contact an administrator.', icon=':material/error:')
                else:
                    # Cache hit -- generation (and therefore the pyotp
                    # import above) already ran earlier in this process, so
                    # this is an ordinary, instant sys.modules lookup, not a
                    # repeat of the cold-import cost. Needed here too since
                    # the verify step below also calls into pyotp.
                    import pyotp

                if not _qr_error and st.session_state.get('totp_qr_cache'):
                    # _qr_ph.container() again, not _qr_ph.empty() followed by
                    # bare st.image() -- .empty() clears the placeholder back
                    # to zero height FIRST, and the replacement content only
                    # lands (and only actually finishes painting once the
                    # image itself decodes) a beat later, as a SEPARATE
                    # element after it -- reported live as the loading box
                    # visibly collapsing shut and then reopening for the QR,
                    # rather than one smoothly stays-expanded swap. Drawing
                    # straight into the same placeholder replaces its content
                    # in one step, so the box never passes through an
                    # empty/collapsed state at all.
                    with _qr_ph.container():
                        st.image(st.session_state.totp_qr_cache, width=176)
                        st.caption('Scan this with your authenticator app, or enter the key '
                                   f'manually: `{st.session_state.totp_setup_secret}`')

                if not _qr_error:
                    with st.form('totp_verify_form', border=False):
                        setup_code = st.text_input(
                            'Enter the 6-digit code to confirm', placeholder='123456',
                            max_chars=6, key='totp_setup_code',
                        )
                        verify_submitted = st.form_submit_button(
                            'Verify and enable', icon=':material/check:', width='stretch',
                        )
                else:
                    verify_submitted = False

                if st.button('Cancel', key='totp_setup_cancel'):
                    st.session_state.totp_setup_secret = None
                    st.session_state.totp_qr_cache = None
                    st.rerun(scope='fragment')
                if verify_submitted:
                    _secret = st.session_state.totp_setup_secret
                    if setup_code and pyotp.TOTP(_secret).verify(setup_code.strip(), valid_window=1):
                        if set_totp_secret(_email, _secret):
                            st.session_state.totp_setup_secret = None
                            st.session_state.totp_qr_cache = None
                            st.success('Two-factor authentication enabled.', icon=':material/check_circle:')
                            st.rerun(scope='fragment')
                        else:
                            st.error('Could not enable two-factor authentication right now. '
                                      'Try again shortly.', icon=':material/error:')
                    else:
                        st.error('Incorrect code. Please try again.', icon=':material/error:')


@st.fragment
def _render_profile_details_card(_email):
    """The Profile page's "Profile details" card, Super Admin editing case
    only -- as its own fragment for the same reason as _render_2fa_card
    above: toggling Edit/Cancel is a widget interaction, and without
    fragment isolation that fades the whole page the same way clicking
    Set up 2FA used to. Collapsed (read-only name/email) by default,
    matching how the read-only Analyst/Administrator/Google branches
    below already look -- Edit swaps in the same form this used to show
    unconditionally.

    _email is threaded in the same way _render_2fa_card's is; _user and
    the account row are both re-read from session_state/the DB fresh on
    every fragment run rather than captured as snapshot arguments, so a
    Cancel-triggered fragment-scoped rerun always reflects the real
    current values.
    """
    _user = st.session_state.user or {}
    _account = get_local_account(_email)
    _editing = st.session_state.get('profile_details_editing', False)
    with card(title='Profile details', sub='Only Super Admins can change their own name or email.'):
        if not _editing:
            st.markdown(f"""
            <div><div class="sf-eyebrow" style="margin-bottom:2px;">Name</div>
                <div style="font-size:13px;font-weight:600;color:{ui.TEXT};">{_user.get('name', '')}</div></div>
            <div style="margin-top:10px;"><div class="sf-eyebrow" style="margin-bottom:2px;">Email</div>
                <div style="font-size:13px;font-weight:600;color:{ui.TEXT};">{_email}</div></div>
            """, unsafe_allow_html=True)
            if st.button('Edit', key='profile_details_edit', icon=':material/edit:', width='stretch'):
                st.session_state.profile_details_editing = True
                st.rerun(scope='fragment')
        else:
            with st.form('profile_details_form', border=False):
                new_name = st.text_input('Name', value=_user.get('name', ''))
                new_email = st.text_input('Work email', value=_email)
                current_pw_profile = st.text_input(
                    'Current password, to confirm', type='password', key='profile_confirm_pw',
                )
                profile_submitted = st.form_submit_button(
                    'Save changes', icon=':material/check:', width='stretch',
                )
            if st.button('Cancel', key='profile_details_cancel'):
                st.session_state.profile_details_editing = False
                st.rerun(scope='fragment')
            if profile_submitted:
                clean_new_email = new_email.strip().lower()
                if not _account or not verify_password(current_pw_profile, _account['password_hash']):
                    st.error('Current password is incorrect.', icon=':material/error:')
                elif not new_name.strip():
                    st.error('Name cannot be empty.', icon=':material/error:')
                elif not clean_new_email or '@' not in clean_new_email:
                    st.error('Enter a valid email address.', icon=':material/error:')
                elif new_name.strip() == _user.get('name') and clean_new_email == _email:
                    st.info('Nothing to save; name and email are unchanged.', icon=':material/info:')
                elif update_local_account_profile(_email, new_name.strip(), clean_new_email):
                    # A full st.rerun(), not scope='fragment' -- unlike Cancel/
                    # Edit above, this changes st.session_state.user, which the
                    # sidebar, page_header's account-menu button, and this same
                    # page's own "Account details" card above all read too, and
                    # those live OUTSIDE this fragment. A fragment-scoped rerun
                    # would leave all of them showing the old name/email until
                    # some later full rerun happened to come along.
                    st.session_state.user = {
                        **_user, 'name': new_name.strip(), 'email': clean_new_email,
                    }
                    st.session_state.profile_details_editing = False
                    st.success('Profile updated.', icon=':material/check_circle:')
                    st.rerun()
                else:
                    st.error('Could not save those changes; that email may already be in use.',
                              icon=':material/error:')


@st.fragment
def _render_change_password_card(_email):
    """The Profile page's "Change password" card, as its own fragment --
    same reasoning as _render_2fa_card and _render_profile_details_card
    above. Collapsed by default behind a single "Change password"
    button rather than always showing three open password fields.
    """
    _account = get_local_account(_email)
    _editing = st.session_state.get('change_password_editing', False)
    with card(title='Change password'):
        if not _editing:
            st.caption('Update the password used to sign in.')
            if st.button('Change password', key='change_password_start',
                          icon=':material/lock_reset:', width='stretch'):
                st.session_state.change_password_editing = True
                st.rerun(scope='fragment')
        else:
            with st.form('change_password_form', border=False):
                current_pw = st.text_input('Current password', type='password', key='cp_current')
                new_pw = st.text_input('New password', type='password', key='cp_new')
                confirm_pw = st.text_input('Confirm new password', type='password', key='cp_confirm')
                change_submitted = st.form_submit_button(
                    'Update password', icon=':material/check:', width='stretch',
                )
            if st.button('Cancel', key='change_password_cancel'):
                st.session_state.change_password_editing = False
                st.rerun(scope='fragment')
            if change_submitted:
                # Re-verify the CURRENT password server-side rather than
                # trusting the already-authenticated session alone -- this
                # is the standard "confirm you're still you" step before a
                # sensitive change, and it doubles as proof the account
                # (and the DB) is actually reachable right now.
                if not _account or not verify_password(current_pw, _account['password_hash']):
                    st.error('Current password is incorrect.', icon=':material/error:')
                elif len(new_pw) < 8:
                    st.error('New password must be at least 8 characters.', icon=':material/error:')
                elif new_pw != confirm_pw:
                    st.error("New passwords don't match.", icon=':material/error:')
                elif update_local_account_password(_email, new_pw):
                    st.session_state.change_password_editing = False
                    st.success('Password updated.', icon=':material/check_circle:')
                    st.rerun(scope='fragment')
                else:
                    st.error('Could not update your password right now. Try again shortly.',
                              icon=':material/error:')


init_session_state()

# layout is now a fixed 'wide', not conditional on authenticated the way
# it used to be -- switching Streamlit's OWN layout mode between runs
# (centered while signed out, wide once signed in) turned out to be
# exactly what caused the login page's own elements to visibly "expand"
# for a moment during the transition into the dashboard: confirmed live,
# the login card's actual narrowing/centering was ALREADY handled purely
# by CSS (.block-container max-width, see auth.py's _render_auth_shell()),
# so the layout= switch was redundant with that AND the sole cause of an
# extra, avoidable flash on top of it. A single fixed layout removes the
# mode-switch entirely; the CSS-only narrowing still works exactly the
# same regardless of which layout mode it's overriding.
st.set_page_config(
    page_title='Donor Forecasting | Stroke Foundation',
    page_icon=str(TITLE_LOGO_PATH) if TITLE_LOGO_PATH.exists() else '🫀',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Checked (and, if set, handled -- rendering a loading screen and
# stopping this run) before handle_google_redirect() or render_login()
# get any chance to render dashboard-adjacent content -- see
# complete_sign_out()'s own docstring in auth.py for why this needs to
# run this early.
complete_sign_out()

handle_google_redirect()

if not st.session_state.authenticated:
    render_login()

# Auto-load the most recent run once per session -- guarded by pipeline_run
# so this only ever runs on a session's first script pass, not on every
# rerun (a widget click reruns the whole script; we don't want a DB
# round-trip on every single interaction). Run History lets you switch to
# an older run later; this is just what a fresh session opens to.
#
# render_startup_progress() wraps this specifically -- confirmed live
# against the real database, this step takes ~3.8s on the first
# post-login run of a freshly-started app (~2.7s of that is just the
# TCP/TLS/auth handshake establishing the connection, before either
# query below even runs), and previously nothing on screen indicated why
# -- the "Signing in" transition screen just sat there. That connection
# latency itself isn't something app code can reduce (see the sticky
# bar's own docstring in ui.py for the full investigation). auth.py's
# render_login() already shows this same bar the instant the login form
# is submitted (its own first DB call pays this same connection cost);
# a fresh placeholder here just continues that same bar into this run
# rather than leaving a gap, since a placeholder object doesn't survive
# across the st.rerun() between the two.
if not st.session_state.pipeline_run and db_configured():
    _startup_bar = st.empty()
    render_startup_progress(_startup_bar)
    _runs = list_dashboard_runs()
    if not _runs.empty:
        _latest_id = _runs.iloc[0]['run_id']
        _cached_state, _cached_at = load_dashboard_run(_latest_id)
        if _cached_state:
            restore_dashboard_state(_cached_state)
            st.session_state.viewing_run_id  = _latest_id
            st.session_state.data_source     = 'cached'
            st.session_state.data_loaded_at  = _cached_at
    render_startup_progress(_startup_bar, done=True)

# The "Signing in" state (set in auth.py's render_login(), survives the
# st.rerun() into this run) used to end HERE, unconditionally, right
# after the block above -- but that's before this run has rendered any
# of the actual page content (KPIs, charts) for whichever page the user
# lands on, which still has to happen below. Clearing the flag that
# early meant nothing covered THAT render -- reported live as the
# Overview page appearing blank for a moment, then its content
# "flashing" in incrementally rather than appearing whole at once.
# Instead, if we're still in this state, open the SAME full-screen
# overlay used everywhere else in the login->dashboard transition (via
# a placeholder held open, not cleared, for the rest of this script) --
# matched by a single clearing point at the very end of the file, right
# after the page's own body finishes, which is the first point this run
# has actually finished producing the whole page. Streamlit streams
# elements to the browser in the order they're added, so everything
# queued in between (sidebar, page content) reaches the browser -- just
# hidden under this fixed, opaque overlay -- before the "remove the overlay"
# instruction does; the user only ever sees a blank instant, then the
# complete, already-finished page, never the gap in between.
_signing_in_overlay_ph = None
if st.session_state.get('is_signing_in'):
    _signing_in_overlay_ph = st.empty()
    with _signing_in_overlay_ph.container():
        render_transition_spinner('Signing in')

page = st.session_state.page
# page_header() (called once per page, inside each page's own routing
# block below) reads these two straight from session_state to render the
# "Export CSV" button it now carries -- see its docstring in ui.py for
# why that's a session_state read rather than a param threaded through
# every one of its eight call sites. Administrator-only: these tables can
# include raw per-donor detail (e.g. the Donor Lifetime Value page's
# export), not just aggregates, same reasoning as the LTV page's own
# "download all donor predictions" button and the Data Pipeline/delete-run
# gates. Gated here, once, rather than in page_header() itself, so this
# stays the one place that decides who gets it.
_is_admin = (st.session_state.user or {}).get('role') in ('Administrator', 'Super Admin')
_export_csv = build_page_export_csv(page) if _is_admin else None
st.session_state.page_export_csv = _export_csv
st.session_state.page_export_filename = (
    f"sf_{page.lower().replace(' ', '_')}_export.csv" if _export_csv else None
)

# One-shot flag: True only for the single rerun immediately following a
# sidebar nav click or the account menu's "My profile" (both set this
# right before their own st.rerun() -- see render_sidebar() and
# page_header() in ui.py), read and reset here so it can never leak
# into a later, unrelated rerun (a widget tweak, a Data Pipeline
# upload, the pipeline's own multi-stage run all rerun the script the
# same way at the DOM level, but aren't a page navigation).
_nav_just_happened = bool(st.session_state.get('nav_loading'))
st.session_state.nav_loading = False

# Opaque full-viewport cover, held open through this entire page's
# render (cleared at the very end, alongside the sign-in overlay --
# search _nav_overlay_ph below) -- see render_nav_transition_overlay()'s
# docstring for why this replaced an earlier CSS-only blur/spinner:
# reproduced live, that approach left the previous page's stale content
# AND a just-opened account-menu popover (portal-rendered outside the
# main content area entirely) visible alongside the new page's content
# until the run finished, showing as a duplicated page header/subtitle
# sandwiching the real one. Placed before the cached-run banner below
# so the overlay also covers that during the transition -- the banner
# still reveals itself right as the transition ends, same as before.
#
# Created BEFORE inject_global_css()/render_sidebar(), not after --
# reported live as an intermittent brief flash of the OLD page (Users
# and Profile specifically, though never reliably reproducible on
# demand, which pointed at something timing-dependent rather than a
# structural per-page difference) before the loader ever appeared.
# Root-caused via a live DOM probe: render_sidebar() runs its own DB
# reads on every single render, not just when navigating to the pages
# they back -- count_new_runs()/count_pending_requests() for the Run
# History/Users nav badges -- cached, but with a short TTL specifically
# because they run on every page's sidebar; a cache miss (typically
# after the user's been reading a page for a while, letting that TTL
# lapse -- exactly the pattern of pausing to read a dashboard before
# opening the account menu or the Users page) makes that a genuine,
# variable-latency DB round trip. With the overlay created AFTER
# render_sidebar(), the script spent that whole round trip with no
# overlay in the DOM yet at all, leaving Streamlit's own native stale-
# content dimming as the only visible feedback until the overlay
# finally appeared. Moving creation ahead of both calls means the
# overlay is already covering the page before either one gets a chance
# to run long, regardless of which page's badge query happens to miss
# its cache -- not a fix specific to Users/Profile, since any page's
# navigation could have hit this depending on timing.
#
# The placeholder itself is created UNCONDITIONALLY, on every single
# run -- reported live (surviving a full browser + server restart, so
# not a stale-session artifact): opening the account-menu popover is
# its own rerun, separate from the later rerun the "My profile" button
# itself triggers, and that first rerun does NOT set nav_loading, so
# the placeholder used to not exist for it at all. That made two
# back-to-back reruns -- one without an element at this script
# position, one with -- structurally different at the exact same
# position in the element tree, which is the kind of mismatch that can
# break the frontend's reconciliation between runs instead of cleanly
# replacing one run's elements with the next's, however it manifests
# for a given browser/OS/network's timing. Keeping this element's
# EXISTENCE unconditional (only its CONTENT and the sleep/clear below
# are conditional) means every rerun has the identical structure at
# this position regardless of what triggered it.
_nav_overlay_ph = st.empty()
if _nav_just_happened:
    with _nav_overlay_ph.container():
        render_nav_transition_overlay()
# Stashed in session_state so clear_nav_overlay() (ui.py) can reach this
# SAME placeholder from an early-exit st.stop() deep inside a page body
# (a permission gate, an empty-state screen) -- those live in ui.py and
# in page bodies below, both too far from this module-level variable to
# close over it directly, and st.stop() means they can never rely on
# the normal end-of-script clearing block further down ever running.
# See clear_nav_overlay()'s own docstring for why this matters: without
# it, that clearing code simply never runs on those pages, leaving the
# overlay open indefinitely.
st.session_state['_nav_overlay_ph'] = _nav_overlay_ph
st.session_state['_nav_overlay_active'] = _nav_just_happened

# Runs AFTER the overlay above is already up, not before -- see that
# block's own comment for why (render_sidebar()'s badge queries can be a
# real, variable-latency DB round trip on a cache miss, and the overlay
# needs to already be covering the page before that can happen, not
# after). BG/LINE/TEAL (used by render_nav_transition_overlay() above,
# via ui.py's module-level palette constants) still hold whatever
# _apply_palette() set them to on the PREVIOUS rerun at this point --
# fine here since a dark-mode toggle triggers its own plain st.rerun(),
# never a nav-triggered one, so _nav_just_happened is never True on the
# one rerun where those constants would actually be changing.
inject_global_css()
render_sidebar()
# The fixed white topbar -- rendered once, for every authenticated page,
# right after the sidebar. Holds the sidebar-collapse hamburger, the logo
# + current page name, the notification bell, and the account popover
# (My profile / dark mode / Log out), which used to live per-page in
# page_header()'s right column. `page` is already resolved above, so the
# popover's per-page key suffix works from here.
render_topbar()

# CACHED_RUN_BANNER_PAGES, not every page -- reported live, this used to
# show (and could auto-reveal, via the nav_loading flag "My profile" now
# also sets) on Data Pipeline, Run History, Users, and Profile too, none
# of which have anything to do with "which pipeline run's data you're
# viewing" -- and on Profile specifically, whose page_header() layout
# isn't built to accommodate a fixed top strip the way the dashboard
# pages' is, that read as the banner "not filling" properly.
if st.session_state.data_source == 'cached' and page in CACHED_RUN_BANNER_PAGES:
    loaded_str = (pd.to_datetime(st.session_state.data_loaded_at).strftime('%d %b %Y, %H:%M')
                  if st.session_state.data_loaded_at else 'a previous run')
    render_cached_run_banner(loaded_str, nav_triggered=_nav_just_happened)

# Rendered globally (not just on the Data Pipeline page) since a completed
# run now redirects straight to Overview -- the success message needs to
# still be visible on whichever page the user lands on.
if st.session_state.pipeline_success_message and not st.session_state.pipeline_running:
    st.success(st.session_state.pipeline_success_message, icon=':material/check_circle:')
    st.session_state.pipeline_success_message = None

# Same pattern as the success banner above, for the failure path the
# pipeline_running block's own try/except/finally sets -- rendered
# globally rather than only on Data Pipeline for the same reason: a
# failed run stays on Data Pipeline today (unlike a successful one,
# which redirects to Overview), but showing it here too means it
# survives if the user navigates away before reading it.
if st.session_state.pipeline_error_message and not st.session_state.pipeline_running:
    st.error(st.session_state.pipeline_error_message, icon=':material/error:')
    st.session_state.pipeline_error_message = None

# Shared right-aligned page-header metadata (Base44-style "SESSION ..." /
# "RUN-..." label) -- every dashboard page shows which run it's reading
# from, live or reloaded from history.
page_meta = f'RUN {st.session_state.viewing_run_id}' if st.session_state.viewing_run_id else 'LIVE SESSION'


# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
if page == 'Data Pipeline':
    page_header('Data pipeline', 'Upload Salesforce exports',
                'Upload all four CSV files below. The pipeline runs automatically once all '
                'four are received and validated.',
                meta=f'SESSION {datetime.now().strftime("%d %b %Y").upper()}')

    # Administrator-only -- running the pipeline mutates the shared
    # dashboard state every viewer sees next, so this is gated the same
    # way delete-run and Export CSV are. The sidebar already hides this
    # page's nav entry for non-admins (see render_sidebar()); this is the
    # defense-in-depth backstop in case session_state.page ever ends up
    # 'Data Pipeline' some other way. page_header() still has to render
    # first, same reason as every other page's empty_state() guard -- see
    # its own docstring. Administrator-or-above -- unlike Users, running
    # the pipeline isn't account-management, an ordinary Administrator
    # keeps this.
    if (st.session_state.user or {}).get('role') not in ('Administrator', 'Super Admin'):
        with card():
            st.markdown(f"""
            <div style="text-align:center;padding:28px;">
                <div style="font-size:14px;font-weight:700;color:{ui.TEXT};margin-bottom:6px;">
                    Administrators only
                </div>
                <div style="font-size:12.5px;color:{ui.MIST};max-width:480px;margin:0 auto;">
                    Running the pipeline updates the shared dashboard data everyone sees, so it's
                    restricted to Administrators. Contact your admin if you need a fresh run.
                </div>
            </div>
            """, unsafe_allow_html=True)
        clear_nav_overlay()
        st.stop()

    if st.session_state.data_source == 'cached' and db_configured():
        st.caption(':material/bolt: Dashboards are currently showing a saved run, loaded instantly from '
                   'history. Uploading new files below adds a new run without affecting past ones. '
                   'See the **Run History** page to browse or reload any previous run.')

    ICON_CARD  = '<rect x="1" y="4" width="22" height="16" rx="2"/><line x1="1" y1="10" x2="23" y2="10"/>'
    ICON_LOOP  = '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>'
    ICON_USERS = '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>'
    ICON_TARGET = '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'

    REQUIRED_PAYMENTS  = ['Recurring Payment ID', 'Schedule Date', 'Created Date', 'Success', 'Amount']
    REQUIRED_RECURRING = ['Recurring Payment ID', 'Contact ID', 'Campaign', 'Donation Amount', 'Recruitment Date']
    REQUIRED_CONTACTS  = ['Contact ID']
    REQUIRED_CAMPAIGNS = ['Campaign', 'Campaign Type']

    running = st.session_state.pipeline_running

    # Everything above the "Forecasting pipeline"/"Execution log" cards --
    # the flow diagram, the 4 upload slots, and the Start button (moved
    # here, below the uploads, rather than living in the pipeline card's
    # own header -- per explicit request, "start" reads more naturally
    # as the next action after choosing files than as a title-bar
    # control) -- lives in ONE st.empty() placeholder specifically so it
    # can be cleared in one call once the run starts (see upload_area.
    # empty() right below this block): a run can take several minutes,
    # and the upload widgets/flow diagram are irrelevant once it's
    # actually in progress, per explicit request ("only the bottom 2
    # elements... should be visible"). Streamlit still needs these
    # upload_slot() calls to actually EXECUTE every run (that's the only
    # way to get f_pay/f_rec/f_con/f_cam's current value at all, running
    # or not) -- clearing the placeholder right after removes their
    # rendered output from the page without preventing that.
    #
    # key='pipeline_upload_area' -- confirmed live, this container (even
    # with border left at its default) was rendering as ONE bordered
    # card wrapping the flow diagram, uploads, AND the button together,
    # reported as "these are supposed to be individual elements... they
    # all are under 1 box". Same root cause already documented elsewhere
    # in ui.py (see .st-key-page_header_block/.st-key-page_header_row):
    # any st.container(), keyed or not, carries the same overflow=
    # "visible" attribute a real border=True container does, which is
    # what inject_global_css()'s global card-style rule actually keys
    # off. A stable key gives this one container a specific CSS class
    # (st-key-pipeline_upload_area) so it can be un-styled the same way
    # those two already are, instead of looking like an accidental card.
    upload_area = st.empty()
    with upload_area.container(key='pipeline_upload_area'):
        st.markdown(f"""
        <div class="sf-pipeline">
            <div class="sf-pipe-step">
                <div class="sf-pipe-icon p-teal">
                    <svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                </div>
                <div><div class="sf-pipe-label">Upload</div><div class="sf-pipe-desc">4 CSV files</div></div>
            </div>
            <div class="sf-pipe-arrow">›</div>
            <div class="sf-pipe-step">
                <div class="sf-pipe-icon p-navy">
                    <svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                </div>
                <div><div class="sf-pipe-label">Clean</div><div class="sf-pipe-desc">Standardise</div></div>
            </div>
            <div class="sf-pipe-arrow">›</div>
            <div class="sf-pipe-step">
                <div class="sf-pipe-icon p-navy">
                    <svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7"/>
                    <rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>
                    <rect x="3" y="14" width="7" height="7"/></svg>
                </div>
                <div><div class="sf-pipe-label">Build master</div><div class="sf-pipe-desc">Join all 4</div></div>
            </div>
            <div class="sf-pipe-arrow">›</div>
            <div class="sf-pipe-step">
                <div class="sf-pipe-icon p-purple">
                    <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5z"/>
                    <path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
                </div>
                <div><div class="sf-pipe-label">Forecast</div><div class="sf-pipe-desc">Run model</div></div>
            </div>
            <div class="sf-pipe-arrow">›</div>
            <div class="sf-pipe-step">
                <div class="sf-pipe-icon p-amber">
                    <svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7"/>
                    <rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>
                    <rect x="3" y="14" width="7" height="7"/></svg>
                </div>
                <div><div class="sf-pipe-label">Dashboard</div><div class="sf-pipe-desc">Views update</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Called here, once, rather than from inject_global_css() (which
        # runs on every page) -- this is the only page with any
        # st.file_uploader() widgets, so this is the only page that needs
        # it. See its own docstring in ui.py for why this used to run
        # globally and what that cost every other page.
        render_upload_progress_tracker()

        st.markdown('<div class="sf-eyebrow">Source files</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            f_pay, pay_valid = upload_slot('Payments.csv', 'Every card charge attempt · up to 1024 MB',
                                            ICON_CARD, 'p-teal', 'up_p', REQUIRED_PAYMENTS, disabled=running)
            f_rec, rec_valid = upload_slot('Recurring Payments.csv', 'Every donor regular-giving signup',
                                            ICON_LOOP, 'p-navy', 'up_r', REQUIRED_RECURRING, disabled=running)
        with col2:
            f_con, con_valid = upload_slot('Contacts.csv', 'One row per unique donor',
                                            ICON_USERS, 'p-purple', 'up_c', REQUIRED_CONTACTS, disabled=running)
            f_cam, cam_valid = upload_slot('Campaigns.csv', 'One row per recruitment campaign · includes CPA',
                                            ICON_TARGET, 'p-amber', 'up_ca', REQUIRED_CAMPAIGNS, disabled=running)

        all_up    = all([f_pay, f_rec, f_con, f_cam])
        all_valid = all([pay_valid, rec_valid, con_valid, cam_valid])

        if all_up and not all_valid:
            st.error('One or more files don\'t match what\'s expected for their slot. Fix the file(s) flagged '
                      'above before running the pipeline.', icon=':material/error:')

        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        start_clicked = st.button(
            'Start pipeline', type='primary', icon=':material/play_arrow:',
            width='stretch', disabled=running or not all_up or not all_valid,
        )

    # 6 stages -- sBG/BG-NBD and the gift-waterfall bridge were dropped
    # along with their Income Forecast options above, since nothing else
    # reads their output. Donor LTV (Pareto/NBD + Gamma-Gamma) was also
    # pulled out of this run: it was the slowest stage (~5 min) and only
    # the separate Donor Lifetime Value page reads its output, so it's
    # now fit lazily on that page's first visit instead of blocking
    # every pipeline run (see cached_ltv_fit() near the top of this file).
    N_STAGES = 6
    STAGE_DEFS = [
        (1, 'Ingestion & Schema Validation', 'ETL'),
        (2, 'Master File Build',             'ETL'),
        (3, 'Linear Trend Baseline',          'Trend'),
        (4, 'ML Forecast Model',              'Gradient Boosting'),
        (5, 'Stock-Flow Model',               'SARIMA + Cohort Survival'),
        (6, 'Dashboard Publish',              'Export'),
    ]

    # "Forecasting pipeline" (progress) and "Execution log" -- hidden
    # entirely until a run is actually in progress, per explicit request
    # ("should only be visible after the run pipeline button is clicked
    # and a pipeline is running"), rather than always shown with a
    # resting "0%"/"awaiting Start pipeline" state as before. Once
    # running is True, upload_area.empty() above has already cleared
    # the upload section, so these two cards are the ONLY thing left on
    # the page -- exactly the "only the bottom 2 elements" behaviour
    # asked for, from both directions now (upload section hides once
    # running; these two cards stay hidden until it starts).
    if running:
        upload_area.empty()

        with st.container(border=True):
            st.markdown('<div class="sf-card-title">Forecasting pipeline</div>', unsafe_allow_html=True)
            st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)
            progress_slot = st.empty()
            overall_progress(progress_slot, 0, N_STAGES)
            # Placeholders only -- deliberately NOT pre-rendered as
            # 'pending'. Each stage's row only appears once its own turn
            # actually comes (see the running block below), so the card
            # shows stages arriving one at a time as the run progresses,
            # not a full skeleton upfront.
            stages = {n: st.empty() for n, _, _ in STAGE_DEFS}

        st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
        with card('Execution log', 'Real-time, timestamped log of this run'):
            log_slot = st.empty()
            log = new_execution_log(log_slot)

    if start_clicked:
        st.session_state.pipeline_running = True
        st.rerun()

    if st.session_state.pipeline_running:
        from pipeline.build_master import build_master

        run_started = datetime.now()
        tmp = tempfile.mkdtemp()

        def save(f, name):
            path = os.path.join(tmp, name)
            with open(path, 'wb') as out:
                out.write(f.getbuffer())
            return path

        try:
            stage_row(stages[1], 1, 'Ingestion & Schema Validation', 'ETL', 'running')
            log(':material/check_circle: Files received')
            p_path  = save(f_pay, 'Payments.csv')
            r_path  = save(f_rec, 'Recurring Payments.csv')
            c_path  = save(f_con, 'Contacts.csv')
            ca_path = save(f_cam, 'Campaigns.csv')
            stage_row(stages[1], 1, 'Ingestion & Schema Validation', 'ETL', 'done', '4/4 files')
            overall_progress(progress_slot, 1, N_STAGES)

            stage_row(stages[2], 2, 'Master File Build', 'ETL', 'running')

            # Real backend stdout capture -- clean.py/build_master.py already
            # print(..., flush=True) genuine diagnostic lines ([diag] chunk
            # counts, row totals) as they work. Redirecting stdout into a
            # buffer and flushing it into the Execution Log on every progress
            # tick surfaces the actual terminal output, not a hand-written
            # narration of it -- this is where nearly all of that output
            # happens, so it's the highest-value place to capture it.
            stdout_buf = io.StringIO()

            def flush_stdout():
                captured = stdout_buf.getvalue()
                if captured.strip():
                    log(captured.rstrip('\n'))
                stdout_buf.truncate(0)
                stdout_buf.seek(0)

            def on_progress(rows):
                stage_row(stages[2], 2, 'Master File Build', 'ETL', 'running', f'{rows:,} rows processed')
                flush_stdout()

            with contextlib.redirect_stdout(stdout_buf):
                master = build_master(p_path, r_path, ca_path, c_path, progress_callback=on_progress)
            flush_stdout()
            log(f':material/check_circle: Master file built, {len(master):,} donor-month rows')
            stage_row(stages[2], 2, 'Master File Build', 'ETL', 'done', f'{len(master):,} rows')
            overall_progress(progress_slot, 2, N_STAGES)

            run_pipeline_models(master, stages=stages, progress=(progress_slot, 2), log=log)

            stage_row(stages[6], 6, 'Dashboard Publish', 'Export', 'running')
            if db_configured():
                dashboard_state = build_dashboard_state()
                summary = {
                    'run_by': (st.session_state.user or {}).get('email'),
                    'donor_count': st.session_state.donor_count,
                    'total_income': st.session_state.total_income,
                    'mape_ml': st.session_state.mape, 'mape_linear': st.session_state.mape_linear,
                    'mape_ltv': (st.session_state.ltv_metrics['income_holdout_mape']
                                 if st.session_state.ltv_metrics else None),
                    'mape_stockflow': st.session_state.mape_stockflow,
                    'mape_sbg': st.session_state.mape_sbg, 'mape_bgnbd': st.session_state.mape_bgnbd,
                    'mape_gw': st.session_state.mape_gw,
                    'duration_seconds': (datetime.now() - run_started).total_seconds(),
                }
                saved_run_id = save_dashboard_run(dashboard_state, summary)
                if saved_run_id:
                    stored_kb = len(gzip.compress(json.dumps(dashboard_state).encode(), compresslevel=9)) / 1024
                    log(f':material/check_circle: Run saved to history, {saved_run_id} '
                        f'({stored_kb:.0f} KB), reload it anytime from Run History, no re-upload needed')
                    stage_row(stages[6], 6, 'Dashboard Publish', 'Export', 'done', f'{stored_kb:.0f} KB saved')
                    st.session_state.viewing_run_id = saved_run_id
                    st.session_state.data_source     = 'live'
                    st.session_state.data_loaded_at  = datetime.now()
                else:
                    log(f':material/warning: Run history save skipped, '
                        f'{st.session_state.get("db_error", "unknown error")}')
                    stage_row(stages[6], 6, 'Dashboard Publish', 'Export', 'warning', 'History save skipped')
            else:
                log(':material/check_circle: Dashboard views published (no history backend configured)')
                stage_row(stages[6], 6, 'Dashboard Publish', 'Export', 'done', 'No history backend')
            overall_progress(progress_slot, N_STAGES, N_STAGES)
            log('Pipeline complete, dashboards refreshed')

            st.session_state.pipeline_success_message = (
                f'Pipeline complete, {len(master):,} rows · '
                f'{master["recurring_payment_id"].nunique():,} donor signups · '
                f'ML forecast MAPE {st.session_state.mape:.1f}% (linear baseline {st.session_state.mape_linear:.1f}%)'
            )
            st.session_state.pipeline_running = False
            st.session_state.page = 'Overview'
            st.rerun()
        except Exception as exc:
            # Outer safety net for file ingestion, build_master(), and the
            # dashboard-publish step -- none of those had one before (the
            # ML/stock-flow model stages already degrade gracefully via
            # their own try/except inside run_pipeline_models(), see
            # there). Confirmed live in a security review: without this,
            # an uncaught exception here (e.g. a malformed CSV) showed
            # Streamlit's raw default traceback in the browser -- internal
            # file paths, library internals -- to whichever Administrator
            # triggered it, and left pipeline_running stuck True
            # afterward, disabling navigation with no way back in short of
            # signing out and back in.
            log(f':material/error: Pipeline failed: {exc}')
            log(traceback.format_exc())
            st.session_state.pipeline_error_message = (
                f'The pipeline run failed: {exc}. See the Execution Log above for '
                f'details, or try again; this usually means one of the uploaded '
                f'files has an unexpected format.'
            )
            st.rerun()
        finally:
            # Unconditional -- runs whether the try block succeeded (where
            # it's a harmless no-op re-set, pipeline_running is already
            # False by then) or raised, so a crash can never leave the run
            # stuck in a state that disables navigation with no way out.
            st.session_state.pipeline_running = False

    if st.session_state.pipeline_run:
        st.markdown('<div style="height:2px;"></div>', unsafe_allow_html=True)
        mape_delta = None
        if st.session_state.mape_linear:
            mape_delta = f'{st.session_state.mape - st.session_state.mape_linear:+.1f}pp vs linear'
        with st.container(horizontal=True):
            kpi('Rows in master file', f'{st.session_state.master_rows:,}', icon=':material/table_rows:', accent=ui.TEAL)
            kpi('Donor signups', f'{st.session_state.donor_count:,}', icon=':material/how_to_reg:', accent=ui.BLUE)
            kpi('Total income reconciled', f'${st.session_state.total_income:,.0f}', icon=':material/payments:', accent=ui.PURPLE)
            kpi('ML forecast accuracy', f'{st.session_state.mape:.1f}% MAPE', delta=mape_delta,
                delta_color='inverse', icon=':material/verified:', accent=ui.AMBER)


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Overview':
    # page_header() (and the account cluster it now carries -- see its
    # docstring) has to render before the empty_state() guard below can
    # st.stop() the script, so the subtitle here has to tolerate the
    # pre-pipeline-run state where these session_state fields are still
    # None, not just the populated case the f-string below assumed when
    # this always ran after the guard.
    overview_sub = (
        f'{st.session_state.donor_count:,} donor signups · '
        f'{st.session_state.campaign_type_count} campaign types · '
        f'{st.session_state.supplier_count} suppliers'
    ) if st.session_state.pipeline_run else ''
    page_header('Dashboard', 'Overview', overview_sub, meta=page_meta, info=(
        'The 12-month forecast KPI and the chart below are both a BLEND of every available '
        'total-income method (ML, linear trend, stock-flow); the dashed line is the average '
        'across methods, and the shaded band is how far those methods disagree, not a statistical '
        'confidence interval. Donor Lifetime Value answers a different question (what today\'s '
        'existing donors are worth, assuming no further recruitment) and is deliberately excluded '
        'from this blend; see that page for the LTV view.'
    ))

    if not st.session_state.pipeline_run:
        empty_state()

    forecast_df = st.session_state.forecast_df
    monthly     = st.session_state.monthly
    mape        = st.session_state.mape

    current_active = monthly.iloc[-1]['active_donors']
    prev_active    = monthly.iloc[-2]['active_donors']
    current_income = monthly.iloc[-1]['total_income']
    prev_income    = monthly.iloc[-2]['total_income']
    total_hist   = st.session_state.total_income

    # ── Blended forecast across every independently-validated total-income
    # method (ML, Linear, Stock-flow). Donor rollup is deliberately excluded
    # here: it only estimates income from today's existing donors (assumes
    # zero future recruitment) -- a narrower, different question -- so
    # blending it in would drag the average down in a way that misrepresents
    # total org income. See the Income Forecast page for that model on its
    # own terms. ──
    horizon = 24
    with st.spinner('Blending forecast methods…'):
        linear_df, mape_lin, _, _ = cached_linear_forecast(monthly, n_train=24, n_forecast=horizon)
    methods = {
        'ML forecast': (forecast_df.head(horizon)['predicted_income'].reset_index(drop=True), mape),
        'Linear trend': (linear_df.head(horizon)['predicted_income'].reset_index(drop=True), mape_lin),
    }
    sf_zone12_state = st.session_state.sf_zone12
    sf_zone3_state  = st.session_state.sf_zone3
    sf_walkforward_state = st.session_state.sf_walkforward
    if sf_zone12_state is not None:
        sf_series = sf_zone12_state['predicted_income'].reset_index(drop=True)
        if horizon > len(sf_zone12_state) and sf_zone3_state and 'Base' in sf_zone3_state:
            extra = sf_zone3_state['Base'].head(horizon - len(sf_zone12_state))['predicted_income'].reset_index(drop=True)
            sf_series = pd.concat([sf_series, extra], ignore_index=True)
        methods['Stock-flow model'] = (sf_series.head(horizon), st.session_state.mape_stockflow)

    blended = pd.DataFrame({name: series.values for name, (series, _) in methods.items()})
    blended_avg = blended.mean(axis=1)
    blended_min = blended.min(axis=1)
    blended_max = blended.max(axis=1)
    blended_slope = float(np.polyfit(range(1, horizon + 1), blended_avg.values, 1)[0])

    method_totals_12 = {name: float(series.head(12).sum()) for name, (series, _) in methods.items()}
    total_12_avg = sum(method_totals_12.values()) / len(method_totals_12)
    total_12_min = min(method_totals_12.values())
    total_12_max = max(method_totals_12.values())

    method_mapes = [m for _, m in methods.values() if m is not None]
    mape_avg = sum(method_mapes) / len(method_mapes)

    income_delta = (current_income - prev_income) / prev_income * 100 if prev_income else 0
    active_delta = int(current_active - prev_active)

    with st.container(horizontal=True):
        kpi('12-month forecast', f'${total_12_avg:,.0f}',
            delta=f'range \\${total_12_min:,.0f} to \\${total_12_max:,.0f} across {len(methods)} methods',
            icon=':material/trending_up:', accent=ui.TEAL, delta_color='off')
        kpi('Active donors', f'{current_active:,.0f}', delta=f'{active_delta:+,} vs prior month',
            icon=':material/group:', accent=ui.BLUE)
        kpi('Current monthly income', f'${current_income:,.0f}', delta=f'{income_delta:+.1f}% vs prior month',
            icon=':material/payments:', accent=ui.PURPLE)
        kpi('Forecast accuracy', f'{mape_avg:.1f}% MAPE', delta=f'avg across {len(methods)} methods',
            icon=':material/verified:', accent=ui.AMBER, delta_color='off')

    recent = monthly.tail(24)
    t_act  = list(range(1, len(recent) + 1))
    t_fore = list(range(len(recent) + 1, len(recent) + horizon + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_fore + t_fore[::-1],
        y=list(blended_max.values) + list(blended_min.values)[::-1],
        fill='toself', fillcolor='rgba(0,137,123,0.10)',
        line=dict(color='rgba(0,0,0,0)'), name=f'Spread across {len(methods)} methods', hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=t_act, y=recent['total_income'].values,
        mode='lines+markers', name='Actual',
        line=dict(color=ui.TEXT, width=2.5), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=t_fore, y=blended_avg.values,
        mode='lines+markers', name='Blended forecast',
        line=dict(color=ui.TEAL, width=2.5, dash='dash'),
        marker=dict(size=5, symbol='square'),
    ))
    fig.add_vline(x=len(recent) + 0.5, line_dash='dot', line_color=ui.LINE, opacity=0.8)
    fig.update_layout(yaxis=dict(tickformat='$,.0f'))

    with card('Monthly income: actual vs blended forecast',
              f'Last 24 months actual · average of {", ".join(methods.keys())} · '
              f'shaded band shows where the methods disagree',
              tag=f'{len(methods)} methods blended', tag_color='green', info=(
                  'Solid line (months to the left of the dotted vertical line) is real historical '
                  'income. Dashed line (to the right) is the forecast: the average across every '
                  'available method. The shaded teal band is NOT a confidence interval; it is the '
                  'spread between the methods\' own predictions, so a wide band means the methods '
                  'disagree more about that month, not that any one of them is less certain.'
              )):
        chart(fig, 280)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        sup = st.session_state.supplier_summary
        fig2 = go.Figure(go.Pie(
            labels=sup['supplier'], values=sup['total_income'],
            hole=0.55, marker_colors=ui.COLOURS, textinfo='percent', textfont_size=10,
        ))
        fig2.update_layout(showlegend=True,
                            legend=dict(orientation='v', x=1, y=0.5, font=dict(size=9)),
                            margin=dict(l=0, r=0, t=0, b=0))
        with card('Income by supplier', info=(
            'Each slice is one supplier\'s share of total historical income, not a forecast. '
            'Hover a slice for its exact dollar total; see Supplier Insights for retention and '
            'lifetime-value comparisons, not just income share.'
        )):
            chart(fig2, 240)

    with col2:
        camp = st.session_state.campaign_summary
        fig3 = go.Figure(go.Pie(
            labels=camp['campaign_type'], values=camp['total_income'],
            hole=0.55, marker_colors=ui.COLOURS, textinfo='percent', textfont_size=10,
        ))
        fig3.update_layout(showlegend=True,
                            legend=dict(orientation='v', x=1, y=0.5, font=dict(size=9)),
                            margin=dict(l=0, r=0, t=0, b=0))
        with card('Income by campaign type', info=(
            'Each slice is one campaign type\'s share of total historical income, not a forecast. '
            'See Campaign ROI for cost-per-acquisition and retention by campaign type, not just '
            'income share.'
        )):
            chart(fig3, 240)

    with col3:
        summary = pd.DataFrame({
            'Metric': ['Total rows', 'Unique signups', 'Unique donors', 'Suppliers',
                       'Campaign types', 'Total income', 'Forecast MAPE'],
            'Value': [
                f'{st.session_state.master_rows:,}',
                f'{st.session_state.donor_count:,}',
                f'{st.session_state.contact_count:,}',
                f'{st.session_state.supplier_count}',
                f'{st.session_state.campaign_type_count}',
                f'${total_hist:,.0f}',
                f'{mape:.1f}%',
            ],
        })
        with card('Dataset summary', info=(
            'Snapshot of the currently loaded dataset; raw payment rows, unique signups/donors, '
            'and the ML forecast\'s own validation MAPE. Not a forecast itself, just what this run '
            'was built from.'
        )):
            st.dataframe(summary, hide_index=True, width='stretch', height=282)

    if db_configured():
        _recent_runs = list_dashboard_runs().head(3)
        if not _recent_runs.empty:
            _recent_runs = _recent_runs.copy()
            _recent_runs['run_at'] = pd.to_datetime(_recent_runs['run_at'])
            with card('Latest pipeline runs', 'Most recent completed runs, see Run History for the full list',
                      tag='Completed', tag_color='green', info=(
                          'Each row is one completed pipeline run, most recent first. Click Run History '
                          'in the sidebar to reload any of these (or an older one) and view every '
                          'dashboard exactly as it looked at that point, without re-uploading data.'
                      )):
                st.dataframe(
                    _recent_runs[['run_id', 'run_at', 'run_by', 'donor_count', 'total_income']].rename(columns={
                        'run_id': 'Run ID', 'run_at': 'Completed', 'run_by': 'Run by',
                        'donor_count': 'Donors', 'total_income': 'Total income',
                    }),
                    hide_index=True, width='stretch',
                    column_config={
                        'Completed': st.column_config.DatetimeColumn(format='D MMM, HH:mm'),
                        'Donors': st.column_config.NumberColumn(format='%d'),
                        'Total income': st.column_config.NumberColumn(format='dollar'),
                    },
                )


# ══════════════════════════════════════════════════════════════════════════════
# INCOME FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Income Forecast':
    page_header('Forecasting', 'Monthly income forecast',
                'Three independent methods, validated the same way, forecasting the same thing: '
                'a top-down trend model, a machine-learning model, and a stock-flow model that '
                'forecasts recruitment, retention, and gift size separately.',
                meta=page_meta, info=(
                    'These three methods are shown for COMPARISON, not blended into one number here '
                    '(see Overview for the blended view). Linear trend is a simple baseline. ML '
                    '(gradient boosting) predicts each month directly rather than chaining one-step '
                    'predictions that compound their own error. Stock-flow never forecasts income '
                    'directly; it forecasts recruits, lapse rate and average gift separately, then '
                    'derives income from those via Income = Active donors x Average gift. The '
                    'stock-flow MAPE comes from 3 rolling walk-forward windows, the other two use one '
                    'fixed 12-month holdout, so it is not directly comparable in absolute terms.'
                ))

    if not st.session_state.pipeline_run:
        empty_state()

    monthly           = st.session_state.monthly

    sf_zone12_state      = st.session_state.sf_zone12
    sf_zone3_state       = st.session_state.sf_zone3
    sf_walkforward_state = st.session_state.sf_walkforward
    sf_assumptions_state = st.session_state.sf_assumptions

    # Model/horizon pickers rendered here, BEFORE the comparison-table fits
    # below, not after them as before -- these two widgets don't depend on
    # the comparison table at all, but used to sit textually after it, so
    # on a genuine cache miss (a freshly-loaded run whose `monthly` hasn't
    # been fit yet) they'd sit blocked behind a real gradient-boosting
    # refit before ever becoming interactive. Streamlit renders top to
    # bottom in code order -- reordering the code is what actually moves
    # them earlier, a st.container() placeholder alone wouldn't (the slow
    # fit call below still has to finish before ANY later line runs,
    # container or not; only genuinely relocating independent code ahead
    # of slow work gets it painted sooner). The model-specific control
    # that used to share this row (training window for Linear, etc.)
    # stays below, since it genuinely needs model_choice decided first.
    model_options = ['ML forecast', 'Linear trend', 'Stock-flow model']
    fc1, fc2 = st.columns([1.8, 1.3], vertical_alignment='bottom')
    with fc1:
        model_choice = st.segmented_control(
            'Forecast model', model_options, default='ML forecast', key='fc_model',
        )
    with fc2:
        horizon = st.segmented_control('Forecast horizon', [12, 18, 24], default=24,
                                        key='fc_h', format_func=lambda x: f'{x} months')
    model_choice = model_choice or 'ML forecast'
    horizon = horizon or 24

    # Both rows below read straight out of session_state instead of calling
    # cached_ml_forecast/cached_linear_forecast again -- forecast_df/mape
    # and forecast_df_linear/mape_linear are the EXACT result of those same
    # two calls (monthly, n_forecast=24 / n_train=24, n_forecast=24), always
    # already computed by run_pipeline_models() and sitting in session_state
    # (or restored straight from a saved run's DB blob, itself never a
    # recompute either) before this page can even render. Calling them again
    # here only ever worked because @st.cache_data usually still had that
    # exact call warm; on any process restart since (this cache is in-memory
    # only, wiped every `streamlit run`) or the first visit after a Run
    # History switch to a run this process hasn't fit before, it silently
    # became a full ~300-estimator gradient-boosting refit just to redraw a
    # table whose numbers already existed -- confirmed as a real cause of
    # this page feeling slow independent of whatever horizon is selected
    # below. Reading session_state instead makes this table's cost zero
    # regardless of cache state. ml_importances empty means the pipeline's
    # own ML fit fell back to linear (not enough monthly history) -- the
    # same condition the old inline call's `except ValueError` was catching,
    # so the ML row is skipped here exactly the same way in that case.
    comp_rows = []
    if st.session_state.ml_importances:
        comp_rows.append({
            'Method': 'ML forecast (gradient boosting)',
            '12-month total': st.session_state.forecast_df.head(12)['predicted_income'].sum(),
            '24-month total': st.session_state.forecast_df['predicted_income'].sum(),
            'Validation MAPE': st.session_state.mape,
        })
    comp_rows.append({
        'Method': 'Linear trend',
        '12-month total': st.session_state.forecast_df_linear.head(12)['predicted_income'].sum(),
        '24-month total': st.session_state.forecast_df_linear['predicted_income'].sum(),
        'Validation MAPE': st.session_state.mape_linear,
    })

    if sf_zone12_state is not None:
        sf_24m = sf_zone12_state['predicted_income'].sum()
        if sf_zone3_state and 'Base' in sf_zone3_state:
            sf_24m += sf_zone3_state['Base'].head(24 - len(sf_zone12_state))['predicted_income'].sum()
        comp_rows.append({
            'Method': 'Stock-flow model (recruits × retention × gift)',
            '12-month total': sf_zone12_state.head(12)['predicted_income'].sum(),
            '24-month total': sf_24m,
            'Validation MAPE': st.session_state.mape_stockflow,
        })

    with card('Method comparison',
              'Trained on 2019 onward · error measured in $, not just a vibe check. The stock-flow '
              'model is validated on 3 rolling 12-month walk-forward windows, while the others use one '
              'fixed 12-month holdout, so its MAPE is not directly comparable, just directionally so.',
              tag=f'{len(comp_rows)} methods', tag_color='blue', info=(
                  'One row per method, each trained and validated independently; not a leaderboard '
                  'to just pick the lowest MAPE from. Compare the 12/24-month totals across rows to '
                  'see how much the methods actually agree on, and remember the stock-flow row\'s '
                  'MAPE is measured differently (3 rolling windows vs one fixed holdout for the '
                  'others), so it is only directionally comparable.'
              )):
        st.dataframe(
            pd.DataFrame(comp_rows), hide_index=True, width='stretch',
            column_config={
                '12-month total': st.column_config.NumberColumn(format='dollar'),
                '24-month total': st.column_config.NumberColumn(format='dollar'),
                'Validation MAPE': st.column_config.NumberColumn(format='%.1f%%'),
            },
        )

    # model_choice/horizon themselves were already picked further up, before
    # the comparison-table fits -- fc3 is just the leftover per-model
    # contextual slot (a caption, or the Linear model's training-window
    # control) that genuinely does need model_choice decided first, so it
    # stays here rather than moving up with the other two.
    fc3 = st.container()
    use_ml = model_choice == 'ML forecast'

    importances = {}
    if model_choice == 'ML forecast':
        with fc3:
            st.caption('Trained on 2019 onward · direct multi-step (no recursive chaining) · '
                       '12-month holdout, not a separate training window.')
        try:
            with st.spinner('Fitting the ML forecast…'):
                forecast_df, mape, importances, slope = cached_ml_forecast(monthly, n_forecast=horizon)
            model_desc = ('Gradient-boosted trend + residual model · direct multi-step, trained on 2019+ · '
                          'validated on a 12-month holdout')
            holdback_label = '12-month holdout'
        except ValueError as exc:
            st.warning(str(exc))
            forecast_df, mape, slope, _ = cached_linear_forecast(monthly, n_train=24, n_forecast=horizon)
            model_desc = 'Linear trend (fallback, not enough history for the ML model)'
            holdback_label = '12-month holdout'
            use_ml = False
        n_train = min(36, len(monthly))
    elif model_choice == 'Linear trend':
        with fc3:
            n_train = st.segmented_control('Training window', [18, 24, 36, 48], default=24,
                                            key='fc_t', format_func=lambda x: f'{x}-month window')
        n_train = n_train or 24
        forecast_df, mape, slope, _ = cached_linear_forecast(monthly, n_train=n_train, n_forecast=horizon)
        model_desc = f'Linear trend · last {n_train} months (from 2019 onward) · validated on a 12-month holdout'
        holdback_label = '12-month holdout'
    elif model_choice == 'Stock-flow model':
        with fc3:
            st.caption('Forecasts recruits (SARIMA), lapse rate (cohort survival), and gift size (trend) '
                       'separately, then derives income through the accounting identity, never forecasts '
                       'income directly. Statistical range is months 1 to 18. Months 19 to 36 are shown as named '
                       'scenarios further down this page, not a point forecast.')
        n_train = min(36, len(monthly))
        holdback_label = '3-window walk-forward avg'
        if sf_zone12_state is None:
            st.warning(st.session_state.sf_error or 'The stock-flow model is not available for this dataset.')
            forecast_df, mape, slope, _ = cached_linear_forecast(monthly, n_train=24, n_forecast=horizon)
            model_desc = 'Linear trend (fallback, stock-flow model unavailable)'
        else:
            base_df = sf_zone12_state[['calendar_month', 'predicted_income', 'band_low', 'band_high']].copy()
            if horizon > len(base_df) and sf_zone3_state and 'Base' in sf_zone3_state:
                need = horizon - len(base_df)
                extra = sf_zone3_state['Base'][['calendar_month', 'predicted_income']].head(need).copy()
                extra['band_low'] = extra['predicted_income'] * 0.80
                extra['band_high'] = extra['predicted_income'] * 1.20
                base_df = pd.concat([base_df, extra], ignore_index=True)
                st.caption(f'Months 19 to {horizon} above use the Base scenario\'s central assumptions to fill '
                           f'this chart. See the scenario comparison below for the full Conservative/Optimistic range.')
            forecast_df = base_df.head(horizon)
            mape = st.session_state.mape_stockflow
            slope = float(np.polyfit(forecast_df['calendar_month'], forecast_df['predicted_income'], 1)[0])
            model_desc = ('Recruits × retention × gift, derived via the accounting identity · trained from '
                          '2019 · validated on 3 rolling 12-month walk-forward windows, error in $')

    fore  = forecast_df.head(horizon)
    total = fore['predicted_income'].sum()

    with st.container(horizontal=True):
        kpi(f'{horizon}-month total', f'${total:,.0f}', delta='Projected income',
            icon=':material/summarize:', accent=ui.TEAL, delta_color='off')
        kpi('Avg monthly income', f'${fore["predicted_income"].mean():,.0f}', delta='Per month',
            icon=':material/calendar_month:', accent=ui.BLUE, delta_color='off')
        kpi('Validation MAPE', f'{mape:.1f}%', delta=holdback_label,
            icon=':material/verified:', accent=ui.PURPLE, delta_color='off')
        kpi('Underlying trend', f'${slope:+,.0f}/mo', delta='Long-run income slope',
            icon=':material/show_chart:', accent=ui.AMBER, delta_color='off')

    recent = monthly.tail(n_train)
    t_act  = list(range(1, len(recent) + 1))
    t_fore = list(range(len(recent) + 1, len(recent) + horizon + 1))
    split  = len(recent) + 0.5

    fig = go.Figure()
    fig.add_vrect(x0=split, x1=max(t_fore) + 0.5, fillcolor='rgba(0,137,123,0.05)',
                  line_width=0, layer='below')
    if 'band_low' in fore.columns:
        band_lo, band_hi = fore['band_low'].values, fore['band_high'].values
        band_name = 'Confidence band (±5% months 1 to 6, ±12% months 7 to 18)'
    else:
        band_lo, band_hi = fore['predicted_income'] * 0.95, fore['predicted_income'] * 1.05
        band_name = '±5% confidence band'
    fig.add_trace(go.Scatter(
        x=t_fore + t_fore[::-1],
        y=list(band_hi) + list(band_lo)[::-1],
        fill='toself', fillcolor='rgba(0,137,123,0.10)',
        line=dict(color='rgba(0,0,0,0)'), name=band_name, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=t_act, y=recent['total_income'].values,
        mode='lines+markers', name='Actual monthly income',
        line=dict(color=ui.TEXT, width=2.5), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=t_fore, y=fore['predicted_income'].values,
        mode='lines+markers', name=f'{horizon}-month forecast',
        line=dict(color=ui.TEAL, width=2.5, dash='dash'),
        marker=dict(size=5, symbol='square'),
    ))
    fig.add_vline(x=split, line_dash='dot', line_color=ui.MIST, opacity=0.8,
                  annotation_text='Forecast start', annotation_font_size=10,
                  annotation_font_color=ui.MIST)
    fig.update_layout(
        yaxis=dict(tickformat='$,.0f', title='Monthly income ($)'),
        xaxis=dict(title='Month number'),
    )

    with card(f'{horizon}-month income forecast', model_desc,
              tag=f'MAPE {mape:.1f}%', tag_color='green', info=(
                  'Solid line is real historical income (the training window shown depends on the '
                  'model selected above). Dashed line, right of the "Forecast start" marker, is this '
                  'ONE selected method\'s own prediction; not blended with the other methods (see '
                  'Overview for that). The shaded band widens further out because near-term months '
                  'are always more certain than distant ones.'
              )):
        chart(fig, 340)

    if use_ml and importances:
        FEATURE_LABELS = {
            'horizon': 'Months ahead being forecast',
            'origin_month': 'Calendar month at forecast time',
            'target_month_sin': 'Seasonality (month of year)',
            'target_month_cos': 'Seasonality (month of year)',
            'roll_mean_3': '3-month rolling average',
            'roll_mean_6': '6-month rolling average',
            'roll_std_3': '3-month volatility',
        }
        imp = pd.Series(importances)
        imp.index = [FEATURE_LABELS.get(i, f'Income {i.split("_")[-1]} month(s) ago') for i in imp.index]
        imp = imp.groupby(imp.index).sum().sort_values(ascending=True).tail(6)

        fig_imp = go.Figure(go.Bar(
            x=imp.values, y=imp.index, orientation='h',
            marker_color=ui.TEAL,
            text=[f'{v:.0%}' for v in imp.values], textposition='outside',
            textfont=dict(size=11, color=ui.TEXT),
        ))
        fig_imp.update_layout(
            xaxis=dict(title='Relative importance', tickformat='.0%'),
            yaxis=dict(title=''), showlegend=False,
        )
        with card('What is driving this forecast',
                  'Relative importance of each engineered feature in the gradient-boosting model',
                  tag='Model transparency', tag_color='blue', info=(
                      'Longer bars mean the model relied on that input more when making its '
                      'predictions; e.g. "3-month rolling average" being longest means recent '
                      'income trend drives most of the forecast, not seasonality or older history. '
                      'This shows relative weighting inside the model, not proof any one factor '
                      'causes income to move.'
                  )):
            chart(fig_imp, 240)

    col1, col2 = st.columns([3, 1])
    with col1:
        table = fore[['calendar_month', 'predicted_income']].rename(
            columns={'calendar_month': 'Month', 'predicted_income': 'Predicted income'})
        with card('Forecast table', info=(
            'The exact predicted income for every month in the chart above, month 1 being the '
            'first month after "Forecast start"; use this for the precise numbers behind the line.'
        )):
            st.dataframe(
                table, hide_index=True, width='stretch', height=300,
                column_config={
                    'Month': st.column_config.NumberColumn(format='%d'),
                    'Predicted income': st.column_config.NumberColumn(format='dollar'),
                },
            )

    with col2:
        summary = pd.DataFrame({
            'Metric': ['12-month total', '24-month total', 'Month 1', f'Month {horizon}'],
            'Value': [
                f'${forecast_df.head(12)["predicted_income"].sum():,.0f}',
                f'${forecast_df["predicted_income"].sum():,.0f}',
                f'${fore.iloc[0]["predicted_income"]:,.0f}',
                f'${fore.iloc[-1]["predicted_income"]:,.0f}',
            ],
        })
        with card('Summary', info=(
            'Quick totals for this forecast: the 12- and 24-month sums, plus the very first and '
            'last month\'s predicted income, so you can see the trajectory without reading the '
            'whole table.'
        )):
            st.dataframe(summary, hide_index=True, width='stretch', height=180)
            st.download_button('Download CSV', data=fore.to_csv(index=False),
                                file_name='sf_forecast.csv', mime='text/csv',
                                icon=':material/download:', width='stretch')

    if model_choice == 'Stock-flow model' and sf_zone3_state:
        st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
        zone3_fig = go.Figure()
        scenario_colors = {'Conservative': ui.AMBER, 'Base': ui.TEAL, 'Optimistic': ui.PURPLE}
        for name, df3 in sf_zone3_state.items():
            zone3_fig.add_trace(go.Scatter(
                x=df3['calendar_month'], y=df3['predicted_income'],
                mode='lines+markers', name=name,
                line=dict(color=scenario_colors.get(name, ui.TEXT), width=2.5), marker=dict(size=5),
            ))
        zone3_fig.update_layout(
            yaxis=dict(tickformat='$,.0f', title='Monthly income ($)'),
            xaxis=dict(title='Month number'),
        )
        with card('Zone 3: strategic scenarios (months 19 to 36)',
                  'Not a statistical point forecast: confidence beyond 18 months is too low for that. '
                  'Three named scenarios, each built by scaling the same fitted recruitment, retention, '
                  'and gift models. The organisation should own which of these it plans around.',
                  tag='Named scenarios', tag_color='orange', info=(
                      'Three separate lines, each a distinct assumption set (Conservative/Base/'
                      'Optimistic), not three individually-fitted models and not a range around one '
                      'central prediction. Pick which scenario the organisation is planning around; '
                      'the table below each line shows the actual assumptions (recruitment, '
                      'retention, gift growth) behind it.'
                  )):
            chart(zone3_fig, 320)

            assum_rows = []
            for name in ['Conservative', 'Base', 'Optimistic']:
                if name not in sf_assumptions_state:
                    continue
                row = {'Scenario': name}
                row.update(sf_assumptions_state[name])
                df3 = sf_zone3_state[name]
                row['Month 36 income'] = f"${df3.iloc[-1]['predicted_income']:,.0f}"
                assum_rows.append(row)
            st.dataframe(pd.DataFrame(assum_rows), hide_index=True, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# RETENTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Retention Analysis':
    page_header('Retention analysis', 'Donor retention curves',
                '% of donors still active at each month since recruitment · observed from historical data.',
                meta=page_meta, info=(
                    'Each line is the % of donors from one segment still actively giving at each '
                    'month since they were recruited; observed directly from historical payment '
                    'data, not modeled or forecast. A line that drops faster means that segment '
                    'loses donors sooner after recruitment. Use the controls above the chart to '
                    'segment by supplier, campaign type, or recruitment year, and to change how '
                    'many months of tenure are shown.'
                ))

    if not st.session_state.pipeline_run:
        empty_state()

    rc1, rc2 = st.columns([2, 3], vertical_alignment='bottom')
    with rc1:
        segment = st.segmented_control(
            'Segment by', ['supplier', 'campaign_type', 'recruit_year'], default='supplier',
            format_func=lambda x: x.replace('_', ' ').title(), key='ret_seg',
        )
    with rc2:
        max_m = st.segmented_control('Months to show', [12, 24, 36, 48, 60], default=36,
                                      key='ret_m', format_func=lambda x: f'{x} months')
    segment = segment or 'supplier'
    max_m = max_m or 36

    ret_df = st.session_state.retention_by_segment.get(segment, pd.DataFrame())

    if ret_df.empty:
        with card():
            st.markdown(f'<div style="text-align:center;padding:24px;color:{ui.MIST};">'
                        f'Not enough data for this segment.</div>', unsafe_allow_html=True)
    else:
        ret_df = ret_df[ret_df['tenure_months'] <= max_m]
        fig = go.Figure()
        for i, grp_val in enumerate(ret_df[segment].unique()):
            grp = ret_df[ret_df[segment] == grp_val]
            fig.add_trace(go.Scatter(
                x=grp['tenure_months'], y=grp['retention_pct'],
                mode='lines', name=str(grp_val),
                line=dict(color=ui.COLOURS[i % len(ui.COLOURS)], width=2),
            ))
        fig.add_hline(y=50, line_dash='dot', line_color=ui.MIST, opacity=0.7,
                      annotation_text='50% retention', annotation_font_size=10)
        fig.update_layout(
            yaxis=dict(range=[0, 105], ticksuffix='%', title='Donors still active (%)'),
            xaxis=dict(title='Months since recruitment'),
        )
        with card(f'Retention by {segment.replace("_", " ").title()}',
                  '% of donors still active at each tenure month',
                  tag='Observed', tag_color='blue', info=(
                      'Each colored line is one group in the segment selected above; e.g. one '
                      'supplier, or one campaign type. The y-axis is % of that group\'s donors still '
                      'giving at each month since recruitment; a line that drops faster loses donors '
                      'sooner. The dotted horizontal line marks 50% retention as a reference point, '
                      'not a target.'
                  )):
            chart(fig, 360)

        milestones = [3, 6, 12, 18, 24]
        rows = []
        for grp_val in ret_df[segment].unique():
            grp = ret_df[ret_df[segment] == grp_val]
            row = {segment.replace('_', ' ').title(): str(grp_val)}
            for m in milestones:
                match = grp[grp['tenure_months'] == m]['retention_pct']
                row[f'Month {m}'] = float(match.values[0]) if len(match) else None
            rows.append(row)
        milestone_df = pd.DataFrame(rows)

        col_cfg = {f'Month {m}': st.column_config.NumberColumn(format='%.1f%%') for m in milestones}
        with card('Retention at key milestones', info=(
            'The same retention curve above, read off at fixed tenure milestones (3/6/12/18/24 '
            'months) for exact numbers instead of eyeballing the chart. A blank cell means that '
            'group has no donors who have reached that many months of tenure yet.'
        )):
            st.dataframe(milestone_df, hide_index=True, width='stretch', column_config=col_cfg)


# ══════════════════════════════════════════════════════════════════════════════
# DONOR LIFETIME VALUE
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Donor Lifetime Value':
    page_header('Donor value', 'Donor lifetime value',
                'Pareto/NBD predicts how many more months each existing donor will keep giving. '
                'Gamma-Gamma predicts their expected donation size, a per-donor $ forecast, not an '
                'aggregate average. Scoped to today\'s donor base only (assumes zero future '
                'recruitment). See Income forecast for total org income including recruitment.',
                meta=page_meta, info=(
                    'This answers a DIFFERENT question from Income Forecast: not "what will the '
                    'organisation earn," but "what are today\'s existing donors worth going forward," '
                    'assuming no further recruitment. The Predicted 12/24-month value KPIs are the '
                    'individual per-donor predictions summed across the whole donor base. The bar '
                    'chart ranks the highest-value individual donors; the histogram shows how '
                    'predicted value is distributed across everyone. Fit fresh each live session, not '
                    'saved with historical runs; reload a past run from Run History and this page '
                    'won\'t have a value to show.'
                ))

    if not st.session_state.pipeline_run:
        empty_state()

    ltv_results = st.session_state.ltv_results
    ltv_tuning  = st.session_state.ltv_tuning
    ltv_metrics = st.session_state.ltv_metrics

    # Fit lazily, right here, on first visit -- this model is no longer
    # fit as part of "Run pipeline" (it was the slowest stage, ~5 min, and
    # this is the only page that reads its output; see cached_ltv_fit()
    # near the top of this file). Cached on `master`, so navigating away
    # and back doesn't refit. Only possible in the same session that ran
    # the pipeline live -- `master` itself is never persisted to Run
    # History (too large to store) -- so a reloaded historical run can't
    # refit it after the fact and falls through to the message below.
    if ltv_results is None and st.session_state.get('master') is not None:
        with st.spinner('Fitting Pareto/NBD + Gamma-Gamma donor lifetime-value model…'):
            try:
                ltv_results, ltv_tuning, ltv_metrics, ltv_monthly = cached_ltv_fit(st.session_state.master)
                st.session_state.ltv_results = ltv_results
                st.session_state.ltv_tuning  = ltv_tuning
                st.session_state.ltv_metrics = ltv_metrics
                st.session_state.ltv_monthly = ltv_monthly
                st.session_state.ltv_error   = None
                counts, edges = np.histogram(ltv_results['predicted_ltv_24m'].dropna(), bins=30)
                st.session_state.ltv_histogram = {'bin_edges': edges.tolist(), 'counts': counts.tolist()}
            except ValueError as exc:
                ltv_results = None
                st.session_state.ltv_error = str(exc)

    if ltv_results is None:
        with card():
            st.markdown(f"""
            <div style="text-align:center;padding:28px;">
                <div style="font-size:14px;font-weight:700;color:{ui.TEXT};margin-bottom:6px;">
                    Donor lifetime value model unavailable
                </div>
                <div style="font-size:12.5px;color:{ui.MIST};max-width:480px;margin:0 auto;">
                    {st.session_state.ltv_error or 'Run a fresh pipeline in this session to compute this model, '
                     'it isn\'t stored with saved runs from history.'}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.container(horizontal=True):
            kpi('Donors modelled', f'{ltv_metrics["total_donors"]:,}',
                delta=f'{ltv_metrics["gamma_gamma_eligible"]:,} with repeat gifts',
                icon=':material/groups:', accent=ui.TEAL, delta_color='off')
            kpi('Predicted 12-month value', f'${ltv_metrics["predicted_value_12m"]:,.0f}',
                delta='existing donors only', icon=':material/savings:', accent=ui.BLUE, delta_color='off')
            kpi('Predicted 24-month value', f'${ltv_metrics["predicted_value_24m"]:,.0f}',
                delta='existing donors only', icon=':material/savings:', accent=ui.PURPLE, delta_color='off')
            kpi('Holdout MAPE', f'{ltv_metrics["holdout_mape"]:.1f}%',
                delta=f'penalizer {ltv_metrics["best_penalizer"]:g}',
                icon=':material/verified:', accent=ui.AMBER, delta_color='off')

        col1, col2 = st.columns([3, 2])
        with col1:
            top15 = ltv_results.sort_values('predicted_ltv_24m', ascending=False).head(15)
            fig = go.Figure(go.Bar(
                x=top15['predicted_ltv_24m'], y=top15['contact_id'].astype(str),
                orientation='h', marker_color=ui.TEAL,
                text=[f'${v:,.0f}' for v in top15['predicted_ltv_24m']],
                textposition='outside', textfont=dict(size=10, color=ui.TEXT),
            ))
            fig.update_layout(
                xaxis=dict(title='Predicted 24-month value ($)', tickformat='$,.0f'),
                yaxis=dict(title='', autorange='reversed'), showlegend=False,
            )
            with card('Top 15 donors by predicted value',
                      'Highest predicted lifetime value over the next 24 months', tag='Ranked', info=(
                          'Each bar is one individual donor (by contact ID), ranked by their own '
                          'predicted 24-month value; not a segment or supplier average. Use this to '
                          'identify specific high-value donors worth prioritising, e.g. for '
                          'stewardship outreach.'
                      )):
                chart(fig, 380)

        with col2:
            # Pre-binned (not a raw-value go.Histogram) so this chart works
            # identically whether ltv_results is the live full donor table
            # or just the cached top-50 -- the bin counts themselves are
            # computed once in run_pipeline_models() from the full table
            # and cached directly, since 30 (edge, count) pairs is tiny
            # regardless of donor count, unlike the raw per-donor values.
            hist = st.session_state.ltv_histogram or {'bin_edges': [], 'counts': []}
            edges, counts = hist['bin_edges'], hist['counts']
            centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
            widths = [edges[i + 1] - edges[i] for i in range(len(counts))]
            # ui.TEAL, not ui.TEXT -- ui.TEXT is the text-colour variable, not a
            # decoration colour; using it here made this chart's bars a
            # flat, mismatched navy next to the teal "Top 15 donors"
            # chart beside it (and, being ui.TEXT specifically, an odd
            # near-white in dark mode too, since that variable tracks
            # whatever the current body text colour is, not an accent).
            fig2 = go.Figure(go.Bar(x=centers, y=counts, width=widths, marker_color=ui.TEAL))
            fig2.update_layout(
                xaxis=dict(title='Predicted 24-month value ($)', tickformat='$,.0f'),
                yaxis=dict(title='Number of donors'), showlegend=False,
            )
            with card('Distribution of predicted value',
                      'How predicted 24-month value is spread across all donors', info=(
                          'A histogram, not a ranking: each bar is a $ range, and its height is how '
                          'many donors fall in that range. A tall bar near $0 with a long low tail to '
                          'the right means most donors are predicted to give modestly, with a small '
                          'number of high-value outliers; the same donors shown individually in '
                          '"Top 15 donors" to the left.'
                      )):
                chart(fig2, 380)

        col3, col4 = st.columns([3, 2])
        with col3:
            table = ltv_results.sort_values('predicted_ltv_24m', ascending=False).head(50).rename(columns={
                'recurring_payment_id': 'Signup ID', 'contact_id': 'Contact',
                'frequency': 'Repeat gifts', 'monetary_value': 'Avg gift ($)',
                'expected_donation_value': 'Expected next gift ($)',
                'predicted_ltv_12m': 'Predicted 12m value', 'predicted_ltv_24m': 'Predicted 24m value',
            })[['Signup ID', 'Contact', 'Repeat gifts', 'Avg gift ($)', 'Expected next gift ($)',
                'Predicted 12m value', 'Predicted 24m value']]
            with card('Top 50 donors', 'Ranked by predicted 24-month value', tag='Actionable list', tag_color='blue', info=(
                'One row per donor. "Repeat gifts" is how many donations Pareto/NBD has actually '
                'observed from them (more repeat gifts generally means a more confident prediction). '
                '"Expected next gift" and the 12/24-month value columns are per-donor forecasts, not '
                'historical totals.'
            )):
                st.dataframe(
                    table, hide_index=True, width='stretch', height=340,
                    column_config={
                        'Avg gift ($)': st.column_config.NumberColumn(format='dollar'),
                        'Expected next gift ($)': st.column_config.NumberColumn(format='dollar'),
                        'Predicted 12m value': st.column_config.NumberColumn(format='dollar'),
                        'Predicted 24m value': st.column_config.NumberColumn(format='dollar'),
                    },
                )
                # Administrator-or-above -- this is the full per-donor table
                # (contact_id-level), not an aggregate, same reasoning as
                # the page-wide Export CSV button (see page_header()).
                if (st.session_state.user or {}).get('role') in ('Administrator', 'Super Admin'):
                    st.download_button(
                        'Download all donor predictions (CSV)',
                        data=ltv_results.to_csv(index=False),
                        file_name='sf_donor_ltv_predictions.csv', mime='text/csv',
                        icon=':material/download:', width='stretch',
                    )

        with col4:
            tune_table = ltv_tuning.rename(columns={
                'penalizer': 'Penalizer', 'mape': 'Holdout MAPE (%)',
                'mae': 'Holdout MAE', 'aggregate_error': 'Aggregate error (%)',
            })
            with card('Model validation', 'Penalizer grid search on a 12-month holdout, trained from 2019', tag='Tuning', tag_color='orange', info=(
                'Each row tried a different penalizer (a regularization setting) for Pareto/NBD; the '
                'one used elsewhere on this page is whichever row had the lowest holdout MAPE, shown '
                'in the "Holdout MAPE" KPI above. The caption below the table compares the '
                'Gamma-Gamma model\'s predicted average gift to what was actually observed; the '
                'closer those two numbers, the better calibrated the gift-size prediction is.'
            )):
                st.dataframe(
                    tune_table, hide_index=True, width='stretch', height=200,
                    column_config={
                        'Holdout MAPE (%)': st.column_config.NumberColumn(format='%.1f'),
                        'Holdout MAE': st.column_config.NumberColumn(format='%.3f'),
                        'Aggregate error (%)': st.column_config.NumberColumn(format='%.1f'),
                    },
                )
                st.caption(
                    f'Gamma-Gamma: observed avg gift \\${ltv_metrics["observed_avg_donation"]:,.2f} vs '
                    f'model-expected \\${ltv_metrics["expected_avg_donation"]:,.2f}.'
                )


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLIER INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Supplier Insights':
    page_header('Supplier insights', 'Which recruiting firms perform best',
                'Retention, lifetime value and income contribution by supplier.',
                meta=page_meta, info=(
                    'Compares suppliers by more than raw signup volume: retention (do their donors '
                    'stick around), average gift size, and total income contribution. A supplier '
                    'with fewer signups but stronger retention or gift size can outperform one that '
                    'recruits more donors who lapse quickly. Use the monthly trend chart to select '
                    'suppliers and compare their income trajectory over time.'
                ))

    if not st.session_state.pipeline_run:
        empty_state()

    sup = st.session_state.supplier_summary
    top = sup.iloc[0]

    with st.container(horizontal=True):
        kpi('Top supplier', top['supplier'], delta='By total income',
            icon=':material/military_tech:', accent=ui.TEAL, delta_color='off')
        kpi('Top supplier income', f'${top["total_income"]:,.0f}', delta='Historical total',
            icon=':material/payments:', accent=ui.BLUE, delta_color='off')
        kpi('Total suppliers', str(len(sup)), delta='In dataset',
            icon=':material/store:', accent=ui.PURPLE, delta_color='off')
        kpi('Avg gift (top)', f'${top["avg_gift"]:,.2f}', delta='Per payment',
            icon=':material/percent:', accent=ui.AMBER, delta_color='off')

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(
            x=sup['supplier'], y=sup['total_income'],
            marker_color=ui.COLOURS[:len(sup)],
            text=[f'${v/1e6:.2f}M' for v in sup['total_income']],
            textposition='outside', textfont=dict(size=11, color=ui.TEXT),
        ))
        fig.update_layout(yaxis=dict(tickformat='$,.0f', title='Total income ($)'), showlegend=False)
        with card('Total income by supplier', info=(
            'Total historical income attributed to donors each supplier recruited, not a per-donor '
            'average; a supplier with more signups will tend to rank higher here even if its '
            'individual donors give less. Compare against "Active donors by supplier" and the '
            'summary table\'s "Avg gift" column for the fuller picture.'
        )):
            chart(fig, 280)

    with col2:
        fig2 = go.Figure(go.Bar(
            x=sup['supplier'], y=sup['active_donors'],
            marker_color=ui.COLOURS[:len(sup)],
            text=[f'{v:,}' for v in sup['active_donors']],
            textposition='outside', textfont=dict(size=11, color=ui.TEXT),
        ))
        fig2.update_layout(yaxis=dict(title='Active donors'), showlegend=False)
        with card('Active donors by supplier', info=(
            'How many currently-active donors each supplier has recruited in total, not how many '
            'they recruited this month. A supplier can look strong here purely on volume even if a '
            'large share of those donors give small amounts; see Retention Analysis to check how '
            'well each supplier\'s donors are retained over time.'
        )):
            chart(fig2, 280)

    sup_monthly = st.session_state.supplier_monthly
    all_sups = [s for s in sup['supplier'].dropna().unique() if s != 'Unknown']

    with card('Monthly income trend by supplier',
              'Select suppliers to compare their monthly income trajectory', info=(
                  'Real historical monthly income, not a forecast; one line per supplier selected '
                  'in the box below. Use this to spot which suppliers are trending up or down over '
                  'time, not just their all-time totals shown in the bar charts above.'
              )):
        selected = st.multiselect('Compare suppliers', options=all_sups, default=all_sups[:3],
                                   label_visibility='collapsed')
        if selected:
            fig3 = go.Figure()
            for i, s in enumerate(selected):
                grp = sup_monthly[sup_monthly['supplier'] == s]
                fig3.add_trace(go.Scatter(
                    x=grp['donor_month'], y=grp['total_income'],
                    mode='lines', name=s,
                    line=dict(color=ui.COLOURS[i % len(ui.COLOURS)], width=2),
                ))
            fig3.update_layout(yaxis=dict(tickformat='$,.0f', title='Monthly income ($)'),
                                xaxis_title='Month')
            chart(fig3, 260)
        else:
            st.caption('Select at least one supplier to see the trend line.')

    sup_display = sup.rename(columns={
        'supplier': 'Supplier', 'total_income': 'Total income',
        'active_donors': 'Active donors', 'avg_gift': 'Avg gift',
    })
    with card('Supplier summary', info=(
        'One row per supplier with the exact numbers behind the charts above. "Avg gift" is per '
        'payment, not per donor; a useful check against "Total income" and "Active donors" for '
        'whether a supplier\'s strength is volume, gift size, or both.'
    )):
        st.dataframe(
            sup_display, hide_index=True, width='stretch',
            column_config={
                'Total income': st.column_config.NumberColumn(format='dollar'),
                'Active donors': st.column_config.NumberColumn(format='%d'),
                'Avg gift': st.column_config.NumberColumn(format='dollar'),
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# CAMPAIGN ROI
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Campaign ROI':
    page_header('Campaign insights', 'Which campaigns perform best',
                'Retention, income and return on acquisition cost by campaign type.',
                meta=page_meta, info=(
                    'Compares campaign types by income AND cost per acquisition (CPA), not just which '
                    'recruited the most donors. A campaign with a higher CPA can still be worth it if '
                    'it recruits donors who give more or stay longer; check this page alongside '
                    'Retention Analysis and Supplier Insights for that fuller picture. "Free Sales" '
                    'campaigns (CPA of $0) are shown separately since a $/donor cost comparison '
                    'doesn\'t apply to them.'
                ))

    if not st.session_state.pipeline_run:
        empty_state()

    roi = st.session_state.campaign_summary

    best = roi.sort_values('total_income', ascending=False).iloc[0]
    has_cpa = 'avg_cpa' in roi.columns
    cpa_vals = roi[roi['avg_cpa'] > 0]['avg_cpa'] if has_cpa else pd.Series(dtype=float)
    cpa_range = f'\\${cpa_vals.min():.0f} to \\${cpa_vals.max():.0f}' if len(cpa_vals) else 'No CPA data'

    with st.container(horizontal=True):
        kpi('Top income campaign', best['campaign_type'], delta=f'\\${best["total_income"]:,.0f} total',
            icon=':material/campaign:', accent=ui.TEAL, delta_color='off')
        kpi('Campaign types', str(len(roi)), delta='In dataset',
            icon=':material/category:', accent=ui.BLUE, delta_color='off')
        kpi('CPA range', cpa_range, delta='Non-zero cost campaigns',
            icon=':material/paid:', accent=ui.PURPLE, delta_color='off')
        kpi('Free-acquisition income', f'${roi[roi["avg_cpa"] == 0]["total_income"].sum():,.0f}' if has_cpa else 'N/A',
            delta='Free Sales campaigns', icon=':material/redeem:', accent=ui.AMBER, delta_color='off')

    col1, col2 = st.columns(2)
    with col1:
        sorted_roi = roi.sort_values('total_income', ascending=False)
        fig = go.Figure(go.Bar(
            x=sorted_roi['campaign_type'], y=sorted_roi['total_income'],
            marker_color=ui.COLOURS[:len(roi)],
            text=[f'${v/1e6:.2f}M' for v in sorted_roi['total_income']],
            textposition='outside', textfont=dict(size=11, color=ui.TEXT),
        ))
        fig.update_layout(yaxis=dict(tickformat='$,.0f', title='Total income ($)'), showlegend=False)
        with card('Total income by campaign type', info=(
            'Total historical income from donors each campaign type recruited; driven by both how '
            'many donors it recruited and how much they give, not cost-efficiency. Check against '
            '"Average cost per acquisition" to see whether a high-income campaign type was also '
            'expensive to run.'
        )):
            chart(fig, 280)

    with col2:
        if has_cpa:
            fig2 = go.Figure(go.Bar(
                x=roi['campaign_type'], y=roi['avg_cpa'],
                marker_color=ui.COLOURS[:len(roi)],
                text=[f'${v:.0f}' if v > 0 else 'Free' for v in roi['avg_cpa']],
                textposition='outside', textfont=dict(size=11, color=ui.TEXT),
            ))
            fig2.update_layout(yaxis=dict(tickformat='$,.0f', title='Avg CPA ($)'), showlegend=False)
            with card('Average cost per acquisition', tag='CPA data', tag_color='orange', info=(
                'Average $ spent to recruit one donor, by campaign type; "Free" bars are $0-cost '
                'channels (e.g. organic Sales campaigns). A lower CPA is not automatically better: '
                'weigh it against that same campaign type\'s income and retention, not on its own.'
            )):
                chart(fig2, 280)
        else:
            with card('Average cost per acquisition'):
                st.markdown(f'<div style="text-align:center;padding:24px;color:{ui.MIST};">'
                            f'CPA data not found in Campaigns file.</div>', unsafe_allow_html=True)

    cm = st.session_state.campaign_monthly
    fig3 = go.Figure()
    for i, ct in enumerate(cm['campaign_type'].dropna().unique()):
        grp = cm[cm['campaign_type'] == ct]
        fig3.add_trace(go.Scatter(
            x=grp['donor_month'], y=grp['total_income'],
            mode='lines', name=ct,
            line=dict(color=ui.COLOURS[i % len(ui.COLOURS)], width=2),
        ))
    fig3.update_layout(yaxis=dict(tickformat='$,.0f', title='Monthly income ($)'), xaxis_title='Month')
    with card('Monthly income by campaign type', info=(
        'Real historical monthly income, not a forecast; one line per campaign type. Use this to '
        'spot which campaign types are trending up or down, not just their all-time totals shown in '
        'the bar chart above.'
    )):
        chart(fig3, 260)

    roi_display = roi.sort_values('total_income', ascending=False).rename(columns={
        'campaign_type': 'Campaign type', 'total_income': 'Total income',
        'active_donors': 'Active donors', 'avg_gift': 'Avg gift', 'avg_cpa': 'Avg CPA',
    })
    col_cfg = {
        'Total income': st.column_config.NumberColumn(format='dollar'),
        'Active donors': st.column_config.NumberColumn(format='%d'),
        'Avg gift': st.column_config.NumberColumn(format='dollar'),
    }
    if has_cpa:
        col_cfg['Avg CPA'] = st.column_config.NumberColumn(format='dollar')
    with card('Campaign summary', info=(
        'One row per campaign type with the exact numbers behind the charts above; income, active '
        'donors, average gift, and average CPA side by side, for comparing cost against return '
        'directly instead of switching between charts.'
    )):
        st.dataframe(roi_display, hide_index=True, width='stretch', column_config=col_cfg)


# ══════════════════════════════════════════════════════════════════════════════
# FORECAST VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Forecast Verification':
    page_header('Forecasting', 'Forecast verification',
                'Compare any two saved forecast runs side by side, and check both against the '
                'actual fund income now on record.',
                meta=page_meta, info=(
                    'Pick any two runs. Run A is the baseline (the loaded run by default), '
                    'Run B is compared against it; swap either freely to verify historical '
                    'forecasts too. Each run forecast from its own last actual month, so an '
                    'older run\'s forecast overlaps months that have since happened; that '
                    'overlap is the verification. "Actual fund income on record" is the union '
                    'of the runs\' actuals, so a month one run only forecast, but another has '
                    'since recorded as real data, gets verified.'
                ))

    if not st.session_state.pipeline_run:
        empty_state()

    def _fv_notice(head, body):
        with card():
            st.markdown(
                f'<div style="text-align:center;padding:28px;">'
                f'<div style="font-size:14px;font-weight:700;color:{ui.TEXT};margin-bottom:6px;">{head}</div>'
                f'<div style="font-size:12.5px;color:{ui.MIST};max-width:520px;margin:0 auto;">{body}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        clear_nav_overlay()
        st.stop()

    if not db_configured():
        _fv_notice('Verification needs run history',
                   'No database connection is configured, so past runs are not saved and there '
                   'is nothing to compare the current forecast against.')

    fv_runs = list_dashboard_runs()
    fv_cur_id = st.session_state.viewing_run_id
    fv_others = fv_runs[fv_runs['run_id'] != fv_cur_id] if not fv_runs.empty else fv_runs
    if fv_others.empty:
        _fv_notice('Only one run so far',
                   'There needs to be at least one other saved run to compare against. Run the '
                   'pipeline again from the Data Pipeline page, then come back here.')

    fv_all = fv_runs.copy()
    fv_all['run_at'] = pd.to_datetime(fv_all['run_at'])
    fv_labels = {
        r['run_id']: f"{r['run_id']}   ·   {r['run_at']:%d %b %Y, %H:%M}   ·   {r['run_by'] or 'unknown'}"
        for _, r in fv_all.iterrows()
    }
    _fv_ids = list(fv_labels)
    _fv_a_def = fv_cur_id if fv_cur_id in _fv_ids else _fv_ids[0]
    _fv_b_def = next((i for i in _fv_ids if i != _fv_a_def), _fv_a_def)

    fp1, fp2, fp3 = st.columns([1.4, 1.4, 1], vertical_alignment='bottom')
    with fp1:
        fv_a_id = st.selectbox('Run A · baseline', _fv_ids, index=_fv_ids.index(_fv_a_def),
                               format_func=lambda x: fv_labels[x], key='fv_a',
                               help='Defaults to the run loaded right now. Pick any saved run.')
    with fp2:
        fv_b_id = st.selectbox('Run B · compared against A', _fv_ids, index=_fv_ids.index(_fv_b_def),
                               format_func=lambda x: fv_labels[x], key='fv_b')
    with fp3:
        fv_model = st.segmented_control(
            'Forecast model',
            ['ML forecast', 'Linear trend', 'Stock-flow model', 'Blended'],
            default='ML forecast', key='fv_model') or 'ML forecast'

    if fv_a_id == fv_b_id:
        _fv_notice('Pick two different runs',
                   'Run A and Run B are the same run. Choose a different run in one of the '
                   'dropdowns to compare them.')

    fv_is_sf = fv_model == 'Stock-flow model'
    fv_is_blend = fv_model == 'Blended'

    if fv_is_sf:
        st.caption('Stock-flow is validated on 3 rolling walk-forward windows, the other two on '
                   'one fixed 12-month holdout, so its MAPE is only directionally comparable. Months '
                   '19 to 24 use the Base scenario\'s central assumptions.')
    elif fv_is_blend:
        st.caption('Blended is the month-by-month mean of every available method (ML forecast, '
                   'Linear trend, Stock-flow), the same combined forecast the Overview page '
                   'shows. Its MAPE is the average of those methods\' own validation scores.')

    def _sf_combined(zone12, zone3, n=24):
        """Stock-flow's saved forecast is the statistical window (months
        1–18) only; months 19–n come from the 'Base' scenario, exactly the
        way the Income Forecast page stitches them. Returns a
        calendar_month / predicted_income frame (up to n rows), or None if
        stock-flow wasn't produced for that run."""
        if zone12 is None or getattr(zone12, 'empty', True):
            return None
        df = zone12[['calendar_month', 'predicted_income']].copy()
        if n > len(df) and zone3 and 'Base' in zone3 and not zone3['Base'].empty:
            extra = zone3['Base'][['calendar_month', 'predicted_income']].head(n - len(df)).copy()
            df = pd.concat([df, extra], ignore_index=True)
        df = df.head(n).reset_index(drop=True)
        df['calendar_month'] = range(1, len(df) + 1)
        return df

    def _blend_fc(parts, n=24):
        """Month-by-month mean of every available total-income method
        (ML, Linear, Stock-flow) the same blend the Overview page shows.
        `parts` is a list of predicted_income-bearing frames; returns a
        calendar_month / predicted_income frame, or None if none are usable."""
        series = [p['predicted_income'].head(n).reset_index(drop=True)
                  for p in parts
                  if p is not None and not getattr(p, 'empty', True)
                  and 'predicted_income' in getattr(p, 'columns', [])]
        if not series:
            return None
        m = min(len(s) for s in series)
        avg = pd.concat([s.head(m) for s in series], axis=1).mean(axis=1)
        return pd.DataFrame({'calendar_month': range(1, m + 1), 'predicted_income': avg.values})

    def _mape_mean(vals):
        got = [v for v in vals if v is not None]
        return sum(got) / len(got) if got else None

    _run_at = dict(zip(fv_all['run_id'], fv_all['run_at']))
    _MODEL_FC_KEY = {'ML forecast': 'ml', 'Linear trend': 'linear', 'Stock-flow model': 'stockflow'}

    def _fv_side(run_id):
        """Everything one side of the comparison needs for the selected
        model: pulled from that run's saved dashboard state, or live from
        session_state when it is the run currently loaded (freshest, and
        covers a live session whose autosave is a moment behind)."""
        if run_id == fv_cur_id and st.session_state.get('pipeline_run'):
            sc = {'total_income': st.session_state.total_income,
                  'donor_count': st.session_state.donor_count,
                  'mape': st.session_state.mape,
                  'mape_linear': st.session_state.mape_linear,
                  'mape_stockflow': st.session_state.mape_stockflow}
            actuals = st.session_state.monthly.copy()
            fcs = {'ml': st.session_state.forecast_df,
                   'linear': st.session_state.forecast_df_linear,
                   'stockflow': st.session_state.sf_zone12}
            z3 = st.session_state.sf_zone3
        else:
            state, _ = load_dashboard_run(run_id)
            if not state:
                return None
            sc = state.get('scalars', {})
            actuals = pd.DataFrame(state.get('monthly', []))
            _f = state.get('forecasts') or {}
            fcs = {k: pd.DataFrame(_f.get(k) or []) for k in ('ml', 'linear', 'stockflow')}
            z3 = {k: pd.DataFrame(v) for k, v in (state.get('sf_zone3') or {}).items()}

        _sfz = fcs.get('stockflow')
        _sfz = None if _sfz is None or getattr(_sfz, 'empty', True) else _sfz
        if fv_is_blend:
            fc = _blend_fc([fcs.get('ml'), fcs.get('linear'), _sf_combined(_sfz, z3)])
            mape = _mape_mean([sc.get('mape'), sc.get('mape_linear'), sc.get('mape_stockflow')])
        elif fv_is_sf:
            fc = _sf_combined(_sfz, z3)
            mape = sc.get('mape_stockflow')
        else:
            _k = _MODEL_FC_KEY[fv_model]
            fc = fcs.get(_k)
            mape = sc.get('mape' if _k == 'ml' else f'mape_{_k}')
        return {'actuals': actuals, 'fc': fc, 'mape': mape,
                'scalars': sc, 'run_at': _run_at.get(run_id)}

    with st.spinner('Loading the selected runs…'):
        _a, _b = _fv_side(fv_a_id), _fv_side(fv_b_id)
    if _a is None or _b is None:
        st.error('Could not load one of the selected runs; it may have been deleted.',
                 icon=':material/error:')
        clear_nav_overlay()
        st.stop()

    cur_actuals, cur_fc, cur_mape = _a['actuals'], _a['fc'], _a['mape']
    cmp_actuals, cmp_fc, cmp_mape = _b['actuals'], _b['fc'], _b['mape']
    cur_sc, cmp_sc = _a['scalars'], _b['scalars']
    fv_a_at, fv_b_at = _a['run_at'], _b['run_at']

    # "Actual fund income on record" -- the UNION of the two selected runs'
    # actuals plus the newest run's, because runs are built from different
    # data vintages: a month one run only forecast is often already a
    # recorded actual in another. For any shared month the value from the
    # most recently executed run wins.
    fv_newest_id = fv_all.iloc[0]['run_id']
    _actual_srcs = [(_run_at.get(fv_a_id, pd.Timestamp.min), cur_actuals),
                    (_run_at.get(fv_b_id, pd.Timestamp.min), cmp_actuals)]
    if fv_newest_id not in (fv_a_id, fv_b_id):
        _ns, _ = load_dashboard_run(fv_newest_id)
        _actual_srcs.append((_run_at.get(fv_newest_id, pd.Timestamp.min),
                             pd.DataFrame((_ns or {}).get('monthly', []))))
    _act_frames = [df.assign(_pri=at) for at, df in _actual_srcs
                   if df is not None and not getattr(df, 'empty', True) and 'donor_month' in df.columns]
    if _act_frames:
        actuals_now = (pd.concat(_act_frames, ignore_index=True)
                         .sort_values(['donor_month', '_pri'], ascending=[True, False])
                         .drop_duplicates('donor_month', keep='first')
                         .drop(columns='_pri')
                         .sort_values('donor_month')
                         .reset_index(drop=True))
    else:
        actuals_now = st.session_state.monthly.copy()

    if (cur_fc is None or getattr(cur_fc, 'empty', True)
            or cmp_fc is None or getattr(cmp_fc, 'empty', True)
            or 'donor_month' not in getattr(cur_actuals, 'columns', [])
            or 'donor_month' not in getattr(cmp_actuals, 'columns', [])):
        _fv_notice(f'{fv_model} not saved for both runs',
                   f'One of the selected runs has no saved {fv_model.lower()} to line up. '
                   'Try a different forecast model, or pick other runs.')

    def _fc_months(actuals_df, fc_df):
        """Real calendar month for each row of a forecast_df (whose own
        'calendar_month' column is only a 1..N offset) -- the month right
        after that run's last actual month, then monthly from there."""
        last = pd.Period(str(actuals_df['donor_month'].iloc[-1]), 'M')
        return [(last + i).to_timestamp() for i in range(1, len(fc_df) + 1)]

    cur_fc_x = _fc_months(cur_actuals, cur_fc)
    cmp_fc_x = _fc_months(cmp_actuals, cmp_fc)

    def _tot(fc_df, n):
        return float(fc_df.head(n)['predicted_income'].sum())

    cur_12, cur_24 = _tot(cur_fc, 12), _tot(cur_fc, 24)
    cmp_12, cmp_24 = _tot(cmp_fc, 12), _tot(cmp_fc, 24)

    def _money_delta(value, base):
        d = value - base
        return f'{"+" if d >= 0 else "-"}${abs(d):,.0f} vs Run A'

    def _mape_delta(value, base):
        if value is None or base is None:
            return None
        d = value - base
        return f'{d:+.1f} pts vs Run A'

    # ── Side-by-side metrics ──────────────────────────────────────────────
    mc1, mc2 = st.columns(2)
    with mc1:
        _sub = f'{fv_a_id} · {fv_a_at:%d %b %Y}' if fv_a_at else fv_a_id
        _a_tag = 'BASELINE · LOADED' if fv_a_id == fv_cur_id else 'BASELINE'
        with card('Run A', _sub, tag=_a_tag, tag_color='blue'):
            with st.container(horizontal=True):
                kpi('12-month forecast', f'${cur_12:,.0f}', delta='Projected income',
                    delta_color='off', k='cur')
                kpi('24-month forecast', f'${cur_24:,.0f}', delta='Projected income',
                    delta_color='off', k='cur')
            with st.container(horizontal=True):
                kpi('Validation MAPE', f'{cur_mape:.1f}%' if cur_mape is not None else 'N/A',
                    delta='Measured at run time', delta_color='off', k='cur')
                kpi('Fund income on record',
                    f'${cur_sc.get("total_income"):,.0f}' if cur_sc.get('total_income') else 'N/A',
                    delta=f'{int(cur_sc["donor_count"]):,} donors'
                          if cur_sc.get('donor_count') else '',
                    delta_color='off', k='cur')
    with mc2:
        _sub = f'{fv_b_id} · {fv_b_at:%d %b %Y}' if fv_b_at else fv_b_id
        with card('Run B', _sub, tag='COMPARED', tag_color='violet'):
            with st.container(horizontal=True):
                kpi('12-month forecast', f'${cmp_12:,.0f}', delta=_money_delta(cmp_12, cur_12),
                    delta_color='off', k='cmp')
                kpi('24-month forecast', f'${cmp_24:,.0f}', delta=_money_delta(cmp_24, cur_24),
                    delta_color='off', k='cmp')
            with st.container(horizontal=True):
                kpi('Validation MAPE', f'{cmp_mape:.1f}%' if cmp_mape is not None else 'N/A',
                    delta=_mape_delta(cmp_mape, cur_mape), delta_color='off', k='cmp')
                kpi('Fund income on record',
                    f'${cmp_sc.get("total_income"):,.0f}' if cmp_sc.get('total_income') else 'N/A',
                    delta=(f'{int(cmp_sc["donor_count"]):,} donors'
                           if cmp_sc.get('donor_count') else ''),
                    delta_color='off', k='cmp')

    # ── One timeline: both forecasts against the real actuals ─────────────
    act = actuals_now.copy()
    act['ts'] = act['donor_month'].apply(lambda m: pd.Period(str(m), 'M').to_timestamp())
    act = act[act['ts'] >= (min(cur_fc_x + cmp_fc_x) - pd.DateOffset(months=18))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=act['ts'], y=act['total_income'], mode='lines+markers', name='Actual fund income',
        line=dict(color=ui.TEXT, width=2.5), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=cmp_fc_x, y=cmp_fc['predicted_income'], mode='lines+markers', name='Run B forecast',
        line=dict(color=ui.PURPLE, width=2.5, dash='dot'), marker=dict(size=5, symbol='diamond'),
    ))
    fig.add_trace(go.Scatter(
        x=cur_fc_x, y=cur_fc['predicted_income'], mode='lines+markers', name='Run A forecast',
        line=dict(color=ui.TEAL, width=2.5, dash='dash'), marker=dict(size=5, symbol='square'),
    ))
    if len(act):
        fig.add_vline(x=act['ts'].iloc[-1], line_dash='dot', line_color=ui.MIST, opacity=0.8,
                      annotation_text='Actuals end', annotation_font_size=10,
                      annotation_font_color=ui.MIST)
    fig.update_layout(yaxis=dict(tickformat='$,.0f', title='Monthly income ($)'), xaxis_title='Month')
    with card('Forecast vs actual fund income',
              'Both runs\' forecasts on one timeline. Where a forecast line sits left of '
              '"Actuals end" it is overlapping real data: the gap between it and the solid '
              'line is that forecast\'s error.',
              tag=fv_model, tag_color='blue'):
        chart(fig, 360)

    # ── Realized accuracy: forecast months that now have actuals ──────────
    def _verify(fc_df, fc_x, actuals_df):
        ai = (actuals_df.assign(_k=actuals_df['donor_month'].astype(str))
                        .set_index('_k')['total_income'])
        rows = []
        for ts, pred in zip(fc_x, fc_df['predicted_income']):
            k = pd.Timestamp(ts).strftime('%Y-%m')
            if k in ai.index:
                actual = float(ai.loc[k])
                if actual:
                    rows.append({'Month': k, 'Forecast': float(pred), 'Actual': actual,
                                 'Error %': (pred - actual) / actual * 100})
        return pd.DataFrame(rows)

    v_cur = _verify(cur_fc, cur_fc_x, actuals_now)
    v_cmp = _verify(cmp_fc, cmp_fc_x, actuals_now)

    # Difference months -- forecast by one run, since recorded as an actual
    # by the other. These are the months the verification actually bites on.
    _vm = set()
    for _vdf in (v_cur, v_cmp):
        if not _vdf.empty:
            _vm.update(_vdf['Month'])
    _verif_months = sorted(_vm)
    if _verif_months:
        _mlabels = ', '.join(pd.Timestamp(f'{m}-01').strftime('%b %Y') for m in _verif_months)
        st.markdown(
            f'<div style="border:1px solid {ui.LINE};border-left:3px solid {ui.TEAL};'
            f'background:rgba(0,137,123,0.05);padding:10px 14px;margin:2px 0 6px;'
            f'font-size:12.5px;color:{ui.TEXT};">'
            f'<b>{len(_verif_months)} forecast month{"s" if len(_verif_months) != 1 else ""} '
            f'now verifiable against later actuals:</b> {_mlabels}. '
            f'Forecast by one run, since recorded as actual fund data by the other; '
            f'the realized-accuracy tables below give the error for each.'
            f'</div>', unsafe_allow_html=True)

    _vcol_cfg = {
        'Forecast': st.column_config.NumberColumn(format='dollar'),
        'Actual': st.column_config.NumberColumn(format='dollar'),
        'Error %': st.column_config.NumberColumn(format='%.1f%%'),
    }

    vc1, vc2 = st.columns(2)
    for _dfv, _stated, _col, _who in (
        (v_cur, cur_mape, vc1, 'Run A'),
        (v_cmp, cmp_mape, vc2, 'Run B'),
    ):
        with _col:
            if _dfv.empty:
                with card(f'{_who}: realized accuracy'):
                    st.caption('This forecast is still entirely in the future relative to the '
                               'actuals on record, so there is nothing to verify yet.')
            else:
                realized = float(_dfv['Error %'].abs().mean())
                _tag = f'realized MAPE {realized:.1f}%'
                with card(f'{_who}: realized accuracy',
                          f'{len(_dfv)} forecast month(s) now have actual fund data.',
                          tag=_tag, tag_color='green' if (_stated is None or realized <= _stated + 2) else 'orange',
                          info=('"Realized MAPE" is the mean absolute error of this run\'s forecast '
                                'over the months that have actually happened since: the real-world '
                                'check on the "Validation MAPE" that was estimated at run time.')):
                    if _stated is not None:
                        st.caption(f'Stated validation MAPE at run time: {_stated:.1f}%  ·  '
                                   f'realized: {realized:.1f}%')
                    st.dataframe(_dfv, hide_index=True, width='stretch', column_config=_vcol_cfg)


# ══════════════════════════════════════════════════════════════════════════════
# RUN HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Run History':
    page_header('Run history', 'Past pipeline runs',
                'Every completed pipeline run is saved automatically. Reload any past run '
                'to explore its dashboards in full, without re-uploading data.',
                meta=f'ACTIVE: {st.session_state.viewing_run_id or "NONE"}')
    # Resets the sidebar's "new run" badge baseline to now -- see
    # count_new_runs()/mark_runs_seen() in db.py. Called on every render of
    # this page (cheap no-op UPDATE, and db.py already fails soft if
    # unreachable), not just the first, so the badge can never re-appear
    # for runs that were already visible the last time this page was open.
    if (_u := st.session_state.user) and _u.get('email'):
        mark_runs_seen(_u['email'])

    if not db_configured():
        with card():
            st.markdown(f"""
            <div style="text-align:center;padding:28px;">
                <div style="font-size:14px;font-weight:700;color:{ui.TEXT};margin-bottom:6px;">
                    Run history isn't available
                </div>
                <div style="font-size:12.5px;color:{ui.MIST};max-width:480px;margin:0 auto;">
                    No database connection is configured, so past runs aren't saved. Every
                    pipeline run still works, it just won't be reloadable later.
                </div>
            </div>
            """, unsafe_allow_html=True)
        clear_nav_overlay()
        st.stop()

    runs = list_dashboard_runs()

    if runs.empty:
        with card():
            st.markdown(f"""
            <div style="text-align:center;padding:28px;">
                <div style="font-size:14px;font-weight:700;color:{ui.TEXT};margin-bottom:6px;">
                    No runs saved yet
                </div>
                <div style="font-size:12.5px;color:{ui.MIST};max-width:480px;margin:0 auto;">
                    Run the pipeline from the Data Pipeline page. It'll be saved here
                    automatically once it finishes.
                </div>
            </div>
            """, unsafe_allow_html=True)
        clear_nav_overlay()
        st.stop()

    runs['run_at'] = pd.to_datetime(runs['run_at'])
    total_run_count = len(runs)
    runs = runs.head(10)  # only the 10 most recent runs are shown/searchable
    latest = runs.iloc[0]
    viewing_id = st.session_state.viewing_run_id

    def _fmt_duration(secs):
        if pd.isna(secs):
            return 'N/A'
        secs = int(secs)
        return f'{secs // 60}m {secs % 60}s' if secs >= 60 else f'{secs}s'

    with st.container(horizontal=True):
        kpi('Total runs', str(total_run_count),
            delta=f'Showing {len(runs)} most recent' if total_run_count > len(runs) else 'Saved to history',
            icon=':material/history:', accent=ui.TEAL, delta_color='off')
        kpi('Latest run', latest['run_at'].strftime('%d %b, %H:%M'),
            delta=latest['run_by'] or 'Unknown user', icon=':material/schedule:',
            accent=ui.BLUE, delta_color='off')
        kpi('Donors (latest)', f'{int(latest["donor_count"]):,}' if pd.notna(latest['donor_count']) else 'N/A',
            delta='In most recent run', icon=':material/groups:', accent=ui.PURPLE, delta_color='off')
        kpi('Income (latest)', f'${latest["total_income"]:,.0f}' if pd.notna(latest['total_income']) else 'N/A',
            delta='Total, most recent run', icon=':material/payments:', accent=ui.AMBER, delta_color='off')
        if 'duration_seconds' in runs.columns and runs['duration_seconds'].notna().any():
            kpi('Avg run duration', _fmt_duration(runs['duration_seconds'].mean()),
                delta=f'Latest: {_fmt_duration(latest.get("duration_seconds"))}',
                icon=':material/timer:', accent=ui.TEAL2, delta_color='off')

    chrono = runs.sort_values('run_at')
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Scatter(
            x=chrono['run_at'], y=chrono['total_income'], mode='lines+markers',
            line=dict(color=ui.TEAL, width=2), marker=dict(size=7),
        ))
        fig.update_layout(yaxis=dict(tickformat='$,.0f', title='Total income ($)'), xaxis_title='Run')
        with card('Total income by run'):
            chart(fig, 260)

    with col2:
        fig2 = go.Figure()
        mape_cols = [
            ('mape_ml', 'ML forecast', ui.TEAL), ('mape_linear', 'Linear trend', ui.PURPLE),
            ('mape_ltv', 'Donor rollup', ui.AMBER), ('mape_stockflow', 'Stock-flow', ui.BLUE),
        ]
        for col, label, colour in mape_cols:
            if col in chrono.columns and chrono[col].notna().any():
                fig2.add_trace(go.Scatter(
                    x=chrono['run_at'], y=chrono[col], mode='lines+markers', name=label,
                    line=dict(color=colour, width=2), marker=dict(size=6),
                ))
        fig2.update_layout(yaxis=dict(ticksuffix='%', title='Holdout MAPE'), xaxis_title='Run')
        with card('Model accuracy by run'):
            chart(fig2, 260)

    runs_display = runs.copy()
    # st.dataframe can't colour individual cells conditionally, so a
    # checkmark glyph stands in for the reference app's green check icon --
    # every saved run got here by completing successfully, so this is
    # always "done", never a fabricated status.
    runs_display['status'] = '✓ Completed'
    if 'duration_seconds' in runs_display.columns:
        runs_display['duration_seconds'] = runs_display['duration_seconds'].apply(_fmt_duration)
    runs_display = runs_display.rename(columns={
        'status': 'Status', 'run_id': 'Run ID', 'run_at': 'Run time', 'run_by': 'Run by',
        'duration_seconds': 'Duration',
        'donor_count': 'Donors', 'total_income': 'Total income',
        'mape_ml': 'ML MAPE', 'mape_linear': 'Linear MAPE', 'mape_ltv': 'Rollup MAPE',
        'mape_stockflow': 'Stock-flow MAPE',
    })
    col_order = ['Status', 'Run ID', 'Run time', 'Run by', 'Duration', 'Donors', 'Total income',
                 'ML MAPE', 'Linear MAPE', 'Rollup MAPE', 'Stock-flow MAPE']
    runs_display = runs_display[[c for c in col_order if c in runs_display.columns]]
    with card('All runs'):
        st.dataframe(
            runs_display, hide_index=True, width='stretch',
            column_config={
                'Status': st.column_config.TextColumn(),
                'Run time': st.column_config.DatetimeColumn(format='D MMM YYYY, HH:mm'),
                'Donors': st.column_config.NumberColumn(format='%d'),
                'Total income': st.column_config.NumberColumn(format='dollar'),
                'ML MAPE': st.column_config.NumberColumn(format='%.1f%%'),
                'Linear MAPE': st.column_config.NumberColumn(format='%.1f%%'),
                'Rollup MAPE': st.column_config.NumberColumn(format='%.1f%%'),
                'Stock-flow MAPE': st.column_config.NumberColumn(format='%.1f%%'),
            },
        )

    with card('Load or delete a run', f'Searching the {len(runs)} most recent runs'):
        # Wrapped in a form -- typing used to trigger a full-script rerun on
        # every keystroke (sidebar, page header, badge counts, everything),
        # not just this card. A form batches those into one rerun on submit
        # (Enter, or the button) instead of one per character.
        with st.form('run_search_form', border=False):
            # Input + submit button on one row, capped well short of the
            # card's full width -- reported live as too wide (first at
            # full width, then again at an even 50/50 used/unused split --
            # explicitly not that either, wanted closer to 80/20). The
            # button was ALSO wrapping to its own line below the input
            # (st.form's default vertical stacking) instead of sitting
            # beside it. search_col:btn_col:spacer is 6:2:2 -- the two
            # together are 80% of the row, the trailing spacer column is
            # never used, just reserved so the other two don't stretch to
            # fill it.
            search_col, btn_col, _spacer = st.columns([6, 2, 2], vertical_alignment='bottom')
            with search_col:
                search = st.text_input(
                    'Search by Run ID', placeholder='Search by Run ID…', icon=':material/search:',
                )
            with btn_col:
                st.form_submit_button('Search', icon=':material/search:', width='stretch')
        filtered = runs[runs['run_id'].str.contains(search.strip(), case=False, na=False)] if search.strip() else runs

        if filtered.empty:
            st.caption(f'No runs match "{search}".')
        else:
            # Run ID only, per instruction -- "currently viewing" is already
            # communicated separately below (disabled Load button + caption),
            # so it doesn't need to be baked into the option text too.
            run_labels = {row['run_id']: row['run_id'] for _, row in filtered.iterrows()}
            # Same half-width treatment as the search row above -- this
            # was stretching to the card's full width.
            select_col, _spacer = st.columns([1, 1])
            with select_col:
                picked_id = st.selectbox(
                    'Choose a run', options=list(run_labels.keys()),
                    format_func=lambda rid: run_labels[rid], label_visibility='collapsed',
                )

            # Equal-width Load/Delete -- was [3, 1] (Load much wider than
            # Delete), reported as should be the same size.
            c1, c2 = st.columns(2)
            with c1:
                load_disabled = picked_id == viewing_id
                if st.button('Load this run for full analysis', icon=':material/bolt:',
                             type='primary', width='stretch', disabled=load_disabled):
                    with st.spinner('Loading run…'):
                        picked_state, picked_at = load_dashboard_run(picked_id)
                    if picked_state:
                        restore_dashboard_state(picked_state)
                        st.session_state.viewing_run_id = picked_id
                        st.session_state.data_source    = 'cached'
                        st.session_state.data_loaded_at = picked_at
                        st.session_state.page = 'Overview'
                        st.rerun()
                    else:
                        st.error('Could not load that run. It may have been deleted, or the '
                                  'database is unreachable.')
                if load_disabled:
                    st.caption('This is the run currently shown across the dashboards.')

            with c2:
                # Administrator-only -- permanent deletion, same reasoning
                # as the Data Pipeline page-access gate above. Not a
                # st.stop() here -- this sits inside a `with c2:` column
                # context, and st.stop() halts the ENTIRE script, not
                # just this column, which would silently truncate
                # anything a later edit adds below this block. A plain
                # if/else keeps the gate scoped to just this column.
                # Administrator-or-above.
                if (st.session_state.user or {}).get('role') not in ('Administrator', 'Super Admin'):
                    st.caption('Only Administrators can delete runs.')
                else:
                    confirm_key = f'confirm_delete_{picked_id}'
                    if st.session_state.get(confirm_key):
                        if st.button('Confirm delete', icon=':material/delete_forever:',
                                      width='stretch'):
                            delete_dashboard_run(picked_id)
                            st.session_state.pop(confirm_key, None)
                            if picked_id == viewing_id:
                                st.session_state.pipeline_run = False
                                st.session_state.viewing_run_id = None
                                st.session_state.data_source = None
                            st.rerun()
                        if st.button('Cancel', width='stretch'):
                            st.session_state.pop(confirm_key, None)
                            st.rerun()
                    else:
                        if st.button('Delete this run', icon=':material/delete:', width='stretch'):
                            st.session_state[confirm_key] = True
                            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# USERS
# ══════════════════════════════════════════════════════════════════════════════
# One unified account-management page -- used to be two separate pages
# (Access Requests for Google sign-in, Local Accounts for email+password),
# split by which table each login method's account lived in. Merged per
# explicit request: a Super Admin manages every account from one place
# regardless of how that person signs in, with full revoke/promote/demote
# power over both, including restoring a Google user's access after
# revoking it (see list_denied_users() in db.py -- previously a revoked
# Google user simply vanished from every admin view with no way back
# except the affected person re-requesting themselves).
elif page == 'Users':
    render_action_button_css()
    page_header('Administration', 'Users',
                'Create and manage every sign-in account; Google and email+password alike. '
                'Approve or deny new Google requests, change anyone\'s role, and revoke or '
                'restore access. A signed-in user manages their own password and two-factor '
                'authentication from their Profile page (account menu in the page header).',
                meta=page_meta)

    # Super-Admin-only -- grants/revokes/promotes account access for
    # BOTH login methods. Not just Administrator any more: this page can
    # hand out Administrator (and Super Admin) itself, so an ordinary
    # Administrator having access to it would let them promote themselves
    # or anyone else. The sidebar already hides this page's nav entry
    # accordingly (see render_sidebar()'s is_super_admin filter); this is
    # the defense-in-depth backstop in case session_state.page ever ends
    # up here some other way.
    if (st.session_state.user or {}).get('role') != 'Super Admin':
        with card():
            st.markdown(f"""
            <div style="text-align:center;padding:28px;">
                <div style="font-size:14px;font-weight:700;color:{ui.TEXT};margin-bottom:6px;">
                    Super Admins only
                </div>
                <div style="font-size:12.5px;color:{ui.MIST};max-width:480px;margin:0 auto;">
                    Creating or managing user accounts is restricted to Super Admins.
                </div>
            </div>
            """, unsafe_allow_html=True)
        clear_nav_overlay()
        st.stop()

    if not db_configured():
        with card():
            st.markdown(f"""
            <div style="text-align:center;padding:28px;">
                <div style="font-size:14px;font-weight:700;color:{ui.TEXT};margin-bottom:6px;">
                    User management isn't available
                </div>
                <div style="font-size:12.5px;color:{ui.MIST};max-width:480px;margin:0 auto;">
                    No database connection is configured, so there's no account store to manage.
                    Anyone on an allowed Google domain can sign in directly until one is set up
                    (see handle_google_redirect() in auth.py).
                </div>
            </div>
            """, unsafe_allow_html=True)
        clear_nav_overlay()
        st.stop()

    current_email = (st.session_state.user or {}).get('email', '')

    # ── Pending Google requests ──
    pending = list_pending_requests()

    if pending.empty:
        with card():
            st.markdown(f"""
            <div style="text-align:center;padding:28px;">
                <div style="font-size:14px;font-weight:700;color:{ui.TEXT};margin-bottom:6px;">
                    No pending requests
                </div>
                <div style="font-size:12.5px;color:{ui.MIST};max-width:480px;margin:0 auto;">
                    New Google sign-in requests from allowed domains will show up here.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        pending['requested_at'] = pd.to_datetime(pending['requested_at'])
        with card(f'{len(pending)} pending', 'Google sign-in requests; oldest first'):
            for i, row in pending.iterrows():
                rc1, rc2, rc3 = st.columns([3, 2, 2], vertical_alignment='center')
                with rc1:
                    st.markdown(f"**{row['name'] or row['email']}**")
                    st.caption(row['email'])
                with rc2:
                    st.caption(f"Requested {row['requested_at'].strftime('%d %b %Y, %H:%M')}")
                with rc3:
                    ac1, ac2 = st.columns(2)
                    with ac1:
                        if st.button('Approve', key=f"approve_{row['email']}", icon=':material/check:',
                                     type='primary', width='stretch'):
                            decide_access_request(
                                row['email'], approve=True,
                                decided_by=(st.session_state.user or {}).get('email', ''),
                            )
                            st.rerun()
                    with ac2:
                        if st.button('Deny', key=f"deny_{row['email']}", icon=':material/close:',
                                     width='stretch'):
                            decide_access_request(
                                row['email'], approve=False,
                                decided_by=(st.session_state.user or {}).get('email', ''),
                            )
                            st.rerun()
                if i != pending.index[-1]:
                    st.markdown(f'<div style="height:1px;background:{ui.LINE};margin:10px 0;"></div>',
                                unsafe_allow_html=True)

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

    # ── All accounts -- Google and local, one combined list, per explicit
    # request (used to be two separate cards split by login method).
    # Sorted by most recent activity (decided_at for Google, created_at
    # for local) so the newest change to anyone's access sits at the top
    # regardless of which table it actually lives in -- the split is
    # invisible here, surfaced only via the small type pill next to each
    # row's role, and in which action buttons that row gets (Role/Revoke
    # for Google, Manage/Delete for local -- the two tables support
    # different operations, so the actions can't be identical, only the
    # listing is unified). ──
    approved = list_approved_users()
    accounts = list_local_accounts()

    # Built as plain dicts rather than a pandas assign()/concat() -- both
    # list_approved_users() and list_local_accounts() fall back to a bare,
    # columnless pd.DataFrame() on a query error (see their own docstrings
    # in db.py), and indexing a specific column out of THAT would raise,
    # not just render an empty state. A plain Python list sidesteps that
    # entirely: iterrows() over a genuinely columnless DataFrame just
    # yields nothing, same as an ordinary empty result.
    combined_rows = []
    for _, r in approved.iterrows():
        combined_rows.append({
            'email': r['email'], 'name': r['name'], 'role': r['role'], 'kind': 'Google',
            'date': pd.to_datetime(r['decided_at']) if pd.notna(r['decided_at']) else None,
        })
    for _, r in accounts.iterrows():
        combined_rows.append({
            'email': r['email'], 'name': r['name'], 'role': r['role'], 'kind': 'Local',
            'date': pd.to_datetime(r['created_at']) if pd.notna(r['created_at']) else None,
        })
    combined_rows.sort(key=lambda x: x['date'] or pd.Timestamp.min, reverse=True)
    all_accounts = pd.DataFrame(combined_rows, columns=['email', 'name', 'role', 'kind', 'date'])
    all_accounts = all_accounts.rename(columns={'kind': '_kind', 'date': '_date'})

    if all_accounts.empty:
        with card():
            st.markdown(f"""
            <div style="text-align:center;padding:28px;">
                <div style="font-size:14px;font-weight:700;color:{ui.TEXT};margin-bottom:6px;">
                    No accounts yet
                </div>
                <div style="font-size:12.5px;color:{ui.MIST};max-width:480px;margin:0 auto;">
                    Approved Google sign-ins and local accounts you create below will show
                    up here together, with controls to change role, reset password, revoke,
                    or delete.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        with card(f'{len(all_accounts)} accounts',
                   'Google and local sign-ins; change role, reset password, revoke, or delete'):
            last_idx = len(all_accounts) - 1
            for i, row in all_accounts.iterrows():
                is_google = row['_kind'] == 'Google'
                is_self = row['email'] == current_email
                uc1, uc2, uc3 = st.columns([3, 2, 3], vertical_alignment='center')
                with uc1:
                    label = row['name'] or row['email']
                    if is_self:
                        label += ' (you)'
                    st.markdown(f"**{label}**")
                    st.caption(row['email'])
                with uc2:
                    pill_class = 'sf-pill-blue' if is_google else 'sf-pill-gray'
                    st.markdown(f'<span class="sf-pill {pill_class}">{row["_kind"]}</span>',
                                unsafe_allow_html=True)
                    verb = 'since' if is_google else 'created'
                    detail = (f"{row['role']} · {verb} {row['_date'].strftime('%d %b %Y')}"
                              if pd.notna(row['_date']) else row['role'])
                    st.caption(detail)
                with uc3:
                    vc1, vc2 = st.columns(2)
                    if is_google:
                        with vc1:
                            with st.popover('Role', icon=':material/settings:', width='stretch',
                                              disabled=is_self):
                                role_options = ['Analyst', 'Administrator', 'Super Admin']
                                new_role_pick = st.selectbox(
                                    'Role', role_options,
                                    index=role_options.index(row['role']) if row['role'] in role_options else 0,
                                    key=f"user_role_pick_{row['email']}",
                                )
                                if new_role_pick != row['role']:
                                    if st.button('Update role', key=f"user_role_update_{row['email']}",
                                                 icon=':material/check:', width='stretch'):
                                        if update_user_role(row['email'], new_role_pick):
                                            st.success('Role updated.', icon=':material/check_circle:')
                                            st.rerun()
                                        else:
                                            st.error('Could not update that role.', icon=':material/error:')
                        with vc2:
                            # Two-step confirm -- revoking access is
                            # destructive (see revoke_user_access()'s own
                            # docstring for why this sets status='denied'
                            # rather than deleting the row), not a single
                            # misclick away. Can't revoke your own access
                            # from here -- same reasoning as not being
                            # able to change your own role, avoids locking
                            # yourself out by accident.
                            confirm_key = f"confirm_revoke_{row['email']}"
                            if is_self:
                                st.button('Revoke', key=f"revoke_btn_{row['email']}",
                                          icon=':material/block:', width='stretch', disabled=True)
                            elif st.session_state.get(confirm_key):
                                if st.button('Confirm', key=f"confirm_revoke_btn_{row['email']}",
                                             icon=':material/block:', width='stretch'):
                                    revoke_user_access(row['email'], decided_by=current_email)
                                    st.session_state.pop(confirm_key, None)
                                    st.rerun()
                            else:
                                if st.button('Revoke', key=f"revoke_btn_{row['email']}",
                                             icon=':material/block:', width='stretch'):
                                    st.session_state[confirm_key] = True
                                    st.rerun()
                    else:
                        with vc1:
                            # Manage (password reset + role change) is
                            # disabled for your own row, same as Google's
                            # Role popover above -- role change is the
                            # risk (a Super Admin could demote or delete
                            # their own only-Super-Admin account and lock
                            # everyone out, self included); password reset
                            # is bundled into the same popover, so it's
                            # disabled along with it, not because it's
                            # risky, but a self-service "Change password"
                            # already exists in the account menu (page
                            # header) for exactly that case.
                            with st.popover('Manage', icon=':material/settings:', width='stretch',
                                              disabled=is_self):
                                with st.form(f"reset_pw_form_{row['email']}", border=False):
                                    reset_pw = st.text_input(
                                        'New password', type='password', key=f"reset_pw_input_{row['email']}",
                                    )
                                    reset_submitted = st.form_submit_button('Set new password',
                                                                              icon=':material/check:', width='stretch')
                                if reset_submitted:
                                    if len(reset_pw) < 8:
                                        st.error('Password must be at least 8 characters.', icon=':material/error:')
                                    elif update_local_account_password(row['email'], reset_pw):
                                        st.success('Password reset.', icon=':material/check_circle:')
                                    else:
                                        st.error('Could not reset that password.', icon=':material/error:')

                                st.markdown(f'<div style="height:1px;background:{ui.LINE};margin:10px 0;"></div>',
                                            unsafe_allow_html=True)

                                role_options = ['Analyst', 'Administrator', 'Super Admin']
                                new_role_pick = st.selectbox(
                                    'Role', role_options, index=role_options.index(row['role'])
                                    if row['role'] in role_options else 0,
                                    key=f"role_pick_{row['email']}",
                                )
                                if new_role_pick != row['role']:
                                    if st.button('Update role', key=f"role_update_{row['email']}",
                                                 icon=':material/check:', width='stretch'):
                                        if update_local_account_role(row['email'], new_role_pick):
                                            st.success('Role updated.', icon=':material/check_circle:')
                                            st.rerun()
                                        else:
                                            st.error('Could not update that role.', icon=':material/error:')

                                # Recovery path for someone locked out after
                                # losing their authenticator device -- a TOTP
                                # secret is never displayed or exported once
                                # set (see db.py's set_totp_secret()), so
                                # there's no way for them to get back in
                                # without either this or a whole new account.
                                # Only shown at all when totp_enabled is
                                # actually true for this row -- no button to
                                # click for the (default) case where 2FA was
                                # never turned on.
                                if row.get('totp_enabled'):
                                    st.markdown(f'<div style="height:1px;background:{ui.LINE};margin:10px 0;"></div>',
                                                unsafe_allow_html=True)
                                    st.caption('Two-factor authentication is enabled for this account.')
                                    if st.button('Disable their two-factor authentication',
                                                 key=f"totp_admin_disable_{row['email']}",
                                                 icon=':material/remove_moderator:', width='stretch'):
                                        if clear_totp_secret(row['email']):
                                            st.success('Two-factor authentication disabled.',
                                                       icon=':material/check_circle:')
                                            st.rerun()
                                        else:
                                            st.error('Could not disable two-factor authentication right now.',
                                                      icon=':material/error:')
                        with vc2:
                            # Same two-step confirm pattern as Run
                            # History's delete-run action -- a destructive
                            # action, not a single misclick away. Can't
                            # delete your own account from here -- same
                            # reasoning as Google's Revoke being disabled
                            # for self above: deleting the account you're
                            # currently signed in as would sign you out
                            # mid-session with no way back in as that
                            # identity, and if it were the only Super
                            # Admin account, no one left could undo it.
                            confirm_key = f"confirm_delete_local_{row['email']}"
                            if is_self:
                                st.button('Delete', key=f"delete_btn_{row['email']}",
                                          icon=':material/delete:', width='stretch', disabled=True)
                            elif st.session_state.get(confirm_key):
                                if st.button('Confirm', key=f"confirm_delete_btn_{row['email']}",
                                             icon=':material/delete_forever:', width='stretch'):
                                    delete_local_account(row['email'])
                                    st.session_state.pop(confirm_key, None)
                                    st.rerun()
                            else:
                                if st.button('Delete', key=f"delete_btn_{row['email']}",
                                             icon=':material/delete:', width='stretch'):
                                    st.session_state[confirm_key] = True
                                    st.rerun()
                if i != last_idx:
                    st.markdown(f'<div style="height:1px;background:{ui.LINE};margin:10px 0;"></div>',
                                unsafe_allow_html=True)

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

    # ── Create local account ──
    with card('Create local account', 'A new email + password sign-in'):
        # The form key carries a nonce that's bumped on a SUCCESSFUL
        # create -- a new key makes Streamlit treat it as a fresh form, so
        # every field resets to its default. Only on success, so a
        # validation error (bad email, short password) keeps what was
        # typed instead of wiping it.
        _cf_nonce = st.session_state.get('_create_acct_nonce', 0)
        with st.form(f'create_local_account_form_{_cf_nonce}', border=False):
            cc1, cc2 = st.columns(2)
            with cc1:
                new_email = st.text_input('Email', placeholder='name@strokefoundation.org.au')
                new_name = st.text_input('Name')
            with cc2:
                new_role = st.selectbox('Role', ['Analyst', 'Administrator', 'Super Admin'])
                new_password = st.text_input('Password', type='password',
                                              help='At least 8 characters. Share this with them directly, '
                                                   'not over an insecure channel; they can change it '
                                                   'themselves afterward from the account menu.')
            create_submitted = st.form_submit_button('Create account', icon=':material/person_add:',
                                                       type='primary')
        if create_submitted:
            clean_email = new_email.strip().lower()
            clean_name = new_name.strip()
            if not clean_email or '@' not in clean_email:
                st.error('Enter a valid email address.', icon=':material/error:')
            elif not clean_name:
                st.error('Enter a name.', icon=':material/error:')
            elif len(new_password) < 8:
                st.error('Password must be at least 8 characters.', icon=':material/error:')
            elif create_local_account(clean_email, clean_name, new_role, new_password):
                st.success(f'Account created for {clean_email}.', icon=':material/check_circle:')
                st.session_state['_create_acct_nonce'] = _cf_nonce + 1
                st.rerun()
            else:
                st.error('Could not create that account; that email may already have one.',
                          icon=':material/error:')

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

    # ── Revoked Google access -- restore directly, no dependency on the
    # revoked person re-requesting themselves (see list_denied_users()'s
    # own docstring in db.py for why this section exists at all). Kept
    # separate from the unified list above rather than shown inline with
    # a "revoked" tag, so a Super Admin skimming "who currently has
    # access" isn't scanning past accounts that don't. ──
    denied = list_denied_users()
    if not denied.empty:
        with card(f'{len(denied)} revoked', 'Google sign-ins with revoked access; restore if needed'):
            for i, row in denied.iterrows():
                dc1, dc2, dc3 = st.columns([3, 2, 2], vertical_alignment='center')
                with dc1:
                    st.markdown(f"**{row['name'] or row['email']}**")
                    st.caption(row['email'])
                with dc2:
                    decided = pd.to_datetime(row['decided_at']) if pd.notna(row['decided_at']) else None
                    detail = f"Was {row['role']} · revoked {decided.strftime('%d %b %Y')}" if decided else f"Was {row['role']}"
                    st.caption(detail)
                with dc3:
                    # Not destructive (unlike Revoke above) -- restoring
                    # someone's access is easy to undo again with another
                    # click, so this skips the two-step confirm pattern.
                    # Restores at whatever role they held before being
                    # revoked (decide_access_request() only ever touches
                    # status, never role -- see its own docstring).
                    if st.button('Restore access', key=f"restore_{row['email']}",
                                 icon=':material/how_to_reg:', width='stretch'):
                        decide_access_request(row['email'], approve=True, decided_by=current_email)
                        st.rerun()
                if i != denied.index[-1]:
                    st.markdown(f'<div style="height:1px;background:{ui.LINE};margin:10px 0;"></div>',
                                unsafe_allow_html=True)


elif page == 'Profile':
    # Reached only from the account menu (page_header(), ui.py) -- deliberately
    # not in NAV_SECTIONS/the sidebar, since this is account-scoped, not a
    # dashboard. Every signed-in user, any role, any sign-in method, lands
    # here the same way; what they can actually DO on the page is what
    # varies (see the auth_method/role checks below), not who can open it.
    #
    # Change password and Two-factor authentication used to be expanders in
    # that same account-menu popover -- moved here because they'd outgrown
    # a popover (the 2FA QR-code setup screen especially), and because the
    # profile-details editing below (Super Admin only) never fit there at
    # all. See ui.py's page_header() docstring/comments for that history.
    render_action_button_css()
    _user = st.session_state.user or {}
    _auth_method = st.session_state.get('auth_method')
    _is_local = _auth_method == 'password'
    _email = _user.get('email', '')
    _account = get_local_account(_email) if _is_local else None

    page_header('Account', 'Profile', 'Your account details, sign-in security, and password.')

    with card(title='Account details'):
        col_avatar, col_id = st.columns([1, 5], vertical_alignment='center')
        with col_avatar:
            st.markdown(
                f'<div class="sf-avatar" style="width:56px;height:56px;font-size:19px;">'
                f'{initials(_user.get("name", ""))}</div>',
                unsafe_allow_html=True,
            )
        with col_id:
            st.markdown(f"""
            <div style="font-size:16px;font-weight:700;color:{ui.TEXT};">{_user.get('name', '')}</div>
            <div style="font-size:12.5px;color:{ui.SLATE};">{_email}</div>
            """, unsafe_allow_html=True)
        st.markdown('<div style="height:1px;background:{0};margin:14px 0 10px;"></div>'.format(ui.LINE),
                    unsafe_allow_html=True)
        role_color = {'Super Admin': 'amber', 'Administrator': 'blue', 'Analyst': 'gray'}.get(_user.get('role'), 'gray')
        method_pill = pill('Local sign-in', 'green') if _is_local else pill('Google sign-in', 'blue')
        member_since = (
            f" · Member since {_account['created_at']:%d %b %Y}"
            if _account and _account.get('created_at') else ''
        )
        st.markdown(
            f'{pill(_user.get("role", ""), role_color)} {method_pill}'
            f'<span style="font-size:11.5px;color:{ui.MIST};margin-left:8px;">{member_since}</span>',
            unsafe_allow_html=True,
        )

    # Profile-details editing (name/email) -- Super Admin, local sign-in
    # only. Google's own name/email is Google's identity, not this app's
    # to change (see auth.py's handle_google_redirect() -- it's re-read
    # from the OAuth payload on every sign-in, there's nothing here that
    # editing it would even persist against); Analyst/Administrator local
    # accounts are admin-provisioned the same way passwords are, per the
    # user's own explicit spec for this page -- only a Super Admin edits
    # their own name/email, everyone else just sees it.
    #
    # Side by side, not stacked -- these three (Profile details, Change
    # password, Two-factor authentication) used to be full-width cards
    # top to bottom, which left every "start" button (Edit/Change password/
    # Set up 2FA, all width='stretch') stretched the full page width for
    # no reason -- reported live as looking wrong. st.columns() narrows
    # each card to a third of the page, so the buttons inside stretch to
    # something sane instead. Two columns, not three, on the Google branch
    # below -- there's no separate password/2FA card to give a third
    # column to there, just the one combined "Sign-in security" notice.
    #
    # Equal height AT REST, not tied together while expanding -- reported
    # live, twice now: a first attempt stretched every card to match its
    # tallest sibling (flexbox align-items:stretch, Streamlit's own
    # default for a row of columns), which does line the three up at
    # rest, but also means expanding just ONE of them (e.g. clicking "Set
    # up 2FA", which grows that card to fit the QR code) drags the OTHER
    # TWO taller right along with it, leaving dead space in cards nobody
    # touched -- not what was wanted; only the card actually being
    # interacted with should grow. align-items:flex-start turns that
    # cross-column stretch off entirely, and a shared min-height (matched
    # to Profile details' own at-rest height, the tallest of the three
    # before any of them are expanded) is what lines them up instead --
    # a floor each card can grow past on its own, never a ceiling tying
    # it to its siblings. This container's key scopes both rules to just
    # this row (column/card layouts elsewhere on other pages are
    # deliberately left at their natural per-card height).
    with st.container(key='profile_settings_row'):
        st.html("""
        <style>
        /* A keyed st.container -- even without border=True -- picks up the
        same overflow="visible" attribute a real bordered card() does, so
        without this it would wrap the whole row in an outer card box of
        its own (same un-styling as .st-key-page_header_row elsewhere).
        Plain ".st-key-profile_settings_row" lost that fight, though --
        confirmed live via getMatchedCSSRules that Streamlit's own global
        "[data-testid='stVerticalBlock'][overflow='visible']" rule (two
        attribute selectors) outranks a single class selector on
        specificity, !important on both sides notwithstanding, so the
        outer box was still showing. Repeating that same attribute
        selector ON this class (three selectors together) outranks it
        back and the row actually goes transparent now. */
        [data-testid="stVerticalBlock"][overflow="visible"].st-key-profile_settings_row {
            background: transparent !important; border: none !important; box-shadow: none !important;
        }
        .st-key-profile_settings_row [data-testid="stHorizontalBlock"] { align-items: flex-start !important; }
        .st-key-profile_settings_row [data-testid="stVerticalBlock"][overflow="visible"] {
            min-height: 227px !important;
        }
        /* Pins each card's own trigger button (Edit / Change password /
        Set up 2FA) to the bottom of its card rather than wherever the
        content above it happens to end -- without this, Profile details'
        extra Name/Email lines push ITS button noticeably lower than
        Change password's and 2FA's shorter captions leave theirs sitting
        near the top, so the three buttons visibly don't line up even
        though the cards themselves are the same height. The inner
        content column (not the outer bordered card -- see the nested
        stVerticalBlock structure the other two rules above also rely on)
        is itself a flex column, so margin-top:auto on the last element
        soaks up all the leftover space above it, flush against the
        bottom every time. Scoped to ":has(.stButton)" so it only grabs
        cards whose last element actually IS a button -- the read-only
        Profile details variant (Analyst/Administrator/Google) ends on a
        plain text block instead, and that one is left in its natural
        top-packed flow. */
        .st-key-profile_settings_row [data-testid="stVerticalBlock"][overflow="visible"]
            > [data-testid="stElementContainer"]:last-child:has(.stButton) {
            margin-top: auto !important;
        }
        </style>
        """)
        if _is_local:
            col_profile, col_pw, col_2fa = st.columns(3)
            with col_profile:
                if _user.get('role') == 'Super Admin':
                    _render_profile_details_card(_email)
                else:
                    with card(title='Profile details'):
                        st.caption('Only a Super Admin can change your name or email. Contact one if these '
                                   'need updating.')
                        st.markdown(f"""
                        <div><div class="sf-eyebrow" style="margin-bottom:2px;">Name</div>
                            <div style="font-size:13px;font-weight:600;color:{ui.TEXT};">{_user.get('name', '')}</div></div>
                        <div style="margin-top:10px;"><div class="sf-eyebrow" style="margin-bottom:2px;">Email</div>
                            <div style="font-size:13px;font-weight:600;color:{ui.TEXT};">{_email}</div></div>
                        """, unsafe_allow_html=True)
            # Password and two-factor auth are both local-sign-in-only -- a
            # Google account has no password in this app at all, and its own
            # 2FA (if any) is Google's, not ours (same reasoning as the
            # docstring history above).
            with col_pw:
                _render_change_password_card(_email)
            with col_2fa:
                _render_2fa_card(_email)
        else:
            col_profile, col_sec = st.columns(2)
            with col_profile:
                with card(title='Profile details'):
                    st.caption('Your name and email come from Google and are managed in your Google '
                               'account, not here.')
                    st.markdown(f"""
                    <div><div class="sf-eyebrow" style="margin-bottom:2px;">Name</div>
                        <div style="font-size:13px;font-weight:600;color:{ui.TEXT};">{_user.get('name', '')}</div></div>
                    <div style="margin-top:10px;"><div class="sf-eyebrow" style="margin-bottom:2px;">Email</div>
                        <div style="font-size:13px;font-weight:600;color:{ui.TEXT};">{_email}</div></div>
                    """, unsafe_allow_html=True)
            with col_sec:
                with card(title='Sign-in security'):
                    st.caption('Password and two-factor authentication are managed in your Google account, '
                               'not here; your access to this app is tied to your Google sign-in.')


# Matches the placeholder opened above (search _signing_in_overlay_ph) --
# this is the actual end of the "Signing in" transition: everything this
# run was going to render has now been queued. Clearing the overlay
# here, not right after the post-login restore step further up, is what
# keeps the browser from ever showing a blank/half-built page in between.
if _signing_in_overlay_ph is not None:
    _signing_in_overlay_ph.empty()
    st.session_state.is_signing_in = False

# Matches the placeholder opened right after render_sidebar() (search
# _nav_overlay_ph) -- same reasoning as the sign-in overlay just above:
# clearing only here, after the destination page's entire render has
# been queued, is what guarantees the browser never shows a half-built
# page, and never the previous page's stale content either (the actual
# bug this overlay replaced a plain
# CSS blur to fix -- see render_nav_transition_overlay()'s docstring).
# This is the NORMAL path -- a page that ran all the way through without
# hitting an early st.stop() (a permission gate, an empty-state screen).
# Those call clear_nav_overlay() themselves, right before their own
# st.stop(), since script execution never reaches this point on that
# path at all; both paths share the exact same function so neither one
# can drift out of sync with the other (see its docstring for the
# reasoning behind the sleep, and why this was a live, reported bug on
# every page with an early-exit path -- Users and Data Pipeline's
# permission gates, Run History and every dashboard's empty-data state).
clear_nav_overlay()

