# Code Flow — How Everything Fits Together

This document explains how the modules in this app connect and how data
actually moves through it, from a CSV upload to a chart on screen. For what
each individual dashboard *means*, see `dashboard.md`. For the bigger design
decisions and *why* things are built this way, see `architecture.md`.

## 1. Module map

| File | Responsibility |
|---|---|
| `app.py` | The entry point. Page routing, the pipeline orchestration function, the CSV export builder, and every page's actual content (one big script, one `if/elif` chain per page). |
| `ui.py` | The design system: `inject_global_css()` (the whole app's CSS, injected once per run), reusable components (`card()`, `kpi()`, `chart()`, `page_header()`, `stage_row()`, ...), the sidebar (`render_sidebar()`), and the light/dark palette. |
| `auth.py` | Login (Google OAuth + a hardcoded password fallback), session bootstrapping (`init_session_state()`), sign-out, and the access-request screens (request/pending/denied). |
| `db.py` | All Postgres (Supabase) access: saving/loading past pipeline runs (`dashboard_runs` table) and the access-request queue (`users` table). Every function fails soft — no DB configured, or a connection error, just means "not available," never a crash. |
| `branding.py` | Logo asset loading (`logo.png`, the white-wordmark variant for the navy sidebar, and the favicon crop), as base64 data URIs for embedding in raw HTML. Deliberately has zero imports from `ui.py`/`auth.py` to avoid a circular import between those two. |
| `pipeline/clean.py` | Reads and normalizes the four raw CSVs. `clean_payments()` is the one that matters for memory: it streams the file in 500k-row chunks and aggregates each chunk immediately, rather than holding the whole raw table in memory. |
| `pipeline/build_master.py` | Joins the four cleaned tables into one donor-month panel (`master`) — the single source of truth every model and every page's stats are computed from. |
| `pipeline/forecast.py` | The linear-trend baseline, plus the aggregation helpers (`get_monthly_actuals`, `get_supplier_breakdown`, `get_campaign_roi`, `get_retention_by_segment`) that the dashboard pages read from directly. |
| `pipeline/ml_forecast.py` | The gradient-boosting forecast model (direct multi-step, not recursive). |
| `pipeline/ltv_model.py` | Pareto/NBD + Gamma-Gamma donor lifetime value model (via the `lifetimes` package). |
| `pipeline/stock_flow_forecast.py` + `walkforward.py` + `component_forecast.py` | The stock-flow forecast: `Income(t) = Active(t) × AvgGift(t)`, `Active(t) = Active(t-1) + Recruits(t) - Lapsed(t)`. Three independently-forecast components (SARIMA recruits, cohort-hazard lapse, trend gift size) rolled forward through that identity, never forecasting income directly. |
| `pipeline/sbg_bgnbd_model.py`, `pipeline/gift_waterfall.py` | A fifth forecasting method (sBG/BG-NBD) and an MRR-style monthly $ bridge. **Both are currently unused** — not imported or called anywhere in `app.py`. Their old `session_state`/DB schema keys are kept around (always `None`) so a saved run from when they were active still loads without error. |

## 2. The execution model — everything is a rerun

Streamlit has no persistent server-side objects between interactions: every
click, upload, or navigation **reruns the entire `app.py` script from the
top**. The only thing that survives between reruns is `st.session_state` — a
plain dict-like object. This is the single most important thing to understand
about how this codebase works: there is no "controller" holding state, no
long-lived Python objects — `session_state` *is* the shared state bus that
every function reads from and writes to.

Two consequences that show up everywhere in this code:

- **Expensive computation gets cached** (`@st.cache_data`, `@st.cache_resource`
  in `app.py`/`db.py`), keyed on its inputs, so a rerun that doesn't actually
  change those inputs doesn't refit a model or reopen a DB connection.
- **A widget's own `key=` isn't reliable across a `st.rerun()` that happens
  mid-script**, because Streamlit clears a keyed widget's `session_state`
  entry if that widget wasn't reached in the last completed run. The fix used
  throughout this app (dark mode, the loading-overlay flag, the cached-run
  banner flag): keep the *real* value in a plain `session_state` key that
  nothing ever garbage-collects, and feed it into the widget via `value=`
  rather than trusting the widget's own key to persist.

## 3. Startup sequence (every single run of `app.py`)

1. `_bootstrap_secrets_from_env()` — writes a `secrets.toml` from plain
   environment variables if one doesn't already exist (for hosts like
   Hugging Face Spaces that don't support Streamlit's native secrets file).
2. `init_session_state()` (`auth.py`) — ensures every session_state key the
   app expects exists (`None` by default), so nothing downstream needs to
   guard against a missing key on a brand new session.
3. `handle_google_redirect()` (`auth.py`) — bridges Streamlit's own OIDC
   cookie (`st.user`) into this app's `session_state.authenticated`/`user`.
   This is also where the domain allow-list and the access-request queue are
   enforced — see §6 below.
4. `st.set_page_config(...)` — page title/icon/layout. Layout is
   `'centered'` (narrow, for the login page) until authenticated, then
   `'wide'`.
5. If not authenticated → `render_login()` and `st.stop()`. Nothing below
   this line ever runs for a signed-out user.
6. **Auto-load the most recent run** — if this session hasn't run the
   pipeline yet and a DB is configured, the latest saved run is loaded
   automatically via `restore_dashboard_state()`, so a returning user lands
   on a populated dashboard instead of an empty one.
7. `page = st.session_state.page` is read, the per-page CSV export is built
   (Administrator-only — see §6), then `inject_global_css()` and
   `render_sidebar()` run.
8. The big `if page == 'Data Pipeline': ... elif page == 'Overview': ...`
   chain in `app.py` renders exactly one page's content.

## 4. The data pipeline flow (uploading and running)

```
4 CSVs uploaded
      │
      ▼
pipeline/clean.py        tidy_columns(), per-file cleaning, date parsing
      │                  clean_payments() aggregates to a donor-month panel
      │                  IN CHUNKS (this is the OOM fix — see architecture.md)
      ▼
pipeline/build_master.py  joins payments panel + recurring + campaigns +
      │                    contacts into one `master` DataFrame
      ▼
run_pipeline_models()     (app.py) fits, in order:
      │                     1. Linear trend baseline
      │                     2. ML (gradient boosting) forecast
      │                     3. Stock-flow forecast (SARIMA + cohort survival)
      │                   ...and computes every page-level aggregate
      │                   (supplier/campaign summaries, retention curves)
      ▼
st.session_state          every model's output + every aggregate lands here
      │
      ▼
build_dashboard_state()   (app.py) extracts the DISPLAY-level subset of
      │                    session_state into one JSON-able dict
      ▼
db.save_dashboard_run()   gzip + insert as a new row in `dashboard_runs`
```

Donor LTV (Pareto/NBD + Gamma-Gamma) is **not** in that list — it used to be,
but it was the slowest stage (~5 min) and only the Donor Lifetime Value page
reads its output, so it's fit lazily instead, cached, the first time that
specific page is opened (`cached_ltv_fit()` in `app.py`).

Reloading a past run skips all of this — `restore_dashboard_state()` (the
inverse of `build_dashboard_state()`) populates the exact same
`session_state` keys directly from the saved JSON blob, so every page
behaves identically whether its data came from a live run or history.

## 5. Page routing flow

```
ui.NAV_SECTIONS            the sidebar's nav structure (section → items)
      │
      ▼
render_sidebar()            renders one st.button per item; a click sets
      │                     session_state.page and calls st.rerun()
      ▼
app.py's if/elif chain      exactly one branch matches session_state.page
      │
      ▼
page_header(eyebrow,        every page's first call: title block on the
  title, sub, meta=...)     left, the account cluster (avatar+name popover,
      │                     Export CSV) on the right — see ui.py's own
      │                     docstring for why this has to run BEFORE any
      │                     page's `empty_state()` guard, not after.
      ▼
page body                   reads whatever it needs straight out of
                             session_state (never recomputes from `master`
                             on every rerun — `master` itself isn't even
                             kept around after a restored/cached run).
```

## 6. Auth flow

Two sign-in methods, both landing in the same `session_state.user` shape
(`{'email', 'name', 'role'}`):

- **Password** (`render_login()`'s form) — checked against the hardcoded
  `USERS` dict in `auth.py`. Demo/dev credentials only.
- **Google OAuth** (`st.login()` / `handle_google_redirect()`) — gated on:
  1. Domain allow-list (`ALLOWED_GOOGLE_DOMAINS`) or an individual exception
     (`GOOGLE_EXTRA_ALLOWED_EMAILS`) — same as before, a hard block if you
     fail this.
  2. **For everyone except `GOOGLE_ADMIN_EMAILS`**, being on an allowed
     domain is necessary but no longer *sufficient* — they also need an
     `'approved'` row in the `users` DB table. No row → `render_request_access()`.
     Row with `status='pending'` → `render_pending_access()`. `'denied'` →
     `render_denied_access()` (can re-request). Admins skip this queue
     entirely (they're the ones reviewing it) but still get upserted into
     `users` via `upsert_approved_user()`, so that table stays a complete
     record of everyone with access.
  3. If no DB is configured at all, this whole queue is skipped and the
     original domain-only behaviour applies — never lock everyone out just
     because persistence isn't set up.

An Administrator reviews pending requests on the **Access Requests** page
(`app.py`, Administrator-only) — Approve always creates an **Analyst**
account; promoting someone to Administrator is a code change
(`GOOGLE_ADMIN_EMAILS`), not something this queue can grant.

## 7. Role-gated features

`user['role']` is `'Administrator'` or `'Analyst'`. Every gate is a plain
`if user['role'] == 'Administrator':` at the point of use — there's no
central permissions table. Currently gated: the Data Pipeline page (nav
entry hidden + page-body guard), the Access Requests page (same pattern),
deleting a run on Run History, and every "Export CSV" / raw per-donor
download button.

## 8. Theming flow

`inject_global_css()` (`ui.py`) calls `_apply_palette()` first, which
reassigns a set of module-level color constants (`BG`, `TEXT`, `TEAL`, ...)
based on `session_state.dark_mode`, via `global`. Every other function in
`ui.py` reads those same module-level names, so they all pick up the current
theme automatically — Python resolves them at *call* time, not *def* time.
The sidebar is the one deliberate exception: it's hardcoded navy regardless
of `dark_mode`, a fixed brand element independent of the toggle.

Streamlit's own native widgets (alerts, popovers, file uploader, segmented
controls, `st.dataframe`) pull their colors from `config.toml`'s **fixed**
theme, not this dynamic palette — each of those needed its own explicit CSS
override in `inject_global_css()` to actually respect dark mode.

## 9. Recurring pattern: the one-shot `session_state` flag

Two features in this app need something to happen exactly once, on the very
next rerun, and never again after that — without any JavaScript:

- `nav_loading` — set right before a sidebar nav button's `st.rerun()`;
  read and immediately cleared at the top of `inject_global_css()`. This is
  what scopes the blur+spinner loading overlay to page navigation
  specifically, not every rerun in the app.
- `show_cached_banner` — set alongside `data_source = 'cached'` whenever a
  historical run gets loaded; read and cleared where the "Showing the run
  from..." banner renders. Without this, the banner would reappear on every
  single page for the rest of the session, since `data_source` itself
  correctly *stays* `'cached'` for as long as you're viewing that run.

The shape is always the same: set the flag right before whatever triggers
the next run, read-and-clear it at the one point that's supposed to react to
it, so it can never leak into a later, unrelated rerun.
