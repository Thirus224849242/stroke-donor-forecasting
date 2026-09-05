# Architecture

The design decisions behind this app and why they were made. For how the
code connects day-to-day, see `flow.md`. For what each dashboard means, see
`dashboard.md`.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| App framework | Streamlit 1.61 | Rerun-the-script-on-every-interaction model, `session_state` as the shared state bus — see `flow.md` §2. No separate frontend/backend split to maintain. |
| Data processing | pandas + numpy | Standard for a donor-month panel of this size (millions of rows on real data). |
| Forecasting | scipy, scikit-learn (`GradientBoostingRegressor`), statsmodels (SARIMA), `lifetimes` (Pareto/NBD, Gamma-Gamma), `lifelines` | One library per method, not a single "AutoML" black box — each forecast is independently interpretable and independently validated. |
| Charts | Plotly | Interactive, and its theming hooks into the app's own light/dark palette. |
| Persistence | Supabase (managed Postgres) via SQLAlchemy + psycopg2 | Only two tables (`dashboard_runs`, `users`) — no ORM models, just parameterized `text()` queries. Entirely optional: every DB-backed feature fails soft if it isn't configured. |
| Auth | Streamlit's native `st.login()`/`st.user` (Google OIDC via Authlib) + a hardcoded password fallback | No separate auth service to run; the fallback exists for local dev/demo without Google credentials configured. |

## High-level shape

```
┌─────────────┐     4 CSVs      ┌──────────────┐   master   ┌───────────────┐
│   Browser    │ ───────────────▶│ pipeline/     │──panel───▶│ 3 forecast     │
│  (Streamlit  │                 │ clean.py +    │           │ models + every │
│   frontend)  │                 │ build_master  │           │ page aggregate │
└──────┬───────┘                 └──────────────┘           └───────┬───────┘
       │                                                             │
       │  st.session_state (the ONLY thing that survives a rerun)    │
       │◀────────────────────────────────────────────────────────────┘
       │
       │  optionally, on completion
       ▼
┌──────────────┐   gzip'd JSON   ┌────────────────┐
│  db.py        │ ───────────────▶│ Supabase        │
│ save/restore  │◀─────────────── │ (dashboard_runs,│
│               │   on reload     │  users tables)  │
└──────────────┘                 └────────────────┘
```

There is no API layer and no separate backend process — `app.py` *is* the
whole application, running as one Python script that Streamlit re-executes
on every interaction.

## Data model

One table, `master`, is the source of truth for every dashboard: a
donor-month panel (one row per `recurring_payment_id` × `donor_month`),
produced by joining:

- the payments panel (aggregated from raw payment attempts — see the
  memory note below) — `attempts`, `successes`, `failures`, `success_amt`,
  `paid_flag`, `success_rate`
- the recurring-payment signup record — `contact_id`, `campaign_code`,
  `supplier`, `recruitment_date`, `tenure_months`
- the campaign lookup — `campaign_type`, `cost_per_acquisition`
- the contact lookup — demographic fields

`master` itself is **never persisted** — it's rebuilt fresh from the four
CSVs on every live pipeline run, and deliberately excluded from what gets
saved to `dashboard_runs` (too large, and no page needs the raw panel — see
below). What *is* persisted is the derived, display-level output every page
actually reads: aggregated summaries, chart series, table rows, scalar
metrics — plus the full per-donor LTV table specifically, since it's the
one raw-ish table a page (Donor Lifetime Value's CSV download) actually
needs back.

## Forecasting architecture: independent methods, not one model

Every income forecast on this app comes from one of several **independently
fit and independently validated** methods, deliberately not a single
"pick the best one" black box:

1. **Linear trend** — a straightforward baseline, useful as a sanity check
   against the more complex methods.
2. **ML (gradient boosting)** — direct multi-step (each forecast horizon is
   its own model output, not a recursive chain of 1-step predictions that
   compounds its own error).
3. **Stock-flow** — `Income(t) = Active(t) × AvgGift(t)`,
   `Active(t) = Active(t-1) + Recruits(t) - Lapsed(t)`. Income is **never**
   forecast directly here — each component (recruits via SARIMA, lapse
   rate via an empirical cohort-hazard curve, average gift via trend) is
   forecast on its own, then rolled forward through that accounting
   identity, so every dollar of forecast income is traceable back to one
   explicit component assumption. Validated on 3 rolling 12-month
   walk-forward windows (not one fixed holdout), which is why its MAPE
   isn't directly comparable to the other two methods' single-holdout
   MAPE.
4. **Donor LTV (Pareto/NBD + Gamma-Gamma)** — a *different question*
   entirely: not "total org income," but "what are today's existing
   donors worth" (assumes zero future recruitment). Kept separate from the
   Overview page's blended forecast for exactly that reason.
5. **sBG / BG-NBD + Gift Waterfall** — a fifth method and an MRR-style
   bridge, both implemented (`pipeline/sbg_bgnbd_model.py`,
   `pipeline/gift_waterfall.py`) but **not currently wired into the
   pipeline** — their old `session_state`/DB schema keys are kept as
   always-`None` placeholders purely so a historical saved run from when
   they were active still loads without a schema error.

The Overview page's headline "12-month forecast" is the *average* of
whichever total-income methods are currently available, with the spread
across methods shown as the uncertainty range — not a single model's
confidence interval.

## Memory engineering: why the pipeline doesn't OOM on real data

The Payments export is the one file that's genuinely large on real data
(millions of rows spanning years). Two things make it tractable:

1. **Streamed, per-chunk aggregation** (`pipeline/clean.py`'s
   `clean_payments()`) — reads 500k rows at a time, aggregates each chunk
   to the donor-month grain immediately, and only ever holds one raw chunk
   plus a running numeric-only panel in memory. The earlier version held
   every raw row in a list, concatenated the whole thing, then grouped it
   once — three copies of a multi-million-row table alive simultaneously,
   which is what was actually causing the deploy host's OOM kill.
2. **A 2019+ cutoff** (`min_period` in `build_master`/`clean_payments`) —
   every forecasting model already treats pre-2019 history as unreliable
   and excludes it before fitting, so those rows are dropped during
   loading rather than carried through the whole pipeline just to be
   discarded later. On the real dataset this turns out to remove
   relatively few rows (most already fall after 2019) — it's a minor
   assist, not the primary fix; #1 is what actually solved the OOM.

This cutoff only applies to Payments.csv. Recurring Payments, Contacts, and
Campaigns are loaded in full, unfiltered — they're one row per signup/
contact/campaign (not per payment event), so they're not what's driving
file size, and filtering them by date would silently break the join for
any donor recruited before 2019 who's still active today.

## Persistence architecture

Two tables, both optional (the app runs without a database, with reduced
functionality):

- **`dashboard_runs`** — one row per completed pipeline run. A handful of
  small summary columns (donor count, income, each method's MAPE) sit
  alongside a gzip'd JSON blob of the full display-level state, so the Run
  History list/trend view is cheap (it never decompresses every run's blob
  just to build a list) while a full reload (`load_dashboard_run`) gets
  everything back.
- **`users`** — the access-control table backing the request/approve queue
  (see below). `email` is the primary key; `status` is `'pending'`,
  `'approved'`, or `'denied'`.

Every function in `db.py` follows the same fail-soft contract: no
connection configured, or a connection error, returns `None`/`False`/an
empty DataFrame — never raises past the caller. The rest of the app never
needs to know *why* persistence isn't available, only that it isn't.

## Auth & access-control architecture

Two independent gates, both in `auth.py`'s `handle_google_redirect()`:

1. **Domain allow-list** (`ALLOWED_GOOGLE_DOMAINS`,
   `GOOGLE_EXTRA_ALLOWED_EMAILS`) — a hard block, unchanged from the
   original design. This is a *code-level* allow-list (redeploy required
   to change it), by design — it's the outer perimeter (who's even allowed
   to ask), not the inner one (who's actually been granted access).
2. **DB-backed approval queue** (`users` table) — for everyone except
   `GOOGLE_ADMIN_EMAILS`, passing gate #1 gets you a request-access screen,
   not immediate entry. An Administrator approves or denies from the
   Access Requests page. This is deliberately a *simple* queue: approval
   only ever grants Analyst, never Administrator — promoting someone to
   Administrator stays a `GOOGLE_ADMIN_EMAILS` code change, so there's
   always a human-reviewed, out-of-band step before anyone gets elevated
   access, not a button any admin could click by accident.

Role (`Administrator`/`Analyst`) gates individual features by a plain
`if user['role'] == 'Administrator':` check at each point of use — there's
no central permissions table or middleware. Currently gated: running the
pipeline, deleting a saved run, reviewing access requests, and every raw/
per-donor CSV export. Everything else (every dashboard page) is read-only
for any signed-in user.

## UI architecture: one dynamic palette, one fixed exception

`ui.py` implements a small design system on top of Streamlit's own
component set: `card()`, `kpi()`, `chart()`, `page_header()`, `stage_row()`,
plus the sidebar and the global CSS injection. Colors are module-level
Python constants (`TEXT`, `TEAL`, `BG`, ...) reassigned at the top of every
render by `_apply_palette()` based on `session_state.dark_mode` — every
other function in the file picks up the current theme automatically since
Python resolves those names at call time.

The one deliberate exception is the **sidebar**, which is a fixed Stroke
Foundation navy regardless of the dark-mode toggle — a constant brand
element, not something that should flip with the reader's preference.

Streamlit's own native widget chrome (alerts, popovers, the file uploader,
segmented controls, `st.dataframe`'s outer border) reads from
`config.toml`'s **fixed**, non-dynamic theme — each of those needed an
explicit CSS override to actually track dark mode, since the dynamic
palette above only exists at the Python level and has no automatic
connection to Streamlit's own component styling.

## Deployment shape

- `secrets.toml` (git-ignored) carries the Google OAuth client
  credentials and the Postgres connection string — both entirely optional;
  their absence is checked explicitly (`google_auth_configured()`,
  `db_configured()`) and degrades features gracefully rather than
  crashing.
- `_bootstrap_secrets_from_env()` (`app.py`) writes a `secrets.toml` from
  plain environment variables on hosts that don't support Streamlit's
  native secrets file (e.g. Hugging Face Spaces), so the same code runs
  unmodified across hosting environments.
- `.streamlit/config.toml` sets the base (light) theme and the sidebar's
  own fixed navy theme — the latter is what native sidebar chrome (e.g.
  the collapse-arrow row) inherits before any CSS override runs.
