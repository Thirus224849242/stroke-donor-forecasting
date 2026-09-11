# Dashboard Guide How to Read Each Page

One section per page, in sidebar order. For how the numbers on these pages
actually get computed, see `flow.md` (data flow) and `architecture.md`
(modeling approach). Every dashboard page (everything except Data Pipeline
and Access Requests) is read-only for any signed-in user nothing here
changes underlying data.

---

## Overview

**What it's for:** a one-screen summary of the whole donor base and where
income is trending, the default landing page.

- **12-month forecast** the *average* of every independently-validated
  total-income method currently available (ML, Linear, and Stock-flow when
  it's fit), with the range across those methods shown as the delta. Donor
  LTV is deliberately excluded from this blend: it only estimates income
  from *today's existing donors* (assumes zero future recruitment), a
  narrower question than "total org income" see Income Forecast for that
  model on its own terms.
- **Active donors / Current monthly income** the most recent actual
  month in `master`, compared to the prior month.
- **Forecast accuracy** the average validation MAPE across the same
  blended methods. Lower is better; this is a backward-looking accuracy
  score, not a forward confidence measure.
- **Monthly income chart** last 24 actual months, then the blended
  forecast with a shaded band showing where the individual methods
  *disagree* with each other (wider band = less consensus between methods,
  not a statistical confidence interval).
- **Income by supplier / campaign type** historical totals, all-time,
  not filtered to any particular period.
- **Dataset summary** row/donor/supplier counts and the ML forecast's own
  MAPE, useful as a quick sanity check that the loaded run is the one you
  expect.

---

## Income Forecast

**What it's for:** the actual forecasting workbench pick a method, a
horizon, see the number.

Three independent methods, each validated the same way (a real holdout, not
just fit statistics):

| Method | Approach | Validation |
|---|---|---|
| **ML forecast** | Gradient-boosted trend + residual model, direct multi-step (not recursive) | 12-month holdout |
| **Linear trend** | Simple trend fit over a selectable training window | 12-month holdout |
| **Stock-flow model** | `Income = Active donors × Avg gift`, with recruits/lapse/gift each forecast independently and rolled forward through that identity | 3 rolling 12-month walk-forward windows |

Because the stock-flow model's MAPE comes from 3 *different* windows (not
one fixed holdout like the other two), its number is **not directly
comparable** to the others only directionally so. The method-comparison
table at the top states this explicitly.

**Reading the stock-flow model specifically:** it has three zones with
different meanings, only visible when you pick it:

- **Months 1–6:** statistical point forecast, ±5% band.
- **Months 7–18:** same models, ±12% band the wider band reflects
  confidence eroding with horizon, not a methodology change.
- **Months 19–36:** **not a point forecast at all.** Three named scenarios
  (Conservative / Base / Optimistic), each built by scaling the *same*
  fitted recruitment/lapse/gift-growth assumptions rather than refitting
  every scenario is traceable back to one explicit assumption change, and
  the organisation is meant to pick which one it plans around, not treat
  any single line as "the" forecast.

"What is driving this forecast" (ML method only) shows the gradient-boosting
model's real feature importances not a canned explanation.

---

## Retention Analysis

**What it's for:** how long donors keep giving after recruitment, segmented
however you like.

- Segment by **supplier**, **campaign type**, or **recruit year** each
  line is one value of that segment; the curve is the % of that segment's
  donors still actively paying at each month since recruitment (observed
  history, not a forecast).
- The dotted **50% retention** line is a quick eyeballing aid, not a target.
- **Retention at key milestones** turns the same curves into a table (3/6/
  12/18/24 months) for quick side-by-side comparison instead of reading
  values off the chart.
- A segment with too few donors to be meaningful (currently: under 50
  donors in that group) is silently excluded from the underlying data
  if a supplier/campaign/year you expect is missing, that's why.

---

## Donor Lifetime Value

**What it's for:** a per-donor $ forecast for *existing* donors specifically
 "how much is this donor base worth going forward," not "how much will the
org raise" (that's Income Forecast).

- **Pareto/NBD** predicts how many more months each existing donor will
  keep giving (their "alive" probability decaying over time).
- **Gamma-Gamma** predicts their expected donation size, combined with the
  above to produce **Predicted 12/24-month value** per donor.
- Scoped to today's donor base only assumes **zero future recruitment**,
  so this is a floor/existing-base figure, not a total org income number.
- **Holdout MAPE** and the **penalizer** grid-search table are the model's
  own validation a held-out period the model didn't see, with the
  penalizer value that performed best highlighted.
- "Top 15 donors" and "Distribution of predicted value" both use the same
  `predicted_ltv_24m` figure, just as a ranked list vs. a histogram.
- **This model is fit lazily** the first time you open this page in a
  session, not as part of "Run pipeline" (it's the slowest single model,
  ~5 minutes, and nothing else reads its output). Expect a one-time wait on
  first visit each session; it's cached after that. If you're viewing a
  run loaded from **Run History** rather than a live session, this page
  may show "unavailable" the raw data needed to fit it isn't kept in
  history, only a live run's `master` panel has it.

---

## Supplier Insights

**What it's for:** which recruiting firms actually perform, by income,
donor count, and average gift, all-time historical, not forecast.

- **Top supplier** KPIs are the single highest-income supplier by total
  historical income.
- "Total income by supplier" / "Active donors by supplier" are the same
  ranking from two angles a supplier can be #1 by income but not by
  donor count if their donors give larger average gifts.
- "Monthly income trend by supplier" is opt-in (pick suppliers to compare)
  since plotting every supplier at once is usually unreadable.
- The summary table at the bottom is the full underlying data every chart
  on this page is drawn from.

---

## Campaign ROI

**What it's for:** which recruitment campaign *types* are worth the spend,
where CPA (cost per acquisition) data is available.

- **CPA range** and **Average cost per acquisition** only populate if the
  uploaded Campaigns file actually has a CPA column if it doesn't, the
  page says so explicitly rather than showing a misleading zero.
- **Free-acquisition income** is total income from campaign types with
  `avg_cpa == 0` (e.g. organic/referral) genuinely free, not missing
  data, as long as CPA data exists at all.
- Same "total income" / "monthly trend" chart pattern as Supplier Insights,
  one level up the hierarchy (campaign type instead of individual
  supplier).

---

## Run History

**What it's for:** every completed pipeline run is saved automatically
(when a database is configured) this page browses, reloads, or deletes
them.

- Shows the **10 most recent** runs; search by Run ID narrows within that
  set (older runs exist in the database but aren't listed/searchable here).
- **Load this run for full analysis** restores that exact run's data into
  every dashboard page what you see afterward is *exactly* what that run
  produced, not recomputed from current data.
- **Delete this run** is **Administrator-only** and permanent Analysts
  see a note that it's restricted, not the button.
- If no database is configured at all, this page explains that and stops
  the pipeline itself still works, it just won't be reloadable afterward.

---

## Data Pipeline *(Administrator only)*

**What it's for:** uploading the four Salesforce exports and running the
actual forecasting pipeline. Hidden from the sidebar entirely for Analysts
 running it mutates the shared dashboard data every viewer sees next.

- All **four** CSVs (Payments, Recurring Payments, Contacts, Campaigns) are
  required before the pipeline runs automatically.
- The stage list shows exactly what's happening and in what order: file
  ingestion → master file build → Linear baseline → ML forecast →
  Stock-flow model → dashboard publish. (Donor LTV is intentionally not in
  this list see the Donor Lifetime Value section above.)
- If you're currently viewing a run loaded from history, a note says so
  uploading here adds a **new** run without touching the one you were
  looking at.

---

## Access Requests *(Administrator only)*

**What it's for:** reviewing pending Google sign-in requests from people on
an allowed domain who don't have an account yet. Hidden from the sidebar
for Analysts.

- Each pending row shows the requester's name, email, and when they asked.
- **Approve** always creates an **Analyst** account promoting someone to
  Administrator is a separate, code-level change, not something granted
  here.
- **Deny** lets them see they were declined and, if they choose, submit a
  new request later.
