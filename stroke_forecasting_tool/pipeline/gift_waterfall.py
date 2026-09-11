"""
Donor Gift Waterfall an MRR-waterfall-style monthly $ bridge:

    Income(t) ~= Income(t-1) + New(t) + Expansion(t) - Contraction(t) - Churned(t)

Built on each donor's actual paid amount (success_amt) over time, not
current_gift_setting: current_gift_setting is a single snapshot joined onto
every historical row for a donor (see build_master.py), so it never varies
within a donor's history in this panel and carries no expansion/contraction
signal at all (verified directly: 0 of 86,286 donors show more than one
distinct value). success_amt does show real, sustained step-changes per
donor (confirmed on real data), so it's the only field that can drive this.

Payment-timing noise (a donor's F2F retry mechanism can cause a failed
attempt to land in one month and its retry the next, or two attempts to
land in the same month) means a raw month-over-month diff of success_amt
would misfire on both false expansions and false contractions. Instead,
each donor's PAID-month amounts are run-length encoded and short runs
(< MIN_RUN consecutive paid months) are merged into the preceding
confirmed level, treating them as noise rather than a real gift change
a donor's true level only "moves" once it holds for a few months running.
"""

import itertools

import numpy as np
import pandas as pd

from pipeline.component_forecast import sarima_series_forecast

MIN_RUN = 3  # consecutive paid months required to confirm a new level
HOLDOUT_MONTHS = 12  # matches the ML/Linear/Donor-rollup holdout convention elsewhere in the app
TREND_LOOKBACK = 12  # trailing months averaged for the small expansion/contraction buckets


def _confirmed_levels(months, amounts, min_run=MIN_RUN):
    """
    Run-length-encode a donor's rounded paid-month amounts and merge any
    interior run shorter than min_run into the preceding confirmed level
    (the first and last runs are always kept, however short, since they
    anchor the donor's known starting and current level).

    Returns a list of (month, level) marking the calendar month each
    confirmed level first appears, with consecutive duplicate levels
    collapsed after merging.
    """
    vals = [round(a) for a in amounts]
    runs, idx = [], 0
    for val, grp in itertools.groupby(vals):
        run_len = sum(1 for _ in grp)
        runs.append([val, run_len, idx])
        idx += run_len

    kept = [runs[0]]
    for i in range(1, len(runs)):
        val, length, start = runs[i]
        is_last = i == len(runs) - 1
        if length < min_run and not is_last:
            continue
        kept.append([val, length, start])

    collapsed = [kept[0]]
    for val, length, start in kept[1:]:
        if val == collapsed[-1][0]:
            continue
        collapsed.append([val, length, start])

    return [(months[start], val) for val, _length, start in collapsed]


def build_gift_waterfall(master, min_period='2019-01'):
    """
    Returns (monthly_waterfall, donor_events).

    monthly_waterfall: one row per calendar month with new_volume,
    expansion_volume, contraction_volume, churned_volume, income (actual),
    bridged_income (income(t-1) + new + expansion - contraction - churned),
    and residual (income - bridged_income) the part of the month-to-month
    change NOT explained by a confirmed level change or churn. This is
    expected to be non-zero: a donor having one bad or one lucky payment
    month while still active (retry noise) moves income without being a
    real level change, and isn't bucketed here. Reported explicitly rather
    than forced to reconcile, so it's visible how much of the picture this
    first-pass waterfall actually explains.

    donor_events: per-donor confirmed-level event log (recurring_payment_id,
    month, level, event_type) for auditability every dollar in the
    monthly waterfall traces back to a row here.
    """
    panel = master[master['donor_month'] >= pd.Period(min_period, freq='M')]
    last_period = panel['donor_month'].max()
    panel = panel[panel['donor_month'] < last_period]

    months = pd.period_range(panel['donor_month'].min(), panel['donor_month'].max(), freq='M')

    # last calendar month each donor had >=1 payment attempt (matches the
    # "active" definition in walkforward.py) -- used only to attribute churn,
    # kept separate from the paid-amount level-tracking above so a donor's
    # gift level isn't disturbed by months they simply failed to pay.
    last_active = panel[panel['attempts'] > 0].groupby('recurring_payment_id')['donor_month'].max()
    panel_last_month = months[-1]

    paid = panel[panel['paid_flag']].sort_values(['recurring_payment_id', 'donor_month'])

    events = []
    for pid, g in paid.groupby('recurring_payment_id', sort=False):
        levels = _confirmed_levels(g['donor_month'].tolist(), g['success_amt'].tolist())
        if not levels:
            continue

        first_month, first_level = levels[0]
        events.append((pid, first_month, first_level, 'new'))

        prev_level = first_level
        for month, level in levels[1:]:
            etype = 'expansion' if level > prev_level else 'contraction'
            events.append((pid, month, level - prev_level, etype))
            prev_level = level

        donor_last_active = last_active.get(pid)
        if donor_last_active is not None and donor_last_active < panel_last_month:
            churn_month = donor_last_active + 1
            if churn_month in months:
                events.append((pid, churn_month, prev_level, 'churn'))

    donor_events = pd.DataFrame(events, columns=['recurring_payment_id', 'month', 'amount', 'event_type'])

    monthly = pd.DataFrame({'month': months})
    for etype, col in [('new', 'new_volume'), ('expansion', 'expansion_volume'),
                        ('contraction', 'contraction_volume'), ('churn', 'churned_volume')]:
        sub = donor_events[donor_events['event_type'] == etype]
        vol = sub.groupby('month')['amount'].sum().abs()
        monthly[col] = monthly['month'].map(vol).fillna(0.0).values

    income = paid.groupby('donor_month')['success_amt'].sum().reindex(months, fill_value=0.0)
    monthly['income'] = income.values
    monthly['bridged_income'] = (
        monthly['income'].shift(1).fillna(monthly['income'].iloc[0])
        + monthly['new_volume'] + monthly['expansion_volume']
        - monthly['contraction_volume'] - monthly['churned_volume']
    )
    monthly['residual'] = monthly['income'] - monthly['bridged_income']
    monthly['month'] = monthly['month'].dt.to_timestamp()

    return monthly, donor_events


def _mape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = actual != 0
    if not nonzero.any():
        return float('nan')
    return float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)


def _trend_forecast(y, horizon, lookback=TREND_LOOKBACK):
    """Trailing-mean forecast for the small, noise-dominated expansion/
    contraction buckets (well under 1% of monthly income each -- verified
    on real data: ~0.20% and ~0.06% respectively) -- a seasonal model
    would just be fitting noise on buckets that small, so a flat trailing
    average is the honest choice. Clipped at 0 since these are
    non-negative dollar volumes."""
    y = np.asarray(y, dtype=float)
    level = max(float(y[-lookback:].mean()), 0.0)
    return np.repeat(level, horizon)


def _roll_income_forward(last_income, new_fc, expansion_fc, contraction_fc, churned_fc):
    """Rolls the bridge identity forward from a single known anchor -- the
    last REAL actual income, not a chain of prior forecasts --
    income(t) = income(t-1) + new(t) + expansion(t) - contraction(t) -
    churned(t), recursively for len(new_fc) months. Mirrors how
    stock_flow_forecast.py rolls Active(t) forward through its own
    identity rather than forecasting income directly."""
    incomes = []
    prev = last_income
    for n, e, c, ch in zip(new_fc, expansion_fc, contraction_fc, churned_fc):
        prev = max(prev + n + e - c - ch, 0.0)
        incomes.append(prev)
    return np.array(incomes)


def forecast_gift_waterfall(master, min_period='2019-01', horizon=24, holdout_months=HOLDOUT_MONTHS,
                             covid_start='2020-01', covid_end='2021-06', progress_callback=None):
    """
    Uses the gift waterfall's own monthly $ bridge as a forecaster in its
    own right: forecast each of the four bridge components independently
    -- new_volume and churned_volume (the two that actually move the
    needle, ~6-7% of monthly income each on real data) get a proper
    seasonal SARIMA model with the same COVID dummy used elsewhere in
    this pipeline; expansion_volume and contraction_volume (consistently
    under 0.2% of monthly income each) get a simple trailing average,
    since a seasonal model would just be fitting noise on buckets that
    small -- then roll the bridge identity forward from the last known
    actual income. Never forecasts income directly, same philosophy as
    stock_flow_forecast.py's Active(t) x AvgGift(t).

    The panel's very first month is always excluded from fitting and
    validation: build_gift_waterfall()'s bridged_income formula has no
    real "previous month" to diff against there, so every donor's opening
    balance registers as "new" and roughly doubles that one row -- a
    boundary artifact of the bridge, not a real data point (confirmed on
    real data: that row alone showed a 100% residual, ~9x every other
    month, while every other month combined shows median 1.1% residual).

    Returns (monthly_df, mape, metrics):
      monthly_df: calendar_month (1..horizon), predicted_income, plus the
        four bucket forecasts for transparency/audit.
      mape: holdout MAPE from a REAL multi-step rolled forecast over the
        holdout window (not a 1-step-ahead metric) -- matches how the
        stock-flow model is validated, since compounding error over the
        rolled horizon is exactly the risk this method needs to be
        honest about, not hidden by only ever checking one step ahead.
      metrics: {'holdout_mape', 'new_volume_mape', 'churned_volume_mape',
        'n_months_fit'} -- component-level detail for auditability.
    """
    def note(msg):
        if progress_callback:
            progress_callback(msg)

    note('Building the monthly gift waterfall…')
    monthly, _events = build_gift_waterfall(master, min_period=min_period)
    hist = monthly.iloc[1:].reset_index(drop=True)  # drop the boundary-artifact first row
    if len(hist) < holdout_months + 24:
        raise ValueError(
            f'Not enough history for the gift-waterfall forecast: {len(hist)} usable months, '
            f'need at least {holdout_months + 24} (holdout + a real fitting window).'
        )

    months_all = pd.PeriodIndex(pd.to_datetime(hist['month']).dt.to_period('M'))

    note('Validating on a 12-month holdout (full multi-step rolled forecast, not 1-step-ahead)…')
    train = hist.iloc[:-holdout_months]
    test  = hist.iloc[-holdout_months:]
    train_months = months_all[:-holdout_months]

    new_fc_h   = sarima_series_forecast(train['new_volume'], train_months, holdout_months, covid_start, covid_end)
    churn_fc_h = sarima_series_forecast(train['churned_volume'], train_months, holdout_months, covid_start, covid_end)
    exp_fc_h   = _trend_forecast(train['expansion_volume'], holdout_months)
    con_fc_h   = _trend_forecast(train['contraction_volume'], holdout_months)

    income_fc_h = _roll_income_forward(float(train['income'].iloc[-1]), new_fc_h, exp_fc_h, con_fc_h, churn_fc_h)
    mape       = _mape(test['income'].values, income_fc_h)
    new_mape   = _mape(test['new_volume'].values, new_fc_h)
    churn_mape = _mape(test['churned_volume'].values, churn_fc_h)

    note('Fitting on the full history for the production forecast…')
    new_fc   = sarima_series_forecast(hist['new_volume'], months_all, horizon, covid_start, covid_end)
    churn_fc = sarima_series_forecast(hist['churned_volume'], months_all, horizon, covid_start, covid_end)
    exp_fc   = _trend_forecast(hist['expansion_volume'], horizon)
    con_fc   = _trend_forecast(hist['contraction_volume'], horizon)
    income_fc = _roll_income_forward(float(hist['income'].iloc[-1]), new_fc, exp_fc, con_fc, churn_fc)

    monthly_df = pd.DataFrame({
        'calendar_month': range(1, horizon + 1),
        'predicted_income': income_fc,
        'new_volume': new_fc, 'expansion_volume': exp_fc,
        'contraction_volume': con_fc, 'churned_volume': churn_fc,
    })
    metrics = {
        'holdout_mape': mape, 'new_volume_mape': new_mape, 'churned_volume_mape': churn_mape,
        'n_months_fit': len(hist),
    }
    return monthly_df, mape, metrics
