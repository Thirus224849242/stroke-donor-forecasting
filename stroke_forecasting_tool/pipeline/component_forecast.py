"""
The three component forecasters feeding pipeline.walkforward.run_walk_forward:

- forecast_recruits_sarima: SARIMA(p,d,q)(P,D,Q,12) on the monthly recruits
  series, with a binary COVID exogenous dummy (Option B from the problem
  doc: keep all 84+ months, let the model learn true seasonal indices, and
  isolate the anomaly's effect explicitly rather than discarding data).

- make_cohort_lapse_forecaster: a population roll-forward driven by an
  empirical per-tenure hazard curve. Corrects the immortal-time bias found
  earlier in ltv_model.py's tenure analysis (dividing by ALL-ever-recruited
  donors understates retention, since recently-recruited donors haven't had
  the chance to reach high tenure yet) by only counting, at each tenure tau,
  donors who were recruited early enough to have reached tau+1 by the
  cutoff date.

- forecast_avg_gift: a capped linear trend. avg_gift is empirically the
  smoothest, most predictable of the three components (0.8-4.4% naive MAPE
  in the walkforward smoke test), so it doesn't need SARIMA-level machinery
  -- just a trend fit with growth capped at a sane annualized rate so it
  can't runaway-extrapolate past the last 36 months of data.

Each forecaster's own recruitment/hazard assumptions are self-contained
(they don't consume each other's forecasts) so each component's walk-forward
MAPE reflects that component's own model quality, not compounded error from
the others -- matching the problem doc's "one MAPE per component" principle.
"""

import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

DEFAULT_SARIMA_ORDERS = ((1, 1, 1), (0, 1, 1), (1, 0, 1), (2, 1, 1), (1, 1, 0))
DEFAULT_SEASONAL_ORDERS = ((1, 1, 0, 12), (0, 1, 1, 12), (1, 1, 1, 12), (0, 1, 0, 12))


def _covid_dummy(periods, covid_start='2020-01', covid_end='2021-06'):
    start, end = pd.Period(covid_start, freq='M'), pd.Period(covid_end, freq='M')
    mask = (periods >= start) & (periods <= end)
    return np.asarray(mask, dtype=float).reshape(-1, 1)


def forecast_recruits_sarima(train, horizon, covid_start='2020-01', covid_end='2021-06',
                              orders=DEFAULT_SARIMA_ORDERS, seasonal_orders=DEFAULT_SEASONAL_ORDERS):
    """
    Light grid search over a handful of (p,d,q)(P,D,Q,12) combinations,
    selected by AIC on the training window, with a COVID dummy exogenous
    regressor. Falls back to a flat trailing average if every candidate
    fails to fit (e.g. a very short training window).
    """
    months = pd.PeriodIndex(pd.to_datetime(train['month']).dt.to_period('M'))
    y = train['recruits'].astype(float).values
    exog = _covid_dummy(months, covid_start, covid_end)

    best_aic, best_res = np.inf, None
    for order, sorder in itertools.product(orders, seasonal_orders):
        try:
            res = SARIMAX(y, exog=exog, order=order, seasonal_order=sorder,
                           enforce_stationarity=False, enforce_invertibility=False).fit(
                disp=False, maxiter=200, method='lbfgs',
            )
            if not res.mle_retvals.get('converged', True):
                continue
            if np.isfinite(res.aic) and res.aic < best_aic:
                best_aic, best_res = res.aic, res
        except Exception:
            continue

    if best_res is None:
        return np.repeat(y[-6:].mean(), horizon)

    future_months = pd.period_range(months[-1] + 1, periods=horizon, freq='M')
    future_exog = _covid_dummy(future_months, covid_start, covid_end)
    fc = best_res.get_forecast(steps=horizon, exog=future_exog).predicted_mean
    return np.clip(np.asarray(fc), a_min=0, a_max=None)


def build_tenure_hazard(panel, cutoff_period, max_tenure=36, min_denom=30):
    """
    hazard[tau] = 1 - P(active at tenure tau+1 | active at tenure tau),
    counted only over donors recruited early enough to have had the chance
    to reach tau+1 by cutoff_period (recruit_m + tau + 1 <= cutoff_period).
    This is the eligible-denominator fix for the immortal-time bias: a
    cohort recruited last month can't have reached tenure 23 yet, and must
    not be counted as part of "everyone who could have stayed to tenure 23"
    for either the numerator or the denominator.

    Falls back to the nearest tenure with enough data (>= min_denom donors)
    when a given tenure's eligible pool is too small to trust.
    """
    active = panel[panel['attempts'] > 0][['recurring_payment_id', 'recruit_m', 'tenure_months']]
    active = active[active['tenure_months'] <= max_tenure + 1]
    pivot = active.groupby(['recurring_payment_id', 'tenure_months']).size().unstack(fill_value=0) > 0
    recruit_m = active.drop_duplicates('recurring_payment_id').set_index('recurring_payment_id')['recruit_m']

    raw_hazard, denom_size = {}, {}
    for tau in range(0, max_tenure + 1):
        eligible_ids = recruit_m[(recruit_m + tau + 1) <= cutoff_period].index
        eligible_ids = pivot.index.intersection(eligible_ids)
        if len(eligible_ids) == 0 or tau not in pivot.columns:
            continue
        sub = pivot.loc[eligible_ids]
        active_at_tau = sub[tau]
        active_at_tau1 = sub[tau + 1] if (tau + 1) in sub.columns else pd.Series(False, index=sub.index)
        denom = int(active_at_tau.sum())
        denom_size[tau] = denom
        if denom == 0:
            continue
        survived = int((active_at_tau & active_at_tau1).sum())
        raw_hazard[tau] = 1 - survived / denom

    hazard, last_good = {}, 0.05
    for tau in range(0, max_tenure + 1):
        if tau in raw_hazard and denom_size.get(tau, 0) >= min_denom:
            last_good = raw_hazard[tau]
        hazard[tau] = last_good
    return hazard


def make_cohort_lapse_forecaster(master, min_period='2019-01', max_tenure=36, recruit_window=12):
    """
    Returns a forecast_lapse_rate(train, horizon) closure for
    walkforward.run_walk_forward. Internally re-derives the hazard curve
    and the active population's tenure histogram as of each window's own
    training cutoff (never looks past it), then rolls the population
    forward month by month: existing tenure buckets lapse at their
    empirical hazard rate, survivors age by one tenure, and new recruits
    (assumed at the training window's own trailing recruitment rate --
    deliberately not the SARIMA recruits forecast, so this component's
    walk-forward MAPE reflects the lapse model alone) enter at tenure 0.
    """
    panel_all = master[master['donor_month'] >= pd.Period(min_period, freq='M')]

    def forecast_lapse_rate(train, horizon):
        cutoff = pd.Period(pd.to_datetime(train['month'].iloc[-1]).strftime('%Y-%m'), freq='M')
        panel = panel_all[panel_all['donor_month'] <= cutoff]
        hazard = build_tenure_hazard(panel, cutoff, max_tenure=max_tenure)

        active_now = panel[(panel['attempts'] > 0) & (panel['donor_month'] == cutoff)]
        pop = {}
        for tau, cnt in active_now.groupby('tenure_months').size().items():
            bucket = min(int(tau), max_tenure)
            pop[bucket] = pop.get(bucket, 0) + cnt

        recruit_rate = float(train['recruits'].tail(recruit_window).mean())

        rates = []
        for _ in range(horizon):
            active_prev_total = sum(pop.values())
            new_pop, lapsed_total = {}, 0.0
            for tau, cnt in pop.items():
                h = hazard.get(tau, hazard[max_tenure])
                survivors = cnt * (1 - h)
                lapsed_total += cnt * h
                next_tau = min(tau + 1, max_tenure)
                new_pop[next_tau] = new_pop.get(next_tau, 0.0) + survivors
            new_pop[0] = new_pop.get(0, 0.0) + recruit_rate
            pop = new_pop
            rates.append(lapsed_total / active_prev_total if active_prev_total > 0 else 0.0)
        return np.array(rates)

    return forecast_lapse_rate


def forecast_avg_gift(train, horizon, max_annual_growth=0.06, lookback=36):
    """
    Linear trend on the trailing `lookback` months, with the fitted slope
    capped at a max_annual_growth annualized rate so a noisy trend can't
    runaway-extrapolate over a long horizon. avg_gift is the most stable of
    the three components (see module docstring), so a simple capped trend
    already tracks it well without needing a seasonal model.
    """
    sub = train.tail(min(lookback, len(train)))
    y = sub['avg_gift'].astype(float).values
    x = np.arange(len(y))
    slope, _intercept = np.polyfit(x, y, 1)

    last_val = float(y[-1])
    monthly_growth_cap = (1 + max_annual_growth) ** (1 / 12) - 1
    capped_slope = np.clip(slope, -abs(last_val) * monthly_growth_cap, abs(last_val) * monthly_growth_cap)

    return last_val + capped_slope * np.arange(1, horizon + 1)
