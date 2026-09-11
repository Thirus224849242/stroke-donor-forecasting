"""
Production entry point for the stock-flow forecast:

    Income(t) = Active(t) x AvgGift(t)
    Active(t) = Active(t-1) + Recruits(t) - Lapsed(t)

Combines walkforward.py (component series + validation harness) and
component_forecast.py (the three component models) into the full 36-month
forecast with the 3-zone confidence architecture:

  Zone 1 (months 1-6):   statistical forecast, +/-5% band
  Zone 2 (months 7-18):  statistical forecast, +/-12% band (same central
                          models as Zone 1, just a wider band -- confidence
                          erodes with horizon even though the point
                          forecast doesn't change methodology)
  Zone 3 (months 19-36): NOT a point forecast. Three named scenarios
                          (Conservative / Base / Optimistic), each built by
                          scaling the SAME fitted recruits/hazard/gift
                          models rather than refitting from scratch, so
                          every scenario is traceable back to one explicit
                          recruitment, lapse, or gift-growth assumption.

Zone 1/2 bands are the doc's stated policy (a fixed schedule), not derived
from the 3-window walk-forward variance -- 3 windows isn't enough history
to estimate a reliable confidence interval, so the walk-forward MAPEs are
reported alongside for transparency instead of used to size the bands.
"""

import numpy as np
import pandas as pd

from pipeline.walkforward import build_monthly_components, run_walk_forward
from pipeline.component_forecast import (
    forecast_recruits_sarima, forecast_avg_gift,
    make_cohort_lapse_forecaster, build_tenure_hazard,
)

MAX_TENURE = 36
ZONE1_MONTHS = 6
ZONE2_MONTHS = 18   # zone 1 + zone 2 combined
ZONE1_BAND = 0.05
ZONE2_BAND = 0.12

SCENARIOS = {
    'Conservative': dict(growth_multiplier=0.85, hazard_multiplier=1.15, annual_gift_growth=0.025),
    'Base':         dict(growth_multiplier=1.00, hazard_multiplier=1.00, annual_gift_growth=0.04),
    'Optimistic':   dict(growth_multiplier=1.20, hazard_multiplier=0.85, annual_gift_growth=0.07),
}


def _scaled_windows(n, holdout=12, n_windows=3, step=6):
    """3 rolling 12-month holdout windows spaced `step` months apart,
    ending at the most recent available data -- scales down gracefully if
    there isn't enough history for all 3."""
    windows, end = [], n
    for _ in range(n_windows):
        train_end = end - holdout
        if train_end < holdout:
            break
        windows.append({'train_end': train_end, 'test_start': train_end + 1, 'test_end': end})
        end -= step
    windows.reverse()
    if not windows:
        train_end = max(n - holdout, 1)
        windows = [{'train_end': train_end, 'test_start': train_end + 1, 'test_end': n}]
    return tuple(windows)


def _tenure_population(panel, cutoff, max_tenure=MAX_TENURE):
    active_now = panel[(panel['attempts'] > 0) & (panel['donor_month'] == cutoff)]
    pop = {}
    for tau, cnt in active_now.groupby('tenure_months').size().items():
        bucket = min(int(tau), max_tenure)
        pop[bucket] = pop.get(bucket, 0) + cnt
    return pop


def _roll_month(pop, hazard, recruits_this_month, hazard_multiplier=1.0, max_tenure=MAX_TENURE):
    active_prev_total = sum(pop.values())
    new_pop, lapsed_total = {}, 0.0
    for tau, cnt in pop.items():
        h = min(hazard.get(tau, hazard[max_tenure]) * hazard_multiplier, 0.95)
        survivors = cnt * (1 - h)
        lapsed_total += cnt * h
        next_tau = min(tau + 1, max_tenure)
        new_pop[next_tau] = new_pop.get(next_tau, 0.0) + survivors
    new_pop[0] = new_pop.get(0, 0.0) + recruits_this_month
    active_total = sum(new_pop.values())
    lapse_rate = lapsed_total / active_prev_total if active_prev_total > 0 else 0.0
    return new_pop, active_total, lapse_rate


def build_production_forecast(master, min_period='2019-01', horizon=36,
                               covid_start='2020-01', covid_end='2021-06',
                               progress_callback=None):
    """
    Returns a dict:
      walkforward_summary: DataFrame, one row per validation window (income,
        recruits, lapse rate, avg gift MAPE) -- proper rolling-origin
        validation, not a single fixed holdout.
      components_df: the historical monthly recruits/active/lapse_rate/
        avg_gift/income series.
      zone12_forecast: DataFrame, months 1-18, one central statistical
        forecast with zone, band_low, band_high columns.
      zone3_scenarios: {'Conservative'/'Base'/'Optimistic': DataFrame},
        months 19-36 (or fewer if horizon < 36), continuing from zone 2's
        ending state under each scenario's assumptions.
      assumptions: {scenario_name: {label: value}} for UI display -- every
        scenario's assumption table, human-readable.
    """
    def note(msg):
        if progress_callback:
            progress_callback(msg)

    note('Building monthly recruits/active/lapse/gift series…')
    comp = build_monthly_components(master, min_period=min_period)
    n = len(comp)

    note('Running 3-window walk-forward validation…')
    windows = _scaled_windows(n)
    lapse_forecaster = make_cohort_lapse_forecaster(master, min_period=min_period)
    wf_summary, _wf_detail = run_walk_forward(
        comp, forecast_recruits_sarima, lapse_forecaster, forecast_avg_gift, windows=windows,
    )

    note('Fitting final recruits, lapse, and gift models on the full history…')
    recruits_fc = forecast_recruits_sarima(comp, horizon, covid_start, covid_end)
    gift_fc = forecast_avg_gift(comp, horizon)

    cutoff = pd.Period(pd.to_datetime(comp['month'].iloc[-1]).strftime('%Y-%m'), freq='M')
    panel = master[(master['donor_month'] >= pd.Period(min_period, freq='M')) & (master['donor_month'] <= cutoff)]
    hazard = build_tenure_hazard(panel, cutoff, max_tenure=MAX_TENURE)
    pop = _tenure_population(panel, cutoff, max_tenure=MAX_TENURE)

    note('Rolling Zone 1 and 2 forward (months 1 to 18)…')
    zone12_len = min(ZONE2_MONTHS, horizon)
    rows = []
    for m in range(zone12_len):
        pop, active_total, lapse_rate = _roll_month(pop, hazard, recruits_fc[m])
        income = active_total * gift_fc[m]
        zone = 1 if m < ZONE1_MONTHS else 2
        band = ZONE1_BAND if zone == 1 else ZONE2_BAND
        rows.append({
            'calendar_month': m + 1, 'zone': zone, 'recruits': float(recruits_fc[m]),
            'lapse_rate': lapse_rate, 'active': active_total, 'avg_gift': float(gift_fc[m]),
            'predicted_income': income, 'band_low': income * (1 - band), 'band_high': income * (1 + band),
        })
    zone12_df = pd.DataFrame(rows)

    zone3_len = max(horizon - zone12_len, 0)
    zone3_scenarios, assumptions = {}, {}
    if zone3_len > 0:
        note('Building Zone 3 scenarios (months 19 to 36)…')
        for name, params in SCENARIOS.items():
            pop3 = dict(pop)
            gift_level = float(gift_fc[zone12_len - 1]) if zone12_len > 0 else float(comp['avg_gift'].iloc[-1])
            monthly_gift_growth = (1 + params['annual_gift_growth']) ** (1 / 12) - 1
            rows3 = []
            for m in range(zone3_len):
                idx = min(zone12_len + m, horizon - 1)
                recruits_m = float(recruits_fc[idx]) * params['growth_multiplier']
                pop3, active_total, lapse_rate = _roll_month(
                    pop3, hazard, recruits_m, hazard_multiplier=params['hazard_multiplier'],
                )
                gift_level *= (1 + monthly_gift_growth)
                income = active_total * gift_level
                rows3.append({
                    'calendar_month': zone12_len + m + 1, 'zone': 3, 'recruits': recruits_m,
                    'lapse_rate': lapse_rate, 'active': active_total, 'avg_gift': gift_level,
                    'predicted_income': income,
                })
            zone3_scenarios[name] = pd.DataFrame(rows3)
            assumptions[name] = {
                'Recruitment': f"{(params['growth_multiplier'] - 1) * 100:+.0f}% vs the central forecast",
                'Lapse / retention': f"{(params['hazard_multiplier'] - 1) * 100:+.0f}% vs the empirical hazard curve",
                'Gift growth': f"{params['annual_gift_growth'] * 100:.1f}%/year",
            }

    return {
        'walkforward_summary': wf_summary,
        'components_df': comp,
        'zone12_forecast': zone12_df,
        'zone3_scenarios': zone3_scenarios,
        'assumptions': assumptions,
    }
