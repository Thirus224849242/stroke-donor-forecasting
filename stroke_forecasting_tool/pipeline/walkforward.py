"""
Walk-forward validation harness for the stock-flow forecasting architecture:

    Income(t) = Active(t) x AvgGift(t)
    Active(t) = Active(t-1) + Recruits(t) - Lapsed(t)

Income is never forecast directly. Each component (recruits, lapse rate,
average gift) is forecast independently, Active(t) is rolled forward through
the identity, and Income(t) is derived -- so every dollar of forecast income
is traceable back to one of the three component assumptions.

This module only supplies the monthly component series and the validation
harness. Plug in the actual component forecasters (SARIMA for recruits,
cohort survival curve for lapse rate, trend+inflation for avg gift) via the
forecast_recruits / forecast_lapse_rate / forecast_avg_gift callables passed
to run_walk_forward -- the harness itself is agnostic to what's inside them.
"""

import numpy as np
import pandas as pd

# Three independent 12-month holdout windows sharing 84 months of history,
# each trained on everything before its own window -- same idea as the
# problem doc's Window 1/2/3, expressed as row offsets into components_df.
DEFAULT_WINDOWS = (
    {'train_end': 60, 'test_start': 61, 'test_end': 72},
    {'train_end': 66, 'test_start': 67, 'test_end': 78},
    {'train_end': 72, 'test_start': 73, 'test_end': 84},
)


def build_monthly_components(master, min_period=None):
    """
    Derive the three observable monthly series (plus income) from the raw
    donor-month panel produced by build_master().

    - recruits(t):   count of donors whose recruit_m == t
    - active(t):     count of donors with >=1 payment ATTEMPT in month t.
                      A failed-but-attempted payment still means the
                      recurring schedule is live; only a month with zero
                      attempts means the donor has genuinely stopped. This
                      matches the F2F retry mechanism (a donor can fail
                      several attempts in a row and still be "active").
    - lapsed(t):     solved as the residual active(t-1) + recruits(t) -
                      active(t) that makes the accounting identity hold
                      exactly, clipped at 0 (a donor reactivating after a
                      gap would otherwise show as negative lapse -- rare,
                      but real; reactivations are not separately tracked
                      here and will slightly understate lapse in the month
                      they occur).
    - avg_gift(t):   income(t) / active(t) -- revenue per ACTIVE donor
                      (attempts>0 basis), i.e. ARPU as the reference MRR
                      doc defines it: total revenue / active accounts, NOT
                      total revenue / paying accounts. Using paid-donor
                      count as the denominator here would silently inflate
                      any forecast that multiplies it back by active(t),
                      since active(t) counts ~1.4-1.6x more donors than
                      paid_count (many "active" donors fail every attempt
                      in a given month but are still on the books) --
                      confirmed on real data and the actual root cause of
                      an early ~40% income over-forecast in this project.
    - income(t):     total success_amt that month == active(t) x
                      avg_gift(t) by construction.

    Excludes the partial trailing month, same convention as the rest of the
    pipeline (get_monthly_actuals, ltv_model._donor_transactions).
    """
    panel = master
    if min_period is not None:
        panel = panel[panel['donor_month'] >= pd.Period(min_period, freq='M')]

    last_period = panel['donor_month'].max()
    panel = panel[panel['donor_month'] < last_period]

    months = pd.period_range(panel['donor_month'].min(), panel['donor_month'].max(), freq='M')

    recruits = (
        panel.drop_duplicates('recurring_payment_id')
             .groupby('recruit_m')['recurring_payment_id'].size()
             .reindex(months, fill_value=0)
    )
    active = (
        panel[panel['attempts'] > 0]
             .groupby('donor_month')['recurring_payment_id'].nunique()
             .reindex(months, fill_value=0)
    )
    paid = panel[panel['paid_flag']]
    income = paid.groupby('donor_month')['success_amt'].sum().reindex(months, fill_value=0.0)
    avg_gift = (income / active.replace(0, np.nan)).fillna(0.0)

    lapsed = (active.shift(1).fillna(active.iloc[0]) + recruits - active).clip(lower=0)
    lapse_rate = (lapsed / active.shift(1).replace(0, np.nan)).fillna(0.0)

    df = pd.DataFrame({
        'recruits': recruits.values,
        'active': active.values,
        'lapsed': lapsed.values,
        'lapse_rate': lapse_rate.values,
        'avg_gift': avg_gift.values,
        'income': income.values,
    })
    df.insert(0, 'month', months.to_timestamp())
    return df.reset_index(drop=True)


def _mape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = actual != 0
    if not nonzero.any():
        return float('nan')
    return float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)


def run_walk_forward(components_df, forecast_recruits, forecast_lapse_rate, forecast_avg_gift,
                      windows=DEFAULT_WINDOWS):
    """
    Run identical rolling-origin train/test splits against all three
    component forecasters, roll Active(t) forward through the accounting
    identity, and derive Income(t) -- never forecast directly.

    forecast_recruits(train_df, horizon)   -> array[horizon]
    forecast_lapse_rate(train_df, horizon) -> array[horizon]
    forecast_avg_gift(train_df, horizon)   -> array[horizon]
        Each receives components_df.iloc[:train_end] (only data available
        at that origin) and returns a forecast array of length horizon.
        Swap in SARIMA / cohort survival / trend+inflation here.

    Returns one row per window: each component's MAPE plus the combined
    income MAPE, all on identical holdout periods so they're directly
    comparable -- and a second DataFrame with the month-by-month detail
    for inspection/plotting.
    """
    if components_df['month'].is_monotonic_increasing is False:
        raise ValueError('components_df must be sorted by month ascending.')

    summary_rows = []
    detail_rows = []

    for w in windows:
        train = components_df.iloc[:w['train_end']]
        test = components_df.iloc[w['test_start'] - 1:w['test_end']]
        horizon = len(test)
        if horizon == 0:
            raise ValueError(f'Window {w} has zero test months; check components_df has enough history.')

        recruits_fc = np.asarray(forecast_recruits(train, horizon), dtype=float)
        lapse_rate_fc = np.asarray(forecast_lapse_rate(train, horizon), dtype=float)
        avg_gift_fc = np.asarray(forecast_avg_gift(train, horizon), dtype=float)

        active_fc, active_prev = [], float(train['active'].iloc[-1])
        for r, lr in zip(recruits_fc, lapse_rate_fc):
            lapsed = active_prev * lr
            active_prev = max(active_prev + r - lapsed, 0.0)
            active_fc.append(active_prev)
        active_fc = np.array(active_fc)
        income_fc = active_fc * avg_gift_fc

        label = f"{w['test_start']}-{w['test_end']}"
        summary_rows.append({
            'window': label,
            'train_months': w['train_end'],
            'recruits_mape': _mape(test['recruits'], recruits_fc),
            'lapse_rate_mape': _mape(test['lapse_rate'], lapse_rate_fc),
            'avg_gift_mape': _mape(test['avg_gift'], avg_gift_fc),
            'income_mape': _mape(test['income'], income_fc),
        })
        detail_rows.append(pd.DataFrame({
            'window': label,
            'month': test['month'].values,
            'recruits_actual': test['recruits'].values, 'recruits_forecast': recruits_fc,
            'lapse_rate_actual': test['lapse_rate'].values, 'lapse_rate_forecast': lapse_rate_fc,
            'avg_gift_actual': test['avg_gift'].values, 'avg_gift_forecast': avg_gift_fc,
            'active_forecast': active_fc,
            'income_actual': test['income'].values, 'income_forecast': income_fc,
        }))

    return pd.DataFrame(summary_rows), pd.concat(detail_rows, ignore_index=True)
