import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


def get_monthly_actuals(master):
    """Return monthly active donors and income from master file."""
    monthly = (
        master[master['paid_flag'] == True]
              .groupby('donor_month')
              .agg(
                  active_donors=('recurring_payment_id', 'nunique'),
                  total_income =('success_amt',          'sum'),
              )
              .reset_index()
    )
    monthly['donor_month'] = monthly['donor_month'].astype(str)
    # Exclude last month — likely partial
    monthly = monthly.iloc[:-1].copy()
    return monthly


def fit_linear_forecast(monthly, n_train=24, n_forecast=24, holdback=12, min_period='2019-01'):
    """
    Fit a linear trend on the last n_train months of income (restricted to
    min_period onward — older history is treated as unreliable).
    Validate on the last `holdback` months held back.
    Return forecast DataFrame, MAPE, slope and intercept.
    """
    monthly     = monthly[monthly['donor_month'] >= min_period]
    recent      = monthly.tail(n_train).copy()
    recent['t'] = range(1, len(recent) + 1)
    income      = recent['total_income'].values
    t           = recent['t'].values

    def linear(t, a, b):
        return a * t + b

    popt, _ = curve_fit(linear, t, income)
    a, b    = popt

    # Validate — hold back last `holdback` months
    train_i   = income[:-holdback]
    actual_v  = income[-holdback:]
    t_train   = t[:-holdback]
    t_val     = t[-holdback:]
    popt_v, _ = curve_fit(linear, t_train, train_i)
    pred_v    = linear(t_val, *popt_v)
    mape      = float(np.mean(np.abs(
        (pred_v - actual_v) / actual_v
    )) * 100)

    # Generate forecast
    t_fore   = np.arange(len(income) + 1, len(income) + n_forecast + 1)
    forecast = np.clip(linear(t_fore, a, b), 0, None)

    forecast_df = pd.DataFrame({
        'calendar_month':   range(1, n_forecast + 1),
        'predicted_income': np.round(forecast, 2),
    })

    return forecast_df, mape, float(a), float(b)


def get_supplier_breakdown(master):
    """Monthly income and active donors per supplier."""
    return (
        master[master['paid_flag'] == True]
              .groupby(['supplier', 'donor_month'])
              .agg(
                  active_donors=('recurring_payment_id', 'nunique'),
                  total_income =('success_amt',          'sum'),
              )
              .reset_index()
    )


def get_campaign_roi(master):
    """Income, donors and CPA per campaign type."""
    roi = (
        master[master['paid_flag'] == True]
              .groupby('campaign_type')
              .agg(
                  total_income  =('success_amt',          'sum'),
                  active_donors =('recurring_payment_id', 'nunique'),
                  avg_gift      =('success_amt',          'mean'),
              )
              .reset_index()
    )
    if 'cost_per_acquisition' in master.columns:
        cpa = (
            master.groupby('campaign_type')['cost_per_acquisition']
                  .mean()
                  .reset_index()
                  .rename(columns={'cost_per_acquisition': 'avg_cpa'})
        )
        roi = roi.merge(cpa, on='campaign_type', how='left')
    return roi


def get_retention_by_segment(master, segment='supplier'):
    """Retention percentage at each tenure month, segmented."""
    records = []
    for grp_val in master[segment].dropna().unique():
        grp_df = master[master[segment] == grp_val]
        n      = grp_df['recurring_payment_id'].nunique()
        if n < 50:
            continue
        monthly_ret = (
            grp_df[grp_df['paid_flag'] == True]
                  .groupby('tenure_months')['recurring_payment_id']
                  .nunique()
                  .reset_index()
        )
        monthly_ret['retention_pct'] = monthly_ret['recurring_payment_id'] / n * 100
        monthly_ret[segment]         = str(grp_val)
        records.append(monthly_ret)
    if not records:
        return pd.DataFrame()
    df = pd.concat(records, ignore_index=True)
    return df[df['tenure_months'] >= 1]
