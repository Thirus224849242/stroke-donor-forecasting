import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import GradientBoostingRegressor

MIN_PERIOD = '2019-01'
RECENCY_HALF_LIFE_MONTHS = 30


def _linear(t, a, b):
    return a * t + b


def _month_numbers(monthly):
    return pd.PeriodIndex(monthly['donor_month'], freq='M').month.to_numpy()


def _origin_features(residual, month_num, i, n_lags):
    """Features knowable at origin month i, built only from real observed
    history up to and including i — never from a previous prediction."""
    feat = {}
    for lag in range(1, n_lags + 1):
        j = i - lag + 1
        feat[f'lag_{lag}'] = residual[j] if j >= 0 else residual[0]
    w3 = residual[max(0, i - 2):i + 1]
    w6 = residual[max(0, i - 5):i + 1]
    feat['roll_mean_3'] = float(np.mean(w3))
    feat['roll_mean_6'] = float(np.mean(w6))
    feat['roll_std_3']  = float(np.std(w3, ddof=1)) if len(w3) > 1 else 0.0
    feat['origin_month'] = int(month_num[i])
    return feat


def _target_row(origin_feat, horizon, target_month):
    row = dict(origin_feat)
    row['horizon'] = horizon
    row['target_month_sin'] = np.sin(2 * np.pi * target_month / 12)
    row['target_month_cos'] = np.cos(2 * np.pi * target_month / 12)
    return row


def _build_training_table(residual, month_num, n_lags, max_horizon, half_life=RECENCY_HALF_LIFE_MONTHS):
    """One row per (origin, horizon) pair: features knowable at the origin
    month, the horizon distance, and the *real* residual that occurred at
    origin + horizon. Turns ~n months of history into ~n x max_horizon
    training examples — the standard 'direct multi-step' trick for getting
    a tree model enough data to learn a horizon-dependent pattern without
    ever chaining predictions into each other.

    Each row also gets a recency weight, exponentially decaying by how many
    months before the end of this series its *origin* falls (half-life
    default 30 months). With 6-7+ years of history, early low-volume years
    can behave quite differently from recent steady-state years (real
    example: this donor base showed +15-16% July->August jumps in
    2019-2020 but a flat +/-1% in 2021-2025) — trained unweighted, those
    early years get an equal vote and can distort a learned seasonal
    pattern that no longer reflects how the program actually behaves now.
    Recency weighting keeps every month of history in the training set
    (nothing is discarded) while trusting recent patterns more."""
    n = len(residual)
    rows = []
    for i in range(n_lags - 1, n):
        origin_feat = _origin_features(residual, month_num, i, n_lags)
        weight = 0.5 ** ((n - 1 - i) / half_life)
        for h in range(1, max_horizon + 1):
            j = i + h
            if j >= n:
                break
            row = _target_row(origin_feat, h, int(month_num[j]))
            row['target'] = residual[j]
            row['weight'] = weight
            rows.append(row)
    return pd.DataFrame(rows)


def _make_model():
    return GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.04,
        subsample=0.85, random_state=42,
    )


def fit_ml_forecast(monthly, n_forecast=24, holdback=12, n_lags=6, min_period=MIN_PERIOD):
    """
    Direct multi-step gradient-boosted trend + residual forecast.

    The linear trend is fit on the full (filtered) series and handled
    linearly — trees can't extrapolate a slope, so the slope is never their
    job. A GradientBoostingRegressor then learns the residual (actual minus
    trend) as a function of lag/rolling/seasonal features AND how many
    months out it's forecasting ("horizon"), trained on every
    (origin-month, horizon) pair in history.

    Critically, every prediction — validation or production — is made from
    the *same fixed, real* origin: nothing is ever fed a previous
    prediction as an input. A recursive one-step-at-a-time forecast (predict
    month 1, feed it in to predict month 2, ...) compounds small biases over
    a long horizon and smooths out the real month-to-month variability the
    actual series has; this avoids that failure mode entirely.

    Trains only on data from `min_period` onward (older history is treated
    as unreliable) and validates on a walk-forward holdout of the most
    recent `holdback` months — train on everything before it, forecast
    those months directly from the last training origin, compare to what
    actually happened.

    Returns forecast_df (calendar_month, predicted_income), mape (on the
    holdout), a dict of feature importances, and the underlying linear
    trend slope (income $/month).
    """
    monthly = monthly[monthly['donor_month'] >= min_period].reset_index(drop=True)
    income = monthly['total_income'].to_numpy(dtype=float)
    n = len(income)

    n_lags = max(1, min(n_lags, n // 6))
    min_needed = n_lags + holdback + 12
    if n < min_needed:
        raise ValueError(
            f'Not enough monthly history from {min_period} onward ({n} months) to train '
            f'the ML forecast with a {holdback}-month holdout. Need at least ~{min_needed} months.'
        )

    month_num = _month_numbers(monthly)
    t_full = np.arange(1, n + 1, dtype=float)

    popt, _ = curve_fit(_linear, t_full, income)
    a, b = float(popt[0]), float(popt[1])
    trend_full = _linear(t_full, a, b)
    residual = income - trend_full

    # ── Validation: train on everything before the holdout, forecast the
    # holdout months directly from that single fixed origin ──
    train_end = n - holdback
    train_table = _build_training_table(residual[:train_end], month_num[:train_end], n_lags, max_horizon=holdback)
    if train_table.empty:
        raise ValueError('Not enough training history to build validation features.')

    feature_cols = [c for c in train_table.columns if c not in ('target', 'weight')]
    val_model = _make_model()
    val_model.fit(train_table[feature_cols], train_table['target'], sample_weight=train_table['weight'])

    origin = train_end - 1
    origin_feat = _origin_features(residual[:train_end], month_num[:train_end], origin, n_lags)
    val_rows = [_target_row(origin_feat, h, int(month_num[origin + h])) for h in range(1, holdback + 1)]
    pred_resid_val = val_model.predict(pd.DataFrame(val_rows)[feature_cols])

    trend_val   = trend_full[origin + 1: origin + 1 + holdback]
    actual_val  = income[origin + 1: origin + 1 + holdback]
    pred_income_val = np.clip(trend_val + pred_resid_val, 0, None)
    mape = float(np.mean(np.abs((pred_income_val - actual_val) / actual_val)) * 100)

    # ── Final model: train on ALL available (filtered) history ──
    max_h = max(n_forecast, holdback)
    full_table = _build_training_table(residual, month_num, n_lags, max_horizon=max_h)
    model = _make_model()
    model.fit(full_table[feature_cols], full_table['target'], sample_weight=full_table['weight'])
    importances = dict(zip(feature_cols, model.feature_importances_.round(4).tolist()))

    last_origin_feat = _origin_features(residual, month_num, n - 1, n_lags)
    forecast_rows = [
        _target_row(last_origin_feat, h, (int(month_num[-1]) + h - 1) % 12 + 1)
        for h in range(1, n_forecast + 1)
    ]
    pred_resid = model.predict(pd.DataFrame(forecast_rows)[feature_cols])
    t_fore = np.arange(n + 1, n + n_forecast + 1, dtype=float)
    trend_fore = _linear(t_fore, a, b)
    forecasts = np.clip(trend_fore + pred_resid, 0, None)

    forecast_df = pd.DataFrame({
        'calendar_month':   range(1, n_forecast + 1),
        'predicted_income': np.round(forecasts, 2),
    })

    return forecast_df, mape, importances, a
