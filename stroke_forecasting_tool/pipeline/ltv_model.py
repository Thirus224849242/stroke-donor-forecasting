import numpy as np
import pandas as pd
from lifetimes import GammaGammaFitter, ParetoNBDFitter
from lifetimes.utils import calibration_and_holdout_data, summary_data_from_transaction_data
from sklearn.metrics import mean_absolute_error

EPSILON = 1e-6
PENALIZER_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)
GAMMA_PENALIZER = 0.01
WEEKS_PER_MONTH = 52 / 12
MIN_PERIOD = '2019-01'
RANDOM_SEED = 42
PNBD_RESTARTS = 5


def _donor_transactions(master, min_period=MIN_PERIOD):
    """One row per donor per month with a successful payment — the event
    stream Pareto/NBD and Gamma-Gamma are fit on. Only transactions from
    min_period onward are kept (older history is treated as unreliable),
    so donor recency/frequency/T are all computed relative to that window.

    The most recent calendar month in `master` is always dropped as likely
    partial/still-accumulating — the same convention get_monthly_actuals()
    uses for the linear/ML models. Without this, the LTV model's "now" (and
    therefore its month-1 forecast) silently lands one calendar month ahead
    of what the other two models — and the shared chart x-axis — treat as
    the last known actual month, which shows up as an unexplained jump at
    the forecast boundary."""
    paid = master[master['paid_flag'] == True].copy()
    last_period = paid['donor_month'].max()
    paid = paid[paid['donor_month'] < last_period]
    paid['transaction_date'] = paid['donor_month'].dt.to_timestamp()
    txn = paid.rename(columns={'success_amt': 'donation_amount'})[
        ['recurring_payment_id', 'contact_id', 'transaction_date', 'donation_amount']
    ]
    # Pareto/NBD and Gamma-Gamma both require strictly positive, non-null
    # monetary values and dates — defensively drop anything that would
    # otherwise silently break the fit or the RFM summary.
    txn = txn.dropna(subset=['recurring_payment_id', 'transaction_date', 'donation_amount'])
    txn = txn[txn['donation_amount'] > 0]
    txn = txn[txn['transaction_date'] >= pd.Timestamp(min_period)]
    return txn


def _clean_rfm(df, freq_col='frequency', recency_col='recency', t_col='T'):
    """Guard against two known lifetimes edge cases that its own input
    validation rejects outright:
      1. A single-transaction donor (frequency == 0) whose recency comes out
         as a tiny non-zero value due to floating-point rounding in the
         library's date-bucketing — mathematically it must be exactly 0
         (there's no gap between a first transaction and itself).
      2. A donor observed at exactly their last possible moment, where
         recency == T — nudged down by an epsilon so the model can
         distinguish "still active" from "T itself".
    """
    df = df.copy()
    zero_freq = df[freq_col] == 0
    df.loc[zero_freq, recency_col] = 0.0
    mask_eq = (~zero_freq) & (df[recency_col] == df[t_col])
    df.loc[mask_eq, recency_col] = df.loc[mask_eq, recency_col] - EPSILON
    return df


def _expected_value_series(freq, monetary):
    """Gamma-Gamma expected donation value for every donor in freq/monetary's
    index, falling back to the eligible-donor mean for donors Gamma-Gamma
    can't score (frequency == 0, i.e. no repeat gift yet)."""
    eligible = (freq > 0) & (monetary > 0)
    value = pd.Series(np.nan, index=freq.index)
    if eligible.any():
        ggf = GammaGammaFitter(penalizer_coef=GAMMA_PENALIZER)
        ggf.fit(freq[eligible], monetary[eligible])
        value.loc[eligible] = ggf.conditional_expected_average_profit(freq[eligible], monetary[eligible])
    fallback_pool = monetary[monetary > 0]
    fallback = float(value.mean()) if value.notna().any() else (
        float(fallback_pool.mean()) if len(fallback_pool) else 0.0
    )
    return value.fillna(fallback), int(eligible.sum())


def _cumulative_purchases_by_month(model, frequency, recency, T, n_months):
    return {
        m: model.conditional_expected_number_of_purchases_up_to_time(m * WEEKS_PER_MONTH, frequency, recency, T)
        for m in range(1, n_months + 1)
    }


def _monthly_rollup(cum_by_month, expected_value, n_months):
    prev_cum = pd.Series(0.0, index=expected_value.index)
    rows = []
    for m in range(1, n_months + 1):
        cum = cum_by_month[m]
        incremental = (cum - prev_cum).clip(lower=0)
        prev_cum = cum
        rows.append({'calendar_month': m, 'predicted_income': round(float((incremental * expected_value).sum()), 2)})
    return pd.DataFrame(rows)


def fit_ltv_model(master, holdout_months=12, horizons=(12, 24), penalizer_grid=PENALIZER_GRID,
                   monthly_horizon=24, progress_callback=None, min_period=MIN_PERIOD):
    """
    Donor-level lifetime value: Pareto/NBD predicts how many more months a
    donor will give in, Gamma-Gamma predicts their expected donation size,
    multiplied together for a predicted $ LTV per donor.

    Also rolls the per-donor predictions up into a monthly aggregate series
    (calendar_month, predicted_income) — the same shape as the linear/ML
    forecasts — and validates it on the same holdback window with an
    income-based MAPE, so all three forecasting methods are comparable.

    Pareto/NBD only ever knows about donors who already exist, so this
    rollup structurally assumes zero future recruitment — it answers "what
    will today's donor base alone contribute," not "what will total org
    income be." For total income including future recruitment, use the
    Linear trend or ML forecast, which model the aggregate monthly series
    directly. (An earlier version of this function tried to bridge that gap
    with a bottom-up new-donor accumulation added on top of this rollup; it
    was removed because it double-counted the recruitment already baked
    into the existing donor base's steady-state size and produced forecasts
    that grew past every historical bound.)

    Only transactions from min_period onward are used (older history is
    treated as unreliable), and validation holds back the most recent
    holdout_months (default 12) — train on 2019 through a year ago, test
    on the last 12 months, same regime as the ML forecast.

    Fits with the L-BFGS-B optimizer rather than lifetimes' Nelder-Mead
    default: on a donor base in the tens of thousands, Nelder-Mead can take
    a minute or more per fit (and is more prone to landing on NaN/infinite
    predictions), where L-BFGS-B converges in a couple of seconds.

    Also fits with a fixed random seed and PNBD_RESTARTS (default 5) restarts,
    keeping the best: Pareto/NBD's likelihood surface can be poorly
    conditioned for some donor populations, and a single restart from the
    library's fixed default starting point can land on meaningfully
    different parameters run to run on identical data (observed directly —
    alpha landing anywhere from ~8 to ~350 across three otherwise-identical
    runs). Restarts make the fit both reproducible and closer to the true
    optimum, at a still-small cost (a few seconds per restart).

    Returns (donor_results, tuning_df, metrics, monthly_forecast_df).
    """
    def note(msg):
        if progress_callback:
            progress_callback(msg)

    # Pareto/NBD's likelihood surface can be poorly conditioned for some donor
    # populations — without a fixed seed and multiple restarts, two runs on
    # identical data can converge to meaningfully different parameters (seen
    # in practice: alpha landing anywhere from ~8 to ~350 run to run). A fixed
    # seed plus a few restarts (keeping the best) makes the fit reproducible.
    np.random.seed(RANDOM_SEED)

    txn = _donor_transactions(master, min_period=min_period)
    if txn['recurring_payment_id'].nunique() < 100:
        raise ValueError(
            'Not enough donors with successful payments to fit a reliable '
            'Pareto/NBD model (need at least ~100).'
        )

    note('Splitting into calibration/holdout donor histories…')
    full_end = txn['transaction_date'].max()
    calibration_end = full_end - pd.DateOffset(months=holdout_months)

    cal_holdout = calibration_and_holdout_data(
        txn, customer_id_col='recurring_payment_id', datetime_col='transaction_date',
        calibration_period_end=calibration_end, observation_period_end=full_end,
        monetary_value_col='donation_amount', freq='W',
    )
    if cal_holdout.empty:
        raise ValueError(
            f'The calibration/holdout split produced no donors (calibration ends '
            f'{calibration_end.date()}, data runs to {full_end.date()}) — the '
            f'donor history may be too short for a {holdout_months}-month holdback.'
        )
    cal_holdout = _clean_rfm(cal_holdout, 'frequency_cal', 'recency_cal', 'T_cal')

    # ── 1. Penalizer grid search, scored on donor repeat-transaction counts ──
    tuning_rows, failures, fitted = [], [], {}
    for i, penalizer in enumerate(penalizer_grid, 1):
        note(f'Testing penalizer {i}/{len(penalizer_grid)} ({penalizer:g})…')
        try:
            model = ParetoNBDFitter(penalizer_coef=penalizer)
            model.fit(cal_holdout['frequency_cal'], cal_holdout['recency_cal'], cal_holdout['T_cal'],
                      fit_method='L-BFGS-B', iterative_fitting=PNBD_RESTARTS)
            pred = model.conditional_expected_number_of_purchases_up_to_time(
                cal_holdout['duration_holdout'], cal_holdout['frequency_cal'],
                cal_holdout['recency_cal'], cal_holdout['T_cal'],
            )
            if pred.isna().any() or np.isinf(pred).any():
                failures.append(f'penalizer {penalizer}: model produced NaN/infinite predictions '
                                 f'({int(pred.isna().sum())} NaN, {int(np.isinf(pred).sum())} inf of {len(pred)})')
                continue

            actual = cal_holdout['frequency_holdout']
            mae = float(mean_absolute_error(actual, pred))
            non_zero = actual > 0
            if not non_zero.any():
                failures.append(f'penalizer {penalizer}: no donors had repeat gifts in the holdout window '
                                 f'(cannot compute MAPE)')
                continue
            mape = float(np.mean(np.abs((actual[non_zero] - pred[non_zero]) / actual[non_zero])) * 100)
            agg_error = float(abs(pred.sum() - actual.sum()) / actual.sum() * 100)

            fitted[penalizer] = model
            tuning_rows.append({
                'penalizer': penalizer, 'mape': mape, 'mae': mae, 'aggregate_error': agg_error,
            })
        except Exception as exc:
            failures.append(f'penalizer {penalizer}: {type(exc).__name__}: {exc}')
            continue

    tuning_df = pd.DataFrame(tuning_rows)
    if tuning_df.empty:
        detail = ' | '.join(failures) if failures else 'no diagnostic detail available'
        raise ValueError(
            f'Pareto/NBD failed to produce a usable fit for any penalizer value tried '
            f'({len(cal_holdout)} donors in the calibration set). Details: {detail}'
        )
    tuning_df = tuning_df.sort_values('mape').reset_index(drop=True)

    best_penalizer = float(tuning_df.iloc[0]['penalizer'])
    holdout_mape = float(tuning_df.iloc[0]['mape'])
    holdout_mae = float(tuning_df.iloc[0]['mae'])

    # ── 2. Income-based validation: actual vs predicted $ per holdout month ──
    # Reuse the winning penalizer's already-fitted calibration model — no need to re-fit it.
    note('Validating income holdout…')
    cal_pnbd = fitted[best_penalizer]
    cal_expected_value, _ = _expected_value_series(cal_holdout['frequency_cal'], cal_holdout['monetary_value_cal'])
    cum_by_month_cal = _cumulative_purchases_by_month(
        cal_pnbd, cal_holdout['frequency_cal'], cal_holdout['recency_cal'], cal_holdout['T_cal'], holdout_months,
    )
    income_forecast_holdout = _monthly_rollup(cum_by_month_cal, cal_expected_value, holdout_months)

    holdout_txn = txn[txn['transaction_date'] > calibration_end].copy()
    holdout_txn['month_idx'] = (
        (holdout_txn['transaction_date'].dt.year - calibration_end.year) * 12
        + (holdout_txn['transaction_date'].dt.month - calibration_end.month)
    )
    actual_by_month = holdout_txn.groupby('month_idx')['donation_amount'].sum()

    income_validation = income_forecast_holdout.rename(columns={'predicted_income': 'predicted'})
    income_validation['actual'] = income_validation['calendar_month'].map(actual_by_month).fillna(0.0)

    nz = income_validation['actual'] > 0
    income_holdout_mape = (
        float(np.mean(np.abs((income_validation.loc[nz, 'actual'] - income_validation.loc[nz, 'predicted'])
                              / income_validation.loc[nz, 'actual'])) * 100)
        if nz.any() else float('nan')
    )

    # ── 3. Final models fit on ALL data, for production forecasts ──
    note('Fitting the final model on full donor history…')
    donor_summary = summary_data_from_transaction_data(
        txn, customer_id_col='recurring_payment_id', datetime_col='transaction_date',
        monetary_value_col='donation_amount', observation_period_end=full_end, freq='W',
    )
    donor_summary = _clean_rfm(donor_summary)

    final_pnbd = ParetoNBDFitter(penalizer_coef=best_penalizer)
    final_pnbd.fit(donor_summary['frequency'], donor_summary['recency'], donor_summary['T'],
                    fit_method='L-BFGS-B', iterative_fitting=PNBD_RESTARTS)

    note(f'Projecting {monthly_horizon} months forward for every donor…')
    max_month = max(monthly_horizon, *horizons)
    cum_by_month = _cumulative_purchases_by_month(
        final_pnbd, donor_summary['frequency'], donor_summary['recency'], donor_summary['T'], max_month,
    )

    donor_summary['expected_donation_value'], n_eligible = _expected_value_series(
        donor_summary['frequency'], donor_summary['monetary_value'],
    )

    for h in horizons:
        donor_summary[f'predicted_transactions_{h}m'] = cum_by_month[h]
        donor_summary[f'predicted_ltv_{h}m'] = cum_by_month[h] * donor_summary['expected_donation_value']

    lookup = txn.drop_duplicates('recurring_payment_id').set_index('recurring_payment_id')['contact_id']
    donor_results = donor_summary.join(lookup).reset_index().rename(columns={'index': 'recurring_payment_id'})

    # ── 4. Monthly aggregate rollup — existing donors only ──
    #
    # Pareto/NBD structurally only knows about donors who already exist, so
    # this rollup assumes zero future recruitment and is scoped to "what will
    # today's donor base alone contribute." An earlier version of this
    # function tried to add a bottom-up "new donor recruitment" component on
    # top of this (accumulating trailing recruitment-rate x retention-curve
    # x gift-size forward from month 1). That was removed: the existing
    # 80k+-donor base already IS the steady-state result of ~90 months of
    # this org's recruitment running continuously, so adding a second,
    # independent "ramp up from zero" accumulation on top double-counted
    # that dynamic and produced forecasts that grew past every historical
    # bound (~$650k/month vs a ~$450k historical ceiling) within the 24-month
    # window. The retention-by-tenure curve driving it also had a real
    # tenure=0→1 discontinuity (2.7%→62%) from payment-processing timing,
    # which made the accumulation jump sharply in its first two months.
    # Total org income *including* future recruitment is what the Linear
    # trend and ML forecast models already estimate, from the aggregate
    # monthly series — that is the right tool for that question.
    note('Rolling up existing-donor forecast by month…')
    existing_donor_rollup = _monthly_rollup(cum_by_month, donor_summary['expected_donation_value'], monthly_horizon)
    monthly_forecast_df = existing_donor_rollup

    gamma_eligible = donor_summary[(donor_summary['frequency'] > 0) & (donor_summary['monetary_value'] > 0)]

    metrics = {
        'total_donors': int(len(donor_summary)),
        'gamma_gamma_eligible': n_eligible,
        'best_penalizer': best_penalizer,
        'holdout_mape': holdout_mape,                  # donor repeat-transaction-count MAPE
        'holdout_mae': holdout_mae,
        'income_holdout_mape': income_holdout_mape,    # $ MAPE, comparable to linear/ML
        'observed_avg_donation': float(gamma_eligible['monetary_value'].mean()),
        'expected_avg_donation': float(gamma_eligible['expected_donation_value'].mean()),
    }
    for h in horizons:
        metrics[f'predicted_transactions_{h}m'] = float(donor_results[f'predicted_transactions_{h}m'].sum())
        metrics[f'predicted_value_{h}m'] = float(donor_results[f'predicted_ltv_{h}m'].sum())

    return donor_results, tuning_df, metrics, monthly_forecast_df
