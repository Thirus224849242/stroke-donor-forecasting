"""
sBG (shifted Beta-Geometric) donor-retention forecast, with a per-supplier
BG/NBD fallback a fifth forecasting method alongside Linear/ML/Donor
rollup (Pareto-NBD)/Stock-flow.

Both models are fit and directly compared as independent forecast methods
(not a manual-pick-one fallback pair):
  - sBG (Fader & Hardie, "How to Project Customer Retention," 2007) assumes
    every donor is on the SAME fixed period-length schedule and, once
    lapsed, never returns -- a discrete-time "died" process. Good fit for
    regular monthly F2F recurring giving. No equivalent exists in the
    `lifetimes` package, so this is a from-scratch closed-form MLE
    implementation, verified by recovering known alpha/beta parameters
    from synthetic data (see pipeline tests).
  - BG/NBD (Fader, Hardie & Lee, 2005) relaxes the fixed-schedule
    assumption -- donors can skip periods and still be "alive". This was
    originally ported from a from-scratch reference implementation (the
    notebook this was adapted from), but that version produces incorrect
    -- occasionally negative -- expected-transaction values whenever the
    fitted `a` parameter lands below 1 (a common, not rare, outcome; only
    verified-safe for a>1 by coincidence of the reference notebook's own
    example data). Confirmed via a direct comparison against
    lifetimes.BetaGeoFitter on identical fitted parameters: lifetimes
    returns small sane values, the from-scratch version returns figures
    in the hundreds, wrong by 3+ orders of magnitude. Uses
    lifetimes.BetaGeoFitter instead -- the same well-tested library this
    pipeline already depends on for the Pareto/NBD donor LTV model.

Unlike the notebook this was adapted from (which read a separate
merged_transactions.csv), both models are fit directly from this app's own
`master` donor-month panel -- no separate data source.
"""

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data
from scipy.optimize import minimize
from scipy.special import betaln

MIN_PERIOD = '2019-01'
MIN_DONORS_PER_SUPPLIER = 30
HOLDOUT_MONTHS = 6
SBG_MAPE_THRESHOLD = 15.0       # suppliers under this use sBG; over it fall back to BG/NBD
PENALIZER_GRID = (0.0, 0.01, 0.05, 0.1, 0.5)
RANDOM_SEED = 42


# ═══════════════════════════════════════════════════════════════════════
# sBG (shifted Beta-Geometric)
# ═══════════════════════════════════════════════════════════════════════
#
# Heterogeneity in each donor's constant per-period retention probability
# is captured by a Beta(alpha, beta) mixing distribution across the cohort.
# Fit on AGGREGATE counts (how many donors lapsed at each tenure month, how
# many are still active/censored) rather than one term per donor -- donors
# sharing the same tenure/censoring status have an identical likelihood
# contribution, so aggregating first is exact, not an approximation, and
# is what the original Fader-Hardie paper itself does.

def _sbg_survival(alpha, beta, t):
    """P(a donor is still active beyond period t) = B(alpha, beta+t) / B(alpha, beta).
    t=0 must return 1.0 (everyone is active before period 1 starts)."""
    t = np.asarray(t, dtype=float)
    return np.exp(betaln(alpha, beta + t) - betaln(alpha, beta))


def _sbg_negloglik(params, died_by_tenure, still_active, max_t, penalizer):
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return 1e10
    surv = _sbg_survival(alpha, beta, np.arange(0, max_t + 1))  # surv[t] = P(active beyond t)
    p_churn = surv[:-1] - surv[1:]  # p_churn[t-1] = P(churn exactly in period t), t=1..max_t
    died_counts = np.array([died_by_tenure.get(t, 0) for t in range(1, max_t + 1)])
    with np.errstate(divide='ignore'):
        churn_terms = np.where(died_counts > 0, died_counts * np.log(np.maximum(p_churn, 1e-300)), 0.0)
    ll = churn_terms.sum()
    if still_active:
        ll += still_active * np.log(max(surv[-1], 1e-300))
    penalty = penalizer * (alpha ** 2 + beta ** 2)
    return -ll + penalty


def _fit_sbg(died_by_tenure, still_active, max_t, penalizer, seed=RANDOM_SEED):
    """MLE fit with multiple random restarts -- a single starting point can
    land the optimizer on a degenerate boundary solution without warning
    (same rationale as this file's BG/NBD fit, and ltv_model.py's Pareto/NBD
    fit elsewhere in this pipeline)."""
    rng = np.random.default_rng(seed)
    starts = [[1.0, 1.0]] + [rng.uniform(0.1, 5.0, 2) for _ in range(4)]
    best = None
    for x0 in starts:
        res = minimize(_sbg_negloglik, x0, args=(died_by_tenure, still_active, max_t, penalizer),
                        method='L-BFGS-B', bounds=[(1e-3, 500)] * 2)
        if not np.isfinite(res.fun):
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        raise RuntimeError('sBG: no restart converged')
    return float(best.x[0]), float(best.x[1])  # alpha, beta


def _tenure_counts(master, supplier, min_period, cutoff_month):
    """died_by_tenure: {tenure_month: number of this supplier's donors who
    lapsed at that tenure}, still_active: number still active (censored) as
    of cutoff_month, max_t: the longest tenure any donor could have reached.
    A donor is "lapsed at tenure t" if their last month with a payment
    attempt was t months after recruitment and cutoff_month is later than
    that (matches the "active" definition in walkforward.py/
    gift_waterfall.py: attempts > 0, not just successful payments, so a
    donor who kept trying but failing isn't miscounted as an early lapse)."""
    panel = master[(master['donor_month'] >= pd.Period(min_period, freq='M'))
                    & (master['donor_month'] < cutoff_month)]
    if supplier is not None:
        panel = panel[panel['supplier'] == supplier]
    if panel.empty:
        return {}, 0, 0

    last_active = (panel[panel['attempts'] > 0]
                   .groupby('recurring_payment_id')
                   .agg(last_month=('donor_month', 'max'), recruit_m=('recruit_m', 'first')))
    last_active['tenure_at_last'] = (
        last_active['last_month'].astype('int64') - last_active['recruit_m'].astype('int64')
    )
    last_active = last_active[last_active['tenure_at_last'] >= 1]  # tenure 0 = recruitment month itself
    if last_active.empty:
        return {}, 0, 0

    still_active_mask = last_active['last_month'] == (cutoff_month - 1)
    still_active = int(still_active_mask.sum())
    died = last_active.loc[~still_active_mask, 'tenure_at_last']
    died_by_tenure = died.value_counts().to_dict()
    max_t = int(last_active['tenure_at_last'].max())
    return died_by_tenure, still_active, max_t


# ═══════════════════════════════════════════════════════════════════════
# BG/NBD (Fader, Hardie & Lee 2005) -- fallback model
# ═══════════════════════════════════════════════════════════════════════

def _fit_bgnbd(x, t_x, T, penalizer):
    """Fits via lifetimes.BetaGeoFitter -- see the module docstring for why
    this isn't a from-scratch implementation (unlike sBG above). Raises on
    failure to fit, same contract the from-scratch version had, so callers
    don't need to change."""
    bg = BetaGeoFitter(penalizer_coef=penalizer)
    bg.fit(x, t_x, T)
    p = bg.params_
    return p['r'], p['alpha'], p['a'], p['b']


def _bgnbd_expected_tx(t, x, t_x, T, r, alpha, a, b):
    """Expected number of transactions in the next t periods, conditional on
    calibration-period history (Fader, Hardie & Lee 2005, eq. 10) --
    delegates to lifetimes.BetaGeoFitter's implementation."""
    bg = BetaGeoFitter()
    bg.params_ = {'r': r, 'alpha': alpha, 'a': a, 'b': b}
    return bg.conditional_expected_number_of_purchases_up_to_time(t, x, t_x, T)


def _supplier_bgnbd_rfm(master, supplier, min_period, cutoff_month):
    """Weekly-unit frequency/recency/T per donor, via lifetimes' own
    summary_data_from_transaction_data (freq='W') -- matching the time unit
    ltv_model.py's Pareto/NBD fit already uses elsewhere in this pipeline.
    Only the fitting/forecast MATH below is from-scratch BG/NBD; RFM
    construction is the same well-tested library helper, just fed this
    app's own master panel instead of a separate transactions file."""
    panel = master[(master['donor_month'] >= pd.Period(min_period, freq='M'))
                    & (master['donor_month'] < cutoff_month)
                    & (master['paid_flag'] == True)]
    if supplier is not None:
        panel = panel[panel['supplier'] == supplier]
    if panel.empty:
        return pd.DataFrame()
    txn = panel[['recurring_payment_id']].copy()
    txn['transaction_date'] = panel['donor_month'].dt.to_timestamp()
    rfm = summary_data_from_transaction_data(
        txn, customer_id_col='recurring_payment_id', datetime_col='transaction_date', freq='W',
    )
    return rfm


# ═══════════════════════════════════════════════════════════════════════
# Per-supplier model selection, validation, and combined forecast
# ═══════════════════════════════════════════════════════════════════════

def fit_sbg_bgnbd_models(master, min_period=MIN_PERIOD, holdout_months=HOLDOUT_MONTHS,
                          monthly_horizon=24, progress_callback=None):
    """
    Fits BOTH sBG and BG/NBD INDEPENDENTLY for every supplier with enough
    donors (pooling small suppliers below MIN_DONORS_PER_SUPPLIER into one
    "Other" group, so no supplier's income is silently dropped), and
    combines each model's own per-supplier projections into its OWN full
    monthly income forecast -- so sBG and BG/NBD are each a standalone,
    directly comparable forecast method (same shape as fit_linear_forecast/
    fit_ml_forecast/etc.), not one merged "whichever fit better" series.

    Returns (supplier_results, sbg_monthly_df, bgnbd_monthly_df, metrics).
    supplier_results has one row per supplier PER MODEL attempted, so both
    models' holdout MAPE are visible side by side for the same supplier.
    metrics has 'sbg_holdout_mape' and 'bgnbd_holdout_mape' as separate
    donor-count-weighted averages across suppliers.
    """
    def note(msg):
        if progress_callback:
            progress_callback(msg)

    panel = master[master['donor_month'] >= pd.Period(min_period, freq='M')]
    if panel.empty:
        raise ValueError(f'No data from {min_period} onward to fit sBG/BG-NBD on.')

    full_cutoff = panel['donor_month'].max()  # excluded below, same "drop likely-partial last month" rule as elsewhere
    calib_cutoff = full_cutoff - holdout_months

    donor_counts = panel.groupby('supplier')['recurring_payment_id'].nunique()
    big_suppliers = donor_counts[donor_counts >= MIN_DONORS_PER_SUPPLIER].index.tolist()
    small_suppliers = donor_counts[donor_counts < MIN_DONORS_PER_SUPPLIER].index.tolist()

    groups = [(s, s) for s in big_suppliers]
    if small_suppliers:
        groups.append(('Other (small suppliers)', small_suppliers))

    results = []
    sbg_curves = {}    # label -> (alpha, beta, current_active, avg_gift)
    bgnbd_curves = {}  # label -> (r, alpha, a, b, x_mean, t_x_mean, T_mean, current_active, avg_gift)

    for i, (label, supplier_filter) in enumerate(groups, 1):
        note(f'Fitting supplier {i}/{len(groups)} ({label})…')
        is_pooled = isinstance(supplier_filter, list)

        def _panel_for(cutoff):
            p = panel[panel['donor_month'] < cutoff]
            if is_pooled:
                return p[p['supplier'].isin(supplier_filter)]
            return p[p['supplier'] == supplier_filter]

        cal_panel = _panel_for(calib_cutoff)
        n_donors = cal_panel['recurring_payment_id'].nunique()
        if n_donors < 5:
            results.append({'supplier': label, 'n_donors': n_donors, 'model': None,
                             'status': 'skipped (too few donors)', 'mape': None,
                             'current_active_donors': None, 'avg_gift': None})
            continue

        if is_pooled:
            # _tenure_counts filters by a single supplier value; for the pooled
            # group, build it directly off the already-filtered pooled panel instead.
            sub_panel = master[(master['donor_month'] >= pd.Period(min_period, freq='M'))
                                & (master['donor_month'] < calib_cutoff)
                                & (master['supplier'].isin(supplier_filter))]
            died_cal, active_cal, max_t_cal = {}, 0, 0
            last_active = (sub_panel[sub_panel['attempts'] > 0]
                           .groupby('recurring_payment_id')
                           .agg(last_month=('donor_month', 'max'), recruit_m=('recruit_m', 'first')))
            if not last_active.empty:
                last_active['tenure_at_last'] = (
                    last_active['last_month'].astype('int64') - last_active['recruit_m'].astype('int64')
                )
                last_active = last_active[last_active['tenure_at_last'] >= 1]
                still_mask = last_active['last_month'] == (calib_cutoff - 1)
                active_cal = int(still_mask.sum())
                died_cal = last_active.loc[~still_mask, 'tenure_at_last'].value_counts().to_dict()
                max_t_cal = int(last_active['tenure_at_last'].max()) if len(last_active) else 0
        else:
            died_cal, active_cal, max_t_cal = _tenure_counts(master, supplier_filter, min_period, calib_cutoff)

        holdout_panel = panel[(panel['donor_month'] >= calib_cutoff) & (panel['donor_month'] < full_cutoff)
                               & (panel['paid_flag'] == True)]
        holdout_panel = (holdout_panel[holdout_panel['supplier'].isin(supplier_filter)] if is_pooled
                          else holdout_panel[holdout_panel['supplier'] == supplier_filter])
        # sBG predicts DONOR SURVIVAL, so its validation basis is how many of
        # the calibration-period active donors were still active (paid) at
        # any point during the holdout window -- a donor-count comparison.
        actual_active_holdout = holdout_panel['recurring_payment_id'].nunique()
        # BG/NBD predicts a TRANSACTION COUNT (a donor giving all 6 months of
        # a 6-month holdout contributes 6, not 1) -- comparing that against
        # a unique-donor count would be apples-to-oranges and inflate MAPE
        # by roughly the average holdout giving frequency regardless of fit
        # quality, so it gets its own, transaction-count validation basis.
        actual_tx_holdout = len(holdout_panel)

        full_panel_donors = _panel_for(full_cutoff)
        avg_gift = full_panel_donors.loc[full_panel_donors['paid_flag'] == True, 'success_amt'].mean()
        avg_gift = float(avg_gift) if pd.notna(avg_gift) else 0.0
        # Attempts-based "active" (matches the convention used everywhere
        # else in this pipeline -- gift_waterfall.py, walkforward.py, and
        # sBG's own _tenure_counts above -- so sBG and BG/NBD scale by the
        # same donor count).
        current_active = int(full_panel_donors[full_panel_donors['donor_month'] == (full_cutoff - 1)][
            'recurring_payment_id'].nunique())
        # Separately, PAID-transaction-based "active" -- used only to pick
        # which donors' RFM rows represent "still active" below. The RFM
        # itself is built purely from paid transactions (see
        # _supplier_bgnbd_rfm), so a donor's recency only lines up with
        # "now" if their most recent *paid* month is the most recent one,
        # not merely their most recent attempt.
        paid_active_ids = full_panel_donors[
            (full_panel_donors['donor_month'] == (full_cutoff - 1)) & (full_panel_donors['paid_flag'] == True)
        ]['recurring_payment_id'].unique()

        # ── sBG ──
        if max_t_cal >= 2 and (died_cal or active_cal):
            try:
                alpha, beta = _fit_sbg(died_cal, active_cal, max_t_cal, penalizer=0.01)
                surv_now = _sbg_survival(alpha, beta, max_t_cal)
                surv_holdout_end = _sbg_survival(alpha, beta, max_t_cal + holdout_months)
                # of the donors active now, the fraction expected to still be
                # active `holdout_months` later:
                pred_active_holdout = active_cal * (surv_holdout_end / max(surv_now, 1e-9))
                sbg_mape = abs(pred_active_holdout - actual_active_holdout) / max(actual_active_holdout, 1) * 100
                results.append({'supplier': label, 'n_donors': n_donors, 'model': 'sBG', 'status': 'fit ok',
                                 'mape': sbg_mape, 'current_active_donors': current_active, 'avg_gift': avg_gift})
                if current_active and avg_gift:
                    sbg_curves[label] = (alpha, beta, current_active, avg_gift)
            except Exception as exc:
                results.append({'supplier': label, 'n_donors': n_donors, 'model': 'sBG',
                                 'status': f'fit failed ({exc})', 'mape': None,
                                 'current_active_donors': None, 'avg_gift': None})
        else:
            results.append({'supplier': label, 'n_donors': n_donors, 'model': 'sBG',
                             'status': 'skipped (not enough tenure history)', 'mape': None,
                             'current_active_donors': None, 'avg_gift': None})

        # ── BG/NBD ──
        try:
            rfm_cal = _supplier_bgnbd_rfm(master, None if is_pooled else supplier_filter, min_period, calib_cutoff)
        except Exception:
            rfm_cal = pd.DataFrame()

        if rfm_cal.empty:
            results.append({'supplier': label, 'n_donors': n_donors, 'model': 'BG/NBD',
                             'status': 'skipped (no calibration transactions)', 'mape': None,
                             'current_active_donors': None, 'avg_gift': None})
        else:
            x, t_x, T = rfm_cal['frequency'].values, rfm_cal['recency'].values, rfm_cal['T'].values
            best = None
            for pen in PENALIZER_GRID:
                try:
                    r, alpha, a, b = _fit_bgnbd(x, t_x, T, pen)
                    weeks_holdout = holdout_months * (52 / 12)
                    pred_tx = _bgnbd_expected_tx(weeks_holdout, x, t_x, T, r, alpha, a, b).sum()
                    m = abs(pred_tx - actual_tx_holdout) / max(actual_tx_holdout, 1) * 100
                    if best is None or m < best[0]:
                        best = (m, (r, alpha, a, b))
                except Exception:
                    continue
            if best is None:
                results.append({'supplier': label, 'n_donors': n_donors, 'model': 'BG/NBD',
                                 'status': 'fit failed (no penalizer converged)', 'mape': None,
                                 'current_active_donors': None, 'avg_gift': None})
            else:
                bgnbd_mape, (r, alpha, a, b) = best
                results.append({'supplier': label, 'n_donors': n_donors, 'model': 'BG/NBD', 'status': 'fit ok',
                                 'mape': bgnbd_mape, 'current_active_donors': current_active, 'avg_gift': avg_gift})
                if current_active and avg_gift:
                    # The mean (x, t_x, T) of only the currently-*paid*-active
                    # donors (as of "now", not the calibration cutoff used to
                    # fit/validate above) -- used below to project an
                    # "average donor" forward. Critically, this must be
                    # restricted to donors who are still active, not the
                    # full population's mean RFM: donors who churned long
                    # ago have recency far below T, and averaging them in
                    # drags the "typical donor" profile down to look mostly
                    # inactive, which was making BG/NBD's projection predict
                    # near-zero future transactions even for cohorts that
                    # are, in reality, still giving reliably every month
                    # (verified: this fix took a synthetic near-certain
                    # monthly giver's 1-month-ahead prediction from 0.004 to
                    # 0.93 -- a ~250x error from using the wrong population).
                    # Still a coarser approximation than sBG's exact cohort
                    # projection (one "average" donor stands in for the
                    # whole active cohort's heterogeneity).
                    try:
                        full_rfm = _supplier_bgnbd_rfm(master, None if is_pooled else supplier_filter,
                                                        min_period, full_cutoff)
                        active_rfm = full_rfm.loc[full_rfm.index.isin(paid_active_ids)]
                        rfm_means = ((float(active_rfm['frequency'].mean()), float(active_rfm['recency'].mean()),
                                      float(active_rfm['T'].mean())) if not active_rfm.empty else (0.0, 0.0, 1.0))
                    except Exception:
                        rfm_means = (0.0, 0.0, 1.0)
                    bgnbd_curves[label] = (r, alpha, a, b) + rfm_means + (current_active, avg_gift)

    def _combine_sbg(curves):
        income = np.zeros(monthly_horizon)
        months_ahead = np.arange(1, monthly_horizon + 1)
        for alpha, beta, current_active, avg_gift in curves.values():
            # _sbg_survival(alpha, beta, 0) == 1.0 always (B(a,b)/B(a,b)), so
            # this is directly the fraction of today's active cohort still
            # active `t` months from now -- no separate baseline needed.
            income += current_active * _sbg_survival(alpha, beta, months_ahead) * avg_gift
        return income

    def _combine_bgnbd(curves):
        income = np.zeros(monthly_horizon)
        months_ahead = np.arange(1, monthly_horizon + 1)
        weeks_ahead = months_ahead * (52 / 12)
        for r, alpha, a, b, x_mean, t_x_mean, T_mean, current_active, avg_gift in curves.values():
            per_donor = _bgnbd_expected_tx(weeks_ahead, x_mean, t_x_mean, T_mean, r, alpha, a, b)
            per_donor_incremental = np.diff(np.concatenate([[0], per_donor]))
            income += current_active * np.clip(per_donor_incremental, 0, None) * avg_gift
        return income

    sbg_monthly_df = pd.DataFrame({
        'calendar_month': range(1, monthly_horizon + 1),
        'predicted_income': np.round(_combine_sbg(sbg_curves), 2),
    })
    bgnbd_monthly_df = pd.DataFrame({
        'calendar_month': range(1, monthly_horizon + 1),
        'predicted_income': np.round(_combine_bgnbd(bgnbd_curves), 2),
    })

    results_df = pd.DataFrame(results)

    def _weighted_mape(model_name):
        if results_df.empty or 'model' not in results_df.columns:
            return None
        sub = results_df[(results_df['model'] == model_name) & results_df['mape'].notna()]
        if sub.empty:
            return None
        return float(np.average(sub['mape'], weights=sub['n_donors']))

    metrics = {
        'sbg_holdout_mape': _weighted_mape('sBG'),
        'bgnbd_holdout_mape': _weighted_mape('BG/NBD'),
        'n_suppliers_sbg': len(sbg_curves),
        'n_suppliers_bgnbd': len(bgnbd_curves),
    }

    return results_df, sbg_monthly_df, bgnbd_monthly_df, metrics
