import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from auth import handle_google_redirect, init_session_state, render_login
from db import db_configured, get_run_actuals, get_run_forecasts, get_runs, save_run
from pipeline.build_master import build_master
from pipeline.forecast import (
    fit_linear_forecast,
    get_campaign_roi,
    get_monthly_actuals,
    get_retention_by_segment,
    get_supplier_breakdown,
)
from pipeline.ml_forecast import fit_ml_forecast
from pipeline.ltv_model import fit_ltv_model
from pipeline.stock_flow_forecast import build_production_forecast
from ui import (
    AMBER, BLUE, COLOURS, LINE, MIST, PURPLE, TEAL, TEXT,
    card, chart, empty_state, inject_global_css, kpi,
    page_header, render_sidebar, render_topbar, upload_slot,
)


# ── Cached forecast fits ────────────────────────────────────────────────────
# fit_ml_forecast in particular refits a 300-estimator, direct-multi-step
# gradient-boosting model — several pages call it just to populate a
# comparison table, and Streamlit reruns the *entire* script on every widget
# interaction, so without caching that refit happened on every click, even
# ones unrelated to the forecast (this was the actual cause of the page
# feeling slow to load/interact with, not just a missing spinner). monthly
# only changes after a new pipeline run, which naturally produces a
# different cache key, so no manual invalidation is needed.
@st.cache_data(show_spinner=False)
def cached_ml_forecast(monthly, n_forecast=24):
    return fit_ml_forecast(monthly, n_forecast=n_forecast)


@st.cache_data(show_spinner=False)
def cached_linear_forecast(monthly, n_train=24, n_forecast=24):
    return fit_linear_forecast(monthly, n_train=n_train, n_forecast=n_forecast)

init_session_state()
handle_google_redirect()

st.set_page_config(
    page_title='Stroke Foundation — Donor Forecasting',
    page_icon='🫀',
    layout='centered' if not st.session_state.authenticated else 'wide',
    initial_sidebar_state='expanded',
)

if not st.session_state.authenticated:
    render_login()

inject_global_css()
render_sidebar()
render_topbar()

page = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
if page == 'Data Pipeline':
    page_header('Data pipeline', 'Upload Salesforce exports',
                'Upload all four CSV files below. The pipeline runs automatically once all '
                'four are received and validated.')

    st.markdown(f"""
    <div class="sf-pipeline">
        <div class="sf-pipe-step">
            <div class="sf-pipe-icon p-teal">
                <svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            </div>
            <div><div class="sf-pipe-label">Upload</div><div class="sf-pipe-desc">4 CSV files</div></div>
        </div>
        <div class="sf-pipe-arrow">›</div>
        <div class="sf-pipe-step">
            <div class="sf-pipe-icon p-navy">
                <svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            </div>
            <div><div class="sf-pipe-label">Clean</div><div class="sf-pipe-desc">Standardise</div></div>
        </div>
        <div class="sf-pipe-arrow">›</div>
        <div class="sf-pipe-step">
            <div class="sf-pipe-icon p-navy">
                <svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7"/>
                <rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>
                <rect x="3" y="14" width="7" height="7"/></svg>
            </div>
            <div><div class="sf-pipe-label">Build master</div><div class="sf-pipe-desc">Join all 4</div></div>
        </div>
        <div class="sf-pipe-arrow">›</div>
        <div class="sf-pipe-step">
            <div class="sf-pipe-icon p-purple">
                <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5z"/>
                <path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
            </div>
            <div><div class="sf-pipe-label">Forecast</div><div class="sf-pipe-desc">Run model</div></div>
        </div>
        <div class="sf-pipe-arrow">›</div>
        <div class="sf-pipe-step">
            <div class="sf-pipe-icon p-amber">
                <svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7"/>
                <rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>
                <rect x="3" y="14" width="7" height="7"/></svg>
            </div>
            <div><div class="sf-pipe-label">Dashboard</div><div class="sf-pipe-desc">Views update</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ICON_CARD  = '<rect x="1" y="4" width="22" height="16" rx="2"/><line x1="1" y1="10" x2="23" y2="10"/>'
    ICON_LOOP  = '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>'
    ICON_USERS = '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>'
    ICON_TARGET = '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'

    REQUIRED_PAYMENTS  = ['Recurring Payment ID', 'Schedule Date', 'Created Date', 'Success', 'Amount']
    REQUIRED_RECURRING = ['Recurring Payment ID', 'Contact ID', 'Campaign', 'Donation Amount', 'Recruitment Date']
    REQUIRED_CONTACTS  = ['Contact ID']
    REQUIRED_CAMPAIGNS = ['Campaign', 'Campaign Type']

    running = st.session_state.pipeline_running

    col1, col2 = st.columns(2)
    with col1:
        f_pay, pay_valid = upload_slot('Payments.csv', 'Every card charge attempt · up to 1024 MB',
                                        ICON_CARD, 'p-teal', 'up_p', REQUIRED_PAYMENTS, disabled=running)
        f_rec, rec_valid = upload_slot('Recurring Payments.csv', 'Every donor regular-giving signup',
                                        ICON_LOOP, 'p-navy', 'up_r', REQUIRED_RECURRING, disabled=running)
    with col2:
        f_con, con_valid = upload_slot('Contacts.csv', 'One row per unique donor',
                                        ICON_USERS, 'p-purple', 'up_c', REQUIRED_CONTACTS, disabled=running)
        f_cam, cam_valid = upload_slot('Campaigns.csv', 'One row per recruitment campaign · includes CPA',
                                        ICON_TARGET, 'p-amber', 'up_ca', REQUIRED_CAMPAIGNS, disabled=running)

    all_up    = all([f_pay, f_rec, f_con, f_cam])
    all_valid = all([pay_valid, rec_valid, con_valid, cam_valid])
    n_up = sum(bool(x) for x in [f_pay, f_rec, f_con, f_cam])

    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

    if not all_up:
        st.progress(n_up / 4, text=f'{n_up} of 4 files uploaded — upload the remaining files to enable the pipeline')
    elif not all_valid:
        st.error('One or more files don\'t match what\'s expected for their slot — fix the file(s) flagged '
                  'above before running the pipeline.', icon=':material/error:')
    else:
        run = st.button('Run pipeline', type='primary', icon=':material/play_arrow:',
                         width='stretch', disabled=running)
        if run:
            st.session_state.pipeline_running = True
            st.rerun()

    if st.session_state.pipeline_success_message and not running:
        st.success(st.session_state.pipeline_success_message, icon=':material/check_circle:')
        st.session_state.pipeline_success_message = None

    if st.session_state.pipeline_running:
        tmp = tempfile.mkdtemp()

        def save(f, name):
            path = os.path.join(tmp, name)
            with open(path, 'wb') as out:
                out.write(f.getbuffer())
            return path

        with st.status('Running the forecasting pipeline…', expanded=True) as status:
            with st.spinner('Saving uploads to a secure temporary workspace…'):
                st.write(':material/check_circle: Files received')
                p_path  = save(f_pay, 'Payments.csv')
                r_path  = save(f_rec, 'Recurring Payments.csv')
                c_path  = save(f_con, 'Contacts.csv')
                ca_path = save(f_cam, 'Campaigns.csv')

            with st.spinner('Cleaning and validating all four files, then building the master donor-month file…'):
                prog = st.progress(0.0)

                def on_progress(rows):
                    pct = min(rows / 6_208_604, 1.0)
                    prog.progress(pct, text=f'{rows:,} payment rows processed')

                master = build_master(p_path, r_path, ca_path, c_path, progress_callback=on_progress)
                prog.progress(1.0, text=f'Master file built — {len(master):,} donor-month rows')
                st.write(f':material/check_circle: Master file built — {len(master):,} donor-month rows')

            with st.spinner('Fitting the linear-trend baseline (trained from 2019, 12-month holdout)…'):
                monthly = get_monthly_actuals(master)
                forecast_df_linear, mape_linear, linear_slope, _ = cached_linear_forecast(monthly, n_train=24, n_forecast=24)
                st.write(f':material/check_circle: Linear baseline fitted — {mape_linear:.1f}% MAPE')

            with st.spinner('Training the gradient-boosting forecast model (direct multi-step, trained from 2019, 12-month holdout)…'):
                try:
                    forecast_df_ml, mape_ml, importances, trend_slope = cached_ml_forecast(monthly, n_forecast=24)
                    st.write(f':material/check_circle: ML forecast model trained — {mape_ml:.1f}% MAPE')
                except ValueError:
                    st.write(':material/warning: Not enough monthly history for the ML model — using the linear baseline instead.')
                    forecast_df_ml, mape_ml, importances, trend_slope = forecast_df_linear, mape_linear, {}, linear_slope

            with st.spinner('Fitting Pareto/NBD + Gamma-Gamma donor lifetime-value model…'):
                ltv_progress = st.empty()

                def on_ltv_progress(msg):
                    ltv_progress.caption(msg)

                try:
                    ltv_results, ltv_tuning, ltv_metrics, ltv_monthly = fit_ltv_model(
                        master, progress_callback=on_ltv_progress)
                    ltv_error = None
                    ltv_progress.empty()
                    st.write(f':material/check_circle: Donor LTV model fitted — '
                             f'{ltv_metrics["income_holdout_mape"]:.1f}% MAPE')
                except ValueError as exc:
                    ltv_results, ltv_tuning, ltv_metrics, ltv_monthly = None, None, None, None
                    ltv_error = str(exc)
                    ltv_progress.empty()
                    st.write(f':material/warning: Donor LTV model skipped — {ltv_error}')

            with st.spinner('Fitting the stock-flow model (recruits × retention × gift, SARIMA + cohort survival)…'):
                sf_progress = st.empty()

                def on_sf_progress(msg):
                    sf_progress.caption(msg)

                try:
                    sf_result = build_production_forecast(master, progress_callback=on_sf_progress)
                    sf_error = None
                    sf_progress.empty()
                    sf_mape = sf_result['walkforward_summary']['income_mape'].mean()
                    st.write(f':material/check_circle: Stock-flow model fitted — '
                             f'{sf_mape:.1f}% avg MAPE across 3 walk-forward windows')
                except Exception as exc:
                    sf_result = None
                    sf_error = str(exc)
                    sf_progress.empty()
                    st.write(f':material/warning: Stock-flow model skipped — {sf_error}')

            with st.spinner('Finalising and updating dashboard views…'):
                st.session_state.master             = master
                st.session_state.monthly            = monthly
                st.session_state.forecast_df        = forecast_df_ml
                st.session_state.mape               = mape_ml
                st.session_state.ml_importances     = importances
                st.session_state.ml_trend_slope     = trend_slope
                st.session_state.forecast_df_linear = forecast_df_linear
                st.session_state.mape_linear        = mape_linear
                st.session_state.ltv_results        = ltv_results
                st.session_state.ltv_tuning         = ltv_tuning
                st.session_state.ltv_metrics        = ltv_metrics
                st.session_state.ltv_error          = ltv_error
                st.session_state.ltv_monthly        = ltv_monthly
                if sf_result is not None:
                    st.session_state.sf_walkforward = sf_result['walkforward_summary']
                    st.session_state.sf_components  = sf_result['components_df']
                    st.session_state.sf_zone12      = sf_result['zone12_forecast']
                    st.session_state.sf_zone3       = sf_result['zone3_scenarios']
                    st.session_state.sf_assumptions = sf_result['assumptions']
                st.session_state.sf_error           = sf_error
                st.session_state.pipeline_run       = True
                forecast_df, mape = forecast_df_ml, mape_ml
                st.write(':material/check_circle: Dashboard views updated')

            if db_configured():
                with st.spinner('Saving run to history…'):
                    # .mean() on a pandas Series returns numpy.float64, which
                    # psycopg2 can't bind directly (it was raising a bizarre
                    # "schema np does not exist" error, misreading the repr).
                    sf_mape = float(sf_result['walkforward_summary']['income_mape'].mean()) if sf_result else None
                    saved_run_id = save_run(
                        run_by=(st.session_state.user or {}).get('email'),
                        master_rows=len(master),
                        donor_count=int(master['recurring_payment_id'].nunique()),
                        total_income=float(master[master['paid_flag'] == True]['success_amt'].sum()),
                        mapes={
                            'linear': mape_linear, 'ml': mape_ml,
                            'ltv': ltv_metrics['income_holdout_mape'] if ltv_metrics else None,
                            'stockflow': sf_mape,
                        },
                        forecasts={
                            'linear': forecast_df_linear, 'ml': forecast_df_ml,
                            'ltv': ltv_monthly, 'stockflow': sf_result['zone12_forecast'] if sf_result else None,
                        },
                        actuals_df=monthly,
                    )
                    if saved_run_id:
                        st.write(f':material/check_circle: Run saved to history — {saved_run_id}')
                    else:
                        st.write(f':material/warning: Run history save skipped — {st.session_state.get("db_error", "unknown error")}')

            status.update(label='Pipeline complete', state='complete', expanded=False)

        st.session_state.pipeline_success_message = (
            f'Pipeline complete — {len(master):,} rows · '
            f'{master["recurring_payment_id"].nunique():,} donor signups · '
            f'ML forecast MAPE {mape:.1f}% (linear baseline {mape_linear:.1f}%)'
        )
        st.session_state.pipeline_running = False
        st.rerun()

    if st.session_state.pipeline_run:
        st.markdown('<div style="height:2px;"></div>', unsafe_allow_html=True)
        m = st.session_state.master
        mape_delta = None
        if st.session_state.mape_linear:
            mape_delta = f'{st.session_state.mape - st.session_state.mape_linear:+.1f}pp vs linear'
        with st.container(horizontal=True):
            kpi('Rows in master file', f'{len(m):,}', icon=':material/table_rows:', accent=TEAL)
            kpi('Donor signups', f'{m["recurring_payment_id"].nunique():,}', icon=':material/how_to_reg:', accent=BLUE)
            kpi('Total income reconciled', f'${m["success_amt"].sum():,.0f}', icon=':material/payments:', accent=PURPLE)
            kpi('ML forecast accuracy', f'{st.session_state.mape:.1f}% MAPE', delta=mape_delta,
                delta_color='inverse', icon=':material/verified:', accent=AMBER)


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Overview':
    if not st.session_state.pipeline_run:
        empty_state()

    master      = st.session_state.master
    forecast_df = st.session_state.forecast_df
    monthly     = st.session_state.monthly
    mape        = st.session_state.mape

    current_active = monthly.iloc[-1]['active_donors']
    prev_active    = monthly.iloc[-2]['active_donors']
    current_income = monthly.iloc[-1]['total_income']
    prev_income    = monthly.iloc[-2]['total_income']
    total_hist   = master[master['paid_flag'] == True]['success_amt'].sum()

    page_header('Dashboard', 'Overview',
                f'{master["recurring_payment_id"].nunique():,} donor signups · '
                f'{master["campaign_type"].nunique()} campaign types · '
                f'{master["supplier"].nunique()} suppliers')

    # ── Blended forecast across every independently-validated total-income
    # method (ML, Linear, Stock-flow). Donor rollup is deliberately excluded
    # here: it only estimates income from today's existing donors (assumes
    # zero future recruitment) -- a narrower, different question -- so
    # blending it in would drag the average down in a way that misrepresents
    # total org income. See the Income Forecast page for that model on its
    # own terms. ──
    horizon = 24
    with st.spinner('Blending forecast methods…'):
        linear_df, mape_lin, _, _ = cached_linear_forecast(monthly, n_train=24, n_forecast=horizon)
    methods = {
        'ML forecast': (forecast_df.head(horizon)['predicted_income'].reset_index(drop=True), mape),
        'Linear trend': (linear_df.head(horizon)['predicted_income'].reset_index(drop=True), mape_lin),
    }
    sf_zone12_state = st.session_state.sf_zone12
    sf_zone3_state  = st.session_state.sf_zone3
    sf_walkforward_state = st.session_state.sf_walkforward
    if sf_zone12_state is not None:
        sf_series = sf_zone12_state['predicted_income'].reset_index(drop=True)
        if horizon > len(sf_zone12_state) and sf_zone3_state and 'Base' in sf_zone3_state:
            extra = sf_zone3_state['Base'].head(horizon - len(sf_zone12_state))['predicted_income'].reset_index(drop=True)
            sf_series = pd.concat([sf_series, extra], ignore_index=True)
        sf_mape = float(sf_walkforward_state['income_mape'].mean()) if sf_walkforward_state is not None else None
        methods['Stock-flow model'] = (sf_series.head(horizon), sf_mape)

    blended = pd.DataFrame({name: series.values for name, (series, _) in methods.items()})
    blended_avg = blended.mean(axis=1)
    blended_min = blended.min(axis=1)
    blended_max = blended.max(axis=1)
    blended_slope = float(np.polyfit(range(1, horizon + 1), blended_avg.values, 1)[0])

    method_totals_12 = {name: float(series.head(12).sum()) for name, (series, _) in methods.items()}
    total_12_avg = sum(method_totals_12.values()) / len(method_totals_12)
    total_12_min = min(method_totals_12.values())
    total_12_max = max(method_totals_12.values())

    method_mapes = [m for _, m in methods.values() if m is not None]
    mape_avg = sum(method_mapes) / len(method_mapes)

    income_delta = (current_income - prev_income) / prev_income * 100 if prev_income else 0
    active_delta = int(current_active - prev_active)

    with st.container(horizontal=True):
        kpi('12-month forecast', f'${total_12_avg:,.0f}',
            delta=f'range \\${total_12_min:,.0f}–\\${total_12_max:,.0f} across {len(methods)} methods',
            icon=':material/trending_up:', accent=TEAL, delta_color='off')
        kpi('Active donors', f'{current_active:,.0f}', delta=f'{active_delta:+,} vs prior month',
            icon=':material/group:', accent=BLUE)
        kpi('Current monthly income', f'${current_income:,.0f}', delta=f'{income_delta:+.1f}% vs prior month',
            icon=':material/payments:', accent=PURPLE)
        kpi('Forecast accuracy', f'{mape_avg:.1f}% MAPE', delta=f'avg across {len(methods)} methods',
            icon=':material/verified:', accent=AMBER, delta_color='off')

    recent = monthly.tail(24)
    t_act  = list(range(1, len(recent) + 1))
    t_fore = list(range(len(recent) + 1, len(recent) + horizon + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_fore + t_fore[::-1],
        y=list(blended_max.values) + list(blended_min.values)[::-1],
        fill='toself', fillcolor='rgba(0,137,123,0.10)',
        line=dict(color='rgba(0,0,0,0)'), name=f'Spread across {len(methods)} methods', hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=t_act, y=recent['total_income'].values,
        mode='lines+markers', name='Actual',
        line=dict(color=TEXT, width=2.5), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=t_fore, y=blended_avg.values,
        mode='lines+markers', name='Blended forecast',
        line=dict(color=TEAL, width=2.5, dash='dash'),
        marker=dict(size=5, symbol='square'),
    ))
    fig.add_vline(x=len(recent) + 0.5, line_dash='dot', line_color=LINE, opacity=0.8)
    fig.update_layout(yaxis=dict(tickformat='$,.0f'))

    with card('Monthly income — actual vs blended forecast',
              f'Last 24 months actual · average of {", ".join(methods.keys())} · '
              f'shaded band shows where the methods disagree',
              tag=f'{len(methods)} methods blended', tag_color='green'):
        chart(fig, 280)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        sup = (master[master['paid_flag'] == True].groupby('supplier')['success_amt']
               .sum().reset_index().sort_values('success_amt', ascending=False))
        fig2 = go.Figure(go.Pie(
            labels=sup['supplier'], values=sup['success_amt'],
            hole=0.55, marker_colors=COLOURS, textinfo='percent', textfont_size=10,
        ))
        fig2.update_layout(showlegend=True,
                            legend=dict(orientation='v', x=1, y=0.5, font=dict(size=9)),
                            margin=dict(l=0, r=0, t=0, b=0))
        with card('Income by supplier'):
            chart(fig2, 240)

    with col2:
        camp = (master[master['paid_flag'] == True].groupby('campaign_type')['success_amt']
                .sum().reset_index().sort_values('success_amt', ascending=False))
        fig3 = go.Figure(go.Pie(
            labels=camp['campaign_type'], values=camp['success_amt'],
            hole=0.55, marker_colors=COLOURS, textinfo='percent', textfont_size=10,
        ))
        fig3.update_layout(showlegend=True,
                            legend=dict(orientation='v', x=1, y=0.5, font=dict(size=9)),
                            margin=dict(l=0, r=0, t=0, b=0))
        with card('Income by campaign type'):
            chart(fig3, 240)

    with col3:
        summary = pd.DataFrame({
            'Metric': ['Total rows', 'Unique signups', 'Unique donors', 'Suppliers',
                       'Campaign types', 'Total income', 'Forecast MAPE'],
            'Value': [
                f'{len(master):,}',
                f'{master["recurring_payment_id"].nunique():,}',
                f'{master["contact_id"].nunique():,}',
                f'{master["supplier"].nunique()}',
                f'{master["campaign_type"].nunique()}',
                f'${total_hist:,.0f}',
                f'{mape:.1f}%',
            ],
        })
        with card('Dataset summary'):
            st.dataframe(summary, hide_index=True, width='stretch', height=282)


# ══════════════════════════════════════════════════════════════════════════════
# INCOME FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Income Forecast':
    if not st.session_state.pipeline_run:
        empty_state()

    monthly           = st.session_state.monthly
    ltv_monthly_state = st.session_state.ltv_monthly
    ltv_metrics_state = st.session_state.ltv_metrics

    sf_zone12_state      = st.session_state.sf_zone12
    sf_zone3_state       = st.session_state.sf_zone3
    sf_walkforward_state = st.session_state.sf_walkforward
    sf_assumptions_state = st.session_state.sf_assumptions

    page_header('Forecasting', 'Monthly income forecast',
                'Four independent methods, validated the same way, forecasting the same thing: '
                'a top-down trend model, a machine-learning model, a bottom-up rollup of every donor, '
                'and a stock-flow model that forecasts recruitment, retention, and gift size separately.')

    comp_rows = []
    with st.spinner('Comparing forecast methods…'):
        try:
            cmp_ml_df, cmp_ml_mape, _, _ = cached_ml_forecast(monthly, n_forecast=24)
            comp_rows.append({
                'Method': 'ML forecast (gradient boosting)',
                '12-month total': cmp_ml_df.head(12)['predicted_income'].sum(),
                '24-month total': cmp_ml_df['predicted_income'].sum(),
                'Validation MAPE': cmp_ml_mape,
            })
        except ValueError:
            pass

        cmp_lin_df, cmp_lin_mape, _, _ = cached_linear_forecast(monthly, n_train=24, n_forecast=24)
        comp_rows.append({
            'Method': 'Linear trend',
            '12-month total': cmp_lin_df.head(12)['predicted_income'].sum(),
            '24-month total': cmp_lin_df['predicted_income'].sum(),
            'Validation MAPE': cmp_lin_mape,
        })

    if ltv_monthly_state is not None:
        comp_rows.append({
            'Method': 'Donor rollup (Pareto/NBD + Gamma-Gamma)',
            '12-month total': ltv_monthly_state.head(12)['predicted_income'].sum(),
            '24-month total': ltv_monthly_state['predicted_income'].sum(),
            'Validation MAPE': ltv_metrics_state['income_holdout_mape'],
        })

    if sf_zone12_state is not None:
        sf_24m = sf_zone12_state['predicted_income'].sum()
        if sf_zone3_state and 'Base' in sf_zone3_state:
            sf_24m += sf_zone3_state['Base'].head(24 - len(sf_zone12_state))['predicted_income'].sum()
        comp_rows.append({
            'Method': 'Stock-flow model (recruits × retention × gift)',
            '12-month total': sf_zone12_state.head(12)['predicted_income'].sum(),
            '24-month total': sf_24m,
            'Validation MAPE': float(sf_walkforward_state['income_mape'].mean()),
        })

    with card('Method comparison',
              'Trained on 2019 onward · error measured in $ — not just a vibe check. The stock-flow '
              'model is validated on 3 rolling 12-month walk-forward windows; the others on one fixed '
              '12-month holdout, so its MAPE is not directly comparable, just directionally so.',
              tag=f'{len(comp_rows)} methods', tag_color='blue'):
        st.dataframe(
            pd.DataFrame(comp_rows), hide_index=True, width='stretch',
            column_config={
                '12-month total': st.column_config.NumberColumn(format='dollar'),
                '24-month total': st.column_config.NumberColumn(format='dollar'),
                'Validation MAPE': st.column_config.NumberColumn(format='%.1f%%'),
            },
        )

    model_options = ['ML forecast', 'Linear trend', 'Donor rollup', 'Stock-flow model']
    fc1, fc2, fc3 = st.columns([1.8, 1.3, 1.9], vertical_alignment='bottom')
    with fc1:
        model_choice = st.segmented_control(
            'Forecast model', model_options, default='ML forecast', key='fc_model',
        )
    with fc2:
        horizon = st.segmented_control('Forecast horizon', [12, 18, 24], default=24,
                                        key='fc_h', format_func=lambda x: f'{x} months')
    model_choice = model_choice or 'ML forecast'
    horizon = horizon or 24
    use_ml = model_choice == 'ML forecast'

    importances = {}
    if model_choice == 'ML forecast':
        with fc3:
            st.caption('Trained on 2019 onward · direct multi-step (no recursive chaining) · '
                       '12-month holdout, not a separate training window.')
        try:
            with st.spinner('Fitting the ML forecast…'):
                forecast_df, mape, importances, slope = cached_ml_forecast(monthly, n_forecast=horizon)
            model_desc = ('Gradient-boosted trend + residual model · direct multi-step, trained on 2019+ · '
                          'validated on a 12-month holdout')
            holdback_label = '12-month holdout'
        except ValueError as exc:
            st.warning(str(exc))
            forecast_df, mape, slope, _ = cached_linear_forecast(monthly, n_train=24, n_forecast=horizon)
            model_desc = 'Linear trend (fallback — not enough history for the ML model)'
            holdback_label = '12-month holdout'
            use_ml = False
        n_train = min(36, len(monthly))
    elif model_choice == 'Linear trend':
        with fc3:
            n_train = st.segmented_control('Training window', [18, 24, 36, 48], default=24,
                                            key='fc_t', format_func=lambda x: f'{x}-month window')
        n_train = n_train or 24
        forecast_df, mape, slope, _ = cached_linear_forecast(monthly, n_train=n_train, n_forecast=horizon)
        model_desc = f'Linear trend · last {n_train} months (from 2019 onward) · validated on a 12-month holdout'
        holdback_label = '12-month holdout'
    elif model_choice == 'Donor rollup':
        with fc3:
            st.caption('Pareto/NBD predicts how many more months each existing donor gives; Gamma-Gamma '
                       'predicts their gift size. Existing donors only — assumes zero future recruitment, '
                       'so it will run below the Linear/ML totals. For total org income including future '
                       'recruitment, use Linear trend or ML forecast.')
        n_train = min(36, len(monthly))
        holdback_label = '12-month holdout'
        if ltv_monthly_state is None:
            st.warning(st.session_state.ltv_error or 'The donor lifetime-value model is not available for this dataset.')
            forecast_df, mape, slope, _ = cached_linear_forecast(monthly, n_train=24, n_forecast=horizon)
            model_desc = 'Linear trend (fallback — donor rollup unavailable)'
        else:
            forecast_df = ltv_monthly_state.head(horizon).copy()
            mape = ltv_metrics_state['income_holdout_mape']
            slope = float(np.polyfit(forecast_df['calendar_month'], forecast_df['predicted_income'], 1)[0])
            model_desc = ('Pareto/NBD + Gamma-Gamma, existing donors only · trained from 2019 · '
                          'validated on a 12-month holdout, error in $')
    else:  # Stock-flow model
        with fc3:
            st.caption('Forecasts recruits (SARIMA), lapse rate (cohort survival), and gift size (trend) '
                       'separately, then derives income through the accounting identity — never forecasts '
                       'income directly. Statistical range is months 1–18; months 19–36 are shown as named '
                       'scenarios further down this page, not a point forecast.')
        n_train = min(36, len(monthly))
        holdback_label = '3-window walk-forward avg'
        if sf_zone12_state is None:
            st.warning(st.session_state.sf_error or 'The stock-flow model is not available for this dataset.')
            forecast_df, mape, slope, _ = cached_linear_forecast(monthly, n_train=24, n_forecast=horizon)
            model_desc = 'Linear trend (fallback — stock-flow model unavailable)'
        else:
            base_df = sf_zone12_state[['calendar_month', 'predicted_income', 'band_low', 'band_high']].copy()
            if horizon > len(base_df) and sf_zone3_state and 'Base' in sf_zone3_state:
                need = horizon - len(base_df)
                extra = sf_zone3_state['Base'][['calendar_month', 'predicted_income']].head(need).copy()
                extra['band_low'] = extra['predicted_income'] * 0.80
                extra['band_high'] = extra['predicted_income'] * 1.20
                base_df = pd.concat([base_df, extra], ignore_index=True)
                st.caption(f'Months 19–{horizon} above use the Base scenario\'s central assumptions to fill '
                           f'this chart — see the scenario comparison below for the full Conservative/Optimistic range.')
            forecast_df = base_df.head(horizon)
            mape = float(sf_walkforward_state['income_mape'].mean())
            slope = float(np.polyfit(forecast_df['calendar_month'], forecast_df['predicted_income'], 1)[0])
            model_desc = ('Recruits × retention × gift, derived via the accounting identity · trained from '
                          '2019 · validated on 3 rolling 12-month walk-forward windows, error in $')

    fore  = forecast_df.head(horizon)
    total = fore['predicted_income'].sum()

    with st.container(horizontal=True):
        kpi(f'{horizon}-month total', f'${total:,.0f}', delta='Projected income',
            icon=':material/summarize:', accent=TEAL, delta_color='off')
        kpi('Avg monthly income', f'${fore["predicted_income"].mean():,.0f}', delta='Per month',
            icon=':material/calendar_month:', accent=BLUE, delta_color='off')
        kpi('Validation MAPE', f'{mape:.1f}%', delta=holdback_label,
            icon=':material/verified:', accent=PURPLE, delta_color='off')
        kpi('Underlying trend', f'${slope:+,.0f}/mo', delta='Long-run income slope',
            icon=':material/show_chart:', accent=AMBER, delta_color='off')

    recent = monthly.tail(n_train)
    t_act  = list(range(1, len(recent) + 1))
    t_fore = list(range(len(recent) + 1, len(recent) + horizon + 1))
    split  = len(recent) + 0.5

    fig = go.Figure()
    fig.add_vrect(x0=split, x1=max(t_fore) + 0.5, fillcolor='rgba(0,137,123,0.05)',
                  line_width=0, layer='below')
    if 'band_low' in fore.columns:
        band_lo, band_hi = fore['band_low'].values, fore['band_high'].values
        band_name = 'Confidence band (±5% months 1–6, ±12% months 7–18)'
    else:
        band_lo, band_hi = fore['predicted_income'] * 0.95, fore['predicted_income'] * 1.05
        band_name = '±5% confidence band'
    fig.add_trace(go.Scatter(
        x=t_fore + t_fore[::-1],
        y=list(band_hi) + list(band_lo)[::-1],
        fill='toself', fillcolor='rgba(0,137,123,0.10)',
        line=dict(color='rgba(0,0,0,0)'), name=band_name, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=t_act, y=recent['total_income'].values,
        mode='lines+markers', name='Actual monthly income',
        line=dict(color=TEXT, width=2.5), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=t_fore, y=fore['predicted_income'].values,
        mode='lines+markers', name=f'{horizon}-month forecast',
        line=dict(color=TEAL, width=2.5, dash='dash'),
        marker=dict(size=5, symbol='square'),
    ))
    fig.add_vline(x=split, line_dash='dot', line_color=MIST, opacity=0.8,
                  annotation_text='Forecast start', annotation_font_size=10,
                  annotation_font_color=MIST)
    fig.update_layout(
        yaxis=dict(tickformat='$,.0f', title='Monthly income ($)'),
        xaxis=dict(title='Month number'),
    )

    with card(f'{horizon}-month income forecast', model_desc,
              tag=f'MAPE {mape:.1f}%', tag_color='green'):
        chart(fig, 340)

    if use_ml and importances:
        FEATURE_LABELS = {
            'horizon': 'Months ahead being forecast',
            'origin_month': 'Calendar month at forecast time',
            'target_month_sin': 'Seasonality (month of year)',
            'target_month_cos': 'Seasonality (month of year)',
            'roll_mean_3': '3-month rolling average',
            'roll_mean_6': '6-month rolling average',
            'roll_std_3': '3-month volatility',
        }
        imp = pd.Series(importances)
        imp.index = [FEATURE_LABELS.get(i, f'Income {i.split("_")[-1]} month(s) ago') for i in imp.index]
        imp = imp.groupby(imp.index).sum().sort_values(ascending=True).tail(6)

        fig_imp = go.Figure(go.Bar(
            x=imp.values, y=imp.index, orientation='h',
            marker_color=TEAL,
            text=[f'{v:.0%}' for v in imp.values], textposition='outside',
            textfont=dict(size=11, color=TEXT),
        ))
        fig_imp.update_layout(
            xaxis=dict(title='Relative importance', tickformat='.0%'),
            yaxis=dict(title=''), showlegend=False,
        )
        with card('What is driving this forecast',
                  'Relative importance of each engineered feature in the gradient-boosting model',
                  tag='Model transparency', tag_color='blue'):
            chart(fig_imp, 240)

    col1, col2 = st.columns([3, 1])
    with col1:
        table = fore[['calendar_month', 'predicted_income']].rename(
            columns={'calendar_month': 'Month', 'predicted_income': 'Predicted income'})
        with card('Forecast table'):
            st.dataframe(
                table, hide_index=True, width='stretch', height=300,
                column_config={
                    'Month': st.column_config.NumberColumn(format='%d'),
                    'Predicted income': st.column_config.NumberColumn(format='dollar'),
                },
            )

    with col2:
        summary = pd.DataFrame({
            'Metric': ['12-month total', '24-month total', 'Month 1', f'Month {horizon}'],
            'Value': [
                f'${forecast_df.head(12)["predicted_income"].sum():,.0f}',
                f'${forecast_df["predicted_income"].sum():,.0f}',
                f'${fore.iloc[0]["predicted_income"]:,.0f}',
                f'${fore.iloc[-1]["predicted_income"]:,.0f}',
            ],
        })
        with card('Summary'):
            st.dataframe(summary, hide_index=True, width='stretch', height=180)
            st.download_button('Download CSV', data=fore.to_csv(index=False),
                                file_name='sf_forecast.csv', mime='text/csv',
                                icon=':material/download:', width='stretch')

    if model_choice == 'Stock-flow model' and sf_zone3_state:
        st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
        zone3_fig = go.Figure()
        scenario_colors = {'Conservative': AMBER, 'Base': TEAL, 'Optimistic': PURPLE}
        for name, df3 in sf_zone3_state.items():
            zone3_fig.add_trace(go.Scatter(
                x=df3['calendar_month'], y=df3['predicted_income'],
                mode='lines+markers', name=name,
                line=dict(color=scenario_colors.get(name, TEXT), width=2.5), marker=dict(size=5),
            ))
        zone3_fig.update_layout(
            yaxis=dict(tickformat='$,.0f', title='Monthly income ($)'),
            xaxis=dict(title='Month number'),
        )
        with card('Zone 3 — strategic scenarios (months 19–36)',
                  'Not a statistical point forecast: confidence beyond 18 months is too low for that. '
                  'Three named scenarios, each built by scaling the same fitted recruitment, retention, '
                  'and gift models — the organisation should own which of these it plans around.',
                  tag='Named scenarios', tag_color='orange'):
            chart(zone3_fig, 320)

            assum_rows = []
            for name in ['Conservative', 'Base', 'Optimistic']:
                if name not in sf_assumptions_state:
                    continue
                row = {'Scenario': name}
                row.update(sf_assumptions_state[name])
                df3 = sf_zone3_state[name]
                row['Month 36 income'] = f"${df3.iloc[-1]['predicted_income']:,.0f}"
                assum_rows.append(row)
            st.dataframe(pd.DataFrame(assum_rows), hide_index=True, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# RETENTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Retention Analysis':
    if not st.session_state.pipeline_run:
        empty_state()

    master = st.session_state.master
    page_header('Retention analysis', 'Donor retention curves',
                '% of donors still active at each month since recruitment · observed from historical data.')

    rc1, rc2 = st.columns([2, 3], vertical_alignment='bottom')
    with rc1:
        segment = st.segmented_control(
            'Segment by', ['supplier', 'campaign_type', 'recruit_year'], default='supplier',
            format_func=lambda x: x.replace('_', ' ').title(), key='ret_seg',
        )
    with rc2:
        max_m = st.slider('Months to show', 12, 60, 36, key='ret_m')
    segment = segment or 'supplier'

    ret_df = get_retention_by_segment(master, segment)

    if ret_df.empty:
        with card():
            st.markdown(f'<div style="text-align:center;padding:24px;color:{MIST};">'
                        f'Not enough data for this segment.</div>', unsafe_allow_html=True)
    else:
        ret_df = ret_df[ret_df['tenure_months'] <= max_m]
        fig = go.Figure()
        for i, grp_val in enumerate(ret_df[segment].unique()):
            grp = ret_df[ret_df[segment] == grp_val]
            fig.add_trace(go.Scatter(
                x=grp['tenure_months'], y=grp['retention_pct'],
                mode='lines', name=str(grp_val),
                line=dict(color=COLOURS[i % len(COLOURS)], width=2),
            ))
        fig.add_hline(y=50, line_dash='dot', line_color=MIST, opacity=0.7,
                      annotation_text='50% retention', annotation_font_size=10)
        fig.update_layout(
            yaxis=dict(range=[0, 105], ticksuffix='%', title='Donors still active (%)'),
            xaxis=dict(title='Months since recruitment'),
        )
        with card(f'Retention by {segment.replace("_", " ").title()}',
                  '% of donors still active at each tenure month',
                  tag='Observed', tag_color='blue'):
            chart(fig, 360)

        milestones = [3, 6, 12, 18, 24]
        rows = []
        for grp_val in ret_df[segment].unique():
            grp = ret_df[ret_df[segment] == grp_val]
            row = {segment.replace('_', ' ').title(): str(grp_val)}
            for m in milestones:
                match = grp[grp['tenure_months'] == m]['retention_pct']
                row[f'Month {m}'] = float(match.values[0]) if len(match) else None
            rows.append(row)
        milestone_df = pd.DataFrame(rows)

        col_cfg = {f'Month {m}': st.column_config.NumberColumn(format='%.1f%%') for m in milestones}
        with card('Retention at key milestones'):
            st.dataframe(milestone_df, hide_index=True, width='stretch', column_config=col_cfg)


# ══════════════════════════════════════════════════════════════════════════════
# DONOR LIFETIME VALUE
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Donor Lifetime Value':
    if not st.session_state.pipeline_run:
        empty_state()

    page_header('Donor value', 'Donor lifetime value',
                'Pareto/NBD predicts how many more months each existing donor will keep giving; '
                'Gamma-Gamma predicts their expected donation size — a per-donor $ forecast, not an '
                'aggregate average. Scoped to today\'s donor base only (assumes zero future '
                'recruitment) — see Income forecast for total org income including recruitment.')

    ltv_results = st.session_state.ltv_results
    ltv_tuning  = st.session_state.ltv_tuning
    ltv_metrics = st.session_state.ltv_metrics

    if ltv_results is None:
        with card():
            st.markdown(f"""
            <div style="text-align:center;padding:28px;">
                <div style="font-size:14px;font-weight:700;color:{TEXT};margin-bottom:6px;">
                    Donor lifetime value model unavailable
                </div>
                <div style="font-size:12.5px;color:{MIST};max-width:480px;margin:0 auto;">
                    {st.session_state.ltv_error or 'The model could not be fit on this dataset.'}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.container(horizontal=True):
            kpi('Donors modelled', f'{ltv_metrics["total_donors"]:,}',
                delta=f'{ltv_metrics["gamma_gamma_eligible"]:,} with repeat gifts',
                icon=':material/groups:', accent=TEAL, delta_color='off')
            kpi('Predicted 12-month value', f'${ltv_metrics["predicted_value_12m"]:,.0f}',
                delta='existing donors only', icon=':material/savings:', accent=BLUE, delta_color='off')
            kpi('Predicted 24-month value', f'${ltv_metrics["predicted_value_24m"]:,.0f}',
                delta='existing donors only', icon=':material/savings:', accent=PURPLE, delta_color='off')
            kpi('Holdout MAPE', f'{ltv_metrics["holdout_mape"]:.1f}%',
                delta=f'penalizer {ltv_metrics["best_penalizer"]:g}',
                icon=':material/verified:', accent=AMBER, delta_color='off')

        col1, col2 = st.columns([3, 2])
        with col1:
            top15 = ltv_results.sort_values('predicted_ltv_24m', ascending=False).head(15)
            fig = go.Figure(go.Bar(
                x=top15['predicted_ltv_24m'], y=top15['contact_id'].astype(str),
                orientation='h', marker_color=TEAL,
                text=[f'${v:,.0f}' for v in top15['predicted_ltv_24m']],
                textposition='outside', textfont=dict(size=10, color=TEXT),
            ))
            fig.update_layout(
                xaxis=dict(title='Predicted 24-month value ($)', tickformat='$,.0f'),
                yaxis=dict(title='', autorange='reversed'), showlegend=False,
            )
            with card('Top 15 donors by predicted value',
                      'Highest predicted lifetime value over the next 24 months', tag='Ranked'):
                chart(fig, 380)

        with col2:
            fig2 = go.Figure(go.Histogram(
                x=ltv_results['predicted_ltv_24m'], marker_color=TEXT, nbinsx=30,
            ))
            fig2.update_layout(
                xaxis=dict(title='Predicted 24-month value ($)', tickformat='$,.0f'),
                yaxis=dict(title='Number of donors'), showlegend=False,
            )
            with card('Distribution of predicted value',
                      'How predicted 24-month value is spread across all donors'):
                chart(fig2, 380)

        col3, col4 = st.columns([3, 2])
        with col3:
            table = ltv_results.sort_values('predicted_ltv_24m', ascending=False).head(50).rename(columns={
                'recurring_payment_id': 'Signup ID', 'contact_id': 'Contact',
                'frequency': 'Repeat gifts', 'monetary_value': 'Avg gift ($)',
                'expected_donation_value': 'Expected next gift ($)',
                'predicted_ltv_12m': 'Predicted 12m value', 'predicted_ltv_24m': 'Predicted 24m value',
            })[['Signup ID', 'Contact', 'Repeat gifts', 'Avg gift ($)', 'Expected next gift ($)',
                'Predicted 12m value', 'Predicted 24m value']]
            with card('Top 50 donors', 'Ranked by predicted 24-month value', tag='Actionable list', tag_color='blue'):
                st.dataframe(
                    table, hide_index=True, width='stretch', height=340,
                    column_config={
                        'Avg gift ($)': st.column_config.NumberColumn(format='dollar'),
                        'Expected next gift ($)': st.column_config.NumberColumn(format='dollar'),
                        'Predicted 12m value': st.column_config.NumberColumn(format='dollar'),
                        'Predicted 24m value': st.column_config.NumberColumn(format='dollar'),
                    },
                )
                st.download_button(
                    'Download all donor predictions (CSV)',
                    data=ltv_results.to_csv(index=False),
                    file_name='sf_donor_ltv_predictions.csv', mime='text/csv',
                    icon=':material/download:', width='stretch',
                )

        with col4:
            tune_table = ltv_tuning.rename(columns={
                'penalizer': 'Penalizer', 'mape': 'Holdout MAPE (%)',
                'mae': 'Holdout MAE', 'aggregate_error': 'Aggregate error (%)',
            })
            with card('Model validation', 'Penalizer grid search on a 12-month holdout, trained from 2019', tag='Tuning', tag_color='orange'):
                st.dataframe(
                    tune_table, hide_index=True, width='stretch', height=200,
                    column_config={
                        'Holdout MAPE (%)': st.column_config.NumberColumn(format='%.1f'),
                        'Holdout MAE': st.column_config.NumberColumn(format='%.3f'),
                        'Aggregate error (%)': st.column_config.NumberColumn(format='%.1f'),
                    },
                )
                st.caption(
                    f'Gamma-Gamma: observed avg gift \\${ltv_metrics["observed_avg_donation"]:,.2f} vs '
                    f'model-expected \\${ltv_metrics["expected_avg_donation"]:,.2f}.'
                )


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLIER INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Supplier Insights':
    if not st.session_state.pipeline_run:
        empty_state()

    master = st.session_state.master
    page_header('Supplier insights', 'Which recruiting firms perform best',
                'Retention, lifetime value and income contribution by supplier.')

    sup = (master[master['paid_flag'] == True]
           .groupby('supplier')
           .agg(total_income=('success_amt', 'sum'),
                active_donors=('recurring_payment_id', 'nunique'),
                avg_gift=('success_amt', 'mean'))
           .reset_index().sort_values('total_income', ascending=False))
    top = sup.iloc[0]

    with st.container(horizontal=True):
        kpi('Top supplier', top['supplier'], delta='By total income',
            icon=':material/military_tech:', accent=TEAL, delta_color='off')
        kpi('Top supplier income', f'${top["total_income"]:,.0f}', delta='Historical total',
            icon=':material/payments:', accent=BLUE, delta_color='off')
        kpi('Total suppliers', str(len(sup)), delta='In dataset',
            icon=':material/store:', accent=PURPLE, delta_color='off')
        kpi('Avg gift (top)', f'${top["avg_gift"]:,.2f}', delta='Per payment',
            icon=':material/percent:', accent=AMBER, delta_color='off')

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(
            x=sup['supplier'], y=sup['total_income'],
            marker_color=COLOURS[:len(sup)],
            text=[f'${v/1e6:.2f}M' for v in sup['total_income']],
            textposition='outside', textfont=dict(size=11, color=TEXT),
        ))
        fig.update_layout(yaxis=dict(tickformat='$,.0f', title='Total income ($)'), showlegend=False)
        with card('Total income by supplier'):
            chart(fig, 280)

    with col2:
        fig2 = go.Figure(go.Bar(
            x=sup['supplier'], y=sup['active_donors'],
            marker_color=COLOURS[:len(sup)],
            text=[f'{v:,}' for v in sup['active_donors']],
            textposition='outside', textfont=dict(size=11, color=TEXT),
        ))
        fig2.update_layout(yaxis=dict(title='Active donors'), showlegend=False)
        with card('Active donors by supplier'):
            chart(fig2, 280)

    sup_monthly = get_supplier_breakdown(master)
    sup_monthly['donor_month'] = sup_monthly['donor_month'].astype(str)
    all_sups = [s for s in master['supplier'].dropna().unique() if s != 'Unknown']

    with card('Monthly income trend by supplier',
              'Select suppliers to compare their monthly income trajectory'):
        selected = st.multiselect('Compare suppliers', options=all_sups, default=all_sups[:3],
                                   label_visibility='collapsed')
        if selected:
            fig3 = go.Figure()
            for i, s in enumerate(selected):
                grp = sup_monthly[sup_monthly['supplier'] == s]
                fig3.add_trace(go.Scatter(
                    x=grp['donor_month'], y=grp['total_income'],
                    mode='lines', name=s,
                    line=dict(color=COLOURS[i % len(COLOURS)], width=2),
                ))
            fig3.update_layout(yaxis=dict(tickformat='$,.0f', title='Monthly income ($)'),
                                xaxis_title='Month')
            chart(fig3, 260)
        else:
            st.caption('Select at least one supplier to see the trend line.')

    sup_display = sup.rename(columns={
        'supplier': 'Supplier', 'total_income': 'Total income',
        'active_donors': 'Active donors', 'avg_gift': 'Avg gift',
    })
    with card('Supplier summary'):
        st.dataframe(
            sup_display, hide_index=True, width='stretch',
            column_config={
                'Total income': st.column_config.NumberColumn(format='dollar'),
                'Active donors': st.column_config.NumberColumn(format='%d'),
                'Avg gift': st.column_config.NumberColumn(format='dollar'),
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# CAMPAIGN ROI
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Campaign ROI':
    if not st.session_state.pipeline_run:
        empty_state()

    master = st.session_state.master
    roi = get_campaign_roi(master)

    page_header('Campaign insights', 'Which campaigns perform best',
                'Retention, income and return on acquisition cost by campaign type.')

    best = roi.sort_values('total_income', ascending=False).iloc[0]
    has_cpa = 'avg_cpa' in roi.columns
    cpa_vals = roi[roi['avg_cpa'] > 0]['avg_cpa'] if has_cpa else pd.Series(dtype=float)
    cpa_range = f'\\${cpa_vals.min():.0f} – \\${cpa_vals.max():.0f}' if len(cpa_vals) else 'No CPA data'

    with st.container(horizontal=True):
        kpi('Top income campaign', best['campaign_type'], delta=f'\\${best["total_income"]:,.0f} total',
            icon=':material/campaign:', accent=TEAL, delta_color='off')
        kpi('Campaign types', str(len(roi)), delta='In dataset',
            icon=':material/category:', accent=BLUE, delta_color='off')
        kpi('CPA range', cpa_range, delta='Non-zero cost campaigns',
            icon=':material/paid:', accent=PURPLE, delta_color='off')
        kpi('Free-acquisition income', f'${roi[roi["avg_cpa"] == 0]["total_income"].sum():,.0f}' if has_cpa else 'N/A',
            delta='Free Sales campaigns', icon=':material/redeem:', accent=AMBER, delta_color='off')

    col1, col2 = st.columns(2)
    with col1:
        sorted_roi = roi.sort_values('total_income', ascending=False)
        fig = go.Figure(go.Bar(
            x=sorted_roi['campaign_type'], y=sorted_roi['total_income'],
            marker_color=COLOURS[:len(roi)],
            text=[f'${v/1e6:.2f}M' for v in sorted_roi['total_income']],
            textposition='outside', textfont=dict(size=11, color=TEXT),
        ))
        fig.update_layout(yaxis=dict(tickformat='$,.0f', title='Total income ($)'), showlegend=False)
        with card('Total income by campaign type'):
            chart(fig, 280)

    with col2:
        if has_cpa:
            fig2 = go.Figure(go.Bar(
                x=roi['campaign_type'], y=roi['avg_cpa'],
                marker_color=COLOURS[:len(roi)],
                text=[f'${v:.0f}' if v > 0 else 'Free' for v in roi['avg_cpa']],
                textposition='outside', textfont=dict(size=11, color=TEXT),
            ))
            fig2.update_layout(yaxis=dict(tickformat='$,.0f', title='Avg CPA ($)'), showlegend=False)
            with card('Average cost per acquisition', tag='CPA data', tag_color='orange'):
                chart(fig2, 280)
        else:
            with card('Average cost per acquisition'):
                st.markdown(f'<div style="text-align:center;padding:24px;color:{MIST};">'
                            f'CPA data not found in Campaigns file.</div>', unsafe_allow_html=True)

    cm = (master[master['paid_flag'] == True]
          .groupby(['campaign_type', 'donor_month'])['success_amt'].sum().reset_index())
    cm['donor_month'] = cm['donor_month'].astype(str)
    fig3 = go.Figure()
    for i, ct in enumerate(cm['campaign_type'].dropna().unique()):
        grp = cm[cm['campaign_type'] == ct]
        fig3.add_trace(go.Scatter(
            x=grp['donor_month'], y=grp['success_amt'],
            mode='lines', name=ct,
            line=dict(color=COLOURS[i % len(COLOURS)], width=2),
        ))
    fig3.update_layout(yaxis=dict(tickformat='$,.0f', title='Monthly income ($)'), xaxis_title='Month')
    with card('Monthly income by campaign type'):
        chart(fig3, 260)

    roi_display = roi.sort_values('total_income', ascending=False).rename(columns={
        'campaign_type': 'Campaign type', 'total_income': 'Total income',
        'active_donors': 'Active donors', 'avg_gift': 'Avg gift', 'avg_cpa': 'Avg CPA',
    })
    col_cfg = {
        'Total income': st.column_config.NumberColumn(format='dollar'),
        'Active donors': st.column_config.NumberColumn(format='%d'),
        'Avg gift': st.column_config.NumberColumn(format='dollar'),
    }
    if has_cpa:
        col_cfg['Avg CPA'] = st.column_config.NumberColumn(format='dollar')
    with card('Campaign summary'):
        st.dataframe(roi_display, hide_index=True, width='stretch', column_config=col_cfg)


# ══════════════════════════════════════════════════════════════════════════════
# RUN HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Run History':
    page_header('History', 'Pipeline run history',
                'Every completed pipeline run is snapshotted with a timestamp — not the source data '
                'itself (it changes too often for that to be worth caching), just what each run '
                'produced: summary MAPEs, each model\'s forecast, and the actuals series as they '
                'looked at that moment.')

    if not db_configured():
        with card():
            st.markdown(f"""
            <div style="text-align:center;padding:36px 20px;">
                <div style="font-size:15px;font-weight:700;color:{TEXT};margin-bottom:6px;">
                    Run history isn't set up yet
                </div>
                <div style="font-size:12.5px;color:{MIST};max-width:420px;margin:0 auto;">
                    Add a Supabase connection string to .streamlit/secrets.toml under [database] —
                    every pipeline run will start saving automatically, no other setup needed.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        runs = get_runs()
        if runs.empty:
            with card():
                st.markdown(f"""
                <div style="text-align:center;padding:36px 20px;">
                    <div style="font-size:15px;font-weight:700;color:{TEXT};margin-bottom:6px;">
                        No runs saved yet
                    </div>
                    <div style="font-size:12.5px;color:{MIST};max-width:420px;margin:0 auto;">
                        Run the pipeline on the Data Pipeline page — every completed run is saved
                        here automatically from now on.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            latest = runs.iloc[0]
            with st.container(horizontal=True):
                kpi('Runs saved', f'{len(runs):,}', icon=':material/history:', accent=TEAL, delta_color='off')
                kpi('Latest run', pd.to_datetime(latest['run_at']).strftime('%d %b, %H:%M'),
                    icon=':material/schedule:', accent=BLUE, delta_color='off')
                if pd.notna(latest['mape_ml']):
                    kpi('Latest ML MAPE', f"{latest['mape_ml']:.1f}%", icon=':material/verified:',
                        accent=PURPLE, delta_color='off')
                if pd.notna(latest['mape_stockflow']):
                    kpi('Latest stock-flow MAPE', f"{latest['mape_stockflow']:.1f}%",
                        icon=':material/insights:', accent=AMBER, delta_color='off')

            model_cols = [('mape_linear', 'Linear', TEAL), ('mape_ml', 'ML', BLUE),
                          ('mape_ltv', 'Donor rollup', PURPLE), ('mape_stockflow', 'Stock-flow', AMBER)]
            trend = runs.sort_values('run_at')
            fig = go.Figure()
            for col, name, color in model_cols:
                if trend[col].notna().any():
                    fig.add_trace(go.Scatter(
                        x=trend['run_at'], y=trend[col], mode='lines+markers', name=name,
                        line=dict(color=color, width=2),
                    ))
            fig.update_layout(yaxis=dict(title='MAPE (%)', ticksuffix='%'), xaxis=dict(title='Run'))
            with card('Validation accuracy over time', 'Every model\'s MAPE, one point per pipeline run',
                      tag=f'{len(runs)} runs', tag_color='blue'):
                chart(fig, 260)

            runs_display = runs.rename(columns={
                'run_at': 'Run at', 'run_by': 'Run by', 'master_rows': 'Rows',
                'donor_count': 'Donors', 'total_income': 'Total income',
                'mape_linear': 'Linear MAPE', 'mape_ml': 'ML MAPE',
                'mape_ltv': 'LTV MAPE', 'mape_stockflow': 'Stock-flow MAPE',
            })
            with card('All runs'):
                st.dataframe(
                    runs_display, hide_index=True, width='stretch',
                    column_config={
                        'Total income': st.column_config.NumberColumn(format='dollar'),
                        'Linear MAPE': st.column_config.NumberColumn(format='%.1f%%'),
                        'ML MAPE': st.column_config.NumberColumn(format='%.1f%%'),
                        'LTV MAPE': st.column_config.NumberColumn(format='%.1f%%'),
                        'Stock-flow MAPE': st.column_config.NumberColumn(format='%.1f%%'),
                    },
                )

            run_options = [f"{r['run_id']} — {pd.to_datetime(r['run_at']).strftime('%d %b %Y, %H:%M')}"
                           for _, r in runs.iterrows()]
            picked = st.selectbox('View a specific run', run_options)
            picked_run_id = picked.split(' — ')[0]

            run_forecasts = get_run_forecasts(picked_run_id)
            run_actuals = get_run_actuals(picked_run_id)

            if not run_forecasts.empty:
                model_labels = {'linear': 'Linear', 'ml': 'ML', 'ltv': 'Donor rollup', 'stockflow': 'Stock-flow'}
                model_colors = {'linear': TEAL, 'ml': BLUE, 'ltv': PURPLE, 'stockflow': AMBER}

                fig2 = go.Figure()
                n_act = 0
                if not run_actuals.empty:
                    run_actuals = run_actuals.sort_values('donor_month')
                    n_act = len(run_actuals)
                    fig2.add_trace(go.Scatter(
                        x=list(range(1, n_act + 1)), y=run_actuals['total_income'],
                        mode='lines+markers', name='Actual', line=dict(color=TEXT, width=2.5),
                    ))
                for model_name in run_forecasts['model'].unique():
                    sub = run_forecasts[run_forecasts['model'] == model_name].sort_values('calendar_month')
                    t_fore = list(range(n_act + 1, n_act + len(sub) + 1))
                    fig2.add_trace(go.Scatter(
                        x=t_fore, y=sub['predicted_income'], mode='lines+markers',
                        name=model_labels.get(model_name, model_name),
                        line=dict(color=model_colors.get(model_name, MIST), width=2, dash='dash'),
                    ))
                fig2.update_layout(yaxis=dict(tickformat='$,.0f', title='Monthly income ($)'),
                                    xaxis=dict(title='Month number'))
                with card(f'Snapshot — {picked}', 'Reconstructed exactly as it looked when this run completed'):
                    chart(fig2, 300)
