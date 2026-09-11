import pandas as pd
import numpy as np
from pipeline.clean import (
    clean_contacts, clean_campaigns,
    clean_recurring, clean_payments,
    VALID_STATES
)


def build_master(
    payments_path,
    recurring_path,
    campaigns_path,
    contacts_path,
    progress_callback=None,
    min_period='2019-01'
):
    # ── Load lookup tables ────────────────────────────────
    # TEMP diagnostic prints (flush=True to defeat stdout buffering on the
    # deploy host) -- pinpointing exactly where the app dies before an OOM
    # kill, since the process leaves no traceback when it happens. Remove
    # once the real bottleneck is confirmed.
    contacts  = clean_contacts(contacts_path)
    print(f'[diag] contacts loaded: {len(contacts):,} rows', flush=True)
    campaigns = clean_campaigns(campaigns_path)
    print(f'[diag] campaigns loaded: {len(campaigns):,} rows', flush=True)
    recurring = clean_recurring(recurring_path)
    print(f'[diag] recurring loaded: {len(recurring):,} rows', flush=True)

    if 'campaign' in recurring.columns:
        recurring = recurring.rename(columns={'campaign': 'campaign_code'})
    if 'campaign' in campaigns.columns:
        campaigns = campaigns.rename(columns={'campaign': 'campaign_code'})

    recurring['recruit_m'] = pd.to_datetime(
        recurring['recruitment_date'], errors='coerce'
    ).dt.to_period('M')

    # ── Process payments ──────────────────────────────────
    # clean_payments() now aggregates to the donor-month panel itself,
    # per-chunk, as it streams -- it never holds the full raw payments
    # table in memory. That (not the min_period filter, which turns out
    # to remove very little of this real dataset) is what was blowing
    # past the deploy host's RAM ceiling: the old code held every raw row
    # in a list, then concatenated the whole thing, then grouped it --
    # three copies of a multi-million-row table alive at once.
    print(f'[diag] starting payments read, min_period={min_period}', flush=True)
    panel = clean_payments(payments_path, progress_callback, min_period=min_period)
    print(f'[diag] payments aggregated: {len(panel):,} donor-month rows', flush=True)

    panel['paid_flag']    = panel['successes'] > 0
    panel['success_rate'] = panel['successes'] / panel['attempts'].clip(lower=1)

    # ── Join recurring payments ───────────────────────────
    rec_cols = [
        'recurring_payment_id', 'contact_id', 'campaign_code',
        'donation_amount', 'supplier', 'recruitment_date', 'recruit_m'
    ]
    rec_cols = [c for c in rec_cols if c in recurring.columns]

    panel = panel.merge(recurring[rec_cols], on='recurring_payment_id', how='left')

    panel['recruitment_date'] = pd.to_datetime(
        panel['recruitment_date'], errors='coerce'
    )
    panel['tenure_months'] = (
        panel['donor_month'].astype('int64') -
        panel['recruit_m'].astype('int64')
    )

    # ── Join campaigns ────────────────────────────────────
    camp_cols = ['campaign_code']
    if 'campaign_type' in campaigns.columns:
        camp_cols.append('campaign_type')
    if 'cost_per_acquisition' in campaigns.columns:
        camp_cols.append('cost_per_acquisition')

    if 'campaign_code' in campaigns.columns and 'campaign_code' in panel.columns:
        panel = panel.merge(campaigns[camp_cols], on='campaign_code', how='left')

    # ── Join contacts ─────────────────────────────────────
    if 'mailing_state_province' in contacts.columns:
        contacts['state'] = (
            contacts['mailing_state_province']
            .astype(str).str.strip().str.upper()
            .where(lambda s: s.isin(VALID_STATES))
        )
    if 'mailing_zip_postal_code' in contacts.columns:
        contacts = contacts.rename(columns={'mailing_zip_postal_code': 'postcode'})

    con_cols = ['contact_id']
    for c in ['gender', 'birth_year', 'state', 'postcode']:
        if c in contacts.columns:
            con_cols.append(c)

    if 'contact_id' in panel.columns and 'contact_id' in contacts.columns:
        panel = panel.merge(contacts[con_cols], on='contact_id', how='left')

    # ── Derived columns ───────────────────────────────────
    panel['recruit_year']  = panel['recruit_m'].dt.year
    panel['recruit_month'] = panel['recruit_m'].dt.month

    if 'birth_year' in panel.columns:
        panel['age_at_recruit'] = panel['recruit_year'] - panel['birth_year']

    # Remove only genuine errors  negative tenure
    # tenure_months = 0 means payment in recruitment month  valid
    panel = panel[panel['tenure_months'] >= 0].copy()

    # Final rename
    if 'donation_amount' in panel.columns:
        panel = panel.rename(columns={'donation_amount': 'current_gift_setting'})

    print(f'Master file: {len(panel):,} rows · {panel["recurring_payment_id"].nunique():,} donors · {panel["supplier"].nunique()} suppliers', flush=True)

    return panel