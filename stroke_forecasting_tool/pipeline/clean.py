import pandas as pd
import numpy as np

VALID_STATES  = ['QLD', 'NSW', 'VIC', 'WA', 'SA', 'TAS', 'ACT', 'NT']
GAP_THRESHOLD = 90


def tidy_columns(df):
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(' ', '_')
                  .str.replace('/', '_')
    )
    return df


def parse_date(series):
    return pd.to_datetime(series, dayfirst=True, errors='coerce')


def clean_contacts(path):
    df = pd.read_csv(path)
    df = tidy_columns(df)
    gender_map = {'F': 'Female', 'Female': 'Female', 'M': 'Male', 'Male': 'Male'}
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map(gender_map).fillna(df['gender'])
    return df


def clean_campaigns(path):
    df = pd.read_csv(path)
    df = tidy_columns(df)
    if 'start_date' in df.columns:
        df['start_date'] = parse_date(df['start_date'])
    if 'cost_per_acquisition' in df.columns:
        df['cost_per_acquisition'] = pd.to_numeric(
            df['cost_per_acquisition'], errors='coerce'
        ).fillna(0)
    return df


def clean_recurring(path):
    df = pd.read_csv(path)
    df = tidy_columns(df)
    for col in ['start_date', 'recruitment_date', 'acquisition_date']:
        if col in df.columns:
            df[col] = parse_date(df[col])
    if 'supplier' in df.columns:
        df['supplier'] = df['supplier'].fillna('Unknown').str.strip()
    return df


def clean_payments(path, progress_callback=None, min_period=None):
    """
    Returns the donor-month PANEL directly (recurring_payment_id,
    donor_month, attempts, successes, failures, success_amt) -- not raw
    payment-level rows. Nothing downstream of build_master() ever touches
    individual payments (payment_id/method_of_payment were parsed and
    then never used again), so on a multi-year, multi-million-row
    Payments.csv, holding every raw row in memory just to group them once
    at the end was the actual cause of the deploy's OOM kill: the final
    concat+groupby needs the whole raw table AND its concatenated copy in
    memory at the same time, on top of string columns (Success, Amount,
    etc.) that are pure overhead once aggregated.

    Aggregating per chunk instead keeps peak memory to roughly one
    500k-row raw chunk plus a running numeric-only panel, which is far
    smaller than the full raw table at any point. attempts/successes/
    failures/success_amt are all sums, so aggregating in two stages
    (per-chunk, then across chunks) is mathematically identical to
    aggregating the whole table at once.

    min_period: e.g. '2019-01' -- rows scheduled before this are dropped
    per-chunk before aggregation (every downstream model already treats
    pre-2019 history as unreliable and filters it out before fitting).
    On this real dataset it turns out to remove very little -- most rows
    already fall after 2019 -- so it's a minor assist, not the main fix;
    the per-chunk aggregation above is what actually addresses the OOM.
    """
    partials = []
    total    = 0
    cutoff   = pd.Timestamp(min_period) if min_period else None
    usecols  = ['Recurring Payment ID', 'Schedule Date', 'Created Date', 'Success', 'Amount']
    for chunk in pd.read_csv(
        path, chunksize=500_000, usecols=usecols,
        dtype={'Success': 'string', 'Amount': 'string'}
    ):
        total += len(chunk)
        chunk = chunk.rename(columns={'Recurring Payment ID': 'recurring_payment_id'})

        sched       = pd.to_datetime(chunk['Schedule Date'], dayfirst=True, errors='coerce')
        created     = pd.to_datetime(chunk['Created Date'],  dayfirst=True, errors='coerce')
        gap         = (created - sched).dt.days.abs()
        use_created = sched.isna() & (gap <= GAP_THRESHOLD)
        date        = sched.where(~use_created, created)
        keep        = date.notna()
        if cutoff is not None:
            keep = keep & (date >= cutoff)

        is_success = chunk['Success'].str.strip().str.upper().eq('YES')
        amount_num = pd.to_numeric(chunk['Amount'], errors='coerce').fillna(0)

        part = pd.DataFrame({
            'recurring_payment_id': chunk.loc[keep, 'recurring_payment_id'].values,
            'donor_month':          date[keep].dt.to_period('M').values,
            'is_success':           is_success[keep].values,
            'success_amt':          amount_num[keep].where(is_success[keep], 0.0).values,
        }).groupby(['recurring_payment_id', 'donor_month']).agg(
            attempts    = ('is_success',  'count'),
            successes   = ('is_success',  'sum'),
            success_amt = ('success_amt', 'sum'),
        ).reset_index()
        part['failures'] = part['attempts'] - part['successes']
        partials.append(part)

        # TEMP diagnostic (flush=True to defeat stdout buffering on the
        # deploy host) -- remove once the OOM bottleneck is confirmed.
        print(f'[diag] payments chunk: {total:,} rows read, partial panel {len(part):,} rows', flush=True)
        if progress_callback:
            progress_callback(total)

    print(f'[diag] all chunks read, combining {len(partials)} partial panels...', flush=True)
    combined = pd.concat(partials, ignore_index=True)
    panel = (
        combined.groupby(['recurring_payment_id', 'donor_month'])
                .agg(attempts=('attempts', 'sum'),
                     successes=('successes', 'sum'),
                     failures=('failures', 'sum'),
                     success_amt=('success_amt', 'sum'))
                .reset_index()
    )
    print(f'[diag] payments aggregation done: {len(panel):,} donor-month rows', flush=True)
    return panel