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
    min_period: e.g. '2019-01' -- rows scheduled before this are dropped
    per-chunk, before they ever join the `chunks` list. Every downstream
    model (ltv_model, forecast, ml_forecast, stock_flow_forecast,
    gift_waterfall, component_forecast) already treats pre-2019 history as
    unreliable and filters it out before fitting -- so on a multi-year
    Payments.csv, those older rows are pure dead weight sitting in memory
    for the whole pipeline run and are the main lever for lowering peak
    RAM (the thing that was OOM-killing the deploy) without touching any
    model's behavior.
    """
    chunks = []
    total  = 0
    kept   = 0
    cutoff = pd.Timestamp(min_period) if min_period else None
    for chunk in pd.read_csv(
        path, chunksize=500_000,
        dtype={'Success': 'string', 'Amount': 'string'}
    ):
        total += len(chunk)

        # Rename before processing so join keys are consistent
        chunk = chunk.rename(columns={
            'Recurring Payment ID': 'recurring_payment_id',
            'Payment ID':           'payment_id',
            'Method of Payment':    'method_of_payment',
            'Schedule Date':        'Schedule Date',
            'Created Date':         'Created Date',
            'Success':              'Success',
            'Amount':               'Amount',
        })

        sched       = pd.to_datetime(chunk['Schedule Date'], dayfirst=True, errors='coerce')
        created     = pd.to_datetime(chunk['Created Date'],  dayfirst=True, errors='coerce')
        gap         = (created - sched).dt.days.abs()
        use_created = sched.isna() & (gap <= GAP_THRESHOLD)
        date        = sched.where(~use_created, created)
        keep        = date.notna()
        if cutoff is not None:
            keep = keep & (date >= cutoff)
        chunk       = chunk[keep].copy()
        date        = date[keep]
        chunk['schedule_date'] = date
        chunk['amount_num']    = pd.to_numeric(chunk['Amount'], errors='coerce').fillna(0)
        ok                     = chunk['Success'].str.strip().str.upper().eq('YES')
        chunk['is_success']    = ok
        chunk['success_amt']   = chunk['amount_num'].where(ok, 0.0)
        chunks.append(chunk)
        kept += len(chunk)
        # TEMP diagnostic (flush=True to defeat stdout buffering on the
        # deploy host) -- remove once the OOM bottleneck is confirmed.
        print(f'[diag] payments chunk: {total:,} read so far, {kept:,} kept post-filter', flush=True)
        if progress_callback:
            progress_callback(total)

    print(f'[diag] all chunks read, concatenating {len(chunks)} chunks ({kept:,} rows)...', flush=True)
    result = pd.concat(chunks, ignore_index=True)
    print('[diag] concat done', flush=True)
    return result