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


def clean_payments(path, progress_callback=None):
    chunks = []
    total  = 0
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
        chunk       = chunk[date.notna()].copy()
        date        = date[date.notna()]
        chunk['schedule_date'] = date
        chunk['amount_num']    = pd.to_numeric(chunk['Amount'], errors='coerce').fillna(0)
        ok                     = chunk['Success'].str.strip().str.upper().eq('YES')
        chunk['is_success']    = ok
        chunk['success_amt']   = chunk['amount_num'].where(ok, 0.0)
        chunks.append(chunk)
        if progress_callback:
            progress_callback(total)

    return pd.concat(chunks, ignore_index=True)