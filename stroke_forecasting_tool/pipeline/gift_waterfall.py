"""
Donor Gift Waterfall — an MRR-waterfall-style monthly $ bridge:

    Income(t) ~= Income(t-1) + New(t) + Expansion(t) - Contraction(t) - Churned(t)

Built on each donor's actual paid amount (success_amt) over time, not
current_gift_setting: current_gift_setting is a single snapshot joined onto
every historical row for a donor (see build_master.py), so it never varies
within a donor's history in this panel and carries no expansion/contraction
signal at all (verified directly: 0 of 86,286 donors show more than one
distinct value). success_amt does show real, sustained step-changes per
donor (confirmed on real data), so it's the only field that can drive this.

Payment-timing noise (a donor's F2F retry mechanism can cause a failed
attempt to land in one month and its retry the next, or two attempts to
land in the same month) means a raw month-over-month diff of success_amt
would misfire on both false expansions and false contractions. Instead,
each donor's PAID-month amounts are run-length encoded and short runs
(< MIN_RUN consecutive paid months) are merged into the preceding
confirmed level, treating them as noise rather than a real gift change —
a donor's true level only "moves" once it holds for a few months running.
"""

import itertools

import numpy as np
import pandas as pd

MIN_RUN = 3  # consecutive paid months required to confirm a new level


def _confirmed_levels(months, amounts, min_run=MIN_RUN):
    """
    Run-length-encode a donor's rounded paid-month amounts and merge any
    interior run shorter than min_run into the preceding confirmed level
    (the first and last runs are always kept, however short, since they
    anchor the donor's known starting and current level).

    Returns a list of (month, level) marking the calendar month each
    confirmed level first appears, with consecutive duplicate levels
    collapsed after merging.
    """
    vals = [round(a) for a in amounts]
    runs, idx = [], 0
    for val, grp in itertools.groupby(vals):
        run_len = sum(1 for _ in grp)
        runs.append([val, run_len, idx])
        idx += run_len

    kept = [runs[0]]
    for i in range(1, len(runs)):
        val, length, start = runs[i]
        is_last = i == len(runs) - 1
        if length < min_run and not is_last:
            continue
        kept.append([val, length, start])

    collapsed = [kept[0]]
    for val, length, start in kept[1:]:
        if val == collapsed[-1][0]:
            continue
        collapsed.append([val, length, start])

    return [(months[start], val) for val, _length, start in collapsed]


def build_gift_waterfall(master, min_period='2019-01'):
    """
    Returns (monthly_waterfall, donor_events).

    monthly_waterfall: one row per calendar month with new_volume,
    expansion_volume, contraction_volume, churned_volume, income (actual),
    bridged_income (income(t-1) + new + expansion - contraction - churned),
    and residual (income - bridged_income) — the part of the month-to-month
    change NOT explained by a confirmed level change or churn. This is
    expected to be non-zero: a donor having one bad or one lucky payment
    month while still active (retry noise) moves income without being a
    real level change, and isn't bucketed here. Reported explicitly rather
    than forced to reconcile, so it's visible how much of the picture this
    first-pass waterfall actually explains.

    donor_events: per-donor confirmed-level event log (recurring_payment_id,
    month, level, event_type) for auditability — every dollar in the
    monthly waterfall traces back to a row here.
    """
    panel = master[master['donor_month'] >= pd.Period(min_period, freq='M')]
    last_period = panel['donor_month'].max()
    panel = panel[panel['donor_month'] < last_period]

    months = pd.period_range(panel['donor_month'].min(), panel['donor_month'].max(), freq='M')

    # last calendar month each donor had >=1 payment attempt (matches the
    # "active" definition in walkforward.py) -- used only to attribute churn,
    # kept separate from the paid-amount level-tracking above so a donor's
    # gift level isn't disturbed by months they simply failed to pay.
    last_active = panel[panel['attempts'] > 0].groupby('recurring_payment_id')['donor_month'].max()
    panel_last_month = months[-1]

    paid = panel[panel['paid_flag']].sort_values(['recurring_payment_id', 'donor_month'])

    events = []
    for pid, g in paid.groupby('recurring_payment_id', sort=False):
        levels = _confirmed_levels(g['donor_month'].tolist(), g['success_amt'].tolist())
        if not levels:
            continue

        first_month, first_level = levels[0]
        events.append((pid, first_month, first_level, 'new'))

        prev_level = first_level
        for month, level in levels[1:]:
            etype = 'expansion' if level > prev_level else 'contraction'
            events.append((pid, month, level - prev_level, etype))
            prev_level = level

        donor_last_active = last_active.get(pid)
        if donor_last_active is not None and donor_last_active < panel_last_month:
            churn_month = donor_last_active + 1
            if churn_month in months:
                events.append((pid, churn_month, prev_level, 'churn'))

    donor_events = pd.DataFrame(events, columns=['recurring_payment_id', 'month', 'amount', 'event_type'])

    monthly = pd.DataFrame({'month': months})
    for etype, col in [('new', 'new_volume'), ('expansion', 'expansion_volume'),
                        ('contraction', 'contraction_volume'), ('churn', 'churned_volume')]:
        sub = donor_events[donor_events['event_type'] == etype]
        vol = sub.groupby('month')['amount'].sum().abs()
        monthly[col] = monthly['month'].map(vol).fillna(0.0).values

    income = paid.groupby('donor_month')['success_amt'].sum().reindex(months, fill_value=0.0)
    monthly['income'] = income.values
    monthly['bridged_income'] = (
        monthly['income'].shift(1).fillna(monthly['income'].iloc[0])
        + monthly['new_volume'] + monthly['expansion_volume']
        - monthly['contraction_volume'] - monthly['churned_volume']
    )
    monthly['residual'] = monthly['income'] - monthly['bridged_income']
    monthly['month'] = monthly['month'].dt.to_timestamp()

    return monthly, donor_events
