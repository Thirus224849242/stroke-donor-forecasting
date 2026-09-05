# How Your Data Is Handled

What happens to the four files you upload, from the moment they're received to the finished dataset every dashboard is built from. This describes what the code actually does, not a summary of intent.

## The four files

| File | What it contains | Row grain |
|---|---|---|
| `Payments.csv` | Every card charge attempt | One row per payment attempt |
| `Recurring Payments.csv` | Every donor's regular-giving signup | One row per signup |
| `Contacts.csv` | Donor details | One row per donor |
| `Campaigns.csv` | Recruitment campaigns, including cost per acquisition | One row per campaign |

Nothing is written anywhere permanent until all four are uploaded and validated together. Each file is checked against the column names it's expected to have first, and the pipeline won't start if any file is missing a column it needs.

## Step 1: Cleaning each file individually

Every file goes through the same first pass: column names are lowercased and standardised (so "Contact ID" and "contact_id" are treated the same, regardless of how your export formats them), and every date column is parsed.

Beyond that, each file gets its own specific handling:

**Contacts.** Gender values are standardised (`F`/`Female` to `Female`, `M`/`Male` to `Male`; anything else is left as provided). Postal state is checked against the eight real Australian states and territories (QLD, NSW, VIC, WA, SA, TAS, ACT, NT). A value that doesn't match one of these exactly is treated as unknown for reporting purposes, rather than kept in whatever raw form it arrived in.

**Campaigns.** Cost per acquisition is converted to a number. Anything that can't be read as a number (blank, text, a formatting artifact) is treated as $0 rather than dropping the campaign.

**Recurring Payments.** A missing supplier is labelled "Unknown" rather than left blank, so every donor still shows up correctly in supplier-level reporting even if that field wasn't filled in on the original signup.

**Payments.** This file only reads five columns from your export: the recurring payment ID, the scheduled date, the created date, the success flag, and the amount. Every other column in your original `Payments.csv` (payment ID, method of payment, and anything else your export includes) is not carried into the dataset at all. Nothing downstream of this pipeline ever needed those fields, so they're set aside at the first read instead of being stored and then ignored later.

For each payment, the scheduled date is used as its date. If the scheduled date isn't recorded, the row is currently excluded rather than falling back to the created date (see "A known gap in the current logic" below).

Payments are then grouped down to one row per donor per month immediately, not kept as individual transactions: the number of attempts, successes, failures, and the total successfully collected amount, for that donor in that month. This is the main thing to understand about the finished dataset: it is not a transaction-level ledger, it's a monthly summary per donor. That was a deliberate choice, since nothing downstream (any dashboard, any forecast) needs individual transaction rows, only the monthly totals.

## Step 2: Filtering to a reliable time window

Payments scheduled before January 2019 are excluded before anything else happens. Older records are treated as unreliable history that every forecasting model already discounts, so they're removed at the source instead of being carried through and ignored later.

## Step 3: Building the combined dataset

Starting from the donor-month payment summary, three more things are joined in:

1. Recruitment details, from Recurring Payments: which campaign the donor was recruited through, their donation amount setting, their supplier, and their recruitment date.
2. Campaign details, from Campaigns: the campaign type and its cost per acquisition, matched by campaign code.
3. Donor details, from Contacts: gender, birth year, state, and postcode, matched by donor or contact ID.

Every one of these three joins is a left join. A payment record is never discarded just because a matching recruitment, campaign, or contact record couldn't be found. If a link is missing, that specific field is simply left blank on that row; the payment itself, and every dollar in it, stays in the dataset.

From the recruitment date and the donor's birth year, two things are calculated: how many months into their giving relationship each payment falls (tenure), and the donor's age at the time they were recruited.

## Step 4: The one row-removal rule

After everything above, exactly one rule removes rows from the finished dataset: a payment can't be dated before the donor's own recorded recruitment date. A payment in the same month a donor was recruited is kept, since that's a normal, valid first gift. Only a payment that appears to predate the donor's recruitment entirely is treated as a genuine data error and excluded. This is the only place data is removed for a reason other than falling outside the reporting window (Step 2) or missing an essential field (dates with no schedule, described above).

## What this means for the numbers you see

- Every dashboard, KPI, and forecast is built from the same single dataset produced by the steps above. Nothing on any individual page applies its own separate filtering.
- The dataset behind every dashboard is a monthly summary, not a transaction ledger. If you need transaction-level detail for a specific donor or period, that would need to come from a separate export, not this tool's dashboards.
- Every completed pipeline run is saved in full, so the exact dataset behind any past run can be reloaded and compared later. Nothing is overwritten when a new run is uploaded.

## A known gap in the current logic

The scheduled-date-missing fallback described in Step 1 (falling back to the created date when the scheduled date is absent, provided the two dates are close together) is written into the code but, on inspection, doesn't currently trigger as intended. Rows with no scheduled date are excluded rather than recovered via the created date. In practice this only affects payment rows that are missing their scheduled date entirely, which is expected to be a small share of a typical export. It's worth being aware of, and we can tighten it up if it turns out to matter for your data.
