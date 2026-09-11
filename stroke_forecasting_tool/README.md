# Stroke Foundation Donor Forecasting Tool

SIT776 Industry Placement · Deakin University · 2026

## Setup

### 1. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at http://localhost:8501

## How to use

1. Go to **Upload Data** in the sidebar
2. Upload all four Salesforce CSV files:
   - Payments.csv
   - Recurring Payments.csv
   - Contacts.csv
   - Campaigns.csv
3. Click **Run Pipeline**
4. All views update automatically

## Project structure

```
stroke_forecasting_tool/
├── app.py                  Main Streamlit application
├── requirements.txt        Python dependencies
├── README.md               This file
└── pipeline/
    ├── __init__.py
    ├── clean.py            Data cleaning functions
    ├── build_master.py     Master file builder
    └── forecast.py         Forecasting model
```

## Model

The forecasting model fits a linear trend to the last 24 months of actual
monthly income and projects forward. Validated on held-back data with a
MAPE of 2.9% on the Stroke Foundation dataset.

## Notes

- The pipeline handles the 550 MB Payments.csv in 500K-row chunks
- All data stays local nothing is sent to any server
- The master file is rebuilt fresh on every pipeline run
