---
title: Stroke Foundation Donor Forecasting
emoji: 📈
colorFrom: teal
colorTo: indigo
sdk: streamlit
sdk_version: "1.61.1"
app_file: stroke_forecasting_tool/app.py
pinned: false
---

# Donor Forecasting Tool

Internal donor-forecasting dashboard for Stroke Foundation Australia's
Face-to-Face regular giving program. See
`stroke_forecasting_tool/README.md` for the full project README, and
`stroke_forecasting_tool/hosting-options.md` for a comparison of hosting
options including this one.

## Configuration on this host

This Space reads its secrets from environment variables (Repository
secrets in the Space's Settings tab) rather than a committed
`secrets.toml` -- see `_bootstrap_secrets_from_env()` at the top of
`stroke_forecasting_tool/app.py`. Set these five:

- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REDIRECT_URI` -- this Space's own URL + `/oauth2callback`
- `GOOGLE_COOKIE_SECRET`
- `DATABASE_URL`

Google sign-in also needs this Space's exact URL added as an authorized
redirect URI in the Google Cloud Console OAuth client used above.
