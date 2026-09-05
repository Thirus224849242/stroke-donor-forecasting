# Hosting Options

Where this app can run, what each option costs, and what trade-off it makes. Written for a decision, not for developers (see `architecture.md` if you need the technical detail behind any of this).

## The short version

The app is currently running on Streamlit Community Cloud's free tier. It works and costs nothing, but that tier caps every app at 1 GB of RAM, and a full pipeline run (reading the CSVs, building the donor-month table, fitting three forecasting models) on a real, multi-hundred-MB `Payments.csv` uses most of that. We've already done real work to reduce that memory footprint, streaming the largest file in chunks instead of loading it whole, but a hard 1 GB ceiling is a hard 1 GB ceiling. If the donor base keeps growing, or the payments file gets meaningfully bigger, the free tier will eventually stop being enough no matter how much we optimize the code.

Recommendation: stay on the free tier for now, since it costs nothing and currently works, but budget for a move to a paid host once the data size grows much further, or if pipeline runs start failing. Streamlit Community Cloud's paid tier and Render are the two lowest-effort upgrades.

## What the app needs from a host, regardless of which one we pick

- Runs a single Python web process (`streamlit run app.py`). No special infrastructure.
- Needs enough RAM to hold the uploaded CSVs and the models being fit at once (currently tight at 1 GB, comfortable at 2 GB or more).
- Needs five configuration values (a Google sign-in client ID and secret, and a database connection string) supplied either as a `secrets.toml` file or as plain environment variables. The app already supports both, so no host is ruled out by this requirement.
- The data itself lives in Supabase, a separate, already-hosted database, regardless of where the app runs. Moving hosts does not mean moving or re-entering data.

## Options

### 1. Streamlit Community Cloud, free tier (current)
- Cost: $0.
- RAM: 1 GB, a hard limit that isn't adjustable.
- Effort to set up: none, already running here.
- Best for: early testing and low-stakes demos, which is where we are today.
- Risk: the memory ceiling described above. A pipeline run can fail outright if the uploaded data grows past what 1 GB can hold, and there's no way to buy more headroom on this tier. The only way up is the next option.

### 2. Streamlit Community Cloud, paid workspace
- Cost: starts in the low tens of USD/month per app (Streamlit's own pricing page has current figures, worth checking before committing since these change).
- RAM: meaningfully higher than the free tier, with the exact limit set per plan rather than fixed at 1 GB.
- Effort to set up: minimal, same platform and deployment process, just a plan upgrade. No code changes needed.
- Best for: the natural next step from where we are now, if the free tier's RAM ceiling becomes the actual blocker, which is the most likely failure mode we'd hit first.

### 3. Render
- Cost: roughly USD $7 to $25/month depending on the RAM tier chosen (their pricing page has current figures).
- RAM: choose the tier (1 GB, 2 GB, 4 GB, and so on), so this directly solves the memory ceiling with room to grow.
- Effort to set up: low to moderate. Needs a `Dockerfile` or a simple build/start command (Render can run this app directly via `streamlit run app.py --server.port $PORT`), plus the same five configuration values entered as environment variables.
- Best for: a genuine step up in headroom without managing a server ourselves, a managed platform rather than raw infrastructure.

### 4. Railway or Fly.io
- Cost: similar shape to Render, usage-based, roughly USD $5 to $20/month for the RAM this app needs.
- RAM: chosen per deployment, same flexibility as Render.
- Effort to set up: comparable to Render (a small config file, environment variables, one deploy command).
- Best for: an equally reasonable alternative to Render. The choice between the three mostly comes down to preference and familiarity, not a functional difference for this app.

### 5. Hugging Face Spaces
- Cost: free tier available, with paid tiers adding more RAM and CPU if needed.
- RAM: the free tier is comparable to Streamlit Cloud's free tier (limited); paid "Spaces hardware" tiers go well beyond it.
- Effort to set up: low. The app already has built-in support for this host's environment-variable-based secrets (see `architecture.md`), so this was clearly considered as an option during development.
- Best for: another free-tier fallback if Streamlit Community Cloud's free tier specifically becomes unavailable for some reason, though it doesn't solve the RAM ceiling on its own free tier either.

### 6. A cloud VM (AWS EC2, Azure, or Google Cloud), self-managed
- Cost: most flexible, as little as around USD $10/month for a small always-on instance, or on-demand pricing if it only needs to run during business hours.
- RAM: entirely our choice, no ceiling beyond what we pay for.
- Effort to set up: meaningfully higher. We, or a hosting engineer, are responsible for provisioning the server, keeping the OS patched, restarting the app if it crashes, and setting up HTTPS. None of that is handled for us the way it is on the options above.
- Best for: only worth it once the app's needs (RAM, control, cost at scale) outgrow what a managed platform like Render offers. Not a starting point.

## Comparison at a glance

| Option | Monthly cost | RAM ceiling | Setup effort | Who manages the server |
|---|---|---|---|---|
| Streamlit Cloud (free) | $0 | 1 GB (fixed) | None (current) | Streamlit |
| Streamlit Cloud (paid) | Low tens of USD | Higher, plan-dependent | Minimal | Streamlit |
| Render | ~$7 to $25 | Chosen per tier | Low to moderate | Render |
| Railway / Fly.io | ~$5 to $20 | Chosen per tier | Low to moderate | Railway / Fly.io |
| Hugging Face Spaces | Free / paid tiers | Free tier limited, paid tiers higher | Low | Hugging Face |
| Self-managed VM | ~$10+ (usage-dependent) | No ceiling (our choice) | High | Us |

(Pricing figures above are approximate and change over time. Confirm current rates on each provider's own pricing page before committing to one.)

## What moving hosts actually involves

Because the app already reads its configuration from either a secrets file or environment variables, and the data lives in Supabase separately from wherever the app itself runs, moving from the current free tier to any option above is not a rebuild. It's:
1. Point the new host at the same GitHub repository.
2. Enter the same five configuration values (Google sign-in credentials, database connection string) as that host's environment variables.
3. Deploy.

No code changes, no data migration, and no user-facing disruption beyond a brief switch of the app's URL, which can be pointed at a custom domain on any of the paid options if wanted.
