# Ownership Transfer Plan - Donor Forecasting Tool

This document sets out how ownership of the Donor Forecasting Tool moves
from the development team to Stroke Foundation. Four services are
involved. Three transfer as part of the standard handover; the fourth
(Google sign-in) is optional and only needs action if Stroke Foundation
chooses to use it. Each section below is broken into the exact steps
whoever performs the work would actually follow.

## Summary

| # | Service | What it is | Currently under | Action |
|---|---|---|---|---|
| 1 | GitHub repository | The app's source code | Developer's personal GitHub account | Transfer ownership to Stroke Foundation |
| 2 | Supabase project | The database (pipeline run history, user accounts, access requests) | Developer's personal Supabase account | Migrate to a Stroke Foundation owned Supabase account |
| 3 | Hosting | Where the live app runs | Developer's personal Streamlit Cloud account | Transfer or re-deploy under a Stroke Foundation owned account |
| 4 | Google sign-in (OAuth) | Powers the "Continue with Google" sign-in option | Developer's personal Google Cloud project | Optional - only needed if Stroke Foundation wants Google sign-in |

## Recommended order

1. GitHub repository
2. Supabase project
3. Hosting
4. Google sign-in, only if wanted
5. Configuration update (always last, once everything above is confirmed
   working under Stroke Foundation's own accounts)

---

## 1. GitHub repository

GitHub has a built-in ownership transfer that preserves the full commit
history - no rebuild required.

**Steps:**
1. Stroke Foundation creates a GitHub account or, better, a GitHub
   organisation (an organisation is the right choice for a nonprofit,
   since ownership then isn't tied to any one person's individual
   account).
2. Stroke Foundation shares the organisation or account name with the
   developer.
3. The developer opens the repository's **Settings -> General -> Danger
   Zone -> Transfer ownership**, and enters that name.
4. Stroke Foundation receives a transfer request by email/GitHub
   notification and accepts it.
5. The repository, with its complete commit history intact, now belongs
   to Stroke Foundation. The developer's access can be reviewed/removed
   from the repo's collaborator settings once everything else below is
   confirmed working.

## 2. Supabase project (database)

This is where every pipeline run, user account, and access request is
stored.

**Steps:**
1. Stroke Foundation creates their own Supabase account (the free tier is
   sufficient to start, the same as the current setup).
2. Check whether project transfer is available: in the current project's
   **Settings -> General**, look for a "Transfer project" option. This is
   the preferred path since it moves all existing data across with no
   manual work, and is available on some Supabase plans.
3. If transfer is available: the developer initiates the transfer to
   Stroke Foundation's new account/organisation; Stroke Foundation
   accepts it.
4. If transfer is not available (fallback path): Stroke Foundation creates
   a fresh, empty Supabase project, and the developer exports the three
   tables (`dashboard_runs`, `users`, `local_accounts`) from the old
   project and imports them into the new one. No data is lost either way,
   it is simply a question of how many manual steps it takes.
5. Either path ends with a new database connection string, from that
   project's **Settings -> Database -> Connection string**. This value
   becomes part of the app's configuration (Section 5).

## 3. Hosting - where the live app runs

The app currently runs on Streamlit Community Cloud's free tier. That
tier caps every app at 1 GB of RAM, which a full pipeline run (processing
the Salesforce CSV exports and fitting three forecasting models) already
uses most of. It works today, but it is worth Stroke Foundation deciding
upfront whether to stay on it or move somewhere with more headroom, since
that decision changes the steps below.

### Option A - stay on Streamlit Community Cloud (simplest)

- Cost: $0/month. Ceiling: 1 GB RAM, fixed, not adjustable on this tier.
- The app's URL becomes something like `<app-name>.streamlit.app`,
  Streamlit's own default domain. A custom domain (for example a
  `stroke-forecast.strokefoundation.org.au` subdomain) is **not
  available on the free tier**; it requires Streamlit's paid workspace
  tier (see Option B).

**Steps:**
1. Stroke Foundation creates their own free account at
   share.streamlit.io.
2. Once the GitHub repository has been transferred (Section 1), Stroke
   Foundation clicks "New app" in the Streamlit Cloud dashboard and
   selects that repository, the `main` branch, and
   `stroke_forecasting_tool/app.py` as the entry-point file.
3. Before the first deploy, the configuration values from Section 5 are
   pasted into that new app's **Settings -> Secrets** page.
4. Deploy. The app is now live under Stroke Foundation's own account.
5. The developer's original app deployment (under their personal
   account) can be deleted once the new one is confirmed working.

### Option B - a host with more RAM and/or a custom domain

If Stroke Foundation wants more headroom than 1 GB, or wants the app to
live under Stroke Foundation's own domain rather than `streamlit.app`,
the realistic options are:

| Host | Cost | RAM | Custom domain |
|---|---|---|---|
| Streamlit Cloud (paid workspace) | Low tens of USD/month | Higher, plan-dependent | Yes, on paid plans |
| Render | Roughly US$7-25/month | Chosen per tier (1-4 GB+) | Yes, free - point a DNS record at Render, they issue the HTTPS certificate automatically |
| Railway / Fly.io | Roughly US$5-20/month | Chosen per tier | Yes, similar setup to Render |

**Steps (using Render as the example; Railway/Fly.io are near-identical):**
1. Stroke Foundation creates a Render account and chooses a paid plan
   with enough RAM (2 GB is a comfortable starting point).
2. Once the GitHub repository has been transferred (Section 1), create a
   new "Web Service" in Render and connect that repository.
3. Set the start command to:
   `streamlit run stroke_forecasting_tool/app.py --server.port $PORT --server.address 0.0.0.0`
4. Add the configuration values from Section 5 under that service's
   **Environment** tab as environment variables (not a secrets file -
   Render does not use `secrets.toml`).
5. Deploy. Render provides a default `<service-name>.onrender.com` URL
   immediately.
6. Only if a custom domain is wanted: in Render's **Settings -> Custom
   Domains**, add the desired domain (for example
   `forecast.strokefoundation.org.au`); Render shows a DNS record
   (usually a CNAME) to create. Stroke Foundation's IT/domain
   administrator adds that record in their own DNS provider. Render
   issues an HTTPS certificate automatically once the DNS record is
   detected, usually within a few minutes to a few hours.
7. Whichever URL is now live (the `.onrender.com` one, or the custom
   domain if set up), that exact URL feeds into Section 5's
   `redirect_uri` value if Google sign-in is used.

**Recommendation:** start with Option A (it's what's already running,
and costs nothing), and revisit Option B only if the 1 GB ceiling
becomes a real problem or a Stroke Foundation branded domain is wanted.

## 4. Google sign-in (optional)

The app supports two ways to sign in: **Google sign-in** and **local
email + password accounts**, managed entirely from within the app's own
Users page. Local accounts work fully independently of Google sign-in;
Stroke Foundation is not required to set up Google sign-in at all for
the app to be fully usable. If it is not wanted, skip this whole section
and move to Section 5, leaving the four `auth` values there blank.

**Steps, if Google sign-in is wanted:**
1. Someone with admin rights on Stroke Foundation's own Google Workspace
   (or any Google account, if there is no Workspace) goes to
   console.cloud.google.com/apis/credentials and creates a new project
   (this is a new project under Stroke Foundation's own Google account,
   not a transfer of the developer's existing one).
2. Under that project, "Create credentials" -> "OAuth client ID" ->
   Application type "Web application".
3. Configure the OAuth consent screen: app name, support email, and the
   `openid`, `email`, and `profile` scopes (the defaults). For
   internal-only use, this can stay in "Testing" mode with Stroke
   Foundation staff added as test users, avoiding Google's verification
   review process entirely.
4. Under "Authorized redirect URIs", add the app's live URL (from
   Section 3) plus `/oauth2callback` - for example
   `https://stroke-forecast.streamlit.app/oauth2callback`. This exact
   value is needed again in Section 5, and the two must always match.
5. Google now shows a Client ID and Client secret. These become two of
   the values entered in Section 5.
6. Once Stroke Foundation's own Google sign-in is confirmed working, the
   developer's personal email address (currently listed as a temporary
   administrator for testing purposes) is removed from `auth.py`'s
   `GOOGLE_ADMIN_EMAILS` and `GOOGLE_EXTRA_ALLOWED_EMAILS` lists - the
   one small code change anywhere in this handover, and only needed for
   this specific cleanup step.

## 5. Configuration - the final step, and exactly where it lives

Once Sections 1-3 (and Section 4, if used) are complete under Stroke
Foundation's own accounts, the app's configuration values are updated to
point at the new database and, if used, the new Google credentials. This
is a deliberate final step, not an afterthought: as long as the
developer's original values remain in use anywhere, the developer
technically still has working access to the database and sign-in.
Updating every value below is what makes the handover complete rather
than symbolic.

**No source code needs to change for any of this** (aside from the one
small cleanup line noted in Section 4, step 6). Every value below is
configuration, and lives in one of two places depending on which host
was chosen in Section 3:

- **Streamlit Community Cloud:** the app's own **Settings -> Secrets**
  page in the Streamlit Cloud dashboard (a text box, not a file in the
  repository). Whatever is pasted there is written into a
  `secrets.toml` file automatically by Streamlit's own platform when
  the app starts.
- **Render, Railway, Fly.io, or any other host:** that host's own
  **environment variables** page or setting. The app already has
  built-in support for reading configuration from environment variables
  instead of a file (`_bootstrap_secrets_from_env()` at the top of
  `app.py`), so no code change is needed for this - it was built in
  from the start specifically so the app is not locked to one host.

**The values, and their exact names on each side:**

| What it is | Streamlit Secrets key | Environment variable name (other hosts) |
|---|---|---|
| Google OAuth client ID | `[auth] client_id` | `GOOGLE_CLIENT_ID` |
| Google OAuth client secret | `[auth] client_secret` | `GOOGLE_CLIENT_SECRET` |
| OAuth redirect URL | `[auth] redirect_uri` | `GOOGLE_REDIRECT_URI` |
| Sign-in cookie secret | `[auth] cookie_secret` | `GOOGLE_COOKIE_SECRET` |
| Database connection string | `[database] url` | `DATABASE_URL` |

The four `auth`/Google values only need to be set at all if Section 4 is
being used; if not, they are simply left blank and the app runs with
local accounts only.

**On Streamlit Cloud specifically**, the Secrets page expects TOML
format, so the four Google values are grouped under a `[auth]` heading
and the database value under `[database]`, for example:

```toml
[auth]
client_id = "<value from Section 4>"
client_secret = "<value from Section 4>"
redirect_uri = "<app URL from Section 3>/oauth2callback"
cookie_secret = "<any long random string>"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

[database]
url = "<value from Section 2>"
```

**The one value that depends on which host and domain were chosen in
Section 3: `redirect_uri` / `GOOGLE_REDIRECT_URI`.** It must be set to
the app's actual public URL plus `/oauth2callback`. Two places need this
exact same value, or Google sign-in fails with a mismatch error:

1. The `redirect_uri` / `GOOGLE_REDIRECT_URI` configuration value above.
2. The **Authorized redirect URIs** list on the Google Cloud OAuth
   client itself (Section 4, step 4) - this lives on Google's side, not
   in this app's configuration, and is easy to miss since it is a
   different system entirely. If the app's URL ever changes later (a
   new custom domain, a move to a different host), this is the value
   that needs updating in both places again.

For local development, this same set of values already lives as a
worked example, with placeholder guidance in the comments above it, in
`stroke_forecasting_tool/.streamlit/secrets.toml`. That file is
git-ignored and never committed, so it is not part of the repository
transfer in Section 1; it is recreated fresh by whoever sets up each new
environment.

No password ever needs to be written into a configuration file for
local accounts - those are created, and their passwords chosen, entirely
through the app's own Users page by whoever Stroke Foundation designates
as administrator.

## What this does not involve

- No rebuild of the application. Aside from the one small line noted in
  Section 4, step 6, no code changes are required by any of the steps
  above - this is almost entirely an accounts-and-configuration
  exercise.
- No data re-entry. Every past pipeline run, user account, and access
  request already stored moves across intact.
- No disruption to how the app is used day to day, beyond a brief
  change of URL once hosting moves (Section 3).
