# Handover Runbook  Full Ownership Transfer

What needs to move to the client, in what order, and what each step actually involves. Five separate services are involved right now, all under your own personal accounts  none of them can be handed over by sharing a password or a config file alone, since each has its own account/ownership model.

## The five things that need new ownership

| # | Service | What it is | Currently under |
|---|---|---|---|
| 1 | GitHub repository | The app's source code | Your personal GitHub account (`Thirus224849242`), private repo |
| 2 | Supabase project | The database (run history, access requests, local accounts) | Your personal Supabase account |
| 3 | Google Cloud OAuth app | Powers "Continue with Google" sign-in | Your personal Google Cloud project |
| 4 | Hosting (Streamlit Community Cloud) | Where the live app actually runs | Your personal Streamlit Cloud account |
| 5 | Secrets (`secrets.toml`) | Connects all of the above together | A local file on your machine, not committed to git |

Recommended order: **1 → 2 → 3 → 4 → 5**. Rotate the secrets last, once everything else is confirmed working under the client's own accounts  that way nothing breaks mid-transfer, and you don't end up rotating a credential twice.

## 1. GitHub repository

GitHub has a built-in transfer feature that preserves the full commit history, issues, and PRs  no rebuild needed.

- The client needs a GitHub account or organization first (an organization is the better choice for a nonprofit  it isn't tied to one person leaving).
- From the repo's Settings → General → Danger Zone → **Transfer ownership**, enter the client's GitHub username/org. They accept the transfer, and the repo moves under their name with everything intact.
- After transfer, update your local clone's remote URL (`git remote set-url origin <new-url>`) if you're still doing any work on it.

## 2. Supabase project

- The client needs their own Supabase account (free tier is fine to start, same as this project's current setup).
- Supabase supports project transfer between organizations on some plans  check the "Transfer project" option under the current project's Settings first, since it preserves all existing data (every past pipeline run, every access request, every local account) with zero migration work.
- If transfer isn't available on your plan: the fallback is the client creates a **fresh** Supabase project, and you migrate the data  `pg_dump` from the old project's connection string, `psql`/`pg_restore` into the new one (three tables: `dashboard_runs`, `users`, `local_accounts`). This does lose nothing either, it's just more manual steps.
- Either way, the result is a new connection string for `secrets.toml`'s `[database]` section (step 5).

## 3. Google Cloud OAuth app

This one's worth doing fresh rather than transferring, even though Google Cloud projects technically can be transferred  a nonprofit's Google sign-in should live under **their own Google Workspace admin console**, not inherit a project originally created under a developer's personal Google account. It's a five-minute setup, not a migration:

- The client (someone with admin rights on `strokefoundation.org.au`'s Google Workspace, if they have one  or any Google account otherwise) goes to [console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials), creates a new project, and creates an OAuth Client ID the same way the current one was set up (the exact steps are already documented as comments at the top of `secrets.toml`).
- This produces a new `client_id` and `client_secret` for step 5.
- Once this is live, remove `thirumalreddyenugu@gmail.com` from `ALLOWED_GOOGLE_DOMAINS`'s companion lists in `auth.py` (`GOOGLE_ADMIN_EMAILS` and `GOOGLE_EXTRA_ALLOWED_EMAILS`)  those were dev-access exceptions for you personally, not meant to persist past handover. The code comment above them already says as much.

## 4. Hosting

- The client sets up their own account on whichever host you land on (see `hosting-options.md` for the comparison  Streamlit Community Cloud's free tier is the simplest starting point, same as today).
- They connect that hosting account to the **now-transferred** GitHub repo (step 1) and deploy from it.
- The app's own secrets-bootstrapping code already supports both a `secrets.toml` file and plain environment variables (see `architecture.md`), so this step works the same regardless of which host from `hosting-options.md` they pick.

## 5. Secrets  rotate everything, don't just copy

Once steps 1–4 are done under the client's own accounts, every value in `secrets.toml` needs to be the **new** one, not the old one carried over:

```toml
[auth]
redirect_uri = "<the new host's URL>/oauth2callback"
cookie_secret = "<generate a new random string>"
client_id = "<from step 3>"
client_secret = "<from step 3>"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

[database]
url = "<from step 2>"
```

Why rotate rather than reuse: as long as the *old* credentials stay valid, you (or anyone who ever had a copy of this file) technically still has working access to the client's database and Google sign-in  even after "handing over," until every one of these values is actually changed to something only the client knows. This is the step that makes the handover real, not just symbolic.

**`[local_accounts]` is no longer part of this file at all, by design.** The app now has a Local Accounts admin page (create/reset password/change role/delete, all from the UI)  so nobody, including the client, ever needs to put a real password into a config file to manage local sign-in. The actual handover sequence for accounts:

1. The client sets up their own admin's email in `GOOGLE_ADMIN_EMAILS` (`auth.py`, step 3 above) and signs in with Google. That's Administrator access with zero dependency on anything in `secrets.toml`.
2. From the Local Accounts page, they create whatever local (email+password) accounts they actually want, choosing each password themselves, on the spot, through the form  nothing gets written to a file.
3. **Delete the two demo accounts I created** (`admin@`/`analyst@strokefoundation.org.au`) from that same page, or reset their passwords to something only the client knows. Those were dev/testing credentials I generated  they should not remain valid after handover, the same as every other credential on this list.

If a brand-new database is ever created from scratch (step 2's fallback path) with genuinely zero accounts in it, a `[local_accounts]` section *can* still be added temporarily to bootstrap the very first one  but that's an edge case, not the normal path, and whoever adds it should be the client, choosing their own value, not something carried over from this handover.

## What you're left with afterward

Once all five are transferred and rotated, you (the outgoing developer) should have **no standing access** to the live app, its database, or its Google sign-in  the same way any contractor's access should end when a project hands over. Confirm this explicitly rather than assuming: try logging into the live app afterward with your old personal Gmail and confirm it's rejected, and confirm your old Supabase/GitHub/Google Cloud accounts no longer show the transferred resources.

## What stays the same

- The three documentation files already written (`data-handling.md`, `hosting-options.md`, `architecture.md`/`flow.md`/`dashboard.md`) don't need to change for handover  they describe the app itself, not who owns the infrastructure it runs on.
- No code changes are required by any of steps 1–4  this is entirely an accounts-and-configuration exercise, not a rebuild.
