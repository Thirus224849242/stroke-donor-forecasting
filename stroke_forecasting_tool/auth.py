import streamlit as st

from branding import logo_data_uri
from db import db_configured, get_local_account, get_user, request_access, upsert_approved_user, verify_password

# Local (email+password) login credentials used to live here as a plaintext
# USERS dict -- the Critical finding from the security review. Moved to a
# new local_accounts table in the same Supabase database db.py already
# uses (hashed via db.py's hash_password()/verify_password()), alongside
# Google sign-in, not replacing it -- see render_login()'s local-form
# handler below, which now calls get_local_account() + verify_password()
# instead of checking a dict. Same two accounts, same passwords, seeded
# automatically by db.py's _seed_local_accounts() on first connect.

# Google sign-in: only these domains may complete login (this is an
# internal tool, not a public one) -- the org's real domain plus the dev
# team's own (Deakin), since there's no separate staging environment.
# Only these specific addresses get promoted to Administrator -- everyone
# else on an allowed domain lands as Analyst. There's no user database
# backing OAuth logins, so this allow-list is the only thing standing in
# for one.
ALLOWED_GOOGLE_DOMAINS = {'strokefoundation.org.au', 'deakin.edu.au'}
GOOGLE_ADMIN_EMAILS = {'admin@strokefoundation.org.au', 'thirumalreddyenugu@gmail.com'}
# Individual exceptions to the domain restriction -- for dev/demo access
# from an account that isn't on either allowed domain (e.g. a personal
# Gmail used for testing). Remove before this app is shared beyond the dev
# team.
GOOGLE_EXTRA_ALLOWED_EMAILS = {'thirumalreddyenugu@gmail.com'}

SESSION_KEYS = [
    'master', 'forecast_df', 'monthly', 'mape',
    'pipeline_run', 'page', 'authenticated', 'user', 'auth_method', 'pending_2fa', 'totp_setup_secret',
    'totp_qr_cache',
    'is_signing_in', 'is_signing_out',
    'ml_importances', 'ml_trend_slope',
    'forecast_df_linear', 'mape_linear', 'mape_stockflow',
    'ltv_results', 'ltv_tuning', 'ltv_metrics', 'ltv_error', 'ltv_monthly', 'ltv_histogram',
    'pipeline_running', 'pipeline_success_message', 'pipeline_error_message',
    'sf_walkforward', 'sf_components', 'sf_zone12', 'sf_zone3', 'sf_assumptions', 'sf_error',
    'db_error',
    'data_source', 'data_loaded_at', 'viewing_run_id',
    'master_rows', 'donor_count', 'total_income',
    'contact_count', 'supplier_count', 'campaign_type_count',
    'supplier_summary', 'campaign_summary', 'supplier_monthly', 'campaign_monthly',
    'retention_by_segment',
    'sbg_results', 'sbg_monthly', 'bgnbd_monthly', 'sbg_metrics', 'mape_sbg', 'mape_bgnbd', 'sbg_error',
    'gw_monthly', 'mape_gw', 'gw_metrics', 'gw_error',
]


def init_session_state():
    for k in SESSION_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None
    if st.session_state.authenticated is None:
        st.session_state.authenticated = False
    if st.session_state.page is None:
        st.session_state.page = 'Overview'
    if st.session_state.pipeline_running is None:
        st.session_state.pipeline_running = False


def google_auth_configured() -> bool:
    """True once secrets.toml's [auth] section has real Google credentials
    (not the placeholder scaffold) -- lets the login page skip the Google
    button entirely until it's actually set up, rather than erroring."""
    try:
        auth = st.secrets.get('auth')
        if not auth:
            return False
        client_id = auth.get('client_id', '')
        return bool(client_id) and not client_id.startswith('REPLACE_WITH_')
    except Exception:
        return False


def handle_google_redirect():
    """Call once, early, on every run -- before the render_login() gate.
    Streamlit's own OIDC cookie (st.user) is separate from our
    session_state; this is what bridges the two right after a user comes
    back from Google's redirect. Rejects anyone outside the allowed
    domain immediately, same as before.

    For everyone else, being on an allowed domain is necessary but no
    longer SUFFICIENT on its own for a non-admin: they also need an
    'approved' row in the `users` DB table (db.py) -- requested via
    render_request_access() below and granted by a Super Admin on the
    Access Requests page (app.py).

    Three roles now, not two -- 'Super Admin' > 'Administrator' >
    'Analyst'. Only Super Admin can create/delete accounts, approve/deny/
    revoke Google access, and promote/demote anyone's role (Local
    Accounts + Access Requests pages, app.py). Administrator keeps
    everything else it always had (Data Pipeline, CSV exports, deleting a
    run) but no account-management power at all -- back to how it worked
    before those two pages existed. Every role can change their own
    password from the account menu if they're signed in locally.

    GOOGLE_ADMIN_EMAILS is a ONE-TIME BOOTSTRAP SEED, not a standing
    override -- it's only ever consulted the very first time an email
    signs in (record is None below), to create their initial 'users' row
    as an auto-approved Super Admin (the highest tier, not just
    Administrator -- someone has to be able to set everyone else up),
    skipping the request queue (they're the ones who'd review it; gating
    them behind their own approval would be circular). From then on the
    database is the sole source of truth: a Super Admin can promote,
    demote, or remove that person's access entirely through the app's own
    UI, and it takes effect on their very next login -- editing this list
    again has no further effect on them.
    """
    if not google_auth_configured():
        return
    if st.session_state.authenticated or not st.user.is_logged_in:
        return

    email = (st.user.email or '').strip().lower()
    domain = email.rsplit('@', 1)[-1] if '@' in email else ''
    name = st.user.get('name') or (email.split('@')[0].replace('.', ' ').title() if email else 'User')

    allowed = domain in ALLOWED_GOOGLE_DOMAINS or email in GOOGLE_EXTRA_ALLOWED_EMAILS
    if not email or not allowed:
        allowed_list = ' or '.join(f'@{d}' for d in sorted(ALLOWED_GOOGLE_DOMAINS))
        st.html("""
        <style>[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }</style>
        """)
        st.error(
            f'{email or "This account"} is not a {allowed_list} account. '
            f'Access is restricted to Stroke Foundation staff and the dev team.',
            icon=':material/block:',
        )
        if st.button('Back to sign in'):
            st.logout()
        st.stop()

    # No DB configured means there's nowhere to persist a request queue OR
    # to check for an existing record at all -- GOOGLE_ADMIN_EMAILS is the
    # only signal available in that case, same "fails soft" pattern every
    # other DB-backed feature in this app follows (see db.py's own module
    # docstring).
    if not db_configured():
        role = 'Super Admin' if email in GOOGLE_ADMIN_EMAILS else 'Analyst'
        st.session_state.authenticated = True
        st.session_state.auth_method = 'google'
        st.session_state.user = {'email': email, 'name': name, 'role': role}
        return

    # THE DATABASE IS THE SOURCE OF TRUTH ONCE A RECORD EXISTS. GOOGLE_ADMIN_EMAILS
    # is checked ONLY in the record-is-None branch below -- a one-time bootstrap
    # seed for a brand-new email that's never signed in before, not a standing
    # override. Previously this list was checked FIRST and re-applied
    # role='Administrator' on every single login for these emails, permanently
    # overwriting whatever an admin had set for them in the `users` table via
    # the Local/Access management UIs -- a real bug (reported live), not just a
    # style issue: it meant an email on this list could never actually be
    # demoted, and meant this hardcoded list was a permanent, undocumented
    # backdoor with no way to revoke it short of editing this file and
    # redeploying. Checking the database FIRST fixes both: once a row exists
    # for an email (created either by this bootstrap path or by the ordinary
    # request-access flow), every later login reads its role/status from
    # there -- an admin can promote, demote, or remove that access entirely
    # through the UI, and it takes effect on that person's very next login,
    # with zero code change and zero redeploy, regardless of what this list
    # still says.
    record = get_user(email)

    if record is None:
        if email in GOOGLE_ADMIN_EMAILS:
            upsert_approved_user(email, name, role='Super Admin', decided_by=email)
            st.session_state.authenticated = True
            st.session_state.auth_method = 'google'
            st.session_state.user = {'email': email, 'name': name, 'role': 'Super Admin'}
            return
        render_request_access(email, name)
        return
    if record['status'] == 'pending':
        render_pending_access(email)
        return
    if record['status'] == 'denied':
        render_denied_access(email, name)
        return

    # approved -- role comes from the database, full stop, even for an
    # email that's also in GOOGLE_ADMIN_EMAILS.
    st.session_state.authenticated = True
    st.session_state.auth_method = 'google'
    st.session_state.user = {
        'email': email,
        'name': record.get('name') or name,
        'role': record.get('role') or 'Analyst',
    }


def initials(name: str) -> str:
    parts = [p for p in name.split() if p]
    if not parts:
        return '?'
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def sign_out():
    """Two-step sign-out, not one -- this function sets a flag and reruns;
    the actual state-clearing happens in complete_sign_out() below, called
    at the very top of app.py on the NEXT run, before any dashboard
    content renders. Clearing everything HERE (an earlier version) meant
    the still-rendering dashboard -- everything below wherever this was
    called from, e.g. the account popover in the page header -- got its
    data pulled out from under it mid-script, then Streamlit streamed
    that half-torn-down state to the browser for a moment before the new
    run (driven by this same st.rerun()) replaced it with the login page:
    reported live as data visibly "dropping out" right before logging out.
    Splitting it into two runs means the browser only ever sees a clean
    loading screen in between the full dashboard and the login page,
    never that broken intermediate frame.

    The overlay itself is now rendered HERE too, immediately, not only in
    complete_sign_out() on the next run -- reported live as a white
    "freeze"/fade flash appearing BEFORE the "Signing out" screen, not
    just going straight to it. Root cause: the instant st.rerun() below
    fires, Streamlit's own frontend starts fading every element of THIS
    (still-visible) dashboard toward transparent, since none of them will
    be reproduced by the upcoming run -- that fade is what read as
    "freezing"/turning white, and it was happening in the gap before
    complete_sign_out() got a chance to paint anything on the new run.
    Painting the exact same overlay here, in this still-live run, closes
    that gap entirely: the browser has something solid covering the
    dashboard from the very first frame of the transition, not just from
    whenever the next run's script happens to reach it."""
    st.html("""
    <style>
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
    [data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
    #stFloatingOverlayPortal { display: none !important; }
    </style>
    """)
    render_transition_spinner('Signing out')
    st.session_state['_signing_out'] = True
    st.rerun()


def render_transition_spinner(label: str):
    """Small branded loading indicator for the brief moment between two
    major auth-state transitions (signing in, signing out) -- rendered
    right before a screen-clearing st.rerun()/st.logout() so the browser
    has something clean to show instead of the previous screen's now-
    stale content lingering visibly until the next run's real content
    arrives. Not private to this module (no leading underscore) since
    app.py also holds this open, via its own placeholder, through the
    authenticated page's own first render after signing in -- see its
    is_signing_in handling near the top of the page-routing section.
    Three call sites in total: complete_sign_out() below (dashboard ->
    login), render_login()'s successful-submit branch (login ->
    dashboard transition, the brief moment before the rerun), and
    app.py (continuing that same dashboard transition through the new
    page's own render) -- factored out here specifically so all of them
    look and behave identically.

    position:fixed + inset:0 + an opaque background, not a plain
    height:80vh block flowing in normal document order -- confirmed live
    (via an artificial delay here, long enough to actually see this
    frame rather than it flashing past): the old height:80vh version
    only ADDED this spinner block after whatever else was already on the
    page, it never covered it. render_login()'s call site never showed
    that, because it first clears its own content via page.empty()
    before calling this -- but complete_sign_out() has no such
    placeholder to clear (the dashboard content it's transitioning away
    from was rendered by the PREVIOUS run, not this one, so there's
    nothing here to call .empty() on), so the old dashboard cards stayed
    fully visible underneath/around this spinner for the whole gap
    before the next rerun replaced them -- reported live as "broken CSS
    a moment before the login loads". A fixed, full-viewport, opaque
    overlay covers that regardless of what's still sitting in the
    document behind it.

    z-index 999999999, not 999999 -- ui.py's .sf-cached-run-banner (the
    "Showing the run from..." peekaboo strip) uses z-index: 999999 too,
    and reported live: revealed while this overlay is up (a stray mouse
    touch near the very top edge of the window is all its own JS needs
    to trigger that, unrelated to anything auth-related), it painted ON
    TOP of this overlay -- equal z-index falls back to DOM/paint order,
    and that banner's element is inserted later in the very same run
    that opened this overlay (app.py calls it after this placeholder is
    created). A comfortably higher z-index here wins regardless of
    ordering, without needing to touch the banner's own value."""
    st.markdown(f"""
    <div style="position:fixed;inset:0;z-index:999999999;background:#FFFFFF;
        display:flex;align-items:center;justify-content:center;">
        <div style="text-align:center;">
            <div style="width:34px;height:34px;border-radius:50%;margin:0 auto 16px;
                border:3px solid #E2E8F0;border-top-color:#00897B;
                animation:sf-transition-spin 0.8s linear infinite;"></div>
            <div style="font-family:'Space Grotesk',system-ui,sans-serif;font-size:12.5px;
                font-weight:600;color:#64748B;letter-spacing:0.06em;text-transform:uppercase;">
                {label}
            </div>
        </div>
    </div>
    <style>@keyframes sf-transition-spin {{ to {{ transform: rotate(360deg); }} }}</style>
    """, unsafe_allow_html=True)


def complete_sign_out():
    """Call once, at the very top of app.py, before anything else renders
    -- checks the flag sign_out() sets above and, only if it's set, shows
    a brief branded loading screen and then actually clears session state
    and completes the sign-out. Google needs st.logout() specifically to
    clear Streamlit's own identity cookie (a plain rerun wouldn't); local
    sign-in just needs the plain rerun -- either one raises internally
    and aborts the rest of THIS script run on its own (same as every
    other st.rerun()/st.stop() call in this file), so there's no
    trailing st.stop() needed here the way render_login() and its
    siblings below need one (they're blocking on user input, not
    triggering an immediate transition)."""
    if not st.session_state.get('_signing_out'):
        return
    method = st.session_state.auth_method
    st.html("""
    <style>
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
    [data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
    /* The account-menu popover (ui.py's page_header()) renders its open
    body into #stFloatingOverlayPortal, appended near document.body, not
    as a normal descendant of the page -- confirmed live, its stacking
    layer sits ABOVE render_transition_spinner()'s full-viewport overlay
    (a plain z-index on the overlay isn't enough to cover something in a
    different, higher portal layer -- and #stFloatingOverlayPortal is NOT
    the same element as [data-testid="portal"], a separate, unrelated
    overlay root). If "Log out" is clicked while that menu is still open
    (the normal way to reach it), its now-stale body stayed visible,
    floating on top of the sign-out transition. Hidden outright since
    nothing in this portal has any reason to be visible during a
    full-screen auth transition. */
    #stFloatingOverlayPortal { display: none !important; }
    </style>
    """)
    render_transition_spinner('Signing out')
    # Clearing session_state below is just dict assignment -- no real wait
    # to cover -- so without a deliberate pause here, this whole screen
    # flashed past in well under a frame and landed straight on the login
    # page: reported live as feeling raw/abrupt, and as "the signing-out
    # loader isn't working" even though it WAS rendering, just never for
    # long enough to actually see. A short fixed pause (not a readiness
    # check -- there's nothing here to wait for) is the deliberate fix,
    # matching the same reasoning as the 2FA QR placeholder's pause in
    # app.py's _render_2fa_card().
    import time
    time.sleep(0.6)

    for k in SESSION_KEYS:
        st.session_state[k] = None
    st.session_state.authenticated = False
    st.session_state.page = 'Overview'
    st.session_state.pipeline_running = False
    st.session_state['_signing_out'] = False
    # Set AFTER the wipe above (which would otherwise reset it right back
    # to None, since it's in SESSION_KEYS too) -- this is the user-facing
    # "still transitioning" flag, distinct from the internal one-shot
    # `_signing_out` trigger just cleared on the line above. It stays True
    # across the st.rerun()/st.logout() below and into the very next run,
    # where render_login() -- the only other place that reads it -- clears
    # it once the login form itself is actually the thing being rendered,
    # not merely once this function is done clearing state.
    st.session_state.is_signing_out = True
    if method == 'google':
        st.logout()  # clears Streamlit's identity cookie and reruns itself
    else:
        st.rerun()


def _render_auth_shell():
    """Shared CSS + logo + subtitle for every full-screen auth state
    (sign in, request access, pending, denied) -- factored out of what
    used to be render_login()'s own opening so the request-access
    screens below share its exact visual chrome without duplicating this
    whole block for each one. `.st-key-login_card` styling is reused as-
    is by every one of them too -- they're all just a differently-worded
    card inside the same shell."""
    st.html("""
    <style>
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    footer, [data-testid="stDecoration"], [data-testid="stAppDeployButton"],
    [data-testid="stMainMenu"] { display: none !important; }
    /* Same native top-right "Running..."/Stop indicator hidden in
    ui.py's inject_global_css() -- that function only runs post-login, so
    this screen (submit, wrong password, the 2FA code form, etc.) needs
    its own copy to stay unwanted-icon-free before authentication too. */
    [data-testid="stStatusWidget"] { display: none !important; }
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600&family=Inter:wght@400;500;600&display=swap');
    .stApp {
        background:
            radial-gradient(circle at 1px 1px, rgba(30,30,95,0.06) 1px, transparent 1px) 0 0/26px 26px,
            linear-gradient(135deg, #F0F1F7 0%, #E6F5F2 60%, #D9F0EC 100%);
    }
    .block-container {
        max-width: 440px !important; padding-top: 8vh !important;
        /* padding-left/right explicit here, not left to Streamlit's own
        default -- confirmed live, this page's card was rendering ~300px
        wide instead of the intended ~440px once layout=wide became the
        app's single fixed layout (see app.py's own comment on that
        change): wide layout's own default side padding for
        .block-container is bigger than centered layout's was, and this
        block never overrode padding-left/right itself, only padding-top,
        so it silently inherited whichever default the active layout
        happened to carry. Pinned to a small fixed value instead, so this
        page's actual width is independent of which layout mode the rest
        of the app is in. padding-bottom pinned too for the same reason,
        even though nothing below the card currently depends on it. */
        padding-left: 24px !important; padding-right: 24px !important; padding-bottom: 24px !important;
        margin-left: auto !important; margin-right: auto !important; float: none !important;
    }
    /* The real Stroke Foundation logo -- already carries the wordmark, so
    the old separate "Stroke Foundation" text title underneath is gone
    (would just repeat what the image already says). */
    .sf-login-logo { display: block; height: 64px; width: auto; margin: 0 auto 18px; }
    .sf-login-sub { text-align:center; font-size: 10px; font-weight:500; color:#64748B; margin-bottom: 28px;
        letter-spacing: 0.18em; text-transform: uppercase; font-family: 'Space Grotesk', system-ui, sans-serif; }
    .st-key-login_card { background: #FFFFFF; border: 1px solid #E2E8F0;
        border-radius: 0; padding: 28px 28px 26px;
        box-shadow: 0 1px 3px rgba(30,30,95,0.06); }
    .sf-login-footer { text-align:center; font-size: 11px; color:#64748B; margin-top: 22px;
        font-family: 'Inter', system-ui, sans-serif; }
    .st-key-login_card [data-testid="stForm"] { border: none; padding: 0; }
    .st-key-login_card button[kind="primary"] { width: 100%; margin-top: 6px; }
    .st-key-google_login_btn button {
        width: 100%; background: white !important; color: #3C4043 !important;
        border: 1px solid #DADCE0 !important; font-weight: 500 !important;
        display: flex !important; align-items: center !important; justify-content: center !important;
        gap: 10px !important;
    }
    .st-key-google_login_btn button:hover { background: #F8F9FA !important; border-color: #C6C9CE !important; }
    .st-key-google_login_btn button p::before {
        content: ''; display: inline-block; width: 18px; height: 18px; margin-right: 2px;
        vertical-align: middle; background-size: contain; background-repeat: no-repeat;
        background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCI+PHBhdGggZmlsbD0iI0ZGQzEwNyIgZD0iTTQzLjYxMSwyMC4wODNINDJWMjBIMjR2OGgxMS4zMDNjLTEuNjQ5LDQuNjU3LTYuMDgsOC0xMS4zMDMsOGMtNi42MjcsMC0xMi01LjM3My0xMi0xMmMwLTYuNjI3LDUuMzczLTEyLDEyLTEyYzMuMDU5LDAsNS44NDIsMS4xNTQsNy45NjEsMy4wMzlsNS42NTctNS42NTdDMzQuMDQ2LDYuMDUzLDI5LjI2OCw0LDI0LDRDMTIuOTU1LDQsNCwxMi45NTUsNCwyNGMwLDExLjA0NSw4Ljk1NSwyMCwyMCwyMGMxMS4wNDUsMCwyMC04Ljk1NSwyMC0yMEM0NCwyMi42NTksNDMuODYyLDIxLjM1LDQzLjYxMSwyMC4wODN6Ii8+PHBhdGggZmlsbD0iI0ZGM0QwMCIgZD0iTTYuMzA2LDE0LjY5MWw2LjU3MSw0LjgxOUMxNC42NTUsMTUuMTA4LDE4Ljk2MSwxMiwyNCwxMmMzLjA1OSwwLDUuODQyLDEuMTU0LDcuOTYxLDMuMDM5bDUuNjU3LTUuNjU3QzM0LjA0Niw2LjA1MywyOS4yNjgsNCwyNCw0QzE2LjMxOCw0LDkuNjU2LDguMzM3LDYuMzA2LDE0LjY5MXoiLz48cGF0aCBmaWxsPSIjNENBRjUwIiBkPSJNMjQsNDRjNS4xNjYsMCw5Ljg2LTEuOTc3LDEzLjQwOS01LjE5MmwtNi4xOS01LjIzOEMyOS4yMTEsMzUuMDkxLDI2LjcxNSwzNiwyNCwzNmMtNS4yMDIsMC05LjYxOS0zLjMxNy0xMS4yODMtNy45NDZsLTYuNTIyLDUuMDI1QzkuNTA1LDM5LjU1NiwxNi4yMjcsNDQsMjQsNDR6Ii8+PHBhdGggZmlsbD0iIzE5NzZEMiIgZD0iTTQzLjYxMSwyMC4wODNINDJWMjBIMjR2OGgxMS4zMDNjLTAuNzkyLDIuMjM3LTIuMjMxLDQuMTY2LTQuMDg3LDUuNTcxYzAuMDAxLTAuMDAxLDAuMDAyLTAuMDAxLDAuMDAzLTAuMDAybDYuMTksNS4yMzhDMzYuOTcxLDM5LjIwNSw0NCwzNCw0NCwyNEM0NCwyMi42NTksNDMuODYyLDIxLjM1LDQzLjYxMSwyMC4wODN6Ii8+PC9zdmc+");
    }
    .sf-login-divider { display: flex; align-items: center; gap: 12px; margin: 18px 0;
        font-size: 10px; font-weight: 500; color: #64748B; text-transform: uppercase; letter-spacing: 0.18em; }
    .sf-login-divider::before, .sf-login-divider::after { content: ''; flex: 1; height: 1px; background: #E2E8F0; }
    </style>
    """)

    _logo_uri = logo_data_uri()
    if _logo_uri:
        st.markdown(f'<img class="sf-login-logo" src="{_logo_uri}" alt="Stroke Foundation">',
                    unsafe_allow_html=True)
    else:
        # Falls back to the old placeholder mark if the asset is ever missing.
        st.markdown("""
        <div style="width:56px;height:56px;border-radius:12px;
            background:linear-gradient(135deg,#1E1E5F 0%,#00897B 100%);
            display:flex;align-items:center;justify-content:center;margin:0 auto 18px;
            box-shadow:0 8px 24px rgba(30,30,95,0.25);">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none"
                 stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
        </div>
        <div style="text-align:center;font-family:'Space Grotesk',sans-serif;font-weight:600;
            font-size:24px;color:#1E1E5F;margin-bottom:18px;">Stroke Foundation</div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="sf-login-sub">DONOR FORECASTING PLATFORM</div>', unsafe_allow_html=True)


def _render_auth_footer():
    st.markdown(
        '<div class="sf-login-footer">Stroke Foundation of Australia Â· '
        'Face-to-Face Regular Giving Program</div>',
        unsafe_allow_html=True,
    )


def render_login():
    """Full-screen sign-in page. Renders and stops the script.

    The whole page (shell + card + footer) lives inside ONE st.empty()
    placeholder specifically so the successful-login branch below can
    clear it in a single call -- reported live: without that, a
    successful submit set authenticated=True and called st.rerun()
    immediately, but this run had ALREADY streamed the full login page
    (logo, card, the just-submitted form, footer) to the browser before
    reaching that point, so THAT stayed the last thing on screen while
    the next run's dashboard (sidebar included) started streaming in
    underneath/around it -- a broken hybrid of both screens at once,
    not a clean transition. Clearing the placeholder and rendering
    render_transition_spinner() before the rerun (same technique
    complete_sign_out() uses for the reverse transition) means the
    browser sees login page -> clean spinner -> dashboard instead.

    just_signed_in is set (not an immediate st.rerun() call) from deep
    inside the nested form-handling logic below, then acted on ONLY
    after the `with page.container():` block has fully exited --
    calling page.empty() on a placeholder while still inside its own
    `with ...container():` body would be clearing a container the
    script is actively writing into, not a clean "replace what was
    there before" the way it works once that block has closed.

    st.session_state.pending_2fa holds {email, name, role, secret} for
    the brief window between a correct password and a confirmed TOTP
    code, for a local account that has two-factor auth turned on (see
    ui.py's account-menu setup flow, and db.py's totp_secret column) --
    None the rest of the time, including for every account that never
    enabled it. Its presence, not a separate step counter, is what picks
    which of the two forms below renders; set right before an immediate
    st.rerun() (same "set state, then rerun" pattern the dark-mode toggle
    elsewhere in this app already uses from deep inside nested `with`
    blocks) rather than falling through to render the code form in the
    same pass, so the two forms never both try to render at once."""
    page = st.empty()
    just_signed_in = False
    with page.container():
        _render_auth_shell()

        with st.container(key='login_card'):
            pending = st.session_state.get('pending_2fa')
            if pending:
                st.markdown("**Two-factor authentication**")
                st.caption(f"Enter the 6-digit code from your authenticator app for {pending['email']}.")
                with st.form('totp_form', border=False):
                    code = st.text_input(
                        'Authentication code', placeholder='123456', max_chars=6,
                        icon=':material/pin:',
                    )
                    totp_submitted = st.form_submit_button(
                        'Verify', type='primary', icon=':material/verified_user:',
                    )
                if st.button('Use a different account', key='totp_cancel'):
                    st.session_state.pending_2fa = None
                    st.rerun()
                if totp_submitted:
                    try:
                        # Local import -- pyotp is only ever needed here and in
                        # app.py's Profile-page setup/disable flow, not on the
                        # hot path of every other page load.
                        import pyotp
                        # valid_window=1 accepts the current 30s step plus one
                        # step either side -- standard TOTP tolerance for clock
                        # drift between the server and the user's phone, same
                        # default most authenticator-backed logins use.
                        code_ok = bool(code) and pyotp.TOTP(pending['secret']).verify(
                            code.strip(), valid_window=1,
                        )
                    except Exception:
                        code_ok = False
                        st.error('Two-factor verification is unavailable right now. Try again '
                                  'shortly, or contact an administrator.', icon=':material/error:')
                    else:
                        if code_ok:
                            st.session_state.authenticated = True
                            st.session_state.auth_method = 'password'
                            st.session_state.user = {
                                'email': pending['email'], 'name': pending['name'], 'role': pending['role'],
                            }
                            st.session_state.pending_2fa = None
                            st.session_state.is_signing_in = True
                            just_signed_in = True
                        else:
                            st.error('Incorrect code. Please try again.', icon=':material/error:')
            else:
                # The login form is the destination complete_sign_out() is
                # transitioning to -- reaching this branch at all means
                # it's about to actually render, so this is where the
                # "Signing out" state ends (see complete_sign_out()'s
                # is_signing_out comment in this same file).
                st.session_state.is_signing_out = False
                st.markdown("**Sign in to your account**")
                st.caption('Enter your credentials to access the F2F forecasting dashboard.')

                if google_auth_configured():
                    with st.container(key='google_login_btn'):
                        if st.button('Continue with Google', width='stretch'):
                            st.login()
                    st.markdown('<div class="sf-login-divider">or</div>', unsafe_allow_html=True)

                with st.form('login_form', border=False):
                    email = st.text_input(
                        'Work email', placeholder='name@strokefoundation.org.au',
                        icon=':material/mail:',
                    )
                    password = st.text_input(
                        'Password', type='password', placeholder='Enter your password',
                        icon=':material/lock:',
                    )
                    submitted = st.form_submit_button(
                        'Sign in', type='primary', icon=':material/login:',
                    )

                if submitted:
                    # Set immediately, before the DB call below -- this is
                    # what makes the "Signing in" state deterministic
                    # rather than tied to how fast that call happens to
                    # return. Cleared on every failure/hand-off path below;
                    # left True on an actual success, through to app.py's
                    # post-login section, which is the only other place
                    # that clears it (see its own comment there for why).
                    st.session_state.is_signing_in = True
                    clean_email = email.strip().lower()
                    if not db_configured():
                        # Distinct from a wrong password -- this means local login
                        # can't work at all right now (local_accounts lives in the
                        # same Supabase database as everything else in db.py), not
                        # that this particular attempt failed.
                        st.session_state.is_signing_in = False
                        st.error('Local sign-in is unavailable right now (no database configured). '
                                  'Try Google sign-in instead, or contact an administrator.',
                                  icon=':material/error:')
                    else:
                        # get_local_account() below is the FIRST database call of
                        # the whole session -- st.cache_resource's engine (db.py's
                        # get_engine()) hasn't connected yet, so THIS call, not
                        # the post-login restore step in app.py, is where the
                        # ~2.7s TCP/TLS/auth handshake to Supabase actually gets
                        # paid on a freshly-started server (confirmed live: without
                        # this, the login card just sat there greyed out with no
                        # feedback for the whole handshake, then jumped straight
                        # to the dashboard once app.py's own bar -- now hitting an
                        # already-warm connection -- flashed by too fast to see).
                        # Showing the same sticky bar immediately here, before the
                        # call, means there's no gap where nothing on screen
                        # explains the wait. Local import: ui.py imports from this
                        # module (initials/sign_out) at its own top level, so a
                        # top-level `from ui import ...` here would be a circular
                        # import; deferring it to call time (after both modules
                        # have already finished loading) avoids that.
                        from ui import render_startup_progress
                        _login_bar = st.empty()
                        render_startup_progress(_login_bar, label='Signing in')
                        account = get_local_account(clean_email)
                        if account and verify_password(password, account['password_hash']):
                            # Same domain restriction Google sign-in enforces
                            # (handle_google_redirect() below) -- applied here too
                            # per explicit request, as a defense-in-depth check.
                            # Local accounts are admin-provisioned (there's no
                            # self-service sign-up for this login path), so this
                            # should never actually trip in normal use; it's a
                            # safety net, not the primary gate.
                            domain = clean_email.rsplit('@', 1)[-1] if '@' in clean_email else ''
                            if domain not in ALLOWED_GOOGLE_DOMAINS and clean_email not in GOOGLE_EXTRA_ALLOWED_EMAILS:
                                _login_bar.empty()
                                st.session_state.is_signing_in = False
                                st.error('This account is not on an authorised domain. Contact an administrator.',
                                          icon=':material/block:')
                            elif account.get('totp_secret'):
                                # Password's right, but this account has 2FA on --
                                # not authenticated yet. Cleared rather than left
                                # showing: the wait from here is on the USER typing
                                # a code, not on a network/DB call, so "connecting"
                                # language would be actively misleading. Same reasoning
                                # for is_signing_in -- entering a code is a distinct
                                # interactive step, not a continuation of "signing in";
                                # it's set True again above once that code verifies.
                                _login_bar.empty()
                                st.session_state.is_signing_in = False
                                st.session_state.pending_2fa = {
                                    'email': clean_email, 'name': account['name'],
                                    'role': account['role'], 'secret': account['totp_secret'],
                                }
                                st.rerun()
                            else:
                                st.session_state.authenticated = True
                                st.session_state.auth_method = 'password'
                                st.session_state.user = {
                                    'email': clean_email,
                                    'name': account['name'],
                                    'role': account['role'],
                                }
                                just_signed_in = True
                                # _login_bar deliberately left showing -- just_signed_in
                                # below clears the whole page (this bar included) and
                                # replaces it with the transition spinner before the
                                # rerun, so there's no gap here either. is_signing_in
                                # stays True (set at submit time above) straight through
                                # that handoff into app.py.
                        else:
                            _login_bar.empty()
                            st.session_state.is_signing_in = False
                            # Deliberately the SAME message whether the email has no
                            # local account at all or the password was just wrong --
                            # distinguishing them would let this form be used to
                            # enumerate which emails have accounts.
                            st.error('Incorrect email or password. Please try again.', icon=':material/error:')

        _render_auth_footer()

    if just_signed_in:
        page.empty()
        render_transition_spinner('Signing in')
        st.rerun()
    st.stop()


def render_request_access(email: str, name: str):
    """Shown when a Google sign-in is from an allowed domain but has no
    `users` DB row at all yet -- offers to submit an access request,
    which shows up on the Administrator-only Access Requests page
    (app.py) for someone to approve or deny. Renders and stops the
    script, same as render_login()."""
    _render_auth_shell()

    with st.container(key='login_card'):
        st.markdown("**Request access**")
        st.caption(f'{email} isn\'t set up yet. Submit a request below and an '
                   f'administrator will review it -- you\'ll be able to sign in as '
                   f'soon as it\'s approved.')
        if st.button('Request access', type='primary', icon=':material/send:', width='stretch'):
            request_access(email, name)
            st.rerun()
        if st.button('Back to sign in', width='stretch'):
            st.logout()

    _render_auth_footer()
    st.stop()


def render_pending_access(email: str):
    """Shown on every sign-in attempt while a request is still awaiting
    an administrator's decision -- no action to take here but wait, or
    back out. Renders and stops the script, same as render_login()."""
    _render_auth_shell()

    with st.container(key='login_card'):
        st.markdown("**Request pending**")
        st.caption(f'Your access request for {email} is waiting on an administrator '
                   f'to review it. You\'ll be able to sign in as soon as it\'s approved '
                   f'-- check back soon.')
        if st.button('Back to sign in', width='stretch'):
            st.logout()

    _render_auth_footer()
    st.stop()


def render_denied_access(email: str, name: str):
    """Shown when an administrator has declined the request -- offers to
    submit a fresh one (request_access() resets a 'denied' row back to
    'pending', see its own docstring in db.py). Renders and stops the
    script, same as render_login()."""
    _render_auth_shell()

    with st.container(key='login_card'):
        st.markdown("**Access denied**")
        st.caption(f'Your access request for {email} was declined by an administrator. '
                   f'If you believe this was a mistake, you can submit a new request.')
        if st.button('Request access again', icon=':material/send:', width='stretch'):
            request_access(email, name)
            st.rerun()
        if st.button('Back to sign in', width='stretch'):
            st.logout()

    _render_auth_footer()
    st.stop()
