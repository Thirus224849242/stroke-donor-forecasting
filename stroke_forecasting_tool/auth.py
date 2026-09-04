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
    'pipeline_run', 'page', 'authenticated', 'user', 'auth_method',
    'ml_importances', 'ml_trend_slope',
    'forecast_df_linear', 'mape_linear', 'mape_stockflow',
    'ltv_results', 'ltv_tuning', 'ltv_metrics', 'ltv_error', 'ltv_monthly', 'ltv_histogram',
    'pipeline_running', 'pipeline_success_message',
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
    render_request_access() below and granted by an Administrator on the
    Access Requests page (app.py).

    GOOGLE_ADMIN_EMAILS is a ONE-TIME BOOTSTRAP SEED, not a standing
    override -- it's only ever consulted the very first time an email
    signs in (record is None below), to create their initial 'users' row
    as an auto-approved Administrator, skipping the request queue (they're
    the ones who'd review it; gating them behind their own approval would
    be circular). From then on the database is the sole source of truth:
    an admin can promote, demote, or remove that person's access entirely
    through the app's own UI, and it takes effect on their very next
    login -- editing this list again has no further effect on them.
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
        role = 'Administrator' if email in GOOGLE_ADMIN_EMAILS else 'Analyst'
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
            upsert_approved_user(email, name, role='Administrator', decided_by=email)
            st.session_state.authenticated = True
            st.session_state.auth_method = 'google'
            st.session_state.user = {'email': email, 'name': name, 'role': 'Administrator'}
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
    method = st.session_state.auth_method
    for k in SESSION_KEYS:
        st.session_state[k] = None
    st.session_state.authenticated = False
    st.session_state.page = 'Overview'
    st.session_state.pipeline_running = False
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
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600&family=Inter:wght@400;500;600&display=swap');
    .stApp {
        background:
            radial-gradient(circle at 1px 1px, rgba(30,30,95,0.06) 1px, transparent 1px) 0 0/26px 26px,
            linear-gradient(135deg, #F0F1F7 0%, #E6F5F2 60%, #D9F0EC 100%);
    }
    .block-container {
        max-width: 440px !important; padding-top: 8vh !important;
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
        '<div class="sf-login-footer">Stroke Foundation of Australia · '
        'Face-to-Face Regular Giving Program</div>',
        unsafe_allow_html=True,
    )


def render_login():
    """Full-screen sign-in page. Renders and stops the script."""
    _render_auth_shell()

    with st.container(key='login_card'):
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
            clean_email = email.strip().lower()
            if not db_configured():
                # Distinct from a wrong password -- this means local login
                # can't work at all right now (local_accounts lives in the
                # same Supabase database as everything else in db.py), not
                # that this particular attempt failed.
                st.error('Local sign-in is unavailable right now (no database configured). '
                          'Try Google sign-in instead, or contact an administrator.',
                          icon=':material/error:')
            else:
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
                        st.error('This account is not on an authorised domain. Contact an administrator.',
                                  icon=':material/block:')
                    else:
                        st.session_state.authenticated = True
                        st.session_state.auth_method = 'password'
                        st.session_state.user = {
                            'email': clean_email,
                            'name': account['name'],
                            'role': account['role'],
                        }
                        st.rerun()
                else:
                    # Deliberately the SAME message whether the email has no
                    # local account at all or the password was just wrong --
                    # distinguishing them would let this form be used to
                    # enumerate which emails have accounts.
                    st.error('Incorrect email or password. Please try again.', icon=':material/error:')

    _render_auth_footer()
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
