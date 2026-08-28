import streamlit as st

USERS = {
    'admin@strokefoundation.org.au': {
        'password': 'strokef2f2026',
        'name': 'Admin User',
        'role': 'Administrator',
    },
    'analyst@strokefoundation.org.au': {
        'password': 'strokef2f2026',
        'name': 'Data Analyst',
        'role': 'Analyst',
    },
}

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
    'forecast_df_linear', 'mape_linear',
    'ltv_results', 'ltv_tuning', 'ltv_metrics', 'ltv_error', 'ltv_monthly',
    'pipeline_running', 'pipeline_success_message',
    'sf_walkforward', 'sf_components', 'sf_zone12', 'sf_zone3', 'sf_assumptions', 'sf_error',
    'db_error',
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
    domain immediately rather than letting them into the app."""
    if not google_auth_configured():
        return
    if st.session_state.authenticated or not st.user.is_logged_in:
        return

    email = (st.user.email or '').strip().lower()
    domain = email.rsplit('@', 1)[-1] if '@' in email else ''

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

    st.session_state.authenticated = True
    st.session_state.auth_method = 'google'
    st.session_state.user = {
        'email': email,
        'name': st.user.get('name') or email.split('@')[0].replace('.', ' ').title(),
        'role': 'Administrator' if email in GOOGLE_ADMIN_EMAILS else 'Analyst',
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


def render_login():
    """Full-screen sign-in page. Renders and stops the script."""
    st.html("""
    <style>
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    footer, [data-testid="stDecoration"], [data-testid="stAppDeployButton"],
    [data-testid="stMainMenu"] { display: none !important; }
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    .stApp {
        background:
            radial-gradient(circle at 1px 1px, rgba(30,30,95,0.07) 1.5px, transparent 0) 0 0/28px 28px,
            linear-gradient(135deg, #F7F8FC 0%, #EEF0FA 55%, #E4ECEB 100%);
    }
    .block-container {
        max-width: 440px !important; padding-top: 8vh !important;
        margin-left: auto !important; margin-right: auto !important; float: none !important;
    }
    .sf-login-logo {
        width: 56px; height: 56px; border-radius: 14px;
        background: linear-gradient(135deg, #1E1E5F 0%, #00897B 100%);
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 18px; box-shadow: 0 8px 24px rgba(30,30,95,0.25);
    }
    .sf-login-title { text-align:center; font-family:'Space Grotesk', system-ui, sans-serif; font-weight: 600;
        font-size: 22px; color:#1E1E5F; margin-bottom:2px; letter-spacing: -0.01em; }
    .sf-login-sub { text-align:center; font-size: 12.5px; color:#8888AA; margin-bottom: 28px;
        letter-spacing: 0.03em; font-family: 'IBM Plex Sans', system-ui, sans-serif; }
    .st-key-login_card { background: #FFFFFF; border: 1px solid #E8E8F0;
        border-radius: 14px; padding: 32px 30px 26px;
        box-shadow: 0 20px 50px -12px rgba(30,30,95,0.18); }
    .sf-login-footer { text-align:center; font-size: 11px; color:#8888AA; margin-top: 22px;
        font-family: 'IBM Plex Sans', system-ui, sans-serif; }
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
        font-size: 11px; color: #8888AA; text-transform: uppercase; letter-spacing: 0.06em; }
    .sf-login-divider::before, .sf-login-divider::after { content: ''; flex: 1; height: 1px; background: #E8E8F0; }
    </style>
    """)

    st.markdown("""
    <div class="sf-login-logo">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none"
             stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
        </svg>
    </div>
    <div class="sf-login-title">Stroke Foundation</div>
    <div class="sf-login-sub">DONOR FORECASTING PLATFORM</div>
    """, unsafe_allow_html=True)

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
            user = USERS.get(email.strip().lower())
            if user and user['password'] == password:
                st.session_state.authenticated = True
                st.session_state.auth_method = 'password'
                st.session_state.user = {
                    'email': email.strip().lower(),
                    'name': user['name'],
                    'role': user['role'],
                }
                st.rerun()
            else:
                st.error('Incorrect email or password. Please try again.', icon=':material/error:')

    st.markdown(
        '<div class="sf-login-footer">Stroke Foundation of Australia · '
        'Face-to-Face Regular Giving Program</div>',
        unsafe_allow_html=True,
    )
    st.stop()
