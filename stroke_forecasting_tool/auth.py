import streamlit as st

from branding import logo_data_uri
from db import (
    LOCAL_SESSION_DAYS, create_local_session, db_configured, delete_local_session, get_local_account,
    get_local_session, verify_password,
)

# Local (email+password) login credentials used to live here as a plaintext
# USERS dict -- the Critical finding from the security review. Moved to a
# local_accounts table in the same Supabase database db.py already uses
# (hashed via db.py's hash_password()/verify_password()) -- see
# render_login()'s local-form handler below, which now calls
# get_local_account() + verify_password() instead of checking a dict. Same
# two accounts, same passwords, seeded automatically by db.py's
# _seed_local_accounts() on first connect.
#
# Google sign-in has been removed entirely -- every account is a local
# (email+password) one, created and managed from the Users page.

# Browser cookie name for a local account's persistent session (see
# create_local_session() below and db.py's create_local_session()). Not
# HttpOnly -- it's written via document.cookie from a plain <script>, since
# Streamlit gives app code no way to attach a Set-Cookie response header --
# so it's readable by any JS on this page. Holds only a random opaque
# token (meaningless without the matching, server-only-visible row in
# local_sessions), never a password or anything else sensitive, which is
# the same trade every "remember me" cookie set client-side makes.
LOCAL_SESSION_COOKIE = 'sf_local_session'

SESSION_KEYS = [
    'master', 'forecast_df', 'monthly', 'mape',
    'pipeline_run', 'page', 'authenticated', 'user', 'auth_method', 'local_session_token',
    'pending_2fa', 'totp_setup_secret',
    'totp_qr_cache', 'profile_details_editing', 'change_password_editing',
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


def restore_local_session():
    """Call once, early, on every run (app.py, right after
    complete_sign_out()). A local (email+password) login's
    authenticated/user state lives only in this session's server-memory
    session_state -- so a process restart (e.g. triggered by a long
    pipeline run's memory pressure) just logs that user out with no way
    back except signing in again. This closes that gap: if this run
    doesn't already have an authenticated session, check the browser for
    the persistent-session cookie render_login() sets on a successful
    local sign-in, and restore from it via db.py's get_local_session() if
    it's still valid. Only ever fills in a MISSING session -- never runs
    at all once one already exists, so it can't override a genuinely fresh
    login or a deliberate sign-out."""
    if st.session_state.authenticated:
        return
    token = st.context.cookies.get(LOCAL_SESSION_COOKIE)
    if not token:
        return
    account = get_local_session(token)
    if not account:
        return
    st.session_state.authenticated = True
    st.session_state.auth_method = 'password'
    st.session_state.user = account
    st.session_state.local_session_token = token


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
    st.html(f"""
    <div style="position:fixed;inset:0;z-index:999999999;background:#FFFFFF;
        display:flex;align-items:center;justify-content:center;">
        <div style="text-align:center;">
            <div class="sf-transition-loader" style="margin:0 auto 16px;"></div>
            <div style="font-family:'Space Grotesk',system-ui,sans-serif;font-size:12.5px;
                font-weight:600;color:#64748B;letter-spacing:0.06em;text-transform:uppercase;">
                {label}
            </div>
        </div>
    </div>
    <style>
    /* A solid teal disc masked down to a ring with one small gap (the
    conic-gradient's 10% transparent wedge, subtracted against the
    content-box so the centre stays hollow) -- replaced the old plain
    border-with-a-coloured-top-arc ring per feedback that this reads
    cleaner/more polished. #00897B is this file's own hardcoded light
    teal (this whole screen is deliberately theme-independent, same as
    the old spinner's border-top-color was), not the {{TEAL}} palette
    var ui.py's theme-aware pages use. */
    .sf-transition-loader {{
        width: 34px; padding: 5px; aspect-ratio: 1; border-radius: 50%;
        background: #00897B;
        --_m: conic-gradient(#0000 10%,#000), linear-gradient(#000 0 0) content-box;
        -webkit-mask: var(--_m); mask: var(--_m);
        -webkit-mask-composite: source-out; mask-composite: subtract;
        animation: sf-transition-spin 1s infinite linear;
    }}
    @keyframes sf-transition-spin {{ to {{ transform: rotate(1turn); }} }}
    </style>
    """)


def complete_sign_out():
    """Call once, at the very top of app.py, before anything else renders
    -- checks the flag sign_out() sets above and, only if it's set, shows
    a brief branded loading screen and then actually clears session state
    and completes the sign-out. st.rerun() raises internally and aborts
    the rest of THIS script run on its own (same as every other
    st.rerun()/st.stop() call in this file), so there's no trailing
    st.stop() needed here the way render_login() and its siblings below
    need one (they're blocking on user input, not triggering an
    immediate transition)."""
    if not st.session_state.get('_signing_out'):
        return
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

    # Revoke the persistent session server-side (not just clear the
    # cookie below) so sign-out actually ends it -- otherwise the token
    # in local_sessions would stay valid, and anyone who'd got hold of
    # the cookie could keep using it, until it expired on its own. Read
    # BEFORE the SESSION_KEYS wipe just below, which would otherwise
    # clear local_session_token first (it's one of those keys) and leave
    # nothing here to revoke.
    delete_local_session(st.session_state.get('local_session_token'))
    st.html(f"""
    <script>
    document.cookie = "{LOCAL_SESSION_COOKIE}=; path=/; max-age=0; SameSite=Lax";
    </script>
    """, unsafe_allow_javascript=True)

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
    st.markdown('<div class="sf-login-sub">CADENCE</div>', unsafe_allow_html=True)


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
    render_transition_spinner() before the rerun means the browser sees
    login page -> clean spinner -> dashboard instead.

    render_transition_spinner() is painted BEFORE page.empty(), not
    after -- reported live as an intermittent thin green line (and a
    flash of the "Signing in..." caption) visible in/around the spinner
    for a moment right at submit: page still holds the sticky startup
    bar (render_startup_progress()/_login_bar below) at that instant,
    and page.empty() clearing it plus this function's own opaque overlay
    appearing are two separate forward-messages the frontend doesn't
    always finish reconciling on the same visual frame -- the still-
    live bar can paint through the overlay while it's mid-fade-in.
    Painting the overlay FIRST, while the old page is still fully intact
    underneath, means it's already fully opaque before anything gets
    cleared -- page.empty() right after is then invisible, covered by
    the overlay already sitting on top. Same ordering principle
    complete_sign_out() already uses (paint the covering overlay against
    still-live content, not after clearing it).

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
                st.caption('Enter your credentials to access Cadence.')

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
                        st.error('Sign-in is unavailable right now (no database configured). '
                                  'Contact an administrator.',
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
                            if account.get('totp_secret'):
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
        # This is exactly the point to start a persistent session (see
        # restore_local_session()'s docstring for why local accounts need
        # one at all). A missing
        # token (create_local_session() returned None -- DB unreachable)
        # just means this login won't survive a restart; it's still a
        # perfectly valid signed-in session for as long as this server
        # process keeps running, so nothing here blocks on it.
        token = create_local_session(st.session_state.user['email'])
        if token:
            st.session_state.local_session_token = token
            max_age = LOCAL_SESSION_DAYS * 24 * 60 * 60
            st.html(f"""
            <script>
            document.cookie = "{LOCAL_SESSION_COOKIE}={token}; path=/; max-age={max_age}; SameSite=Lax"
                + (location.protocol === "https:" ? "; Secure" : "");
            </script>
            """, unsafe_allow_javascript=True)
        render_transition_spinner('Signing in')
        page.empty()
        st.rerun()
    st.stop()
