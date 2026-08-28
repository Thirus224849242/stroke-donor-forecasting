import re
from contextlib import contextmanager

import pandas as pd
import streamlit as st

from auth import initials, sign_out

# ── BRAND PALETTE ────────────────────────────────────────────────────────────
# Light, conventional — white/light-gray surfaces, same brand hues as before.
# NAVY does double duty again (literal sidebar bg, and dark text/accent on
# light cards) since that reads fine once the main content area is light too.
BG      = '#F7F8FC'
SURFACE = '#FFFFFF'
SURFACE2= '#F0F1F7'
LINE    = '#E8E8F0'
TEXT    = '#1E1E5F'
MIST    = '#8888AA'
SLATE   = '#4A4A6A'
NAVY    = '#1E1E5F'
TEAL    = '#00897B'
TEAL2   = '#00B5A3'
TEALLT  = '#E0F5F3'
PURPLE  = '#6B2D8B'
PURPLT  = '#F5EEF8'
AMBER   = '#C17D0C'
AMBERLT = '#FEF9E7'
BLUE    = '#1E1E5F'
BLUELT  = '#EAEAF8'
RED     = '#C0392B'
REDLT   = '#FBEAE8'
GREEN   = '#1B7F5A'
COLOURS = [TEAL, PURPLE, AMBER, RED, TEAL2, BLUE]

NAV_ITEMS = [
    ('Overview',           ':material/dashboard:'),
    ('Income Forecast',    ':material/trending_up:'),
    ('Retention Analysis', ':material/groups:'),
    ('Donor Lifetime Value', ':material/savings:'),
    ('Supplier Insights',  ':material/handshake:'),
    ('Campaign ROI',       ':material/campaign:'),
    ('Run History',        ':material/history:'),
    ('Data Pipeline',      ':material/database:'),
]


def _slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def fmt_size(num_bytes: float) -> str:
    if num_bytes >= 1024 * 1024:
        return f'{num_bytes / (1024 * 1024):.1f} MB'
    return f'{num_bytes / 1024:.0f} KB'


# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
def inject_global_css():
    st.html(f"""
    <style>
    footer, [data-testid="stDecoration"], [data-testid="stAppDeployButton"],
    [data-testid="stMainMenu"] {{ display: none !important; }}
    [data-testid="stHeader"] {{ background: transparent; }}

    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&display=swap');

    .stApp {{
        background:
            radial-gradient(circle at 1px 1px, rgba(30,30,95,0.045) 1.5px, transparent 0) 0 0/26px 26px,
            {BG};
    }}
    .block-container {{ padding-top: 1.1rem !important; max-width: 100% !important; }}

    /* ── Consistent card elevation for every bordered container ──
    st.container(border=True) only gets a border from the theme in this
    Streamlit version, not a fill — without this it's invisible when it
    sits directly on the page background. Bordered blocks are the only ones
    carrying an overflow="visible" attribute (needed for the border-radius
    clip), which is a more reliable selector here than emotion-cache class
    hashes. */
    [data-testid="stVerticalBlock"][overflow="visible"] {{
        background: {SURFACE} !important;
        box-shadow: 0 1px 2px rgba(20,20,55,0.04), 0 8px 20px -12px rgba(20,20,55,0.10);
        transition: box-shadow 0.15s ease;
    }}
    [data-testid="stMetric"] {{
        background: {SURFACE} !important;
        box-shadow: 0 1px 2px rgba(20,20,55,0.04), 0 8px 20px -12px rgba(20,20,55,0.10);
        border-radius: 12px !important;
        padding: 14px 16px 12px !important;
    }}
    [data-testid="stMetricValue"] {{ white-space: nowrap; font-family: 'IBM Plex Mono', ui-monospace, monospace !important; }}
    /* Streamlit wraps each card in its own single-child flex-direction:column
    wrapper. The old "flex: 1 1 210px" here set flex-basis: 210px meaning to
    size the card's WIDTH -- but flex-basis sizes along the MAIN axis, which
    for this wrapper is vertical, not horizontal, so it was flooring every
    card's HEIGHT at 210px regardless of content instead. min-width alone
    already achieves the intended width floor and isn't axis-dependent, so
    that's all this needs. */
    [class*="st-key-kpi_"] {{ min-width: 210px; align-self: flex-start !important; }}
    [data-testid="stMetricLabel"] p {{
        font-size: 10.5px !important; font-weight: 700 !important;
        letter-spacing: 0.08em; text-transform: uppercase; color: {MIST} !important;
    }}
    [data-testid="stMetricValue"] {{ color: {TEXT} !important; font-weight: 600 !important; }}

    /* ── Sidebar ── */
    [data-testid="stSidebarUserContent"] {{ display: flex; flex-direction: column; min-height: 100vh; padding-top: 0 !important; }}
    .sf-side-spacer {{ flex: 1 1 auto; }}
    .sf-side-logo {{ padding: 20px 18px 16px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 4px; }}
    .sf-side-mark {{ width: 30px; height: 30px; border-radius: 7px; background: {TEAL};
        display: flex; align-items: center; justify-content: center; margin-bottom: 10px; }}
    .sf-side-name {{ font-size: 13.5px; font-weight: 700; color: white; }}
    .sf-side-sub  {{ font-size: 11px; color: rgba(255,255,255,0.45); margin-top: 1px; }}
    .sf-side-section {{ font-size: 10px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
        color: rgba(255,255,255,0.32); padding: 6px 18px 6px; }}

    [data-testid="stSidebar"] .stButton > button {{
        width: 100% !important; background: transparent !important; color: rgba(255,255,255,0.68) !important;
        border: none !important; border-left: 3px solid transparent !important; border-radius: 0 !important;
        padding: 9px 16px !important; text-align: left !important; font-size: 13px !important;
        font-weight: 500 !important; justify-content: flex-start !important; box-shadow: none !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(255,255,255,0.07) !important; color: white !important;
        border-left-color: rgba(255,255,255,0.22) !important;
    }}
    [data-testid="stSidebar"] .stButton > button p {{ font-size: 13px !important; font-weight: inherit !important; }}

    .sf-side-user {{ display: flex; align-items: center; gap: 10px; padding: 12px 18px 10px; }}
    .sf-avatar {{ width: 32px; height: 32px; border-radius: 50%; background: {TEAL};
        color: white; font-size: 12px; font-weight: 700; display: flex; align-items: center; justify-content: center;
        flex-shrink: 0; }}
    .sf-user-name {{ font-size: 12.5px; font-weight: 600; color: white; line-height: 1.3; }}
    .sf-user-role {{ font-size: 10.5px; color: rgba(255,255,255,0.45); }}
    .sf-side-signout {{ padding: 0 18px 18px; }}
    .sf-side-signout .stButton > button {{
        border: 1px solid rgba(255,255,255,0.14) !important; border-radius: 8px !important;
        color: rgba(255,255,255,0.75) !important; font-size: 12px !important; padding: 7px 12px !important;
    }}
    .sf-side-signout .stButton > button:hover {{ background: rgba(192,57,43,0.18) !important; color: #F5B7B1 !important; border-color: transparent !important; }}

    /* ── Topbar ── */
    .sf-topbar {{
        position: sticky; top: -1.1rem; z-index: 999; margin: -1.1rem -1rem 18px;
        padding: 14px 24px; background: rgba(247,248,252,0.92); backdrop-filter: blur(6px);
        border-bottom: 1px solid {LINE}; display: flex; align-items: center; justify-content: space-between;
    }}
    .sf-breadcrumb {{ font-size: 12px; color: {MIST}; font-weight: 500; }}
    .sf-breadcrumb b {{ color: {TEXT}; font-weight: 700; }}
    .sf-breadcrumb span {{ margin: 0 6px; color: {LINE}; }}
    .sf-topbar-right {{ display: flex; align-items: center; gap: 10px; }}
    .sf-avatar-sm {{ width: 30px; height: 30px; border-radius: 50%; background: {NAVY}; color: white;
        font-size: 11px; font-weight: 700; display: flex; align-items: center; justify-content: center; }}
    .sf-topbar-user .name {{ font-size: 12px; font-weight: 600; color: {TEXT}; line-height: 1.25; }}
    .sf-topbar-user .role {{ font-size: 10.5px; color: {MIST}; }}

    /* ── Page header ── */
    .sf-eyebrow {{ font-size: 10px; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: {TEAL}; margin-bottom: 4px; }}
    .sf-page-title {{ font-family: 'Space Grotesk', system-ui, sans-serif; font-weight: 600; font-size: 24px; color: {TEXT}; margin-bottom: 3px; line-height: 1.2; letter-spacing: -0.01em; }}
    .sf-page-sub {{ font-size: 12.5px; color: {MIST}; margin-bottom: 6px; max-width: 720px; }}

    /* ── Card header (used inside bordered containers) ── */
    .sf-card-title {{ font-size: 13px; font-weight: 700; color: {TEXT}; }}

    /* ── Buttons ── */
    .stButton > button, .stDownloadButton > button {{ font-weight: 600 !important; }}
    .stDownloadButton > button {{ background: {SURFACE} !important; color: {TEXT} !important; border: 1px solid {LINE} !important; }}

    /* ── Pipeline flow diagram ── */
    .sf-pipeline {{ display: flex; align-items: center; background: {SURFACE}; border: 1px solid {LINE};
        border-radius: 12px; padding: 16px 22px; margin-bottom: 18px;
        box-shadow: 0 1px 2px rgba(20,20,55,0.04), 0 8px 20px -12px rgba(20,20,55,0.10); }}
    .sf-pipe-step {{ display: flex; align-items: center; gap: 9px; flex: 1; }}
    .sf-pipe-icon {{ width: 32px; height: 32px; border-radius: 8px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; }}
    .sf-pipe-icon svg {{ width: 15px; height: 15px; stroke-width: 2; fill: none; }}
    .sf-pipe-icon.p-teal, .sf-upload-icon.p-teal     {{ background: {TEALLT}; }}
    .sf-pipe-icon.p-teal svg, .sf-upload-icon.p-teal svg     {{ stroke: {TEAL}; }}
    .sf-pipe-icon.p-navy, .sf-upload-icon.p-navy     {{ background: {BLUELT}; }}
    .sf-pipe-icon.p-navy svg, .sf-upload-icon.p-navy svg     {{ stroke: {BLUE}; }}
    .sf-pipe-icon.p-purple, .sf-upload-icon.p-purple {{ background: {PURPLT}; }}
    .sf-pipe-icon.p-purple svg, .sf-upload-icon.p-purple svg {{ stroke: {PURPLE}; }}
    .sf-pipe-icon.p-amber, .sf-upload-icon.p-amber   {{ background: {AMBERLT}; }}
    .sf-pipe-icon.p-amber svg, .sf-upload-icon.p-amber svg   {{ stroke: {AMBER}; }}
    .sf-pipe-label {{ font-size: 11.5px; font-weight: 700; color: {TEXT}; }}
    .sf-pipe-desc  {{ font-size: 10px; color: {MIST}; margin-top: 1px; }}
    .sf-pipe-arrow {{ color: {LINE}; font-size: 20px; padding: 0 8px; flex-shrink: 0; }}

    /* ── Upload slot ── */
    .sf-upload-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
    .sf-upload-icon {{ width: 34px; height: 34px; border-radius: 9px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
    .sf-upload-icon svg {{ width: 16px; height: 16px; stroke-width: 2; fill: none; }}
    .sf-upload-title {{ font-size: 12.5px; font-weight: 700; color: {TEXT}; }}
    .sf-upload-desc {{ font-size: 10.5px; color: {MIST}; }}
    .sf-upload-done {{ display: flex; align-items: center; gap: 8px; background: {TEALLT}; border-radius: 8px;
        padding: 8px 12px; margin-top: 10px; font-size: 11.5px; font-weight: 600; color: {GREEN}; }}
    .sf-upload-done svg {{ width: 14px; height: 14px; stroke: {GREEN}; stroke-width: 3; flex-shrink: 0; }}
    .sf-upload-error {{ display: flex; align-items: flex-start; gap: 8px; background: {REDLT}; border-radius: 8px;
        padding: 8px 12px; margin-top: 10px; font-size: 11.5px; font-weight: 600; color: {RED}; }}
    .sf-upload-error svg {{ width: 14px; height: 14px; stroke: {RED}; stroke-width: 2.5; flex-shrink: 0; margin-top: 1px; }}

    [data-testid="stFileUploaderDropzone"] {{ border-radius: 10px !important; }}

    /* ── Sidebar disabled while pipeline runs ── */
    [data-testid="stSidebar"] .stButton > button:disabled {{
        opacity: 0.35 !important; cursor: not-allowed !important;
    }}
    .sf-side-running-note {{ font-size: 10.5px; color: rgba(255,255,255,0.4); padding: 2px 18px 8px;
        display: flex; align-items: center; gap: 6px; }}

    /* ── Empty state ── */
    .sf-empty-icon {{ width: 52px; height: 52px; border-radius: 14px; background: {TEALLT};
        display: flex; align-items: center; justify-content: center; margin: 6px auto 16px; }}

    /* ── Dataframe polish ── */
    [data-testid="stDataFrame"] {{ border-radius: 10px !important; overflow: hidden; }}
    </style>
    """)


# ── LAYOUT COMPONENTS ─────────────────────────────────────────────────────────
def render_sidebar():
    user = st.session_state.user or {'name': 'User', 'role': 'Analyst'}
    running = bool(st.session_state.get('pipeline_running'))
    with st.sidebar:
        st.markdown(f"""
        <div class="sf-side-logo">
            <div class="sf-side-mark">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="white"
                     stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
            </div>
            <div class="sf-side-name">Stroke Foundation</div>
            <div class="sf-side-sub">Donor Forecasting Tool</div>
        </div>
        <div class="sf-side-section">Main</div>
        """, unsafe_allow_html=True)

        for label, icon in NAV_ITEMS:
            active = st.session_state.page == label
            key = f'nav_{_slug(label)}'
            if active:
                st.html(f"""<style>
                .st-key-{key} button {{
                    background: rgba(0,137,123,0.20) !important; color: {TEAL2} !important;
                    border-left-color: {TEAL} !important; font-weight: 700 !important;
                }}
                .st-key-{key} button p {{ color: {TEAL2} !important; font-weight: 700 !important; }}
                </style>""")
            if st.button(label, key=key, icon=icon, width='stretch', disabled=running):
                st.session_state.page = label
                st.rerun()

        if running:
            st.markdown("""
            <div class="sf-side-running-note">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                     stroke-width="2.5" stroke-linecap="round"><circle cx="12" cy="12" r="9"/>
                <path d="M12 7v5l3 3"/></svg>
                Navigation locked — pipeline running
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.pipeline_run:
            m = st.session_state.master
            st.markdown(f"""
            <div style="height:1px;background:rgba(255,255,255,0.08);margin:10px 18px;"></div>
            <div style="padding:0 18px;font-size:11px;line-height:1.9;color:rgba(255,255,255,0.5);">
                <span style="color:rgba(255,255,255,0.85);font-weight:600;">Data loaded</span><br>
                {len(m):,} rows &middot; {m['recurring_payment_id'].nunique():,} signups<br>
                Forecast MAPE {st.session_state.mape:.1f}%
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sf-side-spacer"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="height:1px;background:rgba(255,255,255,0.08);margin:4px 18px 10px;"></div>
        <div class="sf-side-user">
            <div class="sf-avatar">{initials(user['name'])}</div>
            <div>
                <div class="sf-user-name">{user['name']}</div>
                <div class="sf-user-role">{user['role']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="sf-side-signout">', unsafe_allow_html=True)
        if st.button('Sign out', key='signout_btn', icon=':material/logout:', width='stretch', disabled=running):
            sign_out()
        st.markdown('</div>', unsafe_allow_html=True)


def render_topbar():
    user = st.session_state.user or {'name': 'User', 'role': 'Analyst'}
    page = st.session_state.page
    st.markdown(f"""
    <div class="sf-topbar">
        <div class="sf-breadcrumb">Dashboard <span>/</span> <b>{page}</b></div>
        <div class="sf-topbar-right">
            <div class="sf-topbar-user" style="text-align:right;">
                <div class="name">{user['name']}</div>
                <div class="role">{user['role']}</div>
            </div>
            <div class="sf-avatar-sm">{initials(user['name'])}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def page_header(eyebrow, title, sub=''):
    st.markdown(f"""
    <div style="margin-bottom:18px;">
        <div class="sf-eyebrow">{eyebrow}</div>
        <div class="sf-page-title">{title}</div>
        {'<div class="sf-page-sub">' + sub + '</div>' if sub else ''}
    </div>
    """, unsafe_allow_html=True)


def kpi(label, value, delta=None, icon=None, accent=TEAL, delta_color='normal', help=None):
    key = f'kpi_{_slug(label)}'
    st.html(f'<style>.st-key-{key} [data-testid="stMetric"] {{ border-top: 3px solid {accent} !important; }}</style>')
    with st.container(key=key):
        st.metric(label, value, delta=delta, icon=icon, border=True,
                  delta_color=delta_color, help=help)


@contextmanager
def card(title=None, sub=None, tag=None, tag_color='green', key=None):
    with st.container(key=key, border=True):
        if title:
            c1, c2 = st.columns([5, 1.4], vertical_alignment='top')
            with c1:
                st.markdown(f'<div class="sf-card-title">{title}</div>', unsafe_allow_html=True)
                if sub:
                    st.caption(sub)
            with c2:
                if tag:
                    st.badge(tag, color=tag_color)
        yield


def empty_state():
    with st.container(border=True):
        st.markdown(f"""
        <div style="text-align:center;padding:36px 20px 8px;">
            <div class="sf-empty-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{TEAL}" stroke-width="2" stroke-linecap="round">
                    <ellipse cx="12" cy="5" rx="9" ry="3"/>
                    <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
                    <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
                </svg>
            </div>
            <div style="font-size:15px;font-weight:700;color:{TEXT};margin-bottom:6px;">No data loaded yet</div>
            <div style="font-size:12.5px;color:{MIST};max-width:360px;margin:0 auto 18px;">
                Upload the four Salesforce exports on the Data Pipeline page to build the master
                file and unlock this view.
            </div>
        </div>
        """, unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 1.2, 1])
        with mid:
            if st.button('Go to Data Pipeline', icon=':material/database:', type='primary', width='stretch'):
                st.session_state.page = 'Data Pipeline'
                st.rerun()
    st.stop()


def plotly_cfg(fig, h=300):
    fig.update_layout(
        height=h, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans', size=11, color=SLATE),
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(orientation='h', y=1.14, font=dict(size=11)),
        xaxis=dict(gridcolor=LINE, linecolor=LINE),
        yaxis=dict(gridcolor=LINE, linecolor=LINE),
        colorway=COLOURS,
    )
    return fig


def chart(fig, h=300):
    st.plotly_chart(plotly_cfg(fig, h), width='stretch', config={'displayModeBar': False})


def _peek_columns(f):
    """Read just the header row of an uploaded CSV without consuming it —
    UploadedFile.getbuffer()/getvalue() ignore the read cursor, so a cheap
    nrows=0 peek here doesn't affect the full read done later for the pipeline."""
    try:
        f.seek(0)
        columns = list(pd.read_csv(f, nrows=0).columns)
        f.seek(0)
        return columns
    except Exception:
        f.seek(0)
        return None


def upload_slot(title, desc, icon_path, icon_cls, key, required_columns=None, disabled=False):
    """Renders an upload card. If required_columns is given, the uploaded
    file's header is checked against it so an obviously wrong file (e.g. the
    wrong CSV dropped into this slot) is caught immediately with a clear
    message instead of failing deep inside the pipeline. Returns (file, valid)."""
    valid = True
    with st.container(border=True):
        st.markdown(f"""
        <div class="sf-upload-head">
            <div class="sf-upload-icon {icon_cls}">
                <svg viewBox="0 0 24 24" stroke="currentColor">{icon_path}</svg>
            </div>
            <div>
                <div class="sf-upload-title">{title}</div>
                <div class="sf-upload-desc">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        f = st.file_uploader('Upload', type='csv', key=key, label_visibility='collapsed', disabled=disabled)
        if f:
            columns = _peek_columns(f) if required_columns else []
            missing = [] if columns is None else [c for c in required_columns or [] if c not in columns]

            if columns is None:
                valid = False
                st.markdown(f"""
                <div class="sf-upload-error">
                    <svg viewBox="0 0 24 24" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
                    </svg>
                    <div>{f.name} couldn't be read as a CSV. Check it isn't corrupted or a different file type.</div>
                </div>
                """, unsafe_allow_html=True)
            elif missing:
                valid = False
                st.markdown(f"""
                <div class="sf-upload-error">
                    <svg viewBox="0 0 24 24" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
                    </svg>
                    <div>Wrong file for this slot? {f.name} is missing: {', '.join(missing)}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="sf-upload-done">
                    <svg viewBox="0 0 24 24" fill="none" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                    {f.name} &middot; {fmt_size(f.size)}
                </div>
                """, unsafe_allow_html=True)
    return f, valid
