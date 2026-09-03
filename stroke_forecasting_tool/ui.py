import re
from contextlib import contextmanager

import pandas as pd
import streamlit as st

from auth import initials, sign_out
from branding import logo_white_data_uri
from db import count_new_runs, count_pending_requests

# ── BRAND PALETTE ────────────────────────────────────────────────────────────
# Swiss financial / data-dense analytics: pure white surfaces, hairline gray
# separators, restrained color in light mode; slate-900 surfaces with the
# same brand hues (brightened for contrast) in dark mode. BG/SURFACE are the
# same in light mode (page and login share white) -- the login page keeps
# its own separate gradient+dot-grid regardless of mode, defined in auth.py.
_LIGHT = dict(
    BG='#FFFFFF', SURFACE='#FFFFFF', SURFACE2='#F1F5F9', LINE='#E2E8F0', GRIDLINE='#F1F5F9',
    TEXT='#1E1E5F', MIST='#94A3B8', SLATE='#64748B', NAVY='#1E1E5F',
    TEAL='#00897B', TEAL2='#00B5A3', TEALLT='#E0F5F3',
    PURPLE='#6B2D8B', PURPLT='#F5EEF8',
    AMBER='#C17D0C', AMBERLT='#FEF9E7',
    BLUE='#1E1E5F', BLUELT='#EAEAF8',
    RED='#C0392B', REDLT='#FBEAE8',
    GREEN='#047857',
)
_DARK = dict(
    BG='#0F172A', SURFACE='#1E293B', SURFACE2='#334155', LINE='#334155', GRIDLINE='#334155',
    # NAVY was left at the same dark '#1E1E5F' in both modes on the
    # theory that a solid colour badge doesn't need to change -- wrong:
    # against a dark page (BG/SURFACE above are dark slate now) a navy
    # that dark reads as barely-there, near-invisible. Brightened to an
    # indigo-400 for dark mode specifically, staying in the same navy/
    # indigo family rather than switching to an unrelated hue.
    TEXT='#F1F5F9', MIST='#94A3B8', SLATE='#CBD5E1', NAVY='#818CF8',
    TEAL='#2DD4BF', TEAL2='#5EEAD4', TEALLT='rgba(45,212,191,0.16)',
    PURPLE='#C4B5FD', PURPLT='rgba(196,181,253,0.16)',
    AMBER='#FBBF24', AMBERLT='rgba(251,191,36,0.16)',
    BLUE='#93C5FD', BLUELT='rgba(147,197,253,0.16)',
    RED='#FCA5A5', REDLT='rgba(252,165,165,0.16)',
    GREEN='#34D399',
)
# Module-level names every other function in this file references via
# f-string interpolation (e.g. "{TEXT}") -- initialized to light mode here
# so anything that somehow runs before inject_global_css() still has a
# sane value, then reassigned by _apply_palette() every rerun based on
# st.session_state.dark_mode. Python looks these names up at CALL time,
# not at the time each function was defined, so mutating them here is
# enough for kpi()/card()/plotly_cfg()/etc. to all pick up the current
# theme without threading a theme parameter through every one of them.
BG, SURFACE, SURFACE2, LINE, GRIDLINE = (
    _LIGHT['BG'], _LIGHT['SURFACE'], _LIGHT['SURFACE2'], _LIGHT['LINE'], _LIGHT['GRIDLINE'])
TEXT, MIST, SLATE, NAVY = _LIGHT['TEXT'], _LIGHT['MIST'], _LIGHT['SLATE'], _LIGHT['NAVY']
TEAL, TEAL2, TEALLT = _LIGHT['TEAL'], _LIGHT['TEAL2'], _LIGHT['TEALLT']
PURPLE, PURPLT = _LIGHT['PURPLE'], _LIGHT['PURPLT']
AMBER, AMBERLT = _LIGHT['AMBER'], _LIGHT['AMBERLT']
BLUE, BLUELT = _LIGHT['BLUE'], _LIGHT['BLUELT']
RED, REDLT = _LIGHT['RED'], _LIGHT['REDLT']
GREEN = _LIGHT['GREEN']
COLOURS = [TEAL, PURPLE, AMBER, RED, TEAL2, BLUE]


def _apply_palette():
    """Reassigns every module-level palette constant above from _LIGHT or
    _DARK based on st.session_state.dark_mode. Call once, first thing,
    in inject_global_css() -- every function below that runs later in
    the same script pass (kpi, card, stage_row, plotly_cfg, render_*,
    ...) will see whichever value was set here, since they all just
    read these same module-level names when they actually run."""
    global BG, SURFACE, SURFACE2, LINE, GRIDLINE, TEXT, MIST, SLATE, NAVY
    global TEAL, TEAL2, TEALLT, PURPLE, PURPLT, AMBER, AMBERLT, BLUE, BLUELT, RED, REDLT, GREEN, COLOURS
    p = _DARK if st.session_state.get('dark_mode') else _LIGHT
    BG, SURFACE, SURFACE2, LINE, GRIDLINE = p['BG'], p['SURFACE'], p['SURFACE2'], p['LINE'], p['GRIDLINE']
    TEXT, MIST, SLATE, NAVY = p['TEXT'], p['MIST'], p['SLATE'], p['NAVY']
    TEAL, TEAL2, TEALLT = p['TEAL'], p['TEAL2'], p['TEALLT']
    PURPLE, PURPLT = p['PURPLE'], p['PURPLT']
    AMBER, AMBERLT = p['AMBER'], p['AMBERLT']
    BLUE, BLUELT = p['BLUE'], p['BLUELT']
    RED, REDLT = p['RED'], p['REDLT']
    GREEN = p['GREEN']
    COLOURS = [TEAL, PURPLE, AMBER, RED, TEAL2, BLUE]

NAV_SECTIONS = [
    ('Pipeline', [
        ('Data Pipeline', ':material/database:'),
    ]),
    ('Dashboards', [
        ('Overview',             ':material/dashboard:'),
        ('Income Forecast',      ':material/trending_up:'),
        ('Retention Analysis',   ':material/groups:'),
        ('Donor Lifetime Value', ':material/savings:'),
        ('Supplier Insights',    ':material/handshake:'),
        ('Campaign ROI',         ':material/campaign:'),
    ]),
    ('Operations', [
        ('Run History',      ':material/history:'),
        ('Access Requests',  ':material/how_to_reg:'),
    ]),
]
NAV_ITEMS = [item for _, items in NAV_SECTIONS for item in items]

# Hidden from the sidebar (and blocked with their own page-level guard in
# app.py, defense in depth) for anyone whose role isn't Administrator --
# see render_sidebar()'s is_admin filter below. Data Pipeline mutates the
# shared dashboard state everyone sees next; Access Requests grants
# account access. Both are sensitive enough to gate, unlike everything
# else in NAV_SECTIONS, which is read-only for any signed-in user.
ADMIN_ONLY_NAV_ITEMS = {'Data Pipeline', 'Access Requests'}


def _slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def fmt_size(num_bytes: float) -> str:
    if num_bytes >= 1024 * 1024:
        return f'{num_bytes / (1024 * 1024):.1f} MB'
    return f'{num_bytes / 1024:.0f} KB'


# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
def inject_global_css():
    _apply_palette()
    # One-shot flag: True only for the single rerun immediately following a
    # sidebar nav click (set there, right before its st.rerun() -- see
    # render_sidebar()), read and immediately reset here so it can never
    # leak into a LATER, unrelated rerun. This is what scopes the loading
    # overlay below to page-to-page navigation specifically -- without it,
    # :has([data-testid="stStatusWidget"]) alone can't tell a nav click
    # apart from any other rerun (a widget tweak, a Data Pipeline upload,
    # the pipeline's own multi-stage run), which was the actual bug behind
    # "it's running on Data Pipeline too": every rerun looks identical at
    # the DOM level, so the overlay showed for all of them until this
    # extra, deliberately-narrow signal was added.
    _nav_loading = st.session_state.get('nav_loading', False)
    st.session_state.nav_loading = False
    _nav_overlay_css = f"""
    [data-testid="stAppViewContainer"]:has([data-testid="stStatusWidget"]) [data-testid="stMain"] {{
        filter: blur(3px);
        transition: filter 0.2s ease 0.2s;
        pointer-events: none;
    }}
    [data-testid="stAppViewContainer"]:has([data-testid="stStatusWidget"])::after {{
        opacity: 1; animation: sf-spin 0.8s linear infinite;
        transition: opacity 0.15s ease 0.2s;
    }}
    """ if _nav_loading else ''
    st.html(f"""
    <style>
    footer, [data-testid="stDecoration"], [data-testid="stAppDeployButton"],
    [data-testid="stMainMenu"] {{ display: none !important; }}
    /* pointer-events:none so this invisible native bar can never again
    swallow real clicks meant for whatever sits under/behind it (this is
    exactly what broke the topbar's popover before) -- every button it
    would normally hold (deploy, hamburger menu) is already hidden via
    display:none above, so nothing here still needs to be clickable.
    stHeader alone wasn't enough -- re-tested live and clicks were STILL
    being swallowed -- because pointer-events doesn't cascade past a
    descendant that sets its own value: stToolbar (nested inside
    stHeader) carries its own pointer-events:auto from Streamlit's own
    styling, which overrides the ancestor's none right back. Needs its
    own explicit override too. */
    [data-testid="stHeader"] {{ background: transparent; pointer-events: none !important; }}
    [data-testid="stToolbar"] {{ pointer-events: none !important; }}
    /* One genuine exception: the "expand sidebar" control that appears in
    the main content area once the sidebar's been collapsed also lives
    inside stToolbar -- confirmed live a real mouse click on it was
    silently swallowed by the blanket pointer-events:none above (only a
    programmatic .click() got through, since that bypasses pointer-events
    entirely), which would have made a collapsed sidebar impossible to
    ever reopen. Icon colour is a fixed navy from config.toml's static
    (non-dynamic) theme, same class of issue as every other native-widget
    dark-mode fix elsewhere in this file -- confirmed live it was near
    invisible against a dark main content area, so it needs the dynamic
    {{TEXT}} colour here too. */
    [data-testid="stExpandSidebarButton"] {{ pointer-events: auto !important; }}
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {{ color: {TEXT} !important; }}

    /* ── Loading overlay (page navigation only) ── */
    /* Streamlit's own indicator during a script run is a tiny top-right
    spinner plus a per-element fade on whatever's stale -- easy to miss,
    per feedback "very primitive". Replaced with a centered spinner over
    a blurred main content area instead, but ONLY for an actual sidebar
    page-to-page navigation -- not for every rerun in general (a widget
    tweak, a Data Pipeline upload, the pipeline's own multi-stage run all
    rerun the script exactly the same way at the DOM level, which was the
    real bug behind an earlier version of this showing up on Data
    Pipeline too: :has([data-testid="stStatusWidget"]) alone can't tell
    those apart from a nav click). The two rules that actually turn the
    overlay on are therefore built conditionally in Python, in
    _nav_overlay_css above -- included in this CSS only on the one rerun
    immediately following a nav click (that one-shot session_state flag
    is set in render_sidebar()'s nav-button handler, right before its
    st.rerun()) -- and spliced in below. Everything here stays a no-op
    (never matches) on every other rerun, since [data-testid=
    "stStatusWidget"] briefly existing is no longer sufficient on its own.
    [data-testid="stStatusWidget"] itself -- confirmed live (polled the
    DOM every 300ms through an artificial delay) -- is only present while
    a script is actually running: appears the instant a rerun starts,
    removed a beat after the new content finishes streaming in, so no
    extra JavaScript is needed to know when to show/hide this.
    A short transition-delay, present only on the WAY IN (not the way
    out), additionally debounces this: a nav rerun that somehow completes
    in well under 200ms wouldn't visibly flash it either. filter:blur()
    is scoped to [data-testid="stMain"] only, not the sidebar, so nav
    stays usable while a background rerun is still settling; the spinner
    itself is a ::after on [data-testid="stAppViewContainer"] (stMain's
    ANCESTOR, not stMain itself) specifically so stMain's blur filter --
    which also applies to any pseudo-element that were its own -- never
    blurs the spinner along with the content behind it. */
    @keyframes sf-spin {{ to {{ transform: translate(-50%, -50%) rotate(360deg); }} }}
    [data-testid="stMain"] {{
        transition: filter 0.2s ease 0s;
    }}
    [data-testid="stAppViewContainer"] {{ position: relative; }}
    [data-testid="stAppViewContainer"]::after {{
        content: '';
        position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
        width: 40px; height: 40px; border-radius: 50%;
        border: 3px solid {LINE}; border-top-color: {TEAL};
        opacity: 0; pointer-events: none; z-index: 1000;
        transition: opacity 0.15s ease 0s;
    }}
    {_nav_overlay_css}

    /* Swiss financial / data-dense analytics: pure white, hairline gray
    separators, tabular numerics, no rounded flourishes or shadows except
    on the login page (styled separately in auth.py). */
    /* color here (not just background) so plain, otherwise-unstyled
    native text -- st.write(), st.caption(), dataframe labels, anything
    that doesn't set its own explicit color -- inherits a readable
    colour in dark mode too, instead of staying stuck at config.toml's
    fixed (non-dynamic) light-mode text colour. Inheritance means our
    own explicit-coloured elements (pills, badges, kpi values) are
    unaffected -- a more specific declared color always wins over an
    inherited one regardless of this. */
    .stApp {{ background: {BG}; color: {TEXT}; }}
    /* Streamlit's own nested wrappers between .stApp and the actual page
    content can carry their own opaque background from config.toml's
    fixed (non-dynamic) backgroundColor -- transparent here so .stApp's
    background (the one line above that actually responds to dark mode)
    is what's visible, not a static white layer painted on top of it. */
    [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {{ background: transparent !important; }}
    /* max-width was 1152px, matching the reference app's own max-w-6xl
    exactly -- confirmed by measuring its live DOM. But that reference
    was only ever viewed on ~1280px-wide browser windows, where the cap
    never actually binds (1280 - 240px sidebar = 1040px available, already
    under 1152px, so no side margin ever appears there in practice). On a
    genuinely wide monitor the same cap creates a large, obvious idle
    margin on both sides that never showed up in the reference screenshots.
    Widened well past normal desktop widths so it only kicks in on very
    large/ultrawide displays, instead of on every ordinary wide window. */
    .block-container {{
        padding-top: 0.75rem !important; padding-bottom: 2.5rem !important;
        padding-left: 2rem !important; padding-right: 2rem !important;
        max-width: 1920px !important; margin: 0 auto !important;
    }}
    /* padding-top was 2.5rem (40px) -- reported live as wasted empty
    space above every page's title, most obviously with the cached-run
    banner retracted (it's position:fixed and hidden above the
    viewport by default, so it was never what reserved this room; nothing
    was). Cut to a small amount just for basic breathing room instead of
    zero, kept uniform across ALL pages/states rather than only when
    data_source=='cached' -- making it conditional on that would shift
    the whole page down/up depending on which data source is active,
    which is worse than a slightly larger fixed gap. stHeader (the
    invisible, pointer-events:none native Streamlit bar above, ~60px
    tall but position:absolute so it never affected this padding either
    way) still sits over part of this reduced gap -- harmless, it
    paints nothing and swallows no clicks. */
    * {{ font-variant-numeric: tabular-nums; }}

    /* Bordered blocks (card/kpi containers) carry an overflow="visible"
    attribute (needed for Streamlit's own border-radius clip, now zeroed
    via config.toml's baseRadius), which is a more reliable selector here
    than emotion-cache class hashes. Hairline border, no shadow, square
    corners -- structure comes from the border, not elevation. */
    [data-testid="stVerticalBlock"][overflow="visible"] {{
        background: {SURFACE} !important;
        border: 1px solid {LINE} !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        transition: border-color 0.15s ease;
    }}
    [data-testid="stVerticalBlock"][overflow="visible"]:hover {{ border-color: {TEAL} !important; }}
    [data-testid="stMetric"] {{
        background: {SURFACE} !important;
        border: 1px solid {LINE} !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        padding: 18px 20px 16px !important;
    }}
    [data-testid="stMetricValue"] {{ white-space: nowrap; font-variant-numeric: tabular-nums; }}
    /* Streamlit wraps each card in its own single-child flex-direction:column
    wrapper. The old "flex: 1 1 210px" here set flex-basis: 210px meaning to
    size the card's WIDTH -- but flex-basis sizes along the MAIN axis, which
    for this wrapper is vertical, not horizontal, so it was flooring every
    card's HEIGHT at 210px regardless of content instead. min-width alone
    already achieves the intended width floor and isn't axis-dependent, so
    that's all this needs. */
    [class*="st-key-kpi_"] {{ min-width: 210px; align-self: flex-start !important; }}
    [data-testid="stMetricLabel"] p {{
        font-size: 10px !important; font-weight: 500 !important;
        letter-spacing: 0.18em; text-transform: uppercase; color: {SLATE} !important;
    }}
    [data-testid="stMetricValue"] {{
        font-weight: 600 !important; font-size: 1.75rem !important;
        font-family: 'Space Grotesk', system-ui, sans-serif !important; color: {TEXT} !important;
    }}
    /* KPI value is uniform navy per spec ("value ... text-navy") -- only
    the delta row is colour-coded (green up / red down), not the headline
    number itself. Confirmed against real Base44 screenshots: every KPI
    value renders in the same dark navy regardless of what it measures. */
    /* KpiCard has no leading icon per spec (label -> value -> delta ->
    subtext only) -- hide whatever icon st.metric's icon= param renders,
    rather than editing every kpi() call site that still passes one. */
    [data-testid="stMetricLabel"] [data-testid^="stIcon"] {{ display: none !important; }}

    /* ── Sidebar ── */
    /* Streamlit's native sidebar header -- the collapse-arrow row above
    the real content -- used to be hidden entirely (display:none) on the
    assumption it always reserved/overlapped space awkwardly. Re-enabled
    per request ("make the sidebar collapsible"). Its native computed
    style differs across Streamlit installs -- this environment's own
    test render showed height:52.5px + margin-bottom:14px, the user's
    real browser showed margin-bottom:-2rem + height:3.75rem (confirmed
    via their own DevTools computed-style paste). margin-bottom:-2rem is
    the correct, wanted value (per explicit confirmation, not a bug), so
    it's pinned here rather than left to whichever default a given
    install happens to ship. height is deliberately left undeclared, not
    set to auto -- no height rule for this selector here at all, per
    explicit request, rather than an explicit auto override.
    display:flex/justify-content/align-items are left alone too, neither
    one was ever the issue. */
    [data-testid="stSidebarHeader"] {{ margin-bottom: -2rem !important; }}
    /* Streamlit's own default here is padding-bottom: 6rem (96px) --
    confirmed via its actual computed style -- which was never actually
    zeroed before (only padding-top was), leaving a large reserved gap
    below the sidebar's real content. */
    [data-testid="stSidebarUserContent"] {{
        display: flex; flex-direction: column; min-height: 100vh;
        padding-top: 0 !important; padding-bottom: 0 !important;
    }}
    .sf-side-spacer {{ flex: 1 1 auto; }}
    /* Fixed Stroke Foundation navy, not the dynamic {{SURFACE}} palette
    variable -- the sidebar is a constant brand element, deliberately
    independent of the light/dark-mode toggle (which only ever affects
    the main content area). Every text colour below is a fixed white/
    rgba-white for the same reason: {{TEXT}}/{{SLATE}}/{{MIST}}/{{LINE}}
    track that toggle, and in light mode {{TEXT}} is literally this same
    navy -- using it here would be invisible-on-navy. This matches the
    original design (see the initial commit) before an earlier request
    ("undo the sidebar changes") reworked it to a white sidebar. */
    [data-testid="stSidebar"] {{ background: #1E1E5F !important; border-right: 1px solid rgba(255,255,255,0.08) !important; }}
    .sf-side-logo {{ padding: 20px 18px 16px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 4px; }}
    /* Real Stroke Foundation logo image -- the white-wordmark variant
    (logo_white_data_uri() in render_sidebar()), since the navy-text
    logo.png used elsewhere in the app would disappear against this
    background. Falls back to the small teal mark below only if the
    asset is ever missing. */
    .sf-side-mark {{ display: block; height: 34px; width: auto; margin-bottom: 8px; }}
    .sf-side-mark-fallback {{ width: 30px; height: 30px; border-radius: 7px; background: {TEAL};
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
    /* justify-content:flex-start on the button itself only positions its
    inner wrapper div within the button -- that wrapper (nearly as wide
    as the button, ~224px of 259px, confirmed by inspecting the actual
    button DOM) has its OWN justify-content:center centering the icon+
    label as a unit inside itself, which flex-start on the outer button
    never reaches. This is what was actually still centered. */
    [data-testid="stSidebar"] .stButton > button > div {{ justify-content: flex-start !important; }}
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

    /* ── Account cluster (avatar+name popover, Export CSV) ── */
    /* Lives inside page_header() now, right-aligned at the same vertical
    level as the page title -- there is no separate topbar any more
    (removed per request). The whole profile trigger (round picture +
    name) is ONE st.popover() button -- previously the picture and the
    name were two separate elements (only the picture was clickable),
    which is exactly the "picture or name should both open sign out" gap
    that fixed. The button's real label text (the name) is left visible
    this time -- unlike the picture-only version, there's no cramped-
    width squeeze here since the button sizes naturally to fit an actual
    name, so Streamlit's own trailing chevron icon can stay too (a
    chevron next to an account name is normal, expected UI, not a bug
    to hide). The round avatar itself is a ::before pseudo-element
    (carrying the initials, injected per-render in page_header() since
    this block doesn't know the current user) sitting in front of the
    real label text, not a separate element -- so clicking anywhere
    across picture+name+chevron is all the same one button. */
    /* The button's own wrapper div (Streamlit adds one per widget, its
    class named after the widget's own key, testid stLayoutWrapper) is
    itself shrink-to-fit -- confirmed live its own width matches the
    button's exactly, NOT the column's -- so it sat flush left against
    the column's actual full-width ancestor (a sibling stVerticalBlock,
    confirmed by walking the live DOM) by default, out of alignment with
    the right-aligned "RUN ..."/meta text directly above it in that same
    column. margin-left:auto pushes this shrink-to-fit wrapper itself to
    the right edge of its parent to match -- justify-content on the
    wrapper's own box (tried first) did nothing, since a box with no
    slack of its own has nothing for justify-content to redistribute. */
    .st-key-topbar_user_menu {{ margin-left: auto; }}
    .st-key-topbar_user_menu button {{
        display: flex !important; align-items: center !important; gap: 8px !important;
        background: transparent !important; border: none !important;
        padding: 4px 10px 4px 4px !important; border-radius: 20px !important;
        box-shadow: none !important;
    }}

    /* ── "Showing a cached run" banner -- peekaboo top bar, hidden above
    the viewport by default, revealed briefly by render_cached_run_
    banner()'s own JS (a persistent, always-on mousemove listener + a
    hook the nav-click script below also calls), not by Python
    conditionally rendering it. A plain st.markdown() div, deliberately
    NOT an st.container(key=...) this time -- the earlier version needed
    one only to hold a real dismiss button; with no button at all here,
    a plain div sidesteps two real bugs the container version hit
    (an unkeyed/keyed container inheriting the global bordered-card
    look, and a wildcard CSS selector accidentally matching more than
    one similarly-prefixed key) entirely, rather than working around
    them again. left:300px, not 0 -- offset to sit beside the sidebar,
    not over it, per explicit request; 300px matches this sidebar's own
    actual rendered width (confirmed live earlier this session), not a
    guess. Colour is {BLUELT}/{TEXT} -- the same tokens already used
    elsewhere for this exact light info-blue, not a new hardcoded
    colour, so it stays theme-aware in dark mode along with everything
    else. pointer-events:none unconditionally: nothing inside it is
    interactive (no dismiss button this version), so it should never be
    able to intercept a click meant for whatever's underneath it during
    its brief appearance. ── */
    .sf-cached-run-banner {{
        position: fixed; top: 0; left: var(--sf-sidebar-w, 300px); right: 0; z-index: 999999;
        background: {BLUELT}; color: {TEXT};
        box-shadow: 0 4px 20px rgba(0,0,0,0.10);
        padding: 10px 20px; font-size: 13.5px; line-height: 1.5;
        pointer-events: none;
        transform: translateY(-110%);
        transition: transform 0.35s cubic-bezier(0.4,0,1,1);
    }}
    .sf-cached-run-banner b {{ font-weight: 700; }}
    /* Slide-down uses its OWN transition (duration + easing), separate
    from the base rule's retract transition above -- confirmed live this
    resolves correctly per direction: adding .sf-banner-visible
    transitions using THIS rule's 0.3s ease-out-ish curve, removing it
    transitions back using the base rule's 0.35s ease-in-ish curve, per
    spec (no bounce/overshoot going down, a slightly different curve
    retracting). */
    .sf-cached-run-banner.sf-banner-visible {{
        transform: translateY(0);
        transition: transform 0.3s cubic-bezier(0.22,1,0.36,1);
    }}
    /* Reported live, found via DevTools: even hidden/retracted, the
    banner still left a 16px gap above the page title. Not this banner's
    own CSS -- confirmed the culprit is Streamlit's own stVerticalBlock
    wrapper (display:flex, gap:1rem, applies BETWEEN every child element
    in a block), which still counts the banner's own zero-height
    stElementContainer wrapper as one of those children even though
    .sf-cached-run-banner itself is position:fixed and out of flow, so a
    full gap was still inserted before the next real element. :has()
    targets JUST that one wrapper (never every stElementContainer --
    would strip intentional spacing everywhere else) and takes it out of
    flex layout entirely via display:contents, so it no longer counts
    toward gap at all; its child (the actual banner div) is unaffected,
    since display:contents only removes the WRAPPER's own box, not its
    children's rendering. */
    [data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] .sf-cached-run-banner) {{
        display: contents;
    }}
    </style>
    """)
    # Split into a second st.html() call here, roughly halfway through --
    # confirmed live, more than once now, that ONE st.html() call carrying
    # this entire stylesheet (~35KB+) can silently render as if it had
    # never been called at all: every rule from it missing from the
    # page's <style> tags, with no error anywhere. Re-lost this split
    # once already when this file was reconstructed from an earlier
    # snapshot predating the fix -- re-added here, and this time the
    # reason it exists is spelled out in-line (not just in a
    # conversation) specifically so it survives the next such
    # reconstruction. Exact threshold not identified; split roughly in
    # half at a safe rule boundary rather than chase the precise cutoff.
    st.html(f"""
    <style>
    .st-key-topbar_user_menu button:hover {{ background: {SURFACE2} !important; }}
    .st-key-topbar_user_menu button::before {{
        display: flex !important; align-items: center !important; justify-content: center !important;
        width: 30px !important; height: 30px !important; min-width: 30px !important;
        border-radius: 50% !important; background: {NAVY} !important; color: white !important;
        font-size: 11px !important; font-weight: 700 !important; flex-shrink: 0 !important;
    }}
    .st-key-topbar_user_menu button p {{
        font-size: 12px !important; font-weight: 600 !important; color: {TEXT} !important; margin: 0 !important;
    }}
    /* Sits directly under the avatar+name button in the same narrow
    column -- deliberately understated (outline, not solid) so it doesn't
    compete with Log out for attention as the primary action in this
    corner. */
    .st-key-topbar_export_btn {{ margin-top: 6px; }}
    .st-key-topbar_export_btn button {{
        background: transparent !important; color: {SLATE} !important;
        border: 1px solid {LINE} !important; font-size: 11px !important;
        font-weight: 600 !important; padding: 4px 10px !important;
        box-shadow: none !important;
    }}
    .st-key-topbar_export_btn button:hover {{
        background: {SURFACE2} !important; color: {TEAL} !important; border-color: {TEAL} !important;
    }}
    /* st.segmented_control (used for "Forecast horizon", "Forecast
    model", "Months to show", "Segment by") is another native widget
    styled from config.toml's fixed theme -- same issue as alerts/
    inputs before, confirmed live: unselected options rendered as solid
    white pills with near-invisible near-white text (inherited from our
    own dark-mode .stApp color rule, sitting on a background that never
    got the same treatment). data-selected="true" marks the active
    option -- kept solid teal, which already looked correct and didn't
    need fixing; only the unselected default needed to stop being
    static white. */
    [data-testid="stButtonGroup"] button {{
        background: {SURFACE} !important; color: {TEXT} !important; border-color: {LINE} !important;
    }}
    [data-testid="stButtonGroup"] button[data-selected="true"] {{
        background: {TEAL} !important; color: white !important; border-color: {TEAL} !important;
    }}
    /* st.file_uploader's dropzone -- and the "Browse files" button
    inside it -- both had the exact same static-white-background/
    near-invisible-text issue, confirmed live. */
    [data-testid="stFileUploaderDropzone"] {{
        background: {SURFACE} !important; color: {TEXT} !important; border-color: {LINE} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        background: {SURFACE2} !important; color: {TEXT} !important; border-color: {LINE} !important;
    }}
    .sf-topbar-menu-name {{ font-size: 13px; font-weight: 600; color: {TEXT}; }}
    .sf-topbar-menu-role {{ font-size: 11px; color: {MIST}; margin-bottom: 10px; }}
    /* The popover's floating panel is Streamlit-native chrome, styled
    from config.toml's fixed (non-dynamic) theme colours -- confirmed
    live it stayed white/light-text even with everything else in dark
    mode. Needs its own explicit override to actually follow dark mode. */
    [data-testid="stPopoverBody"] {{
        background: {SURFACE} !important; color: {TEXT} !important; border-color: {LINE} !important;
    }}
    /* st.info/warning/error/success and st.text_input/selectbox/
    multiselect/number_input/date_input are all Streamlit-native widgets
    styled from config.toml's fixed (non-dynamic) theme colours, exactly
    the same class of issue as stPopoverBody above -- confirmed live in
    dark mode they were the actual "reverse colour except the white
    part" report: our own elements correctly flipped colour, these
    stayed at their static light-mode tint/white regardless of the
    toggle. Each needs its own override for the same reason. Alert
    variants use :has() to pick the right tint per type -- verified
    supported earlier this session; there's no "light green" token in
    the palette for the success variant so it reuses TEALLT/GREEN,
    which is the same colour family already used for success elsewhere
    (kpi() deltas, stage-row "done" state). */
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{
        background: {BLUELT} !important; color: {BLUE} !important;
    }}
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{
        background: {AMBERLT} !important; color: {AMBER} !important;
    }}
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{
        background: {REDLT} !important; color: {RED} !important;
    }}
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{
        background: {TEALLT} !important; color: {GREEN} !important;
    }}
    [data-testid="stTextInputRootElement"],
    [data-testid="stSelectbox"] [role="group"],
    [data-testid="stMultiSelect"] [role="group"],
    [data-testid="stNumberInput"] [role="group"],
    [data-testid="stDateInput"] [role="group"] {{
        background: {SURFACE} !important; border-color: {LINE} !important;
    }}
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] input, [data-testid="stDateInput"] input {{
        color: {TEXT} !important;
    }}
    /* st.dataframe renders its cells via an internal canvas grid that
    reads theme colours at component-init time, not plain styled DOM --
    this can only reach the outer chrome (border/toolbar), not the cell
    content itself, so a dataframe's actual grid will stay light-themed
    even in dark mode. A real limitation of the widget, not something a
    CSS override can fully fix. */
    [data-testid="stDataFrame"] {{ border-color: {LINE} !important; }}

    /* ── Page header -- eyebrow typography per spec: font-heading
    text-[10px] font-medium uppercase tracking-[0.18em] text-slate-500,
    used identically for the page eyebrow and the header's metadata pill. ── */
    .sf-eyebrow {{ font-family: 'Space Grotesk', system-ui, sans-serif; font-size: 10px; font-weight: 500;
        letter-spacing: 0.18em; text-transform: uppercase; color: {SLATE}; margin-bottom: 4px; }}
    .sf-page-title {{ font-family: 'Space Grotesk', system-ui, sans-serif; font-weight: 600; font-size: 24px; color: {TEXT}; margin-bottom: 3px; line-height: 1.2; letter-spacing: -0.01em; }}
    .sf-page-sub {{ font-size: 13px; color: {SLATE}; margin-bottom: 6px; max-width: 720px; }}
    /* page_header()'s optional `info` tooltip: st.subheader(..., help=info)
    instead of the plain .sf-page-title div, so it gets Streamlit's own
    native tooltip icon+bubble (same mechanism kpi() already uses via
    st.metric's help= param) rather than a second, custom-built tooltip
    system. Restyled here to look identical to .sf-page-title -- scoped to
    this one container key so it can never affect an unrelated st.subheader
    anywhere else in the app. Streamlit's own vertical-block gap (normally
    ~1rem between elements) is zeroed on the whole header block so the
    eyebrow/title/sub stack stays exactly as tight as the old single-div
    version did. */
    /* Any st.container(key=...) -- confirmed live, even without
    border=True -- carries the same overflow="visible" attribute a real
    bordered container does, so both of these picked up the global
    bordered-card look (background + hairline border) unintentionally.
    These two are pure layout grouping, not cards -- un-style them the
    same way an accidentally-bordered nested block gets un-styled
    elsewhere in this file. */
    [data-testid="stVerticalBlock"].st-key-page_header_block,
    [data-testid="stVerticalBlock"].st-key-page_header_title,
    [data-testid="stVerticalBlock"].st-key-page_header_row {{
        background: transparent !important; border: none !important; box-shadow: none !important;
    }}
    .st-key-page_header_block {{ margin-bottom: 18px; }}
    .st-key-page_header_block [data-testid="stVerticalBlock"] {{ gap: 0 !important; }}
    /* Reported live: even after the cached-run banner's own wrapper was
    taken out of flex layout (display:contents, see .sf-cached-run-banner
    below), there was STILL a visible gap between the page header row
    (eyebrow/title/sub + account cluster -- an st.columns() call, wrapped
    in this container only because st.columns() itself doesn't take a
    key in this Streamlit version) and whatever content follows it --
    Streamlit's stVerticalBlock gap:1rem applies between EVERY pair of
    top-level children, and this row is one of them. Can't zero that
    gap globally -- it's what spaces every OTHER card from the one below
    it on every page, all the way down; zeroing it here would just move
    the "cards touching each other" bug onto the very first card
    instead of fixing anything. A negative margin-bottom the exact size
    of that gap on THIS ONE row only cancels the specific gap that
    follows the header, leaving every later gap between actual content
    cards untouched. */
    .st-key-page_header_row {{ margin-bottom: -16px; }}
    .st-key-page_header_title [data-testid="stHeading"] {{ margin: 0 !important; padding: 0 !important; }}
    .st-key-page_header_title h3 {{
        font-family: 'Space Grotesk', system-ui, sans-serif !important; font-weight: 600 !important;
        font-size: 24px !important; color: {TEXT} !important; line-height: 1.2 !important;
        letter-spacing: -0.01em !important; margin: 0 0 3px !important;
    }}
    .st-key-page_header_title [data-testid="stTooltipIcon"] svg {{ width: 15px; height: 15px; color: {SLATE}; }}
    .sf-page-meta {{ font-family: 'Space Grotesk', system-ui, sans-serif; font-size: 10px; font-weight: 500;
        letter-spacing: 0.18em; text-transform: uppercase; color: {SLATE}; white-space: nowrap; padding-bottom: 5px; }}
    /* Sits below the account cluster (avatar+name popover, Export CSV)
    in page_header()'s narrow right column now -- was above it, moved
    per request -- instead of bottom-aligned next to the title in a wide
    flex row like the base .sf-page-meta rule above was written for.
    margin-top not margin-bottom now that it trails those elements
    rather than leading them. */
    .sf-page-meta-account {{ text-align: right; padding-bottom: 0; margin-top: 6px; white-space: normal; }}

    /* ── Status pills (Run History, sidebar active-run box, pipeline stage rows) --
    square corners, consistent with the rest of the square-cornered system. ── */
    .sf-pill {{ display: inline-flex; align-items: center; gap: 4px; padding: 3px 9px; border-radius: 0;
        font-size: 10px; font-weight: 600; letter-spacing: 0.02em; white-space: nowrap; }}
    .sf-pill-green  {{ background: {TEALLT};  color: {GREEN}; }}
    .sf-pill-amber  {{ background: {AMBERLT}; color: {AMBER}; }}
    .sf-pill-red    {{ background: {REDLT};   color: {RED}; }}
    .sf-pill-blue   {{ background: {BLUELT};  color: {BLUE}; }}
    .sf-pill-gray   {{ background: {SURFACE2}; color: {SLATE}; }}

    /* ── Pipeline stage rows (Data Pipeline page, running state) ── */
    .sf-stage-row {{ display: flex; align-items: center; gap: 12px; padding: 9px 2px; border-bottom: 1px solid {LINE}; }}
    .sf-stage-row:last-child {{ border-bottom: none; }}
    .sf-stage-num {{ width: 20px; height: 20px; border-radius: 0; flex-shrink: 0; display: flex;
        align-items: center; justify-content: center; font-size: 10.5px; font-weight: 600; }}
    .sf-stage-num.pending {{ background: {SURFACE2}; color: {SLATE}; }}
    .sf-stage-num.running {{ background: {AMBERLT}; color: {AMBER}; }}
    .sf-stage-num.done    {{ background: {TEALLT};  color: {GREEN}; }}
    .sf-stage-num.warning {{ background: {AMBERLT}; color: {AMBER}; }}
    .sf-stage-num.error   {{ background: {REDLT};   color: {RED}; }}
    .sf-stage-name {{ font-size: 12px; font-weight: 600; color: {TEXT}; }}
    .sf-stage-cat  {{ font-size: 10px; color: {SLATE}; margin-top: 1px; }}
    .sf-stage-detail {{ width: 170px; flex-shrink: 0; text-align: right; font-size: 10.5px; color: {SLATE};
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    /* Per-stage progress bar. done/warning/error are real known end-states
    (solid fill) -- running is a sliding indeterminate bar, not a fake
    percentage, since most stages are atomic pass/fail with no genuine
    partial-progress signal to report. */
    .sf-stage-bar-track {{ flex: 1; height: 4px; background: {SURFACE2}; position: relative;
        overflow: hidden; min-width: 60px; }}
    .sf-stage-bar-fill {{ height: 100%; position: absolute; top: 0; }}
    .sf-stage-bar-fill.pending {{ width: 0; }}
    .sf-stage-bar-fill.done    {{ width: 100%; left: 0; background: {GREEN}; }}
    .sf-stage-bar-fill.warning {{ width: 100%; left: 0; background: {AMBER}; }}
    .sf-stage-bar-fill.error   {{ width: 100%; left: 0; background: {RED}; }}
    .sf-stage-bar-fill.running {{ width: 40%; background: {TEAL}; animation: sf-bar-slide 1.1s ease-in-out infinite; }}
    @keyframes sf-bar-slide {{ 0% {{ left: -40%; }} 100% {{ left: 100%; }} }}

    /* ── Overall pipeline progress bar (Data Pipeline page) ── */
    /* margin-bottom is a real fix, not decoration: verified via a
    disposable repro that Streamlit measures this element's wrapping
    container a few px SHORTER than what actually renders (confirmed with
    both a single combined st.markdown() call and two separate ones, and
    with/without going through an st.empty() placeholder -- same result
    every time), so the track was rendering flush against -- visually
    overflowing past -- the card's own bottom padding/border with zero
    clearance. The margin reserves genuine breathing room the container
    itself wasn't accounting for. */
    .sf-overall-track {{ height: 6px; background: {SURFACE2}; position: relative; overflow: hidden; margin-bottom: 8px; }}
    .sf-overall-fill {{ height: 100%; background: {TEAL}; transition: width 0.3s ease; }}

    /* ── Execution log panel -- a real persistent, scrolling, timestamped
    log (separate from the stage list), matching the reference app. ── */
    /* Same container-underestimates-its-own-height issue as the progress
    bar above -- verified the same way -- so this needs the same
    margin-bottom safety margin below it. */
    .sf-exec-log {{ border: 1px solid {LINE}; background: {SURFACE}; padding: 4px 16px;
        max-height: 220px; overflow-y: auto; margin-bottom: 8px; }}
    .sf-exec-log-line {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 11px; line-height: 2.1; color: {SLATE}; white-space: pre-wrap; }}
    .sf-exec-log-time {{ color: {MIST}; margin-right: 10px; }}

    /* ── Card header (used inside bordered containers) --
    font-heading text-sm font-semibold text-navy ── */
    .sf-card-title {{ font-family: 'Space Grotesk', system-ui, sans-serif; font-size: 14px; font-weight: 600; color: {TEXT}; }}
    /* card()'s optional `info` tooltip -- same technique as
    .st-key-page_header_title above (st.subheader(..., help=info)
    restyled to match the plain-div look), but keyed per-title
    (card_title_ + a slugified title) since card() can render many times on one page.
    [class*=...] matches any of those keys by shared prefix, same
    pattern as [class*="st-key-kpi_"] elsewhere in this file. The
    [data-testid="stVerticalBlock"] half of each selector isn't
    decorative -- it raises specificity to match (and, being later in
    this stylesheet, beat) the global bordered-block rule an unkeyed
    container would otherwise inherit; a bare [class*=...] selector
    alone loses that fight despite !important, since CSS ties on
    specificity, not just !important vs not. */
    [data-testid="stVerticalBlock"][class*="st-key-card_title_"] {{
        background: transparent !important; border: none !important; box-shadow: none !important;
    }}
    [class*="st-key-card_title_"] [data-testid="stHeading"] {{ margin: 0 !important; padding: 0 !important; }}
    [class*="st-key-card_title_"] h3 {{
        font-family: 'Space Grotesk', system-ui, sans-serif !important; font-weight: 600 !important;
        font-size: 14px !important; color: {TEXT} !important; line-height: 1.4 !important; margin: 0 !important;
    }}
    [class*="st-key-card_title_"] [data-testid="stTooltipIcon"] svg {{ width: 13px; height: 13px; color: {SLATE}; }}

    /* ── Buttons -- square corners (buttonRadius=0 in config.toml), teal
    primary / outline secondary per spec. ── */
    .stButton > button, .stDownloadButton > button {{ font-weight: 500 !important; font-size: 13px !important; }}
    .stDownloadButton > button {{ background: {SURFACE} !important; color: {SLATE} !important; border: 1px solid {LINE} !important; }}
    .stDownloadButton > button:hover {{ border-color: {TEAL} !important; background: {TEAL} !important; color: white !important; }}

    /* ── Pipeline flow diagram -- flat, hairline border, no shadow ── */
    .sf-pipeline {{ display: flex; align-items: center; background: {SURFACE}; border: 1px solid {LINE};
        border-radius: 0; padding: 16px 22px; margin-bottom: 18px; }}
    .sf-pipe-step {{ display: flex; align-items: center; gap: 9px; flex: 1; }}
    .sf-pipe-icon {{ width: 32px; height: 32px; border-radius: 0; flex-shrink: 0; display: flex; align-items: center; justify-content: center; }}
    .sf-pipe-icon svg {{ width: 15px; height: 15px; stroke-width: 2; fill: none; }}
    .sf-pipe-icon.p-teal, .sf-upload-icon.p-teal     {{ background: {TEALLT}; }}
    .sf-pipe-icon.p-teal svg, .sf-upload-icon.p-teal svg     {{ stroke: {TEAL}; }}
    .sf-pipe-icon.p-navy, .sf-upload-icon.p-navy     {{ background: {BLUELT}; }}
    .sf-pipe-icon.p-navy svg, .sf-upload-icon.p-navy svg     {{ stroke: {BLUE}; }}
    .sf-pipe-icon.p-purple, .sf-upload-icon.p-purple {{ background: {PURPLT}; }}
    .sf-pipe-icon.p-purple svg, .sf-upload-icon.p-purple svg {{ stroke: {PURPLE}; }}
    .sf-pipe-icon.p-amber, .sf-upload-icon.p-amber   {{ background: {AMBERLT}; }}
    .sf-pipe-icon.p-amber svg, .sf-upload-icon.p-amber svg   {{ stroke: {AMBER}; }}
    .sf-pipe-label {{ font-size: 11.5px; font-weight: 600; color: {TEXT}; }}
    .sf-pipe-desc  {{ font-size: 10px; color: {SLATE}; margin-top: 1px; }}
    .sf-pipe-arrow {{ color: {LINE}; font-size: 20px; padding: 0 8px; flex-shrink: 0; }}

    /* ── Upload slot ── */
    .sf-upload-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
    .sf-upload-icon {{ width: 34px; height: 34px; border-radius: 0; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
    .sf-upload-icon svg {{ width: 16px; height: 16px; stroke-width: 2; fill: none; }}
    .sf-upload-title {{ font-size: 12.5px; font-weight: 600; color: {TEXT}; }}
    .sf-upload-desc {{ font-size: 10.5px; color: {SLATE}; }}
    .sf-upload-done {{ display: flex; align-items: center; gap: 6px;
        margin-top: 10px; font-size: 11.5px; font-weight: 600; color: {GREEN}; }}
    .sf-upload-done svg {{ width: 14px; height: 14px; stroke: {GREEN}; stroke-width: 3; flex-shrink: 0; }}
    .sf-upload-error {{ display: flex; align-items: flex-start; gap: 8px; background: {REDLT}; border-radius: 0;
        padding: 8px 12px; margin-top: 10px; font-size: 11.5px; font-weight: 600; color: {RED}; }}
    .sf-upload-error svg {{ width: 14px; height: 14px; stroke: {RED}; stroke-width: 2.5; flex-shrink: 0; margin-top: 1px; }}

    [data-testid="stFileUploaderDropzone"] {{ border-radius: 0 !important; }}

    /* ── Sidebar disabled while pipeline runs ── */
    [data-testid="stSidebar"] .stButton > button:disabled {{
        opacity: 0.35 !important; cursor: not-allowed !important;
    }}
    /* Fixed rgba-white, not {{MIST}} -- same reasoning as the rest of the
    navy sidebar block above. */
    .sf-side-running-note {{ font-size: 10.5px; color: rgba(255,255,255,0.45); padding: 2px 18px 8px;
        display: flex; align-items: center; gap: 6px; }}

    /* ── Empty state ── */
    .sf-empty-icon {{ width: 52px; height: 52px; border-radius: 0; background: {TEALLT};
        display: flex; align-items: center; justify-content: center; margin: 6px auto 16px; }}

    /* ── Dataframe -- flat, square, hairline border via dataframeBorderColor
    (config.toml), no radius override needed now baseRadius=0. ── */
    [data-testid="stDataFrame"] {{ border-radius: 0 !important; overflow: hidden; }}
    </style>
    """)


# ── LAYOUT COMPONENTS ─────────────────────────────────────────────────────────
def _relative_time(dt) -> str:
    """'Loaded 3m ago' / '2h ago' style relative time for the sidebar's
    active-run box -- matches the Base44 prototype's session-age display."""
    if dt is None:
        return ''
    try:
        secs = (pd.Timestamp.now() - pd.Timestamp(dt)).total_seconds()
    except Exception:
        return ''
    if secs < 60:
        return 'just now'
    if secs < 3600:
        return f'{int(secs // 60)}m ago'
    if secs < 86400:
        return f'{int(secs // 3600)}h ago'
    return f'{int(secs // 86400)}d ago'


def render_sidebar():
    user = st.session_state.user or {'name': 'User', 'role': 'Analyst'}
    # Administrator-only: ADMIN_ONLY_NAV_ITEMS (Data Pipeline mutates the
    # shared dashboard state everyone sees next; Access Requests grants
    # account access) are hidden from nav for anyone else -- same role
    # check as app.py's own guard on each of those page bodies (defense
    # in depth, in case session_state.page is ever set to one some other
    # way), and as the delete-run/export-CSV gates elsewhere.
    is_admin = user['role'] == 'Administrator'
    running = bool(st.session_state.get('pipeline_running'))
    with st.sidebar:
        _logo_uri = logo_white_data_uri()
        _mark_html = (f'<img class="sf-side-mark" src="{_logo_uri}" alt="Stroke Foundation">' if _logo_uri
                      else '''<div class="sf-side-mark-fallback">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="white"
                     stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
            </div>
            <div class="sf-side-name">Stroke Foundation</div>''')
        st.markdown(f"""
        <div class="sf-side-logo">
            {_mark_html}
            <div class="sf-side-sub">Donor Forecasting Tool</div>
        </div>
        """, unsafe_allow_html=True)

        for section, items in NAV_SECTIONS:
            if not is_admin:
                items = [item for item in items if item[0] not in ADMIN_ONLY_NAV_ITEMS]
                if not items:
                    continue
            st.markdown(f'<div class="sf-side-section">{section}</div>', unsafe_allow_html=True)
            for label, icon in items:
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
                # Notification badges -- Run History: dashboard_runs saved
                # since this user last opened that page (count_new_runs,
                # cleared by mark_runs_seen() when app.py actually renders
                # it). Access Requests: the live pending-request count, no
                # "seen" state at all -- it's an open item needing a
                # decision, not a feed to catch up on, so the badge just
                # tracks decide_access_request() directly. Both queries
                # are cached (short ttl, see db.py) since this loop runs
                # on every page's sidebar, not just these two pages.
                badge = 0
                if label == 'Run History':
                    badge = count_new_runs(user.get('email', ''))
                elif label == 'Access Requests':
                    badge = count_pending_requests()
                if badge:
                    st.html(f"""<style>
                    .st-key-{key} button {{ position: relative; }}
                    .st-key-{key} button::after {{
                        content: "{badge if badge <= 99 else '99+'}";
                        position: absolute; top: 6px; right: 10px;
                        min-width: 16px; height: 16px; padding: 0 4px;
                        border-radius: 999px; background: {RED}; color: white;
                        font-size: 10px; font-weight: 700; line-height: 16px;
                        text-align: center; pointer-events: none;
                    }}
                    </style>""")
                if st.button(label, key=key, icon=icon, width='stretch', disabled=running):
                    st.session_state.page = label
                    # One-shot signal read (and immediately cleared) by
                    # inject_global_css() on the very next rerun -- this
                    # is what the loading overlay's blur+spinner is
                    # actually keyed on, see its comment there for why a
                    # plain :has([data-testid="stStatusWidget"]) alone
                    # isn't enough to scope it to nav clicks specifically.
                    st.session_state.nav_loading = True
                    st.rerun()

        if running:
            st.markdown("""
            <div class="sf-side-running-note">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                     stroke-width="2.5" stroke-linecap="round"><circle cx="12" cy="12" r="9"/>
                <path d="M12 7v5l3 3"/></svg>
                Navigation locked, pipeline running
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.pipeline_run:
            # Original read st.session_state.master (the raw dataframe)
            # directly here -- that key is no longer always populated (a
            # run loaded from Run History cache only sets master_rows/
            # donor_count, not the full dataframe), so this would throw
            # on that path. Same two numbers, just read from the fields
            # that are actually always present now -- no visual change.
            st.markdown(f"""
            <div style="height:1px;background:rgba(255,255,255,0.08);margin:10px 18px;"></div>
            <div style="padding:0 18px;font-size:11px;line-height:1.9;color:rgba(255,255,255,0.55);">
                <span style="color:white;font-weight:600;">Data loaded</span><br>
                {st.session_state.master_rows:,} rows &middot; {st.session_state.donor_count:,} signups<br>
                Forecast MAPE {st.session_state.mape:.1f}%
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sf-side-spacer"></div>', unsafe_allow_html=True)
        # Sign out no longer lives here -- moved to a popover behind the
        # avatar circle in the topbar (top-right of the main content).
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


def page_header(eyebrow, title, sub='', meta='', info=None):
    """Renders the page's own title block on the left, and -- since there
    is no separate topbar any more (removed per request: the account
    cluster used to sit in its own bar above every page, now it's fixed
    at the same vertical level as the page title instead) -- the account
    cluster on the right: the optional meta label, then the avatar+name
    popover (dark-mode toggle and Log out live inside it), then the
    current page's Export CSV button underneath that.

    User/running/export state is pulled straight from session_state
    rather than taken as params, the same way the old render_topbar()
    read `page`/`user` -- so all eight call sites across app.py stay
    exactly as they were (just eyebrow/title/sub/meta), no threading a
    new argument through every one of them. export_csv/export_filename
    specifically come from build_page_export_csv() in app.py, stashed
    into session_state right before page routing starts.

    IMPORTANT for callers: because this now renders the ONLY way to open
    the account menu (dark mode / log out) on any given page, every page
    must call this BEFORE its own `if not st.session_state.pipeline_run:
    empty_state()` guard, not after -- empty_state() ends the script with
    st.stop(), so calling page_header() afterward would mean it never
    renders at all pre-pipeline-run, silently taking log out and dark
    mode with it. Keep any dynamic bits of `sub` (session_state fields
    that are still None pre-run) guarded accordingly, as Overview's does.

    meta: optional right-aligned uppercase label (e.g. a session date or
    'FY26 FORECAST · AUG 2026') -- matches the Base44 prototype's page-header
    metadata slot. Omit for pages with nothing meaningful to show there.

    info: optional longer explanation ("what is this dashboard, how do I
    read it") shown as a native hover tooltip next to the title -- NOT the
    same job as `sub`, which is always-visible and stays a short one-liner.
    Rendered via st.subheader(..., help=info) rather than custom HTML
    specifically to get Streamlit's own tooltip bubble (already the same
    mechanism kpi() uses for its `help` param) instead of building a
    second, differently-behaved tooltip system from scratch. anchor=False
    drops the auto-generated deep-link icon st.subheader adds by default --
    not useful here, just visual noise next to a title that's already the
    whole page's identity, not one section among several.
    """
    user = st.session_state.get('user') or {'name': 'User', 'role': 'Analyst'}
    running = bool(st.session_state.get('pipeline_running'))
    export_csv      = st.session_state.get('page_export_csv')
    export_filename = st.session_state.get('page_export_filename')

    with st.container(key='page_header_row'):
        col_main, col_account = st.columns([8, 2], vertical_alignment='top', gap='small')
        with col_main:
            with st.container(key='page_header_block'):
                st.markdown(f'<div class="sf-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
                if info:
                    with st.container(key='page_header_title'):
                        st.subheader(title, anchor=False, help=info)
                else:
                    st.markdown(f'<div class="sf-page-title">{title}</div>', unsafe_allow_html=True)
                if sub:
                    st.markdown(f'<div class="sf-page-sub">{sub}</div>', unsafe_allow_html=True)

        with col_account:
            # The avatar circle is a real st.popover() trigger (Sign out lives
            # inside it, not the sidebar). A raw HTML <div> can't contain an
            # actual Streamlit widget as a child -- opening a div in one
            # st.markdown() call and closing it in a later one, with real
            # widgets rendered in between, is NOT a reliable way to fake that
            # nesting (verified: it silently produced a 0x0 empty div with
            # zero children, most likely because unsafe_allow_html markdown
            # sanitizes/rebalances each call's HTML independently rather than
            # leaving a tag open across calls). A real st.popover() inside a
            # real st.columns() cell is what actually works.
            st.html(f"""<style>
            .st-key-topbar_user_menu button::before {{ content: "{initials(user['name'])}"; }}
            </style>""")
            with st.popover(user['name'], key='topbar_user_menu'):
                # Name + status stacked (not "Signed in as X · Role" on one
                # line), a dark-mode toggle, then Log out -- all plain
                # top-to-bottom Streamlit elements, no st.columns anywhere in
                # here, so the popup stays a vertical list.
                st.markdown(f"""
                <div class="sf-topbar-menu-name">{user['name']}</div>
                <div class="sf-topbar-menu-role">{user['role']}</div>
                """, unsafe_allow_html=True)
                # NOT key='dark_mode' directly -- that was the actual bug
                # reported (dark mode silently reverting after navigating to
                # a different page while the toggle still LOOKED on). Root
                # cause, confirmed by tracing session_state through every
                # step of an actual toggle-then-navigate sequence: our nav
                # buttons call st.rerun() the moment they're clicked, from
                # inside render_sidebar() -- which aborts the script before
                # it ever reaches this widget, later in page_header().
                # Streamlit clears a keyed widget's session_state entry
                # whenever that widget wasn't instantiated in the last
                # completed pass, and an early-aborted pass counts as
                # "wasn't instantiated" for everything after the rerun call.
                # So st.session_state dark_mode was being wiped on literally
                # every single navigation, then silently re-created at False
                # the next time this widget did run -- meanwhile the
                # toggle's own rendered switch was showing a stale "on"
                # straight from the DOM the browser hadn't repainted yet,
                # which is why it visually looked on while nothing was
                # actually dark. Fix: keep the actual setting in a plain,
                # non-widget session_state key (dark_mode) that nothing ever
                # garbage-collects, and give the toggle itself a DIFFERENT
                # key -- feeding it value=... explicitly on every run means
                # it never depends on its own key surviving between reruns
                # at all.
                if 'dark_mode' not in st.session_state:
                    st.session_state.dark_mode = False
                wants_dark = st.toggle('Dark mode', value=st.session_state.dark_mode, key='dark_mode_toggle')
                if wants_dark != st.session_state.dark_mode:
                    st.session_state.dark_mode = wants_dark
                    st.rerun()
                if st.button('Log out', key='topbar_signout_btn', icon=':material/logout:',
                             type='primary', width='stretch', disabled=running):
                    sign_out()

            # Fixed under the username row, right-aligned in the same narrow
            # column -- exports every backing table for whichever page is
            # currently open (see build_page_export_csv() in app.py). Sits
            # outside the popover (a plain click, not a menu item) since it's
            # a one-off download, not an account action.
            if export_csv:
                st.download_button(
                    'Export CSV', data=export_csv, file_name=export_filename or 'export.csv',
                    mime='text/csv', icon=':material/download:', key='topbar_export_btn',
                    width='stretch', disabled=running,
                )

            # Moved below the popover (was above it) per request -- profile
            # at the top of this column, meta at the bottom, not the other
            # way around.
            if meta:
                st.markdown(f'<div class="sf-page-meta sf-page-meta-account">{meta}</div>', unsafe_allow_html=True)


def render_cached_run_banner(loaded_str: str, nav_triggered: bool = False):
    """The "Showing the run from..." peekaboo bar shown while viewing a
    saved run from history rather than a live pipeline result. Call
    once, right after render_sidebar(), only when session_state.
    data_source == 'cached'.

    nav_triggered: whether THIS render was itself caused by a sidebar
    nav click -- the caller reads session_state.nav_loading and passes
    it in as a plain bool rather than this function reading it directly,
    since inject_global_css() (called before this, every render) already
    consumes and clears that same one-shot flag for its own nav-only
    loading overlay; by the time this function ran, session_state.
    nav_loading would already be back to False.

    Deliberately NOT shown persistently, and NOT Python/session_state-
    driven the way the rest of this app's conditional UI normally is --
    per explicit request, this is a transient announcement: hidden above
    the viewport by default, revealed for exactly 3 seconds by its own
    client-side JS, triggered only by (a) the mouse touching the very
    top edge of the window, or (b) a sidebar nav click. No manual
    dismiss control at all this version (the earlier one had a close
    button; removed per request, along with the persistent/sticky
    behaviour it went with).

    Two scripts cooperate, both required every time this function
    renders (the banner's own div is recreated fresh on every full
    rerun, so both need to run again too):
    1. An ALWAYS-emitted, self-guarding one (window.__sfBannerInit)
       that attaches the mousemove listener and defines
       window.__sfBannerShow -- guarded so a full rerun doesn't stack a
       second listener on top of the first, which would cause each
       later render to fire the reveal multiple times per actual mouse
       movement. Safe to call on a page where no banner div exists
       (checks for the element itself, not just the flag) -- it simply
       does nothing on those pages.
    2. A ONE-SHOT script, only emitted when session_state.nav_loading is
       True (the exact same one-shot flag ui.py's nav-only loading
       overlay already reads elsewhere in this file, set right before
       st.rerun() in render_sidebar()'s nav-button handler) -- calls
       window.__sfBannerShow() once, satisfying "triggers automatically
       on every navigation". Placed AFTER the always-on script in the
       page's own script order specifically so window.__sfBannerShow is
       already defined by the time this one-shot call tries to use it.
    """
    st.markdown(
        f'<div class="sf-cached-run-banner">⚡ Run from <b>{loaded_str}</b>, '
        f'Go to <b>Run History</b> to '
        f'pick a different run, or <b>Data Pipeline</b> to run fresh data.</div>',
        unsafe_allow_html=True,
    )
    st.html("""
    <script>
    (function() {
        if (window.__sfBannerInit) return;
        window.__sfBannerInit = true;
        var visible = false, hideTimer = null;
        window.__sfBannerShow = function() {
            var el = document.querySelector('.sf-cached-run-banner');
            if (!el || visible) return;  // no banner on this page, or already showing/mid-animation
            visible = true;
            el.classList.add('sf-banner-visible');
            hideTimer = setTimeout(function() {
                el.classList.remove('sf-banner-visible');
                visible = false;
            }, 3000);
        };
        document.addEventListener('mousemove', function(e) {
            if (e.clientY <= 2) window.__sfBannerShow();
        });

        // Keeps --sf-sidebar-w (the banner's left offset, referenced by
        // the CSS rule in inject_global_css()) matched to the sidebar's
        // REAL current width rather than a hardcoded 300px. Reported
        // live: with the sidebar collapsed a fixed 300px left a blank
        // gap at the left edge (fixed first by a CSS-only :has() rule
        // toggling left:0), but then ALSO reported short/misaligned
        // with the sidebar OPEN -- Streamlit's sidebar has a drag
        // handle on its right edge, so its true rendered width can
        // differ from the 300px default once a user has resized it,
        // which a hardcoded value can never track. Reading the real
        // width here fixes both cases with one mechanism instead of
        // two (this replaces that :has() rule entirely).
        var roTarget = null;
        var ro = new ResizeObserver(updateSidebarWidthVar);
        function updateSidebarWidthVar() {
            var sb = document.querySelector('[data-testid="stSidebar"]');
            if (sb && sb !== roTarget) {
                if (roTarget) ro.unobserve(roTarget);
                ro.observe(sb);
                roTarget = sb;
            }
            var expanded = sb ? sb.getAttribute('aria-expanded') !== 'false' : false;
            var w = (sb && expanded) ? sb.getBoundingClientRect().width : 0;
            document.documentElement.style.setProperty('--sf-sidebar-w', w + 'px');
        }
        updateSidebarWidthVar();
        // subtree + body-level (not tied to one sidebar DOM node) so a
        // rerun that swaps the sidebar element for a new one still gets
        // picked up -- updateSidebarWidthVar() re-queries fresh every
        // call and re-targets the ResizeObserver itself when it does.
        new MutationObserver(updateSidebarWidthVar).observe(document.body, {
            subtree: true, attributes: true, attributeFilter: ['aria-expanded', 'style'],
        });
    })();
    </script>
    """, unsafe_allow_javascript=True)
    if nav_triggered:
        st.html("""
        <script>window.__sfBannerShow && window.__sfBannerShow();</script>
        """, unsafe_allow_javascript=True)


def pill(text, color='green'):
    """Small rounded status badge -- 'Completed'/'Warning'/'Failed'-style
    pills used in Run History, the sidebar's active-run box, and pipeline
    stage rows. color: green|amber|red|blue|gray."""
    return f'<span class="sf-pill sf-pill-{color}">{text}</span>'


_STAGE_STATE = {
    'pending': ('pending', '', MIST),
    'running': ('running', '&#9679;', AMBER),   # filled dot
    'done':    ('done',    '&#10003;', GREEN),  # check
    'warning': ('warning', '!', AMBER),
    'error':   ('error',   '&times;', RED),
}


def stage_row(placeholder, number, name, category, state='pending', detail=''):
    """Renders/updates one row of the Data Pipeline page's numbered stage
    list into `placeholder` (an st.empty()). state: pending|running|done|
    warning|error. Call once per stage per state change -- cheap HTML
    re-render, not a full page rerun.

    The progress bar is a real 0%/100% fill for pending/done/warning/error
    (all known end-states), but a sliding *indeterminate* animation for
    'running' -- not a fabricated percentage. Most stages are atomic
    pass/fail with no genuine partial-progress signal to report, so
    showing e.g. "45%" would be a number we don't actually have."""
    cls, glyph, _ = _STAGE_STATE.get(state, _STAGE_STATE['pending'])
    num_html = glyph if state in ('running', 'done', 'warning', 'error') else str(number)
    placeholder.markdown(f"""
    <div class="sf-stage-row">
        <div class="sf-stage-num {cls}">{num_html}</div>
        <div style="width:190px;flex-shrink:0;min-width:0;">
            <div class="sf-stage-name">{name}</div>
            <div class="sf-stage-cat">{category}</div>
        </div>
        <div class="sf-stage-bar-track"><div class="sf-stage-bar-fill {cls}"></div></div>
        <div class="sf-stage-detail">{detail}</div>
    </div>
    """, unsafe_allow_html=True)


def overall_progress(placeholder, done: int, total: int):
    """Renders the Data Pipeline page's 'OVERALL PROGRESS' bar -- a real
    fraction (stages actually completed / total stages), never estimated."""
    pct = int(round(100 * done / total)) if total else 0
    placeholder.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
        <span class="sf-eyebrow" style="margin-bottom:0;">Overall progress</span>
        <span style="font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:600;color:{TEXT};">{pct}%</span>
    </div>
    <div class="sf-overall-track"><div class="sf-overall-fill" style="width:{pct}%;"></div></div>
    """, unsafe_allow_html=True)


_LOG_ICON_SUB = {
    ':material/check_circle:': '&#10003;',
    ':material/warning:': '&#9888;',
    ':material/error:': '&times;',
}


def new_execution_log(placeholder):
    """Creates a fresh, persistent Execution Log panel bound to
    `placeholder` (an st.empty()) and returns a log(msg) callable. Every
    call appends one real HH:MM:SS-timestamped line and re-renders the
    full scrolling panel -- unlike the old approach of writing lines
    inside st.status()'s own collapsible container (which disappears once
    the status collapses), this persists as its own panel on the page,
    matching the reference app's separate 'Execution Log' card."""
    lines = []

    def log(msg: str):
        for token, repl in _LOG_ICON_SUB.items():
            msg = msg.replace(token, repl)
        ts = pd.Timestamp.now().strftime('%H:%M:%S')
        lines.append((ts, msg))
        rows = ''.join(
            f'<div class="sf-exec-log-line"><span class="sf-exec-log-time">{t}</span>{m}</div>'
            for t, m in lines
        )
        placeholder.markdown(f'<div class="sf-exec-log">{rows}</div>', unsafe_allow_html=True)

    return log


def kpi(label, value, delta=None, icon=None, accent=TEAL, delta_color='normal', help=None):
    """Flat KPI tile per spec: label (eyebrow) -> value (uniform navy,
    handled globally) -> delta (colour-coded green/red by st.metric's own
    delta_color logic) -> optional subtext. `accent`/`icon` are accepted
    for call-site compatibility but no longer render distinctly -- every
    KPI value is uniform navy and icon-less per the real spec/reference,
    not colour-coded per tile (see inject_global_css)."""
    key = f'kpi_{_slug(label)}'
    with st.container(key=key):
        st.metric(label, value, delta=delta, icon=icon, border=True,
                  delta_color=delta_color, help=help)


@contextmanager
def card(title=None, sub=None, tag=None, tag_color='green', key=None, info=None):
    """info: optional "how to read this chart/table" explanation, shown as
    a native hover tooltip right next to the TITLE (not `sub`, which stays
    a plain, always-visible caption underneath) -- feedback on an earlier
    version of this that put the icon next to `sub` instead was that it
    read as detached from the header it was actually explaining. Same
    mechanism page_header()'s `info` and kpi()'s `help` both use:
    st.subheader(title, help=info) instead of the plain .sf-card-title
    div, restyled to look identical to it (14px, same as .sf-card-title,
    vs. page_header()'s own 24px). Wrapped in st.container(key=...) the
    same way page_header()'s title is, for the same reason (an unkeyed
    st.container still carries overflow="visible" and would otherwise
    pick up the global bordered-card look) -- but card() can be called
    many times on one page, where page_header() is called once, so this
    key is generated per-title (card_title_<slugified title>) rather than
    a fixed constant, and the matching CSS selector uses [class*=...] to
    match any of them by shared prefix rather than one exact key (see the
    identical [class*="st-key-kpi_"] pattern already used elsewhere in
    this file). Titles are unique per page in every call site today; a
    genuine duplicate would raise Streamlit's own clear
    StreamlitDuplicateElementKey error rather than fail silently."""
    with st.container(key=key, border=True):
        if title:
            c1, c2 = st.columns([5, 1.4], vertical_alignment='top')
            with c1:
                if info:
                    with st.container(key=f'card_title_{_slug(title)}'):
                        st.subheader(title, anchor=False, help=info)
                else:
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
    """Swiss-financial chart chrome: no axis lines, only horizontal
    gridlines (slate-100), muted slate-400 tick labels -- matches the
    spec's Recharts rules (axisLine/tickLine hidden, vertical={false})
    translated to Plotly's equivalent axis properties."""
    fig.update_layout(
        height=h, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, system-ui, sans-serif', size=11, color=MIST),
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(orientation='h', y=1.14, font=dict(size=11)),
        xaxis=dict(showgrid=False, showline=False, zeroline=False, tickfont=dict(color=MIST, size=11)),
        yaxis=dict(showgrid=True, gridcolor=GRIDLINE, showline=False, zeroline=False, tickfont=dict(color=MIST, size=11)),
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
    message instead of failing deep inside the pipeline. Returns (file, valid).

    Header row (icon + title/desc + a status pill on the right) mirrors the
    Base44 prototype's compact single-line source-file rows; the pill is a
    placeholder filled in AFTER validation runs below, so it always reflects
    the real outcome (Pending/Uploaded/Error), never a fake state."""
    valid = True
    with st.container(border=True):
        head_col, pill_col = st.columns([5, 1.6], vertical_alignment='center')
        with head_col:
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
        with pill_col:
            # Nothing shown here before a file is picked -- the native
            # file_uploader below already has its own "Upload"/drag-drop
            # prompt, so a second "Upload" affordance in the header was a
            # duplicate of the one real action. This slot only fills in
            # once there's something to actually report (Uploaded/Error).
            pill_slot = st.empty()

        f = st.file_uploader('Upload', type='csv', key=key, label_visibility='collapsed', disabled=disabled)
        error_msg = None
        if f:
            columns = _peek_columns(f) if required_columns else []
            missing = [] if columns is None else [c for c in required_columns or [] if c not in columns]

            if columns is None:
                valid = False
                error_msg = f"{f.name} couldn't be read as a CSV. Check it isn't corrupted or a different file type."
            elif missing:
                valid = False
                error_msg = f"Wrong file for this slot? {f.name} is missing: {', '.join(missing)}"
            else:
                pill_slot.markdown(
                    f'<div style="text-align:right;">{pill(fmt_size(f.size), "green")}</div>',
                    unsafe_allow_html=True)

            if error_msg:
                pill_slot.markdown(
                    f'<div style="text-align:right;">{pill("Error", "red")}</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="sf-upload-error">
                    <svg viewBox="0 0 24 24" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
                    </svg>
                    <div>{error_msg}</div>
                </div>
                """, unsafe_allow_html=True)
    return f, valid
