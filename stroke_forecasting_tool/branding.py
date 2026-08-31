"""Shared logo asset helper.

Deliberately a standalone leaf module -- no imports from ui.py or auth.py
-- so both can use it without risking the circular-import trap those two
already have to avoid between each other (ui.py imports from auth.py;
auth.py must never import from ui.py as a result).
"""
import base64
from functools import lru_cache
from pathlib import Path

LOGO_PATH = Path(__file__).parent / 'assets' / 'logo.png'
# Square, icon-only crop (no wordmark) for the browser tab favicon --
# the full logo.png is a wide rectangle with text, which browsers squash
# illegibly into a tiny square tab icon.
TITLE_LOGO_PATH = Path(__file__).parent / 'assets' / 'logotitle.png'
# White-wordmark variant -- logo.png's "Stroke" text is navy, which
# disappears against the sidebar's navy background; this one is for
# there specifically (the green brain mark and "FOUNDATION" text are
# unchanged between the two, since green already reads fine on navy).
LOGO_WHITE_PATH = Path(__file__).parent / 'assets' / 'logo-white-stroke.png'


@lru_cache(maxsize=1)
def logo_data_uri() -> str | None:
    """Base64 data: URI for the real Stroke Foundation logo, for embedding
    in raw HTML <img> tags -- Streamlit's markdown HTML can't reference a
    local file path directly, so this is the only reliable way to inline
    it. None if the asset is missing, so callers can fall back to the old
    placeholder mark instead of showing a broken image. Cached since the
    same bytes get re-encoded on every rerun otherwise."""
    if not LOGO_PATH.exists():
        return None
    data = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return f'data:image/png;base64,{data}'


@lru_cache(maxsize=1)
def logo_white_data_uri() -> str | None:
    """Same as logo_data_uri() but the white-wordmark variant -- see
    LOGO_WHITE_PATH above. None if the asset is missing."""
    if not LOGO_WHITE_PATH.exists():
        return None
    data = base64.b64encode(LOGO_WHITE_PATH.read_bytes()).decode()
    return f'data:image/png;base64,{data}'
