"""Generates/loads the key that signs the SSO session cookie (app/main.py's
SessionMiddleware). Deliberately a plain file next to the sqlite DB, not an
AppSetting row like every other setting in this app -- SessionMiddleware has to
be registered synchronously at import time, before the app (and its async DB
engine) has started serving anything, so this can't go through the normal
async get_setting/set_setting path. Persists across restarts the same way the
DB itself does, since both live under the same data_dir volume.
"""

import secrets
from pathlib import Path

_FILENAME = ".session_secret_key"


def get_or_create_session_secret_key(data_dir: Path) -> str:
    path = data_dir / _FILENAME
    if path.exists():
        return path.read_text().strip()
    data_dir.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(32)
    path.write_text(key)
    return key
