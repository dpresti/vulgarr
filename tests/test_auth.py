import base64

from app.auth import _credentials_valid, _is_authenticated
from app.db.session import hash_password, verify_password


def test_password_round_trips():
    stored = hash_password("hunter2")
    assert verify_password("hunter2", stored) is True


def test_wrong_password_fails():
    stored = hash_password("hunter2")
    assert verify_password("wrong", stored) is False


def test_blank_stored_hash_never_matches():
    assert verify_password("anything", "") is False


def _basic_header(user: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


def test_credentials_valid_with_correct_basic_auth():
    stored = hash_password("hunter2")
    assert _credentials_valid(_basic_header("admin", "hunter2"), "admin", stored) is True


def test_credentials_invalid_with_wrong_password():
    stored = hash_password("hunter2")
    assert _credentials_valid(_basic_header("admin", "nope"), "admin", stored) is False


def test_credentials_invalid_with_wrong_username():
    stored = hash_password("hunter2")
    assert _credentials_valid(_basic_header("someone-else", "hunter2"), "admin", stored) is False


def test_credentials_invalid_with_missing_header():
    stored = hash_password("hunter2")
    assert _credentials_valid(None, "admin", stored) is False


def test_credentials_invalid_with_malformed_header():
    stored = hash_password("hunter2")
    assert _credentials_valid("Bearer sometoken", "admin", stored) is False


def test_mode_none_always_authenticated():
    assert _is_authenticated("none", basic_ok=False, has_session_user=False) is True


def test_mode_password_requires_basic_auth():
    assert _is_authenticated("password", basic_ok=True, has_session_user=False) is True
    assert _is_authenticated("password", basic_ok=False, has_session_user=False) is False


def test_mode_password_ignores_session_user():
    # A stray session cookie (e.g. left over from a prior "sso" mode) must never
    # substitute for real Basic Auth credentials in "password" mode.
    assert _is_authenticated("password", basic_ok=False, has_session_user=True) is False


def test_mode_sso_authenticated_via_session():
    assert _is_authenticated("sso", basic_ok=False, has_session_user=True) is True


def test_mode_sso_break_glass_via_basic_auth():
    # The whole point of the break-glass fallback: Basic Auth alone (no SSO
    # session at all) still gets through in "sso" mode.
    assert _is_authenticated("sso", basic_ok=True, has_session_user=False) is True


def test_mode_sso_denied_without_either():
    assert _is_authenticated("sso", basic_ok=False, has_session_user=False) is False
