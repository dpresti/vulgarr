import base64

from app.auth import _credentials_valid
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
