import datetime
import os
import time

from app.domain import is_mkv_path
from app.queue.worker import _delete_backups_older_than, is_within_off_hours_window


def t(s):
    h, m = s.split(":")
    return datetime.time(int(h), int(m))


def test_simple_window_within_same_day():
    assert is_within_off_hours_window(t("02:00"), t("01:00"), t("07:00")) is True
    assert is_within_off_hours_window(t("08:00"), t("01:00"), t("07:00")) is False


def test_window_wrapping_past_midnight():
    # e.g. 22:00 -> 06:00
    assert is_within_off_hours_window(t("23:30"), t("22:00"), t("06:00")) is True
    assert is_within_off_hours_window(t("03:00"), t("22:00"), t("06:00")) is True
    assert is_within_off_hours_window(t("12:00"), t("22:00"), t("06:00")) is False


def test_is_mkv_path():
    assert is_mkv_path("/plex/Movies/Foo (2020)/Foo.mkv") is True
    assert is_mkv_path("/plex/Movies/Foo (2020)/Foo.MKV") is True
    assert is_mkv_path("/plex/Movies/Foo (2020)/Foo.mp4") is False
    assert is_mkv_path("/plex/Movies/Foo (2020)/Foo.mkv.mp4") is False


def _make_backup(root, rel_path: str, age_days: float):
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("orig")
    old_time = time.time() - age_days * 86400
    os.utime(path, (old_time, old_time))
    return path


def test_prune_deletes_only_backups_past_retention(tmp_path):
    old_file = _make_backup(tmp_path, "Movies/Foo.20200101T000000.mkv.orig", age_days=40)
    new_file = _make_backup(tmp_path, "Movies/Bar.20260101T000000.mkv.orig", age_days=5)

    deleted = _delete_backups_older_than(tmp_path, retention_days=30)

    assert deleted == 1
    assert not old_file.exists()
    assert new_file.exists()


def test_prune_removes_emptied_parent_directories(tmp_path):
    old_file = _make_backup(tmp_path, "Movies/Foo (2020)/Foo.20200101T000000.mkv.orig", age_days=40)

    _delete_backups_older_than(tmp_path, retention_days=30)

    assert not old_file.exists()
    assert not old_file.parent.exists()


def test_prune_missing_backup_root_is_a_noop(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert _delete_backups_older_than(missing, retention_days=30) == 0
