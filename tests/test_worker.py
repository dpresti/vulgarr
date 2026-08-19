import datetime

from app.domain import is_mkv_path
from app.queue.worker import is_within_off_hours_window


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
