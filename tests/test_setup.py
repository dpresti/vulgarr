from app.routers.setup import _filter_importable_settings


def test_filters_out_unrecognized_keys():
    result = _filter_importable_settings({"sonarr_url": "http://sonarr:8989", "not_a_real_setting": "x"})
    assert result == {"sonarr_url": "http://sonarr:8989"}


def test_keeps_all_recognized_keys():
    data = {"sonarr_url": "http://sonarr:8989", "auth_mode": "sso", "backups_enabled": True}
    assert _filter_importable_settings(data) == data


def test_empty_dict_in_empty_dict_out():
    assert _filter_importable_settings({}) == {}


def test_non_dict_input_returns_empty():
    assert _filter_importable_settings(["not", "a", "dict"]) == {}
    assert _filter_importable_settings("also not a dict") == {}
    assert _filter_importable_settings(None) == {}


def test_preserves_value_types():
    data = {"concurrency_cap": 4, "off_hours_enabled": True, "trigger_priority": ["sonarr_radarr", "bazarr"]}
    assert _filter_importable_settings(data) == data
