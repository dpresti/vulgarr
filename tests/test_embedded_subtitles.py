from pathlib import Path

from app.integrations import embedded_subtitles, subtitle_lookup
from app.integrations.embedded_subtitles import select_candidate_streams


def _sub(index, codec="subrip", lang="eng", title=None, forced=0):
    tags = {"language": lang}
    if title:
        tags["title"] = title
    return {"index": index, "codec_type": "subtitle", "codec_name": codec, "tags": tags, "disposition": {"forced": forced}}


def test_picks_english_text_tracks_only():
    streams = [
        {"index": 0, "codec_type": "video", "codec_name": "h264"},
        _sub(2),
        _sub(3, lang="ger"),
        _sub(4, codec="ass"),
        _sub(5, codec="hdmv_pgs_subtitle"),
    ]
    probe = select_candidate_streams(streams, "en")
    assert probe.text_candidates == [2, 4]
    assert not probe.has_image_only_match


def test_skips_forced_by_disposition_or_title():
    streams = [_sub(2, forced=1), _sub(3, title="English (Forced)"), _sub(4, title="English SDH")]
    assert select_candidate_streams(streams, "en").text_candidates == [4]


def test_reports_image_only():
    streams = [_sub(2, codec="hdmv_pgs_subtitle"), _sub(3, codec="dvd_subtitle")]
    probe = select_candidate_streams(streams, "en")
    assert probe.text_candidates == []
    assert probe.has_image_only_match


def test_lookup_prefers_full_sidecar_then_embedded_then_forced(tmp_path, monkeypatch):
    monkeypatch.setattr(embedded_subtitles.app_settings, "data_dir", tmp_path / "data")
    media = tmp_path / "media"
    media.mkdir()
    video = media / "Movie (2024).mkv"
    video.write_bytes(b"x")
    forced = media / "Movie (2024).en.forced.srt"
    forced.write_text("1\n00:00:01,000 --> 00:00:02,000\nHola\n")
    assert subtitle_lookup.find_subtitle_for_video(video) == forced

    cached = embedded_subtitles.cached_subtitle_path(video, "en")
    cached.parent.mkdir(parents=True)
    cached.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    assert subtitle_lookup.find_subtitle_for_video(video) == cached

    full = media / "Movie (2024).en.srt"
    full.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    assert subtitle_lookup.find_subtitle_for_video(video) == full


def test_resolve_tries_extraction_when_only_forced(tmp_path, monkeypatch):
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    forced = tmp_path / "Movie.en.forced.srt"
    forced.write_text("")
    extracted = tmp_path / "extracted.srt"
    monkeypatch.setattr(embedded_subtitles.app_settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(subtitle_lookup, "extract_embedded_subtitle", lambda v, lang: extracted)
    assert subtitle_lookup.resolve_subtitle_for_video(video) == extracted

    monkeypatch.setattr(subtitle_lookup, "extract_embedded_subtitle", lambda v, lang: None)
    assert subtitle_lookup.resolve_subtitle_for_video(video) == forced


def test_cache_key_changes_when_file_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(embedded_subtitles.app_settings, "data_dir", tmp_path / "data")
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    first = embedded_subtitles.cached_subtitle_path(video, "en")
    video.write_bytes(b"xyz")
    assert embedded_subtitles.cached_subtitle_path(video, "en") != first
