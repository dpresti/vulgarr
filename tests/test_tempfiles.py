import asyncio
import os
import time

import pytest

from app.mux.tempfiles import STALE_AFTER_SECONDS, remove_orphaned_temp_files, sweep_interrupted_temp_files


def _touch(path, age_seconds=0):
    path.write_bytes(b"x")
    if age_seconds:
        old = time.time() - age_seconds
        os.utime(path, (old, old))
    return path


def test_removes_stale_remux_and_blur_temps_for_this_title(tmp_path):
    video = _touch(tmp_path / "Movie (2025) WEBDL-1080p.mkv")
    remux_tmp = _touch(tmp_path / ".Movie (2025) WEBDL-1080p.spf-tmp.9f396fd5.mkv", STALE_AFTER_SECONDS + 60)
    blur_tmp = _touch(tmp_path / ".Movie (2025) WEBDL-1080p.spf-blur-tmp.42.mkv", STALE_AFTER_SECONDS + 60)

    removed = remove_orphaned_temp_files(video)

    assert sorted(removed) == sorted([remux_tmp, blur_tmp])
    assert video.exists()
    assert not remux_tmp.exists() and not blur_tmp.exists()


def test_keeps_recent_temp_that_a_live_run_may_still_be_writing(tmp_path):
    video = _touch(tmp_path / "Show - S01E01.mkv")
    live_tmp = _touch(tmp_path / ".Show - S01E01.spf-tmp.abcd1234.mkv")

    assert remove_orphaned_temp_files(video) == []
    assert live_tmp.exists()


def test_min_age_zero_removes_recent_temp(tmp_path):
    video = _touch(tmp_path / "Show - S01E01.mkv")
    tmp = _touch(tmp_path / ".Show - S01E01.spf-tmp.abcd1234.mkv")

    assert remove_orphaned_temp_files(video, min_age_seconds=0) == [tmp]


def test_never_touches_other_titles_edits_or_unrelated_files(tmp_path):
    video = _touch(tmp_path / "Show - S01E01.mkv")
    keep = [
        _touch(tmp_path / "Show - S01E01 - Vulgarr Edit.mkv", STALE_AFTER_SECONDS + 60),
        _touch(tmp_path / ".Show - S01E02.spf-tmp.abcd1234.mkv", STALE_AFTER_SECONDS + 60),
        _touch(tmp_path / ".Show - S01E01 extra.spf-tmp.abcd1234.mkv", STALE_AFTER_SECONDS + 60),
        _touch(tmp_path / ".Show - S01E01.spf-tmp.abcd1234.mp4", STALE_AFTER_SECONDS + 60),
        _touch(tmp_path / "Show - S01E01.en.srt", STALE_AFTER_SECONDS + 60),
    ]

    assert remove_orphaned_temp_files(video, min_age_seconds=0) == []
    assert all(p.exists() for p in keep)


def test_missing_directory_is_not_an_error(tmp_path):
    assert remove_orphaned_temp_files(tmp_path / "gone" / "Movie.mkv", min_age_seconds=0) == []


def test_startup_sweep_removes_regardless_of_age(tmp_path):
    video = _touch(tmp_path / "Movie.mkv")
    tmp = _touch(tmp_path / ".Movie.spf-tmp.abcd1234.mkv")

    asyncio.run(sweep_interrupted_temp_files([video, tmp_path / "unmounted" / "Other.mkv"]))

    assert not tmp.exists()
    assert video.exists()


def test_cancel_during_verify_removes_remux_temp(tmp_path, monkeypatch):
    from app.mux import remux

    video = _touch(tmp_path / "Movie.mkv")
    tmp = tmp_path / ".Movie.spf-tmp.abcd1234.mkv"

    async def fake_probe(_bin, _path):
        return {"format": {"duration": "60"}, "streams": []}

    class _Plan:
        new_track_output_indices = []
        existing_clean_indices = []

    async def fake_build(**_kwargs):
        return _touch(tmp)

    async def cancelled_verify(**_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(remux, "probe", fake_probe)
    monkeypatch.setattr(remux, "plan_audio_streams", lambda *a, **k: _Plan())
    monkeypatch.setattr(remux, "build_clean_track", fake_build)
    monkeypatch.setattr(remux, "verify_output", cancelled_verify)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            remux.remux_with_clean_track(
                video_path=video, matches=[], intervals_by_level=[], media_root=tmp_path, backup_root=tmp_path,
                ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe", clean_track_title="Clean", clean_track_language="eng",
            )
        )
    assert not tmp.exists()
    assert video.exists()
