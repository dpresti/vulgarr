from app.audio.mute import MuteInterval
from app.mux.scene_blur import (
    DEFAULT_BLUR_LEVEL,
    _SourceHevcParams,
    _annexb_buf_has_rasl,
    _annexb_buf_last_picture_nal_type,
    _blur_job_fingerprint,
    _build_matching_x265_params,
    _filter_benign_decode_warnings,
    _parse_hevc_trace_fields,
    _parse_keyframe_csv,
    _reencode_video_codec,
    _reset_stale_work_dir,
    blur_level_to_radius_power,
    build_blur_filter,
    plan_audio_segments,
    plan_video_segments,
    radius_power_to_blur_level,
    sibling_edit_path,
)
from pathlib import Path


def test_reencode_codec_matches_hevc_source():
    # Real bug: re-encode segments used to hardcode libx264 regardless of
    # source codec, producing garbage when spliced against an HEVC copy
    # segment (confirmed directly against a real file this session).
    assert _reencode_video_codec("hevc") == "libx265"
    assert _reencode_video_codec("h265") == "libx265"


def test_reencode_codec_matches_other_known_sources():
    # Same splice-corruption bug applies to any non-H.264 source, not just
    # HEVC -- extended to cover every other codec this ffmpeg build has an
    # encoder for.
    assert _reencode_video_codec("vp9") == "libvpx-vp9"
    assert _reencode_video_codec("av1") == "libsvtav1"
    assert _reencode_video_codec("mpeg2video") == "mpeg2video"
    assert _reencode_video_codec("mpeg4") == "mpeg4"


def test_reencode_codec_defaults_to_h264_for_everything_else():
    assert _reencode_video_codec("h264") == "libx264"
    assert _reencode_video_codec("") == "libx264"
    assert _reencode_video_codec("prores") == "libx264"


def _nal_bytes(nal_type: int, extra: bytes = b"\x00\x00\x00") -> bytes:
    """One Annex-B NAL unit: 3-byte start code, then a header byte encoding
    nal_type in bits 1-6 ((byte>>1)&0x3F recovers it), then arbitrary filler."""
    return b"\x00\x00\x01" + bytes([(nal_type << 1) & 0xFF]) + extra


def test_buf_has_rasl_detects_rasl_n():
    assert _annexb_buf_has_rasl(_nal_bytes(8)) is True


def test_buf_has_rasl_detects_rasl_r():
    assert _annexb_buf_has_rasl(_nal_bytes(9)) is True


def test_buf_has_rasl_false_for_cra_with_no_leading_pictures():
    # Real, common case this session found: a CRA with ordinary TRAIL
    # pictures right after it, no RASL at all.
    data = _nal_bytes(21) + _nal_bytes(1) + _nal_bytes(1) + _nal_bytes(0)
    assert _annexb_buf_has_rasl(data) is False


def test_buf_has_rasl_finds_rasl_after_cra_and_parameter_sets():
    # Real observed shape: AUD, VPS, SPS, PPS, SEI..., CRA, then RASL_R.
    data = (
        _nal_bytes(35) + _nal_bytes(32) + _nal_bytes(33) + _nal_bytes(34)
        + _nal_bytes(39) + _nal_bytes(21) + _nal_bytes(9)
    )
    assert _annexb_buf_has_rasl(data) is True


def test_buf_has_rasl_empty_buffer():
    assert _annexb_buf_has_rasl(b"") is False


def test_last_picture_nal_type_finds_trailing_cra():
    # Real observed shape at a copy segment's true end (28 Years Later, live
    # repro this session): parameter sets, then a closing CRA with no RASL
    # after it -- _annexb_buf_has_rasl correctly says "no RASL here", but the
    # CRA itself is still an unsafe splice boundary (confirmed directly: both
    # adjacent segments decode perfectly in isolation, only the spliced
    # result corrupts). This is what catches that case instead.
    data = _nal_bytes(32) + _nal_bytes(33) + _nal_bytes(34) + _nal_bytes(21)
    assert _annexb_buf_last_picture_nal_type(data) == 21


def test_last_picture_nal_type_ignores_trailing_non_picture_nals():
    # SEI/filler after the last real picture NAL shouldn't count -- the
    # picture type itself (here a plain TRAIL_R) is what determines safety.
    data = _nal_bytes(1) + _nal_bytes(39)
    assert _annexb_buf_last_picture_nal_type(data) == 1


def test_last_picture_nal_type_picks_the_last_one_not_the_first():
    data = _nal_bytes(19) + _nal_bytes(1) + _nal_bytes(21)
    assert _annexb_buf_last_picture_nal_type(data) == 21


def test_last_picture_nal_type_none_for_no_picture_nals():
    data = _nal_bytes(32) + _nal_bytes(33) + _nal_bytes(34)
    assert _annexb_buf_last_picture_nal_type(data) is None


def test_last_picture_nal_type_empty_buffer():
    assert _annexb_buf_last_picture_nal_type(b"") is None


# Trimmed, real trace_headers output captured live this session against a
# real 4K HEVC file (28 Years Later) -- exact field-name strings/format this
# session confirmed ffmpeg actually prints, not a guess at the schema. The
# vps_max_dec_pic_buffering_minus1 line uses a deliberately different value
# than the sps_ line to prove the parser keys on the sps_-prefixed field, not
# the vps_-prefixed duplicate.
_TRACE_HEADERS_FIXTURE_WITH_DEBLOCK = """
[trace_headers @ 0x1] 50          general_tier_flag                                           1 = 1
[trace_headers @ 0x1] 136         general_level_idc                                    01111011 = 123
[trace_headers @ 0x1] 145         vps_max_dec_pic_buffering_minus1[0]                     01001 = 9
[trace_headers @ 0x1] 177         sps_max_dec_pic_buffering_minus1[0]                     00111 = 5
[trace_headers @ 0x1] 41          entropy_coding_sync_enabled_flag                            1 = 1
[trace_headers @ 0x1] 43          deblocking_filter_control_present_flag                      1 = 1
[trace_headers @ 0x1] 46          pps_beta_offset_div2                                    00111 = -3
[trace_headers @ 0x1] 51          pps_tc_offset_div2                                      00111 = -3
"""


def test_parse_hevc_trace_fields_with_deblocking():
    params = _parse_hevc_trace_fields(_TRACE_HEADERS_FIXTURE_WITH_DEBLOCK)
    assert params == _SourceHevcParams(
        high_tier=True, level_x265="4.1", dpb_minus1=5, wpp=True, deblock=(-3, -3)
    )


def test_parse_hevc_trace_fields_without_deblocking():
    fixture = _TRACE_HEADERS_FIXTURE_WITH_DEBLOCK.replace(
        "deblocking_filter_control_present_flag                      1 = 1",
        "deblocking_filter_control_present_flag                      0 = 0",
    )
    params = _parse_hevc_trace_fields(fixture)
    assert params.deblock is None


def test_parse_hevc_trace_fields_wpp_off():
    fixture = _TRACE_HEADERS_FIXTURE_WITH_DEBLOCK.replace(
        "entropy_coding_sync_enabled_flag                            1 = 1",
        "entropy_coding_sync_enabled_flag                            0 = 0",
    )
    params = _parse_hevc_trace_fields(fixture)
    assert params.wpp is False


def test_parse_hevc_trace_fields_missing_fields_returns_none():
    # A non-HEVC source, or an unexpected ffmpeg output format -- caller
    # falls back to an unmatched re-encode rather than guessing.
    assert _parse_hevc_trace_fields("") is None
    assert _parse_hevc_trace_fields("not trace_headers output at all") is None


def test_build_matching_x265_params_with_wpp():
    p = _SourceHevcParams(high_tier=True, level_x265="4.1", dpb_minus1=5, wpp=True, deblock=(-3, -3))
    result = _build_matching_x265_params(p)
    assert result == "open-gop=0:high-tier=1:level-idc=4.1:ref=5:pools=4:wpp=1:deblock=-3,-3"


def test_build_matching_x265_params_without_wpp_or_deblock():
    p = _SourceHevcParams(high_tier=False, level_x265="4.0", dpb_minus1=4, wpp=False, deblock=None)
    result = _build_matching_x265_params(p)
    assert result == "open-gop=0:high-tier=0:level-idc=4.0:ref=4:wpp=0"
    assert "pools" not in result


def test_filter_benign_decode_warnings_strips_known_benign_line():
    # Real message captured live this session, twice, on two different
    # re-encodes of the same scene -- a confirmed false positive from
    # ffmpeg's own null-muxer + accurate-seek interaction, not real
    # corruption (see _BENIGN_DECODE_WARNING_RE's own comment).
    err = (
        "[null @ 0x5dbd82a54280] Application provided invalid, non monotonically increasing dts to muxer in stream 0: 251 >= 248\n"
        "[null @ 0x5dbd82a54280] Application provided invalid, non monotonically increasing dts to muxer in stream 0: 251 >= 249"
    )
    assert _filter_benign_decode_warnings(err) == ""


def test_filter_benign_decode_warnings_keeps_real_errors():
    # Real corruption this module has hit before -- must never be filtered.
    err = "[hevc @ 0x1] Could not find ref with POC 4\n[hevc @ 0x1] alignment_bit_equal_to_one=0"
    assert _filter_benign_decode_warnings(err) == err


def test_filter_benign_decode_warnings_mixed_keeps_only_real_error():
    err = (
        "[null @ 0x1] Application provided invalid, non monotonically increasing dts to muxer in stream 0: 5 >= 4\n"
        "[hevc @ 0x1] Could not find ref with POC 4"
    )
    assert _filter_benign_decode_warnings(err) == "[hevc @ 0x1] Could not find ref with POC 4"


def test_filter_benign_decode_warnings_empty_input():
    assert _filter_benign_decode_warnings("") == ""


def test_empty_intervals_is_passthrough():
    expr = build_blur_filter([], "0:v", "blurred")
    assert expr == "[0:v]null[blurred]"


def test_builds_between_expression():
    intervals = [MuteInterval(start=1.0, end=2.0)]
    expr = build_blur_filter(intervals, "0:v", "blurred", radius=25, power=3)
    assert expr == "[0:v]boxblur=luma_radius=25:luma_power=3:chroma_radius=25:chroma_power=3:enable='between(t,1.000,2.000)'[blurred]"


def test_default_radius_and_power_are_90_and_8():
    # Tested via direct visual comparison (raw frame vs. increasing radius/power)
    # to be the point where nothing recognizable remains -- locked in here so an
    # unrelated refactor can't silently soften the default.
    intervals = [MuteInterval(start=1.0, end=2.0)]
    expr = build_blur_filter(intervals, "0:v", "blurred")
    assert "luma_radius=90:luma_power=8:chroma_radius=90:chroma_power=8" in expr


def test_chains_stages_for_many_intervals():
    # Same batching-limit reasoning as build_volume_filter's own test -- ffmpeg's
    # expression parser breaks past ~80-90 between() terms in one enable=
    # expression, batched here in groups of 20 (_MAX_TERMS_PER_STAGE).
    intervals = [MuteInterval(start=float(i * 10), end=float(i * 10 + 1)) for i in range(45)]
    expr = build_blur_filter(intervals, "0:v", "blurred")
    stages = expr.split(";")
    assert len(stages) == 3  # 45 intervals / 20 per stage -> 3 stages
    for stage in stages:
        assert stage.count("between(") <= 20

    assert stages[0].startswith("[0:v]boxblur=")
    assert stages[-1].endswith("[blurred]")
    assert "[blurred_stage0]" in stages[0] and stages[1].startswith("[blurred_stage0]")
    assert "[blurred_stage1]" in stages[1] and stages[2].startswith("[blurred_stage1]")


def test_sibling_edit_path_naming():
    # Plex "Versions" naming convention (flat, same folder) -- confirmed during
    # planning as what actually groups a second file under one library item.
    result = sibling_edit_path(Path("/plex/Movies/Deadpool (2016)/Deadpool (2016) Bluray-1080p.mkv"))
    assert result == Path("/plex/Movies/Deadpool (2016)/Deadpool (2016) Bluray-1080p - Vulgarr Edit.mkv")


def test_sibling_edit_path_preserves_suffix():
    result = sibling_edit_path(Path("/plex/Movies/Foo/Foo (2020) WEBRip.mp4"))
    assert result.suffix == ".mp4"
    assert result.name == "Foo (2020) WEBRip - Vulgarr Edit.mp4"


def test_default_blur_level_maps_to_90_8():
    assert blur_level_to_radius_power(DEFAULT_BLUR_LEVEL) == (90, 8)


def test_blur_level_mapping_is_monotonically_increasing():
    # Each level should be at least as strong as the one before it, so the
    # slider actually reads left-to-right as "lighter to heavier."
    pairs = [blur_level_to_radius_power(level) for level in range(1, 6)]
    for (r1, p1), (r2, p2) in zip(pairs, pairs[1:]):
        assert r2 >= r1 and p2 >= p1


def test_out_of_range_blur_level_falls_back_to_default():
    assert blur_level_to_radius_power(99) == blur_level_to_radius_power(DEFAULT_BLUR_LEVEL)
    assert blur_level_to_radius_power(0) == blur_level_to_radius_power(DEFAULT_BLUR_LEVEL)


def test_radius_power_round_trips_through_level():
    for level in range(1, 6):
        radius, power = blur_level_to_radius_power(level)
        assert radius_power_to_blur_level(radius, power) == level


def test_unrecognized_radius_power_maps_to_nearest_level():
    # Something close to level 4 (90/8) but not an exact preset match --
    # e.g. a value set some other way before this table existed.
    assert radius_power_to_blur_level(95, 8) == 4


def test_no_blur_intervals_is_one_copy_segment():
    segments = plan_video_segments([], keyframe_timestamps=[0.0, 10.0, 20.0], total_duration=30.0)
    assert len(segments) == 1
    assert segments[0].start == 0.0 and segments[0].end == 30.0
    assert segments[0].reencode is False


def test_single_scene_expands_to_nearest_keyframes():
    scenes = [MuteInterval(start=10.0, end=15.0)]
    segments = plan_video_segments(scenes, keyframe_timestamps=[0.0, 8.0, 20.0, 30.0], total_duration=40.0)
    assert [(s.start, s.end, s.reencode) for s in segments] == [
        (0.0, 8.0, False),
        (8.0, 20.0, True),
        (20.0, 40.0, False),
    ]
    # Local time is offset by the re-encode segment's own start (8.0).
    assert segments[1].local_blur_intervals == (MuteInterval(start=2.0, end=7.0),)


def test_scene_with_no_keyframe_before_falls_back_to_segment_start_zero():
    scenes = [MuteInterval(start=2.0, end=4.0)]
    segments = plan_video_segments(scenes, keyframe_timestamps=[10.0, 20.0], total_duration=30.0)
    assert segments[0].start == 0.0
    assert segments[0].reencode is True


def test_scene_with_no_keyframe_after_extends_to_total_duration():
    scenes = [MuteInterval(start=25.0, end=28.0)]
    segments = plan_video_segments(scenes, keyframe_timestamps=[0.0, 10.0, 20.0], total_duration=30.0)
    assert segments[-1].end == 30.0
    assert segments[-1].reencode is True
    # No trailing copy segment -- the re-encode segment already reaches the end.
    assert all(not s.reencode for s in segments[:-1])


def test_close_scenes_merge_into_one_reencode_segment():
    # Both scenes' keyframe-expanded ranges land in the same [10, 20) window.
    scenes = [MuteInterval(start=11.0, end=13.0), MuteInterval(start=16.0, end=18.0)]
    segments = plan_video_segments(scenes, keyframe_timestamps=[0.0, 10.0, 20.0, 30.0], total_duration=30.0)
    reencode_segments = [s for s in segments if s.reencode]
    assert len(reencode_segments) == 1
    assert reencode_segments[0].start == 10.0 and reencode_segments[0].end == 20.0
    assert set(reencode_segments[0].local_blur_intervals) == {
        MuteInterval(start=1.0, end=3.0),
        MuteInterval(start=6.0, end=8.0),
    }


def test_far_apart_scenes_stay_separate_with_copy_segment_between():
    scenes = [MuteInterval(start=15.0, end=17.0), MuteInterval(start=50.0, end=52.0)]
    segments = plan_video_segments(scenes, keyframe_timestamps=[0.0, 10.0, 20.0, 40.0, 60.0], total_duration=70.0)
    reencode_segments = [s for s in segments if s.reencode]
    copy_segments = [s for s in segments if not s.reencode]
    assert len(reencode_segments) == 2
    # Before the first scene, between the two scenes, and after the second.
    assert len(copy_segments) == 3


def test_empty_keyframes_falls_back_to_whole_file_reencode():
    # No usable keyframe data (e.g. a probe failure) -- degrades to exactly the
    # old "re-encode everything" behavior rather than cutting mid-GOP.
    scenes = [MuteInterval(start=10.0, end=15.0)]
    segments = plan_video_segments(scenes, keyframe_timestamps=[], total_duration=40.0)
    assert len(segments) == 1
    assert segments[0].start == 0.0 and segments[0].end == 40.0
    assert segments[0].reencode is True
    assert segments[0].local_blur_intervals == (MuteInterval(start=10.0, end=15.0),)


def test_parse_keyframe_csv_handles_real_ffprobe_trailing_comma():
    # Real bug: ffprobe's `csv=p=0` with one requested field still emits a
    # trailing comma per line ("0.000000,"), which silently made every
    # float(line) call fail and produce an empty list -- caught live when a
    # real Apply run degraded to a full whole-file re-encode instead of the
    # expected per-scene split.
    output = "0.000000,\n1.043000,\n4.046000,\n"
    assert _parse_keyframe_csv(output) == [0.0, 1.043, 4.046]


def test_parse_keyframe_csv_skips_blank_lines():
    assert _parse_keyframe_csv("0.000000,\n\n1.043000,\n") == [0.0, 1.043]


def test_segments_tile_full_duration_contiguously():
    scenes = [MuteInterval(start=5.0, end=6.0), MuteInterval(start=45.0, end=46.0), MuteInterval(start=90.0, end=95.0)]
    keyframes = [float(i) for i in range(0, 100, 4)]
    segments = plan_video_segments(scenes, keyframe_timestamps=keyframes, total_duration=100.0)
    assert segments[0].start == 0.0
    assert segments[-1].end == 100.0
    for a, b in zip(segments, segments[1:]):
        assert a.end == b.start


def test_no_mute_intervals_is_one_copy_segment():
    segments = plan_audio_segments([], total_duration=30.0)
    assert len(segments) == 1
    assert segments[0].start == 0.0 and segments[0].end == 30.0
    assert segments[0].mute is False


def test_single_mute_interval_used_directly_no_keyframe_expansion():
    # Unlike plan_video_segments, no expansion to a nearest boundary --
    # mute_intervals are used exactly as given.
    intervals = [MuteInterval(start=10.0, end=15.0)]
    segments = plan_audio_segments(intervals, total_duration=40.0)
    assert [(s.start, s.end, s.mute) for s in segments] == [
        (0.0, 10.0, False),
        (10.0, 15.0, True),
        (15.0, 40.0, False),
    ]


def test_mute_interval_at_very_start_has_no_leading_copy_segment():
    intervals = [MuteInterval(start=0.0, end=5.0)]
    segments = plan_audio_segments(intervals, total_duration=20.0)
    assert [(s.start, s.end, s.mute) for s in segments] == [(0.0, 5.0, True), (5.0, 20.0, False)]


def test_mute_interval_at_very_end_has_no_trailing_copy_segment():
    intervals = [MuteInterval(start=15.0, end=20.0)]
    segments = plan_audio_segments(intervals, total_duration=20.0)
    assert [(s.start, s.end, s.mute) for s in segments] == [(0.0, 15.0, False), (15.0, 20.0, True)]


def test_overlapping_mute_intervals_merge_into_one_segment():
    # merge_gap_seconds=0.0 (matching how mute_intervals are already merged
    # upstream in apply_scene_blur before this ever runs) only merges
    # touching/overlapping intervals, not merely nearby ones.
    intervals = [MuteInterval(start=10.0, end=12.0), MuteInterval(start=12.0, end=14.0)]
    segments = plan_audio_segments(intervals, total_duration=30.0)
    mute_segments = [s for s in segments if s.mute]
    assert len(mute_segments) == 1
    assert mute_segments[0].start == 10.0 and mute_segments[0].end == 14.0


def test_far_apart_mute_intervals_stay_separate():
    intervals = [MuteInterval(start=5.0, end=6.0), MuteInterval(start=50.0, end=51.0)]
    segments = plan_audio_segments(intervals, total_duration=60.0)
    mute_segments = [s for s in segments if s.mute]
    copy_segments = [s for s in segments if not s.mute]
    assert len(mute_segments) == 2
    assert len(copy_segments) == 3  # before first, between, after second


def test_audio_segments_tile_full_duration_contiguously():
    intervals = [MuteInterval(start=5.0, end=6.0), MuteInterval(start=45.0, end=46.0)]
    segments = plan_audio_segments(intervals, total_duration=100.0)
    assert segments[0].start == 0.0
    assert segments[-1].end == 100.0
    for a, b in zip(segments, segments[1:]):
        assert a.end == b.start


def _fp(**overrides):
    args = dict(
        video_path=Path("/plex/movie.mkv"),
        blur_intervals=[MuteInterval(start=1.0, end=2.0)],
        mute_intervals=[],
        video_crf=23,
        video_preset="medium",
        blur_radius=90,
        blur_power=8,
        audio_bitrate="192k",
    )
    args.update(overrides)
    return _blur_job_fingerprint(**args)


def test_fingerprint_is_deterministic():
    assert _fp() == _fp()


def test_fingerprint_changes_with_blur_intervals():
    assert _fp() != _fp(blur_intervals=[MuteInterval(start=1.0, end=3.0)])


def test_fingerprint_changes_with_mute_intervals():
    assert _fp() != _fp(mute_intervals=[MuteInterval(start=1.0, end=2.0)])


def test_fingerprint_changes_with_encode_settings():
    assert _fp() != _fp(video_crf=18)
    assert _fp() != _fp(video_preset="slow")
    assert _fp() != _fp(blur_radius=45)
    assert _fp() != _fp(blur_power=5)


def test_fingerprint_changes_with_video_path():
    assert _fp() != _fp(video_path=Path("/plex/other_movie.mkv"))


def test_reset_stale_work_dir_keeps_matching_fingerprint(tmp_path):
    (tmp_path / "reencode_0000.ts").write_bytes(b"fake segment data")
    (tmp_path / "fingerprint.txt").write_text("abc123")

    _reset_stale_work_dir(tmp_path, "abc123")

    assert (tmp_path / "reencode_0000.ts").exists()
    assert (tmp_path / "fingerprint.txt").read_text() == "abc123"


def test_reset_stale_work_dir_wipes_mismatched_fingerprint(tmp_path):
    (tmp_path / "reencode_0000.ts").write_bytes(b"stale segment from a different approved-scene set")
    (tmp_path / "fingerprint.txt").write_text("old-fingerprint")

    _reset_stale_work_dir(tmp_path, "new-fingerprint")

    assert not (tmp_path / "reencode_0000.ts").exists()
    assert (tmp_path / "fingerprint.txt").read_text() == "new-fingerprint"


def test_reset_stale_work_dir_on_brand_new_empty_dir(tmp_path):
    # No fingerprint.txt at all -- e.g. a fresh work_dir for a job that's
    # never run before -- should just write the fingerprint, not error.
    _reset_stale_work_dir(tmp_path, "first-run-fingerprint")
    assert (tmp_path / "fingerprint.txt").read_text() == "first-run-fingerprint"
