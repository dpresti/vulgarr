from app.subtitles.parser import parse_srt

SAMPLE = """1
00:00:01,000 --> 00:00:03,500
<i>Hello there,</i> friend.

2
00:00:04,000 --> 00:00:06,250
This is a
multi-line cue.

3
00:00:07,000 --> 00:00:09,000
{\\an8}Tag soup <font color="#ffffff">test</font>.
"""


def test_parses_basic_cues():
    cues = parse_srt(SAMPLE)
    assert len(cues) == 3
    assert cues[0].text == "Hello there, friend."
    assert cues[0].start_seconds == 1.0
    assert cues[0].end_seconds == 3.5


def test_joins_multiline_cue_with_single_space():
    cues = parse_srt(SAMPLE)
    assert cues[1].text == "This is a multi-line cue."


def test_strips_html_and_brace_tags():
    cues = parse_srt(SAMPLE)
    assert cues[2].text == "Tag soup test."


def test_tolerates_missing_blank_line_and_bom():
    content = "﻿1\n00:00:00,500 --> 00:00:01,000\nHi.\n"
    cues = parse_srt(content)
    assert len(cues) == 1
    assert cues[0].text == "Hi."


def test_ignores_malformed_block():
    content = SAMPLE + "\n4\nnot a timestamp\nbroken\n"
    cues = parse_srt(content)
    assert len(cues) == 3
