"""Parse .srt subtitle files into a list of timed cues.

Handles common subtitle formatting quirks: HTML-ish tags (<i>, <b>, <font ...>),
curly/smart quotes, mid-cue line breaks, and byte-order marks.
"""

import re
from dataclasses import dataclass

_TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")
_BRACE_TAG_RE = re.compile(r"\{[^}]+\}")  # ASS-style override tags sometimes embedded in .srt


@dataclass(frozen=True)
class SubtitleCue:
    index: int
    start_seconds: float
    end_seconds: float
    text: str  # tags stripped, lines joined with a single space


def _timestamp_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _clean_text(raw_lines: list[str]) -> str:
    joined = " ".join(raw_lines)
    joined = _TAG_RE.sub(" ", joined)
    joined = _BRACE_TAG_RE.sub(" ", joined)
    joined = joined.replace("’", "'").replace("‘", "'")
    joined = joined.replace("“", '"').replace("”", '"')
    joined = re.sub(r"\s+", " ", joined).strip()
    joined = re.sub(r"\s+([.,!?;:])", r"\1", joined)  # tag-stripping can leave a stray space before punctuation
    return joined


def parse_srt(content: str) -> list[SubtitleCue]:
    """Parse raw .srt file contents into a list of SubtitleCue.

    Tolerant of missing/duplicated blank lines and stray whitespace, since
    subtitles from varied sources are frequently slightly malformed.
    """
    content = content.lstrip("﻿")
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    blocks = re.split(r"\n\s*\n", content.strip())
    cues: list[SubtitleCue] = []

    for block in blocks:
        lines = [line for line in block.split("\n") if line.strip() != ""]
        if not lines:
            continue

        # First line is normally a numeric index, but tolerate it being absent
        # or the timestamp line coming first.
        timestamp_line_idx = None
        for i, line in enumerate(lines[:2]):
            if _TIMESTAMP_RE.search(line):
                timestamp_line_idx = i
                break
        if timestamp_line_idx is None:
            continue  # not a valid cue block

        match = _TIMESTAMP_RE.search(lines[timestamp_line_idx])
        if not match:
            continue

        start = _timestamp_to_seconds(*match.group(1, 2, 3, 4))
        end = _timestamp_to_seconds(*match.group(5, 6, 7, 8))

        text_lines = lines[timestamp_line_idx + 1 :]
        text = _clean_text(text_lines)
        if not text:
            continue

        # Use the running cue count as the index rather than trusting the
        # subtitle file's own numbering, which is sometimes wrong/duplicated.
        cues.append(SubtitleCue(index=len(cues) + 1, start_seconds=start, end_seconds=end, text=text))

    return cues


def parse_srt_file(path: str) -> list[SubtitleCue]:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return parse_srt(f.read())
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode subtitle file with any known encoding: {path}")
