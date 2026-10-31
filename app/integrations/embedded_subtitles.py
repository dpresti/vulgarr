"""Fallback for videos whose subtitles live inside the container (e.g. an mkv
with muxed SRT/ASS tracks) rather than as a sidecar .srt Bazarr dropped next to
it. Extracts the best matching text track to an .srt in the data dir, which
subtitle_lookup.find_subtitle_for_video then picks up like any other file.

Kept out of find_subtitle_for_video itself: probing costs ~0.3s/file and
extracting has to demux the entire video, so doing either for every title on a
full library sync (tens of thousands of them) isn't viable. Callers that are
about to actually process a title call extract_embedded_subtitle explicitly;
find_subtitle_for_video only does the cheap "was it already extracted" check.

Image-based tracks (PGS, VobSub, DVB) are skipped -- they're pictures of text
and would need OCR to become something the wordlist matcher can read.
"""

import hashlib
import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.config import settings as app_settings

logger = logging.getLogger(__name__)

_TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
_IMAGE_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"}

# ISO 639-1 -> the 639-2 forms ffprobe/mkvmerge actually write into tags.
_LANG_CODES = {
    "en": {"en", "eng"},
}

_FORCED_TITLE_RE = re.compile(r"\bforced\b", re.IGNORECASE)
_SRT_CUE_RE = re.compile(r"-->")

_PROBE_TIMEOUT_SECONDS = 60
# Has to read the whole file -- a 4K remux over NFS takes a while.
_EXTRACT_TIMEOUT_SECONDS = 900


def embedded_subtitle_dir() -> Path:
    return app_settings.data_dir / "embedded_subs"


def cached_subtitle_path(video_path: Path, language: str) -> Path | None:
    """Where an extraction of video_path would be stored. Keyed on the file's
    size and mtime too, so an upgraded/replaced file at the same path never
    reuses the old file's subtitles. None if the video doesn't exist."""
    try:
        st = video_path.stat()
    except OSError:
        return None
    key = hashlib.sha1(f"{video_path}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:20]
    return embedded_subtitle_dir() / f"{key}.{language}.srt"


@dataclass(frozen=True)
class EmbeddedSubtitleProbe:
    text_candidates: list[int]  # absolute stream indexes, matching language, not forced
    has_image_only_match: bool  # language matched, but only as PGS/VobSub


def _lang_codes(language: str) -> set[str]:
    return _LANG_CODES.get(language, {language})


def select_candidate_streams(streams: list[dict], language: str) -> EmbeddedSubtitleProbe:
    codes = _lang_codes(language)
    candidates: list[int] = []
    image_match = False
    for s in streams:
        if s.get("codec_type") != "subtitle":
            continue
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
        if (tags.get("language") or "").lower() not in codes:
            continue
        codec = s.get("codec_name") or ""
        if codec in _IMAGE_CODECS:
            image_match = True
            continue
        if codec not in _TEXT_CODECS:
            continue
        # Forced tracks only cover foreign-language/sign lines -- useless for
        # catching profanity in the main dialogue.
        if (s.get("disposition") or {}).get("forced") or _FORCED_TITLE_RE.search(tags.get("title") or ""):
            continue
        candidates.append(s["index"])
    return EmbeddedSubtitleProbe(text_candidates=candidates, has_image_only_match=image_match and not candidates)


def probe_embedded_subtitles(video_path: Path, language: str) -> EmbeddedSubtitleProbe:
    result = subprocess.run(
        [
            app_settings.ffprobe_bin, "-v", "error", "-select_streams", "s",
            "-show_entries", "stream=index,codec_name,codec_type:stream_tags:stream_disposition=forced",
            "-of", "json", str(video_path),
        ],
        capture_output=True, text=True, timeout=_PROBE_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        logger.warning("ffprobe failed on %s: %s", video_path, result.stderr.strip())
        return EmbeddedSubtitleProbe(text_candidates=[], has_image_only_match=False)
    streams = json.loads(result.stdout or "{}").get("streams", [])
    return select_candidate_streams(streams, language)


def _count_cues(path: Path) -> int:
    try:
        return len(_SRT_CUE_RE.findall(path.read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return 0


def extract_embedded_subtitle(video_path: Path, language: str = "en") -> Path | None:
    """Extract the best embedded text subtitle track in `language` to the
    cache dir and return its path, or None if there isn't one. Reuses a prior
    extraction for the same file. Blocking -- run via asyncio.to_thread.

    When several tracks match (commonly a regular + an SDH English track, or
    an unflagged forced track alongside the full one), all of them are pulled
    in the same single demux pass and the one with the most cues wins: that's
    the full dialogue track, and is robust to forced tracks that were never
    tagged as forced."""
    out_path = cached_subtitle_path(video_path, language)
    if out_path is None:
        return None
    if out_path.exists():
        return out_path

    probe = probe_embedded_subtitles(video_path, language)
    if not probe.text_candidates:
        if probe.has_image_only_match:
            logger.info("%s only has image-based (PGS/VobSub) %s subtitles embedded; needs OCR, skipping", video_path, language)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_path.parent) as tmp:
        tmp_dir = Path(tmp)
        cmd = [app_settings.ffmpeg_bin, "-v", "error", "-nostdin", "-y", "-i", str(video_path)]
        outputs = []
        for idx in probe.text_candidates:
            dest = tmp_dir / f"{idx}.srt"
            cmd += ["-map", f"0:{idx}", "-c:s", "srt", str(dest)]
            outputs.append(dest)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=_EXTRACT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning("Timed out extracting embedded subtitles from %s", video_path)
            return None
        if result.returncode != 0:
            logger.warning("ffmpeg failed extracting subtitles from %s: %s", video_path, result.stderr.strip())
            return None
        best = max(outputs, key=_count_cues)
        if _count_cues(best) == 0:
            return None
        best.replace(out_path)
    logger.info("Extracted embedded %s subtitle from %s -> %s", language, video_path, out_path)
    return out_path


def describe_missing_embedded(video_path: Path, language: str) -> str | None:
    """Short user-facing explanation when a file has embedded subtitles we
    can't use, else None."""
    try:
        probe = probe_embedded_subtitles(video_path, language)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.has_image_only_match:
        return "the embedded subtitles are image-based (PGS/VobSub), which can't be read without OCR"
    return None
