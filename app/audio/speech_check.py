"""Speech check: transcribe a title's dialogue and match the word list against
what's actually *said*, not just what the subtitle says.

Subtitles routinely drop or soften swearing -- seen on Godzilla (1998): a
"what the hell is it" subtitled as "what it is", and shouted lines in action
scenes left out entirely -- and subtitle matching alone can never catch those.
Hits here are muted alongside the subtitle matches; their windows also cover
for a subtitle match the forced aligner placed badly, since overlapping
intervals just merge.

Runs Whisper (faster-whisper, CPU int8) over the source track's dialogue
channel. Roughly 5x realtime on this host, so ~25-30 minutes for a feature
film -- the transcript is cached per title so reprocessing (word list edits,
Keep toggles) doesn't pay that again.

VAD threshold is deliberately far below faster-whisper's 0.5 default: at 0.5
it skipped real lines under loud effects (Godzilla's "hell's" at 18:02,
"Jesus Christ" at 20:43); 0.1 caught them at ~30% more runtime. Turning VAD
off entirely caught no more, at 3x the runtime.
"""

import asyncio
import json
import logging
import os
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Awaitable, Callable

from app.audio.source import AudioSource
from app.config import settings as app_settings
from app.subtitles.matcher import CueMatch, ProfanityMatcher, normalize_cue_text
from app.subtitles.parser import SubtitleCue

logger = logging.getLogger(__name__)

SPEECH_MODEL = "small.en"
SAMPLE_RATE = 16000
_VAD_PARAMETERS = {"threshold": 0.1, "speech_pad_ms": 600, "min_silence_duration_ms": 1500}
# Whisper's own per-word probability. Low on purpose -- a missed word reaches
# the viewer, a false hit just mutes a fraction of a second and can be Kept.
MIN_WORD_PROBABILITY = 0.3
# A heard word within this much of a subtitle match for the same term is the
# same utterance, not a new find.
DUPLICATE_TOLERANCE_SECONDS = 1.5
_TRANSCRIPT_VERSION = 1

SpeechProgressCallback = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class HeardWord:
    text: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class HeardSegment:
    start: float
    end: float
    text: str
    words: tuple[HeardWord, ...]


@dataclass(frozen=True)
class HeardMatch:
    match: CueMatch  # cue = the transcribed segment (its text is what was heard)
    windows: tuple[tuple[float, float], ...]  # one per matched span, absolute seconds


_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel

            # Half the cores, same reasoning as forced_align's torch thread cap:
            # the app keeps serving HTTP/DB work on the same CPU meanwhile.
            _model = WhisperModel(
                SPEECH_MODEL,
                device="cpu",
                compute_type="int8",
                cpu_threads=max(1, (os.cpu_count() or 4) // 2),
                download_root=str(app_settings.data_dir / "whisper_models"),
            )
        return _model


def transcript_cache_path(title_id: int) -> Path:
    return app_settings.data_dir / "transcripts" / f"title_{title_id}.json"


def _signature(video_path: Path, source: AudioSource, duration: float) -> dict:
    # Not size/mtime: every successful run rewrites the file (new Clean track),
    # which would invalidate the cache on exactly the reprocesses it's for. The
    # source stream and duration don't change across our own remuxes, but do
    # when the file is replaced by a different release.
    return {
        "version": _TRANSCRIPT_VERSION,
        "model": SPEECH_MODEL,
        "video_path": str(video_path),
        "stream_index": source.stream_index,
        "duration": round(duration, 1),
    }


def _load_cached(path: Path, signature: dict) -> list[HeardSegment] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if data.get("signature") != signature:
        return None
    return [
        HeardSegment(s["start"], s["end"], s["text"], tuple(HeardWord(**w) for w in s["words"]))
        for s in data["segments"]
    ]


def _save_cached(path: Path, signature: dict, segments: list[HeardSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"signature": signature, "segments": [asdict(s) for s in segments]}))
    tmp.replace(path)


def _decode_audio(ffmpeg_bin: str, video_path: Path, source: AudioSource):
    import numpy as np

    proc = subprocess.run(
        [
            ffmpeg_bin, "-nostdin", "-loglevel", "error", "-i", str(video_path),
            *source.ffmpeg_args(), "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _transcribe_blocking(
    ffmpeg_bin: str,
    video_path: Path,
    source: AudioSource,
    duration: float,
    report: Callable[[float], None],
    cancelled: threading.Event,
) -> list[HeardSegment] | None:
    audio = _decode_audio(ffmpeg_bin, video_path, source)
    total = duration or (len(audio) / SAMPLE_RATE) or 1.0
    segments_iter, _ = _get_model().transcribe(
        audio,
        language="en",
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=_VAD_PARAMETERS,
        # Stops one misheard stretch from cascading into the following ones.
        condition_on_previous_text=False,
    )
    segments: list[HeardSegment] = []
    for seg in segments_iter:
        if cancelled.is_set():
            return None
        words = tuple(
            HeardWord(w.word.strip(), round(w.start, 3), round(w.end, 3), round(w.probability, 3))
            for w in (seg.words or [])
            if w.word.strip()
        )
        segments.append(HeardSegment(round(seg.start, 3), round(seg.end, 3), seg.text.strip(), words))
        report(min(1.0, seg.end / total))
    return segments


async def get_transcript(
    *,
    title_id: int,
    video_path: Path,
    source: AudioSource,
    duration: float,
    ffmpeg_bin: str,
    on_progress: SpeechProgressCallback | None = None,
) -> list[HeardSegment]:
    signature = _signature(video_path, source, duration)
    cache_path = transcript_cache_path(title_id)
    cached = await asyncio.to_thread(_load_cached, cache_path, signature)
    if cached is not None:
        return cached

    loop = asyncio.get_running_loop()
    cancelled = threading.Event()

    def report(fraction: float) -> None:
        if on_progress is not None:
            asyncio.run_coroutine_threadsafe(on_progress(fraction), loop)

    try:
        segments = await asyncio.to_thread(
            _transcribe_blocking, ffmpeg_bin, video_path, source, duration, report, cancelled
        )
    except asyncio.CancelledError:
        # to_thread can't interrupt the worker thread -- tell it to stop at the
        # next segment rather than burning CPU for the rest of the movie.
        cancelled.set()
        raise
    if segments is None:
        return []
    await asyncio.to_thread(_save_cached, cache_path, signature, segments)
    return segments


def find_heard_matches(
    segments: list[HeardSegment], matcher: ProfanityMatcher, min_probability: float = MIN_WORD_PROBABILITY
) -> list[HeardMatch]:
    """Run the word list over each transcribed segment, mapping each matched
    span back to the heard words' own timestamps. Built from per-word
    normalized text so span offsets line up with word boundaries exactly."""
    results: list[HeardMatch] = []
    for seg in segments:
        tokens: list[tuple[HeardWord, int, int]] = []
        parts: list[str] = []
        offset = 0
        for w in seg.words:
            norm = " ".join(normalize_cue_text(w.text).split())
            if not norm:
                continue
            tokens.append((w, offset, offset + len(norm)))
            parts.append(norm)
            offset += len(norm) + 1
        if not tokens:
            continue
        cue = SubtitleCue(index=0, start_seconds=seg.start, end_seconds=seg.end, text=" ".join(parts))
        m = matcher.match_cue(cue)
        if m is None:
            continue
        windows: list[tuple[float, float]] = []
        for span in m.spans:
            covering = [w for w, s, e in tokens if s < span.end and e > span.start]
            if not covering or max(w.probability for w in covering) < min_probability:
                continue
            windows.append((min(w.start for w in covering), max(w.end for w in covering)))
        if windows:
            heard_cue = SubtitleCue(index=0, start_seconds=windows[0][0], end_seconds=windows[-1][1], text=seg.text)
            results.append(HeardMatch(match=CueMatch(cue=heard_cue, matched_terms=m.matched_terms, severity=m.severity), windows=tuple(windows)))
    return results


def is_duplicate_of_subtitle(heard: HeardMatch, subtitle_matches: list[CueMatch]) -> bool:
    terms = set(heard.match.matched_terms)
    start, end = heard.windows[0][0], heard.windows[-1][1]
    return any(
        terms & set(m.matched_terms)
        and start < m.cue.end_seconds + DUPLICATE_TOLERANCE_SECONDS
        and end > m.cue.start_seconds - DUPLICATE_TOLERANCE_SECONDS
        for m in subtitle_matches
    )
