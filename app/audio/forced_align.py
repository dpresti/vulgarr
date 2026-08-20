"""Whisper-family forced alignment: locate already-matched words precisely within a
subtitle cue's audio, using torchaudio's MMS CTC forced aligner.

This is forced *alignment*, not open-vocabulary transcription/ASR -- detection stays
subtitle-based (app.subtitles.matcher), unchanged. All this does is take a cue whose
text we already matched against the word list, and ask "given this exact text, where
in the audio does each word actually start/end" -- a fundamentally more constrained
(and more reliable) problem than transcribing unknown speech.

torch/torchaudio are only imported inside functions here, never at module load, so
every processing job that isn't using Whisper-mode precision -- the overwhelming
majority -- pays zero import/model-load cost for this being available.
"""

import asyncio
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.subtitles.matcher import CueMatch, MatchSpan, normalize_cue_text

logger = logging.getLogger(__name__)

ALIGN_SAMPLE_RATE = 16000
# Extra audio pulled in on each side of the cue's own on-screen window -- subtitle
# timing is rarely frame-accurate to the audio, so the real speech for a word near a
# cue's edge can fall just outside the cue's nominal start/end.
WINDOW_PAD_SECONDS = 0.75

_WORD_RE = re.compile(r"\S+")

_bundle: dict = {}


def _get_bundle() -> dict:
    """Lazily load and cache the MMS forced-alignment model/tokenizer/aligner. This
    is a real model download+load (a few hundred MB, cached under $TORCH_HOME after
    the first run) so it only happens once per process, on first actual use."""
    if "model" not in _bundle:
        import torch
        from torchaudio.pipelines import MMS_FA

        _bundle["torch"] = torch
        _bundle["model"] = MMS_FA.get_model()
        _bundle["tokenizer"] = MMS_FA.get_tokenizer()
        _bundle["aligner"] = MMS_FA.get_aligner()
        _bundle["sample_rate"] = MMS_FA.sample_rate
    return _bundle


@dataclass(frozen=True)
class WordWindow:
    """One matched span's precise start/end, in absolute video-time seconds."""

    start: float
    end: float


def _tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Split already-normalized cue text into words, keeping each word's character
    offsets so per-word alignment output (same order as this list) can be mapped
    back to which MatchSpan(s) it covers."""
    return [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _extract_audio_window(ffmpeg_bin: str, video_path: Path, start: float, end: float):
    """Decode [start, end) of video_path's audio to a mono 16kHz float32 tensor via
    ffmpeg directly (raw PCM over a pipe) -- avoids depending on torchaudio's own
    audio-decoding backends, which need extra system libs we don't otherwise use."""
    torch = _bundle["torch"]
    duration = max(0.05, end - start)
    proc = subprocess.run(
        [
            ffmpeg_bin, "-nostdin", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(video_path),
            "-ac", "1", "-ar", str(ALIGN_SAMPLE_RATE), "-f", "s16le", "-",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    import numpy as np

    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(pcm).unsqueeze(0)


def _align_blocking(
    ffmpeg_bin: str,
    video_path: Path,
    window_start: float,
    window_end: float,
    words: list[tuple[str, int, int]],
    spans: tuple[MatchSpan, ...],
) -> list[WordWindow]:
    bundle = _get_bundle()
    torch = bundle["torch"]

    waveform = _extract_audio_window(ffmpeg_bin, video_path, window_start, window_end)

    with torch.inference_mode():
        emission, _ = bundle["model"](waveform)

    tokens = bundle["tokenizer"]([w[0] for w in words])
    token_spans = bundle["aligner"](emission[0], tokens)
    if len(token_spans) != len(words):
        raise ValueError(f"Aligner returned {len(token_spans)} word spans for {len(words)} input words")

    ratio = waveform.size(1) / emission.size(1) / bundle["sample_rate"]
    word_times = [(ts[0].start * ratio, ts[-1].end * ratio) for ts in token_spans]

    return _spans_to_word_windows(words, word_times, spans, window_start)


def _spans_to_word_windows(
    words: list[tuple[str, int, int]],
    word_times: list[tuple[float, float]],
    spans: tuple[MatchSpan, ...],
    window_start: float,
) -> list[WordWindow]:
    """Pure mapping from per-word alignment output (word_times, same order/length as
    words) to one WordWindow per matched span, in absolute video-time seconds. Kept
    torch-free so it's directly unit-testable without the real model."""
    results: list[WordWindow] = []
    for span in spans:
        # A matched span can cover more than one aligned word (multi-word phrases in
        # the word list) -- take the union of every word whose character range
        # overlaps the span's.
        covering = [word_times[i] for i, (_, wstart, wend) in enumerate(words) if wstart < span.end and wend > span.start]
        if not covering:
            raise ValueError(f"No aligned word overlaps matched span {span!r}")
        results.append(
            WordWindow(
                start=window_start + min(t[0] for t in covering),
                end=window_start + max(t[1] for t in covering),
            )
        )
    return results


async def align_matches_for_cue(video_path: Path, ffmpeg_bin: str, match: CueMatch) -> list[WordWindow] | None:
    """One WordWindow per entry in match.spans (same order), in absolute video-time
    seconds -- or None if alignment couldn't be attempted/trusted for this cue, in
    which case the caller should fall back to the proportional-estimate method for
    just that cue rather than failing the whole job."""
    if not match.spans or not match.normalized_length:
        return None

    words = _tokenize_with_offsets(normalize_cue_text(match.cue.text))
    if not words:
        return None

    window_start = max(0.0, match.cue.start_seconds - WINDOW_PAD_SECONDS)
    window_end = match.cue.end_seconds + WINDOW_PAD_SECONDS

    try:
        return await asyncio.to_thread(
            _align_blocking, ffmpeg_bin, video_path, window_start, window_end, words, match.spans
        )
    except Exception:
        logger.warning(
            "Whisper forced alignment failed for cue at %.2fs in %s, falling back to estimate",
            match.cue.start_seconds, video_path, exc_info=True,
        )
        return None
