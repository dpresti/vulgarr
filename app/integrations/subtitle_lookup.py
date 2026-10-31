"""Convention-based lookup for a video file's subtitle, since Bazarr drops
subtitles alongside the video rather than exposing an API we can always rely on.
"""

from pathlib import Path

from app.integrations.embedded_subtitles import cached_subtitle_path, extract_embedded_subtitle

# Ordered by preference: language-tagged variants first, then bare .srt.
_LANG_ALIASES = {
    "en": ["en", "eng"],
}


def find_subtitle_for_video(video_path: Path, preferred_language: str = "en") -> Path | None:
    directory = video_path.parent
    stem = video_path.stem
    if not directory.exists():
        return None

    langs = _LANG_ALIASES.get(preferred_language, [preferred_language])
    candidates: list[str] = []
    for lang in langs:
        candidates.append(f"{stem}.{lang}.srt")
        candidates.append(f"{stem}.{lang}.sdh.srt")
    candidates.append(f"{stem}.srt")
    # Forced subs only cover foreign-language/sign lines, not the main
    # dialogue -- a full track embedded in the video beats them, so they're
    # checked separately, after the embedded-extraction cache below.
    forced_candidates = [f"{stem}.{lang}.forced.srt" for lang in langs]

    # glob() itself is case-sensitive on Linux, which used to undermine the
    # case-insensitive matching below before it even got a chance to run --
    # a video "Movie.Name.2020.mkv" with a subtitle downloaded as
    # "movie.name.2020.en.srt" (a common release/Bazarr casing mismatch)
    # would never be found, since the glob pattern (built from the
    # video's own, differently-cased stem) wouldn't match it at all.
    # List every .srt in the directory instead and do the casing-insensitive
    # comparison ourselves, all the way through.
    stem_lower = stem.lower()
    lower_index = {
        p.name.lower(): p for p in directory.glob("*.srt") if p.name.lower().startswith(stem_lower)
    }
    for candidate in candidates:
        match = lower_index.get(candidate.lower())
        if match is not None:
            return match

    extracted = cached_subtitle_path(video_path, preferred_language)
    if extracted is not None and extracted.exists():
        return extracted

    for candidate in forced_candidates:
        match = lower_index.get(candidate.lower())
        if match is not None:
            return match

    return None


def is_forced_subtitle(path: Path) -> bool:
    return path.name.lower().endswith(".forced.srt")


def resolve_subtitle_for_video(video_path: Path, preferred_language: str = "en") -> Path | None:
    """find_subtitle_for_video, but when that comes up empty (or with only a
    forced track), also tries extracting a full text track embedded in the
    video itself. Slow when it has to extract (demuxes the whole file) --
    only call this for a title that's actually about to be processed, never
    per-title on a library sync, and always via asyncio.to_thread."""
    found = find_subtitle_for_video(video_path, preferred_language)
    if found is not None and not is_forced_subtitle(found):
        return found
    return extract_embedded_subtitle(video_path, preferred_language) or found
