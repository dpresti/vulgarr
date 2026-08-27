"""Convention-based lookup for a video file's subtitle, since Bazarr drops
subtitles alongside the video rather than exposing an API we can always rely on.
"""

from pathlib import Path

# Ordered by preference: language-tagged variants first, then bare .srt.
_LANG_ALIASES = {
    "en": ["en", "eng"],
}


def find_subtitle_for_video(video_path: Path, preferred_language: str = "en") -> Path | None:
    directory = video_path.parent
    stem = video_path.stem
    if not directory.exists():
        return None

    candidates: list[str] = []
    for lang in _LANG_ALIASES.get(preferred_language, [preferred_language]):
        candidates.append(f"{stem}.{lang}.srt")
        candidates.append(f"{stem}.{lang}.sdh.srt")
        candidates.append(f"{stem}.{lang}.forced.srt")
    candidates.append(f"{stem}.srt")

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

    return None
