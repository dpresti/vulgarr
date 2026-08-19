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

    lower_index = {p.name.lower(): p for p in directory.glob(f"{_glob_escape(stem)}*.srt")}
    for candidate in candidates:
        match = lower_index.get(candidate.lower())
        if match is not None:
            return match

    return None


def _glob_escape(s: str) -> str:
    for ch in "[]?*":
        s = s.replace(ch, f"[{ch}]")
    return s
