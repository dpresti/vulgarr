"""Converts a hand-typed, plain-text list of nudity timestamp windows (e.g.
transcribed by a person reading a title's IMDb "Sex & Nudity" Parents Guide
entry, or any other source read manually rather than scraped) into the same
ground-truth fixture JSON schema scripts/bench_scene_detection.py consumes
via its --fixture flag.

Deliberately does not fetch or parse any web page itself -- IMDb's terms
prohibit automated scraping/data-mining of their site even for non-commercial
use (the free bulk dataset also doesn't include the actual scene-description
text, only vote counts), so this only ever operates on text a person already
read and typed by hand. Not a scraper; a format converter.

Input format, one window per line:

    S01E05 12:34-13:10
    S01E05 1:02:34-1:03:10
    S02E01 5:00-5:45

Blank lines and lines starting with # are ignored. Timestamps accept M:SS,
MM:SS, or H:MM:SS. An episode with no lines at all in the input is simply
absent from the output fixture (not written as a null-windows placeholder --
unlike the GoT fixture's Reddit source, there's no "no data recorded, NOT
confirmed clean" distinction to preserve here, since this tool is never fed
a full episode list to begin with, only the ones someone actually looked up).

Usage:
    python3 scripts/import_ground_truth.py --series-title "The Wire" \\
        --input my_timestamps.txt --output tests/fixtures/wire_ground_truth.json

    # Add more episodes to an existing fixture (e.g. more GoT episodes,
    # sourced from IMDb rather than the original Reddit post) without
    # clobbering what's already there:
    python3 scripts/import_ground_truth.py --series-title "Game of Thrones" \\
        --input more_got_timestamps.txt \\
        --output tests/fixtures/got_nudity_ground_truth.json --merge
"""

import argparse
import json
import re
from pathlib import Path

_LINE_RE = re.compile(
    r"^S(?P<season>\d+)E(?P<episode>\d+)\s+(?P<start>[\d:]+)-(?P<end>[\d:]+)$", re.IGNORECASE
)


def _parse_timestamp(value: str) -> float:
    parts = [int(p) for p in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unrecognized timestamp format: {value!r} (expected M:SS or H:MM:SS)")


def parse_input(text: str) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Pure function, no I/O -- returns {(season, episode): [(start, end), ...]},
    windows in the order they appeared in the input (not yet sorted/merged --
    a person transcribing a page top-to-bottom may naturally list them out of
    chronological order across re-reads; sorting happens once, at output
    time, in build_fixture)."""
    windows_by_episode: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for line_num, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE_RE.match(line)
        if match is None:
            raise ValueError(f"Line {line_num}: could not parse {raw_line!r} (expected 'S01E05 12:34-13:10')")
        season = int(match.group("season"))
        episode = int(match.group("episode"))
        start = _parse_timestamp(match.group("start"))
        end = _parse_timestamp(match.group("end"))
        if end <= start:
            raise ValueError(f"Line {line_num}: end ({match.group('end')}) is not after start ({match.group('start')})")
        windows_by_episode.setdefault((season, episode), []).append((start, end))
    return windows_by_episode


def build_fixture(
    series_title: str,
    source: str,
    windows_by_episode: dict[tuple[int, int], list[tuple[float, float]]],
    existing: dict | None = None,
) -> dict:
    """Merges parsed windows into an existing fixture dict (if given -- must
    already have a matching series_title) or builds a fresh one. An episode
    already present in `existing` gets its windows list extended and
    re-sorted, not replaced -- so re-running this against a growing manual
    transcription file is safe to repeat rather than something that has to
    track what was already imported."""
    if existing is not None:
        if existing.get("series_title") != series_title:
            raise ValueError(
                f"--merge target's series_title ({existing.get('series_title')!r}) "
                f"doesn't match --series-title ({series_title!r})"
            )
        fixture = existing
        fixture.setdefault("sources", [])
        if source not in fixture["sources"]:
            fixture["sources"].append(source)
    else:
        fixture = {
            "series_title": series_title,
            "sources": [source],
            "format_note": "windows are [start_seconds, end_seconds] nudity/sex-content windows, "
            "hand-transcribed from the listed source(s) via scripts/import_ground_truth.py "
            "-- not scraped, see that script's docstring for why.",
            "seasons": {},
        }

    seasons: dict[str, list[dict]] = fixture["seasons"]
    for (season, episode), new_windows in windows_by_episode.items():
        season_key = str(season)
        entries = seasons.setdefault(season_key, [])
        existing_entry = next((e for e in entries if e["episode"] == episode), None)
        if existing_entry is None:
            entries.append({"episode": episode, "windows": sorted(new_windows)})
        else:
            merged = sorted(set(tuple(w) for w in existing_entry["windows"]) | set(new_windows))
            existing_entry["windows"] = [list(w) for w in merged]
        entries.sort(key=lambda e: e["episode"])

    return fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--series-title", required=True, help="Must match Title.series_title in the app DB exactly")
    parser.add_argument("--input", type=Path, required=True, help="Plain-text file of 'S01E05 12:34-13:10' lines")
    parser.add_argument("--output", type=Path, required=True, help="Fixture JSON path to write")
    parser.add_argument(
        "--source", default="Manually transcribed from IMDb Parents Guide",
        help="Free-text provenance note stored in the fixture's sources list",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="If --output already exists, add to it instead of overwriting (episode windows are unioned, not replaced)",
    )
    args = parser.parse_args()

    windows_by_episode = parse_input(args.input.read_text())
    if not windows_by_episode:
        raise SystemExit(f"No windows parsed from {args.input} -- check the format (see this script's docstring)")

    existing = None
    if args.merge:
        if not args.output.exists():
            raise SystemExit(f"--merge given but {args.output} doesn't exist yet")
        existing = json.loads(args.output.read_text())
    elif args.output.exists():
        raise SystemExit(f"{args.output} already exists -- pass --merge to add to it, or choose a different --output")

    fixture = build_fixture(args.series_title, args.source, windows_by_episode, existing)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")

    total_windows = sum(len(w) for w in windows_by_episode.values())
    print(f"Wrote {len(windows_by_episode)} episode(s), {total_windows} window(s) to {args.output}")


if __name__ == "__main__":
    main()
