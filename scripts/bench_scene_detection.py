"""Benchmark app.vision scene-detection accuracy against real ground-truth nudity
windows (tests/fixtures/*_ground_truth.json), Phase 0 of the scene-detection
clean/speed/accuracy plan. Read-only against the app database -- scans real media and
scores candidates against ground truth, never writes DetectedScene rows or touches
Title.scene_scan_status. Not part of the running app; a dev tool for tuning
app/vision/classifier.py, app/vision/scene_cluster.py and their settings.

Must run inside the vulgarr-dev container -- needs real ffmpeg, a loaded NudeNet model,
and the real media library mounted at /plex, none of which exist outside it:

    docker exec -w /app vulgarr-dev python3 scripts/bench_scene_detection.py
    docker exec -w /app vulgarr-dev python3 scripts/bench_scene_detection.py --season 1 --limit 2
    docker exec -w /app vulgarr-dev python3 scripts/bench_scene_detection.py --save /data/bench_baseline.json

Uses whatever scene_* settings currently live in this container's own database (the
same settings the real app would use for a scan) -- to benchmark a *changed* setting,
change it via the Settings UI (or set_setting directly) in this same dev instance first.
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.common.intervals import merge_intervals  # noqa: E402
from app.config import settings as app_settings  # noqa: E402
from app.db.models import Title  # noqa: E402
from app.db.session import get_session, get_setting  # noqa: E402
from app.mux.remux import probe  # noqa: E402
from app.scenes.pipeline import _refine_candidate_boundary  # noqa: E402
from app.vision.classifier import scan_video_frames  # noqa: E402
from app.vision.scene_cluster import cluster_scenes  # noqa: E402

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "got_nudity_ground_truth.json"


@dataclass
class EpisodeScore:
    season: int
    episode: int
    title_id: int
    display_name: str
    duration_seconds: float
    scan_seconds: float
    ground_truth: list[list[float]]
    predicted: list[list[float]]
    matched: int
    missed: list[list[float]]
    false_positives: int
    mean_boundary_error_seconds: float | None


def score_episode(
    ground_truth: list[tuple[float, float]], predicted: list[tuple[float, float]]
) -> tuple[int, list[tuple[float, float]], int, list[float]]:
    """Many-to-one overlap matching: a ground-truth window counts as found if
    *any* predicted window overlaps it at all -- one or several, merged or not.
    A predicted window is a false positive only if it overlaps no ground-truth
    window at all -- one predicted window legitimately covering several
    ground-truth windows (this app's scene_merge_gap_seconds is designed to do
    exactly that) is not penalized twice for being "reused." Pure function, no
    I/O -- testable independently of a real scan.

    Used to require covering >=50% of the ground-truth window's own duration
    (see this function's git history for that version), not just any overlap.
    Dropped after directly confirming, on GoT S03E03's [2296,2356] window, that
    it was scoring a *correct* detection as a miss: real BUTTOCKS_EXPOSED
    signal (0.5-0.82 confidence) was accurately found and tightly bounded
    ([2318.0,2329.5]), but that's only 19% of the labeled window's own 60s
    span, because the ground-truth fixture itself labels the whole loosely-
    bounded scene (including non-explicit connecting shots) as one window, not
    just the frames with visible exposure. Requiring majority coverage of a
    label whose own width doesn't reflect real exposure duration measures the
    fixture's labeling looseness, not this app's actual recall -- what
    actually matters for this app's purpose (a human reviews every candidate
    before any edit is made) is whether a real scene produced *any* candidate
    for a reviewer to see at all, not whether that candidate's boundaries
    happen to span most of an arbitrarily-drawn label.
    """
    matched = 0
    missed: list[tuple[float, float]] = []
    boundary_errors: list[float] = []
    for gt in ground_truth:
        overlapping = [p for p in predicted if p[1] > gt[0] and p[0] < gt[1]]
        if overlapping:
            matched += 1
            best = min(overlapping, key=lambda p: abs(p[0] - gt[0]) + abs(p[1] - gt[1]))
            boundary_errors.append(abs(best[0] - gt[0]) + abs(best[1] - gt[1]))
        else:
            missed.append(gt)
    false_positives = sum(1 for p in predicted if not any(p[1] > gt[0] and p[0] < gt[1] for gt in ground_truth))
    return matched, missed, false_positives, boundary_errors


def cluster_level_boundary_errors(
    ground_truth: list[tuple[float, float]], predicted: list[tuple[float, float]], merge_gap_seconds: float
) -> list[float]:
    """Real boundary-error signal: score_episode's per-raw-window boundary_errors
    is misleading whenever several ground-truth windows are close enough together
    (within merge_gap_seconds) that the app's own cluster_scenes would -- by
    design -- merge them into one continuous predicted window (see
    scene_merge_gap_seconds' own DEFAULT_SETTINGS comment). Comparing that one
    wide, intentionally-merged window against each individual sub-window's edges
    manufactures a huge "error" out of what's actually correct behavior (found
    live: a single [562, 693] predicted window against three ground-truth
    sub-windows scored a 95s "boundary error" that was really just the sub-window
    gaps themselves, not a real detection failure).

    This re-groups ground truth into the same merge_gap_seconds clusters the app
    itself would treat as one scene, then compares each cluster's own outer
    [start, end] against the union of whatever predicted window(s) overlap it --
    one real error value per genuine scene, not per fine-grained sub-cut. Pure
    function, no I/O -- testable independently of a real scan."""
    clusters = merge_intervals(ground_truth, merge_gap_seconds)
    errors: list[float] = []
    for cluster_start, cluster_end in clusters:
        overlapping = [p for p in predicted if p[1] > cluster_start and p[0] < cluster_end]
        if not overlapping:
            continue  # no coverage at all -- already counted as a miss by score_episode, not a boundary error
        union_start = min(p[0] for p in overlapping)
        union_end = max(p[1] for p in overlapping)
        errors.append(abs(union_start - cluster_start) + abs(union_end - cluster_end))
    return errors


async def scan_title(session, video_path: Path) -> tuple[list[tuple[float, float]], float]:
    """Runs the same scan+cluster+per-candidate-refine steps
    app.scenes.pipeline.scan_for_scenes does, using this container's current DB
    settings -- deliberately NOT calling scan_for_scenes itself, since that also
    writes DetectedScene rows, deletes prior pending/rejected candidates, and can
    fire the (possibly cost-incurring) Claude Vision verify step. This harness only
    measures the local classifier pipeline's own accuracy."""
    confidence_threshold = float(await get_setting(session, "scene_confidence_threshold"))
    frame_interval = float(await get_setting(session, "scene_frame_interval_seconds"))
    min_duration = float(await get_setting(session, "scene_min_duration_seconds"))
    merge_gap = float(await get_setting(session, "scene_merge_gap_seconds"))
    frame_concurrency = int(await get_setting(session, "scene_frame_classify_concurrency"))
    verify_pad = float(await get_setting(session, "scene_verify_pad_seconds"))
    verify_interval = float(await get_setting(session, "scene_verify_frame_interval_seconds"))
    verify_max_pad = float(await get_setting(session, "scene_verify_max_pad_seconds"))
    high_confidence_override = float(await get_setting(session, "scene_high_confidence_single_frame_threshold"))

    src_probe = await probe(app_settings.ffprobe_bin, video_path)
    duration = float(src_probe["format"].get("duration", 0))

    scores = await scan_video_frames(
        video_path,
        app_settings.ffmpeg_bin,
        duration_seconds=duration,
        frame_interval_seconds=frame_interval,
        concurrency=frame_concurrency,
    )
    candidates = cluster_scenes(
        scores,
        confidence_threshold=confidence_threshold,
        frame_interval_seconds=frame_interval,
        merge_gap_seconds=merge_gap,
        high_confidence_override=high_confidence_override,
    )

    refined: list[tuple[float, float]] = []
    for candidate in candidates:
        start, end, _frac, _window_scores = await _refine_candidate_boundary(
            video_path,
            candidate.start,
            candidate.end,
            verify_pad,
            verify_max_pad,
            verify_interval,
            confidence_threshold,
            frame_concurrency,
        )
        if end - start < min_duration:
            mid = (start + end) / 2
            start, end = max(0.0, mid - min_duration / 2), mid + min_duration / 2
        refined.append((start, end))
    return refined, duration


async def run(
    fixture_path: Path,
    season_filter: int | None,
    episode_filter: int | None,
    limit: int | None,
    save_path: Path | None = None,
) -> list[EpisodeScore]:
    fixture = json.loads(fixture_path.read_text())
    # series_title required, not defaulted to "Game of Thrones" -- an older
    # fixture missing this field is a sign it predates multi-show support and
    # needs the field added explicitly, not a silent guess that could scan the
    # wrong show's episodes under a similarly-numbered season/episode.
    series_title = fixture["series_title"]
    results: list[EpisodeScore] = []
    scanned = 0

    def save_partial() -> None:
        # Written after every episode, not just at the end -- a multi-hour run over
        # 25 episodes losing all its output to a crash/interruption near the end
        # would be a real waste; this way whatever finished is always on disk.
        if save_path is not None:
            save_path.write_text(json.dumps([asdict(r) for r in results], indent=2))

    async with get_session() as session:
        merge_gap_seconds = float(await get_setting(session, "scene_merge_gap_seconds"))
        for season_str, episodes in fixture["seasons"].items():
            season = int(season_str)
            if season_filter is not None and season != season_filter:
                continue
            for entry in episodes:
                if entry["windows"] is None:
                    continue
                episode = entry["episode"]
                if episode_filter is not None and episode != episode_filter:
                    continue
                if limit is not None and scanned >= limit:
                    break

                row = await session.execute(
                    select(Title).where(
                        Title.series_title == series_title,
                        Title.season_number == season,
                        Title.episode_number == episode,
                    )
                )
                title = row.scalar_one_or_none()
                if title is None:
                    print(f"{series_title} S{season:02d}E{episode:02d}: no matching Title row in this DB -- skipped")
                    continue

                ground_truth = [tuple(w) for w in entry["windows"]]
                print(f"S{season:02d}E{episode:02d} ({title.display_name}): scanning...", flush=True)
                start_time = time.monotonic()
                try:
                    predicted, duration = await scan_title(session, Path(title.video_path))
                except Exception as e:
                    # A multi-hour, many-episode run losing everything to one bad file
                    # (corrupt source, transient ffmpeg failure) would waste far more
                    # time than skipping it costs -- log and move on, same
                    # fail-gracefully philosophy as the app's own scan_video_frames.
                    print(f"  SCAN FAILED: {e!r} -- skipped", flush=True)
                    continue
                scan_seconds = time.monotonic() - start_time

                matched, missed, false_positives, _raw_boundary_errors = score_episode(ground_truth, predicted)
                boundary_errors = cluster_level_boundary_errors(ground_truth, predicted, merge_gap_seconds)
                mean_boundary_error = sum(boundary_errors) / len(boundary_errors) if boundary_errors else None

                result = EpisodeScore(
                    season=season,
                    episode=episode,
                    title_id=title.id,
                    display_name=title.display_name,
                    duration_seconds=duration,
                    scan_seconds=scan_seconds,
                    ground_truth=[list(w) for w in ground_truth],
                    predicted=[list(w) for w in predicted],
                    matched=matched,
                    missed=[list(w) for w in missed],
                    false_positives=false_positives,
                    mean_boundary_error_seconds=mean_boundary_error,
                )
                results.append(result)
                scanned += 1
                save_partial()
                print(
                    f"  {matched}/{len(ground_truth)} ground-truth windows matched, "
                    f"{false_positives} false positive(s), "
                    f"boundary error {mean_boundary_error:.1f}s avg" if mean_boundary_error is not None
                    else f"  {matched}/{len(ground_truth)} ground-truth windows matched, "
                    f"{false_positives} false positive(s)",
                    flush=True,
                )
                print(f"  scan took {scan_seconds:.0f}s for a {duration:.0f}s video", flush=True)
            if limit is not None and scanned >= limit:
                break

    return results


def print_summary(results: list[EpisodeScore]) -> None:
    total_gt = sum(len(r.ground_truth) for r in results)
    total_matched = sum(r.matched for r in results)
    total_fp = sum(r.false_positives for r in results)
    all_boundary_errors = [
        e for r in results if r.mean_boundary_error_seconds is not None
        for e in [r.mean_boundary_error_seconds]
    ]
    print("\n=== Summary ===")
    print(f"Episodes scanned: {len(results)}")
    recall = total_matched / total_gt if total_gt else float("nan")
    print(f"Recall: {total_matched}/{total_gt} ground-truth windows matched ({recall:.0%})")
    print(f"False positives: {total_fp}")
    if all_boundary_errors:
        print(f"Mean boundary error: {sum(all_boundary_errors) / len(all_boundary_errors):.1f}s")


def rescore(load_path: Path, merge_gap_seconds: float) -> list[EpisodeScore]:
    """Re-applies score_episode to a previously --save'd run's raw predicted/
    ground_truth windows, without re-scanning anything -- lets scoring logic itself
    be iterated on (cheaply, in seconds) against real already-collected data,
    instead of re-paying hours of ffmpeg+NudeNet for every methodology tweak.
    merge_gap_seconds must be passed explicitly (rather than read live from a DB
    session, the way `run` does) since this is a pure/offline re-analysis path --
    pass whatever scene_merge_gap_seconds the original scan actually used (default
    matches DEFAULT_SETTINGS) if you want cluster_level_boundary_errors to reflect
    the same grouping the app itself would have applied."""
    raw = json.loads(load_path.read_text())
    results = []
    for r in raw:
        ground_truth = [tuple(w) for w in r["ground_truth"]]
        predicted = [tuple(w) for w in r["predicted"]]
        matched, missed, false_positives, _raw_boundary_errors = score_episode(ground_truth, predicted)
        boundary_errors = cluster_level_boundary_errors(ground_truth, predicted, merge_gap_seconds)
        mean_boundary_error = sum(boundary_errors) / len(boundary_errors) if boundary_errors else None
        results.append(EpisodeScore(
            season=r["season"], episode=r["episode"], title_id=r["title_id"], display_name=r["display_name"],
            duration_seconds=r["duration_seconds"], scan_seconds=r["scan_seconds"],
            ground_truth=r["ground_truth"], predicted=r["predicted"],
            matched=matched, missed=[list(w) for w in missed], false_positives=false_positives,
            mean_boundary_error_seconds=mean_boundary_error,
        ))
        print(
            f"S{r['season']:02d}E{r['episode']:02d} ({r['display_name']}): "
            f"{matched}/{len(ground_truth)} matched, {false_positives} false positive(s)"
            + (f", boundary error {mean_boundary_error:.1f}s avg" if mean_boundary_error is not None else "")
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", type=Path, default=FIXTURE_PATH,
        help="Ground-truth fixture JSON to scan against (default: the GoT fixture). "
             "Must have a top-level series_title matching the Title rows to benchmark.",
    )
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max episodes to scan this run")
    parser.add_argument("--save", type=Path, default=None, help="Write raw per-episode results as JSON")
    parser.add_argument(
        "--rescore", type=Path, default=None,
        help="Re-score a previously --save'd JSON file's raw windows without re-scanning",
    )
    parser.add_argument(
        "--merge-gap", type=float, default=25.0,
        help="scene_merge_gap_seconds to use for --rescore's cluster-level boundary-error grouping "
             "(default matches DEFAULT_SETTINGS; pass the value the original scan actually used if it differed)",
    )
    args = parser.parse_args()

    if args.rescore is not None:
        results = rescore(args.rescore, args.merge_gap)
        print_summary(results)
        return

    results = asyncio.run(run(args.fixture, args.season, args.episode, args.limit, save_path=args.save))
    print_summary(results)

    if args.save is not None:
        print(f"\nSaved raw results to {args.save} (written incrementally after each episode)")


if __name__ == "__main__":
    main()
