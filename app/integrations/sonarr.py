"""Minimal Sonarr v3 API client: enough to list the library and resolve file paths."""

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class SonarrEpisodeFile:
    series_id: int
    episode_id: int
    series_title: str
    season_number: int
    episode_number: int
    episode_title: str
    path: str
    poster_url: str | None = None

    @property
    def display_name(self) -> str:
        return f"{self.series_title} - S{self.season_number:02d}E{self.episode_number:02d} - {self.episode_title}"


def _extract_poster_url(images: list[dict]) -> str | None:
    """See the identical helper in app.integrations.radarr -- same reasoning, different
    CDN (TheTVDB instead of TMDB)."""
    return next((img["remoteUrl"] for img in images if img.get("coverType") == "poster" and img.get("remoteUrl")), None)


class SonarrClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._headers = {"X-Api-Key": api_key}
        self._timeout = timeout

    async def _get(self, path: str, params: dict | None = None) -> list | dict:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self.base_url}{path}", headers=self._headers, params=params)
            resp.raise_for_status()
            return resp.json()

    async def _post(self, path: str, json: dict) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self.base_url}{path}", headers=self._headers, json=json)
            resp.raise_for_status()
            return resp.json()

    async def get_tag_labels(self) -> dict[int, str]:
        """All Sonarr tags, keyed by id, lowercased (Sonarr already lowercases labels
        on creation -- this is cheap insurance, not a correction)."""
        tags = await self._get("/api/v3/tag")
        return {t["id"]: t["label"].lower() for t in tags}

    async def get_series_tag_labels(self, series_id: int) -> set[str]:
        """Resolve a series' tag IDs to lowercased label strings. Two GETs (series +
        full tag list) -- not cached, since this only ever runs off a low-volume
        import-webhook event, not a hot path."""
        series = await self._get(f"/api/v3/series/{series_id}")
        tag_ids: list[int] = series.get("tags") or []
        if not tag_ids:
            return set()
        labels_by_id = await self.get_tag_labels()
        return {labels_by_id[tid] for tid in tag_ids if tid in labels_by_id}

    async def list_episode_files(self) -> list[SonarrEpisodeFile]:
        """List every downloaded episode across the whole library, with resolved file paths."""
        series_list = await self._get("/api/v3/series")
        results: list[SonarrEpisodeFile] = []

        for series in series_list:
            series_id = series["id"]
            poster_url = _extract_poster_url(series.get("images", []))
            episodes = await self._get("/api/v3/episode", params={"seriesId": series_id})
            files_by_id = {
                f["id"]: f for f in await self._get("/api/v3/episodefile", params={"seriesId": series_id})
            }
            for ep in episodes:
                file_id = ep.get("episodeFileId")
                if not file_id or file_id not in files_by_id:
                    continue
                results.append(
                    SonarrEpisodeFile(
                        series_id=series_id,
                        episode_id=ep["id"],
                        series_title=series["title"],
                        season_number=ep["seasonNumber"],
                        episode_number=ep["episodeNumber"],
                        episode_title=ep.get("title", ""),
                        path=files_by_id[file_id]["path"],
                        poster_url=poster_url,
                    )
                )
        return results

    async def get_episode_file_path(self, episode_file_id: int) -> str:
        data = await self._get(f"/api/v3/episodefile/{episode_file_id}")
        return data["path"]

    async def trigger_episode_search(self, episode_id: int) -> bool:
        """Trigger an indexer search for a replacement release of this episode (command
        name/payload verified against Sonarr v3's real EpisodeSearchCommand class)."""
        await self._post("/api/v3/command", json={"name": "EpisodeSearch", "episodeIds": [episode_id]})
        return True

    async def get_current_episode_file_path(self, episode_id: int) -> str | None:
        """Re-fetch the episode's *current* file path -- may have changed since we last
        recorded it, e.g. after a search grabs and imports a replacement release."""
        episode = await self._get(f"/api/v3/episode/{episode_id}")
        file_id = episode.get("episodeFileId")
        if not file_id:
            return None
        return await self.get_episode_file_path(file_id)


def parse_import_webhook(payload: dict) -> dict | None:
    """Extract the fields we need from a Sonarr 'On Import'/'On Upgrade' webhook payload."""
    if payload.get("eventType") not in ("Download", "Upgrade"):
        return None
    series = payload.get("series") or {}
    episodes = payload.get("episodes") or []
    episode_file = payload.get("episodeFile") or {}
    if not episodes or not episode_file.get("path"):
        return None

    ep = episodes[0]
    return {
        "series_id": series.get("id"),
        "episode_id": ep.get("id"),
        "series_title": series.get("title", "Unknown"),
        "season_number": ep.get("seasonNumber", 0),
        "episode_number": ep.get("episodeNumber", 0),
        "display_name": f"{series.get('title', 'Unknown')} - "
        f"S{ep.get('seasonNumber', 0):02d}E{ep.get('episodeNumber', 0):02d}",
        "video_path": episode_file["path"],
        # Same reasoning as radarr.parse_import_webhook's poster_url -- the payload's
        # "series" object already carries the images array, no extra API call needed.
        "poster_url": _extract_poster_url(series.get("images", [])),
    }
