"""Minimal Radarr v3 API client: enough to list the library and resolve file paths."""

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class RadarrMovieFile:
    movie_id: int
    title: str
    year: int
    path: str
    poster_url: str | None = None

    @property
    def display_name(self) -> str:
        return f"{self.title} ({self.year})"


def _extract_poster_url(images: list[dict]) -> str | None:
    """Images come back with both a Radarr-local `url` (only reachable from inside the
    Radarr container, and only with the API key) and a `remoteUrl` pointing at the
    original TMDB CDN -- the latter is what a browser can actually load directly."""
    return next((img["remoteUrl"] for img in images if img.get("coverType") == "poster" and img.get("remoteUrl")), None)


class RadarrClient:
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

    async def trigger_movie_search(self, movie_id: int) -> bool:
        """Trigger an indexer search for a replacement release of this movie (command
        name/payload verified against Radarr v3's real MoviesSearchCommand class)."""
        await self._post("/api/v3/command", json={"name": "MoviesSearch", "movieIds": [movie_id]})
        return True

    async def get_current_movie_file_path(self, movie_id: int) -> str | None:
        """Re-fetch the movie's *current* file path -- may have changed since we last
        recorded it, e.g. after a search grabs and imports a replacement release."""
        movie = await self._get(f"/api/v3/movie/{movie_id}")
        movie_file = movie.get("movieFile")
        return movie_file.get("path") if movie_file else None

    async def get_tag_labels(self) -> dict[int, str]:
        """All Radarr tags, keyed by id, lowercased (Radarr already lowercases labels
        on creation -- this is cheap insurance, not a correction)."""
        tags = await self._get("/api/v3/tag")
        return {t["id"]: t["label"].lower() for t in tags}

    async def get_movie_tag_labels(self, movie_id: int) -> set[str]:
        """Resolve a movie's tag IDs to lowercased label strings. Two GETs (movie +
        full tag list) -- not cached, since this only ever runs off a low-volume
        import-webhook event, not a hot path."""
        movie = await self._get(f"/api/v3/movie/{movie_id}")
        tag_ids: list[int] = movie.get("tags") or []
        if not tag_ids:
            return set()
        labels_by_id = await self.get_tag_labels()
        return {labels_by_id[tid] for tid in tag_ids if tid in labels_by_id}

    async def list_movie_files(self) -> list[RadarrMovieFile]:
        movies = await self._get("/api/v3/movie")
        results: list[RadarrMovieFile] = []
        for movie in movies:
            movie_file = movie.get("movieFile")
            if not movie.get("hasFile") or not movie_file:
                continue
            results.append(
                RadarrMovieFile(
                    movie_id=movie["id"],
                    title=movie["title"],
                    year=movie.get("year", 0),
                    path=movie_file["path"],
                    poster_url=_extract_poster_url(movie.get("images", [])),
                )
            )
        return results


def parse_import_webhook(payload: dict) -> dict | None:
    """Extract the fields we need from a Radarr 'On Import'/'On Upgrade' webhook payload."""
    if payload.get("eventType") not in ("Download", "Upgrade"):
        return None
    movie = payload.get("movie") or {}
    movie_file = payload.get("movieFile") or {}
    if not movie_file.get("path"):
        return None

    return {
        "movie_id": movie.get("id"),
        "display_name": f"{movie.get('title', 'Unknown')} ({movie.get('year', '')})",
        "video_path": movie_file["path"],
        # The webhook payload's "movie" object carries the same images array as the
        # REST API's Movie resource, so this needs no extra API call -- but a new
        # Title row created straight off the webhook (see routers/webhooks.py) was
        # never getting a poster at all, since this went unextracted here and the
        # webhook handler had nothing to pass upsert_title.
        "poster_url": _extract_poster_url(movie.get("images", [])),
    }
