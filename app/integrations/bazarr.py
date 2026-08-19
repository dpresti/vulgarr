"""Bazarr has no built-in generic webhook -- it has a "Custom Post-Processing"
command hook that runs a shell command with variables substituted after a
subtitle is downloaded. We ask the user to point that command at `curl` hitting
our /webhooks/bazarr endpoint with those variables as query params (documented
in the README). This module also parses that query-param payload, and provides
a client for triggering an on-demand subtitle search+download via Bazarr's own
API (verified against a real running Bazarr 1.6.0 instance's /api/swagger.json,
not guessed -- endpoint names/params vary across Bazarr versions).
"""

import httpx


def parse_subtitle_downloaded_event(params: dict) -> dict | None:
    """params come from Bazarr's Custom Post-Processing variables:
    {{episode}} or {{movie}} -> full path to the video file
    {{subtitles}}            -> full path to the subtitle file that was just downloaded
    {{subtitles_language_code2}} -> e.g. "en"
    """
    video_path = params.get("video_path")
    subtitle_path = params.get("subtitle_path")
    if not video_path or not subtitle_path:
        return None

    return {
        "video_path": video_path,
        "subtitle_path": subtitle_path,
        "language": params.get("language", "en"),
    }


class BazarrClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._headers = {"X-API-KEY": api_key}
        self._timeout = timeout

    async def _patch(self, path: str, params: dict) -> bool:
        """PATCH the given endpoint (Bazarr's "auto-search-and-download the best
        match for this language" action). Returns True if the call succeeded --
        Bazarr's API responds 204 whether or not a matching subtitle was actually
        found, so a True return means "search ran", not "a subtitle was downloaded".
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.patch(f"{self.base_url}{path}", headers=self._headers, params=params)
            return resp.status_code < 400

    async def search_episode_subtitle(
        self, *, series_id: int, episode_id: int, language: str = "en", forced: bool = False, hi: bool = False
    ) -> bool:
        return await self._patch(
            "/api/episodes/subtitles",
            {
                "seriesid": series_id,
                "episodeid": episode_id,
                "language": language,
                "forced": str(forced),
                "hi": str(hi),
            },
        )

    async def search_movie_subtitle(
        self, *, radarr_id: int, language: str = "en", forced: bool = False, hi: bool = False
    ) -> bool:
        return await self._patch(
            "/api/movies/subtitles",
            {"radarrid": radarr_id, "language": language, "forced": str(forced), "hi": str(hi)},
        )
