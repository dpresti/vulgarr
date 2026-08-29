"""Minimal DoesTheDogDie.com (DTDD) client: a cheap crowdsourced signal for whether
a title plausibly has nudity/sexual content, checked before running the much more
expensive NudeNet frame scan.

Schema confirmed via a real authenticated request against the live API (DTDD has no
published spec beyond their own site): GET /dddsearch?q=<title text> returns
{"items": [...]}, each item carrying Radarr/Sonarr-style fields (id, name, releaseYear,
imdbId, ...) plus a `stats` field that is a JSON *string* (not a nested object) shaped
like {"topics": {"<topicId>": {"definitelyYes": 0|1, "definitelyNo": 0|1}, ...}}.
A topic id is only present in this dict once the title has at least one vote on it --
absence means "no data yet", not "no". Topic 197 ("there is sexual content") and 279
("there are nude scenes") are DTDD's own topic ids for exactly what IMDb's Parents
Guide groups under "Sex & Nudity".
"""

import json

import httpx

_SEARCH_URL = "https://www.doesthedogdie.com/dddsearch"
_NUDITY_TOPIC_IDS = ("197", "279")

REPORTED_SUMMARY = "Nudity/sexual content reported"
NOT_REPORTED_SUMMARY = "No nudity/sexual content reported"


class DoesTheDogDieClient:
    def __init__(self, api_key: str, timeout: float = 15.0):
        # Accept is not optional -- DTDD silently 302-redirects a request without it
        # to "/" instead of returning JSON (confirmed live: identical request, only
        # difference being this header, is the difference between a 200 with a JSON
        # body and a 302 redirect).
        self._headers = {"X-API-KEY": api_key, "Accept": "application/json"}
        self._timeout = timeout

    async def search(self, query: str) -> list[dict]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(_SEARCH_URL, headers=self._headers, params={"q": query})
            resp.raise_for_status()
            return resp.json().get("items", [])


def _select_best_item(items: list[dict], *, imdb_id: str | None, year: int | None) -> dict | None:
    if not items:
        return None
    if imdb_id:
        exact = next((it for it in items if it.get("imdbId") == imdb_id), None)
        if exact is not None:
            return exact
    if year is not None:
        year_match = next((it for it in items if str(it.get("releaseYear")) == str(year)), None)
        if year_match is not None:
            return year_match
    # Best-effort fallback -- DTDD's own search ranks its best text match first,
    # and neither signal above matched (or wasn't available).
    return items[0]


def summarize_content_advisory(
    items: list[dict], *, imdb_id: str | None = None, year: int | None = None
) -> tuple[str | None, int | None]:
    """Pure, network-call-free (see DoesTheDogDieClient.search above for the actual
    HTTP call) -- returns (summary, item_id). summary is REPORTED_SUMMARY,
    NOT_REPORTED_SUMMARY, or None if DTDD has no matching title or no votes yet on
    either nudity topic for it. item_id is DTDD's own id for the matched item (for
    linking back to the real page), or None alongside a None summary.

    Favors REPORTED over NOT_REPORTED if a topic somehow carries both flags (shouldn't
    normally happen -- definitelyYes/definitelyNo is already DTDD's own resolved
    per-topic verdict, not a raw vote count) -- matches this app's own established
    bias toward more false positives over a missed nudity scene, since only
    NOT_REPORTED ever gates a scan (see the content-advisory-precheck plan).
    """
    item = _select_best_item(items, imdb_id=imdb_id, year=year)
    if item is None or not item.get("stats"):
        return None, None
    topics = json.loads(item["stats"]).get("topics", {})
    saw_no = False
    for topic_id in _NUDITY_TOPIC_IDS:
        topic = topics.get(topic_id)
        if topic is None:
            continue
        if topic.get("definitelyYes"):
            return REPORTED_SUMMARY, item["id"]
        if topic.get("definitelyNo"):
            saw_no = True
    return (NOT_REPORTED_SUMMARY, item["id"]) if saw_no else (None, None)
