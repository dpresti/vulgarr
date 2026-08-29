"""Fixture item dicts below are trimmed real responses (id/imdbId/releaseYear/stats)
captured from a live, authenticated GET /dddsearch call -- see
app/integrations/doesthedogdie.py's module docstring for the confirmed schema."""

import json

from app.integrations.doesthedogdie import (
    NOT_REPORTED_SUMMARY,
    REPORTED_SUMMARY,
    summarize_content_advisory,
)


def _item(item_id: int, imdb_id: str, year: int, topics: dict) -> dict:
    return {"id": item_id, "imdbId": imdb_id, "releaseYear": year, "stats": json.dumps({"topics": topics})}


# Real captured data: A Cure for Wellness (2017) -- both nudity topics voted yes.
_REPORTED_ITEM = _item(
    12881, "tt4731136", 2017, {"197": {"definitelyYes": 1, "definitelyNo": 0}, "279": {"definitelyYes": 1, "definitelyNo": 0}}
)

# Real captured data: Paddington (2014) -- both nudity topics voted no.
_NOT_REPORTED_ITEM = _item(
    10807, "tt1109624", 2014, {"197": {"definitelyYes": 0, "definitelyNo": 1}, "279": {"definitelyYes": 0, "definitelyNo": 1}}
)

# Real captured data: Paddington Goes to School (1986) -- neither nudity topic has
# ever been voted on, so both are absent from stats.topics entirely.
_NO_DATA_ITEM = _item(97857, "tt2053412", 1986, {"153": {"definitelyYes": 0, "definitelyNo": 1}})


def test_reported_when_a_nudity_topic_is_definitely_yes():
    summary, item_id = summarize_content_advisory([_REPORTED_ITEM], imdb_id="tt4731136")
    assert summary == REPORTED_SUMMARY
    assert item_id == 12881


def test_not_reported_when_nudity_topics_are_definitely_no():
    summary, item_id = summarize_content_advisory([_NOT_REPORTED_ITEM], imdb_id="tt1109624")
    assert summary == NOT_REPORTED_SUMMARY
    assert item_id == 10807


def test_no_data_when_nudity_topics_are_unvoted():
    summary, item_id = summarize_content_advisory([_NO_DATA_ITEM], imdb_id="tt2053412")
    assert summary is None
    assert item_id is None


def test_no_data_when_no_items_returned():
    summary, item_id = summarize_content_advisory([], imdb_id="tt0000000")
    assert summary is None
    assert item_id is None


def test_no_data_when_item_has_no_stats():
    summary, item_id = summarize_content_advisory([{"id": 1, "imdbId": "tt1", "releaseYear": 2000}], imdb_id="tt1")
    assert summary is None
    assert item_id is None


def test_prefers_imdb_id_match_over_first_result():
    # First result is a same-named but different title; imdb_id should pick the
    # second one out of the list instead of blindly taking items[0].
    decoy = _item(1, "tt9999999", 2020, {"197": {"definitelyYes": 1, "definitelyNo": 0}})
    summary, item_id = summarize_content_advisory([decoy, _NOT_REPORTED_ITEM], imdb_id="tt1109624")
    assert summary == NOT_REPORTED_SUMMARY
    assert item_id == 10807


def test_falls_back_to_year_match_when_no_imdb_id_given():
    other_year = _item(2, "tt8888888", 1999, {"197": {"definitelyYes": 1, "definitelyNo": 0}})
    summary, item_id = summarize_content_advisory([other_year, _NOT_REPORTED_ITEM], imdb_id=None, year=2014)
    assert summary == NOT_REPORTED_SUMMARY
    assert item_id == 10807


def test_falls_back_to_first_result_when_nothing_matches():
    summary, item_id = summarize_content_advisory([_REPORTED_ITEM], imdb_id=None, year=None)
    assert summary == REPORTED_SUMMARY
    assert item_id == 12881
