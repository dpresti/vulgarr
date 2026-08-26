from app.domain import tags_request_audio, tags_request_video


def test_no_tags_requests_nothing():
    assert tags_request_audio(set()) is False
    assert tags_request_video(set()) is False


def test_audio_tag_requests_only_audio():
    assert tags_request_audio({"vulgarr-audio"}) is True
    assert tags_request_video({"vulgarr-audio"}) is False


def test_video_tag_requests_only_video():
    assert tags_request_audio({"vulgarr-video"}) is False
    assert tags_request_video({"vulgarr-video"}) is True


def test_both_tag_requests_both():
    assert tags_request_audio({"vulgarr-both"}) is True
    assert tags_request_video({"vulgarr-both"}) is True


def test_audio_and_video_tags_together_request_both():
    assert tags_request_audio({"vulgarr-audio", "vulgarr-video"}) is True
    assert tags_request_video({"vulgarr-audio", "vulgarr-video"}) is True


def test_irrelevant_tags_are_ignored():
    tags = {"4k", "anime", "extended"}
    assert tags_request_audio(tags) is False
    assert tags_request_video(tags) is False


def test_relevant_tag_mixed_with_irrelevant_ones_still_matches():
    tags = {"4k", "vulgarr-video", "anime"}
    assert tags_request_audio(tags) is False
    assert tags_request_video(tags) is True


def test_matching_is_case_insensitive():
    assert tags_request_audio({"Vulgarr-Audio"}) is True
    assert tags_request_video({"VULGARR-BOTH"}) is True
