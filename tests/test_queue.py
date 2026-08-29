from dataclasses import dataclass

from app.routers.queue import _group_topbar_jobs


@dataclass
class _FakeJob:
    title_id: int
    title: str  # a plain marker value is enough -- grouping never inspects it


def test_group_topbar_jobs_pairs_audio_and_video_for_same_title():
    audio = [_FakeJob(title_id=1, title="A")]
    video = [_FakeJob(title_id=1, title="A")]
    result = _group_topbar_jobs(audio, video)
    assert len(result) == 1
    assert result[0]["audio_job"] is audio[0]
    assert result[0]["video_job"] is video[0]


def test_group_topbar_jobs_keeps_different_titles_separate():
    audio = [_FakeJob(title_id=1, title="A")]
    video = [_FakeJob(title_id=2, title="B")]
    result = _group_topbar_jobs(audio, video)
    assert len(result) == 2
    assert result[0]["audio_job"] is audio[0] and result[0]["video_job"] is None
    assert result[1]["video_job"] is video[0] and result[1]["audio_job"] is None


def test_group_topbar_jobs_preserves_first_appearance_order():
    audio = [_FakeJob(title_id=2, title="B"), _FakeJob(title_id=1, title="A")]
    video = [_FakeJob(title_id=3, title="C")]
    result = _group_topbar_jobs(audio, video)
    assert [r["title"] for r in result] == ["B", "A", "C"]


def test_group_topbar_jobs_empty_inputs():
    assert _group_topbar_jobs([], []) == []
