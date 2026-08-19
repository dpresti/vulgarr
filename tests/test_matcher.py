from app.domain import Severity
from app.subtitles.matcher import ProfanityMatcher, WordListTerm
from app.subtitles.parser import SubtitleCue


def cue(text, i=1, start=0.0, end=1.0):
    return SubtitleCue(index=i, start_seconds=start, end_seconds=end, text=text)


def make_matcher():
    return ProfanityMatcher(
        [
            WordListTerm(term="damn", severity=Severity.teen),
            WordListTerm(term="poop", severity=Severity.child),
            WordListTerm(term="son of a bitch", severity=Severity.teen),
        ]
    )


def test_case_insensitive_match():
    m = make_matcher()
    result = m.match_cue(cue("Oh, DAMN it!"))
    assert result is not None
    assert result.matched_terms == ("damn",)
    assert result.severity == Severity.teen


def test_punctuation_robust_match():
    m = make_matcher()
    result = m.match_cue(cue("Damn!!"))
    assert result is not None


def test_whole_word_boundary_avoids_substring_false_positive():
    m = make_matcher()
    # "damnation" contains "damn" but should not match with word-boundary matching
    result = m.match_cue(cue("It was pure damnation."))
    assert result is None


def test_child_severity_term_matches():
    m = make_matcher()
    result = m.match_cue(cue("Don't step in the poop."))
    assert result is not None
    assert result.severity == Severity.child


def test_multi_word_phrase_survives_line_wrap_double_space():
    m = make_matcher()
    result = m.match_cue(cue("You son of  a bitch."))
    assert result is not None
    assert result.severity == Severity.teen


def test_no_match_returns_none():
    m = make_matcher()
    assert m.match_cue(cue("Everything is fine here.")) is None


def test_multiple_terms_use_highest_severity():
    m = make_matcher()
    result = m.match_cue(cue("Damn, don't step in the poop."))
    assert result is not None
    assert set(result.matched_terms) == {"damn", "poop"}
    assert result.severity == Severity.teen


def test_match_all_preserves_only_matching_cues():
    m = make_matcher()
    cues = [cue("Fine.", i=1), cue("Damn it.", i=2), cue("Still fine.", i=3)]
    results = m.match_all(cues)
    assert len(results) == 1
    assert results[0].cue.index == 2


def test_hyphenated_compound_still_catches_the_root_word():
    # "shit-biscuit" would merge into "shitbiscuit" if hyphens were stripped instead
    # of treated as separators, breaking whole-word matching against "shit".
    m = ProfanityMatcher([WordListTerm(term="shit", severity=Severity.teen)])
    result = m.match_cue(cue("You shit-biscuit!"))
    assert result is not None
    assert result.matched_terms == ("shit",)


def test_hyphenated_double_compound_catches_both_roots():
    m = ProfanityMatcher(
        [WordListTerm(term="shit", severity=Severity.teen), WordListTerm(term="fuck", severity=Severity.teen)]
    )
    result = m.match_cue(cue("Oh, shit-fuck, that hurt."))
    assert result is not None
    assert set(result.matched_terms) == {"shit", "fuck"}
