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


def test_plural_s_matches_without_a_separate_wordlist_entry():
    # Real recall miss this session: Masters of the Universe (2025) had
    # "assholes", the word list only had the singular "asshole".
    m = ProfanityMatcher([WordListTerm(term="asshole", severity=Severity.teen)])
    result = m.match_cue(cue("You assholes ruined everything."))
    assert result is not None
    assert result.matched_terms == ("asshole",)


def test_plural_es_matches_for_ch_ending_term():
    m = ProfanityMatcher([WordListTerm(term="bitch", severity=Severity.teen)])
    result = m.match_cue(cue("You bitches are all the same."))
    assert result is not None


def test_plural_ies_matches_for_consonant_y_ending_term():
    m = ProfanityMatcher([WordListTerm(term="pussy", severity=Severity.teen)])
    result = m.match_cue(cue("Don't be such pussies."))
    assert result is not None


def test_singular_still_matches_alongside_plural_support():
    m = ProfanityMatcher([WordListTerm(term="asshole", severity=Severity.teen)])
    result = m.match_cue(cue("You asshole."))
    assert result is not None


def test_plural_matching_does_not_create_false_substring_matches():
    # "asshole" gaining an optional "s?" suffix must not start matching
    # inside unrelated longer words -- whole-word boundaries still apply on
    # both sides of the (now variable-length) match.
    m = ProfanityMatcher([WordListTerm(term="ass", severity=Severity.teen)])
    result = m.match_cue(cue("The assassin passed by."))
    assert result is None


def test_plural_matching_skipped_for_non_whole_word_terms():
    # A substring match already matches inside a plural (or any other) form
    # of a larger word, so there's nothing for plural-suffix logic to add --
    # confirms it doesn't, e.g., double up the match or otherwise misbehave.
    m = ProfanityMatcher([WordListTerm(term="ass", severity=Severity.teen, match_whole_word=False)])
    result = m.match_cue(cue("The classes were long."))
    assert result is not None
    assert result.matched_terms == ("ass",)
