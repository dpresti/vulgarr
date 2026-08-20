"""Match subtitle cue text against a configurable, severity-tagged word list."""

import re
import string
from dataclasses import dataclass

from app.domain import SEVERITY_RANK, Severity
from app.subtitles.parser import SubtitleCue

# Punctuation that should not break a word for matching purposes (e.g. contractions).
_KEEP_CHARS = "'"
_STRIP_TABLE = str.maketrans("", "", "".join(c for c in string.punctuation if c not in _KEEP_CHARS))

# Hyphens/dashes are treated as word separators, not deleted -- deleting them would
# merge compounds like "shit-biscuit" or "shit-fuck" into one solid token ("shitbiscuit"),
# which breaks whole-word matching against the individual root terms ("shit", "fuck")
# even though those terms are already in the word list.
_DASH_CHARS = "-‐‑‒–—―"
_DASH_TO_SPACE_TABLE = str.maketrans({c: " " for c in _DASH_CHARS})


@dataclass(frozen=True)
class WordListTerm:
    term: str
    severity: Severity
    match_whole_word: bool = True


@dataclass(frozen=True)
class MatchSpan:
    """Character offsets of one matched occurrence within the cue's normalized text --
    used for optional word-level mute precision (proportional position estimate)."""

    term: str
    severity: Severity
    start: int
    end: int


@dataclass(frozen=True)
class CueMatch:
    cue: SubtitleCue
    matched_terms: tuple[str, ...]
    severity: Severity  # highest severity among matched terms
    spans: tuple[MatchSpan, ...] = ()
    normalized_length: int = 0  # length of the normalized text spans are offset against


def normalize_cue_text(text: str) -> str:
    """Lowercase, split hyphenated compounds into separate words, and strip remaining
    punctuation (except apostrophes) for matching. Also used by forced alignment
    (app.audio.forced_align) to build the same word sequence a MatchSpan's character
    offsets are relative to."""
    text = text.translate(_DASH_TO_SPACE_TABLE)
    return text.lower().translate(_STRIP_TABLE)


def _compile_term_pattern(term: str, whole_word: bool) -> re.Pattern:
    escaped = re.escape(term.lower().strip())
    # Allow the term's internal whitespace to match one-or-more whitespace,
    # so multi-word phrases survive subtitle line-wrapping/double-spacing.
    escaped = escaped.replace(r"\ ", r"\s+")
    if whole_word:
        pattern = rf"\b{escaped}\b"
    else:
        pattern = escaped
    return re.compile(pattern, re.IGNORECASE)


class ProfanityMatcher:
    def __init__(self, terms: list[WordListTerm]):
        self._compiled: list[tuple[re.Pattern, WordListTerm]] = [
            (_compile_term_pattern(t.term, t.match_whole_word), t) for t in terms if t.term.strip()
        ]

    def match_cue(self, cue: SubtitleCue) -> CueMatch | None:
        normalized = normalize_cue_text(cue.text)
        matched: list[str] = []
        spans: list[MatchSpan] = []
        best_severity: Severity | None = None

        for pattern, term in self._compiled:
            # finditer, not search -- a term repeated within one cue (e.g. two separate
            # curse words in a long line) needs a span for each occurrence, or word-level
            # mute precision would only narrow around the first one and miss the rest.
            found = list(pattern.finditer(normalized))
            if found:
                matched.append(term.term)
                for m in found:
                    spans.append(MatchSpan(term=term.term, severity=term.severity, start=m.start(), end=m.end()))
                if best_severity is None or SEVERITY_RANK[term.severity] > SEVERITY_RANK[best_severity]:
                    best_severity = term.severity

        if not matched:
            return None
        spans.sort(key=lambda s: s.start)
        return CueMatch(
            cue=cue,
            matched_terms=tuple(matched),
            severity=best_severity,
            spans=tuple(spans),
            normalized_length=len(normalized),
        )

    def match_all(self, cues: list[SubtitleCue]) -> list[CueMatch]:
        results = []
        for cue in cues:
            m = self.match_cue(cue)
            if m is not None:
                results.append(m)
        return results
