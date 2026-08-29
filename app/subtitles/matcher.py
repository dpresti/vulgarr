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


# Regular English pluralization patterns, checked against a term's own
# normalized text (not the escaped regex) to decide which suffix to add --
# see _add_plural_suffix's docstring for why this exists at all.
_PLURAL_Y_RE = re.compile(r"[^aeiou]y$", re.IGNORECASE)
_PLURAL_ES_RE = re.compile(r"(s|x|z|ch|sh)$", re.IGNORECASE)


def _add_plural_suffix(escaped: str, normalized_term: str) -> str:
    """Extends an already-escaped, already-whitespace-substituted pattern to
    also match the term's regular English plural, hidden from the word list
    UI -- e.g. "asshole" in the list also catches "assholes" without a
    separate entry (a real recall miss this session: Masters of the Universe
    (2025) had "assholes", the list only had the singular).

    Only covers the common regular pluralization patterns, not every
    irregular English plural -- consonant+y -> ies (pussy -> pussies),
    s/x/z/ch/sh -> es (bitch -> bitches, ass -> asses), everything else ->
    plain s (asshole -> assholes) -- since that covers what real single-word
    profanity terms actually follow. Works on the already-escaped string's
    own trailing characters rather than re-deriving escaping rules here:
    re.escape leaves plain ASCII letters like a trailing "y" untouched, so
    slicing it off and replacing it is safe. Only meaningful for whole-word
    matching -- a substring match already matches inside a larger word,
    plural or not, so _compile_term_pattern only calls this in that case."""
    if _PLURAL_Y_RE.search(normalized_term):
        return escaped[:-1] + "(?:y|ies)"
    if _PLURAL_ES_RE.search(normalized_term):
        return escaped + "(?:es)?"
    return escaped + "s?"


def _compile_term_pattern(term: str, whole_word: bool) -> re.Pattern:
    # Normalized the same way match_cue normalizes the cue text it's matched
    # against (dash-to-space, strip punctuation except apostrophes) -- a term
    # entered with a hyphen (e.g. "mother-fucker") previously compiled a
    # pattern requiring a literal hyphen that the normalized cue text (hyphen
    # already converted to a space) could never contain, a silent recall miss.
    normalized = normalize_cue_text(term).strip()
    escaped = re.escape(normalized)
    # Allow the term's internal whitespace to match one-or-more whitespace,
    # so multi-word phrases survive subtitle line-wrapping/double-spacing.
    escaped = escaped.replace(r"\ ", r"\s+")
    if whole_word:
        escaped = _add_plural_suffix(escaped, normalized)
        # (?<!\w)/(?!\w) rather than \b -- \b requires a \w/\W *transition*,
        # which fails for a term ending or starting in an apostrophe (the one
        # punctuation mark normalize_cue_text preserves): apostrophe is \W,
        # so "nothin'" followed by a space (also \W) has no transition and
        # \bnothin'\b could never match, a silent recall miss for any
        # dropped-g/contraction term. The lookaround form only checks what's
        # outside the match, not the match's own edge character, so it
        # behaves identically to \b for ordinary word-char-edged terms while
        # actually working for apostrophe-edged ones.
        pattern = rf"(?<!\w){escaped}(?!\w)"
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
