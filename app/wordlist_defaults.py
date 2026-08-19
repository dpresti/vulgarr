"""Starter word list seeded on first run (only when the word list is empty),
so there's something sensible to start from instead of an empty table.
Fully editable/removable afterward via the Word List page.
"""

from app.domain import Severity

DEFAULT_WORDLIST: list[tuple[str, Severity]] = [
    # mild / scatological
    ("butt", Severity.mild),
    ("poop", Severity.mild),
    ("crap", Severity.mild),
    ("fart", Severity.mild),
    ("freaking", Severity.mild),
    ("dang", Severity.mild),
    ("screw", Severity.mild),
    ("sucks", Severity.mild),
    ("douche", Severity.mild),
    ("badass", Severity.mild),
    # moderate
    ("ass", Severity.moderate),
    ("asshole", Severity.moderate),
    ("bastard", Severity.moderate),
    ("bitch", Severity.moderate),
    ("bullshit", Severity.moderate),
    ("damn", Severity.moderate),
    ("goddamn", Severity.moderate),
    ("hell", Severity.moderate),
    ("dick", Severity.moderate),
    ("piss", Severity.moderate),
    ("prick", Severity.moderate),
    ("whore", Severity.moderate),
    ("slut", Severity.moderate),
    # strong
    ("shit", Severity.strong),
    ("fuck", Severity.strong),
    ("fucking", Severity.strong),
    ("fucked", Severity.strong),
    ("motherfucker", Severity.strong),
    ("cunt", Severity.strong),
    ("cocksucker", Severity.strong),
]
