"""Starter word list seeded on first run (only when the word list is empty),
so there's something sensible to start from instead of an empty table.
Fully editable/removable afterward via the Word List page.
"""

from app.domain import Severity

DEFAULT_WORDLIST: list[tuple[str, Severity]] = [
    # child / scatological -- muted only on the Child track (Teen track lets these through)
    ("butt", Severity.child),
    ("poop", Severity.child),
    ("crap", Severity.child),
    ("fart", Severity.child),
    ("freaking", Severity.child),
    ("dang", Severity.child),
    ("screw", Severity.child),
    ("sucks", Severity.child),
    ("douche", Severity.child),
    ("badass", Severity.child),
    # teen -- muted on both the Child and Teen tracks (old moderate + strong profanity)
    ("ass", Severity.teen),
    ("asshole", Severity.teen),
    ("bastard", Severity.teen),
    ("bitch", Severity.teen),
    ("bullshit", Severity.teen),
    ("damn", Severity.teen),
    ("goddamn", Severity.teen),
    ("hell", Severity.teen),
    ("dick", Severity.teen),
    ("piss", Severity.teen),
    ("prick", Severity.teen),
    ("whore", Severity.teen),
    ("slut", Severity.teen),
    ("shit", Severity.teen),
    ("fuck", Severity.teen),
    ("fucking", Severity.teen),
    ("fucked", Severity.teen),
    ("motherfucker", Severity.teen),
    ("cunt", Severity.teen),
    ("cocksucker", Severity.teen),
]
