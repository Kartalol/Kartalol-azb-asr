"""South Azerbaijani text normalization.

The default ``published`` profile consolidates the rules in the original
``script/utils/text_norm.py``.  It intentionally does not apply broad Arabic
letter folding: doing so would change the labels used by the paper experiments.
The optional ``canonical`` profile additionally folds presentation variants and
Arabic/Persian forms that are normally equivalent in South Azerbaijani text.
"""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Literal

NormalizationProfile = Literal["none", "published", "canonical"]

_DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"
)
_CANONICAL_TRANSLATION = str.maketrans(
    {
        "ي": "ی",  # Arabic yeh -> Persian yeh
        "ى": "ی",  # alef maksura -> Persian yeh
        "ك": "ک",  # Arabic kaf -> Persian kaf
        "ۀ": "هٔ",
        "ة": "ه",
    }
)
_CONTROL_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_PAREN_RE = re.compile(r"\([^)]*\)")
_GUILLEMET_RE = re.compile(r"«[^»]*»")
_NUMBER_RE = re.compile(r"(?<!\w)\d{1,4}(?!\w)")
_SPACE_RE = re.compile(r"\s+")

_ONES = {
    0: "صفر",
    1: "بیر",
    2: "ایکی",
    3: "اۆچ",
    4: "دؤرد",
    5: "بئش",
    6: "آلتی",
    7: "یئددی",
    8: "سککیز",
    9: "دوْققوز",
}
_TENS = {
    10: "اوْن",
    20: "ایگیرمی",
    30: "اوْتوز",
    40: "قؽرخ",
    50: "اللی",
    60: "آلتمیش",
    70: "یئتمیش",
    80: "سکسن",
    90: "دوْخسان",
}


def number_to_azb(number: int) -> str:
    """Spell an integer from 0 through 9999 using the original AZB lexicon."""

    if not 0 <= number <= 9999:
        raise ValueError("number_to_azb supports integers from 0 through 9999")
    if number < 10:
        return _ONES[number]
    parts: list[str] = []
    thousands, remainder = divmod(number, 1000)
    if thousands:
        if thousands > 1:
            parts.append(_ONES[thousands])
        parts.append("مین")
    hundreds, remainder = divmod(remainder, 100)
    if hundreds:
        if hundreds > 1:
            parts.append(_ONES[hundreds])
        parts.append("یۆز")
    tens, ones = divmod(remainder, 10)
    if tens:
        parts.append(_TENS[tens * 10])
    if ones:
        parts.append(_ONES[ones])
    return " ".join(parts)


def _remove_punctuation(text: str) -> str:
    # Match the original cleaner: ASCII punctuation only. Arabic punctuation is
    # retained because removing it would change the published normalization.
    return text.translate(str.maketrans("", "", string.punctuation))


def normalize_azb(
    text: str,
    *,
    profile: NormalizationProfile = "published",
    expand_numbers: bool = True,
) -> str:
    """Normalize South Azerbaijani text using an explicit, reproducible profile.

    ``none`` only coerces to ``str``. ``published`` reproduces the recoverable
    paper-era cleaner. ``canonical`` additionally applies Unicode NFKC, removes
    directional controls, and folds common Arabic/Persian letter variants.
    """

    if profile == "none":
        return str(text)
    if profile not in {"published", "canonical"}:
        raise ValueError(f"Unknown normalization profile: {profile}")

    value = str(text)
    if profile == "canonical":
        value = unicodedata.normalize("NFKC", value)
        value = _CONTROL_RE.sub("", value)
        value = value.translate(_CANONICAL_TRANSLATION)
    value = _PAREN_RE.sub(" ", value)
    value = _GUILLEMET_RE.sub("", value)
    value = value.translate(_DIGIT_TRANSLATION)
    if expand_numbers:
        value = _NUMBER_RE.sub(lambda match: number_to_azb(int(match.group())), value)
    value = _remove_punctuation(value)
    return _SPACE_RE.sub(" ", value).strip()
