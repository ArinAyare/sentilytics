"""Shared text utilities and constants. Used by training, Azure scoring and the backend,
so that training-time and inference-time preprocessing are IDENTICAL."""
import re

LABELS = ["negative", "neutral", "positive"]  # alphabetical == sklearn classes_ order

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HTML = re.compile(r"<[^>]+>")
_HASHTAG = re.compile(r"#(\w+)")
_REPEAT = re.compile(r"(.)\1{3,}")          # sooooo -> sooo
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """Light cleaning. Negations and punctuation (! ?) are deliberately KEPT because they carry sentiment."""
    if text is None:
        return ""
    t = str(text).lower()
    t = _HTML.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _HASHTAG.sub(r"\1", t)
    t = _REPEAT.sub(r"\1\1\1", t)
    t = re.sub(r"[^a-z0-9\s'!?.,]", " ", t)
    return _SPACES.sub(" ", t).strip()
