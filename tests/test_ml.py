from ml.common import clean_text, LABELS


def test_clean_text_strips_noise_keeps_negation():
    t = clean_text("@united <b>NOT</b> good!!! http://x.co #delayed sooooo bad")
    assert "http" not in t and "@" not in t and "<b>" not in t
    assert "not good" in t and "delayed" in t and "sooo" in t and "soooo" not in t


def test_clean_text_handles_none_and_numbers():
    assert clean_text(None) == "" and clean_text(123) == "123"


def test_labels_alphabetical():
    assert LABELS == sorted(LABELS)
