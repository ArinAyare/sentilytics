import pytest

from app import services
from app.repository import MemoryRepo


class StubPredictor:
    def predict_with_source(self, texts):
        out = []
        for t in texts:
            lab = "positive" if "good" in t.lower() else "negative" if "bad" in t.lower() else "neutral"
            probs = {"negative": 0.1, "neutral": 0.1, "positive": 0.1}; probs[lab] = 0.8
            out.append({"label": lab, "confidence": 0.8, "probabilities": probs})
        return out, "stub"


@pytest.fixture
def ctx():
    return MemoryRepo(), StubPredictor()


def test_analyze_text_saves_history(ctx):
    repo, pred = ctx
    d = services.analyze_text("  This is good  ", repo, pred)
    assert d["label"] == "positive" and repo.count() == 1 and d["text"] == "This is good"


@pytest.mark.parametrize("bad", ["", "   ", None, "x" * 6000])
def test_validation_rejects(ctx, bad):
    with pytest.raises(services.ValidationError):
        services.analyze_text(bad, *ctx)


def test_csv_autodetect_and_counts(ctx):
    repo, pred = ctx
    csv = b"id,review\n1,good stuff\n2,bad stuff\n3,whatever\n4,\n"
    job = services.analyze_csv(csv, "r.csv", repo, pred, max_rows=100)
    assert job["text_column"] == "review" and job["total_rows"] == 3
    assert job["counts"] == {"negative": 1, "neutral": 1, "positive": 1}


def test_csv_row_cap_and_missing_column(ctx):
    repo, pred = ctx
    csv = "text\n" + "\n".join(f"good {i}" for i in range(10))
    job = services.analyze_csv(csv.encode(), "r.csv", repo, pred, max_rows=4)
    assert job["total_rows"] == 4 and job["truncated"]
    with pytest.raises(services.ValidationError):
        services.analyze_csv(b"a,b\n1,2\n", "r.csv", repo, pred, 10, text_column="nope")


def test_export_neutralises_formula_injection(ctx):
    repo, pred = ctx
    job = services.analyze_csv(b"text\n=HYPERLINK(\"evil\") good\n", "r.csv", repo, pred, 10)
    out = services.export_csv(job["id"], repo)
    assert "'=HYPERLINK" in out and out.splitlines()[0].startswith("text,sentiment")


def test_analytics_percentages(ctx):
    repo, pred = ctx
    for t in ["good", "good", "bad", "meh"]:
        services.analyze_text(t, repo, pred)
    a = services.analytics(repo)
    assert a["total"] == 4 and a["percentages"]["positive"] == 50.0 and len(a["trend"]) == 1
