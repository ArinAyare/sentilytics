import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.repository import MemoryRepo  # noqa: E402
from tests.test_services import StubPredictor  # noqa: E402


class P(StubPredictor):
    primary, secondary = object(), object()


@pytest.fixture
def client():
    with TestClient(app) as c:
        app.state.repo, app.state.predictor = MemoryRepo(), P()
        yield c


def test_predict_and_history(client):
    r = client.post("/api/predict", json={"text": "really good"})
    assert r.status_code == 200 and r.json()["label"] == "positive"
    assert client.get("/api/history").json()["total"] == 1


def test_predict_empty_rejected(client):
    assert client.post("/api/predict", json={"text": ""}).status_code == 422


def test_batch_flow_and_download(client):
    r = client.post("/api/batch", files={"file": ("a.csv", b"text\ngood\nbad\n", "text/csv")})
    assert r.status_code == 200
    jid = r.json()["id"]
    assert client.get(f"/api/jobs/{jid}/download").text.count("\n") == 3
    assert client.get("/api/analytics").json()["total"] == 2


def test_batch_rejects_non_csv(client):
    assert client.post("/api/batch", files={"file": ("a.txt", b"x", "text/plain")}).status_code == 415
