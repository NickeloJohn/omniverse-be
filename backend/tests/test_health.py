from fastapi.testclient import TestClient

from app.api.v1 import health
from app.main import app

client = TestClient(app)


def test_health_reports_ready_when_database_is_available(monkeypatch) -> None:
    monkeypatch.setattr(health, "database_is_available", lambda: True)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": True}


def test_health_reports_degraded_when_database_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(health, "database_is_available", lambda: False)

    response = client.get("/api/v1/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "database": False}


def test_openapi_docs_are_available() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/v1/health" in response.json()["paths"]