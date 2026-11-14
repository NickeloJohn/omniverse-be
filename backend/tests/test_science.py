from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.models import ScienceObservation, ScienceTarget
from app.utils.mars_coords import geographic_to_local

TARGETS = "/api/v1/science-targets"
OBSERVATIONS = "/api/v1/observations"


@pytest.fixture
def site_id(seeded) -> str:
    return str(seeded.site.id)


def new_target(site_id, **overrides):
    return {"site_id": site_id, "name": "My target", "location": {"x_m": 100.0, "y_m": 200.0}, **overrides}


def test_manual_target_is_proposed_and_distinct_from_seed_data(client, db_session, seeded, site_id):
    response = client.post(TARGETS, json=new_target(
        site_id, description="d", scientific_objective="o", geological_context="g",
        associated_mission="Mars 2020", associated_product_id="prod-1",
        estimated_activity_duration_min=40))
    assert response.status_code == 201
    body = response.json()
    assert body["creation_source"] == "manual" and body["verification_status"] == "proposed"
    assert body["verification_reference"] is None
    assert body["associated_mission"] == "Mars 2020"

    # Geographic metadata is derived server-side and consistent with the local position.
    x, y = geographic_to_local(seeded.site.center_longitude_deg_east, seeded.site.center_latitude_deg,
                               body["longitude_deg_east"], body["latitude_deg"])
    assert (x, y) == pytest.approx((100.0, 200.0), abs=1e-6)

    db_session.expire_all()
    assert db_session.get(ScienceTarget, body["id"]).name == "My target"

    manual = client.get(TARGETS, params={"creation_source": "manual"}).json()
    seeded_only = client.get(TARGETS, params={"creation_source": "demo_seed"}).json()
    assert manual["total"] == 1 and seeded_only["total"] == 3
    assert client.get(TARGETS, params={"verification_status": "nasa_verified"}).json()["total"] == 0


@pytest.mark.parametrize(
    "extra",
    [
        {"verification_status": "nasa_verified"},
        {"creation_source": "nasa_catalog"},
        {"verification_reference": "https://example.com"},
        {"id": str(uuid4())},
    ],
)
def test_clients_cannot_claim_nasa_verification(client, site_id, extra):
    response = client.post(TARGETS, json=new_target(site_id, **extra))
    assert response.status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": ""},
        {"name": "x" * 161},
        {"estimated_activity_duration_min": 0},
        {"estimated_activity_duration_min": 100000},
        {"location": {"x_m": 4001.0, "y_m": 0.0}},
        {"location": {"x_m": 0.0}},
    ],
)
def test_invalid_targets_are_rejected(client, site_id, overrides):
    assert client.post(TARGETS, json=new_target(site_id, **overrides)).status_code == 422


def test_target_unknown_site_and_get_by_id(client, site_id):
    assert client.post(TARGETS, json=new_target(str(uuid4()))).status_code == 404
    created = client.post(TARGETS, json=new_target(site_id)).json()
    assert client.get(f"{TARGETS}/{created['id']}").json() == created
    assert client.get(f"{TARGETS}/{uuid4()}").status_code == 404


def test_targets_filter_by_mission_and_paginate(client, seeded, site_id):
    mission = client.post("/api/v1/missions", json={
        "name": "m", "site_id": site_id, "science_target_ids": [str(seeded.targets[0].id)]}).json()
    by_mission = client.get(TARGETS, params={"mission_id": mission["id"]}).json()
    assert [t["id"] for t in by_mission["items"]] == [str(seeded.targets[0].id)]
    page = client.get(TARGETS, params={"limit": 2, "offset": 2}).json()
    assert page["total"] == 3 and len(page["items"]) == 1


def test_observation_is_recorded_as_manual_simulated(client, db_session, seeded, site_id):
    target_id = str(seeded.targets[0].id)
    response = client.post(OBSERVATIONS, json={
        "site_id": site_id, "target_id": target_id, "observation_type": "geological",
        "title": " Layered rock ", "notes": "n", "measurements": {"grain_size_mm": 0.5, "texture": "fine"}})
    assert response.status_code == 201
    body = response.json()
    assert body["source"] == "manual" and body["data_origin"] == "simulated"
    assert body["title"] == "Layered rock"
    assert body["location"] == {"x_m": 1200.0, "y_m": 800.0}  # inherited from target
    assert body["measurements"] == {"grain_size_mm": 0.5, "texture": "fine"}

    db_session.expire_all()
    assert db_session.scalars(select(ScienceObservation).where(ScienceObservation.id == body["id"])).one()

    listed = client.get(OBSERVATIONS, params={"target_id": target_id}).json()
    assert listed["total"] == 1 and listed["items"][0]["id"] == body["id"]
    assert client.get(OBSERVATIONS).json()["total"] == 2  # + seeded placeholder


@pytest.mark.parametrize("extra", [{"source": "nasa_product"}, {"data_origin": "observed"}, {"eva_session_id": str(uuid4())}])
def test_clients_cannot_forge_nasa_observations(client, site_id, extra):
    payload = {"site_id": site_id, "observation_type": "visual", "title": "t", **extra}
    assert client.post(OBSERVATIONS, json=payload).status_code == 422


def test_observation_validation(client, seeded, site_id):
    base = {"site_id": site_id, "observation_type": "visual", "title": "t"}
    assert client.post(OBSERVATIONS, json={**base, "observation_type": "mystical"}).status_code == 422
    assert client.post(OBSERVATIONS, json={**base, "location": {"x_m": 1e6, "y_m": 0}}).status_code == 422
    assert client.post(OBSERVATIONS, json={**base, "observed_at": "2024-01-01T00:00:00"}).status_code == 422  # naive
    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    assert client.post(OBSERVATIONS, json={**base, "observed_at": future}).status_code == 422
    assert client.post(OBSERVATIONS, json={**base, "target_id": str(uuid4())}).status_code == 404
    assert client.post(OBSERVATIONS, json={**base, "mission_id": str(uuid4())}).status_code == 404
    assert client.post(OBSERVATIONS, json={**base, "site_id": str(uuid4())}).status_code == 404
    too_many = {f"k{i}": 1.0 for i in range(51)}
    assert client.post(OBSERVATIONS, json={**base, "measurements": too_many}).status_code == 422


def test_observation_without_target_or_location_has_no_position(client, site_id):
    body = client.post(OBSERVATIONS, json={"site_id": site_id, "observation_type": "other", "title": "t"}).json()
    assert body["location"] is None and body["latitude_deg"] is None and body["target_id"] is None


def test_unexpected_errors_return_safe_500(client, monkeypatch):
    from app.services import science_service

    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(science_service, "create_science_target", boom)
    response = client.post(TARGETS, json=new_target(str(uuid4())))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error", "message": "An unexpected error occurred"}}
