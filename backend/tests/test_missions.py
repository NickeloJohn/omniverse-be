from uuid import uuid4

import pytest
from sqlalchemy import select

from app.models import ExplorationSite, Mission, MissionTarget

URL = "/api/v1/missions"


@pytest.fixture
def site_id(seeded) -> str:
    return str(seeded.site.id)


def create(client, site_id, **overrides):
    payload = {"name": "  Test traverse  ", "site_id": site_id, **overrides}
    return client.post(URL, json=payload)


def other_site(db_session) -> ExplorationSite:
    from geoalchemy2.shape import from_shape
    from shapely.geometry import box

    from app.services.geo import point_geometry

    site = ExplorationSite(
        slug="other", name="Other", center_longitude_deg_east=10.0, center_latitude_deg=5.0,
        width_m=1000.0, height_m=1000.0, extent=from_shape(box(-500, -500, 500, 500), srid=0),
        habitat_location=point_geometry(0, 0), data_origin="simulated", is_demo=True,
    )
    db_session.add(site)
    db_session.commit()
    return site


def test_create_mission_applies_defaults_and_is_persisted(client, db_session, site_id):
    response = create(client, site_id)
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Test traverse"  # whitespace stripped
    assert body["status"] == "draft" and body["is_demo"] is False
    assert body["start_location"] == {"x_m": -3000.0, "y_m": -2500.0}  # site habitat
    assert body["eva_duration_limit_min"] == 360 and body["return_reserve_min"] == 60
    assert body["constraints"]["max_slope_deg"] == 15.0

    db_session.expire_all()
    stored = db_session.scalars(select(Mission).where(Mission.id == body["id"])).one()
    assert stored.name == "Test traverse" and stored.status.value == "draft"
    assert client.get(f"{URL}/{body['id']}").json() == body


def test_create_mission_with_targets_keeps_order(client, seeded, site_id):
    ids = [str(seeded.targets[2].id), str(seeded.targets[0].id)]
    body = create(client, site_id, science_target_ids=ids).json()
    assert [t["target"]["id"] for t in body["science_targets"]] == ids
    assert [t["sequence"] for t in body["science_targets"]] == [0, 1]


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "   "},
        {"eva_duration_limit_min": 0},
        {"eva_duration_limit_min": 60, "return_reserve_min": 60},
        {"start_location": {"x_m": 5000.0, "y_m": 0.0}},
        {"start_location": {"x_m": "nan", "y_m": 0.0}},
        {"constraints": {"max_slope_deg": 90}},
        {"status": "active"},  # not client-settable on create
        {"science_target_ids": ["3f2c7f55-0000-4000-8000-000000000001"] * 2},
    ],
)
def test_invalid_mission_requests_are_rejected_with_422(client, site_id, overrides):
    response = create(client, site_id, **overrides)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_create_mission_unknown_site_and_target(client, site_id):
    assert client.post(URL, json={"name": "x", "site_id": str(uuid4())}).status_code == 404
    response = create(client, site_id, science_target_ids=[str(uuid4())])
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"


def test_target_from_another_site_is_rejected(client, db_session, seeded, site_id):
    site2 = other_site(db_session)
    response = client.post(URL, json={"name": "x", "site_id": str(site2.id),
                                      "science_target_ids": [str(seeded.targets[0].id)]})
    assert response.status_code == 422


def test_list_missions_filters_and_paginates(client, site_id):
    for i in range(3):
        create(client, site_id, name=f"M{i}")
    page = client.get(URL, params={"limit": 2, "site_id": site_id}).json()
    assert page["total"] == 4 and len(page["items"]) == 2  # 3 new + 1 seeded demo
    drafts = client.get(URL, params={"status": "draft"}).json()
    assert drafts["total"] == 4
    assert client.get(URL, params={"status": "active"}).json()["total"] == 0
    assert client.get(URL, params={"status": "bogus"}).status_code == 422


def test_get_unknown_mission_404(client):
    assert client.get(f"{URL}/{uuid4()}").status_code == 404


def test_patch_updates_fields_and_persists(client, db_session, seeded, site_id):
    mission_id = create(client, site_id).json()["id"]
    response = client.patch(f"{URL}/{mission_id}", json={
        "name": "Renamed", "description": "d", "eva_duration_limit_min": 300,
        "return_reserve_min": 45, "start_location": {"x_m": 100.0, "y_m": -200.0},
        "science_target_ids": [str(seeded.targets[1].id)],
    })
    assert response.status_code == 200
    db_session.expire_all()
    stored = db_session.get(Mission, mission_id)
    assert (stored.name, stored.eva_duration_limit_min, stored.return_reserve_min) == ("Renamed", 300, 45)
    assert [l.target_id for l in stored.target_links] == [seeded.targets[1].id]
    assert stored.updated_at >= stored.created_at
    # Replacing the selection leaves no orphan links behind.
    assert db_session.scalars(select(MissionTarget).where(MissionTarget.mission_id == stored.id)).all() == stored.target_links


def test_patch_rejects_null_for_required_fields_and_bad_values(client, site_id):
    mission_id = create(client, site_id).json()["id"]
    assert client.patch(f"{URL}/{mission_id}", json={"name": None}).status_code == 422
    assert client.patch(f"{URL}/{mission_id}", json={"site_id": str(uuid4())}).status_code == 422
    # reserve is validated against the stored limit as well
    assert client.patch(f"{URL}/{mission_id}", json={"return_reserve_min": 360}).status_code == 422


def test_status_lifecycle_and_invalid_transitions(client, site_id):
    mission_id = create(client, site_id).json()["id"]
    path = f"{URL}/{mission_id}"

    invalid = client.patch(path, json={"status": "active"})  # draft -> active not allowed
    assert invalid.status_code == 409 and "allowed" in invalid.json()["error"]["message"]

    for status in ("planned", "active", "paused", "active", "completed"):
        assert client.patch(path, json={"status": status}).json()["status"] == status

    terminal = client.patch(path, json={"name": "nope"})
    assert terminal.status_code == 409


def test_plan_fields_are_frozen_once_active(client, site_id):
    mission_id = create(client, site_id).json()["id"]
    path = f"{URL}/{mission_id}"
    client.patch(path, json={"status": "planned"})
    client.patch(path, json={"status": "active"})
    assert client.patch(path, json={"eva_duration_limit_min": 400}).status_code == 409
    assert client.patch(path, json={"name": "still editable"}).status_code == 200


def test_failed_update_leaves_no_partial_changes(client, db_session, site_id):
    mission_id = create(client, site_id).json()["id"]
    bad = client.patch(f"{URL}/{mission_id}", json={"status": "planned", "start_location": {"x_m": 9e6, "y_m": 0}})
    assert bad.status_code == 422
    db_session.expire_all()
    assert client.get(f"{URL}/{mission_id}").json()["status"] == "draft"


def test_delete_only_draft_or_aborted(client, site_id):
    draft_id = create(client, site_id).json()["id"]
    assert client.delete(f"{URL}/{draft_id}").status_code == 204
    assert client.get(f"{URL}/{draft_id}").status_code == 404

    active_id = create(client, site_id).json()["id"]
    client.patch(f"{URL}/{active_id}", json={"status": "planned"})
    assert client.delete(f"{URL}/{active_id}").status_code == 409
    client.patch(f"{URL}/{active_id}", json={"status": "aborted"})
    assert client.delete(f"{URL}/{active_id}").status_code == 204
    assert client.delete(f"{URL}/{uuid4()}").status_code == 404
