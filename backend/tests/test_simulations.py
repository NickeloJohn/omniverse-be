from datetime import UTC, datetime
from uuid import uuid4

import pytest
from geoalchemy2.shape import from_shape
from shapely.geometry import LineString
from sqlalchemy import func, select

from app.models import EVASession, Mission, Route, TelemetrySample
from app.models.enums import DataOrigin, EVAStatus, MissionStatus, RouteProfile, RouteStatus
from app.schemas.simulation import ResourceModel

SIMS = "/api/v1/simulations"


@pytest.fixture
def route_fixture(db_session, seeded):
    mission = Mission(
        name="EVA test mission", site_id=seeded.site.id, status=MissionStatus.PLANNED,
        start_location=seeded.site.habitat_location,
        start_longitude_deg_east=seeded.site.center_longitude_deg_east,
        start_latitude_deg=seeded.site.center_latitude_deg,
        eva_duration_limit_min=60, return_reserve_min=5,
        constraints={"max_slope_deg": 15, "block_unknown_terrain": True, "preferred_profile": "safety_first"},
    )
    db_session.add(mission)
    db_session.flush()
    stops = [
        {"sequence": 0, "waypoint_type": "start", "label": "Habitat", "location": {"x_m": 0.0, "y_m": 0.0},
         "science_target_id": None, "activity_duration_min": 0, "distance_from_start_m": 0.0,
         "arrival_time_s": 0.0, "departure_time_s": 0.0, "risk_score": 0.1},
        {"sequence": 1, "waypoint_type": "science_target", "label": "Sample site", "location": {"x_m": 20.0, "y_m": 0.0},
         "science_target_id": str(seeded.targets[0].id), "activity_duration_min": 10, "distance_from_start_m": 20.0,
         "arrival_time_s": 20.0, "departure_time_s": 30.0, "risk_score": 0.7},
        {"sequence": 2, "waypoint_type": "habitat_return", "label": "Habitat", "location": {"x_m": 20.0, "y_m": 10.0},
         "science_target_id": None, "activity_duration_min": 0, "distance_from_start_m": 30.0,
         "arrival_time_s": 50.0, "departure_time_s": 50.0, "risk_score": 0.2},
    ]
    details = {
        "profile": "safety_first", "status": "found", "no_route": None,
        "metrics": {"distance_m": 30.0, "travel_time_s": 40.0, "activity_time_s": 10.0,
                    "total_time_s": 50.0, "total_cost": 60.0, "risk_score": 0.4, "peak_risk": 0.9,
                    "mean_slope_deg": 3.0, "max_slope_deg": 7.0, "elevation_gain_m": 1.0,
                    "elevation_loss_m": 0.0, "high_risk_distance_m": 10.0, "uncertain_distance_m": 2.0},
        "stops": stops,
        "constraints": {"eva_duration_limit_min": 60, "return_reserve_min": 5, "usable_time_s": 3300.0,
                        "within_time_budget": True, "return_planned": True, "max_slope_deg_applied": 15.0,
                        "block_unknown_terrain_applied": True, "violations": []},
        "legs": [], "search": {"expanded_nodes": 4, "elapsed_ms": 1.0},
        "cost_model": {"routing_config_version": "test"}, "data_origin": "simulated", "warnings": [],
    }
    route = Route(
        mission_id=mission.id, profile=RouteProfile.SAFETY_FIRST, status=RouteStatus.FOUND,
        is_selected=True, geometry=from_shape(LineString([(0, 0), (10, 0), (20, 0), (20, 10)]), srid=0),
        distance_m=30.0, estimated_duration_s=40.0, total_cost=60.0, risk_score=0.4,
        max_cell_risk=0.9, mean_slope_deg=3.0, max_slope_deg=7.0,
        elevation_gain_m=1.0, elevation_loss_m=0.0, dataset_version="test-v1",
        cost_model={"details": details, "path_samples": [
            {"distance_m": 0.0, "risk_score": 0.1, "slope_deg": 1.0},
            {"distance_m": 10.0, "risk_score": 0.9, "slope_deg": 7.0},
            {"distance_m": 20.0, "risk_score": 0.7, "slope_deg": 5.0},
            {"distance_m": 30.0, "risk_score": 0.2, "slope_deg": 2.0},
        ]}, data_origin=DataOrigin.SIMULATED,
    )
    db_session.add(route)
    db_session.commit()
    return mission, route


def start(client, route_id, resource_model=None):
    payload = {"route_id": str(route_id)}
    if resource_model is not None:
        payload["resource_model"] = resource_model
    return client.post(SIMS, json=payload)


def advance(client, session_id, seconds):
    return client.post(f"{SIMS}/{session_id}/advance", json={"advance_by_s": seconds})


def test_start_creates_persistent_simulated_session_and_initial_sample(client, db_session, route_fixture):
    mission, route = route_fixture
    response = start(client, route.id)
    assert response.status_code == 201, response.text
    body = response.json()
    state = body["session"]
    assert state["status"] == "running" and state["mission_id"] == str(mission.id) and state["route_id"] == str(route.id)
    assert state["sim_elapsed_s"] == 0 and state["distance_along_route_m"] == 0
    assert state["position"] == {"x_m": 0.0, "y_m": 0.0}
    assert state["oxygen_remaining_pct"] == state["battery_remaining_pct"] == 100
    assert state["telemetry_count"] == 1 and state["latest_telemetry"]["sim_time_s"] == 0
    assert state["data_origin"] == "simulated" and state["latest_telemetry"]["data_origin"] == "simulated"
    assert "authoritative clock" in body["message"]

    eva = db_session.get(EVASession, state["id"])
    assert eva and eva.status == EVAStatus.RUNNING and eva.data_origin == DataOrigin.SIMULATED
    assert eva.started_at is not None and eva.resource_model["version"] == "prototype-resources-v1"
    assert db_session.get(Mission, mission.id).status == MissionStatus.ACTIVE
    sample = db_session.scalars(select(TelemetrySample).where(TelemetrySample.session_id == eva.id)).one()
    assert sample.data_origin == DataOrigin.SIMULATED and sample.risk_level == "low"


def test_advance_interpolates_actual_saved_geometry_and_current_risk(client, db_session, route_fixture):
    mission, route = route_fixture
    session = start(client, route.id).json()["session"]
    response = advance(client, session["id"], 10.0)
    assert response.status_code == 200, response.text
    state, telemetry = response.json()["session"], response.json()["telemetry"]
    assert state["sim_elapsed_s"] == 10.0 and state["distance_along_route_m"] == pytest.approx(10.0)
    assert state["position"] == {"x_m": 10.0, "y_m": 0.0}                 # exact midpoint along stored line
    assert telemetry["speed_mps"] == pytest.approx(1.0)
    assert state["current_risk_score"] == pytest.approx(0.9) and state["risk_level"] == "critical"
    assert any("Critical planned terrain risk" in warning for warning in state["warnings"])
    assert state["estimated_return_time_s"] == pytest.approx(40.0)
    assert telemetry["data_origin"] == "simulated"


def test_logical_clock_holds_position_during_science_activity(client, route_fixture):
    _, route = route_fixture
    session_id = start(client, route.id).json()["session"]["id"]
    first = advance(client, session_id, 20.0).json()["session"]
    assert first["position"] == {"x_m": 20.0, "y_m": 0.0}
    assert first["distance_along_route_m"] == 20.0
    assert first["sim_elapsed_s"] == 20.0
    during = advance(client, session_id, 5.0).json()["session"]
    assert during["position"] == first["position"] and during["distance_along_route_m"] == 20.0
    assert during["sim_elapsed_s"] == 25.0 and during["estimated_return_time_s"] == 25.0
    assert during["oxygen_remaining_pct"] == pytest.approx(100 - (12 * 20 + 15 * 5) / 3600)


def test_resource_rates_are_explicit_configurable_and_deterministic(client, db_session, route_fixture):
    _, route = route_fixture
    custom = ResourceModel(
        version="test-resource-v1", oxygen_use_pct_per_hour_moving=36,
        oxygen_use_pct_per_hour_activity=72, battery_use_pct_per_hour_moving=18,
        battery_use_pct_per_hour_activity=36, oxygen_warning_pct=99.9, battery_warning_pct=99.99,
        risk_warning_threshold=0.95, risk_critical_threshold=0.99,
    )
    state = start(client, route.id, custom.model_dump()).json()["session"]
    one = advance(client, state["id"], 15).json()["session"]
    assert one["oxygen_remaining_pct"] == pytest.approx(100 - 36 * 15 / 3600)
    assert one["battery_remaining_pct"] == pytest.approx(100 - 18 * 15 / 3600)
    assert one["resource_model"]["version"] == "test-resource-v1"
    assert any("oxygen reserve is low" in w for w in one["warnings"])
    assert any("battery reserve is low" in w for w in one["warnings"])


def test_clock_chunking_does_not_change_position_or_resource_state(client, route_fixture):
    _, route = route_fixture
    a = start(client, route.id).json()["session"]["id"]
    advance(client, a, 17)
    state_a = advance(client, a, 9).json()["session"]
    client.post(f"{SIMS}/{a}/reset")
    b = start(client, route.id).json()["session"]["id"]
    state_b = advance(client, b, 26).json()["session"]
    for key in ("sim_elapsed_s", "distance_along_route_m", "position", "oxygen_remaining_pct", "battery_remaining_pct", "estimated_return_time_s"):
        assert state_a[key] == pytest.approx(state_b[key]) if isinstance(state_a[key], (int, float)) else state_a[key] == state_b[key]


def test_pause_resume_freezes_the_explicit_clock_and_syncs_mission_status(client, db_session, route_fixture):
    mission, route = route_fixture
    state = start(client, route.id).json()["session"]
    session_id = state["id"]
    advance(client, session_id, 10)
    paused = client.post(f"{SIMS}/{session_id}/pause")
    assert paused.status_code == 200 and paused.json()["status"] == "paused"
    assert db_session.get(Mission, mission.id).status == MissionStatus.PAUSED
    assert advance(client, session_id, 10).status_code == 409
    assert client.post(f"{SIMS}/{session_id}/pause").status_code == 409
    resumed = client.post(f"{SIMS}/{session_id}/resume")
    assert resumed.status_code == 200 and resumed.json()["status"] == "running"
    assert db_session.get(Mission, mission.id).status == MissionStatus.ACTIVE
    result = advance(client, session_id, 10).json()
    assert result["session"]["sim_elapsed_s"] == 20 and result["session"]["position"] == {"x_m": 20.0, "y_m": 0.0}


def test_advance_clamps_at_route_completion_and_completes_mission(client, db_session, route_fixture):
    mission, route = route_fixture
    session_id = start(client, route.id).json()["session"]["id"]
    response = advance(client, session_id, 1000)
    assert response.status_code == 200
    body = response.json()
    assert body["advanced_by_s"] == 50 and body["completed"] is True
    assert body["session"]["status"] == "completed" and body["session"]["route_progress_pct"] == 100
    assert body["session"]["position"] == {"x_m": 20.0, "y_m": 10.0}
    assert body["session"]["oxygen_remaining_pct"] == pytest.approx(100 - (12 * 40 + 15 * 10) / 3600)
    assert db_session.get(Mission, mission.id).status == MissionStatus.COMPLETED
    assert advance(client, session_id, 1).status_code == 409


def test_return_time_check_warns_when_remaining_eva_time_is_too_short(client, db_session, route_fixture):
    mission, route = route_fixture
    mission.eva_duration_limit_min = 1
    mission.return_reserve_min = 0
    details = dict(route.cost_model["details"])
    details["stops"] = [dict(stop) for stop in details["stops"]]
    details["stops"][1]["departure_time_s"] = 60.0
    details["stops"][1]["activity_duration_min"] = 40.0
    details["stops"][2]["arrival_time_s"] = 80.0
    details["stops"][2]["departure_time_s"] = 80.0
    details["metrics"] = dict(details["metrics"], activity_time_s=40.0, total_time_s=80.0)
    route.cost_model = {**route.cost_model, "details": details}
    db_session.commit()
    state = start(client, route.id).json()["session"]
    assert any("Return-time check" in w for w in state["warnings"])
    assert state["estimated_return_time_s"] == 80
    advanced = advance(client, state["id"], 20).json()["session"]
    assert advanced["estimated_return_time_s"] == 60
    assert any("Return-time check" in w for w in advanced["warnings"])


def test_no_return_leg_uses_explicit_retrace_estimate_warning(client, db_session, route_fixture):
    mission, route = route_fixture
    details = dict(route.cost_model["details"])
    details["stops"][-1]["waypoint_type"] = "science_target"
    route.cost_model = {**route.cost_model, "details": details}
    db_session.commit()
    session_id = start(client, route.id).json()["session"]["id"]
    state = advance(client, session_id, 20).json()["session"]
    assert state["estimated_return_time_s"] == pytest.approx(20.0 * 40.0 / 30.0)
    assert any("based on retracing" in w for w in state["warnings"])


def test_reset_clears_telemetry_and_restores_initial_state(client, db_session, route_fixture):
    mission, route = route_fixture
    session_id = start(client, route.id).json()["session"]["id"]
    advance(client, session_id, 20)
    assert client.get(f"{SIMS}/{session_id}/telemetry").json()["total"] == 2
    reset = client.post(f"{SIMS}/{session_id}/reset")
    assert reset.status_code == 200, reset.text
    body = reset.json()
    state = body["session"]
    assert state["status"] == "created" and state["sim_elapsed_s"] == state["distance_along_route_m"] == 0
    assert state["position"] == {"x_m": 0.0, "y_m": 0.0}
    assert state["oxygen_remaining_pct"] == state["battery_remaining_pct"] == 100
    assert state["telemetry_count"] == 1 and body["data_origin"] == "simulated"
    assert client.get(f"{SIMS}/{session_id}/telemetry").json()["total"] == 1
    assert db_session.get(Mission, mission.id).status == MissionStatus.PLANNED
    assert client.post(f"{SIMS}/{session_id}/resume").status_code == 409


def test_reset_after_completion_reopens_mission_as_planned(client, db_session, route_fixture):
    mission, route = route_fixture
    session_id = start(client, route.id).json()["session"]["id"]
    advance(client, session_id, 100)
    assert db_session.get(Mission, mission.id).status == MissionStatus.COMPLETED
    reset = client.post(f"{SIMS}/{session_id}/reset")
    assert reset.status_code == 200 and reset.json()["session"]["status"] == "created"
    assert db_session.get(Mission, mission.id).status == MissionStatus.PLANNED


def test_persisted_telemetry_is_paginated_and_simulated_only(client, db_session, route_fixture):
    _, route = route_fixture
    session_id = start(client, route.id).json()["session"]["id"]
    for seconds in (5, 5, 10, 10):
        advance(client, session_id, seconds)
    page = client.get(f"{SIMS}/{session_id}/telemetry", params={"limit": 2, "offset": 1})
    assert page.status_code == 200 and page.json()["total"] == 5 and len(page.json()["items"]) == 2
    times = [item["sim_time_s"] for item in page.json()["items"]]
    assert times == [5, 10]
    assert all(item["data_origin"] == "simulated" for item in page.json()["items"])
    assert page.json()["items"][1]["position"] == {"x_m": 10.0, "y_m": 0.0}


def test_state_survives_a_new_request_and_route_id_is_not_client_forged(client, route_fixture):
    _, route = route_fixture
    payload = {"route_id": str(route.id), "mission_id": str(uuid4()), "data_origin": "observed"}
    assert client.post(SIMS, json=payload).status_code == 422
    state = start(client, route.id).json()["session"]
    advance(client, state["id"], 12)
    fetched = client.get(f"{SIMS}/{state['id']}/state")
    assert fetched.status_code == 200 and fetched.json()["sim_elapsed_s"] == 12
    assert fetched.json()["data_origin"] == "simulated"
    assert fetched.json()["latest_telemetry"]["data_origin"] == "simulated"


def test_mission_must_be_planned_and_only_one_active_session_per_mission(client, db_session, route_fixture):
    mission, route = route_fixture
    mission.status = MissionStatus.DRAFT
    db_session.commit()
    assert start(client, route.id).status_code == 201                # draft is promoted to active on start
    assert start(client, route.id).status_code == 409

    mission2 = Mission(
        name="bad state", site_id=mission.site_id, status=MissionStatus.ABORTED,
        start_location=mission.start_location, start_longitude_deg_east=mission.start_longitude_deg_east,
        start_latitude_deg=mission.start_latitude_deg, eva_duration_limit_min=60, return_reserve_min=5,
        constraints={},
    )
    db_session.add(mission2)
    db_session.flush()
    route2 = Route(
        mission_id=mission2.id, profile=RouteProfile.SAFETY_FIRST, status=RouteStatus.FOUND,
        geometry=route.geometry, distance_m=30, estimated_duration_s=40, total_cost=60, risk_score=0.4,
        cost_model=route.cost_model, data_origin=DataOrigin.SIMULATED,
    )
    db_session.add(route2)
    db_session.commit()
    assert start(client, route2.id).status_code == 409


def test_resource_model_and_advance_inputs_are_validated(client, route_fixture):
    _, route = route_fixture
    base = {"route_id": str(route.id)}
    for model in [
        {"oxygen_use_pct_per_hour_moving": -1},
        {"oxygen_use_pct_per_hour_moving": 101},
        {"risk_warning_threshold": 1.1},
        {"risk_warning_threshold": 1, "risk_critical_threshold": 0.8},
        {"unexpected": True},
    ]:
        response = client.post(SIMS, json={**base, "resource_model": model})
        assert response.status_code == 422
    assert client.post(SIMS, json={"route_id": str(uuid4())}).status_code == 404
    state = start(client, route.id).json()["session"]
    for invalid in [0, -1, 86_401, "nan", "inf", "x"]:
        assert client.post(f"{SIMS}/{state['id']}/advance", json={"advance_by_s": invalid}).status_code == 422
    assert client.post(f"{SIMS}/{state['id']}/advance", json={"advance_by_s": 1, "sim_time": 999}).status_code == 422


def test_invalid_state_transitions_and_missing_resources(client, route_fixture):
    _, route = route_fixture
    assert client.get(f"{SIMS}/{uuid4()}/state").status_code == 404
    assert client.post(f"{SIMS}/{uuid4()}/pause").status_code == 404
    assert client.post(f"{SIMS}/not-a-uuid/reset").status_code == 404 or client.post(f"{SIMS}/not-a-uuid/reset").status_code == 422
    state = start(client, route.id).json()["session"]
    sid = state["id"]
    assert client.post(f"{SIMS}/{sid}/resume").status_code == 409
    assert client.post(f"{SIMS}/{sid}/reset").status_code == 200
    assert client.post(f"{SIMS}/{sid}/pause").status_code == 409
    assert client.post(f"{SIMS}/{sid}/advance", json={"advance_by_s": 1}).status_code == 409


def test_database_rejects_non_simulated_rows(client, db_session, route_fixture):
    mission, route = route_fixture
    bad = EVASession(mission_id=mission.id, route_id=route.id, status=EVAStatus.CREATED,
                     data_origin=DataOrigin.OBSERVED, resource_model={})
    db_session.add(bad)
    from sqlalchemy.exc import IntegrityError
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_start_rejects_non_found_route(client, db_session, route_fixture):
    mission, route = route_fixture
    route.status = RouteStatus.NO_ROUTE
    db_session.commit()
    response = start(client, route.id)
    assert response.status_code == 422


def test_resource_rates_are_not_claimed_as_real_measurements(client, route_fixture):
    _, route = route_fixture
    body = start(client, route.id).json()
    assert body["session"]["resource_model"]["version"].startswith("prototype")
    assert "not actual" in body["message"].lower() or "simulated" in body["message"].lower()
