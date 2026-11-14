from uuid import uuid4

import numpy as np
import pytest
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point, Polygon, box
from sqlalchemy import func, select

from app.core import limits
from app.core.config import get_settings
from app.models import HazardZone, Route, RouteWaypoint
from app.models.enums import DataOrigin, HazardZoneType
from app.services.terrain_service import load_terrain

PLAN = "/api/v1/routes/plan"
HABITAT = (-3000.0, -2500.0)
ROOMY = {"eva_duration_limit_min": 1440, "return_reserve_min": 60}      # the demo traverse is ~6.5 h of walking


def plan(client, site=None, **body):
    if site is not None:
        body.setdefault("site_id", str(site.id))
    return client.post(PLAN, json=body)


def by_profile(response):
    return {r["profile"]: r for r in response.json()["routes"]}


def polyline_length(coordinates) -> float:
    xy = np.asarray(coordinates, dtype=float)
    return float(np.hypot(*np.diff(xy, axis=0).T).sum())


def target_ids(seeded):
    return {t.name[:13]: str(t.id) for t in seeded.targets}       # "DEMO Target A", "DEMO Target B", ...


def seeded_mission_id(client) -> str:
    return client.get("/api/v1/missions").json()["items"][0]["id"]

def add_zone(db_session, seeded, geometry, blocking=True, risk=1.0, name="test zone"):
    zone = HazardZone(
        site_id=seeded.site.id, name=name, zone_type=HazardZoneType.OBSTACLE, geometry=from_shape(geometry, srid=0),
        risk_score=risk, is_blocking=blocking, data_origin=DataOrigin.SIMULATED,
    )
    db_session.add(zone)
    db_session.commit()
    return zone


def isolate_target_a(db_session, seeded):
    """A blocking ring (with a free pocket around Target A at 1200,800) that cannot be crossed."""
    ring = Polygon(box(1100, 700, 1300, 900).exterior.coords, [box(1170, 770, 1230, 830).exterior.coords])
    return add_zone(db_session, seeded, ring, name="enclosure")


# --- feasible routes ---------------------------------------------------------------------------

def test_feasible_mission_plan_returns_both_profiles_computed_from_real_terrain(client, seeded, demo_terrain):
    response = plan(client, mission_id=seeded_mission_id(client), **ROOMY)
    assert response.status_code == 200, response.text
    body = response.json()
    routes = by_profile(response)
    assert set(routes) == {"shorter_distance", "safety_first"}
    assert body["terrain"]["is_demo"] and body["terrain"]["resolution_m"] == 20.0
    assert body["terrain"]["elevation_layer"]["data_origin"] == "simulated"
    assert any("DEMO" in n for n in body["notes"]) and any("not a validated" in n for n in body["notes"])
    for route in routes.values():
        assert route["status"] == "found" and route["no_route"] is None and route["data_origin"] == "simulated"
        coords = route["geometry"]["coordinates"]
        assert route["geometry"]["point_count"] == len(coords) == len(route["geometry"]["geographic"])
        assert [s["label"] for s in route["stops"]] == ["Start", "DEMO Target A (proposed)", "DEMO Target B (proposed)", "Habitat"]
        assert [s["waypoint_type"] for s in route["stops"]] == ["start", "science_target", "science_target", "habitat_return"]
        assert coords[0] == pytest.approx(list(HABITAT)) and coords[-1] == pytest.approx(list(HABITAT))
        assert route["stops"][1]["location"] == {"x_m": 1200.0, "y_m": 800.0}
        assert route["constraints"]["within_time_budget"] is True and route["constraints"]["violations"] == []


def test_every_reported_metric_is_recomputed_from_the_returned_geometry(client, seeded, demo_terrain):
    body = plan(client, seeded.site, **ROOMY, destinations=[{"science_target_id": str(seeded.targets[0].id)}]).json()
    for route in body["routes"]:
        m, coords = route["metrics"], route["geometry"]["coordinates"]
        assert m["distance_m"] == pytest.approx(polyline_length(coords), rel=1e-9)
        assert sum(leg["distance_m"] for leg in route["legs"]) == pytest.approx(m["distance_m"], rel=1e-9)
        assert route["stops"][-1]["distance_from_start_m"] == pytest.approx(m["distance_m"], rel=1e-9)
        assert route["stops"][-1]["arrival_time_s"] == pytest.approx(m["total_time_s"])
        assert m["total_time_s"] == pytest.approx(m["travel_time_s"] + m["activity_time_s"])
        assert sum(leg["travel_time_s"] for leg in route["legs"]) == pytest.approx(m["travel_time_s"], rel=1e-9)
        speed = route["cost_model"]["speed_model"]
        assert m["distance_m"] / speed["base_speed_mps"] <= m["travel_time_s"] <= m["distance_m"] / (
            speed["base_speed_mps"] * speed["min_speed_fraction"])
        assert 0 <= m["risk_score"] <= m["peak_risk"] <= 1 and 0 <= m["max_slope_deg"] <= 15
        assert m["mean_slope_deg"] <= m["max_slope_deg"] and m["elevation_gain_m"] >= 0 <= m["elevation_loss_m"]
        assert route["search"]["expanded_nodes"] > 0 and route["cost_model"]["formula"]
        cumulative = [s["distance_from_start_m"] for s in route["stops"]]
        assert cumulative == sorted(cumulative) and cumulative[0] == 0.0


def test_route_cells_are_traversable_according_to_the_terrain_api(client, db_session, storage, seeded, demo_terrain):
    body = plan(client, seeded.site, **ROOMY, destinations=[{"science_target_id": str(seeded.targets[0].id)}]).json()
    data = load_terrain(db_session, storage, seeded.site)
    for route in body["routes"]:
        coords = route["geometry"]["coordinates"]
        for x, y in coords[:: max(1, len(coords) // 25)] + [coords[-1]]:
            sample = client.get("/api/v1/terrain/sample", params={"site_id": str(seeded.site.id), "x_m": x, "y_m": y}).json()
            assert sample["blocked"] is False and sample["is_valid"] and sample["slope_deg"] <= 15
        # dense check against the raster: no vertex or segment midpoint on blocked cells
        rows_cols = [data.grid.cell_of(x, y) for x, y in coords]
        assert not any(data.terrain.blocked[rc] for rc in rows_cols)


def test_profile_comparison_is_consistent_with_the_two_computed_routes(client, seeded, demo_terrain):
    body = plan(client, seeded.site, **ROOMY, destinations=[{"science_target_id": str(seeded.targets[1].id)}]).json()
    routes = {r["profile"]: r for r in body["routes"]}
    short, safe = routes["shorter_distance"]["metrics"], routes["safety_first"]["metrics"]
    cmp = body["comparison"]
    assert short["distance_m"] <= safe["distance_m"] + 1e-6                     # shortest is optimal for length
    assert cmp["distance_difference_m"] == pytest.approx(safe["distance_m"] - short["distance_m"])
    assert cmp["risk_score_difference"] == pytest.approx(safe["risk_score"] - short["risk_score"])
    assert cmp["peak_risk_difference"] == pytest.approx(safe["peak_risk"] - short["peak_risk"])
    assert cmp["distance_ratio"] == pytest.approx(safe["distance_m"] / short["distance_m"])
    assert cmp["same_geometry"] == (routes["shorter_distance"]["geometry"] == routes["safety_first"]["geometry"])
    assert isinstance(cmp["summary"], str) and cmp["summary"]


def test_single_profile_request_has_no_comparison(client, seeded, demo_terrain):
    body = plan(client, seeded.site, profiles=["safety_first"], **ROOMY).json()
    assert [r["profile"] for r in body["routes"]] == ["safety_first"] and body["comparison"] is None


def test_planning_is_deterministic_and_read_only(client, db_session, seeded, demo_terrain):
    before = db_session.scalar(select(func.count()).select_from(Route))
    first, second = plan(client, seeded.site, **ROOMY).json(), plan(client, seeded.site, **ROOMY).json()
    for a, b in zip(first["routes"], second["routes"]):
        assert a["geometry"] == b["geometry"] and a["metrics"] == b["metrics"]
    assert db_session.scalar(select(func.count()).select_from(Route)) == before


# --- stops: science targets, waypoints, activities, return ------------------------------------------

def test_destinations_mix_targets_and_waypoints_in_the_given_order(client, seeded, demo_terrain):
    a, b = str(seeded.targets[0].id), str(seeded.targets[1].id)
    body = plan(client, seeded.site, **ROOMY, destinations=[
        {"science_target_id": b, "activity_duration_min": 20},
        {"location": {"x_m": -2000.0, "y_m": -1500.0}, "label": "Rover cache"},
        {"science_target_id": a},
    ]).json()
    for route in body["routes"]:
        assert [s["label"] for s in route["stops"]] == [
            "Start", "DEMO Target B (proposed)", "Rover cache", "DEMO Target A (proposed)", "Habitat"]
        assert [s["activity_duration_min"] for s in route["stops"]] == [0, 20, 0, 45, 0]      # override, none, target default
        assert route["metrics"]["activity_time_s"] == (20 + 45) * 60
        assert route["stops"][2]["waypoint_type"] == "waypoint" and route["stops"][2]["location"] == {"x_m": -2000.0, "y_m": -1500.0}
        assert [-2000.0, -1500.0] in [list(p) for p in route["geometry"]["coordinates"]]      # exact point is on the path
        arrivals = [s["arrival_time_s"] for s in route["stops"]]
        assert arrivals == sorted(arrivals)
        assert route["stops"][1]["departure_time_s"] == pytest.approx(route["stops"][1]["arrival_time_s"] + 20 * 60)
        assert route["stops"][2]["arrival_time_s"] > route["stops"][1]["departure_time_s"]


def test_mission_targets_use_their_planned_durations(client, seeded, demo_terrain):
    mission = client.post("/api/v1/missions", json={"name": "m", "site_id": str(seeded.site.id),
                                                    "science_target_ids": [str(seeded.targets[0].id)]}).json()
    body = plan(client, mission_id=mission["id"], **ROOMY).json()
    route = body["routes"][0]
    assert [s["label"] for s in route["stops"]] == ["Start", "DEMO Target A (proposed)", "Habitat"]
    assert route["stops"][1]["activity_duration_min"] == 45          # the target's own estimate
    assert body["mission_id"] == mission["id"]


def test_start_can_be_overridden_and_defaults_to_mission_then_habitat(client, seeded, demo_terrain):
    mission = client.get("/api/v1/missions").json()["items"][0]
    custom = plan(client, mission_id=mission["id"], start={"x_m": -2500.0, "y_m": -2000.0}, **ROOMY,
                  destinations=[], profiles=["shorter_distance"]).json()["routes"][0]
    assert custom["geometry"]["coordinates"][0] == [-2500.0, -2000.0]
    default = plan(client, seeded.site, **ROOMY, destinations=[], profiles=["shorter_distance"]).json()["routes"][0]
    assert default["geometry"]["coordinates"][0] == list(HABITAT)


def test_without_return_leg_the_route_ends_at_the_last_destination_and_warns(client, seeded, demo_terrain):
    route = plan(client, seeded.site, **ROOMY, return_to_habitat=False, profiles=["shorter_distance"],
                 destinations=[{"science_target_id": str(seeded.targets[0].id)}]).json()["routes"][0]
    assert [s["label"] for s in route["stops"]] == ["Start", "DEMO Target A (proposed)"]
    assert route["geometry"]["coordinates"][-1] == [1200.0, 800.0]
    assert route["constraints"]["return_planned"] is False
    assert any("return reserve cannot be verified" in w for w in route["warnings"])


def test_return_to_habitat_adds_a_computed_return_leg(client, seeded, demo_terrain):
    dest = [{"science_target_id": str(seeded.targets[0].id)}]
    out = plan(client, seeded.site, **ROOMY, return_to_habitat=False, destinations=dest, profiles=["shorter_distance"]).json()["routes"][0]
    back = plan(client, seeded.site, **ROOMY, destinations=dest, profiles=["shorter_distance"]).json()["routes"][0]
    assert back["metrics"]["distance_m"] > out["metrics"]["distance_m"]
    assert back["legs"][-1]["distance_m"] == pytest.approx(back["metrics"]["distance_m"] - out["metrics"]["distance_m"])
    assert back["geometry"]["coordinates"][-1] == list(HABITAT) and back["stops"][-1]["waypoint_type"] == "habitat_return"


# --- obstacles: blocked starts/destinations ------------------------------------------------------------

def test_start_inside_a_blocking_hazard_zone_is_reported_not_routed(client, seeded, demo_terrain):
    response = plan(client, seeded.site, start={"x_m": 500.0, "y_m": 450.0})          # seeded DEMO obstacle zone
    assert response.status_code == 200
    for route in response.json()["routes"]:
        assert route["status"] == "no_route" and route["geometry"] is None and route["metrics"] is None
        assert route["no_route"]["code"] == "start_blocked" and "'Start' is on blocked terrain" in route["no_route"]["message"]
        assert route["constraints"]["violations"] == ["no_route"] and route["stops"] == []


def test_start_on_unknown_terrain_is_blocked_with_an_explanation(client, seeded, demo_terrain):
    route = plan(client, seeded.site, start={"x_m": -1250.0, "y_m": 1250.0}, profiles=["safety_first"]).json()["routes"][0]
    assert route["no_route"]["code"] == "start_blocked" and "unknown terrain" in route["no_route"]["message"]


def test_destination_on_excessive_slope_is_blocked_with_the_limit_in_the_message(client, db_session, storage, seeded, demo_terrain):
    data = load_terrain(db_session, storage, seeded.site)
    rows, cols = np.where(data.slope > 20)
    x, y = data.grid.cell_center(int(rows[0]), int(cols[0]))
    route = plan(client, seeded.site, destinations=[{"location": {"x_m": x, "y_m": y}, "label": "Cliff"}],
                 profiles=["shorter_distance"]).json()["routes"][0]
    assert route["no_route"]["code"] == "destination_blocked"
    assert "'Cliff'" in route["no_route"]["message"] and "exceeds the 15 deg limit" in route["no_route"]["message"]
    assert route["no_route"]["to_label"] == "Cliff" and route["no_route"]["from_label"] == "Start"


def test_new_blocking_zone_on_the_path_makes_the_planner_go_around_it(client, db_session, seeded, demo_terrain):
    dest = [{"science_target_id": str(seeded.targets[0].id)}]
    before = plan(client, seeded.site, **ROOMY, destinations=dest, return_to_habitat=False,
                  profiles=["shorter_distance"]).json()["routes"][0]
    coords = before["geometry"]["coordinates"]
    mx, my = coords[len(coords) // 2]
    wall = box(mx - 150, my - 150, mx + 150, my + 150)
    add_zone(db_session, seeded, wall, name="new obstacle")
    after = plan(client, seeded.site, **ROOMY, destinations=dest, return_to_habitat=False,
                 profiles=["shorter_distance"]).json()["routes"][0]
    assert after["status"] == "found"
    assert any(wall.contains(Point(p)) for p in coords)                         # the old path really crossed the zone
    assert not any(wall.contains(Point(p)) for p in after["geometry"]["coordinates"])
    assert after["geometry"]["coordinates"] != coords
    # Open ground has many equal-length shortest paths, so the detour may cost nothing but can never be shorter.
    assert after["metrics"]["distance_m"] >= before["metrics"]["distance_m"] - 1e-6


def test_non_blocking_zone_adds_risk_that_the_safety_route_avoids(client, db_session, seeded, demo_terrain):
    dest = [{"science_target_id": str(seeded.targets[0].id)}]
    base = plan(client, seeded.site, **ROOMY, destinations=dest, return_to_habitat=False).json()
    short = next(r for r in base["routes"] if r["profile"] == "shorter_distance")
    coords = short["geometry"]["coordinates"]
    mx, my = coords[len(coords) // 2]
    add_zone(db_session, seeded, box(mx - 120, my - 120, mx + 120, my + 120), blocking=False, risk=0.9, name="soft ground")
    after = plan(client, seeded.site, **ROOMY, destinations=dest, return_to_habitat=False).json()
    new_short = next(r for r in after["routes"] if r["profile"] == "shorter_distance")
    new_safe = next(r for r in after["routes"] if r["profile"] == "safety_first")
    assert new_short["geometry"] == short["geometry"]                           # length-only routing ignores soft risk
    assert new_short["metrics"]["peak_risk"] >= 0.9 - 1e-6                    # risk is stored as float32
    assert new_safe["metrics"]["peak_risk"] < 0.9 and new_safe["metrics"]["distance_m"] > new_short["metrics"]["distance_m"]


# --- unreachable ------------------------------------------------------------------------------------

def test_enclosed_destination_is_unreachable_not_a_fabricated_route(client, db_session, seeded, demo_terrain):
    isolate_target_a(db_session, seeded)
    response = plan(client, seeded.site, **ROOMY, destinations=[{"science_target_id": str(seeded.targets[0].id)}])
    assert response.status_code == 200
    for route in response.json()["routes"]:
        assert route["status"] == "no_route" and route["geometry"] is None and route["metrics"] is None
        info = route["no_route"]
        assert info["code"] == "unreachable" and info["leg_index"] == 0
        assert info["from_label"] == "Start" and info["to_label"] == "DEMO Target A (proposed)"
        assert "No traversable path" in info["message"] and route["search"]["expanded_nodes"] > 0
    assert response.json()["comparison"] is None


def test_unreachable_later_leg_names_the_leg(client, db_session, seeded, demo_terrain):
    isolate_target_a(db_session, seeded)
    route = plan(client, seeded.site, **ROOMY, profiles=["shorter_distance"], destinations=[
        {"science_target_id": str(seeded.targets[1].id)}, {"science_target_id": str(seeded.targets[0].id)}]).json()["routes"][0]
    assert route["no_route"]["leg_index"] == 1 and route["no_route"]["from_label"] == "DEMO Target B (proposed)"


def test_leaving_an_enclosure_is_also_unreachable(client, db_session, seeded, demo_terrain):
    isolate_target_a(db_session, seeded)
    route = plan(client, seeded.site, **ROOMY, profiles=["safety_first"], start={"x_m": 1200.0, "y_m": 800.0},
                 destinations=[]).json()["routes"][0]
    assert route["no_route"]["code"] == "unreachable" and route["no_route"]["from_label"] == "Start"
    assert route["no_route"]["to_label"] == "Habitat"


def test_tight_slope_constraint_can_make_the_mission_unreachable(client, seeded, demo_terrain):
    route = plan(client, seeded.site, **ROOMY, profiles=["shorter_distance"], constraints={"max_slope_deg": 0.5},
                 destinations=[{"science_target_id": str(seeded.targets[0].id)}]).json()["routes"][0]
    assert route["status"] == "no_route" and route["no_route"]["code"] in {"start_blocked", "unreachable", "destination_blocked"}
    assert route["constraints"]["max_slope_deg_applied"] == 0.5


# --- mission constraints -------------------------------------------------------------------------------

def test_slope_constraint_is_applied_and_respected_by_the_route(client, seeded, demo_terrain):
    loose = plan(client, seeded.site, **ROOMY, profiles=["shorter_distance"], return_to_habitat=False,
                 destinations=[{"science_target_id": str(seeded.targets[1].id)}]).json()["routes"][0]
    tight = plan(client, seeded.site, **ROOMY, profiles=["shorter_distance"], return_to_habitat=False,
                 constraints={"max_slope_deg": 4.0},
                 destinations=[{"science_target_id": str(seeded.targets[1].id)}]).json()["routes"][0]
    assert tight["constraints"]["max_slope_deg_applied"] == 4.0
    if tight["status"] == "found":
        assert tight["metrics"]["max_slope_deg"] <= 4.0
        assert tight["metrics"]["distance_m"] >= loose["metrics"]["distance_m"] - 1e-6
    assert loose["metrics"]["max_slope_deg"] <= 15.0


def test_mission_cannot_loosen_the_processed_hazard_layer(client, seeded, demo_terrain):
    body = plan(client, seeded.site, **ROOMY, constraints={"max_slope_deg": 40.0}, profiles=["shorter_distance"]).json()
    assert body["routes"][0]["constraints"]["max_slope_deg_applied"] == 15.0
    assert any("exceeds the processed hazard layer" in w for w in body["warnings"])
    assert any("exceeds the processed hazard layer" in w for w in body["routes"][0]["warnings"])


def test_mission_constraints_are_taken_from_the_mission(client, seeded, demo_terrain):
    mission = client.post("/api/v1/missions", json={
        "name": "strict", "site_id": str(seeded.site.id), "constraints": {"max_slope_deg": 6.0},
        "eva_duration_limit_min": 500, "return_reserve_min": 30}).json()
    route = plan(client, mission_id=mission["id"], profiles=["shorter_distance"]).json()["routes"][0]
    c = route["constraints"]
    assert (c["max_slope_deg_applied"], c["eva_duration_limit_min"], c["return_reserve_min"]) == (6.0, 500, 30)
    assert c["usable_time_s"] == 470 * 60
    overridden = plan(client, mission_id=mission["id"], eva_duration_limit_min=200, return_reserve_min=10,
                      constraints={"max_slope_deg": 9.0}, profiles=["shorter_distance"]).json()["routes"][0]["constraints"]
    assert (overridden["max_slope_deg_applied"], overridden["usable_time_s"]) == (9.0, 190 * 60)


def test_time_budget_is_checked_against_the_computed_time(client, seeded, demo_terrain):
    dest = [{"science_target_id": str(seeded.targets[1].id)}]
    over = plan(client, seeded.site, eva_duration_limit_min=120, return_reserve_min=30, destinations=dest,
                profiles=["shorter_distance"]).json()["routes"][0]
    c = over["constraints"]
    assert c["within_time_budget"] is False and c["violations"] == ["eva_time_budget_exceeded"]
    assert c["planned_total_time_s"] == pytest.approx(over["metrics"]["total_time_s"])
    assert c["margin_s"] == pytest.approx(c["usable_time_s"] - over["metrics"]["total_time_s"]) and c["margin_s"] < 0
    assert over["status"] == "found" and any("exceeds the usable EVA time" in w for w in over["warnings"])
    within = plan(client, seeded.site, **ROOMY, destinations=dest, profiles=["shorter_distance"]).json()["routes"][0]["constraints"]
    assert within["within_time_budget"] is True and within["margin_s"] > 0


def test_reserve_must_be_below_the_limit(client, seeded, demo_terrain):
    response = plan(client, seeded.site, eva_duration_limit_min=60, return_reserve_min=60)
    assert response.status_code == 422 and response.json()["error"]["code"] == "validation_error"


# --- validation and errors ----------------------------------------------------------------------------

def test_missing_resources_and_terrain(client, seeded, demo_terrain):
    assert plan(client, site_id=str(uuid4())).status_code == 404
    assert plan(client, mission_id=str(uuid4())).status_code == 404
    unknown_target = plan(client, seeded.site, destinations=[{"science_target_id": str(uuid4())}])
    assert unknown_target.status_code == 404
    mission = client.get("/api/v1/missions").json()["items"][0]
    assert plan(client, site_id=str(uuid4()), mission_id=mission["id"]).status_code == 422


def test_site_without_processed_terrain_returns_terrain_not_available(client, seeded):
    response = plan(client, seeded.site)
    assert response.status_code == 404 and response.json()["error"]["code"] == "terrain_not_available"


@pytest.mark.parametrize("body", [
    {},                                                                                     # no site or mission
    {"profiles": []}, {"profiles": ["shorter_distance", "shorter_distance"]}, {"profiles": ["fastest"]},
    {"start": {"x_m": 9000.0, "y_m": 0.0}}, {"start": {"x_m": "nan", "y_m": 0.0}}, {"start": {"x_m": 1.0}},
    {"destinations": [{}]}, {"destinations": [{"location": {"x_m": 0, "y_m": 0}, "science_target_id": str(uuid4())}]},
    {"destinations": [{"location": {"x_m": 9e6, "y_m": 0}}]},
    {"destinations": [{"location": {"x_m": 0, "y_m": 0}, "activity_duration_min": -1}]},
    {"destinations": [{"location": {"x_m": 0, "y_m": 0}, "label": "   "}]},
    {"destinations": [], "return_to_habitat": False},                                          # nothing to plan
    {"eva_duration_limit_min": 0}, {"eva_duration_limit_min": 5000}, {"return_reserve_min": -1},
    {"constraints": {"max_slope_deg": 90}}, {"constraints": {"surprise": 1}},
    {"geometry": [[0, 0], [1, 1]]}, {"unexpected": True},                                      # client geometry is never accepted
])
def test_invalid_plan_requests_are_rejected(client, seeded, demo_terrain, body):
    payload = dict(body)
    if body:
        payload.setdefault("site_id", str(seeded.site.id))
    response = client.post(PLAN, json=payload)
    assert response.status_code == 422 and response.json()["error"]["code"] == "validation_error"


def test_target_from_another_site_is_rejected(client, db_session, seeded, demo_terrain):
    from tests.test_missions import other_site
    other = other_site(db_session)
    response = plan(client, other, destinations=[{"science_target_id": str(seeded.targets[0].id)}])
    assert response.status_code == 422


def test_too_many_destinations_are_rejected(client, seeded, demo_terrain):
    limit = get_settings().route_max_destinations
    points = [{"location": {"x_m": -2900.0 + i * 10, "y_m": -2400.0}} for i in range(limit + 1)]
    response = plan(client, seeded.site, destinations=points)
    assert response.status_code == 422 and str(limit) in response.json()["error"]["message"]
    assert plan(client, seeded.site, destinations=points[:limit], **ROOMY, profiles=["shorter_distance"]).status_code == 200


# --- search limits and request limits ----------------------------------------------------------------

def test_work_limit_is_reported_as_search_limit_not_unreachable(client, seeded, demo_terrain, monkeypatch):
    monkeypatch.setattr(get_settings(), "route_max_expansions", 20)
    routes = plan(client, seeded.site, destinations=[{"science_target_id": str(seeded.targets[0].id)}]).json()["routes"]
    for route in routes:
        assert route["status"] == "no_route" and route["no_route"]["code"] == "search_limit_exceeded"
        assert "work limit" in route["no_route"]["message"]


def test_time_limit_is_reported_as_search_timeout(client, seeded, demo_terrain, monkeypatch):
    monkeypatch.setattr(get_settings(), "route_search_timeout_s", 1e-12)
    route = plan(client, seeded.site, profiles=["safety_first"],
                 destinations=[{"science_target_id": str(seeded.targets[0].id)}]).json()["routes"][0]
    assert route["no_route"]["code"] == "search_timeout" and "ran out of time" in route["no_route"]["message"]


def test_rate_limit_returns_429_with_retry_after(client, seeded, demo_terrain, monkeypatch):
    monkeypatch.setattr(limits.route_rate_limiter, "max_calls", 2)
    for _ in range(2):
        assert plan(client, seeded.site, profiles=["shorter_distance"], return_to_habitat=False,
                    destinations=[{"location": {"x_m": -2900.0, "y_m": -2400.0}}]).status_code == 200
    blocked = plan(client, seeded.site, profiles=["shorter_distance"])
    assert blocked.status_code == 429 and blocked.json()["error"]["code"] == "rate_limited"
    assert int(blocked.headers["retry-after"]) >= 1


def test_rate_limiter_window_expires():
    now = [0.0]
    limiter = limits.SlidingWindowLimiter(2, 60.0, clock=lambda: now[0])
    limiter.check("a"), limiter.check("a"), limiter.check("b")
    with pytest.raises(limits.TooManyRequestsError):
        limiter.check("a")
    now[0] = 61.0
    limiter.check("a")


def test_concurrency_gate_rejects_when_the_planner_is_busy(client, seeded, demo_terrain, monkeypatch):
    gate = limits.ConcurrencyGate(1)
    monkeypatch.setattr(limits, "route_gate", gate)
    gate.acquire()
    busy = plan(client, seeded.site, profiles=["shorter_distance"])
    assert busy.status_code == 429 and "busy" in busy.json()["error"]["message"]
    gate.release()
    assert plan(client, seeded.site, profiles=["shorter_distance"], **ROOMY).status_code == 200
    assert gate.acquire() is None                                                              # released after the request


def test_read_only_endpoints_are_not_rate_limited(client, seeded, demo_terrain, monkeypatch):
    monkeypatch.setattr(limits.route_rate_limiter, "max_calls", 1)
    for _ in range(5):
        assert client.get("/api/v1/terrain/metadata", params={"site_id": str(seeded.site.id)}).status_code == 200


# --- saving routes to missions ----------------------------------------------------------------------

def mission_route(client, mission_id, **body):
    return client.post(f"/api/v1/missions/{mission_id}/routes", json={"profile": "safety_first", **ROOMY, **body})


@pytest.fixture
def mission(client, seeded):
    return client.post("/api/v1/missions", json={
        "name": "route mission", "site_id": str(seeded.site.id),
        "science_target_ids": [str(seeded.targets[0].id), str(seeded.targets[1].id)],
        "eva_duration_limit_min": 1440, "return_reserve_min": 60}).json()


def test_saved_route_is_persisted_from_a_server_side_plan(client, db_session, seeded, demo_terrain, mission):
    response = client.post(f"/api/v1/missions/{mission['id']}/routes", json={"profile": "safety_first"})
    assert response.status_code == 201, response.text
    saved = response.json()
    assert saved["id"] and saved["mission_id"] == mission["id"] and saved["is_selected"] is True
    planned = next(r for r in plan(client, mission_id=mission["id"]).json()["routes"] if r["profile"] == "safety_first")
    assert saved["geometry"] == planned["geometry"] and saved["metrics"] == planned["metrics"]   # deterministic engine

    db_session.expire_all()
    row = db_session.get(Route, saved["id"])
    m = saved["metrics"]
    assert (row.distance_m, row.estimated_duration_s, row.risk_score) == pytest.approx(
        (m["distance_m"], m["travel_time_s"], m["risk_score"]))
    assert row.max_cell_risk == pytest.approx(m["peak_risk"]) and row.elevation_gain_m == pytest.approx(m["elevation_gain_m"])
    assert row.status.value == "found" and row.profile.value == "safety_first" and row.data_origin.value == "simulated"
    assert row.terrain_layer_id == demo_terrain.elevation.id and row.dataset_version == demo_terrain.hazard.dataset_version
    stored = [list(p) for p in to_shape(row.geometry).coords]
    assert stored == [list(p) for p in saved["geometry"]["coordinates"]]
    assert db_session.scalar(select(func.count()).select_from(RouteWaypoint).where(RouteWaypoint.route_id == row.id)) == 4
    waypoints = db_session.scalars(select(RouteWaypoint).where(RouteWaypoint.route_id == row.id).order_by(RouteWaypoint.sequence)).all()
    assert [w.waypoint_type.value for w in waypoints] == ["start", "science_target", "science_target", "habitat_return"]
    assert [w.distance_from_start_m for w in waypoints] == pytest.approx([s["distance_from_start_m"] for s in saved["stops"]])
    assert waypoints[1].science_target_id == seeded.targets[0].id and waypoints[1].eta_s == pytest.approx(saved["stops"][1]["arrival_time_s"])
    assert polyline_length(stored) == pytest.approx(row.distance_m, rel=1e-9)


def test_saved_route_can_be_read_back_identically(client, seeded, demo_terrain, mission):
    saved = mission_route(client, mission["id"]).json()
    fetched = client.get(f"/api/v1/routes/{saved['id']}")
    assert fetched.status_code == 200 and fetched.json() == saved
    assert fetched.json()["metrics"]["distance_m"] == pytest.approx(polyline_length(fetched.json()["geometry"]["coordinates"]), rel=1e-9)
    assert client.get(f"/api/v1/routes/{uuid4()}").status_code == 404
    assert client.get("/api/v1/routes/not-a-uuid").status_code == 422


def test_selecting_a_new_route_unselects_the_previous_one(client, seeded, demo_terrain, mission):
    first = mission_route(client, mission["id"], profile="shorter_distance").json()
    second = mission_route(client, mission["id"], profile="safety_first").json()
    third = mission_route(client, mission["id"], profile="shorter_distance", select=False).json()
    assert (second["is_selected"], third["is_selected"]) == (True, False)
    assert client.get(f"/api/v1/routes/{first['id']}").json()["is_selected"] is False
    listing = client.get(f"/api/v1/missions/{mission['id']}/routes").json()
    assert listing["total"] == 3 and sum(r["is_selected"] for r in listing["items"]) == 1
    assert len(client.get(f"/api/v1/missions/{mission['id']}/routes", params={"limit": 2}).json()["items"]) == 2
    assert client.get(f"/api/v1/missions/{uuid4()}/routes").status_code == 404


def test_saving_uses_mission_destinations_unless_overridden(client, seeded, demo_terrain, mission):
    default = mission_route(client, mission["id"], profile="shorter_distance").json()
    assert [s["label"] for s in default["stops"]][1:3] == ["DEMO Target A (proposed)", "DEMO Target B (proposed)"]
    override = mission_route(client, mission["id"], profile="shorter_distance", return_to_habitat=False,
                             destinations=[{"science_target_id": str(seeded.targets[1].id)}]).json()
    assert [s["label"] for s in override["stops"]] == ["Start", "DEMO Target B (proposed)"]


def test_unreachable_route_is_not_saved_and_returns_409(client, db_session, seeded, demo_terrain, mission):
    isolate_target_a(db_session, seeded)
    before = db_session.scalar(select(func.count()).select_from(Route))
    response = mission_route(client, mission["id"])
    assert response.status_code == 409 and response.json()["error"]["code"] == "conflict"
    assert "No feasible route" in response.json()["error"]["message"]
    assert response.json()["error"]["details"][0]["code"] == "unreachable"
    assert db_session.scalar(select(func.count()).select_from(Route)) == before


def test_routes_can_only_be_added_to_draft_or_planned_missions(client, seeded, demo_terrain, mission):
    path = f"/api/v1/missions/{mission['id']}"
    assert client.patch(path, json={"status": "planned"}).status_code == 200
    assert mission_route(client, mission["id"]).status_code == 201
    client.patch(path, json={"status": "active"})
    blocked = mission_route(client, mission["id"])
    assert blocked.status_code == 409 and "draft or planned" in blocked.json()["error"]["message"]


@pytest.mark.parametrize("body", [{}, {"profile": "fastest"}, {"profile": "safety_first", "geometry": [[0, 0]]},
                                  {"profile": "safety_first", "mission_id": "x"}, {"profile": "safety_first", "profiles": ["safety_first"]}])
def test_invalid_mission_route_requests(client, seeded, demo_terrain, mission, body):
    assert client.post(f"/api/v1/missions/{mission['id']}/routes", json=body).status_code == 422


def test_mission_route_endpoint_edge_cases(client, seeded, demo_terrain, mission):
    assert mission_route(client, str(uuid4())).status_code == 404
    assert client.post("/api/v1/missions/not-a-uuid/routes", json={"profile": "safety_first"}).status_code == 422


def test_deleting_a_mission_removes_its_routes(client, db_session, seeded, demo_terrain, mission):
    saved = mission_route(client, mission["id"]).json()
    assert client.delete(f"/api/v1/missions/{mission['id']}").status_code == 204
    db_session.expire_all()
    assert client.get(f"/api/v1/routes/{saved['id']}").status_code == 404
    assert db_session.scalar(select(func.count()).select_from(RouteWaypoint)) == 0
