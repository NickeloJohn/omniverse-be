import math

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.models import EVASession, Mission, Route
from app.models.enums import DataOrigin, EVAStatus, RouteProfile, RouteStatus
from app.utils.mars_coords import geographic_to_local, local_to_geographic

EXPECTED_TABLES = {
    "exploration_sites",
    "data_layers",
    "raster_assets",
    "science_targets",
    "hazard_zones",
    "missions",
    "mission_targets",
    "routes",
    "route_waypoints",
    "eva_sessions",
    "telemetry_samples",
    "science_observations",
}


def test_migrations_create_expected_tables(db_session):
    rows = db_session.execute(
        text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    )
    assert EXPECTED_TABLES <= {r[0] for r in rows}


def test_local_geometry_columns_use_srid_zero_not_an_earth_crs(db_session):
    rows = db_session.execute(
        text("SELECT DISTINCT srid FROM geometry_columns WHERE f_table_name = ANY(:names)"),
        {"names": sorted(EXPECTED_TABLES)},
    ).all()
    assert [r[0] for r in rows] == [0]


def test_seed_is_idempotent_and_labelled_demo(db_session, seeded):
    from scripts.seed_dev_data import seed_development_data

    assert seed_development_data(db_session) is False
    assert seeded.site.is_demo is True
    assert seeded.site.data_origin == DataOrigin.SIMULATED
    assert {t.creation_source.value for t in seeded.targets} == {"demo_seed"}
    assert {t.verification_status.value for t in seeded.targets} == {"proposed"}


def test_eva_session_must_be_simulated(db_session, seeded):
    mission = db_session.query(Mission).one()
    route = Route(mission_id=mission.id, profile=RouteProfile.SAFETY_FIRST,
                  status=RouteStatus.NO_ROUTE, data_origin=DataOrigin.SIMULATED)
    db_session.add(route)
    db_session.flush()
    db_session.add(EVASession(mission_id=mission.id, route_id=route.id, status=EVAStatus.CREATED,
                              data_origin=DataOrigin.OBSERVED))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_local_frame_round_trip_and_scale():
    lon, lat = local_to_geographic(77.58, 18.38, 1000.0, -500.0)
    x, y = geographic_to_local(77.58, 18.38, lon, lat)
    assert x == pytest.approx(1000.0) and y == pytest.approx(-500.0)
    # 1 degree of latitude on the Mars reference sphere is R * pi / 180 metres
    _, lat1 = local_to_geographic(0, 0, 0, math.radians(1) * 3_396_190.0)
    assert lat1 == pytest.approx(1.0)


def test_local_frame_rejects_poles():
    with pytest.raises(ValueError):
        local_to_geographic(0, 90, 1, 1)
