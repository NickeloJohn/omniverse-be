from uuid import uuid4

import numpy as np
import pytest
from geoalchemy2.shape import from_shape
from shapely.geometry import box

from app.models import HazardZone, DataLayer
from app.models.enums import DataOrigin, HazardZoneType
from app.services.terrain_service import load_terrain
from sqlalchemy import select

SAMPLE = "/api/v1/terrain/sample"
META = "/api/v1/terrain/metadata"


def sample(client, site, x, y, **extra):
    return client.get(SAMPLE, params={"site_id": str(site.id), "x_m": x, "y_m": y, **extra})


def first_cell(mask, grid):
    rows, cols = np.where(mask)
    return grid.cell_center(int(rows[0]), int(cols[0]))


# --- /terrain/sample ---------------------------------------------------------------------------

def test_sample_on_open_ground_reports_values_and_provenance(client, seeded, demo_terrain):
    response = sample(client, seeded.site, -3000.0, -2500.0)
    assert response.status_code == 200
    body = response.json()
    assert body["is_valid"] and body["blocked"] is False and body["blocked_reasons"] == []
    assert body["terrain_category"] == "low" and body["method"] == "nearest" and body["resolution_m"] == 20.0
    assert 0 <= body["terrain_risk_score"] <= body["risk_score"] <= 1
    assert body["slope_deg"] < 5 and isinstance(body["max_neighbor_step_m"], float)
    assert (body["row"], body["col"]) == (325, 50)
    prov = body["provenance"]
    assert prov["elevation_layer"]["is_demo"] and prov["elevation_layer"]["data_origin"] == "simulated"
    assert prov["slope_layer"]["layer_type"] == "slope" and prov["hazard_config_version"] == "prototype-slope-v1"
    assert body["notes"][0].startswith("DEMO") and any("not validated NASA" in n for n in body["notes"])
    assert body["latitude_deg"] < 18.38 and body["longitude_deg_east"] < 77.58            # south-west of centre


def test_sample_matches_the_stored_raster(client, db_session, storage, seeded, demo_terrain):
    data = load_terrain(db_session, storage, seeded.site)
    for x, y in [(1210.0, 805.0), (-500.0, 2200.0), (2600.0, -1200.0)]:
        row, col = data.grid.cell_of(x, y)
        body = sample(client, seeded.site, x, y).json()
        assert (body["row"], body["col"]) == (row, col)
        assert body["elevation_m"] == pytest.approx(float(data.elevation[row, col]), abs=1e-4)
        assert body["slope_deg"] == pytest.approx(float(data.slope[row, col]), abs=1e-4)


def test_unknown_terrain_is_reported_invalid_and_blocked_not_filled(client, seeded, demo_terrain):
    body = sample(client, seeded.site, -1250.0, 1250.0).json()      # inside the synthetic NoData patch
    assert body["is_valid"] is False and body["elevation_m"] is None and body["slope_deg"] is None
    assert body["terrain_category"] == "unknown" and body["blocked"] is True
    assert body["blocked_reasons"] == ["unknown_terrain"] and body["terrain_risk_score"] == 1.0


def test_steep_slope_is_excluded_with_reason(client, db_session, storage, seeded, demo_terrain):
    data = load_terrain(db_session, storage, seeded.site)
    x, y = first_cell(data.terrain.category == 3, data.grid)
    body = sample(client, seeded.site, x, y).json()
    assert body["terrain_category"] == "excluded" and body["blocked"] and "slope_above_limit" in body["blocked_reasons"]
    assert body["slope_deg"] > 15 and body["risk_score"] == 1.0


def test_blocking_hazard_zone_blocks_and_is_named(client, seeded, demo_terrain):
    body = sample(client, seeded.site, 500.0, 450.0).json()          # seeded DEMO obstacle zone
    assert body["blocked"] and body["risk_score"] == 1.0 and body["terrain_risk_score"] < 1.0
    assert body["blocked_reasons"] == ["hazard_zone:DEMO obstacle zone (simulated)"]
    zone = body["hazard_zones"][0]
    assert zone["is_blocking"] and zone["data_origin"] == "simulated" and zone["zone_type"] == "obstacle"
    assert sample(client, seeded.site, 520.0, 450.0).json()["blocked"]
    assert sample(client, seeded.site, 800.0, 450.0).json()["hazard_zones"] == []


def test_zone_changes_are_picked_up_without_stale_cache(client, db_session, seeded, demo_terrain):
    assert sample(client, seeded.site, -2000.0, -2000.0).json()["risk_score"] < 0.2
    zone = HazardZone(
        site_id=seeded.site.id, name="soft ground", zone_type=HazardZoneType.UNCERTAIN_TERRAIN,
        geometry=from_shape(box(-2100, -2100, -1900, -1900), srid=0), risk_score=0.6,
        is_blocking=False, data_origin=DataOrigin.SIMULATED,
    )
    db_session.add(zone)
    db_session.commit()
    body = sample(client, seeded.site, -2000.0, -2000.0).json()
    assert body["risk_score"] == pytest.approx(0.6) and body["blocked"] is False
    assert body["hazard_zones"][0]["name"] == "soft ground" and body["blocked_reasons"] == []
    zone.is_blocking = True
    db_session.commit()
    assert sample(client, seeded.site, -2000.0, -2000.0).json()["blocked"] is True


def test_extent_edges_are_inclusive_and_outside_is_rejected(client, seeded, demo_terrain):
    assert sample(client, seeded.site, 4000.0, 4000.0).status_code == 200
    border = sample(client, seeded.site, -4000.0, -4000.0).json()   # corner: elevation exists, but slope needs a full 3x3
    assert border["is_valid"] is True and border["slope_deg"] is None
    assert border["terrain_category"] == "unknown" and border["blocked"] is True
    for x, y in [(4000.01, 0), (0, -4000.01), (1e9, 0)]:
        response = sample(client, seeded.site, x, y)
        assert response.status_code == 422 and response.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize("params", [
    {"x_m": "nan", "y_m": 0}, {"x_m": "inf", "y_m": 0}, {"x_m": "abc", "y_m": 0}, {"y_m": 0}, {"x_m": 0},
])
def test_invalid_coordinates_are_rejected(client, seeded, demo_terrain, params):
    response = client.get(SAMPLE, params={"site_id": str(seeded.site.id), **params})
    assert response.status_code == 422


def test_unknown_site_and_site_without_terrain(client, seeded):
    assert client.get(SAMPLE, params={"site_id": str(uuid4()), "x_m": 0, "y_m": 0}).status_code == 404
    missing = sample(client, seeded.site, 0, 0)                      # seeded, but nothing processed yet
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "terrain_not_available"
    assert client.get(META, params={"site_id": str(seeded.site.id)}).status_code == 404


def test_explicit_layer_selection_and_unknown_layer(client, seeded, demo_terrain):
    ok = sample(client, seeded.site, 0, 0, elevation_layer_id=str(demo_terrain.elevation.id))
    assert ok.status_code == 200
    bad = sample(client, seeded.site, 0, 0, elevation_layer_id=str(uuid4()))
    assert bad.status_code == 404 and bad.json()["error"]["code"] == "terrain_not_available"
    slope_id = str(demo_terrain.slope.id)                              # only elevation layers are valid selectors
    assert sample(client, seeded.site, 0, 0, elevation_layer_id=slope_id).status_code == 404


def test_planned_candidates_are_never_used_as_terrain(client, db_session, seeded, demo_terrain):
    candidate = db_session.scalars(select(DataLayer).where(DataLayer.slug == "nasa-dem-candidate")).one()
    assert candidate.processing_status.value == "planned"
    assert client.get(META, params={"site_id": str(seeded.site.id)}).json()["elevation_layer"]["slug"] == "demo-elevation"


# --- /terrain/metadata -------------------------------------------------------------------------

def test_metadata_describes_grid_layers_stats_and_limits(client, seeded, demo_terrain):
    response = client.get(META, params={"site_id": str(seeded.site.id)})
    assert response.status_code == 200
    body = response.json()
    assert body["is_demo"] is True
    grid = body["grid"]
    assert (grid["width_px"], grid["height_px"], grid["resolution_m"]) == (400, 400, 20.0)
    assert grid["bounds"] == {"min_x_m": -4000.0, "min_y_m": -4000.0, "max_x_m": 4000.0, "max_y_m": 4000.0}
    assert grid["frame"] == "site_local_enu" and grid["proj"].startswith("+proj=eqc") and "6378137" not in grid["proj"]
    assert grid["transform"] == [20.0, 0.0, -4000.0, 0.0, -20.0, 4000.0]
    assert body["elevation_layer"]["data_origin"] == "simulated" and body["slope_layer"]["units"] == "deg"
    assert 0 < body["elevation_stats"]["valid_fraction"] < 1 and body["slope_stats"]["max"] > 15
    assert set(body["hazard_stats"]["categories"]) == {"low", "moderate", "high", "excluded", "unknown"}
    assert body["hazard_config"]["block_unknown"] is True and body["hazard_config"]["slope_limits_deg"] == [5.0, 10.0, 15.0]
    assert body["notes"][0].startswith("DEMO") and any("astronaut-scale" in n for n in body["notes"])


def test_metadata_unknown_site_404(client):
    assert client.get(META, params={"site_id": str(uuid4())}).status_code == 404


# --- /hazards ------------------------------------------------------------------------------------

def test_hazard_zone_listing(client, seeded):
    body = client.get("/api/v1/hazards").json()
    assert body["total"] == 1
    zone = body["items"][0]
    assert zone["data_origin"] == "simulated" and zone["is_blocking"] and zone["risk_score"] == 1.0
    assert zone["geometry"]["coordinate_system"] == "site_local_enu"
    ring = zone["geometry"]["exterior"]
    assert len(ring) == 5 and ring[0] == ring[-1]
    assert {tuple(p) for p in ring} == {(300.0, 300.0), (700.0, 300.0), (700.0, 600.0), (300.0, 600.0)}
    assert client.get("/api/v1/hazards", params={"zone_type": "restricted"}).json()["total"] == 0
    assert client.get("/api/v1/hazards", params={"is_blocking": "true", "site_id": str(seeded.site.id)}).json()["total"] == 1
    assert client.get("/api/v1/hazards", params={"data_origin": "observed"}).json()["total"] == 0
    assert client.get("/api/v1/hazards", params={"zone_type": "bogus"}).status_code == 422
    assert client.get("/api/v1/hazards", params={"limit": 0}).status_code == 422


# --- layers and previews -------------------------------------------------------------------------

def test_layer_detail_exposes_assets_and_lineage_without_paths(client, seeded, demo_terrain, tmp_path):
    slope = client.get(f"/api/v1/layers/{demo_terrain.slope.id}")
    body = slope.json()
    assert body["derived_from_layer_id"] == str(demo_terrain.elevation.id)
    assert body["processing_status"] == "demo" and body["data_origin"] == "simulated" and body["is_demo"]
    assert {a["variable"] for a in body["assets"]} == {"slope_deg", "preview"}
    asset = next(a for a in body["assets"] if a["variable"] == "slope_deg")
    assert asset["pixel_size_x_m"] == 20.0 and asset["bounds"]["max_x_m"] == 4000.0 and len(asset["sha256"]) == 64
    assert "storage_key" not in slope.text and str(tmp_path) not in slope.text and ".tif" not in slope.text
    assert body["processing_metadata"]["stats"]["max"] > 15


def test_layer_listing_filters_show_processed_demo_layers(client, seeded, demo_terrain):
    slopes = client.get("/api/v1/layers", params={"layer_type": "slope"}).json()
    assert [l["slug"] for l in slopes["items"]] == ["demo-slope"]
    hazards = client.get("/api/v1/layers", params={"layer_type": "hazard_cost"}).json()["items"]
    assert hazards[0]["processing_status"] == "demo" and hazards[0]["spatial_resolution_m"] == 20.0


@pytest.mark.parametrize("which,variable", [("elevation", "elevation_m"), ("slope", "slope_deg"), ("hazard", "risk_score")])
def test_preview_is_served_as_png_with_etag(client, seeded, demo_terrain, which, variable):
    layer = getattr(demo_terrain, which)
    response = client.get(f"/api/v1/layers/{layer.id}/preview")
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n" and "max-age" in response.headers["cache-control"]
    preview = next(a for a in layer.assets if a.variable == "preview")
    assert response.headers["etag"] == f'"{preview.sha256}"'


def test_preview_missing_cases(client, db_session, storage, seeded, demo_terrain):
    candidate = db_session.scalars(select(DataLayer).where(DataLayer.slug == "nasa-dem-candidate")).one()
    assert client.get(f"/api/v1/layers/{candidate.id}/preview").status_code == 404
    assert client.get(f"/api/v1/layers/{uuid4()}/preview").status_code == 404
    preview = next(a for a in demo_terrain.slope.assets if a.variable == "preview")
    storage.resolve(preview.storage_key).unlink()
    gone = client.get(f"/api/v1/layers/{demo_terrain.slope.id}/preview")
    assert gone.status_code == 404 and str(storage.root) not in gone.text


def test_preview_cannot_be_used_to_read_arbitrary_files(client, db_session, storage, seeded, demo_terrain, tmp_path):
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"top secret")
    preview = next(a for a in demo_terrain.slope.assets if a.variable == "preview")
    preview.storage_key = "../secret.png"                                # simulate a tampered DB row
    db_session.commit()
    response = client.get(f"/api/v1/layers/{demo_terrain.slope.id}/preview")
    assert response.status_code == 404 and b"top secret" not in response.content


def test_real_terrain_is_preferred_over_demo_when_both_exist(client, db_session, storage, settings, seeded, demo_terrain):
    from app.services import ingestion_service
    from tests.raster_fixtures import local_crs, plane_raster
    from tests.test_terrain_pipeline import make_job

    plane_raster(settings.ingest_inbox / "dem.tif", local_crs(), (-4000, -4000, 4000, 4000), 20.0, 0.05, 0.0, z0=100.0)
    ingestion_service.ingest_job(db_session, storage, settings, make_job(resolution_m=20.0))
    body = sample(client, seeded.site, 0.0, 0.0).json()
    assert body["provenance"]["elevation_layer"]["is_demo"] is False
    assert body["provenance"]["elevation_layer"]["data_origin"] == "observed"
    assert not body["notes"][0].startswith("DEMO")
    assert body["slope_deg"] == pytest.approx(np.degrees(np.arctan(0.05)), abs=0.05)
    forced = sample(client, seeded.site, 0.0, 0.0, elevation_layer_id=str(demo_terrain.elevation.id)).json()
    assert forced["provenance"]["elevation_layer"]["is_demo"] is True
