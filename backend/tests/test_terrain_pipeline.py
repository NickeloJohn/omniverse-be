import json
from datetime import date

import numpy as np
import pytest
from sqlalchemy import func, select

from app.models import DataLayer, RasterAsset
from app.models.enums import DataOrigin, LayerType, ProcessingStatus, RasterRole
from app.processing.errors import ProcessingError, SourceValidationError
from app.processing.hazard import HazardConfig
from app.processing.io import read_geotiff
from app.schemas.ingestion import IngestionJob, RasterSource
from app.services import ingestion_service
from app.services.terrain_service import load_terrain
from tests.raster_fixtures import local_crs, mars_sinusoidal_crs, plane_raster, write_raster

import httpx
from affine import Affine
from rasterio.crs import CRS

SITE_BOUNDS = (-4000.0, -4000.0, 4000.0, 4000.0)


def layer_count(session) -> int:
    return session.scalar(select(func.count()).select_from(DataLayer))


def make_job(**overrides) -> IngestionJob:
    source = RasterSource(
        name="Synthetic TEST DEM (not NASA data)", local_file="dem.tif", product_id="SYNTHETIC-TEST",
        acquisition_date=date(2024, 1, 2), citation="unit-test fixture",
    )
    fields = {"site_slug": "jezero-demo", "layer_prefix": "test-dem", "resolution_m": 100.0, "elevation": source}
    return IngestionJob(**{**fields, **overrides})


@pytest.fixture
def dem_file(settings):
    """Plane z = 0.1 x (5.71 deg) in a sinusoidal Mars CRS with a NoData hole, saved in the ingest inbox."""
    path = plane_raster(settings.ingest_inbox / "dem.tif", mars_sinusoidal_crs(), SITE_BOUNDS, 50.0, 0.1, 0.0, z0=500.0)
    import rasterio
    with rasterio.open(path, "r+") as dst:
        data = dst.read(1)
        data[100:110, 100:110] = -9999.0
        dst.write(data, 1)
    return path


# --- DEMO terrain -----------------------------------------------------------------------------------

def test_demo_terrain_is_registered_as_simulated_with_lineage(db_session, demo_terrain, storage):
    elevation, slope, hazard = demo_terrain.elevation, demo_terrain.slope, demo_terrain.hazard
    for layer in (elevation, slope, hazard):
        assert layer.is_demo and layer.data_origin == DataOrigin.SIMULATED
        assert layer.processing_status == ProcessingStatus.DEMO
        assert layer.spatial_resolution_m == 20.0 and layer.crs.startswith("+proj=eqc")
        assert "NASA" in json.dumps(layer.processing_metadata) or layer.processing_metadata.get("demo_note")
    assert slope.derived_from_layer_id == hazard.derived_from_layer_id == elevation.id
    assert (elevation.slug, slope.slug, hazard.slug) == ("demo-elevation", "demo-slope", "demo-hazard-cost")
    assert elevation.source_name == "MARS-NAV synthetic terrain generator" and elevation.source_url is None
    assert elevation.source_product_id is None and elevation.acquisition_date is None
    assert "not derived from any NASA product" in elevation.processing_metadata["demo_note"]


def test_demo_assets_match_files_on_disk(db_session, demo_terrain, storage):
    assets = db_session.scalars(select(RasterAsset)).all()
    variables = sorted((a.variable, a.role.value) for a in assets)
    assert variables == sorted([
        ("elevation_m", "processed"), ("preview", "display_tiles"), ("slope_deg", "processed"), ("preview", "display_tiles"),
        ("risk_score", "processed"), ("hazard_classes", "processed"), ("preview", "display_tiles"),
    ])
    for asset in assets:
        path = storage.resolve(asset.storage_key)
        assert path.is_file() and path.stat().st_size == asset.size_bytes and storage.sha256(path) == asset.sha256
        assert not asset.storage_key.startswith("/") and (asset.width_px, asset.height_px) == (400, 400)
        assert asset.transform == [20.0, 0.0, -4000.0, 0.0, -20.0, 4000.0]
        assert asset.pixel_size_x_m == 20.0 and asset.pixel_size_y_m == 20.0


def test_stored_rasters_preserve_nodata_and_share_one_grid(db_session, demo_terrain, storage):
    data = load_terrain(db_session, storage, demo_terrain.elevation.site)
    assert data.elevation.shape == data.slope.shape == data.terrain.risk.shape == (400, 400)
    unknown = ~np.isfinite(data.elevation)
    assert unknown.any()
    assert np.isnan(data.slope[unknown]).all()                              # no slope where terrain is unknown
    assert data.terrain.blocked[unknown].all() and (data.terrain.category[unknown] == 255).all()
    elevation_tif = next(a for a in demo_terrain.elevation.assets if a.variable == "elevation_m")
    raw, meta = read_geotiff(storage.resolve(elevation_tif.storage_key))
    assert meta["nodata"] == -9999.0 and np.isnan(raw[0][unknown]).all()


def test_hazard_layer_records_config_stats_and_disclaimers(demo_terrain):
    meta = demo_terrain.hazard.processing_metadata
    assert meta["hazard_config"]["slope_limits_deg"] == [5.0, 10.0, 15.0]
    assert meta["hazard_config"]["block_unknown"] is True
    assert sum(v for k, v in meta["stats"]["categories"].items()) == pytest.approx(1.0, abs=1e-4)
    assert any("not validated NASA astronaut safety limits" in d for d in meta["disclaimers"])
    assert meta["derived_from_slug"] == "demo-elevation"
    slope_meta = demo_terrain.slope.processing_metadata
    assert slope_meta["stats"]["max"] > 15 and slope_meta["stats"]["valid_fraction"] < 1


def test_demo_processing_is_idempotent(db_session, demo_terrain, storage, settings, seeded):
    before = (layer_count(db_session), db_session.scalar(select(func.count()).select_from(RasterAsset)))
    ingestion_service.process_demo_terrain(db_session, storage, settings, seeded.site.slug, resolution_m=20.0)
    after = (layer_count(db_session), db_session.scalar(select(func.count()).select_from(RasterAsset)))
    assert before == after == (5, 7)


def test_demo_site_must_exist(db_session, storage, settings):
    with pytest.raises(ProcessingError, match="not found"):
        ingestion_service.process_demo_terrain(db_session, storage, settings, "nope", 20.0)


def test_demo_resolution_must_divide_site_extent(db_session, storage, settings, seeded):
    with pytest.raises(ProcessingError, match="whole number"):
        ingestion_service.process_demo_terrain(db_session, storage, settings, seeded.site.slug, 30.0)
    assert not (storage.root / "sites").exists()


# --- re-deriving -------------------------------------------------------------------------------------

def test_new_hazard_config_creates_new_hazard_version_and_keeps_slope(db_session, demo_terrain, storage, settings):
    old_version = demo_terrain.hazard.dataset_version
    slope_assets = {a.id for a in demo_terrain.slope.assets}
    strict = HazardConfig(version="strict", slope_limits_deg=(3, 6, 9), risk_at_limits=(0.3, 0.6, 0.9))
    result = ingestion_service.regenerate_hazard(db_session, storage, settings, "jezero-demo", "demo-elevation", strict)
    assert result.hazard.dataset_version != old_version and result.hazard.id == demo_terrain.hazard.id
    assert result.hazard.processing_metadata["hazard_config"]["version"] == "strict"
    assert {a.id for a in result.slope.assets} == slope_assets                       # untouched
    old_dir = storage.resolve(f"sites/jezero-demo/demo-hazard-cost/{old_version}")
    assert old_dir.is_dir()                                                          # previous version is kept
    data = load_terrain(db_session, storage, result.elevation.site)
    assert data.config.version == "strict" and data.config.excluded_above_deg == 9


def test_regenerate_slope_reproduces_the_same_slope_from_stored_elevation(db_session, demo_terrain, storage, settings):
    before = load_terrain(db_session, storage, demo_terrain.elevation.site).slope.copy()
    ingestion_service.regenerate_slope(db_session, storage, settings, "jezero-demo", "demo-elevation")
    from app.services import terrain_service
    terrain_service.clear_cache()
    after = load_terrain(db_session, storage, demo_terrain.elevation.site).slope
    assert np.array_equal(before, after, equal_nan=True)


def test_regenerate_requires_existing_layers(db_session, storage, settings, seeded):
    with pytest.raises(ProcessingError, match="not found"):
        ingestion_service.regenerate_slope(db_session, storage, settings, "jezero-demo", "missing")
    with pytest.raises(ProcessingError, match="No slope layer"):
        ingestion_service.regenerate_hazard(db_session, storage, settings, "jezero-demo", "demo-elevation", HazardConfig())


# --- NASA-style ingestion (local synthetic file) ---------------------------------------------------

def test_ingestion_registers_observed_dem_with_provenance(db_session, storage, settings, seeded, dem_file):
    result = ingestion_service.ingest_job(db_session, storage, settings, make_job())
    elevation, slope, hazard = result.elevation, result.slope, result.hazard
    assert elevation.data_origin == DataOrigin.OBSERVED and elevation.processing_status == ProcessingStatus.PROCESSED
    assert not elevation.is_demo and elevation.slug == "test-dem-elevation"
    assert (slope.data_origin, hazard.data_origin) == (DataOrigin.DERIVED, DataOrigin.DERIVED)
    assert slope.processing_status == hazard.processing_status == ProcessingStatus.PROCESSED
    assert elevation.source_name.startswith("Synthetic TEST DEM") and elevation.source_product_id == "SYNTHETIC-TEST"
    assert elevation.acquisition_date == date(2024, 1, 2) and elevation.units == "m"
    assert elevation.spatial_resolution_m == 100.0 and elevation.nodata_value == -9999.0

    meta = elevation.processing_metadata
    assert meta["source"]["original"]["crs"]["semi_major_m"] == 3396190.0
    assert meta["source"]["original"]["native_resolution_m"] == pytest.approx(50.0)
    assert meta["source"]["acquired_via"] == "local_file" and len(meta["source"]["sha256"]) == 64
    assert [s["step"] for s in meta["steps"]] == ["acquire", "validate", "crop_align"]
    assert meta["steps"][2]["resampling"] == "average"
    assert 0.9 < meta["stats"]["valid_fraction"] < 1.0
    assert any("NoData" in w for w in meta["warnings"])
    assert elevation.dataset_version.startswith("src-") and slope.dataset_version == elevation.dataset_version


def test_ingested_terrain_is_analytically_correct(db_session, storage, settings, seeded, dem_file):
    result = ingestion_service.ingest_job(db_session, storage, settings, make_job())
    data = load_terrain(db_session, storage, seeded.site)
    row, col = data.grid.cell_of(1000.0, -500.0)
    cx, cy = data.grid.cell_center(row, col)
    assert data.elevation[row, col] == pytest.approx(500.0 + 0.1 * cx, abs=0.2)
    interior = data.slope[3:-3, 3:-3]
    finite = interior[np.isfinite(interior)]
    assert finite.mean() == pytest.approx(np.degrees(np.arctan(0.1)), abs=0.05)
    assert (data.terrain.category[3:-3, 3:-3][np.isfinite(interior)] == 1).all()     # 5.7 deg -> moderate


def test_ingestion_nodata_hole_is_unknown_and_blocked(db_session, storage, settings, seeded, dem_file):
    ingestion_service.ingest_job(db_session, storage, settings, make_job())
    data = load_terrain(db_session, storage, seeded.site)
    # source cells 100..109 (50 m) -> a ~5x5 hole in the 100 m grid, plus a 1-cell slope margin
    unknown = ~np.isfinite(data.elevation)
    assert 4 <= unknown.sum() <= 36
    ys, xs = np.where(unknown)
    assert data.terrain.blocked[ys, xs].all() and np.isnan(data.slope[ys, xs]).all()
    margin = ~np.isfinite(data.slope) & np.isfinite(data.elevation)
    interior_margin = margin[2:-2, 2:-2]
    assert interior_margin.sum() > 0                                                  # slope margin around the hole


def test_ingestion_is_idempotent_and_versioned_by_content(db_session, storage, settings, seeded, dem_file):
    first = ingestion_service.ingest_job(db_session, storage, settings, make_job())
    count = db_session.scalar(select(func.count()).select_from(RasterAsset))
    second = ingestion_service.ingest_job(db_session, storage, settings, make_job())
    assert second.elevation.id == first.elevation.id
    assert db_session.scalar(select(func.count()).select_from(RasterAsset)) == count
    other = ingestion_service.ingest_job(db_session, storage, settings, make_job(dataset_version="v2-manual"))
    assert other.elevation.dataset_version == "v2-manual" and other.elevation.id == first.elevation.id
    assert db_session.scalar(select(func.count()).select_from(RasterAsset)) == count + 7   # a full new version set


def test_candidate_layer_is_promoted_not_duplicated(db_session, storage, settings, seeded, dem_file):
    candidate = db_session.scalars(select(DataLayer).where(DataLayer.slug == "nasa-dem-candidate")).one()
    source = make_job().elevation.model_copy(update={"candidate_slug": "nasa-dem-candidate"})
    result = ingestion_service.ingest_job(db_session, storage, settings, make_job(elevation=source))
    assert result.elevation.id == candidate.id and candidate.slug == "test-dem-elevation"
    assert candidate.processing_status == ProcessingStatus.PROCESSED
    assert not db_session.scalars(select(DataLayer).where(DataLayer.slug == "nasa-dem-candidate")).first()


def test_promoting_a_missing_or_processed_candidate_fails(db_session, storage, settings, seeded, dem_file, demo_terrain):
    source = make_job().elevation.model_copy(update={"candidate_slug": "demo-elevation"})   # already processed
    with pytest.raises(ProcessingError, match="unprocessed planned candidate"):
        ingestion_service.ingest_job(db_session, storage, settings, make_job(elevation=source))


def test_imagery_is_aligned_to_the_elevation_grid(db_session, storage, settings, seeded, dem_file):
    img = np.stack([np.full((100, 100), v, np.uint8) for v in (40, 80, 120)])
    img[:, 40:60, 40:60] = 0
    write_raster(settings.ingest_inbox / "img.tif", img, local_crs(), Affine(80, 0, -4000, 0, -80, 4000), nodata=0)
    imagery = RasterSource(name="Synthetic TEST imagery", local_file="img.tif")
    result = ingestion_service.ingest_job(db_session, storage, settings, make_job(imagery=imagery))
    layer = result.imagery
    assert layer.layer_type == LayerType.OPTICAL_IMAGERY and layer.data_origin == DataOrigin.OBSERVED
    assert layer.processing_metadata["aligned_to_elevation_version"] == result.elevation.dataset_version
    asset = next(a for a in layer.assets if a.variable == "imagery")
    data, _ = read_geotiff(storage.resolve(asset.storage_key))
    assert data.shape == (3, 80, 80) and asset.transform == result.elevation.assets[0].transform
    valid = np.isfinite(data[0]) & (data[0] != -9999.0)
    assert valid.sum() == 6400 - 256   # the 20x20-source-pixel (1600 m) hole is exactly 16x16 target cells
    assert [float(data[i][valid].mean()) for i in range(3)] == [40.0, 80.0, 120.0]
    assert any(a.role == RasterRole.DISPLAY_TILES for a in layer.assets)


def test_metadata_never_leaks_filesystem_paths(db_session, storage, settings, seeded, dem_file, tmp_path):
    result = ingestion_service.ingest_job(db_session, storage, settings, make_job())
    blob = json.dumps([l.processing_metadata for l in (result.elevation, result.slope, result.hazard)], default=str)
    blob += json.dumps([a.processing_metadata for l in (result.elevation, result.slope, result.hazard) for a in l.assets], default=str)
    assert str(tmp_path) not in blob and "/private" not in blob and str(storage.root) not in blob


# --- validation failures leave nothing behind ----------------------------------------------------

def assert_nothing_registered(session, storage, expected_layers=4):
    assert layer_count(session) == expected_layers                                     # only the seeded layers
    assert not (storage.root / "sites").exists() or not any((storage.root / "sites").rglob("*.tif"))


def test_earth_crs_source_is_rejected_atomically(db_session, storage, settings, seeded):
    write_raster(settings.ingest_inbox / "dem.tif", np.zeros((50, 50), np.float32), CRS.from_epsg(4326),
                 Affine(0.01, 0, 0, 0, -0.01, 1), units="m")
    with pytest.raises(SourceValidationError, match="not a Mars CRS"):
        ingestion_service.ingest_job(db_session, storage, settings, make_job())
    assert_nothing_registered(db_session, storage)


def test_missing_units_source_is_rejected_atomically(db_session, storage, settings, seeded):
    plane_raster(settings.ingest_inbox / "dem.tif", local_crs(), SITE_BOUNDS, 100.0, 0.1, 0.0, units=None)
    with pytest.raises(SourceValidationError, match="units"):
        ingestion_service.ingest_job(db_session, storage, settings, make_job())
    assert_nothing_registered(db_session, storage)


def test_insufficient_coverage_is_rejected(db_session, storage, settings, seeded):
    plane_raster(settings.ingest_inbox / "dem.tif", local_crs(), (-4000, -4000, -3000, -3000), 100.0, 0.1, 0.0, margin_cells=0)
    with pytest.raises(SourceValidationError, match="valid elevation"):
        ingestion_service.ingest_job(db_session, storage, settings, make_job())
    assert_nothing_registered(db_session, storage)


def test_partial_coverage_above_threshold_warns(db_session, storage, settings, seeded):
    plane_raster(settings.ingest_inbox / "dem.tif", local_crs(), (-4000, -4000, 2000, 4000), 100.0, 0.1, 0.0, margin_cells=0)
    result = ingestion_service.ingest_job(db_session, storage, settings, make_job(min_valid_fraction=0.5))
    assert any("covers only" in w for w in result.warnings) and any("NoData" in w for w in result.warnings)


def test_failure_while_writing_rolls_back_database_and_files(db_session, storage, settings, seeded, dem_file, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(ingestion_service, "write_category_preview", boom)
    with pytest.raises(OSError):
        ingestion_service.ingest_job(db_session, storage, settings, make_job())
    assert_nothing_registered(db_session, storage)
    assert db_session.scalar(select(func.count()).select_from(RasterAsset)) == 0


def test_unknown_site_and_inbox_escape_are_rejected(db_session, storage, settings, seeded, dem_file):
    with pytest.raises(ProcessingError, match="not found"):
        ingestion_service.ingest_job(db_session, storage, settings, make_job(site_slug="atlantis"))
    escape = make_job().elevation.model_copy(update={"local_file": "../outside.tif"})
    with pytest.raises(SourceValidationError):
        ingestion_service.ingest_job(db_session, storage, settings, make_job(elevation=escape))


def test_job_schema_rejects_ambiguous_or_unknown_fields():
    base = {"site_slug": "s", "layer_prefix": "ok-prefix"}
    with pytest.raises(ValueError):
        IngestionJob(**base, elevation={"name": "x"})                                  # no location
    with pytest.raises(ValueError):
        IngestionJob(**base, elevation={"name": "x", "url": "https://trek.nasa.gov/a.tif", "local_file": "a.tif"})
    with pytest.raises(ValueError):
        IngestionJob(**base, elevation={"name": "x", "local_file": "a.tif", "surprise": 1})
    with pytest.raises(ValueError):
        IngestionJob(site_slug="s", layer_prefix="Bad Prefix!", elevation={"name": "x", "local_file": "a.tif"})


def test_download_job_uses_allowlist_and_records_provenance(db_session, storage, settings, seeded, dem_file):
    payload = dem_file.read_bytes()
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=payload)

    url = "https://trek.nasa.gov/synthetic/dem.tif"
    job = make_job(elevation=RasterSource(name="Synthetic download", url=url, product_id="SYNTHETIC-DL"))
    result = ingestion_service.ingest_job(db_session, storage, settings, job, transport=httpx.MockTransport(handler))
    assert seen == [url]
    assert result.elevation.source_url == url
    assert result.elevation.processing_metadata["source"]["acquired_via"] == "download"
    assert (storage.root / "downloads" / "dem.tif").is_file()

    blocked = make_job(elevation=RasterSource(name="bad", url="https://evil.example/dem.tif"))
    with pytest.raises(ProcessingError, match="allowlist"):
        ingestion_service.ingest_job(db_session, storage, settings, blocked, transport=httpx.MockTransport(handler))
    assert seen == [url]                                                                # no request was made
