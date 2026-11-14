import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

from app.processing.crs import validate_mars_crs
from app.processing.errors import ProcessingError, SourceValidationError
from app.processing.grid import TargetGrid, mars_local_crs
from app.utils.mars_coords import geographic_to_local, local_to_geographic
from tests.raster_fixtures import RADIUS, SITE_LAT, SITE_LON, local_crs, make_grid, mars_geographic_crs


def test_local_frame_proj_matches_analytic_mars_conversion():
    geog = mars_geographic_crs()
    for x, y in [(0.0, 0.0), (1234.5, -987.6), (-3999.0, 3999.0)]:
        lon, lat = warp_transform(local_crs(), geog, [x], [y])
        exp_lon, exp_lat = local_to_geographic(SITE_LON, SITE_LAT, x, y)
        assert (lon[0], lat[0]) == pytest.approx((exp_lon, exp_lat), abs=1e-9)
        assert geographic_to_local(SITE_LON, SITE_LAT, exp_lon, exp_lat) == pytest.approx((x, y), abs=1e-6)


def test_local_frame_is_not_an_earth_crs():
    wkt = local_crs().to_wkt()
    assert "3396190" in wkt and "6378137" not in wkt


def test_grid_geometry_and_cell_lookup():
    grid = make_grid(400, 200, 10)
    assert grid.shape == (20, 40) and grid.bounds == (-200, -100, 200, 100)
    assert grid.cell_of(-200, 100) == (0, 0)          # north-west corner
    assert grid.cell_of(199.99, -99.99) == (19, 39)
    assert grid.cell_of(200, -100) == (19, 39)        # far edges are inclusive
    assert grid.cell_of(200.01, 0) is None and grid.cell_of(0, 100.01) is None
    x, y = grid.cell_center(3, 5)
    assert grid.cell_of(x, y) == (3, 5) and (x, y) == (-145.0, 65.0)
    t = grid.transform
    assert (t.a, t.e, t.c, t.f) == (10, -10, -200, 100)


@pytest.mark.parametrize("width,height,res", [(405, 400, 10), (400, 400, 0), (5, 5, 10)])
def test_grid_rejects_bad_dimensions(width, height, res):
    with pytest.raises(ProcessingError):
        TargetGrid.build(width, height, res, local_crs(), 10**7)


def test_grid_enforces_cell_budget():
    with pytest.raises(ProcessingError, match="exceeds"):
        TargetGrid.build(8000, 8000, 10, local_crs(), 1000)


def test_mars_crs_accepts_geographic_and_projected():
    info, warnings = validate_mars_crs(mars_geographic_crs(), RADIUS)
    assert info.is_sphere and not info.is_projected and warnings == []
    info, _ = validate_mars_crs(local_crs(), RADIUS)
    assert info.is_projected and info.semi_major_m == RADIUS


@pytest.mark.parametrize("earth", [CRS.from_epsg(4326), CRS.from_epsg(3857), CRS.from_epsg(32643)])
def test_earth_crs_is_rejected(earth):
    with pytest.raises(SourceValidationError, match="not a Mars CRS"):
        validate_mars_crs(earth, RADIUS)


def test_missing_crs_is_rejected_not_guessed():
    with pytest.raises(SourceValidationError, match="no CRS"):
        validate_mars_crs(None, RADIUS)


def test_different_mars_reference_radius_is_rejected():
    with pytest.raises(SourceValidationError, match="differs from the site frame"):
        validate_mars_crs(mars_geographic_crs(3_389_500.0), RADIUS)


def test_ellipsoid_requires_explicit_opt_in_and_warns():
    ellipsoid = CRS.from_string("+proj=longlat +a=3396190 +b=3376200 +no_defs")
    with pytest.raises(SourceValidationError, match="ellipsoid"):
        validate_mars_crs(ellipsoid, RADIUS)
    info, warnings = validate_mars_crs(ellipsoid, RADIUS, accept_ellipsoid_as_sphere=True)
    assert not info.is_sphere and "positional error" in warnings[0]


def test_non_metre_projected_units_are_rejected():
    feet = CRS.from_string(f"+proj=eqc +lat_ts=18.38 +lat_0=18.38 +lon_0=77.58 +R={RADIUS} +units=ft +no_defs")
    with pytest.raises(SourceValidationError, match="not metres"):
        validate_mars_crs(feet, RADIUS)
