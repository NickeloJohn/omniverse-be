import numpy as np
import pytest
from affine import Affine

from app.processing.errors import SourceValidationError
from app.processing.source import inspect_source, warp_to_grid
from app.processing.terrain import compute_slope_deg
from tests.raster_fixtures import (
    RADIUS, expected_slope_deg, local_crs, make_grid, mars_geographic_crs, mars_sinusoidal_crs,
    plane_raster, write_raster,
)

import math

BOUNDS = (-200.0, -200.0, 200.0, 200.0)


def inspect(path, grid=None, **kwargs):
    kwargs.setdefault("expected_units", None)
    return inspect_source(path, grid or make_grid(), RADIUS, "elevation", **kwargs)


# --- source validation -------------------------------------------------------------------------

def test_valid_source_is_summarised(tmp_path):
    path = plane_raster(tmp_path / "dem.tif", local_crs(), BOUNDS, 5.0, 0.1, 0.0)
    info = inspect(path)
    assert info.units == "m" and info.units_source == "raster" and info.nodata == -9999.0
    assert info.native_resolution_m == pytest.approx(5.0) and info.bbox_overlap_fraction == 1.0
    assert info.crs.is_projected and info.warnings == []
    assert info.summary()["crs"]["semi_major_m"] == RADIUS


def test_source_without_crs_is_rejected(tmp_path):
    path = write_raster(tmp_path / "nocrs.tif", np.zeros((10, 10), "float32"), None,
                        Affine(10, 0, -50, 0, -10, 50), units="m")
    with pytest.raises(SourceValidationError, match="no CRS"):
        inspect(path)


def test_earth_source_is_rejected(tmp_path):
    from rasterio.crs import CRS
    path = write_raster(tmp_path / "earth.tif", np.zeros((10, 10), "float32"), CRS.from_epsg(4326),
                        Affine(0.01, 0, 0, 0, -0.01, 1), units="m")
    with pytest.raises(SourceValidationError, match="not a Mars CRS"):
        inspect(path)


def test_units_must_be_declared_somewhere(tmp_path):
    path = plane_raster(tmp_path / "dem.tif", local_crs(), BOUNDS, 5.0, 0.1, 0.0, units=None)
    with pytest.raises(SourceValidationError, match="units are not declared"):
        inspect(path)
    assert inspect(path, expected_units="m").units_source == "job"


@pytest.mark.parametrize("declared,expected", [("ft", None), (None, "km"), ("m", "km")])
def test_non_metre_or_conflicting_units_are_rejected(tmp_path, declared, expected):
    path = plane_raster(tmp_path / "dem.tif", local_crs(), BOUNDS, 5.0, 0.1, 0.0, units=declared)
    with pytest.raises(SourceValidationError):
        inspect(path, expected_units=expected)


def test_source_outside_site_is_rejected(tmp_path):
    path = plane_raster(tmp_path / "far.tif", local_crs(), (5000, 5000, 5400, 5400), 5.0, 0.1, 0.0)
    with pytest.raises(SourceValidationError, match="does not overlap"):
        inspect(path)


def test_partial_coverage_is_reported(tmp_path):
    path = plane_raster(tmp_path / "half.tif", local_crs(), (-200, -200, 0, 200), 5.0, 0.1, 0.0, margin_cells=0)
    info = inspect(path)
    assert info.bbox_overlap_fraction == pytest.approx(0.5, abs=0.02)
    assert any("covers only" in w for w in info.warnings)


def test_unreadable_source_is_a_validation_error(tmp_path):
    bad = tmp_path / "bad.tif"
    bad.write_bytes(b"not a raster")
    with pytest.raises(SourceValidationError, match="could not be opened"):
        inspect(bad)


# --- alignment, units and projection correctness ---------------------------------------------

def test_same_crs_plane_is_aligned_with_exact_values(tmp_path):
    grid = make_grid()
    path = plane_raster(tmp_path / "dem.tif", local_crs(), BOUNDS, 5.0, 0.1, -0.05, z0=100.0)
    info = inspect(path, grid)
    out = warp_to_grid(path, grid, "elevation", info.native_resolution_m)
    assert out.data.shape == (1, *grid.shape) and out.valid.all()
    xs = np.array([grid.cell_center(0, c)[0] for c in range(grid.width_px)])
    ys = np.array([grid.cell_center(r, 0)[1] for r in range(grid.height_px)])
    expected = 100.0 + 0.1 * xs[None, :] - 0.05 * ys[:, None]
    assert np.allclose(out.data[0], expected, atol=0.05)
    assert out.resampling == "average"  # 5 m source -> 10 m target


@pytest.mark.parametrize("crs_factory,cell", [
    (mars_sinusoidal_crs, 4.0),
    (mars_geographic_crs, 4.0 / (RADIUS * math.pi / 180)),  # ~4 m expressed in degrees
])
def test_reprojection_preserves_metric_slope(tmp_path, crs_factory, cell):
    """A plane sampled in another Mars CRS must keep its slope after alignment to the local frame."""
    grid = make_grid(400, 400, 10)
    a, b = 0.08, -0.06
    path = plane_raster(tmp_path / "dem.tif", crs_factory(), BOUNDS, cell, a, b, z0=50.0)
    info = inspect(path, grid)
    out = warp_to_grid(path, grid, "elevation", info.native_resolution_m)
    assert out.valid.all()
    slope = compute_slope_deg(out.data[0], grid.resolution_m, grid.resolution_m)
    interior = slope[2:-2, 2:-2]
    assert np.isfinite(interior).all()
    assert interior.mean() == pytest.approx(expected_slope_deg(a, b), abs=0.1)
    row, col = grid.cell_of(0, 0)
    cx, cy = grid.cell_center(row, col)
    assert out.data[0][row, col] == pytest.approx(50.0 + a * cx + b * cy, abs=0.1)


def test_upsampling_uses_bilinear(tmp_path):
    grid = make_grid()
    path = plane_raster(tmp_path / "coarse.tif", local_crs(), BOUNDS, 25.0, 0.1, 0.0)
    out = warp_to_grid(path, grid, "elevation", 25.0)
    assert out.resampling == "bilinear" and out.valid.all()


def test_scale_and_offset_are_applied(tmp_path):
    grid = make_grid()
    raw = np.full((60, 60), 1000, dtype=np.int16)
    path = write_raster(tmp_path / "scaled.tif", raw, local_crs(), Affine(10, 0, -300, 0, -10, 300),
                        nodata=-32768, units="m", scale=0.5, offset=10.0)
    out = warp_to_grid(path, grid, "elevation", 10.0)
    assert np.allclose(out.data[0][out.valid], 510.0)


# --- NoData ------------------------------------------------------------------------------------

def test_source_nodata_stays_nodata_with_exact_footprint(tmp_path):
    grid = make_grid()
    z = np.full((60, 60), 100.0, dtype=np.float32)
    z[20:30, 20:30] = -9999.0                              # 10x10 source-cell hole
    path = write_raster(tmp_path / "hole.tif", z, local_crs(), Affine(10, 0, -300, 0, -10, 300),
                        nodata=-9999.0, units="m")
    out = warp_to_grid(path, grid, "elevation", 10.0)
    hole = np.zeros(grid.shape, bool)
    hole[20 - 10 : 30 - 10, 20 - 10 : 30 - 10] = True      # same cells, shifted by the 10-cell margin
    assert (~out.valid == hole).all()
    assert np.isnan(out.data[0][hole]).all()
    assert np.allclose(out.data[0][~hole], 100.0)          # no values leak into or out of the hole


def test_nan_values_are_treated_as_nodata_even_without_nodata_tag(tmp_path):
    grid = make_grid()
    z = np.full((60, 60), 5.0, dtype=np.float32)
    z[30, 30] = np.nan
    path = write_raster(tmp_path / "nan.tif", z, local_crs(), Affine(10, 0, -300, 0, -10, 300), units="m")
    out = warp_to_grid(path, grid, "elevation", 10.0)
    assert (~out.valid).sum() == 1


def test_cells_outside_source_coverage_are_nodata(tmp_path):
    grid = make_grid()
    path = plane_raster(tmp_path / "half.tif", local_crs(), (-200, -200, 0, 200), 5.0, 0.1, 0.0, margin_cells=0)
    out = warp_to_grid(path, grid, "elevation", 5.0)
    left, right = out.valid[:, : grid.width_px // 2], out.valid[:, grid.width_px // 2 + 1 :]
    assert left.all() and not right.any()
    assert np.isnan(out.data[0][~out.valid]).all()


def test_coarse_cell_touching_nodata_is_not_declared_valid(tmp_path):
    """Downsampling must not average valid and invalid cells into a 'valid' value."""
    grid = make_grid()
    z = np.full((120, 120), 7.0, dtype=np.float32)
    z[60, 60] = -9999.0  # source cell covering x 0..5 m, y -5..0 m: inside the site
    path = write_raster(tmp_path / "fine.tif", z, local_crs(), Affine(5, 0, -300, 0, -5, 300),
                        nodata=-9999.0, units="m")
    out = warp_to_grid(path, grid, "elevation", 5.0)
    assert out.resampling == "average" and (~out.valid).sum() >= 1
    assert np.isnan(out.data[0][~out.valid]).all()


def test_imagery_alignment_keeps_bands_and_mask(tmp_path):
    grid = make_grid()
    img = np.stack([np.full((60, 60), v, np.uint8) for v in (10, 20, 30)])
    img[:, 25:35, 25:35] = 0
    path = write_raster(tmp_path / "img.tif", img, local_crs(), Affine(10, 0, -300, 0, -10, 300), nodata=0)
    info = inspect_source_kind(path, grid)
    out = warp_to_grid(path, grid, "imagery", info.native_resolution_m)
    assert out.data.shape == (3, *grid.shape)
    assert [float(out.data[i][out.valid].mean()) for i in range(3)] == [10.0, 20.0, 30.0]
    assert (~out.valid).sum() == 100


def inspect_source_kind(path, grid):
    return inspect_source(path, grid, RADIUS, "imagery", None)
