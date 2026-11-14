import math
from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError
from shapely.geometry import box

from app.processing.demo import generate_demo_elevation
from app.processing.hazard import (
    CATEGORY_EXCLUDED, CATEGORY_HIGH, CATEGORY_LOW, CATEGORY_MODERATE, CATEGORY_UNKNOWN,
    DEFAULT_HAZARD_CONFIG, HazardConfig, HazardZoneSpec, apply_hazard_zones, array_stats,
    category_fractions, classify_terrain,
)
from app.processing.io import FLOAT_NODATA, read_geotiff, write_category_preview, write_geotiff, write_rgb_preview, write_scalar_preview
from app.processing.terrain import compute_slope_deg, max_neighbor_step_m
from tests.raster_fixtures import make_grid


def ramp(rows=9, cols=9, dzdx=0.0, dzdy=0.0, cell=10.0):
    r, c = np.indices((rows, cols))
    return (c * cell * dzdx + (rows - 1 - r) * cell * dzdy).astype(np.float32)


# --- slope -------------------------------------------------------------------------------------

def test_flat_terrain_has_zero_slope_and_borders_are_unknown():
    slope = compute_slope_deg(np.full((7, 7), 123.0, np.float32), 10, 10)
    assert np.allclose(slope[1:-1, 1:-1], 0.0)
    assert np.isnan(slope[0]).all() and np.isnan(slope[-1]).all() and np.isnan(slope[:, 0]).all()


@pytest.mark.parametrize("gradient", [0.05, 0.1763269807, 1.0, 2.0])
def test_slope_matches_analytic_plane_in_any_direction(gradient):
    for dzdx, dzdy in [(gradient, 0), (0, gradient), (-gradient, 0), (gradient * 0.6, gradient * 0.8)]:
        slope = compute_slope_deg(ramp(dzdx=dzdx, dzdy=dzdy), 10, 10)
        assert np.allclose(slope[1:-1, 1:-1], math.degrees(math.atan(math.hypot(dzdx, dzdy))), atol=1e-4)


def test_known_ten_degree_ramp():
    slope = compute_slope_deg(ramp(dzdx=math.tan(math.radians(10))), 10, 10)
    assert slope[4, 4] == pytest.approx(10.0, abs=1e-4)


def test_slope_uses_cell_size_for_unit_correctness():
    z = ramp(dzdx=1.0, cell=10.0)           # +10 m per 10 m cell
    assert compute_slope_deg(z, 10, 10)[4, 4] == pytest.approx(45.0, abs=1e-4)
    assert compute_slope_deg(z, 20, 20)[4, 4] == pytest.approx(math.degrees(math.atan(0.5)), abs=1e-4)


def test_nodata_invalidates_exactly_its_3x3_neighbourhood():
    z = np.zeros((9, 9), np.float32)
    z[4, 4] = np.nan
    slope = compute_slope_deg(z, 10, 10)
    assert np.isnan(slope[3:6, 3:6]).all()
    assert np.isfinite(slope[1:3, 1:-1]).all() and np.isfinite(slope[6:8, 1:-1]).all()
    assert np.isfinite(slope[1:-1, 1:3]).all()
    assert np.isnan(slope).sum() == 9 + (4 * 9 - 4)  # hole margin + border ring


def test_nodata_is_never_filled_with_values():
    z = ramp(dzdx=0.2)
    z[2:4, 2:4] = np.nan
    assert np.isnan(compute_slope_deg(z, 10, 10)[2:4, 2:4]).all()


def test_max_neighbor_step():
    z = np.zeros((5, 5), np.float32)
    z[2, 3] = 4.0
    z[1, 1] = -7.0
    assert max_neighbor_step_m(z, 2, 2) == 7.0
    assert max_neighbor_step_m(z, 0, 4) == 0.0                # corner cell: only in-bounds neighbours count
    z[2, 2] = np.nan
    assert max_neighbor_step_m(z, 2, 2) is None
    assert max_neighbor_step_m(np.full((1, 1), 1.0, np.float32), 0, 0) is None


# --- hazard classification ------------------------------------------------------------------------

@pytest.mark.parametrize("slope,category", [
    (0.0, CATEGORY_LOW), (4.999, CATEGORY_LOW),
    (5.0, CATEGORY_MODERATE), (9.999, CATEGORY_MODERATE),
    (10.0, CATEGORY_HIGH), (15.0, CATEGORY_HIGH),
    (15.001, CATEGORY_EXCLUDED), (40.0, CATEGORY_EXCLUDED),
])
def test_slope_categories_match_prototype_thresholds(slope, category):
    hazard = classify_terrain(np.array([[slope]], np.float32))
    assert hazard.category[0, 0] == category
    assert hazard.blocked[0, 0] == (category == CATEGORY_EXCLUDED)


def test_risk_scores_are_bounded_monotonic_and_explained():
    slopes = np.linspace(0, 60, 601, dtype=np.float32)[None]
    risk = classify_terrain(slopes).risk[0]
    assert risk.min() == 0.0 and risk.max() == 1.0
    assert (np.diff(risk) >= -1e-9).all()
    assert classify_terrain(np.array([[5.0]], np.float32)).risk[0, 0] == pytest.approx(0.25)
    assert classify_terrain(np.array([[15.0]], np.float32)).risk[0, 0] == pytest.approx(0.75)
    assert classify_terrain(np.array([[20.0]], np.float32)).risk[0, 0] == 1.0


def test_unknown_terrain_is_blocked_by_default():
    hazard = classify_terrain(np.array([[np.nan, 1.0]], np.float32))
    assert hazard.category[0, 0] == CATEGORY_UNKNOWN and hazard.blocked[0, 0] and hazard.risk[0, 0] == 1.0
    assert not hazard.blocked[0, 1]


def test_unknown_terrain_can_be_made_traversable_only_by_explicit_config():
    config = HazardConfig(block_unknown=False, unknown_terrain_risk=0.6)
    hazard = classify_terrain(np.array([[np.nan]], np.float32), config)
    assert not hazard.blocked[0, 0] and hazard.risk[0, 0] == pytest.approx(0.6)
    assert hazard.category[0, 0] == CATEGORY_UNKNOWN  # still labelled unknown


def test_custom_limits_are_honoured():
    config = HazardConfig(slope_limits_deg=(2, 4, 6), risk_at_limits=(0.2, 0.4, 0.6))
    hazard = classify_terrain(np.array([[1.0, 3.0, 5.0, 7.0]], np.float32), config)
    assert hazard.category[0].tolist() == [CATEGORY_LOW, CATEGORY_MODERATE, CATEGORY_HIGH, CATEGORY_EXCLUDED]


@pytest.mark.parametrize("kwargs", [
    {"slope_limits_deg": (10, 5, 15)}, {"slope_limits_deg": (0, 5, 10)}, {"slope_limits_deg": (5, 10, 95)},
    {"risk_at_limits": (0.5, 0.4, 0.9)}, {"risk_at_limits": (0.1, 0.5, 1.0)}, {"unknown_terrain_risk": 1.5},
])
def test_invalid_hazard_configs_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        HazardConfig(**kwargs)


def test_category_fractions_and_stats():
    category = np.array([[0, 0, 1, 255]], np.uint8)
    fractions = category_fractions(category)
    assert fractions["low"] == 0.5 and fractions["unknown"] == 0.25 and fractions["excluded"] == 0.0
    stats = array_stats(np.array([1.0, 3.0, np.nan, np.nan]))
    assert stats["valid_fraction"] == 0.5 and stats["mean"] == 2.0 and stats["min"] == 1.0
    assert array_stats(np.array([np.nan])) == {"valid_fraction": 0.0}


# --- hazard zones --------------------------------------------------------------------------------

def zone(geometry, risk=0.9, blocking=True, name="z"):
    return HazardZoneSpec(uuid4(), name, "obstacle", geometry, risk, blocking, "simulated")


def test_blocking_zone_blocks_cells_it_touches_and_nothing_else():
    grid = make_grid(100, 100, 10)
    terrain = classify_terrain(np.zeros(grid.shape, np.float32))
    z = zone(box(-20, -20, 20, 20))
    cost = apply_hazard_zones(terrain, [z], grid.transform)
    mask = cost.zone_masks[z.id]
    assert cost.blocked[mask].all() and (cost.risk[mask] == 1.0).all()
    assert not cost.blocked[~mask].any() and (cost.risk[~mask] == 0.0).all()
    assert 16 <= mask.sum() <= 36
    assert not terrain.blocked.any()                       # input is not mutated


def test_non_blocking_zone_raises_risk_without_lowering_it():
    grid = make_grid(100, 100, 10)
    slope = np.zeros(grid.shape, np.float32)
    slope[5, 5] = 14.0                                      # terrain risk ~0.7
    terrain = classify_terrain(slope)
    z = zone(box(-50, -50, 50, 50), risk=0.4, blocking=False)
    cost = apply_hazard_zones(terrain, [z], grid.transform)
    assert cost.risk[5, 5] == terrain.risk[5, 5]            # max(terrain, zone)
    assert (cost.risk[0, 0] == pytest.approx(0.4)) and not cost.blocked.any()


def test_zone_outside_grid_has_no_effect():
    grid = make_grid(100, 100, 10)
    terrain = classify_terrain(np.zeros(grid.shape, np.float32))
    cost = apply_hazard_zones(terrain, [zone(box(500, 500, 600, 600))], grid.transform)
    assert not cost.blocked.any()


# --- raster I/O and previews -----------------------------------------------------------------------

def test_geotiff_round_trip_preserves_nodata_and_georeferencing(tmp_path):
    grid = make_grid(100, 100, 10)
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    data[3, 3] = np.nan
    write_geotiff(tmp_path / "a.tif", grid, data, FLOAT_NODATA, ["elevation_m"])
    back, meta = read_geotiff(tmp_path / "a.tif")
    assert np.isnan(back[0][3, 3]) and np.nansum(back[0]) == np.nansum(data)
    assert meta["nodata"] == FLOAT_NODATA and meta["transform"] == grid.transform
    assert meta["tags"]["marsnav_local_frame"] == "site_local_enu"
    assert FLOAT_NODATA not in back and not (back == FLOAT_NODATA).any()


def test_geotiff_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError):
        write_geotiff(tmp_path / "bad.tif", make_grid(100, 100, 10), np.zeros((5, 5), np.float32), None)


def test_previews_are_browser_pngs_with_transparent_nodata(tmp_path):
    import rasterio
    values = np.linspace(0, 10, 100, dtype=np.float32).reshape(10, 10)
    values[0, 0] = np.nan
    write_scalar_preview(tmp_path / "s.png", values, 0, 10)
    category = np.full((10, 10), CATEGORY_LOW, np.uint8)
    category[0, 0], category[1, 1] = CATEGORY_UNKNOWN, CATEGORY_EXCLUDED
    write_category_preview(tmp_path / "c.png", category)
    valid = np.ones((10, 10), bool)
    valid[0, 0] = False
    write_rgb_preview(tmp_path / "i.png", np.stack([values] * 3), valid)
    with rasterio.open(tmp_path / "s.png") as src:
        grey_alpha = src.read()
        assert src.count == 2 and grey_alpha[1][0, 0] == 0 and grey_alpha[1][5, 5] == 255
    with rasterio.open(tmp_path / "c.png") as src:
        rgba = src.read()
        assert src.count == 4 and rgba[3][0, 0] == 160 and tuple(rgba[:, 1, 1]) == (200, 30, 30, 220)
    with rasterio.open(tmp_path / "i.png") as src:
        assert src.count == 4 and src.read(4)[0, 0] == 0
    assert (tmp_path / "s.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --- demo terrain ----------------------------------------------------------------------------------

def test_demo_terrain_is_deterministic_and_has_unknown_patch():
    grid = make_grid(8000, 8000, 10)
    first, second = generate_demo_elevation(grid), generate_demo_elevation(grid)
    assert np.array_equal(first, second, equal_nan=True)
    assert np.isnan(first).any() and np.isfinite(first).mean() > 0.95
    assert not np.array_equal(first, generate_demo_elevation(grid, seed=1), equal_nan=True)


def test_demo_terrain_exercises_every_hazard_class_and_keeps_seed_sites_traversable():
    grid = make_grid(8000, 8000, 10)
    hazard = classify_terrain(compute_slope_deg(generate_demo_elevation(grid), 10, 10))
    fractions = category_fractions(hazard.category)
    assert all(fractions[name] > 0 for name in ("low", "moderate", "high", "excluded", "unknown"))
    for x, y in [(-3000, -2500), (1200, 800), (-500, 2200), (2600, -1200)]:
        assert not hazard.blocked[grid.cell_of(x, y)]


def test_demo_terrain_barriers_can_be_walked_around():
    """The habitat must reach every seeded target (4-connected) even with the seeded obstacle zone applied."""
    from collections import deque

    grid = make_grid(8000, 8000, 20)
    hazard = classify_terrain(compute_slope_deg(generate_demo_elevation(grid), 20, 20))
    obstacle = zone(box(300, 300, 700, 600))
    blocked = apply_hazard_zones(hazard, [obstacle], grid.transform).blocked
    start = grid.cell_of(-3000, -2500)
    seen = np.zeros(blocked.shape, bool)
    seen[start] = True
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1] and not seen[nr, nc] and not blocked[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    for x, y in [(1200, 800), (-500, 2200), (2600, -1200)]:
        assert seen[grid.cell_of(x, y)], f"target at {(x, y)} is walled off from the habitat"
