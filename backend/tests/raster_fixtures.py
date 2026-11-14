"""Synthetic raster fixtures. Values are analytic functions of local coordinates, never NASA data."""

import math
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_bounds

from app.processing.grid import TargetGrid, mars_local_crs

SITE_LON, SITE_LAT, RADIUS = 77.58, 18.38, 3_396_190.0


def local_crs() -> CRS:
    return mars_local_crs(SITE_LON, SITE_LAT, RADIUS)


def mars_geographic_crs(radius: float = RADIUS) -> CRS:
    return CRS.from_string(f"+proj=longlat +R={radius} +no_defs")


def mars_sinusoidal_crs() -> CRS:
    return CRS.from_string(f"+proj=sinu +lon_0={SITE_LON} +x_0=0 +y_0=0 +R={RADIUS} +units=m +no_defs")


def make_grid(width_m: float = 400.0, height_m: float = 400.0, resolution_m: float = 10.0) -> TargetGrid:
    return TargetGrid.build(width_m, height_m, resolution_m, local_crs(), 10_000_000)


def write_raster(
    path: Path,
    array: np.ndarray,
    crs: CRS | None,
    transform: Affine,
    nodata: float | None = None,
    units: str | None = None,
    scale: float | None = None,
    offset: float | None = None,
) -> Path:
    data = array if array.ndim == 3 else array[None]
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0],
        dtype=data.dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data)
        if units is not None:
            dst.units = tuple([units] * data.shape[0])
        if scale is not None:
            dst.scales = tuple([scale] * data.shape[0])
        if offset is not None:
            dst.offsets = tuple([offset] * data.shape[0])
    return path


def plane_raster(
    path: Path,
    crs: CRS,
    local_bounds: tuple[float, float, float, float],
    cell_size: float,
    a: float,
    b: float,
    z0: float = 0.0,
    nodata: float | None = -9999.0,
    units: str | None = "m",
    margin_cells: int = 4,
) -> Path:
    """Raster in `crs` covering local_bounds whose value is z0 + a*x_local + b*y_local at every cell centre.

    cell_size is in CRS units (metres for projected, degrees for geographic CRSs).
    """
    min_x, min_y, max_x, max_y = transform_bounds(local_crs(), crs, *local_bounds, densify_pts=21)
    min_x, min_y = min_x - margin_cells * cell_size, min_y - margin_cells * cell_size
    max_x, max_y = max_x + margin_cells * cell_size, max_y + margin_cells * cell_size
    cols, rows = math.ceil((max_x - min_x) / cell_size), math.ceil((max_y - min_y) / cell_size)
    transform = Affine(cell_size, 0, min_x, 0, -cell_size, min_y + rows * cell_size)
    r, c = np.indices((rows, cols))
    xs = (transform.c + (c + 0.5) * cell_size).ravel()
    ys = (transform.f - (r + 0.5) * cell_size).ravel()
    lx, ly = warp_transform(crs, local_crs(), xs.tolist(), ys.tolist())
    z = (z0 + a * np.array(lx) + b * np.array(ly)).reshape(rows, cols).astype(np.float32)
    return write_raster(path, z, crs, transform, nodata=nodata, units=units)


def expected_slope_deg(a: float, b: float) -> float:
    return math.degrees(math.atan(math.hypot(a, b)))
