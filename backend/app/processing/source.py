"""Inspect, validate, crop and align source rasters onto the site-local TargetGrid.

NoData policy: a target cell is valid only if (nearly) all contributing source cells were valid. The
validity mask is resampled on its own, so interpolation can never invent data next to NoData, and
cells outside source coverage are NoData. Nothing is filled or interpolated across gaps.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioError
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from app.processing.crs import CrsInfo, validate_mars_crs
from app.processing.errors import ProcessingError, SourceValidationError
from app.processing.grid import TargetGrid

Kind = Literal["elevation", "imagery"]
METRE_ALIASES = {"m", "meter", "meters", "metre", "metres"}
MIN_CELL_VALID_FRACTION = 0.999


@dataclass
class SourceInfo:
    driver: str
    width_px: int
    height_px: int
    band_count: int
    dtype: str
    nodata: float | None
    crs: CrsInfo
    native_resolution_m: float
    bounds_local_m: tuple[float, float, float, float]
    bbox_overlap_fraction: float
    units: str | None
    units_source: str | None
    scale: float
    offset: float
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "driver": self.driver,
            "size_px": [self.width_px, self.height_px],
            "band_count": self.band_count,
            "dtype": self.dtype,
            "nodata": self.nodata,
            "crs": self.crs.summary(),
            "native_resolution_m": round(self.native_resolution_m, 6),
            "bounds_local_m": list(self.bounds_local_m),
            "bbox_overlap_fraction": round(self.bbox_overlap_fraction, 6),
            "units": self.units,
            "units_source": self.units_source,
            "scale": self.scale,
            "offset": self.offset,
            "warnings": self.warnings,
        }


@dataclass
class WarpResult:
    data: np.ndarray  # (bands, rows, cols); float32 with NaN where invalid for elevation
    valid: np.ndarray  # (rows, cols) bool
    resampling: str
    source_cells_read: int


def _normalise_units(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip().lower()
    return "m" if text in METRE_ALIASES else text


def _native_resolution_m(src: rasterio.DatasetReader, radius_m: float) -> float:
    res_x, res_y = abs(src.transform.a), abs(src.transform.e)
    if src.crs.is_geographic:
        mid_lat = (src.bounds.top + src.bounds.bottom) / 2
        x_m = math.radians(res_x) * radius_m * max(math.cos(math.radians(mid_lat)), 1e-6)
        y_m = math.radians(res_y) * radius_m
        return (x_m + y_m) / 2
    return (res_x + res_y) / 2


def _open(path: Path) -> rasterio.DatasetReader:
    try:
        return rasterio.open(path)
    except RasterioError as exc:
        raise SourceValidationError(f"Source could not be opened as a raster: {exc}") from exc


def inspect_source(
    path: Path,
    grid: TargetGrid,
    frame_radius_m: float,
    kind: Kind,
    expected_units: str | None = None,
    accept_ellipsoid_as_sphere: bool = False,
) -> SourceInfo:
    with _open(path) as src:
        crs_info, warnings = validate_mars_crs(src.crs, frame_radius_m, accept_ellipsoid_as_sphere)
        if src.width < 1 or src.height < 1 or src.count < 1:
            raise SourceValidationError("Source raster is empty")

        units = source_units = None
        scale = offset = 1.0
        if kind == "elevation":
            declared = _normalise_units(src.units[0] if src.units else None) or _normalise_units(
                src.tags(1).get("units") or src.tags().get("units")
            )
            expected = _normalise_units(expected_units)
            if declared and expected and declared != expected:
                raise SourceValidationError(
                    f"Source declares elevation units '{declared}' but the job expects '{expected}'"
                )
            units, source_units = (declared, "raster") if declared else (expected, "job")
            if units is None:
                raise SourceValidationError(
                    "Elevation units are not declared in the raster; state them explicitly in the job"
                )
            if units != "m":
                raise SourceValidationError(f"Elevation units must be metres, got '{units}'")
            scale = float(src.scales[0]) if src.scales and src.scales[0] is not None else 1.0
            offset = float(src.offsets[0]) if src.offsets and src.offsets[0] is not None else 0.0
            if not (np.issubdtype(np.dtype(src.dtypes[0]), np.number)):
                raise SourceValidationError(f"Elevation dtype {src.dtypes[0]} is not numeric")

        try:
            bounds = transform_bounds(src.crs, grid.crs, *src.bounds, densify_pts=21)
        except Exception as exc:
            raise SourceValidationError(f"Source bounds could not be transformed to the site frame: {exc}") from exc
        if not all(math.isfinite(v) for v in bounds):
            raise SourceValidationError("Source bounds are not finite in the site frame")

        g_min_x, g_min_y, g_max_x, g_max_y = grid.bounds
        w = min(bounds[2], g_max_x) - max(bounds[0], g_min_x)
        h = min(bounds[3], g_max_y) - max(bounds[1], g_min_y)
        overlap = max(w, 0.0) * max(h, 0.0) / (grid.width_m * grid.height_m)
        if overlap <= 0:
            raise SourceValidationError("Source does not overlap the site extent")
        if overlap < 1.0:
            warnings.append(f"Source covers only about {overlap:.1%} of the site extent (bounding box)")

        return SourceInfo(
            driver=src.driver,
            width_px=src.width,
            height_px=src.height,
            band_count=src.count,
            dtype=src.dtypes[0],
            nodata=src.nodata,
            crs=crs_info,
            native_resolution_m=_native_resolution_m(src, frame_radius_m),
            bounds_local_m=tuple(bounds),
            bbox_overlap_fraction=overlap,
            units=units,
            units_source=source_units,
            scale=scale,
            offset=offset,
            warnings=warnings,
        )


def _source_window(src: rasterio.DatasetReader, grid: TargetGrid, margin_px: int = 3) -> Window:
    bounds = transform_bounds(grid.crs, src.crs, *grid.bounds, densify_pts=21)
    window = from_bounds(*bounds, transform=src.transform)
    window = Window(
        window.col_off - margin_px,
        window.row_off - margin_px,
        window.width + 2 * margin_px,
        window.height + 2 * margin_px,
    )
    try:
        window = window.intersection(Window(0, 0, src.width, src.height))
    except Exception as exc:
        raise SourceValidationError("Source does not overlap the site extent") from exc
    return window.round_offsets(op="floor").round_lengths(op="ceil")


def warp_to_grid(
    path: Path,
    grid: TargetGrid,
    kind: Kind,
    native_resolution_m: float,
    band_indexes: list[int] | None = None,
    min_cell_valid_fraction: float = MIN_CELL_VALID_FRACTION,
) -> WarpResult:
    """Crop to the site and resample onto the target grid; returns data plus an exact validity mask."""
    resampling = Resampling.average if native_resolution_m < grid.resolution_m * 0.9 else Resampling.bilinear
    with _open(path) as src:
        indexes = band_indexes or list(range(1, src.count + 1))
        window = _source_window(src, grid)
        if window.width < 1 or window.height < 1:
            raise SourceValidationError("Source does not overlap the site extent")
        raw = src.read(indexes, window=window).astype(np.float32)
        masks = src.read_masks(indexes, window=window) > 0
        scales = np.array([src.scales[i - 1] or 1.0 for i in indexes], dtype=np.float32)[:, None, None]
        offsets = np.array([src.offsets[i - 1] or 0.0 for i in indexes], dtype=np.float32)[:, None, None]
        win_transform = src.window_transform(window)
        src_crs = src.crs

    if kind == "elevation":
        raw = raw * scales + offsets
    valid_src = masks.all(axis=0) & np.isfinite(raw).all(axis=0)
    raw = np.where(valid_src[None], raw, np.nan).astype(np.float32)

    out = np.full((len(indexes), *grid.shape), np.nan, dtype=np.float32)
    for i in range(len(indexes)):
        reproject(
            source=np.ascontiguousarray(raw[i]),
            destination=out[i],
            src_transform=win_transform,
            src_crs=src_crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            init_dest_nodata=True,
        )

    fraction = np.zeros(grid.shape, dtype=np.float32)
    reproject(
        source=valid_src.astype(np.float32),
        destination=fraction,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=resampling,
        dst_nodata=0.0,
        init_dest_nodata=True,
    )
    valid = (fraction >= min_cell_valid_fraction) & np.isfinite(out).all(axis=0)
    out[:, ~valid] = np.nan
    return WarpResult(out, valid, resampling.name, int(window.width) * int(window.height))
