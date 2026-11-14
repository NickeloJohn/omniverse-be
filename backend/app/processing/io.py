"""GeoTIFF and browser-preview writers. Every file is written on the shared TargetGrid."""

from pathlib import Path
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

from app.processing.grid import TargetGrid
from app.processing.hazard import (
    CATEGORY_EXCLUDED,
    CATEGORY_HIGH,
    CATEGORY_LOW,
    CATEGORY_MODERATE,
    CATEGORY_UNKNOWN,
)

FLOAT_NODATA = -9999.0
LOCAL_FRAME_TAG = "marsnav_local_frame"

CATEGORY_RGBA = {
    CATEGORY_LOW: (46, 160, 67, 200),
    CATEGORY_MODERATE: (227, 179, 65, 200),
    CATEGORY_HIGH: (240, 120, 40, 210),
    CATEGORY_EXCLUDED: (200, 30, 30, 220),
    CATEGORY_UNKNOWN: (110, 110, 110, 160),
}


def write_geotiff(
    path: Path,
    grid: TargetGrid,
    data: np.ndarray,
    nodata: float | None,
    descriptions: list[str] | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    """Write (bands, rows, cols) or (rows, cols) data; float NaN is stored as the NoData value."""
    if data.ndim == 2:
        data = data[None]
    if data.shape[1:] != grid.shape:
        raise ValueError(f"data shape {data.shape[1:]} does not match grid {grid.shape}")
    if np.issubdtype(data.dtype, np.floating) and nodata is not None:
        data = np.where(np.isfinite(data), data, nodata).astype(data.dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.height_px,
        width=grid.width_px,
        count=data.shape[0],
        dtype=data.dtype,
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(data)
        for index, text in enumerate(descriptions or [], start=1):
            dst.set_band_description(index, text)
        dst.update_tags(**{LOCAL_FRAME_TAG: "site_local_enu", **(tags or {})})


def read_geotiff(path: Path) -> tuple[np.ndarray, dict]:
    """Return (bands, rows, cols) with float NoData converted to NaN, plus basic metadata."""
    with rasterio.open(path) as src:
        data = src.read()
        meta = {"transform": src.transform, "nodata": src.nodata, "tags": src.tags(), "crs": src.crs}
        if np.issubdtype(data.dtype, np.floating) and src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
    return data, meta


def _write_png(path: Path, bands: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # PNG previews are plain images; their bounds live in the layer metadata, so no georeferencing is expected.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(
            path, "w", driver="PNG", height=bands.shape[1], width=bands.shape[2], count=bands.shape[0], dtype="uint8"
        ) as dst:
            dst.write(bands)


def write_scalar_preview(path: Path, values: np.ndarray, vmin: float, vmax: float) -> None:
    """Greyscale + alpha PNG; NoData is fully transparent."""
    valid = np.isfinite(values)
    scaled = np.clip((np.where(valid, values, vmin) - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    grey = (scaled * 255).astype(np.uint8)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    _write_png(path, np.stack([grey, alpha]))


def write_category_preview(path: Path, category: np.ndarray) -> None:
    rgba = np.zeros((4, *category.shape), dtype=np.uint8)
    for code, color in CATEGORY_RGBA.items():
        mask = category == code
        for band in range(4):
            rgba[band][mask] = color[band]
    _write_png(path, rgba)


def write_rgb_preview(path: Path, data: np.ndarray, valid: np.ndarray) -> None:
    """Imagery preview: first 1 or 3 bands stretched to 8 bit (2-98 percentile), NoData transparent."""
    bands = data[:3] if data.shape[0] >= 3 else np.repeat(data[:1], 3, axis=0)
    out = np.zeros((4, *valid.shape), dtype=np.uint8)
    for i in range(3):
        values = bands[i][valid]
        lo, hi = (np.percentile(values, [2, 98]) if values.size else (0.0, 1.0))
        out[i] = (np.clip((np.nan_to_num(bands[i]) - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)
    out[3] = np.where(valid, 255, 0).astype(np.uint8)
    _write_png(path, out)
