"""Terrain ingestion/processing pipeline: acquire -> validate -> crop/align -> slope -> hazard -> register.

All computation happens in memory first; files and DB rows are written afterwards and rolled back as a unit,
so a terrain set is either fully registered or not at all. Source data is never copied into storage (it can be
huge); its checksum and provenance are recorded instead. Demo terrain is explicitly simulated.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import rasterio
from geoalchemy2.shape import from_shape
from shapely.geometry import box
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models import DataLayer, ExplorationSite, RasterAsset
from app.models.enums import DataOrigin, LayerType, ProcessingStatus, RasterRole
from app.processing.demo import DEMO_GENERATOR_VERSION, DEMO_NOTE, generate_demo_elevation
from app.processing.errors import ProcessingError, SourceValidationError
from app.processing.grid import TargetGrid, mars_local_crs, mars_local_proj_string
from app.processing.hazard import (
    DEFAULT_HAZARD_CONFIG,
    DISCLAIMERS,
    HazardConfig,
    array_stats,
    category_fractions,
    classify_terrain,
)
from app.processing.io import (
    FLOAT_NODATA,
    read_geotiff,
    write_category_preview,
    write_geotiff,
    write_rgb_preview,
    write_scalar_preview,
)
from app.processing.source import inspect_source, warp_to_grid
from app.processing.terrain import compute_slope_deg
from app.schemas.ingestion import IngestionJob, RasterSource
from app.services import nasa_service
from app.services.storage import AssetStorage
from app.utils.mars_coords import LOCAL_CRS_NAME, LOCAL_SRID

logger = logging.getLogger("marsnav.ingest")
PIPELINE_VERSION = "terrain-pipeline-v1"
SLOPE_PREVIEW_MAX_DEG = 30.0


@dataclass
class TerrainSet:
    elevation: DataLayer
    slope: DataLayer
    hazard: DataLayer
    imagery: DataLayer | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Run:
    """Tracks directories created by one run so a failure can remove them."""

    storage: AssetStorage
    created_dirs: list[str] = field(default_factory=list)

    def version_dir(self, site_slug: str, layer_slug: str, version: str) -> str:
        key = self.storage.key_for("sites", site_slug, layer_slug, version)
        if not self.storage.resolve(key).exists():
            self.created_dirs.append(key)
        self.storage.resolve(key).mkdir(parents=True, exist_ok=True)
        return key

    def cleanup(self) -> None:
        for key in self.created_dirs:
            self.storage.remove_tree(key)


def grid_for_site(site: ExplorationSite, resolution_m: float, max_cells: int) -> TargetGrid:
    crs = mars_local_crs(site.center_longitude_deg_east, site.center_latitude_deg, site.body_radius_m)
    return TargetGrid.build(site.width_m, site.height_m, resolution_m, crs, max_cells)


def _local_proj(site: ExplorationSite) -> str:
    return mars_local_proj_string(site.center_longitude_deg_east, site.center_latitude_deg, site.body_radius_m)


def _grid_metadata(site: ExplorationSite, grid: TargetGrid) -> dict[str, Any]:
    return {
        "frame": LOCAL_CRS_NAME,
        "proj": _local_proj(site),
        "resolution_m": grid.resolution_m,
        "size_px": [grid.width_px, grid.height_px],
        "bounds_local_m": list(grid.bounds),
        "transform": list(grid.transform)[:6],
    }


def _tool_versions(resampling: str | None = None) -> dict[str, Any]:
    info = {"rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__, "numpy": np.__version__}
    return info | ({"resampling": resampling} if resampling else {})


def _bounds_polygon(grid: TargetGrid):
    return from_shape(box(*grid.bounds), srid=LOCAL_SRID)


def _get_or_create_layer(
    session: Session, site: ExplorationSite, slug: str, layer_type: LayerType, name: str,
    origin: DataOrigin, status: ProcessingStatus, is_demo: bool, candidate_slug: str | None = None,
) -> DataLayer:
    layer = session.scalars(select(DataLayer).where(DataLayer.site_id == site.id, DataLayer.slug == slug)).first()
    if layer is None and candidate_slug:
        layer = session.scalars(
            select(DataLayer).where(DataLayer.site_id == site.id, DataLayer.slug == candidate_slug)
        ).first()
        if layer is not None:
            if layer.processing_status != ProcessingStatus.PLANNED or layer.assets:
                raise ProcessingError(f"Layer '{candidate_slug}' is not an unprocessed planned candidate")
            layer.slug = slug
    if layer is None:
        layer = DataLayer(site_id=site.id, slug=slug, name=name, layer_type=layer_type)
        session.add(layer)
    elif layer.layer_type != layer_type:
        raise ProcessingError(f"Layer '{slug}' exists with a different layer type")
    layer.name = name
    layer.data_origin, layer.processing_status, layer.is_demo = origin, status, is_demo
    return layer


def _replace_assets(session: Session, layer: DataLayer, version: str) -> None:
    for asset in [a for a in layer.assets if a.dataset_version == version]:
        layer.assets.remove(asset)
    session.flush()


def _add_asset(
    layer: DataLayer, storage: AssetStorage, grid: TargetGrid, key: str, role: RasterRole,
    variable: str, version: str, band_count: int, dtype: str, nodata: float | None,
    media_type: str, meta: dict[str, Any],
) -> None:
    path = storage.resolve(key)
    layer.assets.append(
        RasterAsset(
            role=role, variable=variable, storage_key=key, media_type=media_type,
            width_px=grid.width_px, height_px=grid.height_px, band_count=band_count, dtype=dtype,
            pixel_size_x_m=grid.resolution_m, pixel_size_y_m=grid.resolution_m, nodata_value=nodata,
            transform=list(grid.transform)[:6], crs=meta.pop("crs_proj", None), bounds=_bounds_polygon(grid),
            sha256=storage.sha256(path), size_bytes=path.stat().st_size, dataset_version=version,
            processing_metadata=meta,
        )
    )


def _fill_layer(
    layer: DataLayer, site: ExplorationSite, grid: TargetGrid, *, origin: DataOrigin, status: ProcessingStatus,
    is_demo: bool, version: str, metadata: dict[str, Any], units: str | None, nodata: float | None,
    derived_from: DataLayer | None = None, source: RasterSource | None = None, source_name: str | None = None,
) -> None:
    layer.data_origin, layer.processing_status, layer.is_demo = origin, status, is_demo
    layer.spatial_resolution_m = grid.resolution_m
    layer.crs = _local_proj(site)
    layer.units, layer.nodata_value = units, nodata
    layer.dataset_version = version
    layer.processing_metadata = metadata
    layer.bounds = _bounds_polygon(grid)
    layer.derived_from_layer_id = derived_from.id if derived_from is not None else None
    layer.source_name = source.name if source else source_name
    layer.source_url = source.url if source else None
    layer.source_product_id = source.product_id if source else None
    layer.acquisition_date = source.acquisition_date if source else None


def _write_elevation(
    run: _Run, layer: DataLayer, site: ExplorationSite, grid: TargetGrid, elevation: np.ndarray,
    version: str, meta: dict[str, Any],
) -> None:
    key_dir = run.version_dir(site.slug, layer.slug, version)
    tif, png = f"{key_dir}/elevation.tif", f"{key_dir}/preview.png"
    write_geotiff(run.storage.resolve(tif), grid, elevation, FLOAT_NODATA, ["elevation_m"])
    finite = elevation[np.isfinite(elevation)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    write_scalar_preview(run.storage.resolve(png), elevation, lo, hi)
    tools = _tool_versions()
    _add_asset(layer, run.storage, grid, tif, RasterRole.PROCESSED, "elevation_m", version, 1, "float32",
               FLOAT_NODATA, "image/tiff", {**tools, "crs_proj": _local_proj(site), **meta})
    _add_asset(layer, run.storage, grid, png, RasterRole.DISPLAY_TILES, "preview", version, 2, "uint8",
               None, "image/png", {"crs_proj": _local_proj(site), "preview": "greyscale elevation, NoData transparent"})


def _derived_layer(session: Session, site: ExplorationSite, elevation: DataLayer, suffix: str, layer_type: LayerType, name: str) -> DataLayer:
    existing = session.scalars(
        select(DataLayer).where(DataLayer.derived_from_layer_id == elevation.id, DataLayer.layer_type == layer_type)
    ).first()
    if existing is not None:
        return existing
    prefix = elevation.slug.removesuffix("-elevation")
    origin, status, is_demo = _origin_status(elevation)
    return _get_or_create_layer(session, site, f"{prefix}-{suffix}", layer_type, name, origin, status, is_demo)


def _origin_status(elevation: DataLayer) -> tuple[DataOrigin, ProcessingStatus, bool]:
    if elevation.is_demo:
        return DataOrigin.SIMULATED, ProcessingStatus.DEMO, True
    return DataOrigin.DERIVED, ProcessingStatus.PROCESSED, False


def _lineage(elevation: DataLayer) -> dict[str, Any]:
    return {
        "derived_from_layer_id": str(elevation.id),
        "derived_from_slug": elevation.slug,
        "derived_from_version": elevation.dataset_version,
        "derived_from_origin": elevation.data_origin.value,
    }


def derive_slope_layer(
    session: Session, run: _Run, site: ExplorationSite, grid: TargetGrid, elevation_layer: DataLayer,
    elevation: np.ndarray, steps: list[dict[str, Any]],
) -> tuple[DataLayer, np.ndarray]:
    origin, status, is_demo = _origin_status(elevation_layer)
    layer = _derived_layer(session, site, elevation_layer, "slope", LayerType.SLOPE,
                           f"{elevation_layer.name} - slope" if not is_demo else "DEMO synthetic slope")
    version = elevation_layer.dataset_version
    slope = compute_slope_deg(elevation, grid.resolution_m, grid.resolution_m)
    stats = array_stats(slope)
    steps.append({"step": "slope", "method": "Horn 3x3; invalid if any of the 9 cells is NoData", "stats": stats})

    key_dir = run.version_dir(site.slug, layer.slug, version)
    tif, png = f"{key_dir}/slope.tif", f"{key_dir}/preview.png"
    write_geotiff(run.storage.resolve(tif), grid, slope, FLOAT_NODATA, ["slope_deg"])
    write_scalar_preview(run.storage.resolve(png), slope, 0.0, SLOPE_PREVIEW_MAX_DEG)
    session.flush()
    _replace_assets(session, layer, version)
    _add_asset(layer, run.storage, grid, tif, RasterRole.PROCESSED, "slope_deg", version, 1, "float32",
               FLOAT_NODATA, "image/tiff", {**_tool_versions(), "crs_proj": _local_proj(site)})
    _add_asset(layer, run.storage, grid, png, RasterRole.DISPLAY_TILES, "preview", version, 2, "uint8", None,
               "image/png", {"crs_proj": _local_proj(site), "preview": f"greyscale slope 0-{SLOPE_PREVIEW_MAX_DEG:g} deg"})
    _fill_layer(
        layer, site, grid, origin=origin, status=status, is_demo=is_demo, version=version, units="deg",
        nodata=FLOAT_NODATA, derived_from=elevation_layer,
        source_name="Derived by MARS-NAV from " + elevation_layer.name,
        metadata={
            "pipeline_version": PIPELINE_VERSION, **_lineage(elevation_layer), "grid": _grid_metadata(site, grid),
            "method": "Horn 3x3 slope in degrees", "stats": stats, "steps": steps[-1:],
            **({"demo_note": DEMO_NOTE} if is_demo else {}),
        },
    )
    return layer, slope


def derive_hazard_layer(
    session: Session, run: _Run, site: ExplorationSite, grid: TargetGrid, elevation_layer: DataLayer,
    slope_layer: DataLayer, slope: np.ndarray, config: HazardConfig, steps: list[dict[str, Any]],
) -> DataLayer:
    origin, status, is_demo = _origin_status(elevation_layer)
    layer = _derived_layer(session, site, elevation_layer, "hazard-cost", LayerType.HAZARD_COST,
                           "DEMO synthetic hazard cost" if is_demo else f"{elevation_layer.name} - terrain hazard")
    version = f"{elevation_layer.dataset_version}-{_config_digest(config)}"
    hazard = classify_terrain(slope, config)
    stats = {
        "categories": category_fractions(hazard.category),
        "blocked_fraction": round(float(hazard.blocked.mean()), 6),
        "mean_risk": round(float(hazard.risk.mean()), 6),
    }
    steps.append({"step": "hazard", "config_version": config.version, "stats": stats})

    key_dir = run.version_dir(site.slug, layer.slug, version)
    risk_tif, classes_tif, png = f"{key_dir}/risk.tif", f"{key_dir}/classes.tif", f"{key_dir}/preview.png"
    write_geotiff(run.storage.resolve(risk_tif), grid, hazard.risk, None, ["risk_score"])
    write_geotiff(run.storage.resolve(classes_tif), grid,
                  np.stack([hazard.category, hazard.blocked.astype(np.uint8)]), None, ["category", "blocked"])
    write_category_preview(run.storage.resolve(png), hazard.category)
    session.flush()
    _replace_assets(session, layer, version)
    crs = {"crs_proj": _local_proj(site)}
    _add_asset(layer, run.storage, grid, risk_tif, RasterRole.PROCESSED, "risk_score", version, 1, "float32",
               None, "image/tiff", {**_tool_versions(), **crs})
    _add_asset(layer, run.storage, grid, classes_tif, RasterRole.PROCESSED, "hazard_classes", version, 2, "uint8",
               None, "image/tiff", {**_tool_versions(), **crs, "bands": ["category", "blocked"],
                                    "category_codes": {"low": 0, "moderate": 1, "high": 2, "excluded": 3, "unknown": 255}})
    _add_asset(layer, run.storage, grid, png, RasterRole.DISPLAY_TILES, "preview", version, 4, "uint8", None,
               "image/png", {**crs, "preview": "hazard categories RGBA"})
    _fill_layer(
        layer, site, grid, origin=origin, status=status, is_demo=is_demo, version=version, units="risk 0-1",
        nodata=None, derived_from=elevation_layer, source_name="Derived by MARS-NAV from " + slope_layer.name,
        metadata={
            "pipeline_version": PIPELINE_VERSION, **_lineage(elevation_layer), "slope_layer_id": str(slope_layer.id),
            "grid": _grid_metadata(site, grid), "hazard_config": config.model_dump(mode="json"),
            "stats": stats, "disclaimers": DISCLAIMERS, "steps": steps[-1:],
            **({"demo_note": DEMO_NOTE} if is_demo else {}),
        },
    )
    return layer


def _config_digest(config: HazardConfig) -> str:
    return hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:8]


def _get_site(session: Session, slug: str) -> ExplorationSite:
    site = session.scalars(select(ExplorationSite).where(ExplorationSite.slug == slug)).first()
    if site is None:
        raise ProcessingError(f"Exploration site '{slug}' not found")
    return site


def process_demo_terrain(
    session: Session, storage: AssetStorage, settings: Settings, site_slug: str, resolution_m: float = 10.0,
    hazard: HazardConfig = DEFAULT_HAZARD_CONFIG,
) -> TerrainSet:
    """DEMO mode: synthetic terrain, labelled simulated. Kept separate from NASA ingestion."""
    site = _get_site(session, site_slug)
    grid = grid_for_site(site, resolution_m, settings.max_grid_cells)
    run = _Run(storage)
    steps: list[dict[str, Any]] = [{"step": "generate_demo_terrain", "generator": DEMO_GENERATOR_VERSION}]
    try:
        elevation = generate_demo_elevation(grid)
        layer = _get_or_create_layer(session, site, "demo-elevation", LayerType.ELEVATION, "DEMO synthetic elevation",
                                     DataOrigin.SIMULATED, ProcessingStatus.DEMO, True)
        version = f"{DEMO_GENERATOR_VERSION}-r{resolution_m:g}m"
        stats = array_stats(elevation)
        steps.append({"step": "elevation_stats", "stats": stats})
        session.flush()
        _replace_assets(session, layer, version)
        _write_elevation(run, layer, site, grid, elevation, version, {"generator": DEMO_GENERATOR_VERSION})
        _fill_layer(
            layer, site, grid, origin=DataOrigin.SIMULATED, status=ProcessingStatus.DEMO, is_demo=True, version=version,
            units="m", nodata=FLOAT_NODATA, source_name="MARS-NAV synthetic terrain generator",
            metadata={"pipeline_version": PIPELINE_VERSION, "grid": _grid_metadata(site, grid), "stats": stats,
                      "demo_note": DEMO_NOTE, "generator": DEMO_GENERATOR_VERSION, "steps": steps[:2]},
        )
        session.flush()
        slope_layer, slope = derive_slope_layer(session, run, site, grid, layer, elevation, steps)
        hazard_layer = derive_hazard_layer(session, run, site, grid, layer, slope_layer, slope, hazard, steps)
        session.commit()
    except Exception:
        session.rollback()
        run.cleanup()
        raise
    return TerrainSet(layer, slope_layer, hazard_layer)


def _acquire(source: RasterSource, storage: AssetStorage, settings: Settings, transport) -> nasa_service.AcquiredSource:
    if source.url is not None:
        return nasa_service.download_dataset(
            source.url, storage.root / "downloads", settings.nasa_allowed_hosts,
            settings.max_download_mb * 1024 * 1024, settings.download_timeout_s, source.expected_sha256, transport,
        )
    return nasa_service.locate_local_file(source.local_file or "", settings.ingest_inbox, source.expected_sha256)


def _source_metadata(source: RasterSource, acquired: nasa_service.AcquiredSource, info) -> dict[str, Any]:
    return {
        "name": source.name, "url": source.url, "product_id": source.product_id,
        "acquisition_date": source.acquisition_date.isoformat() if source.acquisition_date else None,
        "citation": source.citation, "acquired_via": acquired.kind,
        "local_file": source.local_file, "sha256": acquired.sha256, "size_bytes": acquired.size_bytes,
        "vertical_datum": source.vertical_datum, "original": info.summary(),
    }


def ingest_job(
    session: Session, storage: AssetStorage, settings: Settings, job: IngestionJob, transport=None,
) -> TerrainSet:
    """Ingest an observed NASA/USGS DEM (and optional imagery) and derive slope and hazard layers."""
    site = _get_site(session, job.site_slug)
    grid = grid_for_site(site, job.resolution_m, settings.max_grid_cells)
    run = _Run(storage)
    steps: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        acquired = _acquire(job.elevation, storage, settings, transport)
        steps.append({"step": "acquire", "via": acquired.kind, "sha256": acquired.sha256, "size_bytes": acquired.size_bytes})
        info = inspect_source(acquired.path, grid, site.body_radius_m, "elevation", job.elevation.units,
                              job.elevation.accept_ellipsoid_as_sphere)
        warnings += info.warnings
        steps.append({"step": "validate", "crs": info.crs.summary(), "units": info.units,
                      "units_source": info.units_source, "bbox_overlap_fraction": round(info.bbox_overlap_fraction, 6)})

        warped = warp_to_grid(acquired.path, grid, "elevation", info.native_resolution_m, [job.elevation.band])
        elevation = warped.data[0]
        valid_fraction = float(warped.valid.mean())
        if valid_fraction < job.min_valid_fraction:
            raise SourceValidationError(
                f"Only {valid_fraction:.1%} of the site has valid elevation; at least "
                f"{job.min_valid_fraction:.1%} is required"
            )
        if valid_fraction < 1.0:
            warnings.append(f"{1 - valid_fraction:.1%} of the site is NoData (unknown terrain, blocked by default)")
        stats = array_stats(elevation)
        steps.append({"step": "crop_align", "resampling": warped.resampling, "valid_fraction": round(valid_fraction, 6),
                      "native_resolution_m": round(info.native_resolution_m, 4), "target_resolution_m": grid.resolution_m})

        version = job.dataset_version or f"src-{acquired.sha256[:12]}-r{job.resolution_m:g}m"
        slug = f"{job.layer_prefix}-elevation"
        layer = _get_or_create_layer(session, site, slug, LayerType.ELEVATION, job.elevation.name,
                                     DataOrigin.OBSERVED, ProcessingStatus.PROCESSED, False,
                                     job.elevation.candidate_slug)
        session.flush()
        _replace_assets(session, layer, version)
        _write_elevation(run, layer, site, grid, elevation, version,
                         {"resampling": warped.resampling, "source_sha256": acquired.sha256})
        _fill_layer(
            layer, site, grid, origin=DataOrigin.OBSERVED, status=ProcessingStatus.PROCESSED, is_demo=False,
            version=version, units="m", nodata=FLOAT_NODATA, source=job.elevation,
            metadata={"pipeline_version": PIPELINE_VERSION, "source": _source_metadata(job.elevation, acquired, info),
                      "grid": _grid_metadata(site, grid), "stats": stats, "steps": list(steps), "warnings": warnings,
                      "classification": "observed: resampled/cropped NASA/USGS product; no values invented or filled"},
        )
        session.flush()
        slope_layer, slope = derive_slope_layer(session, run, site, grid, layer, elevation, steps)
        hazard_layer = derive_hazard_layer(session, run, site, grid, layer, slope_layer, slope, job.hazard, steps)

        imagery_layer = None
        if job.imagery is not None:
            imagery_layer = _ingest_imagery(session, storage, settings, run, site, grid, job, version, transport, warnings)
        session.commit()
    except Exception:
        session.rollback()
        run.cleanup()
        raise
    return TerrainSet(layer, slope_layer, hazard_layer, imagery_layer, warnings)


def _ingest_imagery(
    session: Session, storage: AssetStorage, settings: Settings, run: _Run, site: ExplorationSite,
    grid: TargetGrid, job: IngestionJob, elevation_version: str, transport, warnings: list[str],
) -> DataLayer:
    source = job.imagery
    assert source is not None
    acquired = _acquire(source, storage, settings, transport)
    info = inspect_source(acquired.path, grid, site.body_radius_m, "imagery", None, source.accept_ellipsoid_as_sphere)
    warnings += [f"imagery: {w}" for w in info.warnings]
    warped = warp_to_grid(acquired.path, grid, "imagery", info.native_resolution_m, source.band_indexes)
    version = job.dataset_version or f"src-{acquired.sha256[:12]}-r{job.resolution_m:g}m"
    layer = _get_or_create_layer(session, site, f"{job.layer_prefix}-imagery", LayerType.OPTICAL_IMAGERY,
                                 source.name, DataOrigin.OBSERVED, ProcessingStatus.PROCESSED, False,
                                 source.candidate_slug)
    session.flush()
    _replace_assets(session, layer, version)
    key_dir = run.version_dir(site.slug, layer.slug, version)
    tif, png = f"{key_dir}/imagery.tif", f"{key_dir}/preview.png"
    write_geotiff(run.storage.resolve(tif), grid, warped.data, FLOAT_NODATA)
    write_rgb_preview(run.storage.resolve(png), warped.data, warped.valid)
    meta = {**_tool_versions(warped.resampling), "crs_proj": _local_proj(site)}
    _add_asset(layer, run.storage, grid, tif, RasterRole.PROCESSED, "imagery", version, warped.data.shape[0],
               "float32", FLOAT_NODATA, "image/tiff", dict(meta))
    _add_asset(layer, run.storage, grid, png, RasterRole.DISPLAY_TILES, "preview", version, 4, "uint8", None,
               "image/png", {"crs_proj": _local_proj(site), "preview": "2-98 percentile stretch, NoData transparent"})
    valid_fraction = float(warped.valid.mean())
    _fill_layer(
        layer, site, grid, origin=DataOrigin.OBSERVED, status=ProcessingStatus.PROCESSED, is_demo=False,
        version=version, units=None, nodata=FLOAT_NODATA, source=source,
        metadata={"pipeline_version": PIPELINE_VERSION, "source": _source_metadata(source, acquired, info),
                  "grid": _grid_metadata(site, grid), "valid_fraction": round(valid_fraction, 6),
                  "aligned_to_elevation_version": elevation_version,
                  "note": "Aligned to the elevation grid; pixel values are source digital numbers resampled, not radiometrically calibrated."},
    )
    return layer


def _load_layer_array(storage: AssetStorage, layer: DataLayer, variable: str) -> np.ndarray:
    asset = next(
        (a for a in layer.assets if a.dataset_version == layer.dataset_version and a.variable == variable), None
    )
    if asset is None:
        raise ProcessingError(f"Layer '{layer.slug}' has no '{variable}' raster for version {layer.dataset_version}")
    data, _ = read_geotiff(storage.resolve(asset.storage_key))
    return data[0]


def _processed_elevation_layer(session: Session, site: ExplorationSite, slug: str) -> DataLayer:
    layer = session.scalars(select(DataLayer).where(DataLayer.site_id == site.id, DataLayer.slug == slug)).first()
    if layer is None or layer.layer_type != LayerType.ELEVATION:
        raise ProcessingError(f"Elevation layer '{slug}' not found for site '{site.slug}'")
    return layer


def regenerate_slope(
    session: Session, storage: AssetStorage, settings: Settings, site_slug: str, elevation_slug: str,
    hazard: HazardConfig = DEFAULT_HAZARD_CONFIG,
) -> TerrainSet:
    """Recompute slope from a stored elevation raster, then hazard (which depends on slope)."""
    site = _get_site(session, site_slug)
    elevation_layer = _processed_elevation_layer(session, site, elevation_slug)
    resolution = elevation_layer.spatial_resolution_m or 0.0
    grid = grid_for_site(site, resolution, settings.max_grid_cells)
    run, steps = _Run(storage), []
    try:
        elevation = _load_layer_array(storage, elevation_layer, "elevation_m")
        slope_layer, slope = derive_slope_layer(session, run, site, grid, elevation_layer, elevation, steps)
        hazard_layer = derive_hazard_layer(session, run, site, grid, elevation_layer, slope_layer, slope, hazard, steps)
        session.commit()
    except Exception:
        session.rollback()
        run.cleanup()
        raise
    return TerrainSet(elevation_layer, slope_layer, hazard_layer)


def regenerate_hazard(
    session: Session, storage: AssetStorage, settings: Settings, site_slug: str, elevation_slug: str,
    hazard: HazardConfig,
) -> TerrainSet:
    """Recompute only the hazard layer from the stored slope raster with a new configuration."""
    site = _get_site(session, site_slug)
    elevation_layer = _processed_elevation_layer(session, site, elevation_slug)
    slope_layer = session.scalars(
        select(DataLayer).where(DataLayer.derived_from_layer_id == elevation_layer.id, DataLayer.layer_type == LayerType.SLOPE)
    ).first()
    if slope_layer is None:
        raise ProcessingError("No slope layer exists for this elevation layer; run generate_slope first")
    grid = grid_for_site(site, elevation_layer.spatial_resolution_m or 0.0, settings.max_grid_cells)
    run, steps = _Run(storage), []
    try:
        slope = _load_layer_array(storage, slope_layer, "slope_deg")
        hazard_layer = derive_hazard_layer(session, run, site, grid, elevation_layer, slope_layer, slope, hazard, steps)
        session.commit()
    except Exception:
        session.rollback()
        run.cleanup()
        raise
    return TerrainSet(elevation_layer, slope_layer, hazard_layer)
