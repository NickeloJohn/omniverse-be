"""Terrain loading and point sampling for the API and, later, the route planner.

Processed grids are cached in memory keyed by layer ids and dataset versions, so repeated map queries do not
re-read rasters; a new dataset version naturally gets a new cache entry.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import NotFoundError
from app.models import DataLayer, ExplorationSite
from app.models.enums import LayerType, ProcessingStatus
from app.processing.errors import ProcessingError
from app.processing.grid import TargetGrid
from app.processing.hazard import (
    CATEGORY_EXCLUDED,
    CATEGORY_NAMES,
    CATEGORY_UNKNOWN,
    DISCLAIMERS,
    CostGrid,
    HazardConfig,
    TerrainHazard,
)
from app.processing.io import read_geotiff
from app.processing.terrain import max_neighbor_step_m
from app.repositories import hazards as hazard_repo
from app.services.geo import ensure_within_site, geographic_for
from app.services.hazard_service import get_cost_grid
from app.services.ingestion_service import grid_for_site
from app.services.storage import AssetStorage

USABLE_STATUSES = (ProcessingStatus.PROCESSED, ProcessingStatus.DEMO)
_CACHE: "OrderedDict[tuple, TerrainData]" = OrderedDict()
_CACHE_SIZE = 3
_LOCK = threading.Lock()


class TerrainNotAvailable(NotFoundError):
    code = "terrain_not_available"


@dataclass
class TerrainData:
    key: tuple
    site_id: UUID
    grid: TargetGrid
    elevation: np.ndarray
    slope: np.ndarray
    terrain: TerrainHazard
    config: HazardConfig
    elevation_ref: dict[str, Any]
    slope_ref: dict[str, Any]
    hazard_ref: dict[str, Any]

    @property
    def is_demo(self) -> bool:
        return bool(self.elevation_ref["is_demo"])


def _current_assets(layer: DataLayer) -> dict[str, Any]:
    return {a.variable: a for a in layer.assets if a.dataset_version == layer.dataset_version}


def select_terrain_layers(
    session: Session, site_id: UUID, elevation_layer_id: UUID | None = None
) -> tuple[DataLayer, DataLayer, DataLayer]:
    """Prefer a real (non-demo) processed terrain set, then the newest; derivatives follow the lineage link."""
    stmt = select(DataLayer).where(
        DataLayer.site_id == site_id,
        DataLayer.layer_type == LayerType.ELEVATION,
        DataLayer.processing_status.in_(USABLE_STATUSES),
    )
    if elevation_layer_id is not None:
        stmt = stmt.where(DataLayer.id == elevation_layer_id)
    candidates = [
        layer for layer in session.scalars(stmt.order_by(DataLayer.is_demo, DataLayer.updated_at.desc()))
        if "elevation_m" in _current_assets(layer)
    ]
    if not candidates:
        raise TerrainNotAvailable("No processed terrain is available for this site")
    elevation = candidates[0]

    def derived(layer_type: LayerType, variable: str) -> DataLayer:
        layer = session.scalars(
            select(DataLayer).where(
                DataLayer.derived_from_layer_id == elevation.id,
                DataLayer.layer_type == layer_type,
                DataLayer.processing_status.in_(USABLE_STATUSES),
            )
        ).first()
        if layer is None or variable not in _current_assets(layer):
            raise TerrainNotAvailable(f"Terrain derivative '{layer_type.value}' is missing for '{elevation.slug}'")
        return layer

    return elevation, derived(LayerType.SLOPE, "slope_deg"), derived(LayerType.HAZARD_COST, "risk_score")


def load_terrain(
    session: Session, storage: AssetStorage, site: ExplorationSite, elevation_layer_id: UUID | None = None
) -> TerrainData:
    elevation_layer, slope_layer, hazard_layer = select_terrain_layers(session, site.id, elevation_layer_id)
    key = tuple((layer.id, layer.dataset_version) for layer in (elevation_layer, slope_layer, hazard_layer))
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]

    settings = get_settings()
    grid = grid_for_site(site, elevation_layer.spatial_resolution_m or 0.0, settings.max_grid_cells)

    def array(layer: DataLayer, variable: str, band: int = 0) -> np.ndarray:
        data, _ = read_geotiff(storage.resolve(_current_assets(layer)[variable].storage_key))
        if data.shape[1:] != grid.shape:
            raise ProcessingError(f"Stored raster '{layer.slug}/{variable}' does not match the site grid")
        return data[band]

    category = array(hazard_layer, "hazard_classes", 0).astype(np.uint8)
    blocked = array(hazard_layer, "hazard_classes", 1).astype(bool)
    terrain = TerrainHazard(array(hazard_layer, "risk_score").astype(np.float32), category, blocked)
    config = HazardConfig.model_validate(hazard_layer.processing_metadata["hazard_config"])
    data = TerrainData(key, site.id, grid, array(elevation_layer, "elevation_m"), array(slope_layer, "slope_deg"),
                       terrain, config, _layer_ref(elevation_layer), _layer_ref(slope_layer), _layer_ref(hazard_layer))
    with _LOCK:
        _CACHE[key] = data
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return data


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def _layer_ref(layer: DataLayer) -> dict[str, Any]:
    return {
        "id": layer.id, "slug": layer.slug, "name": layer.name, "layer_type": layer.layer_type,
        "data_origin": layer.data_origin, "processing_status": layer.processing_status,
        "dataset_version": layer.dataset_version, "is_demo": layer.is_demo,
        "spatial_resolution_m": layer.spatial_resolution_m, "units": layer.units,
        "source_name": layer.source_name,
    }


def _notes(data_is_demo: bool, warnings: list[str]) -> list[str]:
    notes = list(DISCLAIMERS)
    if data_is_demo:
        notes.insert(0, "DEMO: synthetic terrain generated by MARS-NAV; contains no NASA measurements.")
    return notes + warnings


def terrain_metadata(
    session: Session, site: ExplorationSite, elevation_layer_id: UUID | None = None
) -> dict[str, Any]:
    elevation, slope, hazard = select_terrain_layers(session, site.id, elevation_layer_id)
    grid_meta = elevation.processing_metadata.get("grid", {})
    min_x, min_y, max_x, max_y = grid_meta.get("bounds_local_m", [0, 0, 0, 0])
    warnings = list(elevation.processing_metadata.get("warnings", []))
    return {
        "site_id": site.id,
        "is_demo": elevation.is_demo,
        "grid": {
            "frame": grid_meta.get("frame"), "proj": grid_meta.get("proj"),
            "resolution_m": grid_meta.get("resolution_m"),
            "width_px": grid_meta.get("size_px", [0, 0])[0], "height_px": grid_meta.get("size_px", [0, 0])[1],
            "bounds": {"min_x_m": min_x, "min_y_m": min_y, "max_x_m": max_x, "max_y_m": max_y},
            "transform": grid_meta.get("transform"),
        },
        "elevation_layer": _layer_ref(elevation),
        "slope_layer": _layer_ref(slope),
        "hazard_layer": _layer_ref(hazard),
        "elevation_stats": elevation.processing_metadata.get("stats", {}),
        "slope_stats": slope.processing_metadata.get("stats", {}),
        "hazard_stats": hazard.processing_metadata.get("stats", {}),
        "hazard_config": hazard.processing_metadata.get("hazard_config", {}),
        "notes": _notes(elevation.is_demo, warnings),
    }


def sample_point(
    session: Session, storage: AssetStorage, site: ExplorationSite, x_m: float, y_m: float,
    elevation_layer_id: UUID | None = None,
) -> dict[str, Any]:
    """Nearest-cell terrain sample. Unknown cells are reported as invalid and blocked, never filled."""
    ensure_within_site(site, x_m, y_m, "point")
    data = load_terrain(session, storage, site, elevation_layer_id)
    cost: CostGrid = get_cost_grid(session, data)
    row, col = data.grid.cell_of(x_m, y_m) or (None, None)
    if row is None:
        raise ProcessingError("Point lies outside the terrain grid")

    elevation, slope = float(data.elevation[row, col]), float(data.slope[row, col])
    category = int(cost.category[row, col])
    reasons: list[str] = []
    if category == CATEGORY_EXCLUDED:
        reasons.append("slope_above_limit")
    if category == CATEGORY_UNKNOWN and data.config.block_unknown:
        reasons.append("unknown_terrain")

    zones = []
    for zone in hazard_repo.zones_for_site(session, site.id):
        mask = cost.zone_masks.get(zone.id)
        if mask is None or not mask[row, col]:
            continue
        zones.append({
            "id": zone.id, "name": zone.name, "zone_type": zone.zone_type, "is_blocking": zone.is_blocking,
            "risk_score": zone.risk_score, "data_origin": zone.data_origin,
        })
        if zone.is_blocking:
            reasons.append(f"hazard_zone:{zone.name}")

    lon, lat = geographic_for(site, x_m, y_m)
    valid = bool(np.isfinite(elevation))
    step = max_neighbor_step_m(data.elevation, row, col)
    return {
        "site_id": site.id, "x_m": x_m, "y_m": y_m, "longitude_deg_east": lon, "latitude_deg": lat,
        "row": row, "col": col, "method": "nearest", "resolution_m": data.grid.resolution_m,
        "is_valid": valid,
        "elevation_m": elevation if valid else None,
        "slope_deg": slope if np.isfinite(slope) else None,
        "max_neighbor_step_m": step,
        "terrain_risk_score": float(data.terrain.risk[row, col]),
        "risk_score": float(cost.risk[row, col]),
        "terrain_category": CATEGORY_NAMES[category],
        "blocked": bool(cost.blocked[row, col]),
        "blocked_reasons": reasons,
        "hazard_zones": zones,
        "provenance": {
            "elevation_layer": data.elevation_ref,
            "slope_layer": data.slope_ref,
            "hazard_layer": data.hazard_ref,
            "hazard_config_version": data.config.version,
        },
        "notes": _notes(data.is_demo, []),
    }
