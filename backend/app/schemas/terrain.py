from typing import Annotated, Any
from uuid import UUID

from geoalchemy2 import WKBElement
from geoalchemy2.shape import to_shape
from pydantic import BaseModel, BeforeValidator

from app.models.enums import DataOrigin, HazardZoneType, LayerType, ProcessingStatus
from app.schemas.common import LocalBounds, Timestamped


class LayerRef(BaseModel):
    id: UUID
    slug: str
    name: str
    layer_type: LayerType
    data_origin: DataOrigin
    processing_status: ProcessingStatus
    dataset_version: str
    is_demo: bool
    spatial_resolution_m: float | None
    units: str | None
    source_name: str | None


class GridMetadata(BaseModel):
    frame: str | None
    proj: str | None
    resolution_m: float | None
    width_px: int
    height_px: int
    bounds: LocalBounds
    transform: list[float] | None


class TerrainMetadataRead(BaseModel):
    site_id: UUID
    is_demo: bool
    grid: GridMetadata
    elevation_layer: LayerRef
    slope_layer: LayerRef
    hazard_layer: LayerRef
    elevation_stats: dict[str, Any]
    slope_stats: dict[str, Any]
    hazard_stats: dict[str, Any]
    hazard_config: dict[str, Any]
    notes: list[str]


class HazardZoneHit(BaseModel):
    id: UUID
    name: str
    zone_type: HazardZoneType
    is_blocking: bool
    risk_score: float
    data_origin: DataOrigin


class TerrainProvenance(BaseModel):
    elevation_layer: LayerRef
    slope_layer: LayerRef
    hazard_layer: LayerRef
    hazard_config_version: str


class TerrainSampleRead(BaseModel):
    site_id: UUID
    x_m: float
    y_m: float
    longitude_deg_east: float
    latitude_deg: float
    row: int
    col: int
    method: str
    resolution_m: float
    # True when an elevation value exists for the cell. Slope may still be missing at borders/NoData margins.
    is_valid: bool
    elevation_m: float | None
    slope_deg: float | None
    max_neighbor_step_m: float | None
    terrain_risk_score: float
    risk_score: float
    terrain_category: str
    blocked: bool
    blocked_reasons: list[str]
    hazard_zones: list[HazardZoneHit]
    provenance: TerrainProvenance
    notes: list[str]


class LocalPolygon(BaseModel):
    coordinate_system: str = "site_local_enu"
    exterior: list[tuple[float, float]]


def _polygon(value: Any) -> Any:
    if isinstance(value, WKBElement):
        return {"exterior": [(x, y) for x, y in to_shape(value).exterior.coords]}
    return value


class HazardZoneRead(Timestamped):
    site_id: UUID
    name: str
    description: str | None
    zone_type: HazardZoneType
    risk_score: float
    is_blocking: bool
    data_origin: DataOrigin
    source: str | None
    geometry: Annotated[LocalPolygon, BeforeValidator(_polygon)]
