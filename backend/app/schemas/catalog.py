from datetime import date, datetime
from typing import Any
from uuid import UUID

from app.models.enums import DataOrigin, LayerType, ProcessingStatus, RasterRole
from app.schemas.common import LocalBoundsField, LocalPointField, ORMModel, Timestamped


class SiteRead(Timestamped):
    slug: str
    name: str
    body: str
    description: str | None
    center_longitude_deg_east: float
    center_latitude_deg: float
    geographic_crs: str
    body_radius_m: float
    local_crs_name: str
    local_frame_description: str
    width_m: float
    height_m: float
    habitat_location: LocalPointField
    data_origin: DataOrigin
    is_demo: bool


class RasterAssetRead(ORMModel):
    """Storage keys are internal and intentionally not exposed."""

    id: UUID
    role: RasterRole
    variable: str
    media_type: str
    width_px: int
    height_px: int
    band_count: int
    dtype: str | None
    pixel_size_x_m: float | None
    pixel_size_y_m: float | None
    nodata_value: float | None
    crs: str | None
    bounds: LocalBoundsField | None
    sha256: str | None
    dataset_version: str
    created_at: datetime


class LayerRead(Timestamped):
    site_id: UUID
    slug: str
    name: str
    description: str | None
    layer_type: LayerType
    data_origin: DataOrigin
    processing_status: ProcessingStatus
    is_demo: bool
    source_name: str | None
    source_url: str | None
    source_product_id: str | None
    acquisition_date: date | None
    spatial_resolution_m: float | None
    crs: str | None
    units: str | None
    nodata_value: float | None
    dataset_version: str
    processing_metadata: dict[str, Any]
    bounds: LocalBoundsField | None
    derived_from_layer_id: UUID | None
    assets: list[RasterAssetRead]
