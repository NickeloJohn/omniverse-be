import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from geoalchemy2 import WKBElement
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, local_geometry, utcnow
from app.models.enums import DataOrigin, LayerType, ProcessingStatus, RasterRole, enum_column

if TYPE_CHECKING:
    from app.models.site import ExplorationSite


class DataLayer(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A scientific or derived dataset with provenance. Layers without ingested rasters stay 'planned'."""

    __tablename__ = "data_layers"
    __table_args__ = (
        UniqueConstraint("site_id", "slug", name="uq_data_layers_site_slug"),
        CheckConstraint(
            "spatial_resolution_m IS NULL OR spatial_resolution_m > 0", name="ck_layers_resolution"
        ),
    )

    site_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("exploration_sites.id", ondelete="CASCADE"), index=True
    )
    slug: Mapped[str] = mapped_column(String(96))
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    layer_type: Mapped[LayerType] = mapped_column(enum_column(LayerType, "layer_type"))

    data_origin: Mapped[DataOrigin] = mapped_column(enum_column(DataOrigin, "data_origin"))
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        enum_column(ProcessingStatus, "processing_status")
    )
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    # Provenance. All nullable: unknown values must stay unknown, never be fabricated.
    source_name: Mapped[str | None] = mapped_column(String(200))
    source_url: Mapped[str | None] = mapped_column(String(500))
    source_product_id: Mapped[str | None] = mapped_column(String(200))
    acquisition_date: Mapped[date | None] = mapped_column(Date)
    spatial_resolution_m: Mapped[float | None] = mapped_column(Float)
    crs: Mapped[str | None] = mapped_column(String(200))
    units: Mapped[str | None] = mapped_column(String(32))
    nodata_value: Mapped[float | None] = mapped_column(Float)
    dataset_version: Mapped[str] = mapped_column(String(64), default="0")
    processing_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    bounds: Mapped[WKBElement | None] = mapped_column(local_geometry("POLYGON"))
    derived_from_layer_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("data_layers.id", ondelete="SET NULL"), index=True
    )

    site: Mapped["ExplorationSite"] = relationship(back_populates="layers")
    assets: Mapped[list["RasterAsset"]] = relationship(
        back_populates="layer", cascade="all, delete-orphan", order_by="RasterAsset.created_at"
    )


class RasterAsset(UUIDPrimaryKeyMixin, Base):
    """A stored raster file for a layer. storage_key is relative to the asset root and never exposed by the API."""

    __tablename__ = "raster_assets"
    __table_args__ = (
        CheckConstraint("width_px > 0 AND height_px > 0", name="ck_assets_dimensions"),
        UniqueConstraint(
            "layer_id", "dataset_version", "role", "variable", name="uq_assets_layer_version_role_variable"
        ),
    )

    layer_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_layers.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[RasterRole] = mapped_column(enum_column(RasterRole, "raster_role"))
    # What the raster holds, e.g. elevation_m, slope_deg, risk_score, hazard_classes, preview.
    variable: Mapped[str] = mapped_column(String(64), server_default="unspecified")
    storage_key: Mapped[str] = mapped_column(String(500))
    media_type: Mapped[str] = mapped_column(String(100))
    width_px: Mapped[int] = mapped_column(Integer)
    height_px: Mapped[int] = mapped_column(Integer)
    band_count: Mapped[int] = mapped_column(Integer, default=1)
    dtype: Mapped[str | None] = mapped_column(String(32))
    pixel_size_x_m: Mapped[float | None] = mapped_column(Float)
    pixel_size_y_m: Mapped[float | None] = mapped_column(Float)
    nodata_value: Mapped[float | None] = mapped_column(Float)
    transform: Mapped[list[float] | None] = mapped_column(JSONB)
    crs: Mapped[str | None] = mapped_column(String(200))
    bounds: Mapped[WKBElement | None] = mapped_column(local_geometry("POLYGON"))
    sha256: Mapped[str | None] = mapped_column(String(64))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    dataset_version: Mapped[str] = mapped_column(String(64), default="0")
    processing_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    layer: Mapped[DataLayer] = relationship(back_populates="assets")
