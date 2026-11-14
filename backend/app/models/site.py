from typing import TYPE_CHECKING

from geoalchemy2 import WKBElement
from sqlalchemy import Boolean, CheckConstraint, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, local_geometry
from app.models.enums import DataOrigin, enum_column
from app.utils.mars_coords import (
    GEOGRAPHIC_CRS_NAME,
    LOCAL_CRS_NAME,
    LOCAL_FRAME_DESCRIPTION,
    MARS_REFERENCE_RADIUS_M,
)

if TYPE_CHECKING:
    from app.models.layers import DataLayer


class ExplorationSite(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "exploration_sites"
    __table_args__ = (
        CheckConstraint("center_latitude_deg BETWEEN -90 AND 90", name="ck_sites_lat_range"),
        CheckConstraint("center_longitude_deg_east BETWEEN -180 AND 360", name="ck_sites_lon_range"),
        CheckConstraint("width_m > 0 AND height_m > 0", name="ck_sites_extent_positive"),
    )

    slug: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(160))
    body: Mapped[str] = mapped_column(String(32), default="Mars")
    description: Mapped[str | None] = mapped_column(Text)

    # Mars geographic metadata, kept separate from the local metric frame used for computation.
    center_longitude_deg_east: Mapped[float] = mapped_column(Float)
    center_latitude_deg: Mapped[float] = mapped_column(Float)
    geographic_crs: Mapped[str] = mapped_column(String(200), default=GEOGRAPHIC_CRS_NAME)
    body_radius_m: Mapped[float] = mapped_column(Float, default=MARS_REFERENCE_RADIUS_M)

    local_crs_name: Mapped[str] = mapped_column(String(64), default=LOCAL_CRS_NAME)
    local_frame_description: Mapped[str] = mapped_column(Text, default=LOCAL_FRAME_DESCRIPTION)
    width_m: Mapped[float] = mapped_column(Float)
    height_m: Mapped[float] = mapped_column(Float)
    extent: Mapped[WKBElement] = mapped_column(local_geometry("POLYGON"))
    habitat_location: Mapped[WKBElement] = mapped_column(local_geometry("POINT"))

    data_origin: Mapped[DataOrigin] = mapped_column(enum_column(DataOrigin, "data_origin"))
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    layers: Mapped[list["DataLayer"]] = relationship(back_populates="site", order_by="DataLayer.slug")
