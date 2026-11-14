import uuid

from geoalchemy2 import WKBElement
from sqlalchemy import Boolean, CheckConstraint, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, local_geometry
from app.models.enums import DataOrigin, HazardZoneType, enum_column


class HazardZone(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Obstacle or restricted area in local metric coordinates. Orbital data cannot resolve astronaut-scale obstacles."""

    __tablename__ = "hazard_zones"
    __table_args__ = (
        CheckConstraint("risk_score BETWEEN 0 AND 1", name="ck_hazards_risk_range"),
    )

    site_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("exploration_sites.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(160))
    description: Mapped[str | None] = mapped_column(Text)
    zone_type: Mapped[HazardZoneType] = mapped_column(enum_column(HazardZoneType, "hazard_zone_type"))
    geometry: Mapped[WKBElement] = mapped_column(local_geometry("POLYGON"))
    risk_score: Mapped[float] = mapped_column(Float)
    is_blocking: Mapped[bool] = mapped_column(Boolean, default=False)
    data_origin: Mapped[DataOrigin] = mapped_column(enum_column(DataOrigin, "data_origin"))
    source: Mapped[str | None] = mapped_column(String(200))
