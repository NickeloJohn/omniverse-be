import uuid
from datetime import datetime
from typing import Any

from geoalchemy2 import WKBElement
from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, local_geometry, utcnow
from app.models.enums import (
    CreationSource,
    DataOrigin,
    ObservationSource,
    ObservationType,
    VerificationStatus,
    enum_column,
)


class ScienceTarget(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Manual targets are proposals; only a NASA-verified status may mark one as verified."""

    __tablename__ = "science_targets"
    __table_args__ = (
        CheckConstraint(
            "estimated_activity_duration_min > 0", name="ck_targets_duration_positive"
        ),
    )

    site_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("exploration_sites.id", ondelete="RESTRICT"), index=True
    )
    name: Mapped[str] = mapped_column(String(160))
    description: Mapped[str | None] = mapped_column(Text)
    scientific_objective: Mapped[str | None] = mapped_column(Text)
    location: Mapped[WKBElement] = mapped_column(local_geometry("POINT"))
    longitude_deg_east: Mapped[float] = mapped_column(Float)
    latitude_deg: Mapped[float] = mapped_column(Float)
    associated_mission: Mapped[str | None] = mapped_column(String(160))
    associated_product_id: Mapped[str | None] = mapped_column(String(200))
    geological_context: Mapped[str | None] = mapped_column(Text)
    estimated_activity_duration_min: Mapped[int] = mapped_column(Integer, default=30)
    creation_source: Mapped[CreationSource] = mapped_column(
        enum_column(CreationSource, "creation_source")
    )
    verification_status: Mapped[VerificationStatus] = mapped_column(
        enum_column(VerificationStatus, "verification_status")
    )
    verification_reference: Mapped[str | None] = mapped_column(String(500))


class ScienceObservation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "science_observations"

    site_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("exploration_sites.id", ondelete="RESTRICT"), index=True
    )
    target_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("science_targets.id", ondelete="SET NULL"), index=True
    )
    mission_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("missions.id", ondelete="SET NULL"), index=True
    )
    eva_session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("eva_sessions.id", ondelete="SET NULL"), index=True
    )
    observation_type: Mapped[ObservationType] = mapped_column(
        enum_column(ObservationType, "observation_type")
    )
    title: Mapped[str] = mapped_column(String(200))
    notes: Mapped[str | None] = mapped_column(Text)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    location: Mapped[WKBElement | None] = mapped_column(local_geometry("POINT"))
    longitude_deg_east: Mapped[float | None] = mapped_column(Float)
    latitude_deg: Mapped[float | None] = mapped_column(Float)
    measurements: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    source: Mapped[ObservationSource] = mapped_column(enum_column(ObservationSource, "observation_source"))
    # Manual entries are 'simulated': recorded in the simulation, not instrument data.
    data_origin: Mapped[DataOrigin] = mapped_column(enum_column(DataOrigin, "data_origin"))
