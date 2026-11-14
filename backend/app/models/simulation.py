import uuid
from datetime import datetime
from typing import Any

from geoalchemy2 import WKBElement
from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, local_geometry, utcnow
from app.models.enums import DataOrigin, EVAStatus, enum_column


class EVASession(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Deterministic simulated EVA. Never actual astronaut telemetry."""

    __tablename__ = "eva_sessions"
    __table_args__ = (
        CheckConstraint("data_origin = 'simulated'", name="ck_eva_sessions_simulated_only"),
        CheckConstraint("sim_elapsed_s >= 0", name="ck_eva_sessions_elapsed_nonnegative"),
        CheckConstraint("distance_along_route_m >= 0", name="ck_eva_sessions_distance_nonnegative"),
        CheckConstraint("oxygen_remaining_pct BETWEEN 0 AND 100", name="ck_eva_sessions_oxygen_range"),
        CheckConstraint("battery_remaining_pct BETWEEN 0 AND 100", name="ck_eva_sessions_battery_range"),
    )

    mission_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("missions.id", ondelete="CASCADE"), index=True
    )
    route_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("routes.id", ondelete="RESTRICT"))
    status: Mapped[EVAStatus] = mapped_column(
        enum_column(EVAStatus, "eva_status"), default=EVAStatus.CREATED
    )
    sim_elapsed_s: Mapped[float] = mapped_column(Float, default=0.0)
    distance_along_route_m: Mapped[float] = mapped_column(Float, default=0.0)
    oxygen_remaining_pct: Mapped[float] = mapped_column(Float, default=100.0)
    battery_remaining_pct: Mapped[float] = mapped_column(Float, default=100.0)
    resource_model: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    data_origin: Mapped[DataOrigin] = mapped_column(
        enum_column(DataOrigin, "data_origin"), default=DataOrigin.SIMULATED
    )

    telemetry: Mapped[list["TelemetrySample"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="TelemetrySample.sim_time_s"
    )


class TelemetrySample(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "telemetry_samples"
    __table_args__ = (
        CheckConstraint("data_origin = 'simulated'", name="ck_telemetry_simulated_only"),
        CheckConstraint("sim_time_s >= 0", name="ck_telemetry_time_nonnegative"),
        CheckConstraint("distance_along_route_m >= 0", name="ck_telemetry_distance_nonnegative"),
        CheckConstraint("oxygen_remaining_pct BETWEEN 0 AND 100", name="ck_telemetry_oxygen_range"),
        CheckConstraint("battery_remaining_pct BETWEEN 0 AND 100", name="ck_telemetry_battery_range"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("eva_sessions.id", ondelete="CASCADE"), index=True
    )
    sim_time_s: Mapped[float] = mapped_column(Float)
    position: Mapped[WKBElement] = mapped_column(local_geometry("POINT"))
    distance_along_route_m: Mapped[float] = mapped_column(Float)
    speed_mps: Mapped[float] = mapped_column(Float)
    oxygen_remaining_pct: Mapped[float] = mapped_column(Float)
    battery_remaining_pct: Mapped[float] = mapped_column(Float)
    estimated_return_time_s: Mapped[float | None] = mapped_column(Float)
    risk_level: Mapped[str | None] = mapped_column(String(32))
    warnings: Mapped[list[str]] = mapped_column(JSONB, default=list)
    data_origin: Mapped[DataOrigin] = mapped_column(
        enum_column(DataOrigin, "data_origin"), default=DataOrigin.SIMULATED
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped[EVASession] = relationship(back_populates="telemetry")
