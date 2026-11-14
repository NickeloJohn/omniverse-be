import uuid
from typing import Any

from geoalchemy2 import WKBElement
from sqlalchemy import Boolean, CheckConstraint, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, local_geometry
from app.models.enums import DataOrigin, RouteProfile, RouteStatus, WaypointType, enum_column


class Route(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Route metrics must be computed from the stored geometry, never hardcoded."""

    __tablename__ = "routes"
    __table_args__ = (
        CheckConstraint("risk_score IS NULL OR risk_score BETWEEN 0 AND 1", name="ck_routes_risk_range"),
    )

    mission_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("missions.id", ondelete="CASCADE"), index=True
    )
    profile: Mapped[RouteProfile] = mapped_column(enum_column(RouteProfile, "route_profile"))
    status: Mapped[RouteStatus] = mapped_column(enum_column(RouteStatus, "route_status"))
    is_selected: Mapped[bool] = mapped_column(Boolean, default=False)
    geometry: Mapped[WKBElement | None] = mapped_column(local_geometry("LINESTRING"))
    distance_m: Mapped[float | None] = mapped_column(Float)
    estimated_duration_s: Mapped[float | None] = mapped_column(Float)
    total_cost: Mapped[float | None] = mapped_column(Float)
    risk_score: Mapped[float | None] = mapped_column(Float)
    max_cell_risk: Mapped[float | None] = mapped_column(Float)
    mean_slope_deg: Mapped[float | None] = mapped_column(Float)
    max_slope_deg: Mapped[float | None] = mapped_column(Float)
    elevation_gain_m: Mapped[float | None] = mapped_column(Float)
    elevation_loss_m: Mapped[float | None] = mapped_column(Float)
    no_route_reason: Mapped[str | None] = mapped_column(String(500))
    terrain_layer_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("data_layers.id", ondelete="SET NULL")
    )
    dataset_version: Mapped[str | None] = mapped_column(String(64))
    cost_model: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    data_origin: Mapped[DataOrigin] = mapped_column(enum_column(DataOrigin, "data_origin"))

    waypoints: Mapped[list["RouteWaypoint"]] = relationship(
        back_populates="route", cascade="all, delete-orphan", order_by="RouteWaypoint.sequence"
    )


class RouteWaypoint(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "route_waypoints"
    __table_args__ = (UniqueConstraint("route_id", "sequence", name="uq_waypoints_route_sequence"),)

    route_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("routes.id", ondelete="CASCADE"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    waypoint_type: Mapped[WaypointType] = mapped_column(enum_column(WaypointType, "waypoint_type"))
    location: Mapped[WKBElement] = mapped_column(local_geometry("POINT"))
    science_target_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("science_targets.id", ondelete="SET NULL")
    )
    distance_from_start_m: Mapped[float | None] = mapped_column(Float)
    eta_s: Mapped[float | None] = mapped_column(Float)
    risk_score: Mapped[float | None] = mapped_column(Float)

    route: Mapped[Route] = relationship(back_populates="waypoints")
