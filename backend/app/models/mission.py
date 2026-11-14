import uuid
from typing import Any

from geoalchemy2 import WKBElement
from sqlalchemy import Boolean, CheckConstraint, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, local_geometry
from app.models.enums import MissionStatus, enum_column
from app.models.science import ScienceTarget


class MissionTarget(Base):
    """Ordered selection of science targets for a mission."""

    __tablename__ = "mission_targets"
    __table_args__ = (
        CheckConstraint(
            "planned_duration_min IS NULL OR planned_duration_min > 0",
            name="ck_mission_targets_duration",
        ),
    )

    mission_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("missions.id", ondelete="CASCADE"), primary_key=True
    )
    target_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("science_targets.id", ondelete="RESTRICT"), primary_key=True, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    planned_duration_min: Mapped[int | None] = mapped_column(Integer)

    target: Mapped[ScienceTarget] = relationship(lazy="joined")


class Mission(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "missions"
    __table_args__ = (
        CheckConstraint("eva_duration_limit_min > 0", name="ck_missions_eva_limit_positive"),
        CheckConstraint("return_reserve_min >= 0", name="ck_missions_reserve_nonnegative"),
        CheckConstraint(
            "return_reserve_min < eva_duration_limit_min", name="ck_missions_reserve_below_limit"
        ),
    )

    name: Mapped[str] = mapped_column(String(160))
    description: Mapped[str | None] = mapped_column(Text)
    site_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("exploration_sites.id", ondelete="RESTRICT"), index=True
    )
    status: Mapped[MissionStatus] = mapped_column(
        enum_column(MissionStatus, "mission_status"), default=MissionStatus.DRAFT, index=True
    )
    start_location: Mapped[WKBElement] = mapped_column(local_geometry("POINT"))
    start_longitude_deg_east: Mapped[float] = mapped_column(Float)
    start_latitude_deg: Mapped[float] = mapped_column(Float)
    eva_duration_limit_min: Mapped[int] = mapped_column(Integer)
    return_reserve_min: Mapped[int] = mapped_column(Integer)
    constraints: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    target_links: Mapped[list[MissionTarget]] = relationship(
        cascade="all, delete-orphan", order_by=MissionTarget.sequence, lazy="selectin"
    )

    @property
    def science_targets(self) -> list[MissionTarget]:
        return self.target_links
