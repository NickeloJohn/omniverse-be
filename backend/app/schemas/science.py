from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from app.models.enums import (
    CreationSource,
    DataOrigin,
    ObservationSource,
    ObservationType,
    VerificationStatus,
)
from app.schemas.common import LocalPoint, LocalPointField, Timestamped

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
LongText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=4000)]
MeasurementValue = float | str | bool


class ScienceTargetCreate(BaseModel):
    """Public creation only yields manual, 'proposed' targets; verification is not client-settable."""

    model_config = ConfigDict(extra="forbid")

    site_id: UUID
    name: ShortText
    description: LongText | None = None
    scientific_objective: LongText | None = None
    location: LocalPoint
    associated_mission: ShortText | None = None
    associated_product_id: ShortText | None = None
    geological_context: LongText | None = None
    estimated_activity_duration_min: int = Field(default=30, ge=1, le=720)


class ScienceTargetRead(Timestamped):
    site_id: UUID
    name: str
    description: str | None
    scientific_objective: str | None
    location: LocalPointField
    longitude_deg_east: float
    latitude_deg: float
    associated_mission: str | None
    associated_product_id: str | None
    geological_context: str | None
    estimated_activity_duration_min: int
    creation_source: CreationSource
    verification_status: VerificationStatus
    verification_reference: str | None


class ObservationCreate(BaseModel):
    """Public creation yields source=manual, data_origin=simulated; NASA product observations come from ingestion only."""

    model_config = ConfigDict(extra="forbid")

    site_id: UUID
    target_id: UUID | None = None
    mission_id: UUID | None = None
    observation_type: ObservationType
    title: ShortText
    notes: LongText | None = None
    observed_at: datetime | None = None
    location: LocalPoint | None = None
    measurements: dict[Annotated[str, StringConstraints(max_length=64)], MeasurementValue] = Field(
        default_factory=dict, max_length=50
    )

    @field_validator("observed_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("observed_at must include a timezone offset")
        return value


class ObservationRead(Timestamped):
    site_id: UUID
    target_id: UUID | None
    mission_id: UUID | None
    eva_session_id: UUID | None
    observation_type: ObservationType
    title: str
    notes: str | None
    observed_at: datetime
    location: LocalPointField | None
    longitude_deg_east: float | None
    latitude_deg: float | None
    measurements: dict[str, MeasurementValue]
    source: ObservationSource
    data_origin: DataOrigin
