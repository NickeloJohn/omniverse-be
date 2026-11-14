from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.enums import DataOrigin, EVAStatus, RiskLevel
from app.schemas.common import LocalPoint


class ResourceModel(BaseModel):
    """Explicit prototype rates; not measured or validated astronaut spacesuit performance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(default="prototype-resources-v1", min_length=1, max_length=64)
    oxygen_use_pct_per_hour_moving: float = Field(default=12.0, ge=0, le=100)
    oxygen_use_pct_per_hour_activity: float = Field(default=15.0, ge=0, le=100)
    battery_use_pct_per_hour_moving: float = Field(default=8.0, ge=0, le=100)
    battery_use_pct_per_hour_activity: float = Field(default=10.0, ge=0, le=100)
    oxygen_warning_pct: float = Field(default=30.0, ge=0, le=100)
    battery_warning_pct: float = Field(default=25.0, ge=0, le=100)
    risk_warning_threshold: float = Field(default=0.6, ge=0, le=1)
    risk_critical_threshold: float = Field(default=0.85, ge=0, le=1)

    @model_validator(mode="after")
    def critical_risk_exceeds_warning(self) -> "ResourceModel":
        if self.risk_critical_threshold <= self.risk_warning_threshold:
            raise ValueError("risk_critical_threshold must exceed risk_warning_threshold")
        return self


class SimulationStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route_id: UUID
    resource_model: ResourceModel = Field(default_factory=ResourceModel)


class SimulationAdvanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    advance_by_s: float = Field(gt=0, le=86_400, allow_inf_nan=False,
                                description="Advance the authoritative simulation clock by this many seconds")


class TelemetryRead(BaseModel):
    id: UUID
    session_id: UUID
    sim_time_s: float
    position: LocalPoint
    distance_along_route_m: float
    speed_mps: float
    oxygen_remaining_pct: float
    battery_remaining_pct: float
    estimated_return_time_s: float | None
    risk_level: RiskLevel
    warnings: list[str]
    data_origin: DataOrigin
    created_at: datetime


class SimulationStateRead(BaseModel):
    id: UUID
    mission_id: UUID
    route_id: UUID
    status: EVAStatus
    sim_elapsed_s: float
    distance_along_route_m: float
    total_route_distance_m: float
    route_progress_pct: float
    position: LocalPoint
    oxygen_remaining_pct: float
    battery_remaining_pct: float
    estimated_return_time_s: float | None
    current_risk_score: float | None
    risk_level: RiskLevel
    warnings: list[str]
    resource_model: ResourceModel
    telemetry_count: int
    latest_telemetry: TelemetryRead | None
    data_origin: DataOrigin
    created_at: datetime
    updated_at: datetime


class SimulationStartedRead(BaseModel):
    session: SimulationStateRead
    mission_status: str
    message: str


class SimulationAdvanceRead(BaseModel):
    session: SimulationStateRead
    telemetry: TelemetryRead
    advanced_by_s: float
    completed: bool


class SimulationResetRead(BaseModel):
    session: SimulationStateRead
    message: str
    data_origin: DataOrigin
