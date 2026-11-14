from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.models.enums import MissionStatus, RouteProfile
from app.schemas.common import LocalPoint, LocalPointField, ORMModel, Timestamped
from app.schemas.science import ScienceTargetRead

MissionName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]


class MissionConstraints(BaseModel):
    """Prototype planning constraints, not validated NASA astronaut safety limits."""

    model_config = ConfigDict(extra="forbid")

    max_slope_deg: float = Field(default=15.0, gt=0, le=45)
    block_unknown_terrain: bool = True
    preferred_profile: RouteProfile = RouteProfile.SAFETY_FIRST


class MissionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: MissionName
    description: str | None = Field(default=None, max_length=4000)
    site_id: UUID
    start_location: LocalPoint | None = Field(
        default=None, description="Defaults to the site habitat location"
    )
    eva_duration_limit_min: int = Field(default=360, ge=1, le=1440)
    return_reserve_min: int = Field(default=60, ge=0, le=1440)
    constraints: MissionConstraints = Field(default_factory=MissionConstraints)
    science_target_ids: list[UUID] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def check_values(self) -> "MissionCreate":
        if self.return_reserve_min >= self.eva_duration_limit_min:
            raise ValueError("return_reserve_min must be less than eva_duration_limit_min")
        if len(set(self.science_target_ids)) != len(self.science_target_ids):
            raise ValueError("science_target_ids must not contain duplicates")
        return self


class MissionUpdate(BaseModel):
    """Partial update. site_id is immutable. Omitted fields are unchanged."""

    model_config = ConfigDict(extra="forbid")

    name: MissionName | None = None
    description: str | None = Field(default=None, max_length=4000)
    status: MissionStatus | None = None
    start_location: LocalPoint | None = None
    eva_duration_limit_min: int | None = Field(default=None, ge=1, le=1440)
    return_reserve_min: int | None = Field(default=None, ge=0, le=1440)
    constraints: MissionConstraints | None = None
    science_target_ids: list[UUID] | None = Field(default=None, max_length=50)

    @model_validator(mode="after")
    def check_values(self) -> "MissionUpdate":
        non_nullable = (
            "name",
            "status",
            "start_location",
            "eva_duration_limit_min",
            "return_reserve_min",
            "constraints",
            "science_target_ids",
        )
        for field in non_nullable:
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        ids = self.science_target_ids
        if ids is not None and len(set(ids)) != len(ids):
            raise ValueError("science_target_ids must not contain duplicates")
        return self


class MissionTargetRead(ORMModel):
    sequence: int
    planned_duration_min: int | None
    target: ScienceTargetRead


class MissionRead(Timestamped):
    name: str
    description: str | None
    site_id: UUID
    status: MissionStatus
    start_location: LocalPointField
    start_longitude_deg_east: float
    start_latitude_deg: float
    eva_duration_limit_min: int
    return_reserve_min: int
    constraints: MissionConstraints
    is_demo: bool
    science_targets: list[MissionTargetRead]
