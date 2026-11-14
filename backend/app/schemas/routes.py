from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.models.enums import DataOrigin, RouteProfile, RouteStatus, WaypointType
from app.schemas.common import LocalPoint
from app.schemas.mission import MissionConstraints
from app.schemas.terrain import LayerRef

Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
BOTH_PROFILES = [RouteProfile.SHORTER_DISTANCE, RouteProfile.SAFETY_FIRST]


class DestinationSpec(BaseModel):
    """One stop: a stored science target or a raw coordinate (waypoint). Stops are visited in the order given."""

    model_config = ConfigDict(extra="forbid")

    science_target_id: UUID | None = None
    location: LocalPoint | None = None
    label: Label | None = None
    activity_duration_min: int | None = Field(default=None, ge=0, le=720)

    @model_validator(mode="after")
    def exactly_one_kind(self) -> "DestinationSpec":
        if (self.science_target_id is None) == (self.location is None):
            raise ValueError("Provide exactly one of science_target_id or location")
        return self


class PlanningInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: LocalPoint | None = Field(default=None, description="Defaults to the mission start, else the site habitat")
    destinations: list[DestinationSpec] | None = Field(
        default=None, max_length=50, description="Defaults to the mission's science targets in order"
    )
    return_to_habitat: bool = True
    eva_duration_limit_min: int | None = Field(default=None, ge=1, le=1440)
    return_reserve_min: int | None = Field(default=None, ge=0, le=1440)
    constraints: MissionConstraints | None = None
    elevation_layer_id: UUID | None = None


class RoutePlanRequest(PlanningInputs):
    site_id: UUID | None = None
    mission_id: UUID | None = None
    profiles: list[RouteProfile] = Field(default_factory=lambda: list(BOTH_PROFILES), min_length=1, max_length=2)

    @model_validator(mode="after")
    def check(self) -> "RoutePlanRequest":
        if self.site_id is None and self.mission_id is None:
            raise ValueError("Provide site_id or mission_id")
        if len(set(self.profiles)) != len(self.profiles):
            raise ValueError("profiles must not contain duplicates")
        return self


class MissionRouteRequest(PlanningInputs):
    """Plan with the mission's context and save the result as one of its routes. Geometry is never client-supplied."""

    profile: RouteProfile
    select: bool = Field(default=True, description="Make this the mission's selected route")


class RouteGeometryRead(BaseModel):
    coordinate_system: str = "site_local_enu"
    point_count: int
    coordinates: list[tuple[float, float]] = Field(description="Local metres (x east, y north) from the site centre")
    geographic: list[tuple[float, float]] = Field(description="Derived (longitude_deg_east, latitude_deg) per vertex")


class RouteMetricsRead(BaseModel):
    """All values are computed from the path geometry; none are configured or estimated up front."""

    distance_m: float
    travel_time_s: float
    activity_time_s: float
    total_time_s: float
    total_cost: float
    risk_score: float = Field(description="Length-weighted mean risk (0-1) along the path")
    peak_risk: float = Field(description="Highest risk (0-1) of any cell on the path")
    mean_slope_deg: float | None
    max_slope_deg: float | None
    elevation_gain_m: float
    elevation_loss_m: float
    high_risk_distance_m: float
    uncertain_distance_m: float = Field(description="Distance within the buffer around unknown terrain")


class RouteStopRead(BaseModel):
    sequence: int
    waypoint_type: WaypointType
    label: str
    location: LocalPoint
    longitude_deg_east: float
    latitude_deg: float
    science_target_id: UUID | None
    activity_duration_min: float
    distance_from_start_m: float
    arrival_time_s: float
    departure_time_s: float
    risk_score: float


class RouteLegRead(BaseModel):
    from_sequence: int
    to_sequence: int
    distance_m: float
    travel_time_s: float


class ConstraintCheckRead(BaseModel):
    eva_duration_limit_min: int
    return_reserve_min: int
    usable_time_s: float
    planned_total_time_s: float
    margin_s: float
    within_time_budget: bool
    return_planned: bool
    max_slope_deg_applied: float
    block_unknown_terrain_applied: bool
    violations: list[str]


class NoRouteRead(BaseModel):
    code: str
    message: str
    leg_index: int | None
    from_label: str | None
    to_label: str | None


class SearchStatsRead(BaseModel):
    expanded_nodes: int
    elapsed_ms: float


class PlannedRouteRead(BaseModel):
    id: UUID | None = None
    mission_id: UUID | None = None
    is_selected: bool = False
    profile: RouteProfile
    status: RouteStatus
    no_route: NoRouteRead | None
    geometry: RouteGeometryRead | None
    metrics: RouteMetricsRead | None
    stops: list[RouteStopRead]
    legs: list[RouteLegRead]
    constraints: ConstraintCheckRead
    search: SearchStatsRead
    cost_model: dict[str, Any]
    data_origin: DataOrigin
    warnings: list[str]


class RouteComparisonRead(BaseModel):
    distance_difference_m: float = Field(description="safety_first minus shorter_distance")
    distance_ratio: float | None
    risk_score_difference: float = Field(description="safety_first minus shorter_distance (negative = safer)")
    peak_risk_difference: float
    travel_time_difference_s: float
    same_geometry: bool
    summary: str


class PlanTerrainRead(BaseModel):
    is_demo: bool
    resolution_m: float
    elevation_layer: LayerRef
    slope_layer: LayerRef
    hazard_layer: LayerRef
    hazard_config_version: str
    routing_config_version: str


class RoutePlanResponse(BaseModel):
    site_id: UUID
    mission_id: UUID | None
    terrain: PlanTerrainRead
    routes: list[PlannedRouteRead]
    comparison: RouteComparisonRead | None
    warnings: list[str]
    notes: list[str]
