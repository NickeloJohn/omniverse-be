from enum import StrEnum

import sqlalchemy as sa


class DataOrigin(StrEnum):
    OBSERVED = "observed"
    DERIVED = "derived"
    SIMULATED = "simulated"


class LayerType(StrEnum):
    OPTICAL_IMAGERY = "optical_imagery"
    ELEVATION = "elevation"
    SLOPE = "slope"
    HAZARD_COST = "hazard_cost"
    GEOLOGY = "geology"
    SPECTRAL = "spectral"
    ENVIRONMENTAL = "environmental"


class ProcessingStatus(StrEnum):
    PLANNED = "planned"
    DOWNLOADED = "downloaded"
    PROCESSED = "processed"
    FAILED = "failed"
    DEMO = "demo"


class RasterRole(StrEnum):
    SOURCE = "source"
    PROCESSED = "processed"
    DISPLAY_TILES = "display_tiles"


class CreationSource(StrEnum):
    MANUAL = "manual"
    NASA_CATALOG = "nasa_catalog"
    DEMO_SEED = "demo_seed"


class VerificationStatus(StrEnum):
    PROPOSED = "proposed"
    UNDER_REVIEW = "under_review"
    NASA_VERIFIED = "nasa_verified"


class ObservationSource(StrEnum):
    MANUAL = "manual"
    NASA_PRODUCT = "nasa_product"
    SIMULATED_EVA = "simulated_eva"


class ObservationType(StrEnum):
    VISUAL = "visual"
    GEOLOGICAL = "geological"
    SPECTRAL = "spectral"
    ENVIRONMENTAL = "environmental"
    HAZARD = "hazard"
    OTHER = "other"


class MissionStatus(StrEnum):
    DRAFT = "draft"
    PLANNED = "planned"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"


class RouteProfile(StrEnum):
    SHORTER_DISTANCE = "shorter_distance"
    SAFETY_FIRST = "safety_first"


class RouteStatus(StrEnum):
    FOUND = "found"
    NO_ROUTE = "no_route"


class WaypointType(StrEnum):
    START = "start"
    SCIENCE_TARGET = "science_target"
    WAYPOINT = "waypoint"
    HABITAT_RETURN = "habitat_return"


class HazardZoneType(StrEnum):
    OBSTACLE = "obstacle"
    RESTRICTED = "restricted"
    UNCERTAIN_TERRAIN = "uncertain_terrain"


class EVAStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"


class RiskLevel(StrEnum):
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


def enum_column(enum_cls: type[StrEnum], name: str) -> sa.Enum:
    """Portable string-backed enum with a CHECK constraint (avoids native PG enum migrations)."""
    return sa.Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda members: [m.value for m in members],
    )
