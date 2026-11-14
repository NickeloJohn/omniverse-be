"""Importing every model module registers its tables on Base.metadata (needed by Alembic)."""

from app.models.base import Base
from app.models.hazard import HazardZone
from app.models.layers import DataLayer, RasterAsset
from app.models.mission import Mission, MissionTarget
from app.models.route import Route, RouteWaypoint
from app.models.science import ScienceObservation, ScienceTarget
from app.models.simulation import EVASession, TelemetrySample
from app.models.site import ExplorationSite

__all__ = [
    "Base",
    "DataLayer",
    "EVASession",
    "ExplorationSite",
    "HazardZone",
    "Mission",
    "MissionTarget",
    "RasterAsset",
    "Route",
    "RouteWaypoint",
    "ScienceObservation",
    "ScienceTarget",
    "TelemetrySample",
]
