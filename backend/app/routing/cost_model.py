"""Route cost model.

Moving between two adjacent traversable cells a -> b of geometric length d (cell size, or cell size * sqrt(2)
for diagonals) costs

    SHORTER_DISTANCE:  d
    SAFETY_FIRST:      d * (f(a) + f(b)) / 2,   f(c) = 1 + w_risk * risk(c)
                                                         + w_slope * (slope(c) / slope_limit)^2
                                                         + w_unc * uncertainty(c)

risk is the 0-1 terrain/hazard risk score, slope(c) is NaN-safe (unknown slope counts as the limit), and
uncertainty falls linearly from 1 next to unknown (NoData) terrain to 0 at `uncertainty_buffer_m`.
Every f(c) >= 1, so the cost of a path is never below its geometric length. That makes the octile-distance
heuristic admissible and consistent (see astar.py). Blocked cells are impassable, not merely expensive.

Travel time uses a configurable walking-speed model that slows linearly with slope. All weights and speeds are
PROTOTYPE values. They are not validated astronaut or spacesuit performance data.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import RouteProfile
from app.routing.grid import RoutingGrid

COST_FUNCTION_DESCRIPTIONS = {
    RouteProfile.SHORTER_DISTANCE: "cost = geometric length over traversable cells",
    RouteProfile.SAFETY_FIRST: (
        "cost = length * mean over the two cells of (1 + w_risk*risk + w_slope*(slope/limit)^2 + w_unc*uncertainty)"
    ),
}


class SpeedModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_speed_mps: float = Field(default=0.8, gt=0, le=3.0)
    # Speed fraction at the slope limit; interpolated linearly from 1.0 on flat ground.
    min_speed_fraction: float = Field(default=0.4, gt=0, le=1.0)


class SafetyWeights(BaseModel):
    model_config = ConfigDict(frozen=True)

    risk_weight: float = Field(default=6.0, ge=0)
    slope_weight: float = Field(default=2.0, ge=0)
    uncertainty_weight: float = Field(default=2.0, ge=0)
    uncertainty_buffer_m: float = Field(default=40.0, ge=0)


class RoutingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str = "prototype-routing-v1"
    speed: SpeedModel = SpeedModel()
    safety: SafetyWeights = SafetyWeights()


DEFAULT_ROUTING_CONFIG = RoutingConfig()


def uncertainty_field(unknown: np.ndarray, resolution_m: float, buffer_m: float) -> np.ndarray:
    """1.0 on cells adjacent to unknown terrain, decreasing linearly to 1/K at K = ceil(buffer/cell) cells away."""
    result = np.zeros(unknown.shape, dtype=np.float32)
    steps = int(np.ceil(buffer_m / resolution_m)) if buffer_m > 0 else 0
    reached = unknown.copy()
    for step in range(1, steps + 1):
        padded = np.pad(reached, 1, constant_values=False)
        grown = np.zeros_like(reached)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                grown |= padded[1 + dr : 1 + dr + reached.shape[0], 1 + dc : 1 + dc + reached.shape[1]]
        new = grown & ~reached
        result[new] = 1.0 - (step - 1) / steps
        reached = grown
    return result


def effective_slope(slope_deg: np.ndarray, slope_limit_deg: float) -> np.ndarray:
    """Slope with unknown values treated as the limit, so unknown never looks gentler than it is."""
    return np.where(np.isfinite(slope_deg), np.minimum(slope_deg, slope_limit_deg), slope_limit_deg)


def cell_factors(grid: RoutingGrid, profile: RouteProfile, config: RoutingConfig) -> np.ndarray:
    """Per-cell cost multiplier f(c) (>= 1) for the profile."""
    if profile == RouteProfile.SHORTER_DISTANCE:
        return np.ones(grid.shape, dtype=np.float64)
    w = config.safety
    slope = effective_slope(grid.slope_deg, grid.slope_limit_deg) / grid.slope_limit_deg
    return (
        1.0
        + w.risk_weight * np.asarray(grid.risk, dtype=np.float64)
        + w.slope_weight * slope.astype(np.float64) ** 2
        + w.uncertainty_weight * np.asarray(grid.uncertainty, dtype=np.float64)
    )


def speed_mps(slope_deg: np.ndarray | float, slope_limit_deg: float, model: SpeedModel) -> np.ndarray | float:
    fraction = np.clip(np.asarray(slope_deg, dtype=np.float64) / slope_limit_deg, 0.0, 1.0)
    return model.base_speed_mps * (1.0 - (1.0 - model.min_speed_fraction) * fraction)
