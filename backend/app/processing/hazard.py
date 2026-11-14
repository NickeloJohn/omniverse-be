"""Terrain hazard classification and risk scoring.

DISCLAIMER: slope categories are configurable PROTOTYPE values, not validated NASA astronaut safety limits.
Orbital elevation data cannot resolve every astronaut-scale obstacle (rocks, small pits, soft soil), so a
low risk score means "low risk at orbital resolution", never "safe".
"""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np
from affine import Affine
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rasterio.features import rasterize
from shapely.geometry.base import BaseGeometry

CATEGORY_LOW, CATEGORY_MODERATE, CATEGORY_HIGH, CATEGORY_EXCLUDED = 0, 1, 2, 3
CATEGORY_UNKNOWN = 255
CATEGORY_NAMES = {
    CATEGORY_LOW: "low",
    CATEGORY_MODERATE: "moderate",
    CATEGORY_HIGH: "high",
    CATEGORY_EXCLUDED: "excluded",
    CATEGORY_UNKNOWN: "unknown",
}
DISCLAIMERS = [
    "Slope categories are prototype configuration values, not validated NASA astronaut safety limits.",
    "Orbital-resolution terrain cannot identify every astronaut-scale obstacle; low risk does not mean safe.",
]


class HazardConfig(BaseModel):
    """Low [0, l0) / Moderate [l0, l1) / High [l1, l2] / Excluded (> l2), in degrees of slope."""

    model_config = ConfigDict(frozen=True)

    version: str = "prototype-slope-v1"
    slope_limits_deg: tuple[float, float, float] = (5.0, 10.0, 15.0)
    # Risk reached at each limit; risk is 0 at flat terrain and interpolates linearly between anchors.
    risk_at_limits: tuple[float, float, float] = (0.25, 0.5, 0.75)
    block_unknown: bool = True
    # Used only when block_unknown is False.
    unknown_terrain_risk: float = Field(default=0.8, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_monotonic(self) -> "HazardConfig":
        limits, risks = self.slope_limits_deg, self.risk_at_limits
        if not (0 < limits[0] < limits[1] < limits[2] <= 89):
            raise ValueError("slope_limits_deg must be strictly increasing within (0, 89]")
        if not (0 < risks[0] < risks[1] < risks[2] < 1):
            raise ValueError("risk_at_limits must be strictly increasing within (0, 1)")
        return self

    @property
    def excluded_above_deg(self) -> float:
        return self.slope_limits_deg[2]


DEFAULT_HAZARD_CONFIG = HazardConfig()


@dataclass
class TerrainHazard:
    risk: np.ndarray  # float32 in [0, 1]
    category: np.ndarray  # uint8, see CATEGORY_*
    blocked: np.ndarray  # bool


@dataclass(frozen=True)
class HazardZoneSpec:
    id: UUID
    name: str
    zone_type: str
    geometry: BaseGeometry
    risk_score: float
    is_blocking: bool
    data_origin: str


@dataclass
class CostGrid:
    risk: np.ndarray
    category: np.ndarray
    blocked: np.ndarray
    zone_masks: dict[UUID, np.ndarray] = field(default_factory=dict)


def classify_terrain(slope_deg: np.ndarray, config: HazardConfig = DEFAULT_HAZARD_CONFIG) -> TerrainHazard:
    """Slope -> risk score, category and blocked mask. Unknown (NaN) cells are blocked by default."""
    slope = np.asarray(slope_deg, dtype=np.float64)
    unknown = ~np.isfinite(slope)
    s = np.where(unknown, 0.0, slope)
    l0, l1, l2 = config.slope_limits_deg

    category = np.full(slope.shape, CATEGORY_LOW, dtype=np.uint8)
    category[s >= l0] = CATEGORY_MODERATE
    category[s >= l1] = CATEGORY_HIGH
    category[s > l2] = CATEGORY_EXCLUDED
    category[unknown] = CATEGORY_UNKNOWN

    risk = np.interp(s, [0.0, l0, l1, l2], [0.0, *config.risk_at_limits])
    excluded = category == CATEGORY_EXCLUDED
    risk[excluded] = 1.0
    risk[unknown] = 1.0 if config.block_unknown else config.unknown_terrain_risk
    blocked = excluded | (unknown & config.block_unknown)
    return TerrainHazard(risk.astype(np.float32), category, blocked)


def apply_hazard_zones(
    terrain: TerrainHazard, zones: list[HazardZoneSpec], transform: Affine
) -> CostGrid:
    """Overlay user-defined zones. A zone affects every cell it touches (conservative).

    Blocking zones block and set risk 1. Other zones raise risk to at least their risk_score.
    """
    risk = terrain.risk.copy()
    blocked = terrain.blocked.copy()
    masks: dict[UUID, np.ndarray] = {}
    for zone in zones:
        mask = rasterize(
            [(zone.geometry, 1)],
            out_shape=risk.shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        ).astype(bool)
        masks[zone.id] = mask
        if zone.is_blocking:
            blocked |= mask
            risk[mask] = 1.0
        else:
            risk[mask] = np.maximum(risk[mask], np.float32(zone.risk_score))
    return CostGrid(risk, terrain.category, blocked, masks)


def category_fractions(category: np.ndarray) -> dict[str, float]:
    total = category.size
    return {
        name: round(float((category == code).sum()) / total, 6) for code, name in CATEGORY_NAMES.items()
    }


def array_stats(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    base: dict[str, Any] = {"valid_fraction": round(finite.size / values.size, 6)}
    if finite.size:
        base |= {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "p95": float(np.percentile(finite, 95)),
        }
    return base
