"""Traversability grid used by the planner, built from processed terrain plus mission constraints."""

from dataclasses import dataclass

import numpy as np

from app.processing.grid import TargetGrid
from app.processing.hazard import CATEGORY_UNKNOWN


@dataclass(frozen=True)
class RoutingConstraints:
    max_slope_deg: float
    block_unknown_terrain: bool = True


@dataclass(frozen=True)
class RoutingGrid:
    grid: TargetGrid
    elevation: np.ndarray       # float32, NaN = unknown terrain
    slope_deg: np.ndarray       # float32, NaN = unknown slope
    risk: np.ndarray            # float32 in [0, 1]
    category: np.ndarray        # uint8 hazard categories
    blocked: np.ndarray         # bool, True = impassable
    uncertainty: np.ndarray     # float32 in [0, 1]
    slope_limit_deg: float      # slope limit actually applied

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    @property
    def resolution_m(self) -> float:
        return self.grid.resolution_m

    def cell_of(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        return self.grid.cell_of(x_m, y_m)

    def traversable(self, row: int, col: int) -> bool:
        return not bool(self.blocked[row, col])


def build_routing_grid(
    grid: TargetGrid,
    elevation: np.ndarray,
    slope_deg: np.ndarray,
    risk: np.ndarray,
    category: np.ndarray,
    blocked: np.ndarray,
    constraints: RoutingConstraints,
    layer_slope_limit_deg: float,
    layer_blocks_unknown: bool,
    uncertainty: np.ndarray,
) -> tuple[RoutingGrid, list[str]]:
    """Apply mission constraints on top of the processed hazard layer. Constraints can only tighten it.

    The stored hazard layer already excludes slopes above its own limit and (by default) unknown terrain, so a
    mission asking for a looser limit gets the layer's limit and a warning, never a silent relaxation.
    """
    warnings: list[str] = []
    slope_limit = min(constraints.max_slope_deg, layer_slope_limit_deg)
    if constraints.max_slope_deg > layer_slope_limit_deg:
        warnings.append(
            f"Mission max_slope_deg {constraints.max_slope_deg:g} exceeds the processed hazard layer's exclusion "
            f"limit {layer_slope_limit_deg:g}; the stricter limit was applied"
        )
    effective_blocked = blocked | (np.isfinite(slope_deg) & (slope_deg > slope_limit))

    block_unknown = constraints.block_unknown_terrain or layer_blocks_unknown
    if layer_blocks_unknown and not constraints.block_unknown_terrain:
        warnings.append(
            "Mission allows unknown terrain but the processed hazard layer blocks it; unknown terrain stays blocked"
        )
    if block_unknown:
        effective_blocked = effective_blocked | (category == CATEGORY_UNKNOWN)

    return (
        RoutingGrid(grid, elevation, slope_deg, risk, category, effective_blocked, uncertainty, float(slope_limit)),
        warnings,
    )
