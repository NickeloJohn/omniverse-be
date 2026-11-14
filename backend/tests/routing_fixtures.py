"""Hand-built routing grids for engine tests. ASCII maps: '#' blocked, '.' open, '~' high risk, '?' unknown terrain."""

import numpy as np

from app.processing.hazard import CATEGORY_HIGH, CATEGORY_LOW, CATEGORY_UNKNOWN
from app.routing.cost_model import uncertainty_field
from app.routing.grid import RoutingGrid
from app.models.enums import WaypointType
from app.routing.planner import Stop
from tests.raster_fixtures import local_crs
from app.processing.grid import TargetGrid


def make_routing_grid(
    ascii_map: list[str],
    res: float = 10.0,
    high_risk: float = 0.8,
    slope: np.ndarray | float = 0.0,
    slope_limit: float = 15.0,
    buffer_m: float = 0.0,
) -> RoutingGrid:
    rows, cols = len(ascii_map), len(ascii_map[0])
    cells = np.array([list(line) for line in ascii_map])
    grid = TargetGrid.build(cols * res, rows * res, res, local_crs(), 10**7)
    unknown = cells == "?"
    blocked = (cells == "#") | unknown
    risk = np.where(cells == "~", high_risk, 0.0).astype(np.float32)
    risk[blocked] = 1.0
    category = np.full((rows, cols), CATEGORY_LOW, np.uint8)
    category[cells == "~"] = CATEGORY_HIGH
    category[unknown] = CATEGORY_UNKNOWN
    elevation = np.zeros((rows, cols), np.float32)
    elevation[unknown] = np.nan
    slope_arr = np.broadcast_to(np.asarray(slope, np.float32), (rows, cols)).copy()
    return RoutingGrid(
        grid, elevation, slope_arr, risk, category, blocked,
        uncertainty_field(unknown, res, buffer_m), slope_limit,
    )


def stop_at(grid: RoutingGrid, row: int, col: int, label: str = "stop", **kwargs) -> Stop:
    x, y = grid.grid.cell_center(row, col)
    kind = kwargs.pop("kind", WaypointType.WAYPOINT)
    return Stop(kind, label, x, y, **kwargs)


def dijkstra_oracle(grid: RoutingGrid, factors: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> float:
    """Independent reference implementation: dense relaxation, no heap, no heuristic."""
    rows, cols = grid.shape
    res = grid.resolution_m
    dist = np.full((rows, cols), np.inf)
    dist[start] = 0.0
    done = np.zeros((rows, cols), bool)
    while True:
        masked = np.where(done, np.inf, dist)
        index = np.unravel_index(np.argmin(masked), masked.shape)
        if not np.isfinite(masked[index]):
            return float(dist[goal])
        done[index] = True
        r, c = index
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or grid.blocked[nr, nc]:
                    continue
                if dr and dc and (grid.blocked[r, nc] or grid.blocked[nr, c]):
                    continue
                step = res * (2 ** 0.5 if dr and dc else 1.0) * (factors[r, c] + factors[nr, nc]) / 2
                dist[nr, nc] = min(dist[nr, nc], dist[r, c] + step)
