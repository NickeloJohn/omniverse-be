"""A* search over the 8-connected routing grid.

Heuristic: octile distance * cell size. Because every cell factor is >= 1 (see cost_model.py) a step never costs
less than its geometric length, so this never overestimates (admissible) and satisfies
h(a) <= cost(a, b) + h(b) (consistent). With a consistent heuristic the first time the goal is popped its cost is
optimal and closed nodes never need reopening. Set use_heuristic=False for plain Dijkstra (used as a test oracle).

Diagonal moves are refused unless both orthogonal neighbours are traversable, so a path can never squeeze through
the gap between two diagonally touching blocked cells.
"""

import heapq
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from app.routing.grid import RoutingGrid

SQRT2 = math.sqrt(2.0)
_CHECK_MASK = 255
MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


class SearchStatus(StrEnum):
    FOUND = "found"
    START_BLOCKED = "start_blocked"
    GOAL_BLOCKED = "goal_blocked"
    UNREACHABLE = "unreachable"
    LIMIT_EXCEEDED = "search_limit_exceeded"
    TIMEOUT = "search_timeout"


@dataclass(frozen=True)
class SearchLimits:
    max_expansions: int = 2_000_000
    timeout_s: float = 15.0


@dataclass
class SearchResult:
    status: SearchStatus
    cells: list[tuple[int, int]]
    cost: float
    expanded: int


class SearchSpace:
    """Flat Python lists of the grid, built once per profile because list access is much faster than numpy scalars."""

    def __init__(self, grid: RoutingGrid, factors: np.ndarray) -> None:
        self.height, self.width = grid.shape
        self.resolution_m = grid.resolution_m
        self.blocked: list[bool] = grid.blocked.ravel().tolist()
        self.factors: list[float] = np.asarray(factors, dtype=np.float64).ravel().tolist()


def octile(dr: int, dc: int, resolution_m: float) -> float:
    dr, dc = abs(dr), abs(dc)
    return resolution_m * (max(dr, dc) + (SQRT2 - 1.0) * min(dr, dc))


def astar(
    space: SearchSpace,
    start: tuple[int, int],
    goal: tuple[int, int],
    limits: SearchLimits = SearchLimits(),
    use_heuristic: bool = True,
    clock: Callable[[], float] = time.monotonic,
) -> SearchResult:
    width, height, res = space.width, space.height, space.resolution_m
    blocked, factors = space.blocked, space.factors
    start_i, goal_i = start[0] * width + start[1], goal[0] * width + goal[1]
    if blocked[start_i]:
        return SearchResult(SearchStatus.START_BLOCKED, [], math.inf, 0)
    if blocked[goal_i]:
        return SearchResult(SearchStatus.GOAL_BLOCKED, [], math.inf, 0)

    goal_r, goal_c = goal
    h = (lambda r, c: octile(r - goal_r, c - goal_c, res)) if use_heuristic else (lambda r, c: 0.0)
    deadline = clock() + limits.timeout_s
    best: dict[int, float] = {start_i: 0.0}
    parent: dict[int, int] = {}
    closed: set[int] = set()
    counter = 0
    heap: list[tuple[float, int, int]] = [(h(*start), counter, start_i)]
    expanded = 0

    while heap:
        _, _, node = heapq.heappop(heap)
        if node in closed:
            continue
        if node == goal_i:
            cells, cursor = [], node
            while True:
                cells.append(divmod(cursor, width))
                if cursor == start_i:
                    break
                cursor = parent[cursor]
            cells.reverse()
            return SearchResult(SearchStatus.FOUND, cells, best[node], expanded)
        closed.add(node)
        expanded += 1
        if expanded & _CHECK_MASK == 1:
            if clock() > deadline:
                return SearchResult(SearchStatus.TIMEOUT, [], math.inf, expanded)
        if expanded > limits.max_expansions:
            return SearchResult(SearchStatus.LIMIT_EXCEEDED, [], math.inf, expanded)

        row, col = divmod(node, width)
        g_node, f_node = best[node], factors[node]
        for dr, dc, length in MOVES:
            nr, nc = row + dr, col + dc
            if nr < 0 or nc < 0 or nr >= height or nc >= width:
                continue
            nxt = nr * width + nc
            if blocked[nxt] or nxt in closed:
                continue
            if dr and dc and (blocked[row * width + nc] or blocked[nr * width + col]):
                continue
            g_new = g_node + res * length * 0.5 * (f_node + factors[nxt])
            if g_new < best.get(nxt, math.inf):
                best[nxt] = g_new
                parent[nxt] = node
                counter += 1
                heapq.heappush(heap, (g_new + h(nr, nc), counter, nxt))
    return SearchResult(SearchStatus.UNREACHABLE, [], math.inf, expanded)


def validate_cell_path(grid: RoutingGrid, cells: list[tuple[int, int]]) -> None:
    """Defensive check that a returned path is contiguous, traversable and never cuts a blocked corner."""
    if not cells:
        raise ValueError("empty path")
    height, width = grid.shape
    for index, (row, col) in enumerate(cells):
        if not (0 <= row < height and 0 <= col < width) or grid.blocked[row, col]:
            raise ValueError(f"path cell {index} {(row, col)} is outside the grid or blocked")
        if index == 0:
            continue
        prev_row, prev_col = cells[index - 1]
        dr, dc = row - prev_row, col - prev_col
        if max(abs(dr), abs(dc)) != 1:
            raise ValueError(f"path is not contiguous at step {index}")
        if dr and dc and (grid.blocked[prev_row, col] or grid.blocked[row, prev_col]):
            raise ValueError(f"path cuts a blocked corner at step {index}")
