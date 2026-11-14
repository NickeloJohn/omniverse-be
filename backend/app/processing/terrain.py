"""Terrain derivatives computed on the aligned elevation grid."""

import numpy as np

NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _padded(array: np.ndarray) -> np.ndarray:
    return np.pad(array, 1, mode="constant", constant_values=np.nan)


def compute_slope_deg(elevation: np.ndarray, cell_x_m: float, cell_y_m: float) -> np.ndarray:
    """Slope in degrees using Horn's 3x3 method.

    A cell is valid only if it and all 8 neighbours are valid; border cells are therefore invalid.
    Unknown terrain is never filled, so NoData also removes a one-cell margin around it.
    """
    z = _padded(np.asarray(elevation, dtype=np.float64))
    rows, cols = elevation.shape

    def at(dr: int, dc: int) -> np.ndarray:
        return z[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]

    a, b, c = at(-1, -1), at(-1, 0), at(-1, 1)
    d, e, f = at(0, -1), at(0, 0), at(0, 1)
    g, h, i = at(1, -1), at(1, 0), at(1, 1)
    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cell_x_m)
    dz_dy = ((a + 2 * b + c) - (g + 2 * h + i)) / (8 * cell_y_m)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    valid = np.isfinite(np.stack([a, b, c, d, e, f, g, h, i])).all(axis=0)
    return np.where(valid, slope, np.nan).astype(np.float32)


def max_neighbor_step_m(elevation: np.ndarray, row: int, col: int) -> float | None:
    """Largest absolute elevation difference between a cell and its valid 8-neighbours, or None."""
    centre = elevation[row, col]
    if not np.isfinite(centre):
        return None
    steps = []
    for dr, dc in NEIGHBOR_OFFSETS:
        r, c = row + dr, col + dc
        if 0 <= r < elevation.shape[0] and 0 <= c < elevation.shape[1] and np.isfinite(elevation[r, c]):
            steps.append(abs(float(elevation[r, c] - centre)))
    return max(steps) if steps else None
