"""Composition of terrain-derived risk with user-defined hazard zones."""

from collections import OrderedDict
from typing import TYPE_CHECKING
import threading

from sqlalchemy.orm import Session

from app.processing.hazard import CostGrid, apply_hazard_zones
from app.repositories import hazards as hazard_repo

if TYPE_CHECKING:
    from app.services.terrain_service import TerrainData

_CACHE: "OrderedDict[tuple, CostGrid]" = OrderedDict()
_CACHE_SIZE = 8
_LOCK = threading.Lock()


def get_cost_grid(session: Session, terrain: "TerrainData") -> CostGrid:
    """Terrain risk overlaid with the site's current hazard zones; cached until a zone changes."""
    rows = hazard_repo.zones_for_site(session, terrain.site_id)
    key = (terrain.key, tuple((z.id, z.updated_at) for z in rows))
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    grid = apply_hazard_zones(terrain.terrain, [hazard_repo.to_spec(z) for z in rows], terrain.grid.transform)
    with _LOCK:
        _CACHE[key] = grid
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return grid


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
