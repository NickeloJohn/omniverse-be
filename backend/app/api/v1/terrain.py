from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.errors import NotFoundError
from app.repositories import catalog
from app.schemas.terrain import TerrainMetadataRead, TerrainSampleRead
from app.services import terrain_service
from app.services.storage import AssetStorage, get_storage

router = APIRouter(prefix="/terrain", tags=["terrain"])
DbSession = Annotated[Session, Depends(get_db)]
Storage = Annotated[AssetStorage, Depends(get_storage)]
Coordinate = Annotated[float, Query(allow_inf_nan=False, description="Metres in the site-local frame")]


def _site_or_404(db: Session, site_id: UUID):
    site = catalog.get_site(db, site_id)
    if site is None:
        raise NotFoundError("Exploration site not found")
    return site


@router.get("/sample", response_model=TerrainSampleRead)
def sample_terrain(
    db: DbSession,
    storage: Storage,
    site_id: UUID,
    x_m: Coordinate,
    y_m: Coordinate,
    elevation_layer_id: UUID | None = None,
) -> TerrainSampleRead:
    site = _site_or_404(db, site_id)
    return TerrainSampleRead.model_validate(
        terrain_service.sample_point(db, storage, site, x_m, y_m, elevation_layer_id)
    )


@router.get("/metadata", response_model=TerrainMetadataRead)
def terrain_metadata(db: DbSession, site_id: UUID, elevation_layer_id: UUID | None = None) -> TerrainMetadataRead:
    site = _site_or_404(db, site_id)
    return TerrainMetadataRead.model_validate(terrain_service.terrain_metadata(db, site, elevation_layer_id))
