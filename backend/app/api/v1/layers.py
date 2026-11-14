from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.errors import NotFoundError
from app.models.enums import DataOrigin, LayerType, RasterRole
from app.processing.errors import ProcessingError
from app.repositories import catalog
from app.schemas.catalog import LayerRead
from app.schemas.common import Page
from app.services.storage import AssetStorage, get_storage

router = APIRouter(prefix="/layers", tags=["layers"])
DbSession = Annotated[Session, Depends(get_db)]
Storage = Annotated[AssetStorage, Depends(get_storage)]


@router.get("", response_model=Page[LayerRead])
def list_layers(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
    site_id: UUID | None = None,
    layer_type: LayerType | None = None,
    data_origin: DataOrigin | None = None,
) -> Page[LayerRead]:
    items, total = catalog.list_layers(db, limit, offset, site_id, layer_type, data_origin)
    return Page[LayerRead](
        items=[LayerRead.model_validate(i) for i in items], total=total, limit=limit, offset=offset
    )


@router.get("/{layer_id}", response_model=LayerRead)
def get_layer(layer_id: UUID, db: DbSession) -> LayerRead:
    layer = catalog.get_layer(db, layer_id)
    if layer is None:
        raise NotFoundError("Data layer not found")
    return LayerRead.model_validate(layer)


@router.get("/{layer_id}/preview", response_class=FileResponse, responses={200: {"content": {"image/png": {}}}})
def get_layer_preview(layer_id: UUID, db: DbSession, storage: Storage) -> FileResponse:
    """Browser-ready PNG overlay of the layer's current dataset version (NoData is transparent)."""
    layer = catalog.get_layer(db, layer_id)
    if layer is None:
        raise NotFoundError("Data layer not found")
    asset = next(
        (
            a for a in layer.assets
            if a.role == RasterRole.DISPLAY_TILES and a.variable == "preview"
            and a.dataset_version == layer.dataset_version
        ),
        None,
    )
    if asset is None:
        raise NotFoundError("This layer has no preview image")
    try:
        path = storage.resolve(asset.storage_key)
    except ProcessingError:
        path = None
    if path is None or not path.is_file():
        raise NotFoundError("Preview image is not available")
    return FileResponse(
        path,
        media_type=asset.media_type,
        headers={"ETag": f'"{asset.sha256}"', "Cache-Control": "public, max-age=300"},
    )
