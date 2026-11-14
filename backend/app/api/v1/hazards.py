from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.enums import DataOrigin, HazardZoneType
from app.repositories import hazards as repo
from app.schemas.common import Page
from app.schemas.terrain import HazardZoneRead

router = APIRouter(prefix="/hazards", tags=["hazards"])
DbSession = Annotated[Session, Depends(get_db)]


@router.get("", response_model=Page[HazardZoneRead])
def list_hazards(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
    site_id: UUID | None = None,
    zone_type: HazardZoneType | None = None,
    is_blocking: bool | None = None,
    data_origin: DataOrigin | None = None,
) -> Page[HazardZoneRead]:
    items, total = repo.list_hazard_zones(db, limit, offset, site_id, zone_type, is_blocking, data_origin)
    return Page[HazardZoneRead](
        items=[HazardZoneRead.model_validate(i) for i in items], total=total, limit=limit, offset=offset
    )
