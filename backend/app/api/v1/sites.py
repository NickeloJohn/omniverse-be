from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.errors import NotFoundError
from app.repositories import catalog
from app.schemas.catalog import SiteRead
from app.schemas.common import Page

router = APIRouter(prefix="/sites", tags=["sites"])
DbSession = Annotated[Session, Depends(get_db)]


@router.get("", response_model=Page[SiteRead])
def list_sites(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[SiteRead]:
    items, total = catalog.list_sites(db, limit, offset)
    return Page[SiteRead](
        items=[SiteRead.model_validate(i) for i in items], total=total, limit=limit, offset=offset
    )


@router.get("/{site_id}", response_model=SiteRead)
def get_site(site_id: UUID, db: DbSession) -> SiteRead:
    site = catalog.get_site(db, site_id)
    if site is None:
        raise NotFoundError("Exploration site not found")
    return SiteRead.model_validate(site)
