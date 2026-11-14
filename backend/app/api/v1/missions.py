from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.enums import MissionStatus
from app.repositories import missions as repo
from app.schemas.common import Page
from app.schemas.mission import MissionCreate, MissionRead, MissionUpdate
from app.services import mission_service

router = APIRouter(prefix="/missions", tags=["missions"])
DbSession = Annotated[Session, Depends(get_db)]


@router.post("", response_model=MissionRead, status_code=status.HTTP_201_CREATED)
def create_mission(payload: MissionCreate, db: DbSession) -> MissionRead:
    return MissionRead.model_validate(mission_service.create_mission(db, payload))


@router.get("", response_model=Page[MissionRead])
def list_missions(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: Annotated[MissionStatus | None, Query(alias="status")] = None,
    site_id: UUID | None = None,
) -> Page[MissionRead]:
    items, total = repo.list_missions(db, limit, offset, status_filter, site_id)
    return Page[MissionRead](
        items=[MissionRead.model_validate(i) for i in items], total=total, limit=limit, offset=offset
    )


@router.get("/{mission_id}", response_model=MissionRead)
def get_mission(mission_id: UUID, db: DbSession) -> MissionRead:
    return MissionRead.model_validate(mission_service.get_mission_or_404(db, mission_id))


@router.patch("/{mission_id}", response_model=MissionRead)
def update_mission(mission_id: UUID, payload: MissionUpdate, db: DbSession) -> MissionRead:
    return MissionRead.model_validate(mission_service.update_mission(db, mission_id, payload))


@router.delete("/{mission_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_mission(mission_id: UUID, db: DbSession) -> Response:
    mission_service.delete_mission(db, mission_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
