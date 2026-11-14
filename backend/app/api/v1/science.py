from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.errors import NotFoundError
from app.models.enums import CreationSource, VerificationStatus
from app.repositories import missions as repo
from app.schemas.common import Page
from app.schemas.science import (
    ObservationCreate,
    ObservationRead,
    ScienceTargetCreate,
    ScienceTargetRead,
)
from app.services import science_service

router = APIRouter(tags=["science"])
DbSession = Annotated[Session, Depends(get_db)]
Limit = Annotated[int, Query(ge=1, le=100)]
Offset = Annotated[int, Query(ge=0)]


@router.get("/science-targets", response_model=Page[ScienceTargetRead])
def list_science_targets(
    db: DbSession,
    limit: Limit = 25,
    offset: Offset = 0,
    site_id: UUID | None = None,
    verification_status: VerificationStatus | None = None,
    creation_source: CreationSource | None = None,
    mission_id: UUID | None = None,
) -> Page[ScienceTargetRead]:
    items, total = repo.list_science_targets(
        db, limit, offset, site_id, verification_status, creation_source, mission_id
    )
    return Page[ScienceTargetRead](
        items=[ScienceTargetRead.model_validate(i) for i in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/science-targets/{target_id}", response_model=ScienceTargetRead)
def get_science_target(target_id: UUID, db: DbSession) -> ScienceTargetRead:
    target = repo.get_science_target(db, target_id)
    if target is None:
        raise NotFoundError("Science target not found")
    return ScienceTargetRead.model_validate(target)


@router.post("/science-targets", response_model=ScienceTargetRead, status_code=status.HTTP_201_CREATED)
def create_science_target(payload: ScienceTargetCreate, db: DbSession) -> ScienceTargetRead:
    return ScienceTargetRead.model_validate(science_service.create_science_target(db, payload))


@router.post("/observations", response_model=ObservationRead, status_code=status.HTTP_201_CREATED)
def create_observation(payload: ObservationCreate, db: DbSession) -> ObservationRead:
    return ObservationRead.model_validate(science_service.create_observation(db, payload))


@router.get("/observations", response_model=Page[ObservationRead])
def list_observations(
    db: DbSession,
    limit: Limit = 25,
    offset: Offset = 0,
    site_id: UUID | None = None,
    target_id: UUID | None = None,
    mission_id: UUID | None = None,
) -> Page[ObservationRead]:
    items, total = repo.list_observations(db, limit, offset, site_id, target_id, mission_id)
    return Page[ObservationRead](
        items=[ObservationRead.model_validate(i) for i in items],
        total=total,
        limit=limit,
        offset=offset,
    )
