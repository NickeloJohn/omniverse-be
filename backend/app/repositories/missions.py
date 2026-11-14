from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Mission, MissionTarget, ScienceObservation, ScienceTarget
from app.models.enums import CreationSource, MissionStatus, VerificationStatus
from app.repositories.pagination import paginate


def list_missions(
    session: Session,
    limit: int,
    offset: int,
    status: MissionStatus | None = None,
    site_id: UUID | None = None,
) -> tuple[list[Mission], int]:
    stmt = select(Mission)
    if status is not None:
        stmt = stmt.where(Mission.status == status)
    if site_id is not None:
        stmt = stmt.where(Mission.site_id == site_id)
    return paginate(session, stmt.order_by(Mission.created_at.desc(), Mission.id), limit, offset)


def get_mission(session: Session, mission_id: UUID) -> Mission | None:
    return session.get(Mission, mission_id)


def get_targets_by_ids(session: Session, ids: list[UUID]) -> dict[UUID, ScienceTarget]:
    if not ids:
        return {}
    rows = session.scalars(select(ScienceTarget).where(ScienceTarget.id.in_(ids)))
    return {t.id: t for t in rows}


def list_science_targets(
    session: Session,
    limit: int,
    offset: int,
    site_id: UUID | None = None,
    verification_status: VerificationStatus | None = None,
    creation_source: CreationSource | None = None,
    mission_id: UUID | None = None,
) -> tuple[list[ScienceTarget], int]:
    stmt = select(ScienceTarget)
    if site_id is not None:
        stmt = stmt.where(ScienceTarget.site_id == site_id)
    if verification_status is not None:
        stmt = stmt.where(ScienceTarget.verification_status == verification_status)
    if creation_source is not None:
        stmt = stmt.where(ScienceTarget.creation_source == creation_source)
    if mission_id is not None:
        stmt = stmt.where(
            ScienceTarget.id.in_(
                select(MissionTarget.target_id).where(MissionTarget.mission_id == mission_id)
            )
        )
    return paginate(session, stmt.order_by(ScienceTarget.name, ScienceTarget.id), limit, offset)


def get_science_target(session: Session, target_id: UUID) -> ScienceTarget | None:
    return session.get(ScienceTarget, target_id)


def list_observations(
    session: Session,
    limit: int,
    offset: int,
    site_id: UUID | None = None,
    target_id: UUID | None = None,
    mission_id: UUID | None = None,
) -> tuple[list[ScienceObservation], int]:
    stmt = select(ScienceObservation)
    if site_id is not None:
        stmt = stmt.where(ScienceObservation.site_id == site_id)
    if target_id is not None:
        stmt = stmt.where(ScienceObservation.target_id == target_id)
    if mission_id is not None:
        stmt = stmt.where(ScienceObservation.mission_id == mission_id)
    stmt = stmt.order_by(ScienceObservation.observed_at.desc(), ScienceObservation.id)
    return paginate(session, stmt, limit, offset)
