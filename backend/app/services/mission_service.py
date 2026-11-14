from uuid import UUID

from sqlalchemy.orm import Session

from app.core.errors import ConflictError, DomainValidationError, NotFoundError
from app.models import Mission, MissionTarget
from app.models.enums import MissionStatus
from app.repositories import catalog, missions
from app.schemas.mission import MissionCreate, MissionUpdate
from app.services.geo import ensure_within_site, geographic_for, point_geometry, point_xy

TERMINAL_STATUSES = {MissionStatus.COMPLETED, MissionStatus.ABORTED}
ALLOWED_TRANSITIONS: dict[MissionStatus, set[MissionStatus]] = {
    MissionStatus.DRAFT: {MissionStatus.PLANNED, MissionStatus.ABORTED},
    MissionStatus.PLANNED: {MissionStatus.DRAFT, MissionStatus.ACTIVE, MissionStatus.ABORTED},
    MissionStatus.ACTIVE: {MissionStatus.PAUSED, MissionStatus.COMPLETED, MissionStatus.ABORTED},
    MissionStatus.PAUSED: {MissionStatus.ACTIVE, MissionStatus.COMPLETED, MissionStatus.ABORTED},
    MissionStatus.COMPLETED: set(),
    MissionStatus.ABORTED: set(),
}
# Fields that define the plan; frozen once an EVA is underway.
PLAN_FIELDS = {
    "start_location",
    "eva_duration_limit_min",
    "return_reserve_min",
    "constraints",
    "science_target_ids",
}


def _build_links(session: Session, site_id: UUID, target_ids: list[UUID]) -> list[MissionTarget]:
    found = missions.get_targets_by_ids(session, target_ids)
    missing = [str(t) for t in target_ids if t not in found]
    if missing:
        raise NotFoundError(
            "Science target not found",
            [{"field": "science_target_ids", "message": f"unknown target id(s): {', '.join(missing)}"}],
        )
    other_site = [str(t) for t in target_ids if found[t].site_id != site_id]
    if other_site:
        raise DomainValidationError(
            "Science targets must belong to the mission's site",
            [{"field": "science_target_ids", "message": f"different site: {', '.join(other_site)}"}],
        )
    return [
        MissionTarget(target_id=tid, sequence=index, target=found[tid])
        for index, tid in enumerate(target_ids)
    ]


def get_mission_or_404(session: Session, mission_id: UUID) -> Mission:
    mission = missions.get_mission(session, mission_id)
    if mission is None:
        raise NotFoundError("Mission not found")
    return mission


def create_mission(session: Session, payload: MissionCreate) -> Mission:
    site = catalog.get_site(session, payload.site_id)
    if site is None:
        raise NotFoundError("Exploration site not found")

    if payload.start_location is not None:
        start_x, start_y = payload.start_location.x_m, payload.start_location.y_m
        ensure_within_site(site, start_x, start_y, "start_location")
    else:
        start_x, start_y = point_xy(site.habitat_location)
    lon, lat = geographic_for(site, start_x, start_y)

    mission = Mission(
        name=payload.name,
        description=payload.description,
        site_id=site.id,
        status=MissionStatus.DRAFT,
        start_location=point_geometry(start_x, start_y),
        start_longitude_deg_east=lon,
        start_latitude_deg=lat,
        eva_duration_limit_min=payload.eva_duration_limit_min,
        return_reserve_min=payload.return_reserve_min,
        constraints=payload.constraints.model_dump(mode="json"),
        is_demo=False,
    )
    mission.target_links = _build_links(session, site.id, payload.science_target_ids)
    session.add(mission)
    session.commit()
    session.refresh(mission)
    return mission


def update_mission(session: Session, mission_id: UUID, payload: MissionUpdate) -> Mission:
    mission = get_mission_or_404(session, mission_id)
    if mission.status in TERMINAL_STATUSES:
        raise ConflictError(f"A {mission.status.value} mission cannot be modified")

    changes = payload.model_dump(exclude_unset=True)
    plan_changes = PLAN_FIELDS & changes.keys()
    if plan_changes and mission.status not in {MissionStatus.DRAFT, MissionStatus.PLANNED}:
        raise ConflictError(
            f"Plan fields ({', '.join(sorted(plan_changes))}) can only change while the mission "
            "is draft or planned"
        )

    new_status = payload.status
    if new_status is not None and new_status != mission.status:
        if new_status not in ALLOWED_TRANSITIONS[mission.status]:
            allowed = ", ".join(sorted(s.value for s in ALLOWED_TRANSITIONS[mission.status]))
            raise ConflictError(
                f"Cannot change status from {mission.status.value} to {new_status.value}; "
                f"allowed: {allowed}"
            )
        mission.status = new_status

    limit = payload.eva_duration_limit_min or mission.eva_duration_limit_min
    reserve = (
        payload.return_reserve_min
        if payload.return_reserve_min is not None
        else mission.return_reserve_min
    )
    if reserve >= limit:
        raise DomainValidationError(
            "return_reserve_min must be less than eva_duration_limit_min",
            [{"field": "return_reserve_min", "message": f"{reserve} >= {limit}"}],
        )
    mission.eva_duration_limit_min = limit
    mission.return_reserve_min = reserve

    if "name" in changes:
        mission.name = payload.name  # type: ignore[assignment]
    if "description" in changes:
        mission.description = payload.description
    if payload.constraints is not None:
        mission.constraints = payload.constraints.model_dump(mode="json")

    site = catalog.get_site(session, mission.site_id)
    if payload.start_location is not None and site is not None:
        x_m, y_m = payload.start_location.x_m, payload.start_location.y_m
        ensure_within_site(site, x_m, y_m, "start_location")
        mission.start_location = point_geometry(x_m, y_m)
        mission.start_longitude_deg_east, mission.start_latitude_deg = geographic_for(site, x_m, y_m)

    if payload.science_target_ids is not None:
        links = _build_links(session, mission.site_id, payload.science_target_ids)
        mission.target_links.clear()
        session.flush()
        mission.target_links.extend(links)

    session.commit()
    session.refresh(mission)
    return mission


def delete_mission(session: Session, mission_id: UUID) -> None:
    mission = get_mission_or_404(session, mission_id)
    if mission.status not in {MissionStatus.DRAFT, MissionStatus.ABORTED}:
        raise ConflictError("Only draft or aborted missions can be deleted")
    session.delete(mission)
    session.commit()
