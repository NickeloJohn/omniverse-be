from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.core.errors import DomainValidationError, NotFoundError
from app.models import ScienceObservation, ScienceTarget
from app.models.enums import (
    CreationSource,
    DataOrigin,
    ObservationSource,
    VerificationStatus,
)
from app.repositories import catalog, missions
from app.schemas.science import ObservationCreate, ScienceTargetCreate
from app.services.geo import ensure_within_site, geographic_for, point_geometry, point_xy


def create_science_target(session: Session, payload: ScienceTargetCreate) -> ScienceTarget:
    site = catalog.get_site(session, payload.site_id)
    if site is None:
        raise NotFoundError("Exploration site not found")
    ensure_within_site(site, payload.location.x_m, payload.location.y_m, "location")
    lon, lat = geographic_for(site, payload.location.x_m, payload.location.y_m)
    target = ScienceTarget(
        site_id=site.id,
        name=payload.name,
        description=payload.description,
        scientific_objective=payload.scientific_objective,
        location=point_geometry(payload.location.x_m, payload.location.y_m),
        longitude_deg_east=lon,
        latitude_deg=lat,
        associated_mission=payload.associated_mission,
        associated_product_id=payload.associated_product_id,
        geological_context=payload.geological_context,
        estimated_activity_duration_min=payload.estimated_activity_duration_min,
        creation_source=CreationSource.MANUAL,
        verification_status=VerificationStatus.PROPOSED,
    )
    session.add(target)
    session.commit()
    return target


def create_observation(session: Session, payload: ObservationCreate) -> ScienceObservation:
    site = catalog.get_site(session, payload.site_id)
    if site is None:
        raise NotFoundError("Exploration site not found")

    target = None
    if payload.target_id is not None:
        target = missions.get_science_target(session, payload.target_id)
        if target is None:
            raise NotFoundError("Science target not found")
        if target.site_id != site.id:
            raise DomainValidationError(
                "Science target belongs to a different site",
                [{"field": "target_id", "message": "target is not part of site_id"}],
            )
    if payload.mission_id is not None:
        mission = missions.get_mission(session, payload.mission_id)
        if mission is None:
            raise NotFoundError("Mission not found")
        if mission.site_id != site.id:
            raise DomainValidationError(
                "Mission belongs to a different site",
                [{"field": "mission_id", "message": "mission is not part of site_id"}],
            )

    observed_at = payload.observed_at or datetime.now(UTC)
    if observed_at > datetime.now(UTC) + timedelta(minutes=1):
        raise DomainValidationError(
            "observed_at cannot be in the future",
            [{"field": "observed_at", "message": "must not be in the future"}],
        )

    if payload.location is not None:
        x_m, y_m = payload.location.x_m, payload.location.y_m
        ensure_within_site(site, x_m, y_m, "location")
    elif target is not None:
        x_m, y_m = point_xy(target.location)
    else:
        x_m = y_m = None

    lon = lat = None
    location = None
    if x_m is not None and y_m is not None:
        lon, lat = geographic_for(site, x_m, y_m)
        location = point_geometry(x_m, y_m)

    observation = ScienceObservation(
        site_id=site.id,
        target_id=payload.target_id,
        mission_id=payload.mission_id,
        observation_type=payload.observation_type,
        title=payload.title,
        notes=payload.notes,
        observed_at=observed_at,
        location=location,
        longitude_deg_east=lon,
        latitude_deg=lat,
        measurements=payload.measurements,
        source=ObservationSource.MANUAL,
        data_origin=DataOrigin.SIMULATED,
    )
    session.add(observation)
    session.commit()
    return observation
