from uuid import UUID

from geoalchemy2.shape import to_shape
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import HazardZone
from app.models.enums import DataOrigin, HazardZoneType
from app.processing.hazard import HazardZoneSpec
from app.repositories.pagination import paginate


def list_hazard_zones(
    session: Session,
    limit: int,
    offset: int,
    site_id: UUID | None = None,
    zone_type: HazardZoneType | None = None,
    is_blocking: bool | None = None,
    data_origin: DataOrigin | None = None,
) -> tuple[list[HazardZone], int]:
    stmt = select(HazardZone)
    if site_id is not None:
        stmt = stmt.where(HazardZone.site_id == site_id)
    if zone_type is not None:
        stmt = stmt.where(HazardZone.zone_type == zone_type)
    if is_blocking is not None:
        stmt = stmt.where(HazardZone.is_blocking == is_blocking)
    if data_origin is not None:
        stmt = stmt.where(HazardZone.data_origin == data_origin)
    return paginate(session, stmt.order_by(HazardZone.name, HazardZone.id), limit, offset)


def zones_for_site(session: Session, site_id: UUID) -> list[HazardZone]:
    return list(session.scalars(select(HazardZone).where(HazardZone.site_id == site_id).order_by(HazardZone.id)))


def to_spec(zone: HazardZone) -> HazardZoneSpec:
    return HazardZoneSpec(
        id=zone.id,
        name=zone.name,
        zone_type=zone.zone_type.value,
        geometry=to_shape(zone.geometry),
        risk_score=zone.risk_score,
        is_blocking=zone.is_blocking,
        data_origin=zone.data_origin.value,
    )
