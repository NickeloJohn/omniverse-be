from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models import DataLayer, ExplorationSite
from app.models.enums import DataOrigin, LayerType
from app.repositories.pagination import paginate


def list_sites(session: Session, limit: int, offset: int) -> tuple[list[ExplorationSite], int]:
    return paginate(session, select(ExplorationSite).order_by(ExplorationSite.name), limit, offset)


def get_site(session: Session, site_id: UUID) -> ExplorationSite | None:
    return session.get(ExplorationSite, site_id)


def list_layers(
    session: Session,
    limit: int,
    offset: int,
    site_id: UUID | None = None,
    layer_type: LayerType | None = None,
    data_origin: DataOrigin | None = None,
) -> tuple[list[DataLayer], int]:
    stmt = select(DataLayer).options(selectinload(DataLayer.assets))
    if site_id is not None:
        stmt = stmt.where(DataLayer.site_id == site_id)
    if layer_type is not None:
        stmt = stmt.where(DataLayer.layer_type == layer_type)
    if data_origin is not None:
        stmt = stmt.where(DataLayer.data_origin == data_origin)
    return paginate(session, stmt.order_by(DataLayer.slug), limit, offset)


def get_layer(session: Session, layer_id: UUID) -> DataLayer | None:
    stmt = select(DataLayer).options(selectinload(DataLayer.assets)).where(DataLayer.id == layer_id)
    return session.scalars(stmt).first()
