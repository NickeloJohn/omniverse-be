from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session


def paginate(session: Session, stmt: Select, limit: int, offset: int) -> tuple[list, int]:
    """Run a SELECT of one ORM entity with a total count; callers add their own ORDER BY."""
    total = session.scalar(select(func.count()).select_from(stmt.order_by(None).subquery())) or 0
    items = list(session.scalars(stmt.limit(limit).offset(offset)).unique())
    return items, total
