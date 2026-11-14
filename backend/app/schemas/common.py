from datetime import datetime
from typing import Annotated, Any, Generic, TypeVar
from uuid import UUID

from geoalchemy2 import WKBElement
from geoalchemy2.shape import to_shape
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


class LocalPoint(BaseModel):
    """Position in the site-local metric frame (metres, +x east, +y north from site centre)."""

    model_config = ConfigDict(extra="forbid")

    x_m: float = Field(allow_inf_nan=False)
    y_m: float = Field(allow_inf_nan=False)


class LocalBounds(BaseModel):
    min_x_m: float
    min_y_m: float
    max_x_m: float
    max_y_m: float


def _point_from_geometry(value: Any) -> Any:
    if isinstance(value, WKBElement):
        point = to_shape(value)
        return {"x_m": point.x, "y_m": point.y}
    return value


def _bounds_from_geometry(value: Any) -> Any:
    if isinstance(value, WKBElement):
        min_x, min_y, max_x, max_y = to_shape(value).bounds
        return {"min_x_m": min_x, "min_y_m": min_y, "max_x_m": max_x, "max_y_m": max_y}
    return value


LocalPointField = Annotated[LocalPoint, BeforeValidator(_point_from_geometry)]
LocalBoundsField = Annotated[LocalBounds, BeforeValidator(_bounds_from_geometry)]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Timestamped(ORMModel):
    id: UUID
    created_at: datetime
    updated_at: datetime
