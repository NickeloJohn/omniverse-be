from geoalchemy2 import WKBElement
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point

from app.core.errors import DomainValidationError
from app.models import ExplorationSite
from app.utils.mars_coords import LOCAL_SRID, local_to_geographic


def point_geometry(x_m: float, y_m: float) -> WKBElement:
    return from_shape(Point(x_m, y_m), srid=LOCAL_SRID)


def point_xy(geometry: WKBElement) -> tuple[float, float]:
    point = to_shape(geometry)
    return point.x, point.y


def ensure_within_site(site: ExplorationSite, x_m: float, y_m: float, field: str) -> None:
    if abs(x_m) > site.width_m / 2 or abs(y_m) > site.height_m / 2:
        raise DomainValidationError(
            f"{field} lies outside the site extent",
            [
                {
                    "field": field,
                    "message": (
                        f"x_m must be within ±{site.width_m / 2:g} and "
                        f"y_m within ±{site.height_m / 2:g} metres of the site centre"
                    ),
                }
            ],
        )


def geographic_for(site: ExplorationSite, x_m: float, y_m: float) -> tuple[float, float]:
    return local_to_geographic(
        site.center_longitude_deg_east, site.center_latitude_deg, x_m, y_m, site.body_radius_m
    )
