"""Validation of source raster coordinate reference systems for Mars.

A source CRS is never guessed: it must be present, describe a Mars-sized body, use metre linear units,
and use the same reference sphere as the site-local frame. Earth CRSs (EPSG:4326, 3857, ...) are rejected.
"""

from dataclasses import dataclass

import pyproj
from rasterio.crs import CRS

from app.processing.errors import SourceValidationError

# Covers Mars 2000 (a = 3396190 m), IAU 2015 mean radius (3389500 m) and polar radius; far from Earth's ~6.37e6 m.
MARS_RADIUS_RANGE_M = (3_370_000.0, 3_400_000.0)


@dataclass(frozen=True)
class CrsInfo:
    semi_major_m: float
    semi_minor_m: float
    is_sphere: bool
    is_projected: bool
    name: str

    def summary(self) -> dict:
        return {
            "name": self.name,
            "semi_major_m": self.semi_major_m,
            "semi_minor_m": self.semi_minor_m,
            "is_sphere": self.is_sphere,
            "is_projected": self.is_projected,
        }


def validate_mars_crs(
    crs: CRS | None,
    frame_radius_m: float,
    accept_ellipsoid_as_sphere: bool = False,
    radius_tolerance_m: float = 1.0,
) -> tuple[CrsInfo, list[str]]:
    """Return (info, warnings) or raise SourceValidationError."""
    if crs is None:
        raise SourceValidationError("Source has no CRS; refusing to guess a coordinate system")
    try:
        parsed = pyproj.CRS.from_wkt(crs.to_wkt())
    except Exception as exc:
        raise SourceValidationError(f"Source CRS could not be parsed: {exc}") from exc

    ellipsoid = parsed.ellipsoid
    if ellipsoid is None:
        raise SourceValidationError("Source CRS has no ellipsoid/sphere definition")
    a, b = float(ellipsoid.semi_major_metre), float(ellipsoid.semi_minor_metre)
    low, high = MARS_RADIUS_RANGE_M
    if not low <= a <= high:
        raise SourceValidationError(
            f"Source CRS is not a Mars CRS (semi-major axis {a:.0f} m; expected {low:.0f}-{high:.0f} m). "
            "Earth CRSs such as EPSG:4326 or EPSG:3857 are rejected."
        )
    if parsed.is_projected:
        factor = crs.linear_units_factor[1]
        if abs(factor - 1.0) > 1e-9:
            raise SourceValidationError(f"Source CRS linear unit factor {factor} is not metres")

    is_sphere = abs(a - b) < 1e-6
    info = CrsInfo(a, b, is_sphere, bool(parsed.is_projected), parsed.name)

    if abs(a - frame_radius_m) > radius_tolerance_m:
        raise SourceValidationError(
            f"Source reference radius {a:.1f} m differs from the site frame radius "
            f"{frame_radius_m:.1f} m; re-project the source with a matching Mars reference first"
        )
    warnings: list[str] = []
    if not is_sphere:
        if not accept_ellipsoid_as_sphere:
            raise SourceValidationError(
                "Source CRS uses an ellipsoid but the site frame is spherical; positions would be "
                "silently distorted. Set accept_ellipsoid_as_sphere to accept the approximation."
            )
        warnings.append(
            f"Ellipsoidal source CRS (b={b:.1f} m) treated as a sphere of radius {frame_radius_m:.1f} m; "
            "positional error may be several km at mid latitudes"
        )
    return info, warnings
