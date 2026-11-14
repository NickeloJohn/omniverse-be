"""Mars coordinate conventions used by MARS-NAV.

Geographic metadata: planetocentric latitude and east-positive longitude (degrees) on a
reference sphere of MARS_REFERENCE_RADIUS_M. This is NOT an Earth CRS and EPSG:3857 is never used.
The CRS of each ingested NASA product is validated and recorded per DataLayer, not assumed here.

Local metric frame ("site_local_enu"): origin at the site centre, +x east, +y north, metres.
Points are converted with a local equirectangular approximation on the reference sphere, which is
adequate for extents of a few tens of km. Geometry columns holding local coordinates use SRID 0
(planar, unitless-CRS) so PostGIS never applies an Earth datum to them.
"""

import math

MARS_REFERENCE_RADIUS_M = 3_396_190.0
GEOGRAPHIC_CRS_NAME = "Mars planetocentric lon/lat (east-positive), spherical reference radius 3396190 m"
LOCAL_CRS_NAME = "site_local_enu"
LOCAL_FRAME_DESCRIPTION = (
    "Site-centred local metric frame: origin at site centre, +x east, +y north, metres; "
    "local equirectangular approximation on the Mars reference sphere."
)
LOCAL_SRID = 0


def local_to_geographic(
    center_lon_deg: float,
    center_lat_deg: float,
    x_m: float,
    y_m: float,
    radius_m: float = MARS_REFERENCE_RADIUS_M,
) -> tuple[float, float]:
    """Return (longitude_deg_east, latitude_deg) for local metric offsets from the site centre."""
    cos_lat = math.cos(math.radians(center_lat_deg))
    if cos_lat < 1e-6:
        raise ValueError("Local frame is undefined at the poles")
    lat = center_lat_deg + math.degrees(y_m / radius_m)
    lon = center_lon_deg + math.degrees(x_m / (radius_m * cos_lat))
    return lon, lat


def geographic_to_local(
    center_lon_deg: float,
    center_lat_deg: float,
    lon_deg: float,
    lat_deg: float,
    radius_m: float = MARS_REFERENCE_RADIUS_M,
) -> tuple[float, float]:
    """Return local (x_m, y_m) offsets from the site centre for a lon/lat position."""
    cos_lat = math.cos(math.radians(center_lat_deg))
    if cos_lat < 1e-6:
        raise ValueError("Local frame is undefined at the poles")
    x = math.radians(lon_deg - center_lon_deg) * radius_m * cos_lat
    y = math.radians(lat_deg - center_lat_deg) * radius_m
    return x, y
