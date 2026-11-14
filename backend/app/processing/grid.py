"""Target raster grid in the site-local Mars metric frame.

All derived layers (elevation, imagery, slope, hazard) share exactly this grid, which is what
"aligned" means here. Row 0 is the northern edge; x grows east, y grows north, origin at site centre.
"""

from dataclasses import dataclass

from affine import Affine
from rasterio.crs import CRS

from app.processing.errors import ProcessingError


def mars_local_proj_string(center_lon_deg: float, center_lat_deg: float, radius_m: float) -> str:
    """PROJ definition of the site-local frame.

    Equirectangular with lat_ts = lat_0 = site latitude on the Mars sphere gives
    x = R*cos(lat0)*dlon, y = R*dlat, i.e. exactly app.utils.mars_coords.local_to_geographic.
    """
    lon, lat, r = (format(v, ".12g") for v in (center_lon_deg, center_lat_deg, radius_m))
    return f"+proj=eqc +lat_ts={lat} +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +R={r} +units=m +no_defs"


def mars_local_crs(center_lon_deg: float, center_lat_deg: float, radius_m: float) -> CRS:
    return CRS.from_string(mars_local_proj_string(center_lon_deg, center_lat_deg, radius_m))


@dataclass(frozen=True)
class TargetGrid:
    width_m: float
    height_m: float
    resolution_m: float
    crs: CRS
    width_px: int
    height_px: int

    @classmethod
    def build(
        cls,
        width_m: float,
        height_m: float,
        resolution_m: float,
        crs: CRS,
        max_cells: int,
    ) -> "TargetGrid":
        if resolution_m <= 0:
            raise ProcessingError("resolution_m must be positive")
        cols, rows = width_m / resolution_m, height_m / resolution_m
        if abs(cols - round(cols)) > 1e-6 or abs(rows - round(rows)) > 1e-6 or min(cols, rows) < 1:
            raise ProcessingError(
                f"Site extent {width_m:g} x {height_m:g} m is not a whole number of "
                f"{resolution_m:g} m cells"
            )
        cols, rows = round(cols), round(rows)
        if cols * rows > max_cells:
            raise ProcessingError(f"Grid of {cols * rows} cells exceeds the limit of {max_cells}")
        return cls(width_m, height_m, resolution_m, crs, cols, rows)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height_px, self.width_px

    @property
    def transform(self) -> Affine:
        r = self.resolution_m
        return Affine(r, 0.0, -self.width_m / 2, 0.0, -r, self.height_m / 2)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) in local metres."""
        return (-self.width_m / 2, -self.height_m / 2, self.width_m / 2, self.height_m / 2)

    def cell_of(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        """(row, col) of the cell containing the point, or None outside the extent (edges inclusive)."""
        min_x, min_y, max_x, max_y = self.bounds
        if not (min_x <= x_m <= max_x and min_y <= y_m <= max_y):
            return None
        col = min(int((x_m - min_x) // self.resolution_m), self.width_px - 1)
        row = min(int((max_y - y_m) // self.resolution_m), self.height_px - 1)
        return row, col

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        min_x, _, _, max_y = self.bounds
        r = self.resolution_m
        return min_x + (col + 0.5) * r, max_y - (row + 0.5) * r
