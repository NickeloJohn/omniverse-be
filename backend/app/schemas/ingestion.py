from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.processing.hazard import DEFAULT_HAZARD_CONFIG, HazardConfig

Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{1,40}$")]
Version = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[A-Fa-f0-9]{64}$")]


class RasterSource(BaseModel):
    """One source raster: a verified URL on an allowlisted host, or a file in the ingest inbox."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200, description="Human-readable product name for provenance")
    url: str | None = None
    local_file: str | None = Field(default=None, description="Path relative to the ingest inbox")
    product_id: str | None = Field(default=None, max_length=200)
    acquisition_date: date | None = None
    citation: str | None = Field(default=None, max_length=500)
    expected_sha256: Sha256 | None = None
    units: str | None = Field(default=None, description="Elevation units; required if the raster does not declare them")
    vertical_datum: str | None = Field(default=None, max_length=120)
    band: int = Field(default=1, ge=1)
    band_indexes: list[int] | None = Field(default=None, description="Imagery bands to align (default all)")
    accept_ellipsoid_as_sphere: bool = False
    candidate_slug: str | None = Field(default=None, description="Promote this planned candidate layer instead of creating one")

    @model_validator(mode="after")
    def exactly_one_location(self) -> "RasterSource":
        if (self.url is None) == (self.local_file is None):
            raise ValueError("Provide exactly one of url or local_file")
        return self


class IngestionJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    site_slug: str = Field(min_length=1, max_length=64)
    layer_prefix: Slug
    resolution_m: float = Field(default=10.0, gt=0, le=1000)
    elevation: RasterSource
    imagery: RasterSource | None = None
    hazard: HazardConfig = DEFAULT_HAZARD_CONFIG
    dataset_version: Version | None = None
    min_valid_fraction: float = Field(default=0.5, ge=0, le=1)
