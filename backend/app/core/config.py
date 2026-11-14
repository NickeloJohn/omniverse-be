from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "MARS-NAV API"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg://marsnav:marsnav@localhost:5434/marsnav"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    log_level: str = "INFO"

    # Raster storage root; asset storage keys are always relative to it.
    data_dir: Path = Path("./data")
    # Operator-supplied local source files must live under this directory.
    ingest_input_dir: Path | None = None
    # Only HTTPS downloads from these official hosts (and their subdomains) are allowed.
    nasa_allowed_hosts: list[str] = Field(
        default_factory=lambda: [
            "trek.nasa.gov",
            "astrogeology.usgs.gov",
            "planetarymaps.usgs.gov",
            "ode.rsl.wustl.edu",
            "pds-geosciences.wustl.edu",
        ]
    )
    max_download_mb: int = Field(default=2048, ge=1)
    download_timeout_s: float = Field(default=60.0, gt=0)
    max_grid_cells: int = Field(default=25_000_000, ge=1)

    # Route planning is CPU heavy, so it is bounded in time, work, concurrency and request rate.
    route_search_timeout_s: float = Field(default=15.0, gt=0)
    route_max_expansions: int = Field(default=2_000_000, ge=1)
    route_max_destinations: int = Field(default=12, ge=1, le=100)
    route_rate_limit_per_min: int = Field(default=30, ge=1)
    route_max_concurrent: int = Field(default=2, ge=1)

    @property
    def ingest_inbox(self) -> Path:
        return self.ingest_input_dir or self.data_dir / "inbox"


@lru_cache
def get_settings() -> Settings:
    return Settings()