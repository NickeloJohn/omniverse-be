"""Seed clearly-labelled DEMO data for local development.

Run from backend/: python -m scripts.seed_dev_data

Nothing here is a NASA measurement. Everything is marked is_demo / simulated / proposed, and NASA
layers are only 'planned' candidates with unknown fields left empty. Idempotent: skips if present.
"""

import logging

from geoalchemy2.shape import from_shape
from shapely.geometry import box
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models import (
    DataLayer,
    ExplorationSite,
    HazardZone,
    Mission,
    MissionTarget,
    ScienceObservation,
    ScienceTarget,
)
from app.models.enums import (
    CreationSource,
    DataOrigin,
    HazardZoneType,
    LayerType,
    MissionStatus,
    ObservationSource,
    ObservationType,
    ProcessingStatus,
    VerificationStatus,
)
from app.schemas.mission import MissionConstraints
from app.services.geo import geographic_for, point_geometry
from app.utils.mars_coords import LOCAL_SRID

logger = logging.getLogger("marsnav.seed")

SITE_SLUG = "jezero-demo"
HALF_EXTENT_M = 4000.0
HABITAT_XY = (-3000.0, -2500.0)
DEMO_TARGETS = [
    ("DEMO Target A (proposed)", "Placeholder: layered-outcrop style target", (1200.0, 800.0), 45),
    ("DEMO Target B (proposed)", "Placeholder: boulder-field margin style target", (-500.0, 2200.0), 30),
    ("DEMO Target C (proposed)", "Placeholder: crater-rim style target", (2600.0, -1200.0), 60),
]
NOT_NASA = "Seeded placeholder; not derived from NASA data."


def _bounds_polygon():
    return from_shape(
        box(-HALF_EXTENT_M, -HALF_EXTENT_M, HALF_EXTENT_M, HALF_EXTENT_M), srid=LOCAL_SRID
    )


def seed_development_data(session: Session) -> bool:
    """Returns True if data was created, False if the demo site already exists."""
    if session.scalar(select(ExplorationSite.id).where(ExplorationSite.slug == SITE_SLUG)):
        return False

    site = ExplorationSite(
        slug=SITE_SLUG,
        name="Jezero Crater region (DEMO extent, 8 km x 8 km)",
        description=(
            "Demonstration extent near Jezero Crater, Mars. Centre coordinates are approximate "
            "and not taken from an ingested NASA product. All seeded content is DEMO data."
        ),
        center_longitude_deg_east=77.58,
        center_latitude_deg=18.38,
        width_m=2 * HALF_EXTENT_M,
        height_m=2 * HALF_EXTENT_M,
        extent=_bounds_polygon(),
        habitat_location=point_geometry(*HABITAT_XY),
        data_origin=DataOrigin.SIMULATED,
        is_demo=True,
    )
    session.add(site)
    session.flush()

    demo_meta = {"note": "Synthetic terrain; contains no NASA measurements. Raster generated in a later phase."}
    candidate_meta = {"note": "Candidate source only. No specific product selected or ingested."}
    layers = [
        ("demo-elevation", "DEMO synthetic elevation", LayerType.ELEVATION, DataOrigin.SIMULATED, True),
        ("demo-slope", "DEMO synthetic slope", LayerType.SLOPE, DataOrigin.SIMULATED, True),
        ("nasa-dem-candidate", "NASA elevation model (candidate, not ingested)", LayerType.ELEVATION, DataOrigin.OBSERVED, False),
        ("nasa-imagery-candidate", "NASA optical imagery (candidate, not ingested)", LayerType.OPTICAL_IMAGERY, DataOrigin.OBSERVED, False),
    ]
    for slug, name, layer_type, origin, is_demo in layers:
        session.add(
            DataLayer(
                site_id=site.id,
                slug=slug,
                name=name,
                layer_type=layer_type,
                data_origin=origin,
                processing_status=ProcessingStatus.PLANNED,
                is_demo=is_demo,
                source_name=None if is_demo else "NASA Mars Trek / USGS Astrogeology (candidate)",
                source_url=None if is_demo else "https://trek.nasa.gov/mars/",
                spatial_resolution_m=10.0 if is_demo else None,
                units="m" if layer_type is LayerType.ELEVATION and is_demo else None,
                dataset_version="demo-0" if is_demo else "unversioned",
                processing_metadata=demo_meta if is_demo else candidate_meta,
                bounds=_bounds_polygon(),
            )
        )

    targets = []
    for name, description, (x_m, y_m), minutes in DEMO_TARGETS:
        lon, lat = geographic_for(site, x_m, y_m)
        target = ScienceTarget(
            site_id=site.id,
            name=name,
            description=description,
            scientific_objective=NOT_NASA,
            geological_context=NOT_NASA,
            location=point_geometry(x_m, y_m),
            longitude_deg_east=lon,
            latitude_deg=lat,
            estimated_activity_duration_min=minutes,
            creation_source=CreationSource.DEMO_SEED,
            verification_status=VerificationStatus.PROPOSED,
        )
        session.add(target)
        targets.append(target)

    session.add(
        HazardZone(
            site_id=site.id,
            name="DEMO obstacle zone (simulated)",
            description="Placeholder blocked area for routing tests.",
            zone_type=HazardZoneType.OBSTACLE,
            geometry=from_shape(box(300, 300, 700, 600), srid=LOCAL_SRID),
            risk_score=1.0,
            is_blocking=True,
            data_origin=DataOrigin.SIMULATED,
            source="MARS-NAV development seed",
        )
    )

    lon, lat = geographic_for(site, *HABITAT_XY)
    mission = Mission(
        name="DEMO proposed traverse",
        description="Placeholder mission for development. Not a real mission plan.",
        site_id=site.id,
        status=MissionStatus.DRAFT,
        start_location=point_geometry(*HABITAT_XY),
        start_longitude_deg_east=lon,
        start_latitude_deg=lat,
        eva_duration_limit_min=360,
        return_reserve_min=60,
        constraints=MissionConstraints().model_dump(mode="json"),
        is_demo=True,
    )
    mission.target_links = [
        MissionTarget(target=targets[0], sequence=0),
        MissionTarget(target=targets[1], sequence=1),
    ]
    session.add(mission)

    session.add(
        ScienceObservation(
            site_id=site.id,
            observation_type=ObservationType.OTHER,
            title="DEMO observation note",
            notes="Seeded placeholder record; not an instrument measurement.",
            source=ObservationSource.MANUAL,
            data_origin=DataOrigin.SIMULATED,
        )
    )
    session.commit()
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if get_settings().environment.lower() == "production":
        raise SystemExit("Refusing to seed demo data in a production environment")
    with SessionLocal() as session:
        created = seed_development_data(session)
    logger.info("Seeded DEMO data" if created else "DEMO data already present; nothing changed")


if __name__ == "__main__":
    main()
