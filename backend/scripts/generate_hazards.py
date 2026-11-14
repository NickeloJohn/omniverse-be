"""Recompute only the hazard layer from the stored slope raster with a given hazard configuration.

    python -m scripts.generate_hazards --site jezero-demo --elevation-layer demo-elevation --hazard-config cfg.json

Slope categories are prototype values, not validated NASA astronaut safety limits.
"""

import argparse

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.services import ingestion_service
from app.services.storage import AssetStorage
from scripts._cli import load_hazard_config, run, setup


def main() -> None:
    setup()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", required=True)
    parser.add_argument("--elevation-layer", required=True, help="Slug of a processed elevation layer")
    parser.add_argument("--hazard-config", help="Hazard configuration JSON (defaults to prototype values)")
    args = parser.parse_args()
    settings = get_settings()

    def action() -> None:
        with SessionLocal() as session:
            result = ingestion_service.regenerate_hazard(
                session, AssetStorage(settings.data_dir), settings, args.site, args.elevation_layer,
                load_hazard_config(args.hazard_config),
            )
        print(f"Hazard layer regenerated: {result.hazard.slug} ({result.hazard.dataset_version})")

    run(action)


if __name__ == "__main__":
    main()
