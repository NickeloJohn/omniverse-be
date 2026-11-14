"""Recompute slope (and then hazard) from a stored elevation layer.

    python -m scripts.generate_slope --site jezero-demo --elevation-layer demo-elevation
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
    parser.add_argument("--hazard-config", help="Optional hazard configuration JSON")
    args = parser.parse_args()
    settings = get_settings()

    def action() -> None:
        with SessionLocal() as session:
            result = ingestion_service.regenerate_slope(
                session, AssetStorage(settings.data_dir), settings, args.site, args.elevation_layer,
                load_hazard_config(args.hazard_config),
            )
        print(f"Slope and hazard regenerated for {result.elevation.slug}")

    run(action)


if __name__ == "__main__":
    main()
