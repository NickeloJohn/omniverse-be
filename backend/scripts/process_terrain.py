"""DEMO terrain: generate synthetic, clearly simulated terrain and derive slope and hazard layers.

    python -m scripts.process_terrain --site jezero-demo --demo [--resolution 10] [--hazard-config cfg.json]

Production NASA ingestion lives in scripts.import_nasa_data and is deliberately separate.
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
    parser.add_argument("--site", required=True, help="Exploration site slug")
    parser.add_argument("--demo", action="store_true", required=True, help="Required: confirms synthetic DEMO data")
    parser.add_argument("--resolution", type=float, default=10.0, help="Cell size in metres")
    parser.add_argument("--hazard-config", help="Optional hazard configuration JSON")
    args = parser.parse_args()

    settings = get_settings()
    if settings.environment.lower() == "production":
        raise SystemExit("Refusing to create DEMO terrain in a production environment")

    def action() -> None:
        with SessionLocal() as session:
            result = ingestion_service.process_demo_terrain(
                session, AssetStorage(settings.data_dir), settings, args.site, args.resolution,
                load_hazard_config(args.hazard_config),
            )
        print(f"DEMO terrain ready: {result.elevation.slug}, {result.slope.slug}, {result.hazard.slug}")

    run(action)


if __name__ == "__main__":
    main()
