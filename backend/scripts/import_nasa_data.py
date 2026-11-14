"""Ingest an observed NASA/USGS Jezero DEM (and optional imagery) described by a job file.

    python -m scripts.import_nasa_data --job path/to/job.json

The job names a verified https URL on an allowlisted host, or a file inside the ingest inbox. Nothing is
downloaded unless the job asks for it, and nothing is invented: missing CRS, units or coverage abort the run.
"""

import argparse
import json

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.schemas.ingestion import IngestionJob
from app.services import ingestion_service
from app.services.storage import AssetStorage
from scripts._cli import run, setup


def main() -> None:
    setup()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True, help="Path to the ingestion job JSON file")
    args = parser.parse_args()

    settings = get_settings()
    with open(args.job, encoding="utf-8") as handle:
        job = IngestionJob.model_validate(json.load(handle))

    def action() -> None:
        with SessionLocal() as session:
            result = ingestion_service.ingest_job(session, AssetStorage(settings.data_dir), settings, job)
        print(f"Ingested terrain set: {result.elevation.slug}, {result.slope.slug}, {result.hazard.slug}")
        for warning in result.warnings:
            print(f"WARNING: {warning}")

    run(action)


if __name__ == "__main__":
    main()
