"""Shared helpers for the offline processing CLIs."""

import json
import logging
import sys
from pathlib import Path

from app.processing.errors import ProcessingError
from app.processing.hazard import DEFAULT_HAZARD_CONFIG, HazardConfig


def setup() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def load_hazard_config(path: str | None) -> HazardConfig:
    if path is None:
        return DEFAULT_HAZARD_CONFIG
    try:
        return HazardConfig.model_validate(json.loads(Path(path).read_text()))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Invalid hazard config: {exc}") from exc


def run(action) -> None:
    """Run a CLI action, turning expected processing errors into a clean non-zero exit."""
    try:
        action()
    except ProcessingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
