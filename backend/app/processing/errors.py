class ProcessingError(Exception):
    """A raster processing step failed; the message is safe to show to an operator."""


class SourceValidationError(ProcessingError):
    """Source data failed validation (CRS, units, coverage, ...) and must not be ingested."""
