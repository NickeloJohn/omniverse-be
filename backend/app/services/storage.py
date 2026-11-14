import hashlib
import re
import shutil
from pathlib import Path

from app.core.config import get_settings
from app.processing.errors import ProcessingError

_SAFE_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def get_storage() -> "AssetStorage":
    return AssetStorage(get_settings().data_dir)


class AssetStorage:
    """File storage rooted at one directory. Keys are relative, validated, and never leave the root."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def key_for(self, *parts: str) -> str:
        for part in parts:
            if not _SAFE_PART.match(part) or part in {".", ".."} or ".." in part:
                raise ProcessingError(f"Unsafe storage path component: {part!r}")
        return "/".join(parts)

    def resolve(self, key: str) -> Path:
        if not key or key.startswith(("/", "\\")) or "\\" in key or ":" in key:
            raise ProcessingError("Invalid storage key")
        parts = key.split("/")
        if any(p in {"", ".", ".."} for p in parts):
            raise ProcessingError("Invalid storage key")
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ProcessingError("Storage key escapes the storage root")
        return path

    def remove_tree(self, key: str) -> None:
        path = self.resolve(key)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
