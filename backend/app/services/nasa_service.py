"""Acquisition of source datasets from official hosts or an operator inbox. Offline tooling only.

Never exposed through public API endpoints. No network access happens at import time.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.processing.errors import ProcessingError, SourceValidationError
from app.services.storage import AssetStorage

MAX_REDIRECTS = 5
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class DownloadError(ProcessingError):
    pass


@dataclass
class AcquiredSource:
    path: Path
    kind: str  # "download" or "local_file"
    sha256: str
    size_bytes: int
    url: str | None = None
    content_type: str | None = None


def check_download_url(url: str, allowed_hosts: list[str]) -> httpx.URL:
    parsed = httpx.URL(url)
    if parsed.scheme != "https":
        raise DownloadError("Only https URLs are allowed")
    if parsed.userinfo:
        raise DownloadError("URLs with embedded credentials are not allowed")
    host = (parsed.host or "").lower().rstrip(".")
    allowed = [h.lower() for h in allowed_hosts]
    if not any(host == h or host.endswith("." + h) for h in allowed):
        raise DownloadError(f"Host '{host}' is not on the allowlist of official data hosts")
    return parsed


def _file_name(url: httpx.URL) -> str:
    name = _SAFE_NAME.sub("_", Path(url.path).name).strip("._") or "download"
    return name[:120]


def download_dataset(
    url: str,
    dest_dir: Path,
    allowed_hosts: list[str],
    max_bytes: int,
    timeout_s: float,
    expected_sha256: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> AcquiredSource:
    """Stream a file over https. Redirects are followed manually so every hop is allowlist-checked."""
    current = check_download_url(url, allowed_hosts)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / _file_name(httpx.URL(url))
    partial = target.with_name(target.name + ".part")

    try:
        with httpx.Client(transport=transport, timeout=timeout_s, follow_redirects=False) as client:
            for _ in range(MAX_REDIRECTS + 1):
                with client.stream("GET", current) as response:
                    if response.is_redirect:
                        current = check_download_url(
                            str(current.join(response.headers.get("location", ""))), allowed_hosts
                        )
                        continue
                    if response.status_code != 200:
                        raise DownloadError(f"Download failed with HTTP {response.status_code}")
                    declared = response.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > max_bytes:
                        raise DownloadError(f"Download of {declared} bytes exceeds the {max_bytes} byte limit")
                    digest, size = hashlib.sha256(), 0
                    with partial.open("wb") as handle:
                        for chunk in response.iter_bytes(1024 * 256):
                            size += len(chunk)
                            if size > max_bytes:
                                raise DownloadError(f"Download exceeds the {max_bytes} byte limit")
                            digest.update(chunk)
                            handle.write(chunk)
                    content_type = response.headers.get("content-type")
                    break
            else:
                raise DownloadError("Too many redirects")
    except httpx.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"Download failed: {exc.__class__.__name__}") from exc
    except DownloadError:
        partial.unlink(missing_ok=True)
        raise

    sha256 = digest.hexdigest()
    if expected_sha256 and sha256 != expected_sha256.lower():
        partial.unlink(missing_ok=True)
        raise DownloadError("Downloaded file checksum does not match expected_sha256")
    partial.replace(target)
    return AcquiredSource(target, "download", sha256, size, str(url), content_type)


def locate_local_file(relative_name: str, inbox: Path, expected_sha256: str | None = None) -> AcquiredSource:
    """Resolve an operator-supplied file strictly inside the inbox directory."""
    root = inbox.resolve()
    if Path(relative_name).is_absolute():
        raise SourceValidationError("local_file must be relative to the ingest inbox")
    path = (root / relative_name).resolve()
    if not path.is_relative_to(root):
        raise SourceValidationError("local_file escapes the ingest inbox")
    if not path.is_file():
        raise SourceValidationError(f"local_file not found in the ingest inbox: {relative_name}")
    sha256 = AssetStorage.sha256(path)
    if expected_sha256 and sha256 != expected_sha256.lower():
        raise SourceValidationError("local_file checksum does not match expected_sha256")
    return AcquiredSource(path, "local_file", sha256, path.stat().st_size)
