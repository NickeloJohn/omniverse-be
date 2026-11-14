import hashlib

import httpx
import pytest

from app.processing.errors import ProcessingError, SourceValidationError
from app.services.nasa_service import (
    DownloadError, check_download_url, download_dataset, locate_local_file,
)
from app.services.storage import AssetStorage

HOSTS = ["trek.nasa.gov", "astrogeology.usgs.gov"]
PAYLOAD = b"synthetic-bytes" * 100


def mock(handler):
    return httpx.MockTransport(handler)


def fetch(tmp_path, url="https://trek.nasa.gov/data/dem.tif", handler=None, **kwargs):
    handler = handler or (lambda request: httpx.Response(200, content=PAYLOAD, headers={"content-type": "image/tiff"}))
    kwargs.setdefault("max_bytes", 10_000_000)
    return download_dataset(url, tmp_path, HOSTS, timeout_s=5, transport=mock(handler), **kwargs)


# --- allowlist -----------------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://trek.nasa.gov/dem.tif",                      # not https
    "https://evil.example/dem.tif",                      # not allowlisted
    "https://trek.nasa.gov.evil.example/dem.tif",        # look-alike suffix
    "https://eviltrek.nasa.gov.example/x.tif",
    "https://user:pw@trek.nasa.gov/dem.tif",             # credentials in URL
    "file:///etc/passwd",
    "ftp://trek.nasa.gov/dem.tif",
])
def test_unsafe_urls_are_rejected_before_any_request(url):
    with pytest.raises(DownloadError):
        check_download_url(url, HOSTS)


def test_allowlisted_hosts_and_subdomains_pass():
    assert check_download_url("https://trek.nasa.gov/a.tif", HOSTS).host == "trek.nasa.gov"
    assert check_download_url("https://data.astrogeology.usgs.gov/a.tif", HOSTS)


def test_download_streams_to_disk_and_records_checksum(tmp_path):
    result = fetch(tmp_path)
    assert result.path.read_bytes() == PAYLOAD and result.kind == "download"
    assert result.sha256 == hashlib.sha256(PAYLOAD).hexdigest() and result.size_bytes == len(PAYLOAD)
    assert not list(tmp_path.glob("*.part"))


def test_expected_checksum_must_match_and_leaves_no_file(tmp_path):
    with pytest.raises(DownloadError, match="checksum"):
        fetch(tmp_path, expected_sha256="0" * 64)
    assert not list(tmp_path.iterdir())
    assert fetch(tmp_path, expected_sha256=hashlib.sha256(PAYLOAD).hexdigest().upper()).path.exists()


def test_size_limit_is_enforced_for_declared_and_streamed_size(tmp_path):
    with pytest.raises(DownloadError, match="exceeds"):
        fetch(tmp_path, max_bytes=100)
    chunked = lambda request: httpx.Response(200, content=iter([b"x" * 80, b"x" * 80]))
    with pytest.raises(DownloadError, match="exceeds"):
        fetch(tmp_path, handler=chunked, max_bytes=100)
    assert not list(tmp_path.iterdir())


def test_http_errors_are_reported_without_leaving_files(tmp_path):
    with pytest.raises(DownloadError, match="HTTP 404"):
        fetch(tmp_path, handler=lambda request: httpx.Response(404))
    assert not list(tmp_path.iterdir())


def test_redirect_to_allowlisted_host_is_followed(tmp_path):
    def handler(request):
        if request.url.host == "trek.nasa.gov":
            return httpx.Response(302, headers={"location": "https://astrogeology.usgs.gov/final.tif"})
        return httpx.Response(200, content=PAYLOAD)
    assert fetch(tmp_path, handler=handler).path.read_bytes() == PAYLOAD


def test_redirect_to_unlisted_host_is_blocked(tmp_path):
    handler = lambda request: httpx.Response(302, headers={"location": "https://evil.example/steal.tif"})
    with pytest.raises(DownloadError, match="allowlist"):
        fetch(tmp_path, handler=handler)


def test_redirect_loops_are_bounded(tmp_path):
    handler = lambda request: httpx.Response(302, headers={"location": "https://trek.nasa.gov/again.tif"})
    with pytest.raises(DownloadError, match="redirects"):
        fetch(tmp_path, handler=handler)


def test_download_file_name_cannot_escape_destination(tmp_path):
    result = fetch(tmp_path / "dl", url="https://trek.nasa.gov/..%2F..%2Fetc%2Fpasswd")
    assert result.path.parent == tmp_path / "dl"


# --- local files -------------------------------------------------------------------------------

def test_local_file_inside_inbox_is_accepted(tmp_path):
    (tmp_path / "dem.tif").write_bytes(PAYLOAD)
    result = locate_local_file("dem.tif", tmp_path)
    assert result.kind == "local_file" and result.sha256 == hashlib.sha256(PAYLOAD).hexdigest()


@pytest.mark.parametrize("name", ["../secret.tif", "/etc/passwd", "sub/../../x.tif", "missing.tif"])
def test_local_file_outside_inbox_or_missing_is_rejected(tmp_path, name):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (tmp_path / "secret.tif").write_bytes(b"x")
    with pytest.raises(SourceValidationError):
        locate_local_file(name, inbox)


def test_local_file_checksum_mismatch_is_rejected(tmp_path):
    (tmp_path / "dem.tif").write_bytes(PAYLOAD)
    with pytest.raises(SourceValidationError, match="checksum"):
        locate_local_file("dem.tif", tmp_path, "f" * 64)


def test_local_file_symlink_escape_is_rejected(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    outside = tmp_path / "outside.tif"
    outside.write_bytes(b"x")
    (inbox / "link.tif").symlink_to(outside)
    with pytest.raises(SourceValidationError, match="escapes"):
        locate_local_file("link.tif", inbox)


# --- asset storage -------------------------------------------------------------------------------

def test_storage_keys_resolve_inside_root(tmp_path):
    storage = AssetStorage(tmp_path)
    key = storage.key_for("sites", "jezero-demo", "demo-elevation", "v1")
    assert key == "sites/jezero-demo/demo-elevation/v1"
    assert storage.resolve(key + "/elevation.tif").is_relative_to(tmp_path.resolve())


@pytest.mark.parametrize("key", ["/etc/passwd", "../x", "a/../../x", "a//b", "a\\b", "C:/x", "", "a/./b"])
def test_storage_rejects_unsafe_keys(tmp_path, key):
    with pytest.raises(ProcessingError):
        AssetStorage(tmp_path).resolve(key)


@pytest.mark.parametrize("part", ["..", ".", "a/b", "", "a b", "-x", "x" * 200, "a..b"])
def test_storage_rejects_unsafe_path_components(tmp_path, part):
    with pytest.raises(ProcessingError):
        AssetStorage(tmp_path).key_for("sites", part)
