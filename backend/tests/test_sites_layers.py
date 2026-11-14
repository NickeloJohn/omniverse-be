from uuid import uuid4

from app.models import RasterAsset
from app.models.enums import RasterRole
from app.utils.mars_coords import LOCAL_CRS_NAME


def test_list_sites_is_paginated(client, seeded):
    response = client.get("/api/v1/sites", params={"limit": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1 and body["limit"] == 1 and body["offset"] == 0
    assert body["items"][0]["slug"] == "jezero-demo"


def test_get_site_exposes_local_frame_and_demo_labels(client, seeded):
    body = client.get(f"/api/v1/sites/{seeded.site.id}").json()
    assert body["is_demo"] is True and body["data_origin"] == "simulated"
    assert body["local_crs_name"] == LOCAL_CRS_NAME
    assert body["habitat_location"] == {"x_m": -3000.0, "y_m": -2500.0}
    assert "3857" not in body["geographic_crs"]


def test_unknown_site_returns_consistent_404(client):
    response = client.get(f"/api/v1/sites/{uuid4()}")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found", "message": "Exploration site not found"}}


def test_malformed_id_and_paging_are_rejected(client):
    bad_id = client.get("/api/v1/sites/not-a-uuid")
    assert bad_id.status_code == 422 and bad_id.json()["error"]["code"] == "validation_error"
    assert client.get("/api/v1/sites", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/sites", params={"limit": 101}).status_code == 422
    assert client.get("/api/v1/sites", params={"offset": -1}).status_code == 422


def test_layers_distinguish_demo_from_unavailable_nasa_layers(client, seeded):
    body = client.get("/api/v1/layers", params={"site_id": str(seeded.site.id)}).json()
    by_slug = {layer["slug"]: layer for layer in body["items"]}
    assert body["total"] == 4
    assert by_slug["demo-elevation"]["data_origin"] == "simulated" and by_slug["demo-elevation"]["is_demo"]
    nasa = by_slug["nasa-dem-candidate"]
    assert nasa["processing_status"] == "planned" and nasa["assets"] == []
    # Unknown provenance stays unknown rather than being invented.
    assert nasa["source_product_id"] is None and nasa["spatial_resolution_m"] is None
    assert nasa["acquisition_date"] is None


def test_layer_filters_and_404(client, seeded):
    observed = client.get("/api/v1/layers", params={"data_origin": "observed"}).json()
    assert observed["total"] == 2
    assert client.get("/api/v1/layers", params={"layer_type": "bogus"}).status_code == 422
    assert client.get(f"/api/v1/layers/{uuid4()}").status_code == 404


def test_layer_detail_never_exposes_storage_paths(client, db_session, seeded):
    layer = client.get("/api/v1/layers", params={"data_origin": "simulated"}).json()["items"][0]
    db_session.add(RasterAsset(layer_id=layer["id"], role=RasterRole.PROCESSED,
                               storage_key="secret/internal/path.tif", media_type="image/tiff",
                               width_px=10, height_px=10))
    db_session.commit()
    response = client.get(f"/api/v1/layers/{layer['id']}")
    assert response.status_code == 200
    assert len(response.json()["assets"]) == 1
    assert "secret/internal" not in response.text and "storage_key" not in response.text
