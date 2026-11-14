# MARS-NAV Backend

Phases 1–6 provide the FastAPI/PostGIS backend, NASA/USGS raster ingestion pipeline, terrain analysis, A* routing, and deterministic simulated EVA APIs. Phase 7 frontend integration is blocked because this workspace has no React app, TypeScript files, package manifest, Zustand stores, TanStack Query setup, UI components, or mock services. No frontend was recreated or modified. Add the existing frontend to this workspace before implementing its adapters and preserving its UI.

## Local development

Requirements: Docker with Compose, and Python 3.12 or newer. The commands below are run from the repository root unless a `cd backend` is shown.

From the repository root, start PostgreSQL/PostGIS (host port 5434 by default; override with `MARSNAV_DB_PORT`):

```sh
docker compose up -d db
```

From the repository root, configure the local Python backend:

```sh
cp backend/.env.example backend/.env
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
alembic upgrade head
python -m scripts.seed_dev_data
python -m scripts.process_terrain --site jezero-demo --demo
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

For container startup from the repository root, run `docker compose up --build`. It starts the PostGIS database and API; run the two seed/DEMO terrain commands above once from `backend/` if the site needs local demo data.

The API is available at `http://localhost:8000`, Swagger UI at `/docs`, and readiness at `/health` or `/api/v1/health`. Health returns HTTP 503 when PostgreSQL is unavailable. Compose uses development-only database credentials; replace them before any non-local deployment.

## Tests

Run the backend suite from `backend/` with `../.venv/bin/python -m pytest -q`. DB tests use a separate `marsnav_test` database in the same PostGIS container (`TEST_DATABASE_URL` overrides it; the name must end in `_test`). Its schema is rebuilt from the Alembic migrations on every run, so migrations are exercised too. If the database is unreachable, DB tests are skipped with a reason. No external NASA services are contacted. Run `../.venv/bin/alembic check` from `backend/` to check model/migration drift. No frontend TypeScript compile, production build, or browser workflow can run until the frontend is present.

## Coordinates

Geographic metadata uses Mars planetocentric, east-positive lon/lat on a spherical reference radius of 3,396,190 m. Computation uses a site-local metric frame (`site_local_enu`: origin at site centre, +x east, +y north, metres) stored as PostGIS geometry with SRID 0, so no Earth datum or EPSG:3857 is applied. See `app/utils/mars_coords.py`. The CRS of each ingested NASA product is to be recorded per data layer.

## Data labelling

- Seed data is DEMO: sites and layers are `is_demo`, simulated data is `data_origin=simulated`, targets are `creation_source=demo_seed` with `verification_status=proposed`.
- NASA layers in the seed are `planned` candidates only; unknown provenance fields (product ID, resolution, date) are left empty rather than invented.
- Public APIs only create `manual`/`proposed` targets and `manual`/`simulated` observations. NASA verification and product-sourced records are not client-settable.

## Terrain pipeline (offline)

All processing is operator-run from the CLI; there are no public ingestion endpoints. Every layer shares one grid in the site-local frame, so imagery, elevation, slope and hazard are aligned by construction. Rasterio ships its own GDAL (no system GDAL needed).

| Step | Behaviour |
| --- | --- |
| Acquire | `url` must be https on an allowlisted host (`NASA_ALLOWED_HOSTS`; every redirect hop is re-checked), size-limited, optional `expected_sha256`; or `local_file` strictly inside `INGEST_INPUT_DIR` (default `data/inbox`) |
| Validate | CRS must exist, be a Mars CRS (Earth CRSs are rejected), use the site frame's reference radius and metre units; elevation units must be metres (declared in the raster or stated in the job); coverage is checked |
| Crop and align | Reads only the window overlapping the site and resamples onto the site grid (bilinear when upsampling, average when downsampling) |
| NoData | Resampled separately: a cell is valid only if all contributing source cells were valid; cells outside coverage are NoData. Nothing is filled or interpolated |
| Slope | Horn 3x3; a cell and its 8 neighbours must be valid, so unknown terrain also removes a one-cell margin |
| Hazard | Slope categories to a 0-1 risk score; unknown cells blocked by default |
| Register | GeoTIFFs and PNG previews written, then DB rows; any failure rolls back rows and removes files created by the run |

```sh
python -m scripts.import_nasa_data --job job.json   # observed NASA/USGS data (see app/schemas/ingestion.py)
python -m scripts.process_terrain --site jezero-demo --demo   # synthetic DEMO terrain, labelled simulated
python -m scripts.generate_slope --site S --elevation-layer SLUG     # recompute slope, then hazard
python -m scripts.generate_hazards --site S --elevation-layer SLUG --hazard-config cfg.json
```

Example job (no NASA product is bundled or assumed; fill in a product you have verified):

```json
{
  "site_slug": "jezero-demo",
  "layer_prefix": "jezero-dem",
  "resolution_m": 10,
  "elevation": {
    "name": "<product name>",
    "local_file": "my_dem.tif",
    "product_id": "<PDS/USGS product id>",
    "acquisition_date": "YYYY-MM-DD",
    "units": "m",
    "candidate_slug": "nasa-dem-candidate"
  }
}
```

`candidate_slug` promotes the seeded planned candidate layer instead of creating a second one. Source rasters are not copied into storage (they can be huge); their SHA-256 and provenance are recorded. Reprocessing writes a new dataset version beside the old one.

Hazard defaults are **prototype values, not validated NASA astronaut safety limits**: low `[0,5)`, moderate `[5,10)`, high `[10,15]`, excluded above 15 degrees. Orbital-resolution elevation cannot identify astronaut-scale obstacles, so low risk never means safe. Hazard zones (user-defined obstacles) are overlaid at query time, not baked into the stored hazard layer. All files live under `DATA_DIR`; the API exposes ids and metadata, never filesystem paths.

## Route planning

`POST /api/v1/routes/plan` runs a real A* search over the processed terrain grid for the requested profiles and returns the routes with a comparison. It is read-only: nothing is saved. `POST /api/v1/missions/{id}/routes` plans on the server with the mission's context and saves the route (client-supplied geometry is never accepted); `GET /api/v1/routes/{id}` and `GET /api/v1/missions/{id}/routes` read saved routes. Mission routes can only be added while the mission is draft or planned.

Request essentials: `site_id` or `mission_id`; optional `start` (default mission start, else habitat); ordered `destinations`, each a stored `science_target_id` or a raw `location` waypoint (default: the mission's targets); `return_to_habitat` (default true); `profiles` (default both); optional `eva_duration_limit_min`, `return_reserve_min` and `constraints` overriding the mission. Stops are visited in the order given; there is no reordering.

- **Movement:** 8-connected, and a diagonal move is refused unless both orthogonal neighbours are traversable, so paths never squeeze between diagonally touching blocked cells. Blocked cells (excluded slope, unknown terrain, blocking hazard zones) are impassable. The route starts and ends at the exact requested coordinates and is optimal over the grid of cell centres.
- **Cost:** `shorter_distance` costs length. `safety_first` costs length times `1 + 6*risk + 2*(slope/limit)^2 + 2*uncertainty` (mean of the two cells of a step); uncertainty is 1 next to unknown terrain and fades to 0 over 40 m. Every factor is at least 1, so the octile-distance heuristic is admissible and consistent. Both profiles are checked against an independent Dijkstra in the tests.
- **Metrics:** distance, travel time, risk (length-weighted mean and peak), slope, elevation gain/loss, high-risk and uncertain distance are recomputed from the final path geometry, segment by segment, never taken from configuration or the search. Per-stop and per-leg figures are cumulative along that geometry.
- **Time:** walking speed is a configurable prototype model (0.8 m/s on flat ground, falling linearly to 40% at the slope limit) plus each stop's activity time. It is not astronaut or spacesuit performance data.
- **Mission constraints:** `max_slope_deg` and `block_unknown_terrain` can only tighten the processed hazard layer; a looser request gets the layer's limit plus a warning. The EVA time budget is `eva_duration_limit_min - return_reserve_min`; a route that exceeds it is still returned, with `within_time_budget=false` and a violation. Without a return leg the reserve cannot be verified and the route says so.
- **Failure is explicit:** a profile with no route returns `status=no_route` with a code (`start_blocked`, `destination_blocked`, `unreachable`, `search_limit_exceeded`, `search_timeout`), the failing leg and a message. Nothing is invented. `search_*` codes mean the search was cut off, not that no path exists.
- **Limits:** per-leg search timeout and expansion cap, at most `ROUTE_MAX_DESTINATIONS` stops, a per-client rate limit (HTTP 429 with `Retry-After`) and a cap on concurrent planner runs. These are in-process limits for a single server.

The outermost ring of cells is always blocked, because slope needs a full 3x3 neighbourhood.

## API (Phase 4 additions)

`GET /terrain/sample?site_id&x_m&y_m` (nearest-cell elevation, slope, risk, blocked reasons, provenance; 404 `terrain_not_available` until terrain is processed), `GET /terrain/metadata?site_id`, `GET /hazards`, `GET /layers/{id}/preview` (PNG, NoData transparent). Real (non-demo) terrain is preferred over DEMO terrain when both exist.

## API (Phase 3)

Under `/api/v1`: `/sites` and `/layers` (read-only), `/missions` (CRUD; delete only draft or aborted), `/science-targets`, `/observations`. Errors use `{"error": {"code", "message", "details?"}}`. Mission status flow: draft ↔ planned → active ⇄ paused → completed, with abort allowed from any non-terminal state. Plan fields freeze once a mission is active.

## EVA simulation (Phase 6)

`POST /simulations` starts an EVA from a persisted, found mission route. The logical clock advances only through `POST /simulations/{id}/advance` with `advance_by_s`; pause, resume, and reset are `POST /pause`, `/resume`, and `/reset` on that session. Read current state with `GET /simulations/{id}/state`; paginated simulated telemetry is at `GET /simulations/{id}/telemetry`. Position follows saved route coordinates and stop timing. The resource rates and thresholds are configurable prototype assumptions, not measured astronaut or suit performance. Session and telemetry origins are constrained to `simulated`.

## Frontend integration handoff

When the existing React app is added, set `VITE_API_URL=http://localhost:8000` and integrate through typed adapters rather than changing component types implicitly. Available API groups: missions (`GET/POST /missions`, `GET/PATCH/DELETE /missions/{id}`, saved routes at `/missions/{id}/routes`); science (`/science-targets`, `/observations`); route planning (`POST /routes/plan`); simulations (`POST /simulations`, then state/advance/pause/resume/reset); layers and terrain (`/layers`, `/terrain/metadata`, `/terrain/sample`); health (`/health`). Swagger at `/docs` and `/openapi.json` is the authoritative request/response contract. Use the existing UI's mock/demo presentation only as an explicit fallback when the API is unavailable; don't label simulated or proposed backend data as NASA-verified.