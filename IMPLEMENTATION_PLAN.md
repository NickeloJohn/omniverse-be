# MARS-NAV Backend Plan

## Phase 1: Repository assessment

The supplied workspace was empty at inspection time: it contains no frontend, package manifest, source files, mocks, or Git metadata. Therefore there are no existing React components, API services, Zustand stores, TypeScript interfaces, or build commands to inspect. Frontend alignment and integration must wait until the working frontend is added to this workspace; no frontend code will be recreated or assumed.

## Planned implementation

1. Establish a modular FastAPI service, environment-based settings, database session management, CORS, container setup, Alembic, and a tested health endpoint.
2. Add persistent domain models and mission/science APIs, with database-backed tests and sample development records.
3. Add offline geospatial ingestion, terrain processing, hazard analysis, and synthetic fixtures clearly classified as simulated.
4. Implement and test bounded A* route planning with documented costs and route metrics.
5. Add deterministic, explicitly simulated EVA state transitions and telemetry.
6. Inspect the actual frontend once available, then integrate through typed API adapters and incremental UI changes.
7. Run the available backend and frontend checks and document remaining data/configuration requirements.

## Phase 2 scope

This phase creates the service foundation only. It does not claim domain persistence, NASA data ingestion, terrain analysis, route planning, simulation, or frontend integration are implemented. PostgreSQL uses the PostGIS image; the first migration enables the extension. Database connectivity is reported by `/health` so a running API with an unavailable database is visibly degraded.

## Coordinate reference policy

Route computation will use local metric coordinates in a documented site-centered Mars projection, not EPSG:3857 or Earth latitude/longitude conventions. The exact projection and geographic metadata will be selected and stored with each exploration site in the domain-model phase, where the actual site extent and dataset metadata can be validated.