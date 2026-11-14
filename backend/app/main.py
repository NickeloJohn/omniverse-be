import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import hazards, layers, missions, routes, science, simulations, sites, terrain
from app.api.v1.health import router as health_router
from app.core.config import get_settings
from app.core.errors import register_exception_handlers

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper())

app = FastAPI(title=settings.app_name, version="0.1.0")
register_exception_handlers(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
app.include_router(health_router, prefix=settings.api_v1_prefix)
app.include_router(health_router)
for module in (sites, layers, terrain, hazards, missions, routes, science, simulations):
    app.include_router(module.router, prefix=settings.api_v1_prefix)