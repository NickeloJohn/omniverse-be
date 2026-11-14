from logging.config import fileConfig

from alembic import context
from geoalchemy2 import Geometry, alembic_helpers
from sqlalchemy import engine_from_config, pool

from app.core.config import get_settings
from app.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

config.set_main_option("sqlalchemy.url", get_settings().database_url.replace("%", "%%"))
target_metadata = Base.metadata

# GeoAlchemy2 helpers handle spatial indexes and skip PostGIS-owned tables (spatial_ref_sys).
def include_object(obj, name, type_, reflected, compare_to):
    # Never drop tables we do not own (e.g. PostGIS tiger/topology tables in the postgis image).
    if type_ == "table" and reflected and name not in target_metadata.tables:
        return False
    return alembic_helpers.include_object(obj, name, type_, reflected, compare_to)


def compare_type(context, inspected_column, metadata_column, inspected_type, metadata_type):
    # GeoAlchemy2 reflects PostGIS SRID 0 as -1 ("unspecified"); they are the same.
    if isinstance(inspected_type, Geometry) and isinstance(metadata_type, Geometry):
        def srid(value: int | None) -> int:
            return 0 if value in (None, -1) else value

        return (
            inspected_type.geometry_type != metadata_type.geometry_type
            or srid(inspected_type.srid) != srid(metadata_type.srid)
        )
    return None


geo_options = {
    "include_object": include_object,
    "compare_type": compare_type,
    "process_revision_directives": alembic_helpers.writer,
    "render_item": alembic_helpers.render_item,
}


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **geo_options,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, **geo_options)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()