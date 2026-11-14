"""simulation resource invariants

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-14 11:31:12.007489
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0004'
down_revision: Union[str, None] = '0003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_check_constraint(
        "ck_eva_sessions_distance_nonnegative", "eva_sessions", "distance_along_route_m >= 0"
    )
    op.create_check_constraint(
        "ck_eva_sessions_oxygen_range", "eva_sessions", "oxygen_remaining_pct BETWEEN 0 AND 100"
    )
    op.create_check_constraint(
        "ck_eva_sessions_battery_range", "eva_sessions", "battery_remaining_pct BETWEEN 0 AND 100"
    )
    op.create_check_constraint("ck_telemetry_time_nonnegative", "telemetry_samples", "sim_time_s >= 0")
    op.create_check_constraint(
        "ck_telemetry_distance_nonnegative", "telemetry_samples", "distance_along_route_m >= 0"
    )
    op.create_check_constraint(
        "ck_telemetry_oxygen_range", "telemetry_samples", "oxygen_remaining_pct BETWEEN 0 AND 100"
    )
    op.create_check_constraint(
        "ck_telemetry_battery_range", "telemetry_samples", "battery_remaining_pct BETWEEN 0 AND 100"
    )


def downgrade() -> None:
    op.drop_constraint("ck_telemetry_battery_range", "telemetry_samples", type_="check")
    op.drop_constraint("ck_telemetry_oxygen_range", "telemetry_samples", type_="check")
    op.drop_constraint("ck_telemetry_distance_nonnegative", "telemetry_samples", type_="check")
    op.drop_constraint("ck_telemetry_time_nonnegative", "telemetry_samples", type_="check")
    op.drop_constraint("ck_eva_sessions_battery_range", "eva_sessions", type_="check")
    op.drop_constraint("ck_eva_sessions_oxygen_range", "eva_sessions", type_="check")
    op.drop_constraint("ck_eva_sessions_distance_nonnegative", "eva_sessions", type_="check")