from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_backend_foundation"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "studies",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("public_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("study_instance_uid", sa.String(length=255), nullable=False),
        sa.Column("source_kind", sa.String(length=64), nullable=False),
        sa.Column("source_metadata", sa.JSON(), nullable=False),
        sa.Column("staging_status", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("public_id"),
        sa.UniqueConstraint("study_instance_uid"),
    )
    op.create_index("ix_studies_public_id", "studies", ["public_id"])
    op.create_index("ix_studies_study_instance_uid", "studies", ["study_instance_uid"])

    op.create_table(
        "series",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("study_id", sa.Integer(), nullable=False),
        sa.Column("series_instance_uid", sa.String(length=255), nullable=False),
        sa.Column("modality", sa.String(length=16), nullable=False),
        sa.Column("series_description", sa.String(length=255), nullable=True),
        sa.Column("protocol_name", sa.String(length=255), nullable=True),
        sa.Column("classification", sa.String(length=64), nullable=False),
        sa.Column("scanner_vendor", sa.String(length=128), nullable=True),
        sa.Column("source_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["study_id"], ["studies.id"]),
        sa.UniqueConstraint("series_instance_uid"),
    )
    op.create_index("ix_series_study_id", "series", ["study_id"])
    op.create_index("ix_series_series_instance_uid", "series", ["series_instance_uid"])

    op.create_table(
        "artifacts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("study_id", sa.Integer(), nullable=False),
        sa.Column("series_id", sa.Integer(), nullable=True),
        sa.Column("artifact_kind", sa.String(length=64), nullable=False),
        sa.Column("storage_root", sa.String(length=64), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("source_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["series_id"], ["series.id"]),
        sa.ForeignKeyConstraint(["study_id"], ["studies.id"]),
    )
    op.create_index("ix_artifacts_series_id", "artifacts", ["series_id"])
    op.create_index("ix_artifacts_study_id", "artifacts", ["study_id"])

    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("study_id", sa.Integer(), nullable=False),
        sa.Column("public_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("job_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("failure_payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["study_id"], ["studies.id"]),
        sa.UniqueConstraint("public_id"),
    )
    op.create_index("ix_jobs_public_id", "jobs", ["public_id"])
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_study_id", "jobs", ["study_id"])

    op.create_table(
        "job_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
    )
    op.create_index("ix_job_events_created_at", "job_events", ["created_at"])
    op.create_index("ix_job_events_job_id", "job_events", ["job_id"])


def downgrade() -> None:
    op.drop_index("ix_job_events_job_id", table_name="job_events")
    op.drop_index("ix_job_events_created_at", table_name="job_events")
    op.drop_table("job_events")
    op.drop_index("ix_jobs_study_id", table_name="jobs")
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_index("ix_jobs_public_id", table_name="jobs")
    op.drop_table("jobs")
    op.drop_index("ix_artifacts_study_id", table_name="artifacts")
    op.drop_index("ix_artifacts_series_id", table_name="artifacts")
    op.drop_table("artifacts")
    op.drop_index("ix_series_series_instance_uid", table_name="series")
    op.drop_index("ix_series_study_id", table_name="series")
    op.drop_table("series")
    op.drop_index("ix_studies_public_id", table_name="studies")
    op.drop_index("ix_studies_study_instance_uid", table_name="studies")
    op.drop_table("studies")
