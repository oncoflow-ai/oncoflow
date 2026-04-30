from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_phase3_results"
down_revision = "0001_backend_foundation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "study_results",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("study_id", sa.Integer(), nullable=False),
        sa.Column("result_kind", sa.String(length=64), nullable=False),
        sa.Column("needs_review", sa.Boolean(), nullable=False),
        sa.Column("summary_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["study_id"], ["studies.id"]),
    )
    op.create_index("ix_study_results_study_id", "study_results", ["study_id"])

    op.create_table(
        "lesion_results",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("study_result_id", sa.Integer(), nullable=False),
        sa.Column("study_id", sa.Integer(), nullable=False),
        sa.Column("lesion_id", sa.String(length=255), nullable=False),
        sa.Column("measurement_payload", sa.JSON(), nullable=False),
        sa.Column("bounding_box", sa.JSON(), nullable=False),
        sa.Column("artifact_refs", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["study_result_id"], ["study_results.id"]),
        sa.ForeignKeyConstraint(["study_id"], ["studies.id"]),
        sa.UniqueConstraint("study_result_id", "lesion_id"),
    )
    op.create_index("ix_lesion_results_study_result_id", "lesion_results", ["study_result_id"])
    op.create_index("ix_lesion_results_study_id", "lesion_results", ["study_id"])


def downgrade() -> None:
    op.drop_index("ix_lesion_results_study_id", table_name="lesion_results")
    op.drop_index("ix_lesion_results_study_result_id", table_name="lesion_results")
    op.drop_table("lesion_results")
    op.drop_index("ix_study_results_study_id", table_name="study_results")
    op.drop_table("study_results")
