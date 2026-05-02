from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0003_nifti_demo"
down_revision = "0002_phase3_results"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "studies",
        sa.Column("acquired_at", sa.Date(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("studies", "acquired_at")
