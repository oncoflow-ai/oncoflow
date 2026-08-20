"""add patients and assignments tables

Revision ID: a1b2c3d4e5f6
Revises: f61c96c5c275
Create Date: 2026-08-18 14:20:00.000000

"""
from datetime import datetime, timezone
from typing import Sequence, Union
from uuid import UUID

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'f61c96c5c275'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SQLITE_BATCH_NAMING_CONVENTION = {
    "fk": "fk_%(table_name)s_%(column_0_name)s",
}


def _backfill_study_patients() -> None:
    """Create one patient per legacy identifier and link every existing study."""
    bind = op.get_bind()
    studies = sa.table(
        "studies",
        sa.column("id", sa.Integer()),
        sa.column("patient_public_id", sa.Uuid()),
        sa.column("patient_id", sa.Integer()),
    )
    patients = sa.table(
        "patients",
        sa.column("id", sa.Integer()),
        sa.column("public_id", sa.Uuid()),
        sa.column("pseudonym", sa.String()),
        sa.column("status", sa.String()),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )

    legacy_ids = {
        UUID(str(value))
        for value in bind.execute(
            sa.select(studies.c.patient_public_id).where(
                studies.c.patient_public_id.is_not(None)
            )
        ).scalars()
    }
    now = datetime.now(timezone.utc)
    for public_id in sorted(legacy_ids, key=str):
        bind.execute(
            patients.insert().values(
                public_id=public_id,
                pseudonym=f"MIG-{public_id}",
                status="active",
                created_at=now,
                updated_at=now,
            )
        )

    patient_rows = bind.execute(
        sa.select(patients.c.id, patients.c.public_id)
    ).all()
    patient_ids = {UUID(str(row.public_id)): row.id for row in patient_rows}
    for public_id in legacy_ids:
        bind.execute(
            studies.update()
            .where(studies.c.patient_public_id == public_id)
            .values(patient_id=patient_ids[public_id])
        )

    orphan_count = bind.scalar(
        sa.select(sa.func.count())
        .select_from(studies)
        .where(studies.c.patient_public_id.is_not(None), studies.c.patient_id.is_(None))
    )
    if orphan_count:
        raise RuntimeError(f"patient backfill left {orphan_count} studies unlinked")


def upgrade() -> None:
    # 1. Create patients table
    op.create_table(
        'patients',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('public_id', sa.Uuid(), nullable=False),
        sa.Column('pseudonym', sa.String(length=128), nullable=False),
        sa.Column('dob', sa.Date(), nullable=True),
        sa.Column('gender', sa.String(length=32), nullable=True),
        sa.Column('diagnosis', sa.String(length=255), nullable=True),
        sa.Column('diagnosis_location', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='active'),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_patients_public_id'), 'patients', ['public_id'], unique=True)
    op.create_index(op.f('ix_patients_pseudonym'), 'patients', ['pseudonym'], unique=True)

    # 2. Create assignments table
    op.create_table(
        'assignments',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('doctor_id', sa.Integer(), nullable=False),
        sa.Column('patient_id', sa.Integer(), nullable=False),
        sa.Column('assigned_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['doctor_id'], ['users.id']),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('doctor_id', 'patient_id', name='uq_doctor_patient_assignment')
    )
    op.create_index(op.f('ix_assignments_doctor_id'), 'assignments', ['doctor_id'], unique=False)
    op.create_index(op.f('ix_assignments_patient_id'), 'assignments', ['patient_id'], unique=False)

    # 3. Add patient_id to studies
    with op.batch_alter_table(
        'studies', naming_convention=SQLITE_BATCH_NAMING_CONVENTION
    ) as batch_op:
        batch_op.add_column(sa.Column('patient_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            'fk_studies_patient_id', 'patients', ['patient_id'], ['id']
        )
        batch_op.create_index(op.f('ix_studies_patient_id'), ['patient_id'], unique=False)

    # 4. Materialize the legacy study identifier as the canonical patient relation.
    _backfill_study_patients()


def downgrade() -> None:
    with op.batch_alter_table(
        'studies', naming_convention=SQLITE_BATCH_NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_constraint('fk_studies_patient_id', type_='foreignkey')
        batch_op.drop_index(op.f('ix_studies_patient_id'))
        batch_op.drop_column('patient_id')

    op.drop_index(op.f('ix_assignments_patient_id'), table_name='assignments')
    op.drop_index(op.f('ix_assignments_doctor_id'), table_name='assignments')
    op.drop_table('assignments')

    op.drop_index(op.f('ix_patients_pseudonym'), table_name='patients')
    op.drop_index(op.f('ix_patients_public_id'), table_name='patients')
    op.drop_table('patients')
