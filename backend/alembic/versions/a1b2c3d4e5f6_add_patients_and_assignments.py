"""add patients and assignments tables

Revision ID: a1b2c3d4e5f6
Revises: f61c96c5c275
Create Date: 2026-08-18 14:20:00.000000

"""
from typing import Sequence, Union
from datetime import datetime, timezone

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

    # 4. Canonicalize every legacy study's patient_public_id into patients and
    # link all studies sharing that identifier to the same patient row.
    bind = op.get_bind()
    metadata = sa.MetaData()
    patients = sa.Table('patients', metadata, autoload_with=bind)
    studies = sa.Table('studies', metadata, autoload_with=bind)
    legacy_studies = bind.execute(
        sa.select(
            studies.c.id,
            studies.c.patient_public_id,
            studies.c.created_at,
            studies.c.updated_at,
        ).where(studies.c.patient_id.is_(None))
    ).mappings().all()

    patient_ids_by_public_id: dict[object, int] = {}
    for study in legacy_studies:
        public_id = study['patient_public_id']
        if public_id is None:
            continue
        patient_id = patient_ids_by_public_id.get(public_id)
        if patient_id is None:
            existing_id = bind.execute(
                sa.select(patients.c.id).where(patients.c.public_id == public_id)
            ).scalar_one_or_none()
            if existing_id is None:
                created_at = study['created_at'] or datetime.now(timezone.utc)
                updated_at = study['updated_at'] or created_at
                result = bind.execute(
                    patients.insert().values(
                        public_id=public_id,
                        pseudonym=f"LEGACY-{public_id}",
                        status='active',
                        created_at=created_at,
                        updated_at=updated_at,
                    )
                )
                patient_id = int(result.inserted_primary_key[0])
            else:
                patient_id = int(existing_id)
            patient_ids_by_public_id[public_id] = patient_id

        bind.execute(
            studies.update()
            .where(studies.c.id == study['id'])
            .values(patient_id=patient_id)
        )


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
