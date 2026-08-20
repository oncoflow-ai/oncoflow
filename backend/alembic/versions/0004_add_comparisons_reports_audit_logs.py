"""add comparisons, reports, and audit_logs tables

Revision ID: 0004_add_comparisons_reports_audit_logs
Revises: a1b2c3d4e5f6
Create Date: 2026-08-20 12:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0004_add_comparisons_reports_audit_logs'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create comparisons table
    op.create_table(
        'comparisons',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('public_id', sa.Uuid(), nullable=False),
        sa.Column('study_a_id', sa.Integer(), nullable=False),
        sa.Column('study_b_id', sa.Integer(), nullable=False),
        sa.Column('volume_a', sa.Float(), nullable=True),
        sa.Column('volume_b', sa.Float(), nullable=True),
        sa.Column('delta_cm3', sa.Float(), nullable=True),
        sa.Column('pct_change', sa.Float(), nullable=True),
        sa.Column('dice_overlap', sa.Float(), nullable=True),
        sa.Column('hd95_mm', sa.Float(), nullable=True),
        sa.Column('growth_rate_cm3_per_day', sa.Float(), nullable=True),
        sa.Column('interpretation_flag', sa.String(length=255), nullable=True),
        sa.Column('recist_ratio', sa.Float(), nullable=True),
        sa.Column('vol_delta_ci_half_cm3', sa.Float(), nullable=True),
        sa.Column('registration_ncc', sa.Float(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['study_a_id'], ['studies.id']),
        sa.ForeignKeyConstraint(['study_b_id'], ['studies.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_comparisons_public_id'), 'comparisons', ['public_id'], unique=True)
    op.create_index(op.f('ix_comparisons_study_a_id'), 'comparisons', ['study_a_id'], unique=False)
    op.create_index(op.f('ix_comparisons_study_b_id'), 'comparisons', ['study_b_id'], unique=False)

    # 2. Create reports table
    op.create_table(
        'reports',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('public_id', sa.Uuid(), nullable=False),
        sa.Column('patient_id', sa.Integer(), nullable=False),
        sa.Column('comparison_id', sa.Integer(), nullable=True),
        sa.Column('pdf_artifact_id', sa.Integer(), nullable=True),
        sa.Column('signature', sa.Text(), nullable=True),
        sa.Column('generated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.id']),
        sa.ForeignKeyConstraint(['comparison_id'], ['comparisons.id']),
        sa.ForeignKeyConstraint(['pdf_artifact_id'], ['artifacts.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_reports_public_id'), 'reports', ['public_id'], unique=True)
    op.create_index(op.f('ix_reports_patient_id'), 'reports', ['patient_id'], unique=False)
    op.create_index(op.f('ix_reports_comparison_id'), 'reports', ['comparison_id'], unique=False)
    op.create_index(op.f('ix_reports_pdf_artifact_id'), 'reports', ['pdf_artifact_id'], unique=False)

    # 3. Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('actor_id', sa.String(length=255), nullable=False),
        sa.Column('action', sa.String(length=128), nullable=False),
        sa.Column('resource_id', sa.String(length=255), nullable=False),
        sa.Column('details', sa.JSON(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_audit_logs_actor_id'), 'audit_logs', ['actor_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_action'), 'audit_logs', ['action'], unique=False)
    op.create_index(op.f('ix_audit_logs_resource_id'), 'audit_logs', ['resource_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_timestamp'), 'audit_logs', ['timestamp'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_audit_logs_timestamp'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_resource_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_action'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_actor_id'), table_name='audit_logs')
    op.drop_table('audit_logs')

    op.drop_index(op.f('ix_reports_pdf_artifact_id'), table_name='reports')
    op.drop_index(op.f('ix_reports_comparison_id'), table_name='reports')
    op.drop_index(op.f('ix_reports_patient_id'), table_name='reports')
    op.drop_index(op.f('ix_reports_public_id'), table_name='reports')
    op.drop_table('reports')

    op.drop_index(op.f('ix_comparisons_study_b_id'), table_name='comparisons')
    op.drop_index(op.f('ix_comparisons_study_a_id'), table_name='comparisons')
    op.drop_index(op.f('ix_comparisons_public_id'), table_name='comparisons')
    op.drop_table('comparisons')
