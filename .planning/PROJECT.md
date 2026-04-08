# OncoFlow: AI-Powered Bone Lesion Tracking System

## What This Is

OncoFlow is a clinical AI pipeline that automatically detects, segments, and tracks bone lesions (benign and malignant) across serial MRI scans. The system registers scans across variable timepoints, matches corresponding tumors automatically, and quantifies interval change through volume metrics, shape analysis, growth rates, and visual heatmaps. Results are delivered via a standalone web interface for radiologists and oncologists at a single-institution clinical pilot.

## Core Value

**Automated, quantitative tumor progression tracking that transforms subjective "eyeball" assessments into objective, reproducible measurements for clinical decision-making.**

If everything else fails, the system must accurately segment bone lesions and report measurable change between timepoints.

## Requirements

### Validated

<!-- Shipped and confirmed valuable (existing codebase functionality) -->

- ✓ React-based web UI with TypeScript and Tailwind CSS — existing
- ✓ Patient management interface with CRUD operations — existing
- ✓ Scan timeline visualization with volume charts — existing
- ✓ Mock API integration layer with axios + React Query — existing
- ✓ Authentication flow with session persistence (Zustand) — existing
- ✓ Component library following atomic design principles — existing

### Active

<!-- Current scope — building toward these -->

#### Milestone 1: Single-Scan Segmentation

- [ ] **SEG-01**: Load and preprocess MRI DICOM files
- [ ] **SEG-02**: Foundation model integration for bone lesion segmentation (any benign/malignant)
- [ ] **SEG-03**: Multi-lesion detection within single anatomical region
- [ ] **SEG-04**: Segmentation quality metrics and confidence scores
- [ ] **SEG-05**: Export segmentation masks in standard format (NIfTI)
- [ ] **SEG-06**: Interactive segmentation refinement UI (optional corrections)

#### Milestone 2: Longitudinal Registration & Change Detection

- [ ] **REG-01**: Rigid registration pipeline for MRI scan alignment
- [ ] **REG-02**: Automated lesion correspondence matching across timepoints
- [ ] **REG-03**: Volume change quantification (absolute and percentage)
- [ ] **REG-04**: Shape metrics calculation (sphericity, surface area, irregularity)
- [ ] **REG-05**: Tumor growth rate computation (velocity in mm³/day)
- [ ] **REG-06**: Visual change heatmaps overlaid on scans
- [ ] **REG-07**: Handle variable timepoint counts per patient (2 to N scans)

#### Milestone 3: Clinical Workflow Integration

- [ ] **WEB-01**: Upload MRI scans via web interface (DICOM drag-and-drop)
- [ ] **WEB-02**: Processing pipeline orchestration (seg → reg → change detection)
- [ ] **WEB-03**: Interactive viewer for segmentation results and heatmaps
- [ ] **WEB-04**: PDF report generation (metrics + visualizations)
- [ ] **WEB-05**: Basic security: authentication, session management, input validation
- [ ] **WEB-06**: Audit logging for clinical pilot (who viewed what, when)
- [ ] **WEB-07**: Error handling and retry logic for long-running ML jobs

### Out of Scope

- **CT scan support** — MRI-only for v1, defers multimodal complexity
- **X-ray integration** — different clinical workflow and resolution constraints
- **PACS integration (DICOM C-STORE/C-FIND)** — standalone system for pilot, defers enterprise integration
- **Whole-body MRI tracking** — focused on single anatomical regions (e.g., femur, pelvis)
- **Real-time processing** — batch processing acceptable for clinical pilot timelines
- **Multi-site deployment** — single-institution pilot scope
- **FDA regulatory submission** — research/pilot validation only, not production clearance
- **HIPAA full compliance** — basic security only (not production-grade PHI handling)
- **Automated treatment recommendations** — decision support out of scope, provides data only

## Context

**Clinical Problem:**
Oncologists and radiologists currently assess tumor progression by manually comparing MRI scans side-by-side. This is subjective, time-intensive, and prone to inter-observer variability. Quantitative metrics (volume, shape, growth rate) require manual segmentation, which is rarely done in practice.

**Technical Environment:**
- **Imaging modality:** MRI (T1, T2, STIR sequences typical for bone lesions)
- **Use cases:** Post-treatment monitoring, clinical trial endpoints, surgical planning
- **Users:** Radiologists (primary reviewers) and oncologists (treatment decisions)
- **Existing codebase:** React/TypeScript frontend with mock API, ready for backend integration

**Prior Work:**
- Frontend UI prototyped with patient/scan management interface
- Mock data structure defined (Patient, Scan, Summary types)
- Component library established (charts, tables, navigation)

**Data Landscape:**
- Using foundation models (pretrained on medical imaging) rather than custom training
- MONAI, TotalSegmentator, nnU-Net are candidate frameworks
- No large annotated dataset required for initial deployment

## Constraints

- **Tech stack**: Python (ML backend), React/TypeScript (frontend) — existing frontend must integrate
- **Timeline**: Clinical pilot timeline (not production deployment) — prioritize core functionality over edge cases
- **Modality**: MRI-only — CT/X-ray support deferred to avoid multimodal complexity
- **Security**: Basic security only (auth, logging) — full HIPAA compliance deferred to production phase
- **Deployment**: Single-institution pilot — multi-site scalability not required
- **Performance**: Batch processing acceptable — real-time inference not required for clinical workflow

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Foundation models over custom training | No large annotated dataset available; pretrained models (MONAI, nnU-Net) exist for bone lesion segmentation | — Pending |
| MRI-only scope | Clinical priority is MRI for soft tissue + bone marrow visibility; CT adds multimodal complexity | — Pending |
| Staged milestones (seg → reg → clinical) | Single-scan segmentation must be robust before longitudinal tracking; crawl-walk-run approach | — Pending |
| Standalone web UI over PACS integration | Faster pilot deployment without enterprise IT dependencies; PACS deferred to production | — Pending |
| React frontend (existing) + Python ML backend | Leverage existing frontend work; Python standard for medical ML pipelines | — Pending |
| Automated lesion matching required | Multiple tumors per region need correspondence tracking; manual matching scales poorly | — Pending |
| Variable timepoint support | Clinical reality: some patients have 2 scans, others have 5+; system must handle any count | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-08 after initialization*
