# OncoFlow

## What This Is

OncoFlow is an MRI-focused AI backend for detecting, segmenting, and tracking bone-related tumors across serial scans. It is being built for both research workflows and clinical-facing review, starting with robust single-scan tumor segmentation and expanding into paired longitudinal comparison, structured reporting, and eventually richer clinical context integration.

## Core Value

Deliver reliable, quantitative tumor tracking from MRI scans so users can move from subjective visual comparison to reproducible lesion measurements.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] MRI studies can be ingested and normalized into analysis-ready volumes
- [ ] Single-scan tumor detection returns lesion masks and structured measurements
- [ ] Paired scans can be aligned and compared lesion-by-lesion across time
- [ ] Backend APIs expose processing status, result artifacts, and clinical-ready summaries
- [ ] Unsupported or incomplete MRI inputs fail with actionable validation errors

### Out of Scope

- CT or other non-MRI modalities — v1 is MRI-only to avoid multimodal complexity
- Arbitrary multi-timepoint longitudinal history — start with paired scan comparison first
- PACS/native hospital system integration — defer enterprise workflow integration until core pipeline works
- Treatment recommendation or clinical decision support — provide measurements and change analysis only
- Full regulatory/compliance hardening — research and pilot deployment come before production controls
- RAG-based patient-context summaries in v1 — keep early milestones centered on imaging reliability first

## Context

The immediate project is a new backend initiative inside the existing OncoFlow repository. There is prior frontend and planning work in the repo, but this initialization is centered on the backend pipeline required to ingest MRI scans, detect all found tumors, segment them, and output tumor-specific measurements and interval changes.

The development path is intentionally staged:
- Milestone 1 establishes robust per-scan segmentation because all downstream longitudinal analysis depends on it.
- Milestone 2 adds registration and change detection between a baseline and follow-up scan.
- Milestone 3 packages the results for API consumers and clinical workflow outputs such as PDF reports.

Primary users are both research users and clinical reviewers. In practice that means the backend should produce machine-readable outputs for experimentation while also supporting interpretable summaries, confidence/QC signals, and durable result artifacts for review.

The HLD also clarifies a few architectural anchors that should guide implementation:
- Clinical inputs arrive primarily as DICOM and are converted into NIfTI for internal numerical processing.
- Segmentation backbone choice should be benchmark-driven. Current research suggests nnU-Net v2 with residual-encoder presets is the leading baseline, with MedNeXt, MONAI DynUNet/SegResNetDS, and SwinUNETR-V2 as challengers and MedSAM2 as an interactive/QC assistant rather than the default automatic model.
- The intended backend stack is Python/FastAPI with asynchronous workers, PostgreSQL for metadata, Redis/Celery for task orchestration, object storage for imaging artifacts, and PDF generation for clinical summaries.
- The long-term product direction includes patient-context summarization via RAG, assignment-aware access control, and privacy-first handling of medical data.

## Constraints

- **Modality**: MRI only — focus the initial system on one imaging modality for better reliability
- **Scope**: Paired longitudinal comparison first — two-scan workflows are the shortest path to useful interval analysis
- **Architecture**: Backend-first delivery — the key outcome is a dependable analysis pipeline, not UI polish
- **Deployment**: Research and pilot clinical use — operational guardrails matter, but enterprise integration can wait
- **Product strategy**: Stage single-scan before longitudinal — segmentation quality is the foundation for all later milestones
- **Data handling**: DICOM in, NIfTI-centered processing — preserve traceability while standardizing internal computation
- **Privacy/security**: Medical data requires de-identification, auditability, and access control from the start even if full production hardening is deferred

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| MRI-only initial scope | Narrowing modality reduces preprocessing and modeling complexity | — Pending |
| Paired scan comparison before full timeline support | Baseline/follow-up comparison is the fastest useful longitudinal workflow | — Pending |
| Start with a narrow MRI input contract | Controlled input assumptions reduce early ingestion and normalization failure modes | — Pending |
| Use DICOM ingestion with NIfTI-centered internal processing | Matches medical workflow inputs while keeping ML processing efficient and standardized | — Pending |
| Use nnU-Net v2 residual encoder as the first benchmark baseline | Current evidence still favors nnU-Net as the safest production starting point for MRI medical segmentation | — Pending |
| Benchmark 2D/2.5D and challenger models before locking production backbone | MRI anisotropy and lesion-size distribution may favor a different model family on local data | — Pending |
| Reserve foundation models for challenger and QC roles first | Promptable models like MedSAM2 are promising but less deterministic for serial quantitative workflows | — Pending |
| Backend returns both raw artifacts and structured lesion data | Research users need machine-readable outputs while clinical users need interpretable summaries | — Pending |
| Confidence/QC outputs are part of core results | Clinical and research review both need visibility into uncertain or low-quality segmentations | — Pending |
| Clinical workflow features come after segmentation and comparison foundations | Reporting and integration are only valuable once core measurements are trustworthy | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check -> still the right priority?
3. Audit Out of Scope -> reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-11 after initialization*
