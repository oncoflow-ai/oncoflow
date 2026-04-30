<!-- GSD:project-start source:PROJECT.md -->
## Project

**OncoFlow**

OncoFlow is an MRI-focused AI backend for detecting, segmenting, and tracking bone-related tumors across serial scans. It is being built for both research workflows and clinical-facing review, starting with robust single-scan tumor segmentation and expanding into paired longitudinal comparison, structured reporting, and eventually richer clinical context integration.

**Core Value:** Deliver reliable, quantitative tumor tracking from MRI scans so users can move from subjective visual comparison to reproducible lesion measurements.

### Constraints

- **Modality**: MRI only — focus the initial system on one imaging modality for better reliability
- **Scope**: Paired longitudinal comparison first — two-scan workflows are the shortest path to useful interval analysis
- **Architecture**: Backend-first delivery — the key outcome is a dependable analysis pipeline, not UI polish
- **Deployment**: Research and pilot clinical use — operational guardrails matter, but enterprise integration can wait
- **Product strategy**: Stage single-scan before longitudinal — segmentation quality is the foundation for all later milestones
- **Data handling**: DICOM in, NIfTI-centered processing — preserve traceability while standardizing internal computation
- **Privacy/security**: Medical data requires de-identification, auditability, and access control from the start even if full production hardening is deferred
<!-- GSD:project-end -->

<!-- GSD:stack-start source:STACK.md -->
## Technology Stack

Technology stack not yet documented. Will populate after codebase mapping or first phase.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
