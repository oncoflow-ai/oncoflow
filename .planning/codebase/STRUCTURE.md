# Codebase Structure

**Analysis Date:** 2024-04-08

## Directory Layout

```
oncoflow/
├── frontend/               # React SPA with TypeScript
│   ├── src/
│   │   ├── api/            # Backend API client functions
│   │   ├── components/     # React components (atomic design)
│   │   │   ├── layout/     # TopNav, page wrappers
│   │   │   ├── patient/    # Patient-specific components
│   │   │   ├── scan/       # Scan/imaging components
│   │   │   └── shared/     # Reusable UI primitives
│   │   ├── data/           # Mock data for development
│   │   ├── lib/            # Utility functions
│   │   ├── pages/          # Route-level components
│   │   ├── store/          # Zustand state management
│   │   ├── test/           # Test files and setup
│   │   ├── types/          # TypeScript type definitions
│   │   ├── main.tsx        # App entry point
│   │   └── router.tsx      # React Router configuration
│   ├── index.html          # HTML entry point
│   ├── package.json        # Dependencies and scripts
│   ├── tsconfig.json       # TypeScript compiler config
│   ├── vite.config.ts      # Vite bundler config
│   └── tailwind.config.ts  # Tailwind CSS config
├── ml/                     # ML exploration (not production)
│   └── exploration/
│       ├── notebooks/      # Jupyter notebooks for research
│       ├── outputs/        # Model outputs and artifacts
│       ├── utils/          # Python helper modules
│       └── requirements.txt
├── data/                   # Patient data (not committed)
│   └── P01/                # Example patient imaging data
├── docs/                   # Project documentation
│   └── superpowers/        # Agent-generated specs/plans
├── project-documents/      # Original project documents (PDFs)
├── .planning/              # GSD codebase mapping outputs
├── .github/                # GitHub configuration
│   ├── agents/             # AI agent configurations
│   └── skills/             # GSD skill definitions
├── IMPLEMENTATION_PLAN.md  # Phase-based development roadmap
└── README.md               # Project overview
```

## Directory Purposes

**`frontend/src/api/`:**
- Purpose: HTTP client and backend communication
- Contains: API function modules grouped by domain (patients, scans, reports, mri)
- Key files: 
  - `client.ts`: Axios instance with base URL configuration
  - `patients.ts`: Patient CRUD operations
  - `scans.ts`: Scan retrieval functions
  - `reports.ts`: AI summary fetching
  - `mri.ts`: MRI viewer URL generation

**`frontend/src/components/`:**
- Purpose: Reusable React components organized by domain
- Contains: TSX files with component logic and Tailwind styling
- Key files:
  - `layout/TopNav.tsx`: Application header with navigation
  - `shared/StatBlock.tsx`: Metric display card
  - `shared/ErrorBanner.tsx`: Error state with retry
  - `shared/EmptyState.tsx`: Zero-data placeholder
  - `patient/PatientTable.tsx`: Patient list table
  - `scan/VolumeChart.tsx`: Recharts visualization
  - `scan/ImagingHistory.tsx`: Scan timeline table

**`frontend/src/pages/`:**
- Purpose: Route-level components (one per URL path)
- Contains: Full-page components that compose smaller components
- Key files:
  - `AuthPage.tsx`: Login form
  - `DashboardPage.tsx`: Patient list view
  - `PatientDetailPage.tsx`: Individual patient view with scans

**`frontend/src/store/`:**
- Purpose: Global client-side state management
- Contains: Zustand stores with middleware
- Key files:
  - `authStore.ts`: Authentication state with sessionStorage persistence

**`frontend/src/types/`:**
- Purpose: TypeScript type definitions
- Contains: Domain model interfaces
- Key files:
  - `index.ts`: Patient, Scan, Summary, Physician, MriUrl interfaces

**`frontend/src/lib/`:**
- Purpose: Shared utility functions
- Contains: Pure helper functions
- Key files:
  - `utils.ts`: Date formatting, volume calculations, Tailwind class merging

**`frontend/src/data/`:**
- Purpose: Mock data for frontend-only development
- Contains: Static data matching TypeScript interfaces
- Key files:
  - `mockData.ts`: Sample patients, scans, summaries

**`frontend/src/test/`:**
- Purpose: Test files and testing utilities
- Contains: Vitest test files, setup configuration
- Key files:
  - `setup.ts`: Vitest global configuration
  - `*.test.tsx`: Component and integration tests

**`ml/exploration/`:**
- Purpose: Research notebooks for model evaluation (isolated from production)
- Contains: Jupyter notebooks, Python utilities, model experiment outputs
- Key files:
  - `notebooks/00_setup_and_data.ipynb`: Environment and data exploration
  - `notebooks/02_nnunet_exploration.ipynb`: nnU-Net segmentation experiments
  - `notebooks/03_medgemma_exploration.ipynb`: MedGemma LLM experiments
  - `notebooks/04_sam3_exploration.ipynb`: SAM3 segmentation experiments
  - `notebooks/05_ensemble_strategies.ipynb`: Multi-model fusion
  - `utils/dicom_utils.py`: DICOM/NIfTI conversion helpers

**`data/`:**
- Purpose: Patient imaging data storage (not version controlled)
- Contains: DICOM series, NIfTI files, segmentation masks
- Key subdirectories:
  - `P01/BraTS/`: BraTS-formatted multi-timepoint data
  - `P01/DICOM/`: Raw DICOM series
  - `P01/tumor segmentation/`: Ground truth masks

**`docs/superpowers/`:**
- Purpose: Agent-generated documentation and specifications
- Contains: Planning documents, design specs
- Key files:
  - `specs/2026-03-15-frontend-design.md`: UI/UX specification
  - `plans/2026-03-15-frontend-scaffold.md`: Implementation plan

**`.planning/codebase/`:**
- Purpose: GSD codebase analysis documents
- Contains: Architecture, conventions, testing patterns
- Generated by: `/gsd-map-codebase` command

## Key File Locations

**Entry Points:**
- `frontend/index.html`: HTML document root
- `frontend/src/main.tsx`: React application initialization
- `frontend/src/router.tsx`: Client-side routing configuration

**Configuration:**
- `frontend/package.json`: NPM dependencies and scripts
- `frontend/vite.config.ts`: Build tool configuration, path aliases, test setup
- `frontend/tsconfig.json`: TypeScript compiler options
- `frontend/tailwind.config.ts`: Tailwind CSS theme customization
- `ml/exploration/requirements.txt`: Python ML dependencies

**Core Logic:**
- `frontend/src/api/`: All backend communication
- `frontend/src/store/authStore.ts`: Authentication state
- `frontend/src/components/`: UI component library
- `frontend/src/pages/`: Page-level orchestration

**Testing:**
- `frontend/src/test/setup.ts`: Vitest configuration
- `frontend/src/test/*.test.tsx`: Component tests
- `frontend/src/test/*.test.ts`: Utility function tests

## Naming Conventions

**Files:**
- React components: `PascalCase.tsx` (e.g., `PatientTable.tsx`)
- Utilities: `camelCase.ts` (e.g., `utils.ts`)
- Test files: `*.test.tsx` or `*.test.ts` (e.g., `PatientTable.test.tsx`)
- Type definitions: `index.ts` within `types/` directory
- Store modules: `camelCase.ts` with "Store" suffix (e.g., `authStore.ts`)

**Directories:**
- Lowercase, hyphenated if multi-word: `components/`, `api/`, `project-documents/`
- Component subdirectories: domain-based grouping (`patient/`, `scan/`, `shared/`)

**Imports:**
- Path alias `@/` maps to `frontend/src/` (configured in `vite.config.ts`)
- Example: `import { useAuthStore } from '@/store/authStore'`

## Where to Add New Code

**New Feature:**
- Primary code: `frontend/src/pages/NewFeaturePage.tsx` (if full page) or `frontend/src/components/{domain}/NewComponent.tsx`
- Tests: `frontend/src/test/NewFeature.test.tsx`
- API calls: `frontend/src/api/newFeature.ts`
- Types: Add to `frontend/src/types/index.ts`
- Routes: Register in `frontend/src/router.tsx`

**New Component/Module:**
- Implementation: 
  - Shared/reusable: `frontend/src/components/shared/ComponentName.tsx`
  - Domain-specific: `frontend/src/components/{patient|scan|layout}/ComponentName.tsx`
- Tests: `frontend/src/test/ComponentName.test.tsx`

**Utilities:**
- Shared helpers: `frontend/src/lib/utils.ts` (or create new module like `frontend/src/lib/calculations.ts`)
- Data formatting: Add to existing functions in `frontend/src/lib/utils.ts`

**New API Endpoint:**
- Client function: `frontend/src/api/{domain}.ts` (group by resource type)
- Return type: Define interface in `frontend/src/types/index.ts`

**New Page Route:**
- Component: `frontend/src/pages/{Feature}Page.tsx`
- Route config: Add to `router.tsx` routes array
- Navigation: Update `TopNav.tsx` or add links in relevant components

**New ML Experiment:**
- Notebook: `ml/exploration/notebooks/{NN}_{name}.ipynb`
- Utilities: `ml/exploration/utils/{module}.py`
- Outputs: `ml/exploration/outputs/{experiment_name}/`

## Special Directories

**`frontend/node_modules/`:**
- Purpose: NPM package cache
- Generated: Yes (via `npm install`)
- Committed: No (listed in `.gitignore`)

**`ml/exploration/outputs/`:**
- Purpose: Model training artifacts, predictions, preprocessed data
- Generated: Yes (by Jupyter notebooks)
- Committed: No (large binary files, reproducible)

**`data/`:**
- Purpose: Patient imaging data (DICOM, NIfTI, masks)
- Generated: No (provided dataset)
- Committed: No (sensitive medical data, large files)

**`.planning/`:**
- Purpose: GSD agent outputs and codebase analysis
- Generated: Yes (by `/gsd-map-codebase` and related commands)
- Committed: Yes (provides context for future development)

**`.github/`:**
- Purpose: GitHub-specific configuration
- Generated: No (manually configured)
- Committed: Yes (CI/CD workflows, agent configs)

---

*Structure analysis: 2024-04-08*
