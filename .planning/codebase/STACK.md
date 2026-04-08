# Technology Stack

**Analysis Date:** 2026-04-08

## Languages

**Primary:**
- TypeScript 5.4+ - Frontend application (`frontend/src/`)
- Python 3.12+ - ML exploration and data processing (`ml/exploration/`)

**Secondary:**
- JavaScript (ES2020) - Build configuration (`frontend/*.config.js`)

## Runtime

**Environment:**
- Node.js v22.15.0 (Frontend)
- Python 3.12.7 (ML)

**Package Manager:**
- npm - Frontend dependencies
- Lockfile: `frontend/package-lock.json` present
- pip - Python dependencies (`ml/exploration/requirements.txt`)

## Frameworks

**Core:**
- React 18.3.0 - UI library
- React Router DOM 6.22.0 - Client-side routing
- Vite 5.2.0 - Build tool and dev server

**Testing:**
- Vitest 1.4.0 - Test runner
- @testing-library/react 15.0.0 - React component testing
- @testing-library/user-event 14.5.0 - User interaction simulation
- jsdom 24.0.0 - DOM environment for tests

**Build/Dev:**
- Vite 5.2.0 - Bundler and dev server with HMR
- TypeScript 5.4.0 - Type checking and compilation
- ESLint 8.57.0 - Linting
- PostCSS 8.4.0 - CSS processing
- Autoprefixer 10.4.0 - CSS vendor prefixing

## Key Dependencies

**Critical:**
- axios 1.6.0 - HTTP client for API communication (`frontend/src/api/client.ts`)
- @tanstack/react-query 5.0.0 - Server state management and caching
- zustand 4.5.0 - Client state management with persistence (`frontend/src/store/authStore.ts`)
- tailwindcss 3.4.0 - Utility-first CSS framework
- react-router-dom 6.22.0 - Declarative routing

**Infrastructure:**
- recharts 2.12.0 - Data visualization charts
- lucide-react 0.363.0 - Icon library
- class-variance-authority 0.7.0 - CVA utility for component variants
- clsx 2.1.0 - Conditional className utility
- tailwind-merge 2.2.0 - Tailwind class merging

**ML/Scientific (Python):**
- torch 2.2+ - Deep learning framework
- torchvision 0.17+ - Computer vision models
- transformers 4.47+ - HuggingFace models (MedGemma-1.5)
- nnunetv2 2.4+ - Medical image segmentation
- numpy 1.26+ - Numerical computing
- pandas 2.2+ - Data manipulation
- scikit-learn 1.4+ - Classical ML algorithms
- pydicom 2.4+ - DICOM file I/O
- nibabel 5.2+ - NIfTI file I/O
- SimpleITK 2.3+ - Medical image processing
- matplotlib 3.8+ - Data visualization
- seaborn 0.13+ - Statistical visualization
- jupyterlab 4.0+ - Interactive notebooks

## Configuration

**Environment:**
- Vite environment variables via `import.meta.env`
- `VITE_API_URL` - Backend API base URL (defaults to `http://localhost:8000`)
- Environment file: `frontend/.env.local` present (not tracked in git)

**Build:**
- `frontend/tsconfig.json` - TypeScript compiler options (ES2020, strict mode, path aliases)
- `frontend/tsconfig.node.json` - Node-specific TypeScript config
- `frontend/vite.config.ts` - Vite bundler config with React plugin and path aliases
- `frontend/tailwind.config.ts` - Tailwind CSS design tokens (custom color palette, fonts)
- `frontend/postcss.config.js` - PostCSS with Tailwind and Autoprefixer

**TypeScript:**
- Target: ES2020
- Module: ESNext
- Module resolution: bundler
- Strict mode enabled
- Path alias: `@/*` → `src/*`
- JSX: react-jsx

**Python ML:**
- `ml/exploration/requirements.txt` - Pinned versions for reproducibility
- Notebooks require optional `HF_TOKEN` for MedGemma access
- nnU-Net requires environment variables: `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`

## Platform Requirements

**Development:**
- Node.js 22+ (inferred from v22.15.0 in use)
- Python 3.12+ for ML exploration
- Optional: CUDA-capable GPU for accelerated ML inference (notebooks fall back to CPU)
- Optional: dcm2niix installed via package manager for DICOM conversion

**Production:**
- Planned deployment target: AWS (per `IMPLEMENTATION_PLAN.md`)
- Planned architecture: FastAPI backend, PostgreSQL, Redis/Celery workers, ML microservices
- Current state: Frontend-only implementation with mock API layer

---

*Stack analysis: 2026-04-08*
