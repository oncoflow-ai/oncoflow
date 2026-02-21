# OncoFlow – ML Exploration

Standalone research notebooks for **Phase 4** of the OncoFlow project. Explores data processing pipeline options and segmentation model comparison using real P01 patient data — **not part of the production app**.

## Data (P01/)
```
P01/
├── BraTS/
│   ├── baseline/   # t1.nii.gz, t1c.nii.gz, t2.nii.gz, fla.nii.gz
│   ├── fu1/ … fu4/ # same structure for 4 follow-up timepoints
├── DICOM/          # Raw DICOM series (T1C, T1w, T2w, FLR, RTP)
├── tumor segmentation/  # GT masks: P01_tumor_mask_{tp}.nii.gz
├── P01_CT.nii.gz
├── P01_RTP.nii.gz
└── P01_brain_mask.nii.gz
```

## Setup

```bash
cd ml/exploration
pip install -r requirements.txt

# Optional: brew install dcm2niix (for Option A2 in notebook 01)

# Optional (for MedGemma): export HF_TOKEN=hf_xxxx
# Optional (for SAM3):     pip install git+https://github.com/facebookresearch/sam3.git
# Optional (for SAM2):     pip install git+https://github.com/facebookresearch/sam2.git

jupyter lab
```

## Notebooks

| # | Notebook | Goal |
|---|----------|------|
| 00 | `00_setup_and_data.ipynb` | Environment check, data exploration, GT volume timeline |
| 01 | `01_preprocessing_comparison.ipynb` | DICOM→NIfTI options: SimpleITK, dcm2niix, N4+zscore, isotropic |
| 02 | `02_nnunet_exploration.ipynb` | nnU-Net v2 dataset prep, fingerprinting, inference |
| 03 | `03_medgemma_exploration.ipynb` | MedGemma-1.5 / LLaVA-Med slice-wise inference |
| 04 | `04_sam3_exploration.ipynb` | SAM3 / SAM2 automatic + prompted segmentation |
| 05 | `05_ensemble_strategies.ipynb` | Majority vote, STAPLE, union, intersection, weighted |
| 06 | `06_benchmark_report.ipynb` | Leaderboard, radar chart, HTML report |

## Notes

- **GPU not required** — All notebooks run in stub mode without GPU/tokens (clearly flagged)
- **MedGemma access** — Set `HF_TOKEN` env var; otherwise LLaVA-Med is used automatically
- **nnU-Net weights** — No official BraTS checkpoint; notebook sets up training structure and plans the config
- **SAM3 API** — Falls back to SAM2 if SAM3 is not installed
