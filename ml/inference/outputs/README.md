# Inference Outputs

This directory contains results from running the OncoFlow inference pipeline.

## Structure

```
outputs/
├── cli_runs/          # CLI command outputs (segment, longitudinal)
│   ├── baseline_vs_fu1/
│   │   ├── comparison.json
│   │   ├── baseline_ensemble_mask.nii.gz
│   │   └── followup_warped_mask.nii.gz
│   └── ...
└── p01_benchmark/     # P01 benchmark harness results
    ├── segmentation_leaderboard.csv
    ├── longitudinal_results.csv
    ├── volume_curve.png
    └── summary.json
```

## Running CLI Examples

### Single-timepoint Segmentation
```bash
python -m ml.inference.cli segment \
  --input data/P01/BraTS/baseline/t1c.nii.gz \
  --out ml/inference/outputs/cli_runs/baseline_seg \
  --verbose
```

### Longitudinal Comparison (with GT masks)
```bash
python -m ml.inference.cli longitudinal \
  --baseline data/P01/BraTS/baseline/t1c.nii.gz \
  --followup data/P01/BraTS/fu1/t1c.nii.gz \
  --baseline-mask "data/P01/tumor segmentation/P01_tumor_mask_baseline.nii.gz" \
  --followup-mask "data/P01/tumor segmentation/P01_tumor_mask_fu1.nii.gz" \
  --out ml/inference/outputs/cli_runs/baseline_vs_fu1 \
  --verbose
```

### P01 Benchmark
```bash
python -m ml.inference.cli p01-benchmark \
  --data data/P01 \
  --out ml/inference/outputs/p01_benchmark \
  --use-gt-masks \
  --verbose
```

## Output Files

### `comparison.json`
Full longitudinal comparison result with:
- Volume metrics (cm³, delta, % change)
- Overlap metrics (Dice, HD95)
- RECIST diameters and ratio
- Registration quality (NCC before/after)
- Interpretation flag (PD/stable/response)

### `segmentation.json`
Per-timepoint segmentation with:
- Per-model volumes and runtime
- Ensemble volume
- Panel agreement score
- ROI bbox used for speed optimization

### `.nii.gz` files
- `*_ensemble_mask.nii.gz` - Final fused segmentation
- `*_warped_mask.nii.gz` - Follow-up mask registered to baseline space
- `*_registered_volume.nii.gz` - Follow-up intensity registered to baseline

## Notes

- All outputs are gitignored except this README and `.gitkeep` files
- Results are cached under `~/.oncoflow/cache/` for faster re-runs
- Use `--no-cache` flag to force re-segmentation
