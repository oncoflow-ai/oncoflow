from __future__ import annotations

from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "P01"
OUTPUT_DIR = REPO_ROOT / "frontend" / "public" / "demo-assets"

DEMO_STAGES = [
    ("baseline", "P01_tumor_mask_baseline.nii.gz", DATA_DIR / "BraTS" / "baseline" / "t1c.nii.gz", 0),
    ("fu1", "P01_tumor_mask_fu1.nii.gz", DATA_DIR / "BraTS" / "fu1" / "t1c.nii.gz", 1),
    ("fu2", "P01_tumor_mask_fu2.nii.gz", DATA_DIR / "BraTS" / "fu2" / "t1c.nii.gz", 2),
    ("fu3", "P01_tumor_mask_fu3.nii.gz", DATA_DIR / "BraTS" / "fu3" / "t1c.nii.gz", 3),
    ("fu4", "P01_tumor_mask_fu4.nii.gz", DATA_DIR / "BraTS" / "fu4" / "t1c.nii.gz", 4),
]

FRAME_SLICES = [72, 78, 82, 86, 92, 98, 102, 106, 110]


def generate_demo_slices() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_dir = DATA_DIR / "tumor segmentation"

    for stage_name, mask_file, img_path, stage_idx in DEMO_STAGES:
        mask_path = mask_dir / mask_file
        if not mask_path.exists() or not img_path.exists():
            print(f"Skipping {stage_name}: {mask_path} or {img_path} not found")
            continue

        mask_nib = nib.load(str(mask_path))
        img_nib = nib.load(str(img_path))

        mask_data = np.asarray(mask_nib.dataobj)
        img_data = np.asarray(img_nib.dataobj)

        for z in FRAME_SLICES:
            if z >= img_data.shape[2]:
                continue

            raw_slice = img_data[:, :, z].astype(float)
            raw_mask = mask_data[:, :, z]

            # Orient correctly (rot90)
            img_slice = np.rot90(raw_slice, 1)
            mask_slice = np.rot90(raw_mask, 1)

            # Normalize MRI grayscale (clip 1.0% - 99.2%)
            pos_pixels = img_slice[img_slice > 0]
            if len(pos_pixels) > 0:
                p_high = np.percentile(pos_pixels, 99.2)
                p_low = np.percentile(pos_pixels, 1.0)
                clipped = np.clip((img_slice - p_low) / max(p_high - p_low, 1e-5), 0, 1)
            else:
                clipped = np.zeros_like(img_slice)

            rgb = np.stack([clipped, clipped, clipped], axis=-1)

            # Mask overlays:
            # Label 3: enhancing tumor (rose/red tint with solid contour)
            # Label 2: edema (cyan contour)
            # Label 1: necrotic core (amber/orange)
            if mask_slice.max() > 0:
                for label, color, alpha in [
                    (2, np.array([0.05, 0.85, 0.95]), 0.28),  # Edema: Cyan
                    (1, np.array([0.95, 0.65, 0.15]), 0.40),  # Necrotic: Amber
                    (3, np.array([0.95, 0.20, 0.40]), 0.50),  # Enhancing: Rose/Red
                ]:
                    submask = mask_slice == label
                    if submask.sum() > 0:
                        for c in range(3):
                            rgb[:, :, c] = np.where(submask, rgb[:, :, c] * (1 - alpha) + color[c] * alpha, rgb[:, :, c])
                        dilated = binary_dilation(submask, iterations=1)
                        boundary = dilated & (~submask)
                        for c in range(3):
                            rgb[:, :, c] = np.where(boundary, color[c] * 0.9 + rgb[:, :, c] * 0.1, rgb[:, :, c])

                overall_mask = mask_slice > 0
                dilated_all = binary_dilation(overall_mask, iterations=1)
                outer_boundary = dilated_all & (~overall_mask)
                for c, val in enumerate([0.1, 0.95, 0.85]):  # teal/cyan highlight
                    rgb[:, :, c] = np.where(outer_boundary, val, rgb[:, :, c])

            im = Image.fromarray((rgb * 255).astype(np.uint8))
            im_resized = im.resize((448, 448), Image.Resampling.BILINEAR)

            out_file = OUTPUT_DIR / f"demo-stage-{stage_idx}-slice-{z}.png"
            im_resized.save(out_file)

            if stage_idx == 0:
                im_resized.save(OUTPUT_DIR / f"p01-t1c-seg-slice-{z}.png")

        print(f"Generated slices for stage {stage_idx} ({stage_name})")


if __name__ == "__main__":
    generate_demo_slices()
