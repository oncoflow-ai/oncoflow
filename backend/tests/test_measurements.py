from __future__ import annotations

import pytest

from app.modules.results.measurements import compute_longest_diameter_mm, compute_volume_mm3


def test_measurement_functions_compute_volume_and_longest_diameter() -> None:
    mask = [
        [
            [1, 1],
            [1, 1],
        ],
        [
            [1, 1],
            [1, 1],
        ],
    ]
    spacing = (1.0, 2.0, 3.0)

    assert compute_volume_mm3(mask, spacing) == 48.0
    assert round(compute_longest_diameter_mm(mask, spacing), 5) == round((1.0**2 + 2.0**2 + 3.0**2) ** 0.5, 5)


def test_measurement_functions_reject_empty_masks_or_missing_spacing() -> None:
    with pytest.raises(ValueError, match="at least one positive voxel"):
        compute_volume_mm3([[[0]]], (1.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="three positive values"):
        compute_longest_diameter_mm([[[1]]], (1.0, 1.0, 0.0))
