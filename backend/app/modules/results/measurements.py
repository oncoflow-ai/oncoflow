from __future__ import annotations

import math
from itertools import combinations


def _validate_mask(mask: list[list[list[int]]]) -> list[tuple[int, int, int]]:
    occupied: list[tuple[int, int, int]] = []
    for z, plane in enumerate(mask):
        for y, row in enumerate(plane):
            for x, value in enumerate(row):
                if value:
                    occupied.append((x, y, z))
    if not occupied:
        raise ValueError("mask must contain at least one positive voxel")
    return occupied


def _validate_spacing(voxel_spacing_mm: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(voxel_spacing_mm) != 3 or any(value <= 0 for value in voxel_spacing_mm):
        raise ValueError("voxel_spacing_mm must contain three positive values")
    return voxel_spacing_mm


def compute_volume_mm3(mask: list[list[list[int]]], voxel_spacing_mm: tuple[float, float, float]) -> float:
    occupied = _validate_mask(mask)
    spacing = _validate_spacing(voxel_spacing_mm)
    return float(len(occupied) * spacing[0] * spacing[1] * spacing[2])


def compute_longest_diameter_mm(mask: list[list[list[int]]], voxel_spacing_mm: tuple[float, float, float]) -> float:
    occupied = _validate_mask(mask)
    spacing = _validate_spacing(voxel_spacing_mm)
    if len(occupied) == 1:
        return 0.0

    def to_mm(point: tuple[int, int, int]) -> tuple[float, float, float]:
        return (point[0] * spacing[0], point[1] * spacing[1], point[2] * spacing[2])

    max_distance = 0.0
    for first, second in combinations(occupied, 2):
        a = to_mm(first)
        b = to_mm(second)
        distance = math.dist(a, b)
        max_distance = max(max_distance, distance)
    return float(max_distance)
