"""
Configuration for the Classical Multi-Signal CV Pipeline.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ClassicalConfig:
    dataset_dir: Path = Path("dataset/dataset")
    output_path: Path = Path("results/results_classical.json")
    cache_dir: Path = Path(".cache/features_classical")

    target_size: tuple[int, int] = (256, 256)
    grid_rows: int = 4
    grid_cols: int = 4
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    weight_color: float = 0.40
    weight_structure: float = 0.40
    weight_dct: float = 0.20

    knn_k: int = 2
    threshold_k: float = 2.2
    min_images_for_detection: int = 3
    fallback_threshold: float = 0.18

    high_anomaly_threshold: float = 0.80
    moderate_anomaly_threshold: float = 0.55

    use_cache: bool = True
