"""
Centralized configuration for the Suspicious Photo Detection pipeline.

All tunable parameters are defined here with sensible defaults.
Override via CLI arguments in run_pipeline.py.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for the full detection pipeline."""

    # --- Paths ---
    dataset_dir: Path = Path("dataset/dataset")
    output_path: Path = Path("results/results.json")
    cache_dir: Path = Path(".cache/features")

    # --- Feature Extraction ---
    target_size: tuple[int, int] = (256, 256)
    grid_rows: int = 3
    grid_cols: int = 3
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    # --- Weights for Multi-Signal Fusion ---
    weight_color: float = 0.45      # Spatial color distribution (branding, signage, walls)
    weight_structure: float = 0.35  # Edge gradients (shutters, counters, geometry)
    weight_dct: float = 0.20        # Frequency layout (perceptual scene structure)

    # --- Anomaly Detection ---
    threshold_k: float = 2.5       # MAD multiplier for outlier threshold
    min_images_for_detection: int = 3  # skip outlets with fewer images
    fallback_threshold: float = 0.18   # fallback when MAD ≈ 0 (very uniform images)
    score_floor: float = 0.0
    score_ceil: float = 1.0

    # --- Reason Generation ---
    high_anomaly_threshold: float = 0.85
    moderate_anomaly_threshold: float = 0.65

    # --- Caching ---
    use_cache: bool = True

    # --- Logging ---
    log_level: str = "INFO"
    log_file: Optional[Path] = None
