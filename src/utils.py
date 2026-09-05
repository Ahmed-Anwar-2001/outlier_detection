"""
Utility functions for I/O, logging setup, and progress helpers.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.config import PipelineConfig


def setup_logging(config: PipelineConfig) -> logging.Logger:
    """Configure structured logging for the pipeline."""
    logger = logging.getLogger("suspicious_photo_detection")
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    # Prevent duplicate handlers on re-init
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if config.log_file:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Write results to JSON with proper formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_outlet_dirs(dataset_dir: Path, extensions: tuple) -> list[tuple[str, Path]]:
    """
    Discover all outlet directories and validate they contain images.

    Returns:
        List of (outlet_id, outlet_path) tuples, sorted by outlet_id.
    """
    outlets = []
    for entry in sorted(dataset_dir.iterdir()):
        if entry.is_dir():
            images = [
                f for f in entry.iterdir()
                if f.suffix.lower() in extensions
            ]
            if images:
                outlets.append((entry.name, entry))
    return outlets


def get_image_files(outlet_dir: Path, extensions: tuple) -> list[Path]:
    """Get sorted list of image files in an outlet directory."""
    return sorted(
        f for f in outlet_dir.iterdir()
        if f.suffix.lower() in extensions
    )
