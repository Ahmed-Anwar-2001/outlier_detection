"""
Configuration for CLIP Vision Embedding Pipeline.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClipConfig:
    dataset_dir: Path = Path("dataset/dataset")
    output_path: Path = Path("results/results_clip.json")
    cache_dir: Path = Path(".cache/features_clip")

    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str = "cpu"
    batch_size: int = 16

    knn_k: int = 2
    threshold_k: float = 2.2
    min_images_for_detection: int = 3
    fallback_threshold: float = 0.15

    use_cache: bool = True
