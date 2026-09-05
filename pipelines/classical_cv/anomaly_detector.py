"""
Classical anomaly detector with multi-instance k-NN distance scoring.
"""
from dataclasses import dataclass
import numpy as np
try:
    from .config import ClassicalConfig
except (ImportError, ValueError):
    from pipelines.classical_cv.config import ClassicalConfig


@dataclass
class ImageResult:
    file_name: str
    cosine_distance: float
    suspicion_score: float
    is_flagged: bool
    reason: str = ""


@dataclass
class OutletResult:
    outlet_id: str
    total_images: int
    image_results: list[ImageResult]
    threshold: float
    median_distance: float
    mad: float
    skipped: bool = False


class ClassicalAnomalyDetector:
    def __init__(self, config: ClassicalConfig):
        self.config = config

    def detect(self, outlet_id: str, file_names: list[str], features_dict: dict[str, np.ndarray]) -> OutletResult:
        n_images = len(file_names)
        embeddings = features_dict.get("combined", np.array([]))

        if n_images < self.config.min_images_for_detection or embeddings.size == 0:
            return OutletResult(
                outlet_id=outlet_id,
                total_images=n_images,
                image_results=[ImageResult(fn, 0.0, 0.0, False) for fn in file_names],
                threshold=0.0, median_distance=0.0, mad=0.0, skipped=True
            )

        sim_matrix = embeddings @ embeddings.T
        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, np.inf)

        min_pairwise_dists = dist_matrix.min(axis=1)
        sorted_dists = np.sort(dist_matrix, axis=1)
        k = max(1, min(self.config.knn_k, n_images - 1))
        knn_dists = sorted_dists[:, :k].mean(axis=1)
        combined_dists = 0.5 * min_pairwise_dists + 0.5 * knn_dists

        median_dist = float(np.median(combined_dists))
        mad = float(np.median(np.abs(combined_dists - median_dist)))
        threshold = median_dist + (self.config.fallback_threshold if mad < 1e-5 else self.config.threshold_k * mad)

        dist_min = float(combined_dists.min())
        dist_max = float(combined_dists.max())

        image_results = []
        for i, fn in enumerate(file_names):
            d = combined_dists[i]
            is_flagged = bool(d > threshold)
            if is_flagged:
                excess = (d - threshold) / max(dist_max - threshold, 1e-5)
                score = round(float(0.65 + 0.35 * excess), 4)
                reason = "Multi-signal anomaly — divergent localized edge structure and spatial color profile relative to outlet visits"
            else:
                ratio = (d - dist_min) / max(threshold - dist_min, 1e-5)
                score = round(float(0.45 * max(0.0, ratio)), 4)
                reason = ""
            image_results.append(ImageResult(fn, float(d), score, is_flagged, reason))

        return OutletResult(outlet_id, n_images, image_results, threshold, median_dist, mad)
