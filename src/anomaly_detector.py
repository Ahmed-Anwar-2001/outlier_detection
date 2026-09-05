"""
Anomaly detection using centroid-based cosine distance with robust MAD thresholding.

For each outlet:
1. Computes centroid across the outlet's photo series.
2. Computes distance of each image from the centroid.
3. Computes component-level deviations (color, structure, scene layout).
4. Determines adaptive threshold via Median Absolute Deviation (MAD).
5. Produces normalized suspicion scores [0, 1] and flags outliers.
"""

import logging
from dataclasses import dataclass

import numpy as np

from src.config import PipelineConfig

logger = logging.getLogger("suspicious_photo_detection")


@dataclass
class ImageResult:
    """Detection result for a single image."""

    file_name: str
    cosine_distance: float
    suspicion_score: float  # normalized 0–1
    is_flagged: bool
    min_pairwise_distance: float
    dist_color: float
    dist_structure: float
    dist_dct: float
    reason: str = ""


@dataclass
class OutletResult:
    """Detection result for an entire outlet."""

    outlet_id: str
    total_images: int
    image_results: list[ImageResult]
    centroid_distances: np.ndarray
    threshold: float
    median_distance: float
    mad: float
    skipped: bool = False
    skip_reason: str = ""


class AnomalyDetector:
    """Robust centroid-based anomaly detector with MAD thresholding."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def detect(
        self,
        outlet_id: str,
        file_names: list[str],
        features_dict: dict[str, np.ndarray],
    ) -> OutletResult:
        """
        Run anomaly detection for a single outlet.

        Args:
            outlet_id: Outlet identifier.
            file_names: List of image filenames.
            features_dict: Dictionary of feature matrices.
        """
        n_images = len(file_names)
        embeddings = features_dict.get("combined", np.array([]))

        # Edge case: insufficient images to form a reliable baseline
        if n_images < self.config.min_images_for_detection:
            logger.info("Skipping %s: only %d images (minimum %d needed)", outlet_id, n_images, self.config.min_images_for_detection)
            return OutletResult(
                outlet_id=outlet_id,
                total_images=n_images,
                image_results=[
                    ImageResult(
                        file_name=fn,
                        cosine_distance=0.0,
                        suspicion_score=0.0,
                        is_flagged=False,
                        min_pairwise_distance=0.0,
                        dist_color=0.0,
                        dist_structure=0.0,
                        dist_dct=0.0,
                    )
                    for fn in file_names
                ],
                centroid_distances=np.zeros(n_images),
                threshold=0.0,
                median_distance=0.0,
                mad=0.0,
                skipped=True,
                skip_reason=f"Insufficient images ({n_images} < {self.config.min_images_for_detection})",
            )

        if embeddings.size == 0 or len(embeddings) != n_images:
            logger.warning("Empty or mismatched embeddings for %s", outlet_id)
            return OutletResult(
                outlet_id=outlet_id,
                total_images=n_images,
                image_results=[],
                centroid_distances=np.array([]),
                threshold=0.0,
                median_distance=0.0,
                mad=0.0,
                skipped=True,
                skip_reason="Extraction failed for images",
            )

        # 1. Compute Centroid
        centroid = np.mean(embeddings, axis=0)
        norm_c = np.linalg.norm(centroid) + 1e-8
        centroid = centroid / norm_c

        # 2. Combined Cosine Distances from Centroid
        sims = embeddings @ centroid
        cosine_distances = 1.0 - sims
        cosine_distances = np.clip(cosine_distances, 0.0, 2.0)

        # 3. Component Centroids and Distances
        color_feats = features_dict.get("color", embeddings)
        struct_feats = features_dict.get("structure", embeddings)
        dct_feats = features_dict.get("dct", embeddings)

        color_cent = np.mean(color_feats, axis=0)
        color_cent = color_cent / (np.linalg.norm(color_cent) + 1e-8)
        dist_color = 1.0 - (color_feats @ color_cent)

        struct_cent = np.mean(struct_feats, axis=0)
        struct_cent = struct_cent / (np.linalg.norm(struct_cent) + 1e-8)
        dist_struct = 1.0 - (struct_feats @ struct_cent)

        dct_cent = np.mean(dct_feats, axis=0)
        dct_cent = dct_cent / (np.linalg.norm(dct_cent) + 1e-8)
        dist_dct = 1.0 - (dct_feats @ dct_cent)

        # 4. Pairwise Distances (Nearest Neighbor)
        sim_matrix = embeddings @ embeddings.T
        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, np.inf)
        min_pairwise_dists = dist_matrix.min(axis=1)

        # 5. Robust MAD-Based Threshold
        median_dist = float(np.median(cosine_distances))
        mad = float(np.median(np.abs(cosine_distances - median_dist)))

        if mad < 1e-5:
            threshold = median_dist + self.config.fallback_threshold
            logger.debug("%s: Uniform series (MAD ≈ 0), using fallback threshold %.4f", outlet_id, threshold)
        else:
            threshold = median_dist + self.config.threshold_k * mad

        # 6. Normalize Suspicion Scores to [0, 1]
        dist_min = float(cosine_distances.min())
        dist_max = float(cosine_distances.max())

        if dist_max - dist_min < 1e-7:
            normalized_scores = np.zeros_like(cosine_distances)
        else:
            normalized_scores = (cosine_distances - dist_min) / (dist_max - dist_min)

        normalized_scores = np.clip(
            normalized_scores, self.config.score_floor, self.config.score_ceil
        )

        # 7. Construct per-image results
        image_results = []
        for i, fn in enumerate(file_names):
            is_flagged = bool(cosine_distances[i] > threshold)
            image_results.append(
                ImageResult(
                    file_name=fn,
                    cosine_distance=float(cosine_distances[i]),
                    suspicion_score=round(float(normalized_scores[i]), 4),
                    is_flagged=is_flagged,
                    min_pairwise_distance=float(min_pairwise_dists[i]),
                    dist_color=float(dist_color[i]),
                    dist_structure=float(dist_struct[i]),
                    dist_dct=float(dist_dct[i]),
                )
            )

        n_flagged = sum(1 for r in image_results if r.is_flagged)
        logger.info(
            "%s: %d images, threshold=%.4f (median=%.4f, MAD=%.4f), %d flagged",
            outlet_id,
            n_images,
            threshold,
            median_dist,
            mad,
            n_flagged,
        )

        return OutletResult(
            outlet_id=outlet_id,
            total_images=n_images,
            image_results=image_results,
            centroid_distances=cosine_distances,
            threshold=threshold,
            median_distance=median_dist,
            mad=mad,
        )
