"""
Contextual, human-readable reason generator for flagged outlet images.

Analyzes which visual component diverged the most:
- Color / signage / banner branding distribution
- Structural / shutter / counter edge orientation
- Global scene geometry and spatial layout
"""

import logging

from src.anomaly_detector import ImageResult, OutletResult
from src.config import PipelineConfig

logger = logging.getLogger("suspicious_photo_detection")


class ReasonGenerator:
    """Generate precise, contextual reasons for flagged images."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def generate(self, outlet_result: OutletResult) -> OutletResult:
        """Attach reasons to all flagged images."""
        if outlet_result.skipped or not outlet_result.image_results:
            return outlet_result

        # Outlet-level baselines
        pairwise_dists = [r.min_pairwise_distance for r in outlet_result.image_results]
        median_pairwise = float(sorted(pairwise_dists)[len(pairwise_dists) // 2])

        color_dists = [r.dist_color for r in outlet_result.image_results]
        struct_dists = [r.dist_structure for r in outlet_result.image_results]
        dct_dists = [r.dist_dct for r in outlet_result.image_results]

        med_color = float(sorted(color_dists)[len(color_dists) // 2])
        med_struct = float(sorted(struct_dists)[len(struct_dists) // 2])
        med_dct = float(sorted(dct_dists)[len(dct_dists) // 2])

        for img in outlet_result.image_results:
            if img.is_flagged:
                img.reason = self._build_reason(
                    img, median_pairwise, med_color, med_struct, med_dct
                )

        return outlet_result

    def _build_reason(
        self,
        img: ImageResult,
        med_pairwise: float,
        med_color: float,
        med_struct: float,
        med_dct: float,
    ) -> str:
        """Synthesize domain-specific human-readable rationale."""
        is_isolated = img.min_pairwise_distance > (med_pairwise * 2.2 + 0.05)

        # Compute relative deviations per visual channel
        dev_color = (img.dist_color - med_color) / (med_color + 1e-4)
        dev_struct = (img.dist_structure - med_struct) / (med_struct + 1e-4)
        dev_dct = (img.dist_dct - med_dct) / (med_dct + 1e-4)

        reasons = []

        if dev_color > 1.2:
            reasons.append("distinct color palette and branding/signage mismatch")
        if dev_struct > 1.2:
            reasons.append("divergent edge geometry and storefront structural lines")
        if dev_dct > 1.2:
            reasons.append("dissimilar overall scene composition and spatial layout")

        if not reasons:
            reasons.append("low visual similarity to the outlet's historical photo cluster")

        primary_desc = ", ".join(reasons)

        if img.suspicion_score >= self.config.high_anomaly_threshold:
            prefix = "Highly anomalous image"
            if is_isolated:
                suffix = "unrelated to all other visits; likely a completely different location"
            else:
                suffix = "major visual discrepancy from the outlet's established identity"
            return f"{prefix} — {primary_desc}; {suffix}"

        elif img.suspicion_score >= self.config.moderate_anomaly_threshold:
            prefix = "Moderate visual inconsistency"
            return f"{prefix} — {primary_desc} compared to typical outlet visits"

        else:
            return f"Minor anomaly — {primary_desc} relative to cluster centroid"
