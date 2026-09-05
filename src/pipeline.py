"""
Pipeline orchestrator — coordinates feature extraction, anomaly detection,
reason generation, and structured output formatting.
"""

import logging
import time
from typing import Any

from tqdm import tqdm

from src.anomaly_detector import AnomalyDetector, OutletResult
from src.config import PipelineConfig
from src.feature_extractor import FeatureExtractor
from src.reason_generator import ReasonGenerator
from src.utils import (
    get_image_files,
    load_outlet_dirs,
    save_results,
    setup_logging,
)

logger = logging.getLogger("suspicious_photo_detection")


class Pipeline:
    """End-to-end suspicious photo detection pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.extractor = FeatureExtractor(config)
        self.detector = AnomalyDetector(config)
        self.reasoner = ReasonGenerator(config)

    def run(self) -> list[dict[str, Any]]:
        """
        Execute the full pipeline across all outlets in the dataset.

        Returns:
            List of schema-compliant per-outlet result dictionaries.
        """
        setup_logging(self.config)
        start_time = time.time()

        outlets = load_outlet_dirs(
            self.config.dataset_dir, self.config.image_extensions
        )
        logger.info("Found %d outlets in %s", len(outlets), self.config.dataset_dir)

        if not outlets:
            logger.error("No outlet directories found at %s", self.config.dataset_dir)
            return []

        all_results = []
        total_flagged = 0
        total_images = 0

        for outlet_id, outlet_path in tqdm(outlets, desc="Evaluating outlets", unit="outlet"):
            try:
                result = self._process_outlet(outlet_id, outlet_path)
                formatted = self._format_result(result)
                all_results.append(formatted)

                total_flagged += len(formatted["flagged_images"])
                total_images += formatted["total_images"]

            except Exception as e:
                logger.error("Error evaluating outlet %s: %s", outlet_id, e, exc_info=True)
                all_results.append(
                    {
                        "outlet_id": outlet_id,
                        "total_images": 0,
                        "flagged_images": [],
                        "ranking": [],
                    }
                )

        save_results(all_results, self.config.output_path)

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Pipeline execution completed in %.2fs", elapsed)
        logger.info(
            "Summary: %d outlets, %d total images evaluated, %d flagged suspicious",
            len(all_results),
            total_images,
            total_flagged,
        )
        logger.info("Structured output written to: %s", self.config.output_path)
        logger.info("=" * 60)

        return all_results

    def _process_outlet(self, outlet_id: str, outlet_path) -> OutletResult:
        """Process a single outlet through the 3 stages."""
        image_paths = get_image_files(outlet_path, self.config.image_extensions)
        file_names, features_dict = self.extractor.extract_outlet(outlet_id, image_paths)
        outlet_result = self.detector.detect(outlet_id, file_names, features_dict)
        explained_result = self.reasoner.generate(outlet_result)
        return explained_result

    def _format_result(self, result: OutletResult) -> dict[str, Any]:
        """Format an OutletResult into the assignment's exact JSON schema."""
        flagged_images = []
        for img in result.image_results:
            if img.is_flagged:
                flagged_images.append(
                    {
                        "file_name": img.file_name,
                        "suspicion_score": img.suspicion_score,
                        "reason": img.reason or "Visually inconsistent with outlet's photo series",
                    }
                )

        # Sort flagged images by descending suspicion score
        flagged_images.sort(key=lambda x: x["suspicion_score"], reverse=True)

        # Ranking: all images in folder ordered from most to least suspicious
        ranked = sorted(
            result.image_results,
            key=lambda r: r.suspicion_score,
            reverse=True,
        )
        ranking = [r.file_name for r in ranked]

        return {
            "outlet_id": result.outlet_id,
            "total_images": result.total_images,
            "flagged_images": flagged_images,
            "ranking": ranking,
        }
