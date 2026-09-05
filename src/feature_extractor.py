"""
Multi-signal visual feature extractor for outlet verification images.

Combines:
1. Spatial Grid Color Distributions (HSV color moments and histograms)
2. Directional Edge & Structural Gradients (storefront shutters, counters, frames)
3. Perceptual Frequency Signatures (2D Discrete Cosine Transform)

Returns normalized unified feature vectors alongside decomposed signals
for precise, human-interpretable outlier reason generation.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.fftpack import dct
from scipy.ndimage import sobel

from src.config import PipelineConfig

logger = logging.getLogger("suspicious_photo_detection")


class FeatureExtractor:
    """Multi-signal visual feature extractor with local caching."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def extract_outlet(
        self, outlet_id: str, image_paths: list[Path]
    ) -> tuple[list[str], dict[str, np.ndarray]]:
        """
        Extract feature embeddings for all images in an outlet.

        Returns:
            Tuple of:
            - file_names: list of filenames
            - features_dict: {
                'combined': (N, D) unified L2-normalized embeddings,
                'color': (N, D_c) color component embeddings,
                'structure': (N, D_s) edge structure embeddings,
                'dct': (N, D_d) perceptual DCT layout embeddings
              }
        """
        file_names = [p.name for p in image_paths]
        cache_path = self._get_cache_path(outlet_id)

        # Check cache
        if self.config.use_cache and cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=True).item()
                if list(cached.get("file_names", [])) == file_names:
                    logger.debug("Cache hit for %s (%d images)", outlet_id, len(file_names))
                    return file_names, cached["features"]
            except Exception as e:
                logger.warning("Failed to read cache for %s (%s), re-extracting", outlet_id, e)

        # Extract features
        combined_list = []
        color_list = []
        struct_list = []
        dct_list = []
        valid_names = []

        for img_path in image_paths:
            feats = self._extract_single(img_path)
            if feats is not None:
                valid_names.append(img_path.name)
                combined_list.append(feats["combined"])
                color_list.append(feats["color"])
                struct_list.append(feats["structure"])
                dct_list.append(feats["dct"])

        if not combined_list:
            empty_dict = {
                "combined": np.array([]),
                "color": np.array([]),
                "structure": np.array([]),
                "dct": np.array([]),
            }
            return valid_names, empty_dict

        features_dict = {
            "combined": np.vstack(combined_list).astype(np.float32),
            "color": np.vstack(color_list).astype(np.float32),
            "structure": np.vstack(struct_list).astype(np.float32),
            "dct": np.vstack(dct_list).astype(np.float32),
        }

        # Cache features
        if self.config.use_cache:
            self._save_cache(cache_path, valid_names, features_dict)

        return valid_names, features_dict

    def _extract_single(self, img_path: Path) -> dict[str, np.ndarray] | None:
        """Extract multi-signal descriptor for a single image."""
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB").resize(self.config.target_size, Image.Resampling.BILINEAR)
                arr_rgb = np.array(img_rgb, dtype=np.float32) / 255.0

                # 1. Color Features (HSV color moments and spatial histograms)
                img_hsv = img.convert("HSV").resize(self.config.target_size, Image.Resampling.BILINEAR)
                arr_hsv = np.array(img_hsv, dtype=np.float32) / 255.0
                color_feat = self._extract_spatial_color(arr_hsv)

                # 2. Structural & Edge Features (horizontal/vertical Sobel gradients)
                gray = 0.2989 * arr_rgb[:, :, 0] + 0.5870 * arr_rgb[:, :, 1] + 0.1140 * arr_rgb[:, :, 2]
                struct_feat = self._extract_edge_structure(gray)

                # 3. Frequency / Global Layout Features (2D-DCT low-frequency coefficients)
                dct_feat = self._extract_dct_layout(gray)

                # Normalize individual vectors
                norm_c = np.linalg.norm(color_feat) + 1e-8
                norm_s = np.linalg.norm(struct_feat) + 1e-8
                norm_d = np.linalg.norm(dct_feat) + 1e-8

                c_unit = color_feat / norm_c
                s_unit = struct_feat / norm_s
                d_unit = dct_feat / norm_d

                # Weighted fusion
                combined = np.concatenate([
                    np.sqrt(self.config.weight_color) * c_unit,
                    np.sqrt(self.config.weight_structure) * s_unit,
                    np.sqrt(self.config.weight_dct) * d_unit,
                ])
                combined = combined / (np.linalg.norm(combined) + 1e-8)

                return {
                    "combined": combined,
                    "color": c_unit,
                    "structure": s_unit,
                    "dct": d_unit,
                }
        except Exception as e:
            logger.warning("Corrupt or unreadable image %s: %s", img_path.name, e)
            return None

    def _extract_spatial_color(self, hsv: np.ndarray) -> np.ndarray:
        """Extract spatial grid color histograms and moments."""
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        r_step = hsv.shape[0] // self.config.grid_rows
        c_step = hsv.shape[1] // self.config.grid_cols

        features = []

        # Global histograms
        h_hist, _ = np.histogram(h, bins=16, range=(0.0, 1.0))
        s_hist, _ = np.histogram(s, bins=8, range=(0.0, 1.0))
        v_hist, _ = np.histogram(v, bins=8, range=(0.0, 1.0))
        features.extend([h_hist, s_hist, v_hist])

        # Grid-level statistics (captures top signage vs middle counter vs bottom pavement)
        for r in range(self.config.grid_rows):
            for c in range(self.config.grid_cols):
                sub_h = h[r * r_step : (r + 1) * r_step, c * c_step : (c + 1) * c_step]
                sub_s = s[r * r_step : (r + 1) * r_step, c * c_step : (c + 1) * c_step]
                sub_v = v[r * r_step : (r + 1) * r_step, c * c_step : (c + 1) * c_step]

                # Mean, std, and compact hue histogram
                cell_h_hist, _ = np.histogram(sub_h, bins=8, range=(0.0, 1.0))
                stats = np.array([
                    np.mean(sub_h), np.std(sub_h),
                    np.mean(sub_s), np.std(sub_s),
                    np.mean(sub_v), np.std(sub_v),
                ])
                features.extend([cell_h_hist, stats])

        return np.concatenate(features).astype(np.float32)

    def _extract_edge_structure(self, gray: np.ndarray) -> np.ndarray:
        """Extract directional edge gradients and vertical band densities."""
        sx = sobel(gray, axis=1)  # vertical lines (posts, borders)
        sy = sobel(gray, axis=0)  # horizontal lines (shutters, shelves)

        mag = np.hypot(sx, sy)

        # Gradient orientation histogram (8 bins: 0 to pi)
        ang = np.arctan2(np.abs(sy), np.abs(sx))
        ang_hist, _ = np.histogram(ang, bins=8, range=(0.0, np.pi / 2.0), weights=mag)

        # Edge distribution across 4 horizontal stripes
        h_step = gray.shape[0] // 4
        stripe_energy = [
            np.mean(mag[i * h_step : (i + 1) * h_step, :])
            for i in range(4)
        ]

        # Horizontal vs vertical dominance ratio
        h_dominance = np.mean(np.abs(sy)) / (np.mean(np.abs(sx)) + 1e-6)

        return np.concatenate([
            ang_hist,
            np.array(stripe_energy),
            np.array([h_dominance]),
        ]).astype(np.float32)

    def _extract_dct_layout(self, gray: np.ndarray) -> np.ndarray:
        """Extract low-frequency 2D-DCT coefficients for scene layout."""
        # Downsample to 32x32 for global frequency response
        small = gray[::8, ::8]
        # 2D DCT
        dct_2d = dct(dct(small.T, norm="ortho").T, norm="ortho")
        # Extract 8x8 low-frequency block (excluding DC component [0,0])
        block = dct_2d[:8, :8].flatten()
        return block[1:].astype(np.float32)

    def _get_cache_path(self, outlet_id: str) -> Path:
        cache_dir = self.config.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{outlet_id}.npy"

    def _save_cache(
        self, cache_path: Path, file_names: list[str], features: dict[str, np.ndarray]
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            cache_path,
            {"file_names": file_names, "features": features},
            allow_pickle=True,
        )
