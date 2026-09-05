"""
Multi-signal classical feature extractor using spatial RGB+HSV moments, Sobel HOG, and DCT.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.fftpack import dct
from scipy.ndimage import sobel

try:
    from .config import ClassicalConfig
except (ImportError, ValueError):
    from pipelines.classical_cv.config import ClassicalConfig


class ClassicalFeatureExtractor:
    def __init__(self, config: ClassicalConfig):
        self.config = config

    def extract_outlet(self, outlet_id: str, image_paths: list[Path]):
        file_names = [p.name for p in image_paths]
        cache_path = self.config.cache_dir / f"{outlet_id}.npy"

        if self.config.use_cache and cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=True).item()
                if list(cached.get("file_names", [])) == file_names:
                    return file_names, cached["features"]
            except Exception:
                pass

        valid_names, combined_list, color_list, struct_list, dct_list = [], [], [], [], []
        for p in image_paths:
            feats = self._extract_single(p)
            if feats:
                valid_names.append(p.name)
                combined_list.append(feats["combined"])
                color_list.append(feats["color"])
                struct_list.append(feats["structure"])
                dct_list.append(feats["dct"])

        if not combined_list:
            return valid_names, {"combined": np.array([]), "color": np.array([]), "structure": np.array([]), "dct": np.array([])}

        features_dict = {
            "combined": np.vstack(combined_list).astype(np.float32),
            "color": np.vstack(color_list).astype(np.float32),
            "structure": np.vstack(struct_list).astype(np.float32),
            "dct": np.vstack(dct_list).astype(np.float32),
        }

        if self.config.use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, {"file_names": valid_names, "features": features_dict}, allow_pickle=True)

        return valid_names, features_dict

    def _extract_single(self, img_path: Path):
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB").resize(self.config.target_size, Image.Resampling.BILINEAR)
                arr_rgb = np.array(img_rgb, dtype=np.float32) / 255.0
                img_hsv = img.convert("HSV").resize(self.config.target_size, Image.Resampling.BILINEAR)
                arr_hsv = np.array(img_hsv, dtype=np.float32) / 255.0

                color_feat = self._extract_spatial_color(arr_rgb, arr_hsv)

                gray = 0.2989 * arr_rgb[:, :, 0] + 0.5870 * arr_rgb[:, :, 1] + 0.1140 * arr_rgb[:, :, 2]
                struct_feat = self._extract_edge_structure(gray)
                dct_feat = self._extract_dct_layout(gray)

                c_unit = color_feat / (np.linalg.norm(color_feat) + 1e-8)
                s_unit = struct_feat / (np.linalg.norm(struct_feat) + 1e-8)
                d_unit = dct_feat / (np.linalg.norm(dct_feat) + 1e-8)

                combined = np.concatenate([
                    np.sqrt(self.config.weight_color) * c_unit,
                    np.sqrt(self.config.weight_structure) * s_unit,
                    np.sqrt(self.config.weight_dct) * d_unit,
                ])
                combined /= (np.linalg.norm(combined) + 1e-8)

                return {"combined": combined, "color": c_unit, "structure": s_unit, "dct": d_unit}
        except Exception:
            return None

    def _extract_spatial_color(self, arr_rgb: np.ndarray, arr_hsv: np.ndarray):
        h, s, v = arr_hsv[:, :, 0], arr_hsv[:, :, 1], arr_hsv[:, :, 2]
        features = []

        h_hist, _ = np.histogram(h, bins=16, range=(0.0, 1.0))
        s_hist, _ = np.histogram(s, bins=8, range=(0.0, 1.0))
        v_hist, _ = np.histogram(v, bins=8, range=(0.0, 1.0))
        features.extend([h_hist / (np.linalg.norm(h_hist) + 1e-6),
                         s_hist / (np.linalg.norm(s_hist) + 1e-6),
                         v_hist / (np.linalg.norm(v_hist) + 1e-6)])

        grid_size = 4
        h_step = arr_rgb.shape[0] // grid_size
        w_step = arr_rgb.shape[1] // grid_size
        for ri in range(grid_size):
            for ci in range(grid_size):
                patch_rgb = arr_rgb[ri * h_step : (ri + 1) * h_step, ci * w_step : (ci + 1) * w_step]
                patch_hsv = arr_hsv[ri * h_step : (ri + 1) * h_step, ci * w_step : (ci + 1) * w_step]
                features.extend([patch_rgb.mean(axis=(0, 1)), patch_rgb.std(axis=(0, 1)),
                                 patch_hsv.mean(axis=(0, 1)), patch_hsv.std(axis=(0, 1))])

        return np.concatenate(features).astype(np.float32)

    def _extract_edge_structure(self, gray: np.ndarray):
        sx = sobel(gray, axis=1)
        sy = sobel(gray, axis=0)
        mag = np.hypot(sx, sy)
        ang = np.arctan2(np.abs(sy), np.abs(sx))

        features = []
        grid_size = 4
        h_step = gray.shape[0] // grid_size
        w_step = gray.shape[1] // grid_size

        for ri in range(grid_size):
            for ci in range(grid_size):
                patch_mag = mag[ri * h_step : (ri + 1) * h_step, ci * w_step : (ci + 1) * w_step]
                patch_ang = ang[ri * h_step : (ri + 1) * h_step, ci * w_step : (ci + 1) * w_step]
                cell_hist, _ = np.histogram(patch_ang, bins=8, range=(0.0, np.pi / 2.0), weights=patch_mag)
                features.append(cell_hist / (np.linalg.norm(cell_hist) + 1e-6))

        h_dominance = np.mean(np.abs(sy)) / (np.mean(np.abs(sx)) + 1e-6)
        stripe_energy = [np.mean(mag[i * (gray.shape[0] // 4) : (i + 1) * (gray.shape[0] // 4), :]) for i in range(4)]
        features.extend([np.array([h_dominance]), np.array(stripe_energy)])

        return np.concatenate(features).astype(np.float32)

    def _extract_dct_layout(self, gray: np.ndarray):
        small = gray[::8, ::8]
        dct_2d = dct(dct(small.T, norm="ortho").T, norm="ortho")
        return dct_2d[:8, :8].flatten()[1:].astype(np.float32)
