"""
CLIP visual feature extractor using open_clip.
"""
from pathlib import Path
import numpy as np
from PIL import Image
try:
    from .config import ClipConfig
except (ImportError, ValueError):
    from pipelines.clip_embeddings.config import ClipConfig


class ClipFeatureExtractor:
    def __init__(self, config: ClipConfig):
        self.config = config
        self.model = None
        self.preprocess = None

    def _lazy_init(self):
        if self.model is None:
            import open_clip
            import torch
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.config.model_name, pretrained=self.config.pretrained
            )
            model.eval()
            self.model = model.to(self.config.device)
            self.preprocess = preprocess

    def extract_outlet(self, outlet_id: str, image_paths: list[Path]):
        file_names = [p.name for p in image_paths]
        cache_path = self.config.cache_dir / f"{outlet_id}.npy"

        if self.config.use_cache and cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=True).item()
                if list(cached.get("file_names", [])) == file_names:
                    return file_names, cached["embeddings"]
            except Exception:
                pass

        self._lazy_init()
        import torch

        embeddings = []
        valid_names = []

        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensor = self.preprocess(img).unsqueeze(0).to(self.config.device)
                with torch.no_grad():
                    feat = self.model.encode_image(tensor)
                    feat /= feat.norm(dim=-1, keepdim=True)
                    embeddings.append(feat.cpu().numpy().flatten())
                valid_names.append(p.name)
            except Exception as e:
                print(f"Error reading {p}: {e}")

        if not embeddings:
            return valid_names, np.array([])

        emb_matrix = np.vstack(embeddings).astype(np.float32)

        if self.config.use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, {"file_names": valid_names, "embeddings": emb_matrix}, allow_pickle=True)

        return valid_names, emb_matrix
