"""
Classical Computer Vision Anomaly Detection Pipeline.
Extracts spatial color distributions (HSV), edge structure (Sobel HOG), and frequency textures (DCT).
Saves results to results/results_classical.json.
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.classical_cv.config import ClassicalConfig
from pipelines.classical_cv.feature_extractor import ClassicalFeatureExtractor
from pipelines.classical_cv.anomaly_detector import ClassicalAnomalyDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Classical CV Suspicious Photo Detection Pipeline")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "dataset" / "dataset", help="Dataset directory")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "results_classical.json", help="Output JSON path")
    parser.add_argument("--outlet", type=str, default=None, help="Process a single outlet ID")
    return parser.parse_args()


def run_classical_pipeline(dataset_dir: Path = None, output_path: Path = None, outlet_id: str = None):
    if dataset_dir is None:
        dataset_dir = PROJECT_ROOT / "dataset" / "dataset"
    if output_path is None:
        output_path = PROJECT_ROOT / "results" / "results_classical.json"

    config = ClassicalConfig(dataset_dir=dataset_dir, output_path=output_path)
    extractor = ClassicalFeatureExtractor(config)
    detector = ClassicalAnomalyDetector(config)

    outlet_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    if outlet_id:
        outlet_dirs = [d for d in outlet_dirs if d.name == outlet_id]

    all_results = []
    print(f"Running Classical CV Pipeline on {len(outlet_dirs)} outlet(s)...")

    for i, o_dir in enumerate(outlet_dirs, 1):
        img_paths = sorted([p for p in o_dir.iterdir() if p.suffix.lower() in config.image_extensions])
        file_names, feats = extractor.extract_outlet(o_dir.name, img_paths)
        res = detector.detect(o_dir.name, file_names, feats)

        flagged = [
            {
                "file_name": r.file_name,
                "suspicion_score": r.suspicion_score,
                "reason": r.reason,
            }
            for r in res.image_results
            if r.is_flagged
        ]
        flagged.sort(key=lambda x: x["suspicion_score"], reverse=True)
        ranked = sorted(res.image_results, key=lambda x: x.cosine_distance, reverse=True)

        all_results.append({
            "outlet_id": o_dir.name,
            "total_images": res.total_images,
            "flagged_images": flagged,
            "ranking": [r.file_name for r in ranked],
        })

        if i % 20 == 0 or i == len(outlet_dirs):
            print(f"  [{i}/{len(outlet_dirs)}] Processed {o_dir.name} (Flagged: {len(flagged)})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    total_flagged = sum(len(r["flagged_images"]) for r in all_results)
    outlets_flagged = sum(1 for r in all_results if r["flagged_images"])
    print(f"\nClassical Pipeline finished:")
    print(f"  - Outlets evaluated: {len(all_results)}")
    print(f"  - Outlets with flags: {outlets_flagged}")
    print(f"  - Total suspicious images: {total_flagged}")
    print(f"  - Output saved to: {output_path}")
    return all_results


if __name__ == "__main__":
    args = parse_args()
    run_classical_pipeline(args.dataset, args.output, args.outlet)
