"""
Groq VLM Suspicious Photo Detection Pipeline (Qwen 3.6 27B).
Semantic visual consensus auditing (scene context, business type, architectural fixtures, signage).
Saves results to results/results_vlm.json.
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pipelines.vlm_groq.config import VlmConfig
from pipelines.vlm_groq.client import GroqVlmClient
from pipelines.vlm_groq.detector import VlmAnomalyDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Groq VLM Suspicious Photo Detection Pipeline (Qwen 3.6 27B)")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "dataset" / "dataset", help="Path to dataset directory")
    parser.add_argument("--outlet", type=str, default=None, help="Evaluate a single outlet ID (e.g. outlet_003a29a9)")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "results_vlm.json", help="Output JSON path")
    parser.add_argument("--api-key", type=str, default=None, help="Optional explicit Groq API key")
    return parser.parse_args()


def process_single_outlet(outlet_dir: Path, client: GroqVlmClient, detector: VlmAnomalyDetector):
    image_paths = sorted([
        p for p in outlet_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    ])

    print(f"\nProcessing {outlet_dir.name} ({len(image_paths)} images)...")
    profiles = []
    for p in image_paths:
        print(f"  Analyzing {p.name} with {client.config.model_name}...")
        prof = client.analyze_image(p)
        scene = prof.get("scene_type")
        cat = prof.get("business_category")
        sign = prof.get("signboard_name")
        arch = prof.get("architectural_features") or []
        print(f"    -> Scene: {scene} | Category: {cat} | Sign: {sign} | Arch: {arch[:3]}")
        profiles.append({"file_name": p.name, "profile": prof})

    res = detector.evaluate_outlet(outlet_dir.name, profiles)
    print(f"  Outlet Consensus Identity -> Category: {res.consensus_category} | Sign: {res.consensus_signboard} | Anchors: {res.consensus_landmarks}")

    flagged = [
        {
            "file_name": e.file_name,
            "suspicion_score": e.suspicion_score,
            "reason": e.reason,
        }
        for e in res.evaluations
        if e.is_flagged
    ]
    flagged.sort(key=lambda x: x["suspicion_score"], reverse=True)
    ranked = sorted(res.evaluations, key=lambda x: x.suspicion_score, reverse=True)

    return {
        "outlet_id": outlet_dir.name,
        "total_images": res.total_images,
        "flagged_images": flagged,
        "ranking": [e.file_name for e in ranked],
    }


def run_vlm_pipeline(dataset_dir: Path = None, output_path: Path = None, outlet_id: str = None, api_key: str = None):
    if dataset_dir is None:
        dataset_dir = PROJECT_ROOT / "dataset" / "dataset"
    if output_path is None:
        output_path = PROJECT_ROOT / "results" / "results_vlm.json"

    config = VlmConfig(dataset_dir=dataset_dir, output_path=output_path)
    client = GroqVlmClient(config, api_key=api_key)
    detector = VlmAnomalyDetector()

    if outlet_id:
        target_dir = dataset_dir / outlet_id
        if not target_dir.exists():
            raise FileNotFoundError(f"Outlet directory not found: {target_dir}")
        outlets = [target_dir]
    else:
        outlets = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

    all_results = []
    print(f"Running VLM Pipeline on {len(outlets)} outlet(s)...")

    for o_dir in outlets:
        record = process_single_outlet(o_dir, client, detector)
        all_results.append(record)

        # Save incrementally
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    total_flagged = sum(len(r["flagged_images"]) for r in all_results)
    outlets_flagged = sum(1 for r in all_results if r["flagged_images"])
    print(f"\nVLM Pipeline finished:")
    print(f"  - Outlets evaluated: {len(all_results)}")
    print(f"  - Outlets with flags: {outlets_flagged}")
    print(f"  - Total suspicious images: {total_flagged}")
    print(f"  - Output saved to: {output_path}")
    return all_results


if __name__ == "__main__":
    args = parse_args()
    run_vlm_pipeline(args.dataset, args.output, args.outlet, args.api_key)
