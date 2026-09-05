"""
Unified Runner for Suspicious Photo Detection Pipelines.

Usage:
  python run.py --pipeline clip                          # Run CLIP pipeline on full dataset
  python run.py --pipeline classical                     # Run Classical CV pipeline on full dataset
  python run.py --pipeline vlm                           # Run Groq VLM pipeline
  python run.py --pipeline all                           # Run all pipelines
  python run.py --pipeline clip --outlet outlet_003a29a9 # Run CLIP on a single outlet

Outputs are saved in the results/ folder:
  - results/results_classical.json
  - results/results_clip.json
  - results/results_vlm.json
"""
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Suspicious Photo Detection in Outlet Verification Images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline",
        choices=["classical", "clip", "vlm", "all"],
        default="clip",
        help="Which pipeline to run: 'classical', 'clip', 'vlm', or 'all' (default: clip)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=BASE_DIR / "dataset" / "dataset",
        help="Path to dataset directory (default: dataset/dataset)",
    )
    parser.add_argument(
        "--outlet",
        type=str,
        default=None,
        help="Optional: Run only on a specific outlet ID (e.g. outlet_003a29a9)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional: Custom output JSON path (defaults to results/results_<pipeline>.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.pipeline in ("classical", "all"):
        print("\n" + "=" * 65)
        print("PIPELINE 1: Classical Multi-Signal CV (Color + Edge + DCT)")
        print("=" * 65)
        from pipelines.classical_cv.run import run_classical_pipeline
        out_path = args.output if (args.output and args.pipeline == "classical") else results_dir / "results_classical.json"
        run_classical_pipeline(dataset_dir=args.dataset, output_path=out_path, outlet_id=args.outlet)

    if args.pipeline in ("clip", "all"):
        print("\n" + "=" * 65)
        print("PIPELINE 2: CLIP Visual Embeddings (OpenCLIP ViT-B-32)")
        print("=" * 65)
        from pipelines.clip_embeddings.run import run_clip_pipeline
        out_path = args.output if (args.output and args.pipeline == "clip") else results_dir / "results_clip.json"
        run_clip_pipeline(dataset_dir=args.dataset, output_path=out_path, outlet_id=args.outlet)

    if args.pipeline in ("vlm", "all"):
        print("\n" + "=" * 65)
        print("PIPELINE 3: Vision-Language Model (Groq Qwen 3.6 27B)")
        print("=" * 65)
        from pipelines.vlm_groq.run import run_vlm_pipeline
        out_path = args.output if (args.output and args.pipeline == "vlm") else results_dir / "results_vlm.json"
        run_vlm_pipeline(dataset_dir=args.dataset, output_path=out_path, outlet_id=args.outlet)

    print("\n" + "=" * 65)
    print("Execution complete. All requested pipeline results are saved in results/")
    print("=" * 65)


if __name__ == "__main__":
    main()
