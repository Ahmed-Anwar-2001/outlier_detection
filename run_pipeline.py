#!/usr/bin/env python3
"""
CLI entry point for the Suspicious Photo Detection pipeline.

Usage:
    python run_pipeline.py
    python run_pipeline.py --dataset dataset/dataset --output results/results.json
    python run_pipeline.py --threshold-k 2.5 --no-cache
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suspicious Photo Detection in Outlet Verification Images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py
  python run_pipeline.py --dataset dataset/dataset --threshold-k 3.0
  python run_pipeline.py --no-cache --output results/custom_results.json
        """,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/dataset"),
        help="Path to root directory containing outlet folders (default: dataset/dataset)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/results.json"),
        help="Output results file path (default: results/results.json)",
    )
    parser.add_argument(
        "--threshold-k",
        type=float,
        default=2.5,
        help="MAD multiplier for outlier threshold (default: 2.5). "
        "Lower = more sensitive, higher = more conservative.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/features"),
        help="Directory for feature cache (default: .cache/features)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable feature cache (force re-extraction)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    from src.config import PipelineConfig

    config = PipelineConfig(
        dataset_dir=args.dataset,
        output_path=args.output,
        cache_dir=args.cache_dir,
        threshold_k=args.threshold_k,
        use_cache=not args.no_cache,
        log_level=args.log_level,
        log_file=args.log_file,
    )

    from src.pipeline import Pipeline

    pipeline = Pipeline(config)
    results = pipeline.run()

    # Summary
    n_outlets_with_flags = sum(1 for r in results if r["flagged_images"])
    total_flagged = sum(len(r["flagged_images"]) for r in results)
    total_images = sum(r["total_images"] for r in results)

    print(f"\n{'='*64}")
    print(f"             SUSPICIOUS PHOTO DETECTION SUMMARY")
    print(f"{'='*64}")
    print(f"  Total Outlets Processed:    {len(results)}")
    print(f"  Total Images Evaluated:     {total_images}")
    print(f"  Outlets with Flagged Media: {n_outlets_with_flags} ({(n_outlets_with_flags/max(1, len(results)))*100:.1f}%)")
    print(f"  Total Suspicious Images:    {total_flagged}")
    print(f"  Results JSON Written to:    {config.output_path}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
