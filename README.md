# Suspicious Photo Detection in Outlet Verification Images

Production-grade outlier detection pipeline that identifies fraudulent or visually inconsistent images across outlet photo histories.

---

## ⚡ Quick Start

### 1. Activate Environment
```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline
```bash
python run_pipeline.py
```
Results will be generated and saved to `results/results.json`.

---

## 🛠️ CLI Options

```bash
python run_pipeline.py [OPTIONS]

Options:
  --dataset PATH       Root directory containing outlet folders (default: dataset/dataset)
  --output PATH        Path for output results JSON (default: results/results.json)
  --threshold-k FLOAT  MAD sensitivity multiplier (default: 2.5)
  --cache-dir PATH     Feature cache directory (default: .cache/features)
  --no-cache           Bypass feature caching and force re-extraction
  --log-level LEVEL    Logging verbosity: DEBUG, INFO, WARNING, ERROR (default: INFO)
```

---

## 🏗️ Architecture

```
                          ┌───────────────────────────┐
                          │   Dataset Loader (159)    │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │ Multi-Signal Visual Feature Extractor               │
             │  • 3x3 Spatial Grid Color Distributions (HSV)       │
             │  • Directional Sobel Edge Gradients (Shutters/Frames)│
             │  • Low-Frequency 2D Discrete Cosine Transform       │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │ Robust Centroid & Anomaly Detector                  │
             │  • Identity Centroid Calculation                    │
             │  • Cosine Distance Scoring                          │
             │  • Adaptive Threshold: Median + k · MAD             │
             │  • Min-Max Normalized Suspicion Scores [0, 1]       │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │ Domain Context Reason Generator                     │
             │  • Identifies Divergent Sub-Channels (Color/Edge)   │
             │  • Nearest-Neighbor Isolation Analysis              │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │    results/results.json   │
                          └───────────────────────────┘
```

---

## 📋 Output Schema

The system produces structured, per-outlet evaluation records conforming strictly to the requested specification:

```json
{
  "outlet_id": "outlet_003a29a9",
  "total_images": 8,
  "flagged_images": [
    {
      "file_name": "image_0008.jpg",
      "suspicion_score": 1.0,
      "reason": "Highly anomalous image — distinct color palette and branding/signage mismatch, divergent edge geometry and storefront structural lines; major visual discrepancy from the outlet's established identity"
    }
  ],
  "ranking": [
    "image_0008.jpg",
    "image_0001.jpg",
    "image_0004.jpg",
    "image_0002.jpg",
    "image_0003.jpg",
    "image_0007.jpg",
    "image_0005.jpg",
    "image_0006.jpg"
  ]
}
```

---

## 📂 Project Structure

```
inteligent_machines/
├── src/
│   ├── __init__.py
│   ├── config.py              # Centralized dataclass configuration
│   ├── feature_extractor.py   # Multi-signal computer vision extractor with caching
│   ├── anomaly_detector.py    # Centroid cosine distance & MAD adaptive thresholding
│   ├── reason_generator.py    # Contextual, interpretable explanation synthesizer
│   ├── pipeline.py            # End-to-end orchestrator & schema builder
│   └── utils.py               # Dataset discovery, structured logging, file I/O
├── run_pipeline.py            # CLI entry point with argparse
├── results/
│   └── results.json           # Final deliverable results file
├── dataset/dataset/           # Verification image photo history folders
├── requirements.txt           # Pinned production dependencies
├── writeup.md                 # 1-page technical write-up & methodology justification
└── README.md                  # System documentation & usage guide
```
