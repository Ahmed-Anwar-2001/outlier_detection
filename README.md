# Suspicious Photo Detection in Outlet Verification Images
## Three Complementary Approaches to Visual Outlier Detection

This repository implements **three distinct, production-grade approaches** to solve the outlet verification photo anomaly detection task defined in [AI_Engineer_Assignment_Suspicious_Photo_Detection.docx.pdf](AI_Engineer_Assignment_Suspicious_Photo_Detection.docx.pdf).

Field verification photos taken over months exhibit **multi-modal legitimate states** (an outlet visited when open with product counters vs. closed with roll-down corrugated iron shutters). Field fraud involves field workers uploading unrelated scenes (residential rooms, roads, personal photos) or wrong storefronts (a pharmacy photo submitted for a grocery shop) to mark visits as "done."

### Summary of the 3 Approaches

| Approach | Name | Core Technology | Methodology & Feature Space | Deliverable File |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Classical Multi-Signal Computer Vision** | CPU-first Computer Vision (OpenCV / NumPy) | $4 \times 4$ Spatial HSV/RGB color moments + directional Sobel edge gradient histograms + low-frequency 2D-DCT layout. Multi-instance $k$-NN with adaptive MAD thresholding. | [`results/results_classical.json`](results/results_classical.json) |
| **2** | **Deep Vision Metric Embeddings (OpenCLIP)** | Foundation Vision Transformer (ViT-B-32) | Continuous 512-dimensional visual-semantic unit hypersphere. Multi-instance nearest-neighbor ($k$-NN) cosine distance with robust Median Absolute Deviation (MAD). | **[`results/results.json`](results/results.json)** (Primary) & [`results/results_clip.json`](results/results_clip.json) |
| **3** | **Vision-Language Model (Groq Qwen 3.6 27B)** | Zero-shot Multimodal VLM Auditing | Structured semantic profiling (scene validity, business category, structural fixtures, permanent signage) + cross-visit consensus reasoning. | [`results/results_vlm.json`](results/results_vlm.json) |

---

## ⚡ Quick Start: Running the 3 Approaches

### 1. Environment Setup
```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Or install dependencies
pip install -r requirements.txt
```

### 2. Execution Commands for Each Approach

```bash
# -------------------------------------------------------------
# Approach 1: Classical Multi-Signal Computer Vision (Zero GPU)
# -------------------------------------------------------------
python run.py --pipeline classical

# -------------------------------------------------------------
# Approach 2: OpenCLIP Deep Vision Embeddings (Primary)
# -------------------------------------------------------------
python run.py --pipeline clip

# -------------------------------------------------------------
# Approach 3: Groq Qwen 3.6 27B Vision-Language Model
# -------------------------------------------------------------
# Run on a sample outlet:
python run.py --pipeline vlm --outlet outlet_003a29a9
# Or run on the full dataset:
python run.py --pipeline vlm

# -------------------------------------------------------------
# Run All 3 Pipelines Sequentially
# -------------------------------------------------------------
python run.py --pipeline all
```
*Note: `python run_pipeline.py` is also available as a backward-compatible entry point.*

---

## 🏗️ Detailed Methodology for Each Approach

```
                                  ┌───────────────────────────┐
                                  │   Dataset Loader (159)    │
                                  └─────────────┬─────────────┘
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                      ▼                                      ▼
┌──────────────────────────┐          ┌──────────────────────────┐          ┌──────────────────────────┐
│  Approach 1: Classical   │          │   Approach 2: OpenCLIP   │          │    Approach 3: Groq VLM  │
│  (Zero-GPU Baseline)     │          │   (Primary Production)   │          │   (Deep Semantic Audit)  │
├──────────────────────────┤          ├──────────────────────────┤          ├──────────────────────────┤
│• 4x4 Spatial HSV moments │          │• ViT-B-32 dense features │          │• Qwen 3.6 27B Vision     │
│• Sobel edge orientation  │          │• 512-dim embedding space │          │• Shop type classification│
│• Low-frequency 2D-DCT    │          │• Invariant to angle/zoom │          │• Facade & shutter anchors│
│• Multi-instance k-NN     │          │• Multi-instance k-NN     │          │• Signboard token overlap │
│• Adaptive MAD threshold  │          │• Adaptive MAD threshold  │          │• Scene validity checking │
└────────────┬─────────────┘          └────────────┬─────────────┘          └────────────┬─────────────┘
             │                                     │                                     │
             ▼                                     ▼                                     ▼
  results_classical.json                 results/results.json                    results_vlm.json
```

### Approach 1: Classical Multi-Signal Computer Vision (`pipelines/classical_cv/`)
- **Visual Representation**: Handcrafted multi-signal visual descriptor fusing 3 complementary sub-vectors into $\mathbf{v}_i \in \mathbb{R}^{320}$:
  1. *Spatial Color Distributions*: $4 \times 4$ spatial grid computing mean ($\mu$) and standard deviation ($\sigma$) across RGB and HSV color spaces, plus global Hue/Saturation histograms.
  2. *Structural Gradient Orientations*: Directional Sobel filters ($\mathbf{S}_x, \mathbf{S}_y$) computing 8-bin gradient orientation histograms across spatial cells, capturing horizontal roll-up shutter ridges, counter lines, and framing.
  3. *Frequency Texture*: Low-frequency 2D Discrete Cosine Transform (DCT) coefficients capturing coarse spatial layout.
- **Scoring & Outlier Detection**: Pairwise cosine distance evaluated via multi-instance $k$-NN ($k=\min(2, N-1)$) against the outlet's visit clusters, with adaptive Median Absolute Deviation (MAD) thresholding.
- **Strength**: Ultra-fast (~15ms per image on CPU), zero neural dependencies, zero GPU required.

### Approach 2: Deep Vision Metric Embeddings (`pipelines/clip_embeddings/`) — Primary
- **Visual Representation**: OpenCLIP ViT-B-32 transforms each image into a continuous 512-dimensional semantic embedding ($\mathbf{e}_i \in \mathbb{S}^{511}$). Pretrained representations provide high invariance to extreme camera perspective, framing, daylight variation, and distance.
- **Scoring Method (Multi-Instance $k$-NN)**: Evaluates consistency against nearest neighbors in the outlet series:
  $$D_{\text{multi}}(i) = 0.5 \times \min_{j \neq i} d(i, j) + 0.5 \times \left(\frac{1}{k} \sum_{j \in \mathcal{N}_k(i)} d(i, j)\right)$$
  This avoids the fatal flaw of single-centroid averaging, allowing legitimate open and closed states to coexist.
- **Outlier Rule (Adaptive MAD)**: $\text{Threshold} = \text{Median}(D) + 2.2 \times \text{MAD}(D)$. Because MAD has a 50% breakdown point, the threshold remains rock-solid even when up to 50% of images in a folder are fraudulent.
- **Strength**: State-of-the-art semantic generalization, fast batched inference, robust clustering.

### Approach 3: Vision-Language Model Semantic Auditing (`pipelines/vlm_groq/`)
- **Visual Representation**: Zero-shot structured profiling with Groq Qwen 3.6 27B extracting:
  - `scene_type` (`storefront_exterior`, `storefront_interior`, `closed_storefront`, `unrelated_scene`, `unclear_blurry`)
  - `business_category` (`grocery_general_store`, `telecom_recharge`, `pharmacy`, `tea_stall_restaurant`, `clothing_tailor`, `non_commercial`)
  - `architectural_features` (`roll_down_shutter`, `corrugated_tin_roof`, `wooden_counter`, `glass_display_cabinet`, `brick_facade`)
  - `signboard_name` and corporate `brand_sponsors`.
- **Scoring & Outlier Detection**: Forms a consensus outlet identity across all visits. Flags true anomalies across orthogonal semantic dimensions:
  1. *Unrelated Scene*: Living room, road, selfie, non-store object (Score: 0.95).
  2. *Business Category Conflict*: Pharmacy photo submitted for grocery history (Score: 0.88).
  3. *Signage Mismatch*: Conflicting storefront name without shared visual anchors (Score: 0.90).
  4. *Multi-Modal Protection*: Closed roll-down shutter matching outlet facade/fixtures is recognized as legitimate (Score: 0.05).
- **Strength**: Human-level reasoning, explicit business classification, and natural language audit explanations.

---

## 🛠️ CLI Options

```bash
python run.py [OPTIONS]

Options:
  --pipeline {clip,classical,vlm,all}   Which pipeline to run (default: clip)
  --dataset PATH                        Root directory of outlet folders (default: dataset/dataset)
  --outlet STR                          Optional: Process a single outlet ID (e.g. outlet_003a29a9)
  --output PATH                         Optional: Custom path for output results JSON
```

---

## 📋 Deliverables & Output Schema

The output strictly follows the schema defined in the assignment specification:

```json
{
  "outlet_id": "outlet_003a29a9",
  "total_images": 8,
  "flagged_images": [
    {
      "file_name": "image_0008.jpg",
      "suspicion_score": 0.92,
      "reason": "Semantic outlier — low visual embedding similarity to outlet visit clusters; distinct background/signage and storefront structure"
    }
  ],
  "ranking": [
    "image_0008.jpg",
    "image_0001.jpg",
    "image_0002.jpg",
    "image_0003.jpg",
    "image_0004.jpg",
    "image_0005.jpg",
    "image_0006.jpg",
    "image_0007.jpg"
  ]
}
```

---

## 📂 Project Structure

```
inteligent_machines/
├── pipelines/
│   ├── classical_cv/          # Multi-Signal Spatial CV (HSV + Sobel + DCT)
│   │   ├── feature_extractor.py
│   │   ├── anomaly_detector.py
│   │   ├── config.py
│   │   └── run.py
│   ├── clip_embeddings/       # OpenCLIP ViT-B-32 Deep Embedding Pipeline (Primary)
│   │   ├── extractor.py
│   │   ├── detector.py
│   │   ├── config.py
│   │   └── run.py
│   └── vlm_groq/              # Groq Qwen 3.6 27B Semantic Visual Audit Pipeline
│       ├── prompts.py         # Visual semantic auditing schema
│       ├── client.py          # Resilient Groq VLM API client with caching
│       ├── detector.py        # Consensus-driven visual outlier detection
│       ├── config.py
│       └── run.py
├── run.py                     # Unified CLI runner (--pipeline {clip,classical,vlm,all})
├── run_pipeline.py            # Backward-compatible CLI entry point
├── results/
│   ├── results.json           # Primary final deliverable file for full dataset (159 outlets)
│   ├── results_clip.json      # Full dataset results from OpenCLIP pipeline
│   ├── results_classical.json # Full dataset results from Classical CV pipeline
│   └── results_vlm.json       # Results from Groq VLM semantic auditing pipeline
├── dataset/dataset/           # Dataset directory (159 outlet folders, 2042 images)
├── requirements.txt           # Python dependencies
├── writeup.md                 # 1-page technical write-up & methodology justification
└── README.md                  # System documentation & usage guide
```

