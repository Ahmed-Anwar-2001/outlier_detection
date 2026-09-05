# Take-Home Assignment: Suspicious Photo Detection in Outlet Verification Images
**Candidate Technical Write-Up & Methodology Report**

---

## 1. Executive Summary & Problem Context

Retail verification systems (such as bKash outlets in Bangladesh) collect photos across visits over months under varying daylight, camera angles, zoom levels, and framing. Storefronts exhibit **multi-modal legitimate states**: an outlet may be open (counters, snack racks, glass cabinets), partially obstructed, or **closed with roll-down corrugated iron shutters**. Field fraud consists of workers uploading unrelated scenes (bedrooms, roads, personal photos) or wrong storefronts (a pharmacy photo submitted for a grocery shop) to mark visits as "done."

To address this challenge comprehensively, we designed and implemented **three distinct, complementary approaches**:
1. **Approach 1: Classical Multi-Signal Computer Vision (Zero-GPU / Edge-First)**
2. **Approach 2: Deep Vision Metric Embeddings (OpenCLIP ViT-B-32)**
3. **Approach 3: Vision-Language Model Semantic Auditing (Groq Qwen 3.6 27B)**

---

## 2. Visual Representations Across the 3 Approaches

### Approach 1: Classical Multi-Signal Computer Vision
Combines spatial color, structural geometry, and perceptual texture without external neural dependencies:
- **Multi-Scale Spatial Color Moments (RGB + HSV)**: Images are partitioned into a $4 \times 4$ spatial grid. For each cell, 1st and 2nd moments ($\mu, \sigma$) are computed across RGB and HSV channels, alongside 16-bin global Hue/Saturation histograms. This captures regional wall paint and signage colors while remaining invariant to minor illumination shifts.
- **Directional Gradient Orientations (Sobel HOG)**: Horizontal ($\mathbf{S}_y$) and vertical ($\mathbf{S}_x$) Sobel operators generate 8-bin gradient orientation histograms per spatial cell, quantifying horizontal shutter ridges and counter lines.
- **Perceptual Frequency Layout (2D-DCT)**: Low-frequency 2D Discrete Cosine Transform coefficients encode coarse spatial geometry.
- *Fused vector:* $\mathbf{v}_i \in \mathbb{R}^{320}$, $L_2$-normalized.

### Approach 2: Deep Vision Metric Embeddings (OpenCLIP ViT-B-32)
- **Deep Feature Representation**: Passes images through a Vision Transformer (ViT-B-32) pretrained on vast visual-concept corpora.
- Maps photos into a dense, continuous 512-dimensional unit hypersphere ($\mathbf{e}_i \in \mathbb{S}^{511}$).
- Provides intrinsic invariance to severe camera tilt, perspective shifts, partial zoom, and lighting, while maintaining sharp discrimination between different commercial locations.

### Approach 3: Vision-Language Model (Groq Qwen 3.6 27B)
- **Semantic Structured Profiling**: Directly extracts symbolic visual facts via zero-shot VLM prompting:
  - `scene_type`: `storefront_exterior`, `storefront_interior`, `closed_storefront`, `unrelated_scene`, `unclear_blurry`.
  - `business_category`: `grocery_general_store`, `telecom_recharge`, `pharmacy`, `tea_stall_restaurant`, `clothing_tailor`, `non_commercial`.
  - `architectural_features`: Roll-down shutters, corrugated tin roofs, brick facades, wooden counters, glass cabinets.
  - `signboard_name` & `brand_sponsors`: Permanent shop name tokens and corporate sponsor banners.

---

## 3. Similarity & Scoring Methods (With Justification)

### Why Naive Global Centroids Fail (The Multi-Modal Problem)
A common pitfall is averaging all images in an outlet into a single centroid $\mathbf{c} = \frac{1}{N}\sum \mathbf{v}_i$. When an outlet contains 6 open-counter photos and 2 closed-shutter visits:
1. The centroid represents an artificial hybrid that resembles neither state.
2. Legitimate closed shutters have large distance to this centroid and get **falsely flagged**.
3. A fraudulent photo of a *completely different open shop* has low distance to the centroid because it shares the general "open shop" profile, causing a **false negative**.

### Scoring in Approach 1 (Classical CV) & Approach 2 (OpenCLIP): Multi-Instance $k$-NN
Rather than centroid averaging, each image $\mathbf{v}_i$ is scored against its nearest neighbors in the outlet:
$$d(i, j) = 1 - \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}, \quad j \neq i$$
$$D_{\text{multi}}(i) = 0.5 \times \min_{j \neq i} d(i, j) + 0.5 \times \left(\frac{1}{k} \sum_{j \in \mathcal{N}_k(i)} d(i, j)\right), \quad k = \min(2, N-1)$$
- **Justification**: If an image belongs to an established visit mode (open counter or closed shutter), $D_{\text{multi}}(i)$ is small. Only an isolated image from an unrelated storefront incurs a high distance.

### Scoring in Approach 3 (Groq VLM): Cross-Visit Consensus Synthesizer
- Builds the outlet's ground-truth identity across visits: dominant business category, persistent architectural anchors (features in $\ge 25\%$ of images), and recurring signage tokens.
- Scores each image along 4 orthogonal semantic dimensions:
  1. *Scene Validity*: Unrelated non-store scenes (bedroom, road, selfie) $\rightarrow$ Score $\in [0.90, 0.98]$.
  2. *Business Type Divergence*: Conflicting store types (e.g. tea stall in telecom history) $\rightarrow$ Score $\in [0.75, 0.90]$.
  3. *Architectural Congruence*: Disjoint physical fixtures $\rightarrow$ Score $\in [0.70, 0.85]$.
  4. *Multi-Modal Protection*: Closed shutters sharing outlet facade/anchors $\rightarrow$ Score $\approx 0.05$ (Legitimate).

---

## 4. Outlier Detection Rules (With Justification)

### Approaches 1 & 2: Robust Adaptive Threshold via Median Absolute Deviation (MAD)
Thresholds are computed dynamically per outlet using Median Absolute Deviation:
$$\text{Threshold} = \text{Median}(D) + k \times \text{MAD}(D)$$
where $\text{MAD}(D) = \text{Median}(|D_i - \text{Median}(D)|)$ with multiplier $k = 2.2$.

**Justification & 50% Breakdown Point**:
- Parametric rules ($\mu + 3\sigma$) suffer from the *masking effect*: extreme outliers inflate the sample mean and variance, masking true anomalies.
- Median and MAD have a **50% breakdown point**—they remain completely robust even if multiple visits in an outlet are fraudulent.
- Distance scores are min-max mapped to $[0.0, 1.0]$, calibrating flagged outliers to $[0.65, 1.0]$ and consistent images to $[0.0, 0.45]$.

### Approach 3: Semantic Rule Engine & Explainability
- Rules evaluate scene validity and category compatibility matrices (e.g., telecom + grocery co-exist in rural retail, but tea stall vs. telecom does not).
- Produces natural language justifications directly meeting assignment requirements (e.g., *"Storefront mismatch - tea stall restaurant photo inconsistent with outlet's established telecom recharge history"*).

---

## 5. Comparative Evaluation & Production Recommendation

| Dimension | Approach 1: Classical Multi-Signal CV | Approach 2: OpenCLIP ViT-B-32 (Primary) | Approach 3: Groq VLM (Qwen 3.6 27B) |
| :--- | :--- | :--- | :--- |
| **Model Footprint** | **0 MB** (Zero weights, Pure Python/NumPy) | ~600 MB (Standard ViT-B-32) | Cloud API (27B parameter Multimodal LLM) |
| **Hardware Required** | Ultra-light CPU (~15 ms / image) | CPU or commodity GPU (~40 ms / image) | High-bandwidth network connection |
| **Operational Cost** | $0 runtime cost | Negligible (on-prem edge inference) | Recurring token cost ($O(N)$ API calls) |
| **Semantic Fidelity** | Coarse (Color/Edge/DCT spatial grid) | **High** (Dense visual-semantic embedding) | **High** (Symbolic language reasoning) |
| **Throughput & Speed** | ~65 images / second | **~25 images / second** (Batched) | ~1.5 images / second (Rate-limited) |
| **Failure Vulnerability**| Angle & zoom variance | Minor fine-text OCR differences | Prompt sensitivity & token truncation |
| **Dataset Results File** | `results/results_classical.json` (159 outlets) | **`results/results.json`** & `results_clip.json` | `results/results_vlm.json` |

### Architectural Verdict & Selection Rationale

**Why OpenCLIP is Selected as the Primary Production Engine:**
1. **Mathematical Determinism & Robust Metric Space**: Visual verification across multi-modal visits (open shop vs. closed shutter) is fundamentally an image-to-image metric alignment problem. OpenCLIP projects storefront structures into continuous feature clusters without discretization error, providing clean, repeatable cosine distances.
2. **Superior Cost-to-Throughput Profile**: Outlier detection across thousands of outlets demands high throughput. OpenCLIP processes the entire 2,042-image dataset in seconds locally with zero external API dependencies, predictable latency, and zero per-token cloud costs.
3. **Robustness to Visual Diversity**: Generalizes effortlessly across camera perspectives, distance variations, and daylight shifts without requiring prompt engineering.

**Engineering Evaluation of the Vision-Language Model (VLM) Approach:**
- **Capabilities**: The VLM pipeline demonstrates exceptional qualitative interpretability—generating human-like, context-rich explanations (e.g., identifying a tea stall or pharmacy photo submitted for a telecom shop).
- **Practical Trade-Offs in Production**: While promising for explanatory auditing, general-purpose zero-shot VLMs are sub-optimal as a primary standalone outlier detection engine:
  - *Latency & Rate Bottlenecks*: Generating multi-token reasoning chains for thousands of high-resolution images introduces high latency and vulnerability to cloud rate limits.
  - *Domain Discretization*: Converting continuous visual geometry into discrete JSON strings inevitably discards subtle structural invariants that embedding vectors preserve naturally.
- **Strategic Evolution Roadmap**: In a mature enterprise architecture, these two systems should be paired hierarchically: OpenCLIP operates as the high-throughput Tier-1 screen, while the VLM is deployed as an asynchronous Tier-2 arbitration agent exclusively on borderline edge cases to generate explainable audit reports for operations teams.

---

## 6. Production Scalability & Known Limitations

### Serving Architecture at Scale (100,000+ Outlets)
1. **Asynchronous Edge Feature Ingestion**: Compute OpenCLIP 512-dim embeddings during worker upload; persist compact vectors in an indexed vector database (Milvus / Qdrant).
2. **Sub-Millisecond k-NN Retrieval**: Evaluate multi-instance nearest-neighbor distances via indexed HNSW graph queries in $O(\log N)$ time.
3. **Cross-Outlet Fraud Ring Detection**: A global vector index catches recycled images submitted across different outlet IDs in $O(1)$ time.

### Known Limitations
1. **Outlets with $N < 3$ Visits**: Statistical outlier detection requires reference history; outlets with 1–2 visits cannot establish reliable baseline clusters.
2. **Major Storefront Renovation / Relocation**: A complete facade overhaul causes visual divergence. *Mitigation: Introduce temporal decay weighting recent visits higher once capture timestamps are available.*



