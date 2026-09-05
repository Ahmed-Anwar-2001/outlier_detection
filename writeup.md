# Take-Home Assignment: Suspicious Photo Detection in Outlet Verification Images
**Candidate Technical Write-Up & Methodology Report**

---

## 1. Approach & Visual Representation Rationale

Field agent verification photos for retail and mobile-recharge outlets (such as bKash outlets in Bangladesh) present unique challenges:
- Captures are taken during different visits under shifting daylight, weather, and camera orientations.
- Storefront identity is defined by a combination of **branding/signage palette** (e.g., magenta bKash banners), **storefront geometry** (horizontal roll-up shutters, counters, display racks), and **spatial scene layout**.
- Individual fraudulent photos consist of random storefronts, indoor spaces, or arbitrary objects photographed to mark visits complete.

To address this reliably without external system dependencies, our production pipeline employs a **Multi-Signal Visual Descriptor** coupled with **Robust Non-Parametric Outlier Detection**:

### Multi-Signal Visual Descriptor
1. **Spatial Grid Color Moments & Histograms (HSV Space)**:
   - Rather than a global color histogram that ignores structure, the image is decomposed into a $3 \times 3$ spatial grid (top signage, middle counter, bottom street/pavement).
   - Each cell extracts 1st and 2nd color moments ($\mu, \sigma$) alongside 16-bin Hue, 8-bin Saturation, and 8-bin Value distributions. This catches signage mismatches and wall repaints while remaining tolerant of minor lighting shifts.
2. **Directional Edge & Structural Gradients**:
   - Sobel horizontal ($\mathbf{S}_y$) and vertical ($\mathbf{S}_x$) gradient operators quantify structural orientations. Horizontal shutter ridges and counter planes produce strong horizontal dominance, contrasting sharply with unstructured outdoor scenes or cluttered indoor rooms.
3. **Perceptual Frequency Signatures (2D-DCT)**:
   - Low-frequency 2D Discrete Cosine Transform coefficients capture coarse spatial geometry and scene layout, providing invariant structural fingerprints.

The sub-vectors are $L_2$-normalized and fused into a unified unit vector $\mathbf{v}_i \in \mathbb{R}^D$, while retaining decomposed sub-channel distances for explainability.

---

## 2. Similarity & Outlier Detection Methodology

### Metric: Cosine Distance from Series Centroid
Given $N$ images for an outlet with normalized feature vectors $\{\mathbf{v}_1, \dots, \mathbf{v}_N\}$:
1. **Outlet Identity Centroid**:
   $$\mathbf{c} = \frac{\sum_{i=1}^N \mathbf{v}_i}{\left\|\sum_{i=1}^N \mathbf{v}_i\right\|_2}$$
   $\mathbf{c}$ represents the visual identity of the outlet across visits.
2. **Cosine Distance**:
   $$d_i = 1 - (\mathbf{v}_i \cdot \mathbf{c})$$

### Adaptive Threshold: Median Absolute Deviation (MAD)
Outlier cutoff thresholds must be computed dynamically per outlet because some outlets exhibit tightly controlled views while others have slight angle variance across visits:
$$\text{Threshold} = \text{Median}(d) + k \times \text{MAD}(d)$$
where $\text{MAD}(d) = \text{Median}(|d_i - \text{Median}(d)|)$ and $k = 2.5$.

**Justification over Standard Deviation ($\mu + 3\sigma$) and IQR**:
- Standard deviation has a **breakdown point of 0%** — a single extreme outlier artificially inflates the standard deviation $\sigma$, raising the threshold and allowing other outliers to hide (masking effect).
- MAD has a **breakdown point of 50%**, ensuring that up to half the images in a folder could be anomalous without corrupting the reference scale.
- For small sample sizes ($N \in [5, 40]$), MAD is markedly more stable than boxplot IQR methods.

### Normalized Suspicion Score & Domain Reason Generation
- Cosine distances are min-max mapped to $[0.0, 1.0]$ within each outlet folder.
- Decomposed sub-signals (color deviation vs. structural edge deviation vs. layout deviation) are evaluated against peer images to synthesize clear, human-readable explanations (e.g., *"divergent edge geometry and storefront structural lines; major visual discrepancy from established identity"*).

---

## 3. Trade-Offs & Alternative Architectures

| Approach | Strengths | Trade-Offs / Drawbacks | Selected? |
|---|---|---|:---:|
| **Multi-Signal CV (Spatial HSV + Sobel + DCT)** | Zero C++ runtime issues, ultra-fast (~100ms/img CPU), completely interpretable channel deviations | Requires domain-tailored feature weighting | **Yes (Primary)** |
| **CLIP (ViT-B/32 or ViT-L/14)** | High semantic abstraction | Heavy weight, requires torch C++ runtime binaries, higher latency | Alternative |
| **ResNet-50 / EfficientNet** | Established classification features | Pretrained on ImageNet object classes; over-indexes on foreground objects rather than storefront architecture | No |
| **Isolation Forest / One-Class SVM** | Non-parametric density estimation | Uninterpretable decision boundaries; poor stability on $N < 10$ samples | No |

---

## 4. Production Scalability & Serving Architecture

To scale this pipeline to **100,000+ outlets**:
1. **Asynchronous Ingestion & Pre-Caching**:
   - Feature extraction runs worker-side during image upload. Vectors (a compact few hundred floats per photo) are persisted in object storage or a vector database.
2. **$O(1)$ Incremental Centroid Updates**:
   - New visits update the running mean vector $\mathbf{c}$ in $O(1)$ time without re-processing historical images.
3. **Cross-Outlet Duplicate Detection (Fraud Rings)**:
   - The same feature vectors can be indexed in an HNSW / FAISS index to detect when the same photo is fraudulently submitted across distinct outlet IDs.

---

## 5. Known Limitations & Failure Modes

1. **Full Storefront Renovations**: If an outlet changes branding or undergoes remodeling, older valid photos may register higher distance to the new centroid. *Mitigation: Rolling temporal decay window when capture timestamps are available.*
2. **Extreme Close-Ups vs. Wide-Angle Shots**: A close-up of a QR code sticker vs. a street-view photo will differ in layout. *Mitigation: Multi-crop or hierarchical clustering when visit clusters have multiple standard photo types (counter vs. signage).*
