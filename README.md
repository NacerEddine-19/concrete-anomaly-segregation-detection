# 🏗️ Concrete Inspector

AI-powered concrete surface anomaly detection and characterisation.

---

## Pipeline

```
Image upload
    │
    ▼
Roboflow (concrete-seg)  ──► Crop each detected concrete region
    │
    ▼ (per crop)
ResNet-18 Classifier ──► crack | crack_segregation | segregation | normal
    │
    ├── [segregation / crack_segregation] ──────────────────────────────┐
    │        │                                                           │
    │        ▼                                                           │
    │   U-Net (best_unet.pth)                                           │
    │        │                                                           │
    │        ▼                                                           │
    │   Binary anomaly mask                                             │
    │        │                                                           │
    │        ▼                                                           │
    │   GAI + SI_ia extraction                                          │
    │        │                                                           │
    │        ▼                                                           │
    │   Defect cluster assignment [0 / 1 / 2]                          │
    │        │                                                           │
    │        ▼                                                           │
    │   Static material cluster means lookup ◄──────────────────────────┘
    │        │
    │        ▼
    │   Full 4-panel inspection card
    │   [Image | Identity | Defect signals | Material means]
    │
    └── [normal / crack] ──► Simple 2-panel card
                              [Image | Class probabilities]
```

---

## Project Structure

```
concrete_inspector/
├── app.py                    # Streamlit entry-point
├── constants.py              # CLASSES, MATERIAL_CLUSTER_MEANS, thresholds
├── requirements.txt
├── saved_models/
│   ├── best_unet.pth         # ← place your U-Net weights here
│   └── best_classifier.pth   # ← place your ResNet-18 weights here
└── core/
    ├── __init__.py
    ├── architectures.py      # UNet, DoubleConv, SimpleCNNSeg, build_classifier
    ├── pipeline.py           # preprocess, classify, segment, extract_features, SI_ia
    ├── roboflow_seg.py       # Roboflow concrete detection wrapper
    └── visualization.py      # render_full_card, render_simple_card
```

---

## Setup

### 1 · Install dependencies
```bash
pip install -r requirements.txt
```

### 2 · Place model weights
```
saved_models/
  best_unet.pth          ← U-Net segmentation (binary anomaly mask)
  best_classifier.pth    ← ResNet-18 4-class classifier
```

### 3 · Configure Roboflow
Either via the Streamlit sidebar, or via environment variables:
```bash
export ROBOFLOW_API_KEY="your_key_here"
export ROBOFLOW_MODEL_ID="concrete-detection-lvb3q/1"
```

### 4 · Run
```bash
streamlit run app.py
```

---

## Architectures (aligned with `Concrete_Anomaly_Detection_v2.ipynb`)

| Model | File | Architecture | Role |
|---|---|---|---|
| U-Net | `best_unet.pth` | `UNet(in=3, out=1, features=[64,128,256,512])` | Pixel-level binary segmentation |
| Classifier | `best_classifier.pth` | ResNet-18 + `Dropout(0.4)→Linear(256)→ReLU→Dropout(0.3)→Linear(4)` | 4-class crop classification |

---

## Segregation Index (SI_ia)

Implemented in `core/pipeline.py → extract_features()`:

```
1. GAI = anomaly_pixels / total_pixels
2. Divide mask into 10×10 grid
3. For each cell: LAI = anomaly_pixels_in_cell / cell_total
4. LAD = |LAI − GAI|
5. LDC = mean(LADs)
6. SI_ia = LDC / (2 × GAI × (1 − GAI)) × 100
```

---

## Material Cluster Means (from `material_set_enriched.ipynb`)

| Cluster | Damage | Strength | Age | Water |
|---|---|---|---|---|
| 0 | Low | 54.5 MPa | 33 days | 163.2 kg/m³ |
| 1 | Medium | 35.8 MPa | 90 days | 185.7 kg/m³ |
| 2 | High | 19.2 MPa | 180 days | 207.4 kg/m³ |

Mapping: `defect_cluster i ↔ material_cluster i` (symmetric severity).

---

## Defect Cluster Assignment Thresholds

| Condition | Cluster |
|---|---|
| GAI < 5% (unreliable) | 0 (low) |
| GAI < 15% AND SI < 30 | 0 (Stage 1) |
| GAI < 35% AND SI ≥ 30 | 1 (Stage 2) |
| GAI ≥ 35% AND SI ≥ 60 | 2 (Stage 3) |
| GAI ≥ 35% AND SI < 40 | 2 (Stage 4) |
| Transitional zone | 1 (medium) |
