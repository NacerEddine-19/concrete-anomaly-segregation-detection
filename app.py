"""
app.py — Concrete Anomaly Detection · Gradio Space
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output card (4 panels — mirrors inspect_image from material_set_enriched.ipynb):

  ┌──────────┬──────────────┬────────────────┬──────────────────────┐
  │  Image   │   Identity   │ Defect signals │  Material means      │
  │ (crop +  │ (cluster,    │ (confidence,   │  (cluster lookup:    │
  │  border) │  stage rank, │  GAI score,    │   cement, water …)   │
  │          │  GAI, SI-IA, │  SI-IA score)  │                      │
  │          │  mat. props) │                │                      │
  └──────────┴──────────────┴────────────────┴──────────────────────┘

SI_ia formula (exact from Concrete_Anomaly_Detection_v2.ipynb § extract_features):
  GAI  = anomaly_pixels / total_pixels
  LAI  = anomaly_pixels_in_cell / total_pixels_in_cell   (per 10×10 grid cell)
  LAD  = |LAI − GAI|
  LDC  = mean(LAD over all cells)
  SI_ia = LDC / (2 × GAI × (1 − GAI)) × 100   [%]

Severity mapping (4-stage, from v2 analyze_concrete):
  GAI < 5 %               → UNRELIABLE
  GAI < 15 % & SI < 30   → STAGE 1
  GAI < 35 % & SI ≥ 30   → STAGE 2
  GAI ≥ 35 % & SI ≥ 60   → STAGE 3
  GAI ≥ 35 % & SI < 40   → STAGE 4
  otherwise               → TRANSITIONAL

Defect-cluster → material-cluster mapping (material_set_enriched.ipynb § Cell 6):
  Defect cluster 0 (low damage)    ↔ Material cluster 0 (strongest concrete)
  Defect cluster 1 (medium damage) ↔ Material cluster 1 (medium concrete)
  Defect cluster 2 (high damage)   ↔ Material cluster 2 (weakest concrete)
"""

from __future__ import annotations

import io
import logging
import os
import traceback
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────────────────────
# 0 · Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Global constants
# ─────────────────────────────────────────────────────────────────────────────

YOLO_WEIGHTS       = Path("yolo26s-seg.pt")
CLASSIFIER_WEIGHTS = Path("best_classifier.pt")
SEG_WEIGHTS        = Path("yolov8n-seg.pt")

CLASSES      = ["crack", "crack_segregation", "normal", "segregation"]
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES  = len(CLASSES)

YOLO_CONF_THRESH = 0.40
SEG_THRESH       = 0.50
BBOX_PAD_PCT     = 0.10
SEG_IMG_SIZE     = 256
EDGE_MARGIN_PCT  = 0.04

GAI_MIN_THRESHOLD = 0.05
SI_GRID_SIZE      = 10

# ── Cluster visual identity (mirrors material_set_enriched.ipynb Cell 7) ─────
CLUSTER_COLORS = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}
CLUSTER_DAMAGE_LABELS = {
    0: "Low damage",
    1: "Medium damage",
    2: "High damage",
}

# ── Pre-computed material cluster means ───────────────────────────────────────
# Source: material_set_enriched.ipynb Cell 5 — KMeans K=3 on UCI concrete
# dataset, re-ranked so cluster 0 = strongest → cluster 2 = weakest.
# UPDATE these values by reading your own concrete_defect_enriched.csv once
# you have run the full notebook pipeline.
#
#  mat_cement              kg/m³
#  mat_blast_furnace_slag  kg/m³
#  mat_fly_ash             kg/m³
#  mat_water               kg/m³
#  mat_superplasticizer    kg/m³
#  mat_coarse_aggregate    kg/m³
#  mat_fine_aggregate      kg/m³
#  mat_strength            MPa
#  mat_age                 days
MATERIAL_CLUSTER_MEANS: dict[int, dict[str, float]] = {
    0: {   # strongest concrete — maps to LOW damage
        "mat_cement":             381.3,
        "mat_blast_furnace_slag": 115.9,
        "mat_fly_ash":             28.2,
        "mat_water":              163.2,
        "mat_superplasticizer":    12.3,
        "mat_coarse_aggregate":   921.5,
        "mat_fine_aggregate":     783.0,
        "mat_strength":            54.5,
        "mat_age":                 33.0,
    },
    1: {   # medium concrete — maps to MEDIUM damage
        "mat_cement":             298.4,
        "mat_blast_furnace_slag":  73.1,
        "mat_fly_ash":             54.6,
        "mat_water":              185.7,
        "mat_superplasticizer":     6.1,
        "mat_coarse_aggregate":   972.3,
        "mat_fine_aggregate":     796.2,
        "mat_strength":            35.8,
        "mat_age":                 90.0,
    },
    2: {   # weakest concrete — maps to HIGH damage
        "mat_cement":             218.6,
        "mat_blast_furnace_slag":  22.5,
        "mat_fly_ash":             87.3,
        "mat_water":              207.4,
        "mat_superplasticizer":     2.8,
        "mat_coarse_aggregate":   998.6,
        "mat_fine_aggregate":     801.1,
        "mat_strength":            19.2,
        "mat_age":               180.0,
    },
}

MAT_DISPLAY_KEYS   = [
    "mat_cement", "mat_blast_furnace_slag", "mat_fly_ash",
    "mat_water",  "mat_superplasticizer",
    "mat_coarse_aggregate", "mat_fine_aggregate",
]
MAT_DISPLAY_LABELS = [
    "Cement", "BF Slag", "Fly Ash",
    "Water",  "Superplast.",
    "Coarse Agg.", "Fine Agg.",
]


# ─────────────────────────────────────────────────────────────────────────────
# 2 · Model architectures  (mirror v2 notebook § Section 3)
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """Classic U-Net with bilinear upsampling (v2 notebook architecture)."""
    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 features: list[int] | None = None) -> None:
        super().__init__()
        features = features or [64, 128, 256, 512]
        self.downs      = nn.ModuleList()
        self.ups        = nn.ModuleList()
        self.pool       = nn.MaxPool2d(2, 2)
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f)); ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x    = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)
        return torch.sigmoid(self.final_conv(x))


def _build_classifier() -> nn.Module:
    """ResNet-18 with the 4-class dropout head from v2 notebook § Section 3."""
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(m.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES),
    )
    return m


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Model loading
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")
_detector:   Optional[YOLO]   = None
_classifier: Optional[nn.Module] = None
_seg_model:  Optional[UNet]  = None


def _load_models() -> tuple[YOLO, nn.Module, UNet]:
    global _detector, _classifier, _seg_model

    if _detector is None:
        if not YOLO_WEIGHTS.exists():
            raise FileNotFoundError(f"YOLO weights not found: '{YOLO_WEIGHTS}'")
        log.info("Loading YOLO …")
        _detector = YOLO(str(YOLO_WEIGHTS))
        _detector.to(DEVICE)

    if _classifier is None:
        if not CLASSIFIER_WEIGHTS.exists():
            raise FileNotFoundError(f"Classifier weights not found: '{CLASSIFIER_WEIGHTS}'")
        log.info("Loading ResNet-18 classifier …")
        _classifier = _build_classifier()
        ckpt = torch.load(CLASSIFIER_WEIGHTS, map_location=DEVICE)
        _classifier.load_state_dict(ckpt.get("model", ckpt), strict=False)
        _classifier.eval()

    if _seg_model is None:
        if not SEG_WEIGHTS.exists():
            raise FileNotFoundError(f"Segmentation weights not found: '{SEG_WEIGHTS}'")
        log.info("Loading UNet …")
        _seg_model = UNet()
        ckpt = torch.load(SEG_WEIGHTS, map_location=DEVICE)
        _seg_model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        _seg_model.eval()

    return _detector, _classifier, _seg_model


# ─────────────────────────────────────────────────────────────────────────────
# 4 · Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

_CLS_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_SEG_TFM = transforms.Compose([
    transforms.Resize((SEG_IMG_SIZE, SEG_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def remove_edge_artifacts(mask: np.ndarray,
                          margin_pct: float = EDGE_MARGIN_PCT) -> np.ndarray:
    """Zero a border strip on all four sides (mirrors v2 helper)."""
    H, W    = mask.shape
    mh      = max(1, int(H * margin_pct))
    mw      = max(1, int(W * margin_pct))
    cleaned = mask.copy()
    cleaned[:mh,  :]  = 0
    cleaned[-mh:, :]  = 0
    cleaned[:,  :mw]  = 0
    cleaned[:, -mw:]  = 0
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# 5 · SI_ia / GAI  (exact formula from v2 notebook § extract_features)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(mask_np: np.ndarray, grid_size: int = 10) -> dict:
    """
    Compute GAI, SI_ia and region statistics from a binary anomaly mask.

    Follows the exact implementation in Concrete_Anomaly_Detection_v2.ipynb
    (Section 4 · extract_features).

    Steps
    ─────
    1. GAI  (Global Aggregate Index) = anomaly_pixels / total_pixels
    2. Divide mask into grid_size × grid_size cells
    3. LAI  (Local Aggregate Index)  = anomaly_in_cell / cell_size
    4. LAD  (Local Absolute Diff)    = |LAI − GAI|
    5. LDC  (Local Distrib. Coeff)   = mean(LAD)
    6. SI_ia = LDC / (2 × GAI × (1 − GAI))  × 100    [%]

    Parameters
    ──────────
    mask_np   : np.ndarray  (H, W)  — values in {0,1} or {0,255}
    grid_size : int  — N for the N×N spatial grid (default 10)

    Returns
    ───────
    dict with keys:
        total_pixels, anomaly_pixels, anomaly_area_pct,
        si_ia_score, num_regions, bounding_boxes, centroids
    """
    # normalise to uint8 {0,255}
    binary = (
        (mask_np * 255).astype(np.uint8)
        if mask_np.max() <= 1
        else mask_np.astype(np.uint8)
    )
    H, W   = binary.shape
    total  = H * W
    anom   = int(np.sum(binary > 0))

    # ── Step 1: GAI ──────────────────────────────────────────────────────────
    gai = anom / total if total > 0 else 0.0

    # ── Steps 2-6: SI_ia ─────────────────────────────────────────────────────
    if gai == 0.0 or gai == 1.0:
        # perfectly homogeneous: no meaningful spatial variation
        si_ia = 0.0
    else:
        h_step = max(1, H // grid_size)
        w_step = max(1, W // grid_size)
        lads   = []

        for row in range(grid_size):
            for col in range(grid_size):
                cell       = binary[row * h_step:(row + 1) * h_step,
                                    col * w_step:(col + 1) * w_step]
                cell_total = cell.size
                if cell_total == 0:
                    continue
                lai = np.sum(cell > 0) / cell_total     # Local Aggregate Index
                lads.append(abs(lai - gai))              # Local Absolute Diff

        ldc   = float(np.mean(lads)) if lads else 0.0   # Local Distribution Coeff
        # Guard the denominator  (2·GAI·(1−GAI) → 0 only at GAI=0 or 1,
        # already handled above, but add a safety floor anyway)
        denom = 2.0 * gai * (1.0 - gai)
        si_ia = (ldc / denom) if denom > 1e-9 else 0.0

    # ── Region stats ─────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bboxes    = [cv2.boundingRect(c) for c in contours]
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            centroids.append(
                (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            )

    return {
        "total_pixels":     total,
        "anomaly_pixels":   anom,
        "anomaly_area_pct": round(gai * 100, 4),       # GAI as %
        "si_ia_score":      round(si_ia * 100, 4),     # SI_ia as %
        "num_regions":      len(contours),
        "bounding_boxes":   bboxes,
        "centroids":        centroids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6 · Severity + defect-cluster assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_severity(gai_pct: float, si_ia: float, cls_name: str) -> tuple[str, int]:
    """
    Map (GAI%, SI_ia) → (stage_label, stage_rank).

    Mirrors the 4-stage rule in v2 analyze_concrete (§ Step 3e).
    Also derives defect_cluster ∈ {0, 1, 2} for the material lookup.

    Returns
    ───────
    stage_label : str   human-readable severity
    stage_rank  : int   numeric rank (0–4 ; 0 = normal / unreliable)
    defect_cluster : int  0=low / 1=medium / 2=high
    """
    if cls_name == "normal":
        return "NORMAL", 0, 0

    gai = gai_pct / 100.0  # convert back to [0,1] for threshold comparisons

    if gai < GAI_MIN_THRESHOLD:
        return "UNRELIABLE", 0, 0

    if gai < 0.15 and si_ia < 30:
        return "STAGE 1", 1, 0          # low damage  → cluster 0

    if gai < 0.35 and si_ia >= 30:
        return "STAGE 2", 2, 1          # medium      → cluster 1

    if gai >= 0.35 and si_ia >= 60:
        return "STAGE 3", 3, 2          # high        → cluster 2

    if gai >= 0.35 and si_ia < 40:
        return "STAGE 4", 4, 2          # high (dense)→ cluster 2

    return "TRANSITIONAL", 1, 1         # between stages → cluster 1


# ─────────────────────────────────────────────────────────────────────────────
# 7 · YOLO concrete-region detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_concrete_rois(img_bgr: np.ndarray,
                         detector: YOLO) -> list[dict]:
    """Return a list of ROI dicts; falls back to full image if YOLO fires nothing."""
    H, W  = img_bgr.shape[:2]
    results = detector(img_bgr, conf=YOLO_CONF_THRESH, verbose=False)
    rois  = []

    if results and len(results[0].boxes):
        for box in results[0].boxes:
            if float(box.conf[0]) < YOLO_CONF_THRESH:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            pw = max(1, int(BBOX_PAD_PCT * (x2 - x1)))
            ph = max(1, int(BBOX_PAD_PCT * (y2 - y1)))
            cx1 = max(0, x1 - pw);  cy1 = max(0, y1 - ph)
            cx2 = min(W, x2 + pw);  cy2 = min(H, y2 + ph)
            crop_rgb = cv2.cvtColor(img_bgr[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)
            rois.append({"bbox": (cx1, cy1, cx2, cy2), "crop_rgb": crop_rgb,
                         "yolo_conf": float(box.conf[0])})

    if not rois:
        rois.append({"bbox": (0, 0, W, H),
                     "crop_rgb": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                     "yolo_conf": None})
    return rois


# ─────────────────────────────────────────────────────────────────────────────
# 8 · Inspection card renderer  (mirrors inspect_image from notebook Cell 8)
# ─────────────────────────────────────────────────────────────────────────────

def render_inspection_card(
    orig_rgb:        np.ndarray,
    image_name:      str,
    crop_rgb:        np.ndarray,
    bbox:            tuple[int, int, int, int],
    cls_name:        str,
    confidence:      float,
    gai_pct:         float,
    si_ia:           float,
    stage_label:     str,
    stage_rank:      int,
    defect_cluster:  int,
    mat_means:       dict[str, float],
) -> Image.Image:
    """
    Build the 4-panel inspection card and return it as a PIL Image.

    Layout (1 row × 4 columns, proportions 1.5 : 1 : 1 : 1.3):
      [Image] [Identity] [Defect signals] [Material means]
    """
    dc     = defect_cluster
    color  = CLUSTER_COLORS[dc]
    damage = CLUSTER_DAMAGE_LABELS[dc]
    title  = f"{image_name}  ·  {damage}"

    fig, axes = plt.subplots(
        1, 4, figsize=(17, 4.8),
        gridspec_kw={"width_ratios": [1.5, 1, 1, 1.3], "wspace": 0.42},
    )
    fig.suptitle(title, fontsize=12, fontweight="bold",
                 x=0.01, ha="left", y=1.02, color="#111")

    # ── Panel 0 : Image with coloured border ─────────────────────────────────
    ax0 = axes[0]
    ax0.imshow(crop_rgb)
    ax0.set_title("Image", fontsize=11, fontweight="bold", pad=6)
    ax0.set_xticks([]); ax0.set_yticks([])
    for spine in ax0.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(3)

    # ── Panel 1 : Identity card ───────────────────────────────────────────────
    ax1 = axes[1]
    ax1.set_facecolor(color + "18")          # very light tint of cluster colour
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    for spine in ax1.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(1.5)
    ax1.set_title("Identity", fontsize=11, fontweight="bold", pad=6)

    identity_rows = [
        ("Defect cluster",  str(dc),                    False),
        ("Stage rank",      str(stage_rank),             False),
        ("Confidence",      f"{confidence:.3f}",         False),
        ("GAI score",       f"{gai_pct:.3f}",            False),
        ("SI-IA score",     f"{si_ia:.3f}",              False),
        ("— Material —",    "",                           True),
        ("Strength",        f"{mat_means.get('mat_strength', 0):.1f} MPa",  False),
        ("Age",             f"{mat_means.get('mat_age', 0):.0f} days",      False),
        ("Water",           f"{mat_means.get('mat_water', 0):.1f} kg/m³",  False),
    ]
    for i, (label, value, italic) in enumerate(identity_rows):
        y = 0.93 - i * 0.098
        ax1.text(0.06, y, label, transform=ax1.transAxes,
                 fontsize=9.5, color="#666",
                 style="italic" if italic else "normal", va="center")
        if value:
            ax1.text(0.94, y, value, transform=ax1.transAxes,
                     fontsize=9.5, fontweight="bold", ha="right",
                     color="#111", va="center")

    # ── Panel 2 : Defect signals bar chart ────────────────────────────────────
    ax2 = axes[2]
    defect_labels = ["SI-IA score", "GAI score", "Confidence"]
    # SI-IA and GAI are percentages (0-100) → normalise to 0-1 for the bar
    raw_vals      = [si_ia / 100.0, gai_pct / 100.0, confidence]
    display_vals  = [si_ia,         gai_pct,           confidence]

    bars = ax2.barh(
        defect_labels, raw_vals,
        color=color, alpha=0.85, edgecolor="white", height=0.52,
    )
    ax2.set_xlim(0, 1.25)
    ax2.set_xlabel("Score", fontsize=10)
    ax2.set_title("Defect signals", fontsize=11, fontweight="bold", pad=6)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    ax2.tick_params(labelsize=10)
    ax2.spines[["top", "right"]].set_visible(False)

    for bar, dval in zip(bars, display_vals):
        ax2.text(
            0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{dval:.3f}",
            va="center", fontsize=10, fontweight="bold", color="white",
        )

    # ── Panel 3 : Material composition bar chart ──────────────────────────────
    ax3 = axes[3]
    mat_vals   = [mat_means.get(k, 0.0) for k in MAT_DISPLAY_KEYS]

    barsM = ax3.barh(
        MAT_DISPLAY_LABELS, mat_vals,
        color=color, alpha=0.85, edgecolor="white", height=0.52,
    )
    ax3.set_xlabel("kg/m³", fontsize=10)
    ax3.set_title(f"Material means (cluster {dc})", fontsize=11,
                  fontweight="bold", pad=6)
    ax3.grid(axis="x", linestyle="--", alpha=0.35)
    ax3.tick_params(labelsize=9)
    ax3.spines[["top", "right"]].set_visible(False)

    x_pad = max(mat_vals) * 0.015 if mat_vals else 5
    for bar, val in zip(barsM, mat_vals):
        ax3.text(
            val + x_pad,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center", fontsize=8.5, color="#333",
        )

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ─────────────────────────────────────────────────────────────────────────────
# 9 · Main inference entry point
# ─────────────────────────────────────────────────────────────────────────────

def predict(pil_image: Image.Image,
            image_name: str = "image") -> tuple[Image.Image, str]:
    """
    Full pipeline:
      1. YOLO → concrete ROI(s)
      2. ResNet-18 → class + confidence
      3. UNet → binary anomaly mask
      4. remove_edge_artifacts → clean mask
      5. extract_features → GAI, SI_ia (exact v2 formula)
      6. assign_severity → stage + defect_cluster
      7. MATERIAL_CLUSTER_MEANS lookup
      8. render_inspection_card → 4-panel PIL Image
    """
    if pil_image is None:
        return None, "⚠️  No image uploaded."

    # ── sanitise the image name ───────────────────────────────────────────────
    safe_name = (image_name.strip() or "image").replace(" ", "_")

    # ── load models ───────────────────────────────────────────────────────────
    try:
        detector, classifier, seg_model = _load_models()
    except FileNotFoundError as exc:
        return None, f"⛔  **Model weights missing:**\n```\n{exc}\n```"
    except Exception:
        return None, f"⛔  Unexpected model-load error:\n```\n{traceback.format_exc()}\n```"

    orig_rgb = np.array(pil_image.convert("RGB"))
    img_bgr  = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

    # ── Step 1: YOLO concrete detection ───────────────────────────────────────
    try:
        rois = detect_concrete_rois(img_bgr, detector)
    except Exception:
        log.warning("YOLO failed — full-image fallback.\n%s", traceback.format_exc())
        rois = [{"bbox": (0, 0, orig_rgb.shape[1], orig_rgb.shape[0]),
                 "crop_rgb": orig_rgb, "yolo_conf": None}]

    # ── Process first ROI (primary detection) ─────────────────────────────────
    roi      = rois[0]
    crop_rgb = roi["crop_rgb"]
    bbox     = roi["bbox"]

    # ── Step 2: Classification ────────────────────────────────────────────────
    try:
        cls_tensor = _CLS_TFM(Image.fromarray(crop_rgb)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(classifier(cls_tensor), dim=1).squeeze()
        conf, idx  = probs.max(0)
        cls_name   = IDX_TO_CLASS[idx.item()]
        confidence = conf.item()
    except Exception:
        log.warning("Classifier failed.\n%s", traceback.format_exc())
        cls_name, confidence = "normal", 0.0

    # ── Steps 3-5: Segmentation + feature extraction ──────────────────────────
    gai_pct, si_ia = 0.0, 0.0
    anomaly_mask   = np.zeros(crop_rgb.shape[:2], dtype=np.uint8)

    if cls_name != "normal":
        try:
            seg_tensor = _SEG_TFM(Image.fromarray(crop_rgb)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                raw_prob = seg_model(seg_tensor)        # (1,1,H,H) already sigmoid
            raw_prob  = F.interpolate(
                raw_prob,
                size=(crop_rgb.shape[0], crop_rgb.shape[1]),
                mode="bilinear", align_corners=False,
            )
            raw_mask  = (raw_prob.squeeze().cpu().numpy() >= SEG_THRESH).astype(np.uint8)
            clean_mask = remove_edge_artifacts(raw_mask)
            feats      = extract_features(clean_mask, grid_size=SI_GRID_SIZE)
            gai_pct    = feats["anomaly_area_pct"]
            si_ia      = feats["si_ia_score"]
            anomaly_mask = clean_mask
        except ZeroDivisionError:
            log.error("ZeroDivisionError in extract_features — mask is degenerate.")
        except Exception:
            log.warning("Segmentation failed.\n%s", traceback.format_exc())

    # ── Step 6: Severity + cluster ────────────────────────────────────────────
    stage_label, stage_rank, defect_cluster = assign_severity(
        gai_pct, si_ia, cls_name
    )

    # ── Step 7: Material lookup ───────────────────────────────────────────────
    mat_means = MATERIAL_CLUSTER_MEANS[defect_cluster]

    # ── Step 8: Render card ───────────────────────────────────────────────────
    card = render_inspection_card(
        orig_rgb       = orig_rgb,
        image_name     = safe_name,
        crop_rgb       = crop_rgb,
        bbox           = bbox,
        cls_name       = cls_name,
        confidence     = confidence,
        gai_pct        = gai_pct,
        si_ia          = si_ia,
        stage_label    = stage_label,
        stage_rank     = stage_rank,
        defect_cluster = defect_cluster,
        mat_means      = mat_means,
    )

    # ── Markdown summary ──────────────────────────────────────────────────────
    dc_label = CLUSTER_DAMAGE_LABELS[defect_cluster]
    report   = (
        f"### {safe_name}  ·  {dc_label}\n\n"
        f"| Field | Value |\n|-------|-------|\n"
        f"| Defect cluster | **{defect_cluster}** |\n"
        f"| Stage | **{stage_label}** (rank {stage_rank}) |\n"
        f"| Class | `{cls_name}` |\n"
        f"| Confidence | {confidence:.3f} |\n"
        f"| GAI score | {gai_pct:.3f} % |\n"
        f"| SI-IA score | {si_ia:.3f} % |\n"
        f"| Est. strength | {mat_means['mat_strength']:.1f} MPa |\n"
        f"| Est. age | {mat_means['mat_age']:.0f} days |\n"
        f"| Est. water | {mat_means['mat_water']:.1f} kg/m³ |\n"
    )
    return card, report


# ─────────────────────────────────────────────────────────────────────────────
# 10 · Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

_DESCRIPTION = """
# 🏗️ Concrete Anomaly — Inspection Card

Upload a photo of a concrete surface.  The pipeline returns a **4-panel
inspection card** identical to the one generated by the research notebook:

| Panel | Content |
|-------|---------|
| **Image** | Detected concrete region with cluster-coloured border |
| **Identity** | Defect cluster, stage rank, GAI, SI-IA, material properties |
| **Defect signals** | Bar chart of Confidence · GAI · SI-IA |
| **Material means** | Estimated mix composition (kg/m³) for the assigned cluster |
"""

_ARTICLE = """
---
### Metric definitions

**GAI** (Global Aggregate Index) = `anomaly_pixels / total_pixels × 100 %`

**SI_ia** (Spatial Index via Image Analysis):
```
LAI  = anomaly_in_cell / cell_size          (per 10×10 grid)
LAD  = |LAI − GAI|
LDC  = mean(LAD)
SI_ia = LDC / (2 × GAI × (1 − GAI)) × 100
```

**Cluster mapping** — defect severity → material strength:
- 🟢 Cluster 0 · Low damage → Strongest concrete
- 🟠 Cluster 1 · Medium damage → Medium concrete
- 🔴 Cluster 2 · High damage → Weakest concrete

*PFE — Concrete Anomaly Detection Pipeline v2/v3*
"""

with gr.Blocks(title="Concrete Anomaly Detection") as demo:
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            inp_image = gr.Image(type="pil", label="Input image")
            inp_name  = gr.Textbox(
                value="image",
                label="Image / filename label",
                placeholder="e.g.  1.jpeg",
                max_lines=1,
            )
            run_btn = gr.Button("🔍  Analyse", variant="primary")

        with gr.Column(scale=2):
            out_card   = gr.Image(type="pil", label="Inspection card")
            out_report = gr.Markdown()

    run_btn.click(
        fn=predict,
        inputs=[inp_image, inp_name],
        outputs=[out_card, out_report],
    )

    gr.Markdown(_ARTICLE)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
