"""

🏗️ Concrete Anomaly Inspector — Gradio App

Runs the full inspection pipeline on one uploaded image:

  1. Roboflow → ROI crop (fallback: full image)

  2. ResNet-18 → classify (crack | crack_segregation | segregation | normal)

  3. U-Net → binary anomaly mask (ALL classes — always shown)

  4. extract_features → GAI + SI_ia

  5. Stage assignment → KMeans (primary) + rule-based (reference)
      — ONLY for segregation / crack_segregation

  6. Inspection card → 4-panel or 2-panel figure

Sensitive config is loaded from a .env file (or HF Spaces secrets).

"""

import os
import io
import cv2
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import gradio as gr

from torchvision import transforms, models
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
MODELS_DIR        = os.getenv("MODELS_DIR", "./saved_models/")
ROBOFLOW_API_KEY  = os.getenv("ROBOFLOW_API_KEY",  "")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "")
SEG_THRESHOLD     = float(os.getenv("SEG_THRESHOLD",  "0.90"))
CONF_THRESHOLD    = float(os.getenv("CONF_THRESHOLD", "0.90"))
SHOW_MASK         = os.getenv("SHOW_MASK", "true").lower() == "true"

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
CLASSES      = ["crack", "crack_segregation", "segregation", "normal"]
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
SEG_IMG_SIZE = 256
CLS_IMG_SIZE = 224

GAI_MIN_THRESHOLD = 0.05
GAI_LOW           = 0.15
GAI_HIGH          = 0.35

MATERIAL_CLUSTER_MEANS = {
    0: {"mat_cement": 381.3, "mat_blast_furnace_slag": 115.9, "mat_fly_ash": 28.2,
        "mat_water": 163.2, "mat_superplasticizer": 12.3,
        "mat_coarse_aggregate": 921.5, "mat_fine_aggregate": 783.0,
        "mat_strength": 54.5, "mat_age": 33.0},
    1: {"mat_cement": 298.4, "mat_blast_furnace_slag": 73.1, "mat_fly_ash": 54.6,
        "mat_water": 185.7, "mat_superplasticizer": 6.1,
        "mat_coarse_aggregate": 972.3, "mat_fine_aggregate": 796.2,
        "mat_strength": 35.8, "mat_age": 90.0},
    2: {"mat_cement": 218.6, "mat_blast_furnace_slag": 22.5, "mat_fly_ash": 87.3,
        "mat_water": 207.4, "mat_superplasticizer": 2.8,
        "mat_coarse_aggregate": 998.6, "mat_fine_aggregate": 801.1,
        "mat_strength": 19.2, "mat_age": 180.0},
}
DAMAGE_LABEL = {0: "Low damage",    1: "Medium damage",  2: "High damage"}
DAMAGE_COLOR = {0: "#2ca02c",       1: "#ff7f0e",          2: "#d62728"}
CLASS_COLOR  = {
    "crack":             (255,  50,  50),
    "crack_segregation": (255, 165,   0),
    "segregation":       ( 50, 100, 255),
    "normal":            ( 50, 200,  50),
}
CLASS_HEX = {
    "crack":             "#e63946",
    "crack_segregation": "#fb5607",
    "segregation":       "#3a86ff",
    "normal":            "#2dc653",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)

        ch = in_channels
        for feat in features:
            self.downs.append(DoubleConv(ch, feat))
            ch = feat
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feat * 2, feat))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)
        return torch.sigmoid(self.final_conv(x))


def get_classifier_architecture(dev):
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(p=0.4), nn.Linear(m.fc.in_features, 256), nn.ReLU(),
        nn.Dropout(p=0.3), nn.Linear(256, 4),
    )
    return m.to(dev)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def _load_models():
    seg_model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512]).to(device)
    seg_model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "best_unet.pth"), map_location=device)
    )
    seg_model.eval()

    classifier = get_classifier_architecture(device)
    classifier.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "best_classifier.pth"), map_location=device)
    )
    classifier.eval()

    km_model  = joblib.load(os.path.join(MODELS_DIR, "kmeans_stages.pkl"))
    km_scaler = joblib.load(os.path.join(MODELS_DIR, "kmeans_scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "kmeans_rank_map.json")) as f:
        km_rank_map = {int(k): int(v) for k, v in json.load(f).items()}

    concrete_seg_model = None
    if ROBOFLOW_API_KEY:
        try:
            from inference import get_model
            concrete_seg_model = get_model(
                model_id=ROBOFLOW_MODEL_ID, api_key=ROBOFLOW_API_KEY
            )
            print("✅ Roboflow model loaded.")
        except Exception as exc:
            print(f"⚠️  Roboflow unavailable ({exc}) — will use full image as ROI.")
    else:
        print("ℹ️  No ROBOFLOW_API_KEY — using full image as ROI.")

    return seg_model, classifier, km_model, km_scaler, km_rank_map, concrete_seg_model


try:
    seg_model, classifier, km_model, km_scaler, km_rank_map, concrete_seg_model = _load_models()
    MODELS_LOADED = True
    print(f"✅ All models loaded on {device}.")
except Exception as _exc:
    MODELS_LOADED = False
    print(f"❌ Model loading failed: {_exc}")


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def _to_tensor(image_rgb: np.ndarray, size: int) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tfm(Image.fromarray(image_rgb)).unsqueeze(0)


def classify_image(model, tensor: torch.Tensor) -> tuple[str, float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(tensor.to(device)), dim=1)
        conf, pred = probs.max(1)
    return IDX_TO_CLASS[pred.item()], conf.item(), probs.squeeze().cpu().numpy()


def segment_image(model, tensor: torch.Tensor, threshold: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(tensor.to(device))
    return (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8)


def extract_features(mask_np: np.ndarray, grid_size: int = 10) -> dict:
    binary = (mask_np * 255).astype(np.uint8) if mask_np.max() <= 1 else mask_np.astype(np.uint8)
    H, W   = binary.shape
    total  = H * W
    anom   = int(np.sum(binary > 0))
    gai    = anom / total if total > 0 else 0.0

    if gai == 0 or gai == 1:
        si_ia = 0.0
    else:
        h_step = max(1, H // grid_size)
        w_step = max(1, W // grid_size)
        lads   = []
        for row in range(grid_size):
            for col in range(grid_size):
                cell = binary[row * h_step:(row + 1) * h_step,
                               col * w_step:(col + 1) * w_step]
                if cell.size == 0:
                    continue
                lads.append(abs(np.sum(cell > 0) / cell.size - gai))
        ldc   = np.mean(lads)
        si_ia = ldc / (2 * gai * (1 - gai))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return {
        "total_pixels":     total,
        "anomaly_pixels":   anom,
        "anomaly_area_pct": round(gai * 100, 4),
        "si_ia_score":      round(si_ia * 100, 4),
        "num_regions":      len(contours),
        "bounding_boxes":   [cv2.boundingRect(c) for c in contours],
    }


def remove_edge_artifacts(mask: np.ndarray, margin_pct: float = 0.04) -> np.ndarray:
    H, W = mask.shape
    mh   = max(1, int(H * margin_pct))
    mw   = max(1, int(W * margin_pct))
    c    = mask.copy()
    c[:mh, :] = c[-mh:, :] = c[:, :mw] = c[:, -mw:] = 0
    return c


def apply_circular_mask(mask: np.ndarray, margin: float = 0.03) -> np.ndarray:
    H, W = mask.shape
    ry = max(1, int(H / 2 * (1 - margin)))
    rx = max(1, int(W / 2 * (1 - margin)))
    e  = np.zeros_like(mask)
    cv2.ellipse(e, (W // 2, H // 2), (rx, ry), 0, 0, 360, 1, -1)
    return (mask * e).astype(mask.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════
def _get_stage_severity(stage_str: str) -> float:
    mapping = {
        "STAGE 4": 4.0,
        "STAGE 3": 3.0,
        "STAGE 2": 2.0,
        "TRANSITIONAL": 1.5,
        "STAGE 1": 1.0,
        "UNRELIABLE": 0.0,
        "N/A": -1.0
    }
    return mapping.get(stage_str, -1.0)


def assign_stage(gai_pct: float, si: float, cls_name: str) -> tuple[str, int]:
    if cls_name in ("normal", "crack"):
        return "N/A", -1
    gai = gai_pct / 100.0
    if gai < GAI_MIN_THRESHOLD:       return "UNRELIABLE",    0
    if gai < GAI_LOW   and si < 30:   return "STAGE 1",       0
    if gai < GAI_HIGH  and si >= 30:  return "STAGE 2",       1
    if gai >= GAI_HIGH and si >= 60:  return "STAGE 3",       2
    if gai >= GAI_HIGH and si < 40:   return "STAGE 4",       2
    return "TRANSITIONAL", 1


def assign_stage_kmeans(gai_pct: float, si: float, cls_name: str) -> tuple[str, int]:
    if cls_name in ("normal", "crack"):
        return "N/A", -1
    if gai_pct / 100.0 < GAI_MIN_THRESHOLD:
        return "UNRELIABLE", 0
    feat    = km_scaler.transform([[gai_pct, si]])
    raw_id  = int(km_model.predict(feat)[0])
    rank    = km_rank_map[raw_id]
    return f"STAGE {rank + 1}", rank


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _mask_overlay(crop: np.ndarray, mask, cls_name: str) -> np.ndarray:
    d = crop.copy()
    if SHOW_MASK and mask is not None and mask.any():
        h, w = crop.shape[:2]
        m  = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        ov = d.copy()
        ov[m > 0] = CLASS_COLOR.get(cls_name, (50, 100, 255))
        d  = cv2.addWeighted(d, 0.55, ov, 0.45, 0)
    return d


# ═══════════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL PLOT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def plot1_roi(orig_img, crop_img, bbox, pred_class, confidence) -> Image.Image:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#f8f9fa")
    fig.suptitle("Stage 1 · Detection & ROI Crop",
                 fontsize=12, fontweight="bold", color="#1c2541", y=1.01)

    cx1, cy1, cx2, cy2 = bbox
    cls_hex = CLASS_HEX.get(pred_class, "#888888")
    W, H    = orig_img.shape[1], orig_img.shape[0]
    is_crop = (cx2 - cx1 < W) or (cy2 - cy1 < H)
    rect_color = "#3a86ff" if is_crop else "#ff7f0e"

    ax = axes[0]
    ax.imshow(orig_img)
    ax.set_title("Original Image", fontsize=11, fontweight="bold", pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    rect = patches.Rectangle((cx1, cy1), cx2 - cx1, cy2 - cy1,
                               linewidth=2.5, edgecolor=rect_color,
                               facecolor="none", linestyle="--")
    ax.add_patch(rect)
    source_label = "Roboflow ROI" if is_crop else "Full Image (no detection)"
    ax.text(cx1 + 4, max(cy1 - 6, 10), source_label,
            color="white", fontsize=8.5, fontweight="bold",
            bbox=dict(facecolor=rect_color, edgecolor="none", pad=2, alpha=0.85))
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc"); sp.set_linewidth(1.2)

    ax2 = axes[1]
    ax2.imshow(crop_img)
    ax2.set_title(f"ROI Crop  [{cx2 - cx1}×{cy2 - cy1} px]",
                  fontsize=11, fontweight="bold", pad=6)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.text(0.02, 0.97, f"{pred_class.replace('_', ' ').upper()}  {confidence * 100:.1f}%",
             transform=ax2.transAxes, ha="left", va="top",
             fontsize=9.5, fontweight="bold", color="white",
             bbox=dict(facecolor=cls_hex, edgecolor="none", pad=3, alpha=0.90))
    for sp in ax2.spines.values():
        sp.set_edgecolor(cls_hex); sp.set_linewidth(3)

    plt.tight_layout()
    return _fig_to_pil(fig)


def plot2_classification(pred_class, confidence, all_probs, crop_img) -> Image.Image:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#f8f9fa")
    fig.suptitle("Stage 2 · ResNet-18 Classification",
                 fontsize=12, fontweight="bold", color="#1c2541", y=1.01)

    cls_hex = CLASS_HEX.get(pred_class, "#888888")

    axes[0].imshow(crop_img)
    axes[0].set_title("ROI Crop", fontsize=11, fontweight="bold", pad=6)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for sp in axes[0].spines.values():
        sp.set_edgecolor(cls_hex); sp.set_linewidth(3)

    bar_colors = [cls_hex if c == pred_class else "#b0bec5" for c in CLASSES]
    bars = axes[1].barh(CLASSES, all_probs, color=bar_colors, alpha=0.9, height=0.55)
    for bar, val in zip(bars, all_probs):
        axes[1].text(bar.get_width() + 0.012,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val * 100:.1f}%",
                     va="center", ha="left", fontsize=9,
                     fontweight="bold", color="#1c2541")
    axes[1].set_title("Class Probabilities", fontsize=11, fontweight="bold", pad=7)
    axes[1].set_xlabel("Probability", fontsize=9)
    axes[1].set_xlim(0, 1.28)
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", linestyle="--", alpha=0.35)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].tick_params(axis="both", labelsize=8.5)
    axes[1].text(0.98, 0.02, f"Confidence: {confidence * 100:.1f}%",
                 transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=9.5, color=cls_hex, fontweight="bold")

    plt.tight_layout()
    return _fig_to_pil(fig)


_LAI_CMAP = LinearSegmentedColormap.from_list(
    "lai_cmap",
    ["#1a78c2", "#ffffff", "#d62728"],
    N=256,
)


def _compute_lai_grid(mask: np.ndarray, grid_size: int = 10) -> np.ndarray:
    H, W = mask.shape
    h_step = max(1, H // grid_size)
    w_step = max(1, W // grid_size)
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r in range(grid_size):
        for c in range(grid_size):
            cell = mask[r * h_step:(r + 1) * h_step,
                        c * w_step:(c + 1) * w_step]
            if cell.size > 0:
                grid[r, c] = np.sum(cell > 0) / cell.size
    return grid


def plot3_lai_heatmap(crop_img, pred_mask, gai_score, si_ia,
                       pred_class, confidence, stage_label) -> Image.Image:
    binary_mask = (pred_mask > 0).astype(np.uint8)
    gai_ref   = gai_score / 100.0
    si_ia_ref = si_ia

    grid_size = 10
    lai_grid  = _compute_lai_grid(binary_mask, grid_size=grid_size)

    H, W = crop_img.shape[:2]
    lai_full = cv2.resize(lai_grid, (W, H), interpolation=cv2.INTER_NEAREST)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("#0f0f0f")

    title_color    = "white"
    subtitle_color = "#aaaaaa"

    axes[0].imshow(crop_img)
    axes[0].set_title("Concrete Crop", color=title_color, fontsize=12, fontweight="bold")
    axes[0].axis("off")
    axes[0].set_facecolor("#0f0f0f")

    vmin = max(0.0, gai_ref - 0.3)
    vmax = min(1.0, gai_ref + 0.3)
    vmin = min(vmin, lai_grid.min())
    vmax = max(vmax, lai_grid.max())
    if vmax == vmin:
        vmax = vmin + 1e-6

    im = axes[1].imshow(
        lai_grid,
        cmap=_LAI_CMAP,
        vmin=vmin, vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cb.set_label("LAI (local anomaly density)", color=subtitle_color, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=subtitle_color)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=subtitle_color)

    cb.ax.axhline(y=(gai_ref - vmin) / (vmax - vmin),
                  color="yellow", linewidth=2, linestyle="--")

    for r in range(grid_size):
        for c in range(grid_size):
            val = lai_grid[r, c]
            fc  = "black" if val > gai_ref else "white"
            axes[1].text(c, r, f"{val:.2f}",
                         ha="center", va="center",
                         fontsize=6, color=fc, fontweight="bold")

    axes[1].set_title(
        f"LAI Heatmap  (GAI={gai_ref * 100:.1f}%)\n"
        f"Yellow dashes = GAI reference",
        color=title_color, fontsize=11, fontweight="bold",
    )
    axes[1].set_xlabel(f"{grid_size}x{grid_size} spatial grid",
                       color=subtitle_color, fontsize=9)
    axes[1].tick_params(colors=subtitle_color)
    for spine in axes[1].spines.values():
        spine.set_edgecolor("#444")

    norm_lai = (lai_full - vmin) / (vmax - vmin + 1e-9)
    norm_lai = np.clip(norm_lai, 0, 1)
    rgba  = _LAI_CMAP(norm_lai)
    rgb_h = (rgba[:, :, :3] * 255).astype(np.uint8)
    blended = cv2.addWeighted(crop_img, 0.5, rgb_h, 0.5, 0)

    h_step = max(1, H // grid_size)
    w_step = max(1, W // grid_size)
    for r in range(1, grid_size):
        cv2.line(blended, (0, r * h_step), (W, r * h_step), (200, 200, 200), 1)
    for c in range(1, grid_size):
        cv2.line(blended, (c * w_step, 0), (c * w_step, H), (200, 200, 200), 1)

    axes[2].imshow(blended)
    stage_text = stage_label.split("—")[-1].strip() if "—" in stage_label else stage_label
    axes[2].set_title(
        f"LAI Overlay  |  SI_ia = {si_ia_ref:.1f}%\n"
        f"Class: {pred_class.upper()} ({confidence * 100:.0f}%)  {stage_text}",
        color=title_color, fontsize=11, fontweight="bold",
    )
    axes[2].axis("off")
    axes[2].set_facecolor("#0f0f0f")

    legend_patches = [
        mpatches.Patch(color="#1a78c2", label=f"Below GAI (<{gai_ref * 100:.1f}%)"),
        mpatches.Patch(color="#ffffff", label=f"≈ GAI ({gai_ref * 100:.1f}%)"),
        mpatches.Patch(color="#d62728", label=f"Above GAI (>{gai_ref * 100:.1f}%)"),
    ]
    axes[2].legend(handles=legend_patches, loc="lower right",
                   fontsize=8, facecolor="#1a1a1a", labelcolor="white",
                   edgecolor="#555")

    for ax in axes:
        ax.set_facecolor("#0f0f0f")

    fig.suptitle(
        f"Detection 1  |  LAI Heatmap Analysis\n"
        f"GAI = {gai_ref * 100:.2f}%   SI_ia = {si_ia_ref:.2f}%   {stage_label}",
        fontsize=13, fontweight="bold", color="white", y=1.01,
    )

    plt.tight_layout()
    return _fig_to_pil(fig)


def plot4_stage_comparison(stage_label_km, stage_label_rb,
                            stage_rank_km, cluster_rb,
                            gai_score, si_ia, crop_img) -> Image.Image:
    STAGE_COLORS = {
        "STAGE 1": "#3a86ff", "STAGE 2": "#ffbe0b",
        "STAGE 3": "#fb5607", "STAGE 4": "#e63946",
        "TRANSITIONAL": "#8338ec", "UNRELIABLE": "#aaaaaa", "N/A": "#cccccc",
    }

    def _panel(ax, stage_str, method_label, gai, si, bg_img, subtitle=""):
        col = STAGE_COLORS.get(stage_str, "#888888")
        ax.set_facecolor("#ffffff"); ax.set_xticks([]); ax.set_yticks([])
        if bg_img is not None:
            ax.imshow(bg_img, alpha=0.45, aspect="auto")
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(3.5)
        ax.set_title(method_label, fontsize=11, fontweight="bold", pad=10, color="#1c2541")
        ax.text(0.5, 0.68, stage_str, transform=ax.transAxes,
                ha="center", va="center", fontsize=24, fontweight="bold", color=col,
                bbox=dict(facecolor="#ffffff", edgecolor="none", pad=4,
                          alpha=0.85, boxstyle="round,pad=0.3"))
        feat_txt = f"GAI  = {gai:.2f} %\nSI   = {si:.2f}\n{'─' * 20}\n{subtitle}"
        ax.text(0.5, 0.24, feat_txt, transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#222222",
                fontfamily="monospace", fontweight="semibold",
                bbox=dict(facecolor="#ffffff", edgecolor=col, linewidth=1,
                          pad=6, alpha=0.9, boxstyle="round,pad=0.5"))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#f8f9fa")
    fig.suptitle("Stage 4 · Stage Assignment Comparison",
                 fontsize=12, fontweight="bold", color="#1c2541", y=1.03)

    km_sub = (f"rank {stage_rank_km}  →  {DAMAGE_LABEL.get(stage_rank_km, '—')}"
              if stage_rank_km >= 0 else "no cluster assigned")
    _panel(axs[0], stage_label_km, "KMeans  (primary)",
           gai_score, si_ia, crop_img, km_sub)

    rb_sub = (f"cluster {cluster_rb}  →  {DAMAGE_LABEL.get(cluster_rb, '—')}"
              if cluster_rb >= 0 else "no cluster assigned")
    _panel(axs[1], stage_label_rb, "Rule-based  (reference)",
           gai_score, si_ia, crop_img, rb_sub)

    agree     = stage_label_km == stage_label_rb
    agree_txt = "✅  Both methods agree" if agree else "⚠️  Methods disagree — Highest stage selected"
    fig.text(0.5, -0.05, agree_txt, ha="center", fontsize=10.5,
             color="#2ca02c" if agree else "#e63946", fontweight="bold",
             bbox=dict(facecolor="#ffffff", edgecolor="#e0e0e0",
                       pad=4, alpha=0.9, boxstyle="round,pad=0.4"))

    plt.tight_layout()
    return _fig_to_pil(fig)


def plot5_inspection_card(crop_img, pred_mask, pred_class,
                           confidence, all_probs,
                           gai_score, si_ia,
                           stage_label, defect_cluster) -> Image.Image:
    MAT_ROWS = [
        ("mat_fine_aggregate",      "Fine Agg."),
        ("mat_coarse_aggregate",    "Coarse Agg."),
        ("mat_superplasticizer",    "Superplast."),
        ("mat_water",               "Water"),
        ("mat_fly_ash",             "Fly Ash"),
        ("mat_blast_furnace_slag",  "BF Slag"),
        ("mat_cement",              "Cement"),
    ]

    if pred_class in ("segregation", "crack_segregation") and defect_cluster >= 0:
        mat   = MATERIAL_CLUSTER_MEANS[defect_cluster]
        color = DAMAGE_COLOR[defect_cluster]

        fig = plt.figure(figsize=(18, 4.8), facecolor="#f8f9fa")
        fig.suptitle(f"Stage 5 · Inspection Card  ·  {DAMAGE_LABEL[defect_cluster]}",
                     fontsize=13, fontweight="bold", color="#1c2541",
                     x=0.005, ha="left", y=1.02)
        gs = gridspec.GridSpec(1, 4, figure=fig,
                               width_ratios=[1.6, 1.1, 1.05, 1.4], wspace=0.36)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(_mask_overlay(crop_img, pred_mask, pred_class))
        ax1.set_title("Image", fontsize=11, fontweight="bold", pad=7)
        ax1.set_xticks([]); ax1.set_yticks([])
        for sp in ax1.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(3.5)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor("#f0f9f0")
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
        for sp in ax2.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2)
        ax2.set_title("Identity", fontsize=11, fontweight="bold", pad=7)
        identity_rows = [
            ("Defect cluster",  str(defect_cluster),         True),
            ("Stage",           stage_label,                 True),
            ("Stage rank",      str(defect_cluster + 1),      True),
            ("Confidence",      f"{confidence * 100:.1f}%",   True),
            ("GAI score",       f"{gai_score:.3f} %",         True),
            ("SI-IA score",     f"{si_ia:.3f}",               True),
            ("— Material —",    "",                           False),
            ("Strength",        f"{mat['mat_strength']} MPa", True),
            ("Age",             f"{mat['mat_age']:.0f} days", True),
            ("Water",           f"{mat['mat_water']} kg/m³",  True),
        ]
        for idx, (lbl, val, bold) in enumerate(identity_rows):
            y = 1 - (idx + 0.5) / len(identity_rows)
            ax2.text(0.06, y, lbl, transform=ax2.transAxes,
                     ha="left", va="center", fontsize=8.4, color="#555")
            if val:
                ax2.text(0.97, y, val, transform=ax2.transAxes,
                         ha="right", va="center",
                         fontsize=8.4,
                         fontweight="bold" if bold else "normal",
                         color="#1c2541")

        ax3 = fig.add_subplot(gs[0, 2])
        signal_vals  = [si_ia, gai_score, confidence * 100]
        signal_names = ["SI-IA score", "GAI score", "Confidence"]
        bars3 = ax3.barh(signal_names, signal_vals,
                         color=color, alpha=0.85, height=0.55)
        for bar, val in zip(bars3, signal_vals):
            ax3.text(bar.get_width() * 0.5,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}", va="center", ha="center",
                     fontsize=9, fontweight="bold", color="white")
        ax3.set_title("Defect Signals", fontsize=11, fontweight="bold", pad=7)
        ax3.set_xlabel("Score", fontsize=9); ax3.invert_yaxis()
        ax3.grid(axis="x", linestyle="--", alpha=0.35)
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.tick_params(axis="both", labelsize=8)

        ax4 = fig.add_subplot(gs[0, 3])
        mk   = [k for k, _ in MAT_ROWS]
        ml   = [n for _, n in MAT_ROWS]
        mv   = [mat[k] for k in mk]
        bars4 = ax4.barh(ml, mv, color=color, alpha=0.85, height=0.6)
        for bar, val in zip(bars4, mv):
            ax4.text(bar.get_width() + max(mv) * 0.015,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}", va="center", ha="left",
                     fontsize=8, fontweight="bold", color="#1c2541")
        ax4.set_title(f"Material Means (cluster {defect_cluster})",
                       fontsize=11, fontweight="bold", pad=7)
        ax4.set_xlabel("kg/m³", fontsize=9)
        ax4.invert_yaxis(); ax4.set_xlim(0, max(mv) * 1.22)
        ax4.grid(axis="x", linestyle="--", alpha=0.35)
        ax4.spines[["top", "right"]].set_visible(False)
        ax4.tick_params(axis="both", labelsize=8)

    else:
        is_normal = pred_class == "normal"
        bc        = "#2ca02c" if is_normal else "#d62728"

        fig = plt.figure(figsize=(11, 4.5), facecolor="#f8f9fa")
        fig.suptitle(f"Stage 5 · Inspection Card  ·  "
                     f"{pred_class.replace('_', ' ').title()}",
                     fontsize=13, fontweight="bold", color="#1c2541",
                     x=0.005, ha="left", y=1.02)
        gs = gridspec.GridSpec(1, 2, figure=fig,
                               width_ratios=[1.4, 1.0], wspace=0.32)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(_mask_overlay(crop_img, pred_mask, pred_class))
        ax1.set_title("Image", fontsize=11, fontweight="bold", pad=7)
        ax1.set_xticks([]); ax1.set_yticks([])
        for sp in ax1.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(3.5)

        ax2 = fig.add_subplot(gs[0, 1])
        bar_cols = [bc if c == pred_class else "#b0bec5" for c in CLASSES]
        bars5 = ax2.barh(CLASSES, all_probs, color=bar_cols, alpha=0.9, height=0.55)
        for bar, val in zip(bars5, all_probs):
            ax2.text(bar.get_width() + 0.012,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val * 100:.1f}%", va="center", ha="left",
                     fontsize=9, fontweight="bold", color="#1c2541")
        ax2.set_title("Class Probabilities", fontsize=11, fontweight="bold", pad=7)
        ax2.set_xlabel("Probability", fontsize=9)
        ax2.set_xlim(0, 1.28); ax2.invert_yaxis()
        ax2.grid(axis="x", linestyle="--", alpha=0.35)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.tick_params(axis="both", labelsize=8.5)
        ax2.text(0.98, 0.02, f"Confidence: {confidence * 100:.1f}%",
                 transform=ax2.transAxes, ha="right", va="bottom",
                 fontsize=9, color=bc, fontweight="bold")

    plt.tight_layout()
    return _fig_to_pil(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY HTML BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
def _build_summary_html(pred_class, confidence, gai_score, si_ia,
                         stage_label_km, stage_label_rb,
                         defect_cluster, num_regions) -> str:
    cls_hex    = CLASS_HEX.get(pred_class, "#888888")
    damage_str = DAMAGE_LABEL.get(defect_cluster, "N/A")
    dam_color  = DAMAGE_COLOR.get(defect_cluster, "#888888")
    agree      = stage_label_km == stage_label_rb
    agree_col  = "#2ca02c" if agree else "#e63946"
    agree_txt  = "✅ Agree" if agree else "⚠️ Disagree — Highest stage selected"
    is_seg_cls = pred_class in ("segregation", "crack_segregation")

    STAGE_COLORS = {
        "STAGE 1": "#3a86ff", "STAGE 2": "#ffbe0b", "STAGE 3": "#fb5607",
        "STAGE 4": "#e63946", "TRANSITIONAL": "#8338ec",
        "UNRELIABLE": "#aaaaaa", "N/A": "#888888",
    }
    stage_col = STAGE_COLORS.get(stage_label_km, "#888888")

    def badge(text, color):
        return (f'<span style="background:{color}22;color:{color};padding:2px 10px;'
                f'border-radius:12px;font-weight:700;font-size:0.85em;'
                f'border:1px solid {color}55">{text}</span>')

    def row(label, value_html):
        return (f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:7px 4px;border-bottom:1px solid #f0f0f0">'
                f'<span style="color:#6b7280;font-size:0.88em;font-weight:500">{label}</span>'
                f'<span>{value_html}</span></div>')

    def code_val(text):
        return (f'<code style="background:#f3f4f6;color:#1c2541;padding:2px 8px;'
                f'border-radius:4px;font-size:0.88em">{text}</code>')

    seg_rows = ""
    if is_seg_cls:
        seg_rows = (
            row("Stage (KMeans)", badge(stage_label_km, stage_col)) +
            row("Stage (Rule-based)", code_val(stage_label_rb)) +
            row("Agreement", f'<span style="color:{agree_col};font-weight:700;font-size:0.88em">{agree_txt}</span>') +
            row("Damage Level", badge(damage_str, dam_color)) +
            row("Anomaly Regions", code_val(str(num_regions)))
        )
    else:
        seg_rows = (
            row("Stage", badge("N/A — not a seg class", "#aaaaaa")) +
            row("Anomaly Regions", code_val(str(num_regions)))
        )

    return f"""
<div style="background:#fff;border-radius:12px;padding:16px 20px;
            border:1px solid #e5e7eb;border-top:4px solid {cls_hex};
            box-shadow:0 1px 4px rgba(0,0,0,0.06);font-family:sans-serif">
  <div style="font-size:1.05em;font-weight:700;color:#1c2541;margin-bottom:12px;
              display:flex;align-items:center;gap:8px">
    🔍 Inspection Summary
  </div>
  {row("Defect Class", badge(pred_class.replace("_", " ").title(), cls_hex))}
  {row("Confidence", f'<span style="color:{cls_hex};font-weight:700;font-family:monospace">{confidence*100:.1f}%</span>')}
  {row("GAI Score", code_val(f"{gai_score:.3f}%"))}
  {row("SI-IA Score", code_val(f"{si_ia:.3f}"))}
  {seg_rows}
</div>
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def run_pipeline(image: Image.Image,
                 seg_threshold: float = SEG_THRESHOLD,
                 conf_threshold: float = CONF_THRESHOLD,
                 show_mask: bool = SHOW_MASK):
    if not MODELS_LOADED:
        raise gr.Error(
            "Models could not be loaded. "
            "Check MODELS_DIR and that all .pth / .pkl / .json files exist."
        )
    if image is None:
        raise gr.Error("Please upload a concrete image first.")

    global SHOW_MASK
    SHOW_MASK = show_mask
    orig_img = np.array(image.convert("RGB"))
    H, W     = orig_img.shape[:2]

    # ── Step 1: Roboflow ROI detection ────────────────────────────────────────
    crop_img = orig_img.copy()
    bbox     = (0, 0, W, H)

    if concrete_seg_model is not None:
        try:
            import supervision as sv
            rf_res = concrete_seg_model.infer(
                cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
            )[0]
            dets = sv.Detections.from_inference(rf_res)
            if dets.confidence is not None:
                dets = dets[dets.confidence >= conf_threshold]
            if len(dets) > 0:
                x1, y1, x2, y2 = dets.xyxy[0].astype(int)
                px  = max(1, int(0.10 * (x2 - x1)))
                py  = max(1, int(0.10 * (y2 - y1)))
                cx1 = max(0, x1 - px);  cy1 = max(0, y1 - py)
                cx2 = min(W, x2 + px + 1);  cy2 = min(H, y2 + py + 1)
                crop_img = orig_img[cy1:cy2, cx1:cx2]
                bbox     = (cx1, cy1, cx2, cy2)
        except Exception as exc:
            print(f"⚠️  Roboflow error: {exc} — falling back to full image.")

    # ── Step 2: Classify (ResNet-18) ─────────────────────────────────────────
    cls_tensor = _to_tensor(crop_img, CLS_IMG_SIZE)
    pred_class, confidence, all_probs = classify_image(classifier, cls_tensor)

    # ── Step 3: Segment (U-Net) — ALWAYS run for all classes ─────────────────
    seg_tensor = _to_tensor(crop_img, SEG_IMG_SIZE)
    mask_raw   = segment_image(seg_model, seg_tensor, seg_threshold)
    aspect     = crop_img.shape[0] / max(crop_img.shape[1], 1)
    if aspect > 1.4:
        pred_mask = apply_circular_mask(
            remove_edge_artifacts(mask_raw, margin_pct=0.05), margin=0.03
        )
    else:
        pred_mask = remove_edge_artifacts(mask_raw, margin_pct=0.04)
    features  = extract_features(pred_mask, grid_size=10)
    gai_score = features["anomaly_area_pct"]
    si_ia     = features["si_ia_score"]

    # ── Step 4: Stage assignment (seg classes only) ───────────────────────────
    stage_label_km, stage_rank_km = assign_stage_kmeans(gai_score, si_ia, pred_class)
    stage_label_rb, cluster_rb    = assign_stage(gai_score, si_ia, pred_class)
    
    # Choose the highest severity stage between KMeans and Rule-Based
    sev_km = _get_stage_severity(stage_label_km)
    sev_rb = _get_stage_severity(stage_label_rb)

    if sev_rb > sev_km:
        stage_label    = stage_label_rb
        defect_cluster = cluster_rb
    else:
        stage_label    = stage_label_km
        defect_cluster = stage_rank_km

    # ── Generate plots ────────────────────────────────────────────────────────
    p1 = plot1_roi(orig_img, crop_img, bbox, pred_class, confidence)
    p2 = plot2_classification(pred_class, confidence, all_probs, crop_img)

    # Stage 3: always generated for every class
    p3 = plot3_lai_heatmap(crop_img, pred_mask, gai_score, si_ia,
                            pred_class, confidence, stage_label)

    # Stage 4: only for segregation / crack_segregation
    p4 = (plot4_stage_comparison(stage_label_km, stage_label_rb,
                                   stage_rank_km, cluster_rb,
                                   gai_score, si_ia, crop_img)
          if pred_class in ("segregation", "crack_segregation") else None)

    p5 = plot5_inspection_card(crop_img, pred_mask, pred_class,
                                confidence, all_probs,
                                gai_score, si_ia,
                                stage_label, defect_cluster)

    summary_html = _build_summary_html(
        pred_class, confidence, gai_score, si_ia,
        stage_label_km, stage_label_rb,
        defect_cluster, features.get("num_regions", 0),
    )

    return p1, p2, p3, p4, p5, summary_html


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE DESCRIPTION CARDS
# ═══════════════════════════════════════════════════════════════════════════════
_CARD_CSS = (
    "background:#f0f4ff;border-left:4px solid {color};"
    "padding:14px 18px;border-radius:8px;margin:6px 0 14px 0"
)

_CLS = (
    "display:inline-block;font-family:monospace;font-size:0.88em;"
    "background:#eef0f8;color:#374151;padding:1px 6px;"
    "border-radius:4px;border:1px solid #d1d5db"
)
_BOLD = "font-weight:700;color:#1c2541"

STAGE_INFO = {
    "s1": ("#3a86ff",
           "🔷 Stage 1 — Detection & ROI Crop",
           "Roboflow detects the concrete surface and crops a tight Region of Interest "
           "(ROI) with a 10% padding. If no detection exceeds the confidence threshold "
           f"({CONF_THRESHOLD:.0%}), the full image is used as the ROI. A dashed bounding "
           "box is overlaid on the original to show what the model found."),
    "s2": ("#fb5607",
           "🤖 Stage 2 — ResNet-18 Classification",
           f"A fine-tuned ResNet-18 (4-class head with dropout regularisation) classifies "
           f"the ROI into <span style='{_CLS}'>crack</span>, "
           f"<span style='{_CLS}'>crack_segregation</span>, "
           f"<span style='{_CLS}'>segregation</span>, or "
           f"<span style='{_CLS}'>normal</span>. "
           "The probability bar chart shows the model's confidence for every "
           "class so you can judge borderline cases."),
    "s3": ("#8338ec",
           "🧩 Stage 3 — U-Net Segmentation & LAI Heatmap",
           f"A U-Net produces a pixel-level binary anomaly mask for "
           f"<span style='{_BOLD}'>every class</span>. "
           "The 10×10 Local Anomaly Intensity (LAI) grid colours each cell by its local "
           "defect density — blue below the global average (GAI), red above it. "
           f"For <span style='{_CLS}'>crack</span> and "
           f"<span style='{_CLS}'>normal</span> images the mask signal will typically be low; "
           "for segregation cases it reveals spatial clustering patterns that drive stage assignment."),
    "s4": ("#ffbe0b",
           "📊 Stage 4 — Stage Assignment (KMeans vs Rule-based)",
           "Two independent methods assign a damage stage using GAI (Global Anomaly "
           "Index) and SI_ia (Segregation Index). "
           f"<span style='{_BOLD}'>Only shown for "
           f"<span style='{_CLS}'>segregation</span> and "
           f"<span style='{_CLS}'>crack_segregation</span> images.</span> "
           "KMeans is the primary data-driven result; the rule-based method uses fixed "
           "thresholds for interpretability. Disagreement is flagged explicitly."),
    "s5": ("#2ca02c",
           "📋 Stage 5 — Final Inspection Card",
           "The full card summarises everything: image with colour-coded mask overlay, "
           "identity panel (class, stage, cluster, GAI, SI-IA, estimated material mix), "
           "defect signal bars, and estimated material composition means derived from the "
           f"KMeans cluster. For <span style='{_CLS}'>crack</span> and "
           f"<span style='{_CLS}'>normal</span> images a simplified 2-panel card is shown."),
}


def _html_card(key: str) -> str:
    color, title, body = STAGE_INFO[key]
    css = _CARD_CSS.format(color=color)
    return (
        f"<div style='{css}'>"
        f"<span style='font-size:1.05em;font-weight:700;color:#1c2541;display:block;margin-bottom:5px'>{title}</span>"
        f"<span style='color:#444;font-size:0.92em;line-height:1.6'>{body}</span>"
        f"</div>"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════════
_PLACEHOLDER_IMG = Image.new("RGB", (600, 120), "#f0f4ff")

_HEADER_HTML = """
<div style="background:linear-gradient(135deg,#1c2541 0%,#2d4a8a 60%,#3a5a9a 100%);
            padding:24px 28px;border-radius:12px;margin-bottom:4px;color:white;
            box-shadow:0 4px 16px rgba(28,37,65,0.25)">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
    <span style="font-size:2em">🏗️</span>
    <h1 style="margin:0;font-size:1.6em;font-weight:800;letter-spacing:-0.02em">
      Concrete Anomaly Inspector
    </h1>
  </div>
  <p style="margin:0 0 16px;opacity:0.8;font-size:0.93em;max-width:680px">
    Upload a concrete surface photo and the full 5-stage inspection pipeline
    runs automatically — from ROI detection to material-level stage assignment.
  </p>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
                 padding:4px 13px;border-radius:20px;font-size:0.8em;font-weight:600">
      🔍 Roboflow ROI
    </span>
    <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
                 padding:4px 13px;border-radius:20px;font-size:0.8em;font-weight:600">
      🤖 ResNet-18
    </span>
    <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
                 padding:4px 13px;border-radius:20px;font-size:0.8em;font-weight:600">
      🧩 U-Net (all classes)
    </span>
    <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
                 padding:4px 13px;border-radius:20px;font-size:0.8em;font-weight:600">
      📊 KMeans Staging
    </span>
    <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
                 padding:4px 13px;border-radius:20px;font-size:0.8em;font-weight:600">
      📋 Inspection Card
    </span>
  </div>
</div>
"""

_CLASS_LEGEND_HTML = """
<div style="display:flex;gap:10px;flex-wrap:wrap;padding:10px 0 4px">
  <span style="font-size:0.82em;font-weight:600;color:#6b7280;align-self:center">Classes:</span>
  <span style="background:#e6394622;color:#e63946;border:1px solid #e6394655;
                padding:3px 11px;border-radius:12px;font-size:0.82em;font-weight:700">crack</span>
  <span style="background:#fb560722;color:#fb5607;border:1px solid #fb560755;
                padding:3px 11px;border-radius:12px;font-size:0.82em;font-weight:700">crack_segregation</span>
  <span style="background:#3a86ff22;color:#3a86ff;border:1px solid #3a86ff55;
                padding:3px 11px;border-radius:12px;font-size:0.82em;font-weight:700">segregation</span>
  <span style="background:#2dc65322;color:#2dc653;border:1px solid #2dc65355;
                padding:3px 11px;border-radius:12px;font-size:0.82em;font-weight:700">normal</span>
</div>
"""

_CSS = """
.gradio-container { max-width: 1160px !important; }
footer { display: none !important; }
.run-btn { font-size: 1.05em !important; letter-spacing: 0.02em !important; }
.stage-accordion .label-wrap { font-weight: 600 !important; }
"""

with gr.Blocks(title="Concrete Anomaly Inspector") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML(_HEADER_HTML)

    # ── Input row ─────────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=290):
            img_input = gr.Image(
                type="pil",
                label="📷 Upload Concrete Image",
                height=270,
            )
            gr.HTML(_CLASS_LEGEND_HTML)
            run_btn = gr.Button(
                "🚀 Run Inspection",
                variant="primary",
                size="lg",
                elem_classes=["run-btn"],
            )

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                gr.HTML(
                    "<p style='font-size:0.82em;color:#6b7280;margin:4px 0 10px'>"
                    "Adjust thresholds for this run. Changes take effect on the next "
                    "<em>Run Inspection</em> click. Defaults come from your <code>.env</code> file.</p>"
                )
                seg_threshold_slider = gr.Slider(
                    minimum=0.50, maximum=0.99, step=0.01,
                    value=SEG_THRESHOLD,
                    label="Segmentation Threshold (SEG_THRESHOLD)",
                    info="U-Net pixel confidence cutoff. Higher → fewer but more certain anomaly pixels.",
                )
                conf_threshold_slider = gr.Slider(
                    minimum=0.30, maximum=0.99, step=0.01,
                    value=CONF_THRESHOLD,
                    label="Detection Confidence (CONF_THRESHOLD)",
                    info="Roboflow ROI confidence cutoff. Lower → accept weaker detections.",
                )
                show_mask_checkbox = gr.Checkbox(
                    value=SHOW_MASK,
                    label="Show Mask Overlay (SHOW_MASK)",
                    info="Blend the anomaly mask colour over the crop in the inspection card.",
                )

        with gr.Column(scale=2, min_width=360):
            gr.HTML(
                "<div style='font-size:0.78em;font-weight:700;color:#9ca3af;"
                "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px'>"
                "Inspection Summary</div>"
            )
            summary_out = gr.HTML(
                value=(
                    "<div style='background:#f9fafb;border-radius:10px;padding:20px;"
                    "border:1px solid #e5e7eb;color:#9ca3af;font-size:0.9em;"
                    "text-align:center;font-family:sans-serif'>"
                    "⬅️ Upload an image and click "
                    "<span style='font-weight:700;color:#6b7280'>Run Inspection</span>"
                    " to see results."
                    "</div>"
                ),
            )

    gr.HTML("<hr style='border:none;border-top:1px solid #e5e7eb;margin:18px 0 10px'>")

    # ── Stage outputs ─────────────────────────────────────────────────────────
    with gr.Accordion("📌 Stage 1 — Detection & ROI Crop", open=True,
                      elem_classes=["stage-accordion"]):
        gr.HTML(_html_card("s1"))
        plot1_out = gr.Image(label="Detection & ROI Crop")

    with gr.Accordion("📌 Stage 2 — Classification", open=True,
                      elem_classes=["stage-accordion"]):
        gr.HTML(_html_card("s2"))
        plot2_out = gr.Image(label="ResNet-18 Classification")

    with gr.Accordion("📌 Stage 3 — U-Net Segmentation & LAI Heatmap", open=True,
                      elem_classes=["stage-accordion"]):
        gr.HTML(_html_card("s3"))
        plot3_out = gr.Image(label="U-Net Segmentation & LAI Heatmap",
                              value=_PLACEHOLDER_IMG)

    with gr.Accordion("📌 Stage 4 — Stage Assignment", open=True,
                      elem_classes=["stage-accordion"]):
        gr.HTML(_html_card("s4"))
        plot4_out = gr.Image(label="KMeans vs Rule-based Stage")
        gr.HTML(
            "<p style='font-size:0.83em;color:#6b7280;margin:6px 0 10px;padding-left:4px'>"
            "ℹ️ This plot is only generated for "
            "<span style='font-weight:700;color:#555'>segregation</span> "
            "and <span style='font-weight:700;color:#555'>crack_segregation</span> images. "
            "It will be blank for "
            "<span style='font-family:monospace;background:#f3f4f6;padding:1px 5px;border-radius:3px'>crack</span>"
            " or "
            "<span style='font-family:monospace;background:#f3f4f6;padding:1px 5px;border-radius:3px'>normal</span>"
            " results.</p>"
        )

    with gr.Accordion("📌 Stage 5 — Final Inspection Card", open=True,
                      elem_classes=["stage-accordion"]):
        gr.HTML(_html_card("s5"))
        plot5_out = gr.Image(label="Final Inspection Card")

    # ── Concrete damage stage reference table ─────────────────────────────────
    with gr.Accordion("📖 Concrete Damage Stage Reference", open=False):
        gr.HTML("""
<div style="overflow-x:auto;margin:6px 0;background:#1a1d2e;border-radius:10px;padding:2px;border:1px solid #2e3350">
<table style="width:100%;border-collapse:collapse;font-size:0.87em;font-family:sans-serif;background:transparent">
  <thead>
    <tr style="background:#111827;text-align:left;border-bottom:2px solid #3a4060">
      <th style="padding:11px 14px;color:#a8b4cc;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;font-size:0.82em">Stage</th>
      <th style="padding:11px 14px;color:#a8b4cc;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;font-size:0.82em">Label</th>
      <th style="padding:11px 14px;color:#a8b4cc;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;font-size:0.82em">GAI range</th>
      <th style="padding:11px 14px;color:#a8b4cc;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;font-size:0.82em">SI_ia range</th>
      <th style="padding:11px 14px;color:#a8b4cc;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;font-size:0.82em">Damage level</th>
      <th style="padding:11px 14px;color:#a8b4cc;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;font-size:0.82em">Visual description</th>
      <th style="padding:11px 14px;color:#a8b4cc;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;font-size:0.82em">Recommended action</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#1e2d22;border-bottom:1px solid #2a3540">
      <td style="padding:9px 14px;font-weight:700;color:#2dc653;font-size:0.95em">NORMAL</td>
      <td style="padding:9px 14px;color:#c8d6c0;font-weight:500">No defect</td>
      <td style="padding:9px 14px;color:#7a9a80;font-family:monospace">&mdash;</td>
      <td style="padding:9px 14px;color:#7a9a80;font-family:monospace">&mdash;</td>
      <td style="padding:9px 14px"><span style="background:#1a3d1e;color:#2dc653;font-weight:700;padding:2px 10px;border-radius:12px;font-size:0.88em">None</span></td>
      <td style="padding:9px 14px;color:#b8ccb4">Uniform surface, no visible segregation or cracking</td>
      <td style="padding:9px 14px;color:#b8ccb4">Routine monitoring only</td>
    </tr>
    <tr style="background:#1a2236;border-bottom:1px solid #2a3540">
      <td style="padding:9px 14px;font-weight:700;color:#5b9fff;font-size:0.95em">STAGE 1</td>
      <td style="padding:9px 14px;color:#c0ccde;font-weight:500">Incipient</td>
      <td style="padding:9px 14px;color:#7a8fb0;font-family:monospace">GAI &lt; 15%</td>
      <td style="padding:9px 14px;color:#7a8fb0;font-family:monospace">SI_ia &lt; 30</td>
      <td style="padding:9px 14px"><span style="background:#162040;color:#5b9fff;font-weight:700;padding:2px 10px;border-radius:12px;font-size:0.88em">Low</span></td>
      <td style="padding:9px 14px;color:#b0bcd0">Sparse, isolated anomaly pixels; damage not yet clustered</td>
      <td style="padding:9px 14px;color:#b0bcd0">Log and monitor; re-inspect in 3&ndash;6 months</td>
    </tr>
    <tr style="background:#26221a;border-bottom:1px solid #2a3540">
      <td style="padding:9px 14px;font-weight:700;color:#f0a500;font-size:0.95em">STAGE 2</td>
      <td style="padding:9px 14px;color:#d4c8a8;font-weight:500">Developing</td>
      <td style="padding:9px 14px;color:#9a8c60;font-family:monospace">GAI 15&ndash;35%</td>
      <td style="padding:9px 14px;color:#9a8c60;font-family:monospace">SI_ia &ge; 30</td>
      <td style="padding:9px 14px"><span style="background:#3a2c00;color:#f0a500;font-weight:700;padding:2px 10px;border-radius:12px;font-size:0.88em">Medium</span></td>
      <td style="padding:9px 14px;color:#c8bc98">Anomaly pixels forming clusters; spatial non-uniformity rising</td>
      <td style="padding:9px 14px;color:#c8bc98">Detailed inspection; consider surface treatment</td>
    </tr>
    <tr style="background:#261e1a;border-bottom:1px solid #2a3540">
      <td style="padding:9px 14px;font-weight:700;color:#ff6b35;font-size:0.95em">STAGE 3</td>
      <td style="padding:9px 14px;color:#d4c0b8;font-weight:500">Advanced</td>
      <td style="padding:9px 14px;color:#9a7060;font-family:monospace">GAI &ge; 35%</td>
      <td style="padding:9px 14px;color:#9a7060;font-family:monospace">SI_ia &ge; 60</td>
      <td style="padding:9px 14px"><span style="background:#3a1800;color:#ff6b35;font-weight:700;padding:2px 10px;border-radius:12px;font-size:0.88em">High</span></td>
      <td style="padding:9px 14px;color:#ccb8ae">Heavy, spatially clustered damage; large contiguous zones</td>
      <td style="padding:9px 14px;color:#ccb8ae">Structural assessment required; repair planning</td>
    </tr>
    <tr style="background:#281818;border-bottom:1px solid #2a3540">
      <td style="padding:9px 14px;font-weight:700;color:#ff4d58;font-size:0.95em">STAGE 4</td>
      <td style="padding:9px 14px;color:#d4b8b8;font-weight:500">Severe</td>
      <td style="padding:9px 14px;color:#9a6060;font-family:monospace">GAI &ge; 35%</td>
      <td style="padding:9px 14px;color:#9a6060;font-family:monospace">SI_ia &lt; 40</td>
      <td style="padding:9px 14px"><span style="background:#3a0808;color:#ff4d58;font-weight:700;padding:2px 10px;border-radius:12px;font-size:0.88em">Critical</span></td>
      <td style="padding:9px 14px;color:#ccaaaa">Diffuse widespread damage covering most of the surface</td>
      <td style="padding:9px 14px;color:#ccaaaa">Immediate intervention; potential load-bearing risk</td>
    </tr>
    <tr style="background:#211a2e;border-bottom:1px solid #2a3540">
      <td style="padding:9px 14px;font-weight:700;color:#a066e8;font-size:0.95em">TRANSITIONAL</td>
      <td style="padding:9px 14px;color:#c8b8d8;font-weight:500">Borderline</td>
      <td style="padding:9px 14px;color:#8070a0;font-family:monospace">15&ndash;35%</td>
      <td style="padding:9px 14px;color:#8070a0;font-family:monospace">Mixed</td>
      <td style="padding:9px 14px"><span style="background:#2a1040;color:#a066e8;font-weight:700;padding:2px 10px;border-radius:12px;font-size:0.88em">Medium</span></td>
      <td style="padding:9px 14px;color:#c0b0d0">Metrics fall between defined stages; ambiguous spatial pattern</td>
      <td style="padding:9px 14px;color:#c0b0d0">Cross-check with KMeans result; manual review advised</td>
    </tr>
    <tr style="background:#1e1e22">
      <td style="padding:9px 14px;font-weight:700;color:#7a7a8e;font-size:0.95em">UNRELIABLE</td>
      <td style="padding:9px 14px;color:#9898a8;font-weight:500">Low signal</td>
      <td style="padding:9px 14px;color:#6a6a78;font-family:monospace">GAI &lt; 5%</td>
      <td style="padding:9px 14px;color:#6a6a78;font-family:monospace">&mdash;</td>
      <td style="padding:9px 14px"><span style="background:#2a2a32;color:#7a7a8e;font-weight:700;padding:2px 10px;border-radius:12px;font-size:0.88em">N/A</span></td>
      <td style="padding:9px 14px;color:#9898a8">Too few anomaly pixels for reliable stage inference</td>
      <td style="padding:9px 14px;color:#9898a8">Lower SEG_THRESHOLD or inspect image quality</td>
    </tr>
  </tbody>
</table>
<p style="font-size:0.78em;color:#5a6680;margin:8px 14px 6px;line-height:1.5">
  <span style="font-weight:700;color:#7a8aaa">GAI</span> = Global Anomaly Index (anomalous pixels / total pixels × 100).&nbsp;
  <span style="font-weight:700;color:#7a8aaa">SI_ia</span> = Segregation Index via Image Analysis (spatial non-uniformity, 0–100).
  Stage boundaries reflect the rule-based method; KMeans may differ on borderline cases.
</p>
</div>
""")

    # ── Wire up ───────────────────────────────────────────────────────────────
    run_btn.click(
        fn=run_pipeline,
        inputs=[img_input, seg_threshold_slider, conf_threshold_slider, show_mask_checkbox],
        outputs=[plot1_out, plot2_out, plot3_out, plot4_out, plot5_out, summary_out],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=_CSS,
    )