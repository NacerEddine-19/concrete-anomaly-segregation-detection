"""
Inspection card renderer — produces the matplotlib figure shown in the
screenshot: [Image] [Identity] [Defect signals] [Material means].

For Normal / Crack classes a simpler 2-panel card is rendered instead.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — safe for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure

from constants import (
    CLASSES, DAMAGE_COLORS, MATERIAL_CLUSTER_MEANS,
)


# ─── colour palette ───────────────────────────────────────────────────────────
_CLASS_COLORS = {
    "crack":            (255,  50,  50),
    "crack_segregation":(255, 165,   0),
    "segregation":      ( 50, 100, 255),
    "normal":           ( 50, 200,  50),
}
_MAT_LABELS = [
    ("mat_fine_aggregate",    "Fine Agg."),
    ("mat_coarse_aggregate",  "Coarse Agg."),
    ("mat_superplasticizer",  "Superplast."),
    ("mat_water",             "Water"),
    ("mat_fly_ash",           "Fly Ash"),
    ("mat_blast_furnace_slag","BF Slag"),
    ("mat_cement",            "Cement"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  FULL INSPECTION CARD  (segregation / crack_segregation)
# ══════════════════════════════════════════════════════════════════════════════

def render_full_card(
    image_name:    str,
    crop_img:      np.ndarray,          # RGB (H, W, 3)
    defect_cluster: int,                # 0 / 1 / 2
    stage_rank:    int,                 # 1 / 2 / 3
    damage_label:  str,                 # "Low damage" etc.
    confidence:    float,               # 0–1
    gai_score:     float,               # 0–100
    si_ia_score:   float,               # 0–100
    pred_mask:     "np.ndarray | None" = None,  # binary mask (H, W)
) -> Figure:
    """
    Produces the 4-panel inspection card shown in the screenshot:
      Panel 1 — crop image with coloured border
      Panel 2 — Identity table
      Panel 3 — Defect signals bar chart
      Panel 4 — Material means (cluster) bar chart
    """
    mat   = MATERIAL_CLUSTER_MEANS[defect_cluster]
    color = DAMAGE_COLORS[defect_cluster]          # hex "#2ca02c" etc.
    rgb   = tuple(int(color.lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4))

    fig = plt.figure(figsize=(17, 4.5), facecolor="#f8f9fa")
    fig.suptitle(
        f"{image_name}  ·  {damage_label}",
        fontsize=13, fontweight="bold", color="#1c2541",
        x=0.01, ha="left", y=1.01,
    )

    gs = gridspec.GridSpec(
        1, 4, figure=fig,
        width_ratios=[1.5, 1.1, 1.0, 1.3],
        wspace=0.38,
    )

    # ── Panel 1: Image ────────────────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[0, 0])
    display = crop_img.copy()
    if pred_mask is not None and pred_mask.any():
        # resize mask to crop size if needed
        h, w = crop_img.shape[:2]
        m = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        overlay = display.copy()
        clr = _CLASS_COLORS.get("segregation", (50, 100, 255))
        overlay[m > 0] = clr
        display = cv2.addWeighted(display, 0.55, overlay, 0.45, 0)

    ax_img.imshow(display)
    ax_img.set_title("Image", fontsize=11, fontweight="bold", pad=6)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3.5)

    # ── Panel 2: Identity table ───────────────────────────────────────────────
    ax_id = fig.add_subplot(gs[0, 1])
    ax_id.set_xticks([]); ax_id.set_yticks([])
    ax_id.set_facecolor("#f0f8f0")
    for spine in ax_id.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2)
    ax_id.set_title("Identity", fontsize=11, fontweight="bold", pad=6)

    rows = [
        ("Defect cluster",  f"{defect_cluster}",          True),
        ("Stage rank",      f"{stage_rank}",               True),
        ("Confidence",      f"{confidence:.3f}",           True),
        ("GAI score",       f"{gai_score:.3f}",            True),
        ("SI-IA score",     f"{si_ia_score:.3f}",          True),
        ("— Material —",    "",                            False),
        ("Strength",        f"{mat['mat_strength']} MPa",  True),
        ("Age",             f"{mat['mat_age']:.0f} days",  True),
        ("Water",           f"{mat['mat_water']} kg/m³",   True),
    ]
    n = len(rows)
    for i, (label, value, bold_val) in enumerate(rows):
        y = 1 - (i + 0.5) / n
        ax_id.text(0.05, y, label, transform=ax_id.transAxes,
                   ha="left", va="center", fontsize=9, color="#555555")
        if value:
            ax_id.text(0.98, y, value, transform=ax_id.transAxes,
                       ha="right", va="center",
                       fontsize=9, fontweight="bold" if bold_val else "normal",
                       color="#1c2541")
    ax_id.set_xlim(0, 1); ax_id.set_ylim(0, 1)

    # ── Panel 3: Defect signals bar chart ─────────────────────────────────────
    ax_def = fig.add_subplot(gs[0, 2])
    sig_labels = ["SI-IA score", "GAI score", "Confidence"]
    sig_values = [si_ia_score, gai_score, confidence]
    # normalise to [0, 1] for display — SI_ia and GAI are 0–100, conf 0–1
    # We display raw values but scale bars by max observed to keep them readable
    _max = max(max(sig_values), 1e-6)
    norm_vals = [v / _max for v in sig_values]

    bars = ax_def.barh(sig_labels, norm_vals, color=color, alpha=0.85)
    for bar, val in zip(bars, sig_values):
        ax_def.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left",
            fontsize=8, fontweight="bold", color="#1c2541",
        )
    ax_def.set_title("Defect signals", fontsize=11, fontweight="bold", pad=6)
    ax_def.set_xlabel("Score", fontsize=9)
    ax_def.set_xlim(0, 1.35)
    ax_def.invert_yaxis()
    ax_def.grid(axis="x", linestyle="--", alpha=0.4)
    ax_def.set_yticks(range(len(sig_labels)))
    ax_def.set_yticklabels(sig_labels, fontsize=8.5)
    ax_def.tick_params(axis="x", labelsize=8)

    # ── Panel 4: Material means bar chart ─────────────────────────────────────
    ax_mat = fig.add_subplot(gs[0, 3])
    mat_keys   = [k for k, _ in _MAT_LABELS]
    mat_names  = [n for _, n in _MAT_LABELS]
    mat_values = [mat[k] for k in mat_keys]

    ax_mat.barh(mat_names, mat_values, color=color, alpha=0.85)
    for i, v in enumerate(mat_values):
        ax_mat.text(v + 5, i, f"{v:.1f}", va="center", fontsize=8,
                    fontweight="bold", color="#1c2541")
    ax_mat.set_title(
        f"Material means (cluster {defect_cluster})",
        fontsize=11, fontweight="bold", pad=6,
    )
    ax_mat.set_xlabel("kg/m³", fontsize=9)
    ax_mat.invert_yaxis()
    ax_mat.set_xlim(0, max(mat_values) * 1.25)
    ax_mat.grid(axis="x", linestyle="--", alpha=0.4)
    ax_mat.tick_params(axis="both", labelsize=8)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIMPLE CARD  (normal / crack)
# ══════════════════════════════════════════════════════════════════════════════

def render_simple_card(
    image_name:  str,
    crop_img:    np.ndarray,       # RGB (H, W, 3)
    cls_name:    str,              # "normal" | "crack"
    confidence:  float,           # 0–1
    all_probs:   np.ndarray,      # shape (4,)
    pred_mask:   "np.ndarray | None" = None,
) -> Figure:
    """
    2-panel card for Normal / Crack detections:
      Panel 1 — crop image with grey/red overlay
      Panel 2 — classification summary (class probabilities)
    """
    is_normal  = cls_name == "normal"
    border_col = "#2ca02c" if is_normal else "#d62728"
    overlay_cl = _CLASS_COLORS.get(cls_name, (180, 180, 180))

    fig = plt.figure(figsize=(11, 4.5), facecolor="#f8f9fa")
    fig.suptitle(
        f"{image_name}  ·  {cls_name.replace('_', ' ').title()}",
        fontsize=13, fontweight="bold", color="#1c2541",
        x=0.01, ha="left", y=1.01,
    )

    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.4, 1.0], wspace=0.35)

    # ── Panel 1: image ────────────────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[0, 0])
    display = crop_img.copy()
    if not is_normal and pred_mask is not None and pred_mask.any():
        h, w = crop_img.shape[:2]
        m    = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        ov   = display.copy()
        ov[m > 0] = overlay_cl
        display = cv2.addWeighted(display, 0.55, ov, 0.45, 0)

    ax_img.imshow(display)
    ax_img.set_title("Image", fontsize=11, fontweight="bold", pad=6)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_edgecolor(border_col); spine.set_linewidth(3.5)

    # ── Panel 2: class probabilities ─────────────────────────────────────────
    ax_prob = fig.add_subplot(gs[0, 1])
    from constants import CLASSES
    colors = ["#d62728" if c == cls_name else "#aec6cf" for c in CLASSES]
    bars   = ax_prob.barh(CLASSES, all_probs, color=colors, alpha=0.85)
    for bar, val in zip(bars, all_probs):
        ax_prob.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%", va="center", ha="left",
            fontsize=8.5, fontweight="bold", color="#1c2541",
        )
    ax_prob.set_title("Class probabilities", fontsize=11, fontweight="bold", pad=6)
    ax_prob.set_xlim(0, 1.35)
    ax_prob.invert_yaxis()
    ax_prob.set_xlabel("Probability", fontsize=9)
    ax_prob.grid(axis="x", linestyle="--", alpha=0.4)
    ax_prob.tick_params(axis="both", labelsize=8.5)

    # Confidence badge
    ax_prob.text(
        0.98, 0.02,
        f"Confidence: {confidence * 100:.1f}%",
        transform=ax_prob.transAxes,
        ha="right", va="bottom",
        fontsize=9, color=border_col, fontweight="bold",
    )

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MASK OVERLAY HELPER  (for the Streamlit sidebar / debug view)
# ══════════════════════════════════════════════════════════════════════════════

def overlay_mask_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    cls_name: str,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend anomaly mask over the image with the class colour."""
    clr     = _CLASS_COLORS.get(cls_name, (255, 0, 0))
    h, w    = img.shape[:2]
    m       = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    overlay[m > 0] = clr
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
