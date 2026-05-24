"""
Concrete Inspector — Streamlit Application
==========================================

Pipeline
--------
1. Upload an image
2. Roboflow segments and crops the concrete area(s)
3. ResNet-18 classifier labels each crop:
      crack | crack_segregation | segregation | normal
4. If the label is [segregation | crack_segregation]:
      → U-Net (best_unet.pth) produces a binary anomaly mask
      → SI_ia (Segregation Index) and GAI are computed
      → Defect cluster is assigned [0 / 1 / 2]
      → Static material cluster means are looked up
      → Full 4-panel inspection card is rendered
5. If the label is [normal | crack]:
      → Simpler 2-panel card with class probabilities is rendered

Model files expected
--------------------
  saved_models/best_unet.pth        — U-Net segmentation weights
  saved_models/best_classifier.pth  — ResNet-18 classifier weights

Environment variables (optional)
---------------------------------
  ROBOFLOW_API_KEY   — your Roboflow API key
  ROBOFLOW_MODEL_ID  — model ID string (e.g. "concrete-seg-v2/1")
"""

import os
import sys
import io

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

# ─── resolve imports regardless of working directory ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import CLASSES, MATERIAL_CLUSTER_MEANS, DAMAGE_COLORS
from core.architectures import UNet, build_classifier
from core.pipeline import (
    preprocess_for_inference,
    preprocess_for_seg,
    classify_image,
    segment_image,
    extract_features,
    remove_edge_artifacts,
    apply_circular_mask,
    assign_defect_cluster,
    get_stage_info,
)
from core.visualization import render_full_card, render_simple_card
from core.roboflow_seg import get_concrete_rois

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Concrete Inspector",
    page_icon="🏗️",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { border-radius: 8px; font-weight: 600; }
    .metric-card {
        background: white; border-radius: 10px;
        padding: 14px 18px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
        margin-bottom: 8px;
    }
    .stage-badge {
        display: inline-block; padding: 4px 12px;
        border-radius: 20px; font-weight: 700;
        font-size: 13px; color: white;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DEVICE
# ══════════════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING  (cached — runs once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading segmentation model (U-Net)…")
def load_unet(model_path: str = "saved_models/best_unet.pth"):
    """Load best_unet.pth — the U-Net segmentation model."""
    model = UNet(in_channels=3, out_channels=1).to(device)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        # Handle state-dict wrapped in 'model_state_dict' key (common checkpoint pattern)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
    else:
        st.warning(
            f"⚠️  `{model_path}` not found. Segmentation will return blank masks.\n\n"
            "Place **best_unet.pth** in the `saved_models/` folder and restart."
        )
    return model


@st.cache_resource(show_spinner="Loading classification model (ResNet-18)…")
def load_classifier(model_path: str = "saved_models/best_classifier.pth"):
    """Load the ResNet-18 4-class classifier."""
    model = build_classifier(device, num_classes=len(CLASSES))
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
    else:
        st.warning(
            f"⚠️  `{model_path}` not found. Classification will return random labels.\n\n"
            "Place **best_classifier.pth** in the `saved_models/` folder and restart."
        )
    return model


@st.cache_resource(show_spinner="Connecting to Roboflow…")
def load_roboflow(api_key: str, model_id: str):
    """Load the Roboflow concrete-segmentation model."""
    try:
        from inference import get_model
        return get_model(model_id=model_id, api_key=api_key)
    except Exception as e:
        st.warning(f"⚠️  Roboflow unavailable: {e}. Will use full image as ROI.")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Gray_square.svg/"
        "120px-Gray_square.svg.png",
        width=60,
    )
    st.title("🏗️ Concrete Inspector")
    st.caption("AI-powered concrete defect analysis")

    st.divider()
    st.subheader("⚙️ Model paths")
    unet_path       = st.text_input("U-Net weights",       value="saved_models/best_unet.pth")
    cls_path        = st.text_input("Classifier weights",  value="saved_models/best_classifier.pth")

    st.divider()
    st.subheader("🌐 Roboflow")
    rf_api_key  = st.text_input(
        "API key", value=os.getenv("ROBOFLOW_API_KEY", ""), type="password"
    )
    rf_model_id = st.text_input(
        "Model ID", value=os.getenv("ROBOFLOW_MODEL_ID", "concrete-detection-lvb3q/1")
    )
    rf_conf     = st.slider("Detection confidence threshold", 0.0, 1.0, 0.50, 0.05)

    st.divider()
    st.subheader("🔬 Segmentation")
    seg_thresh  = st.slider("U-Net threshold", 0.0, 1.0, 0.50, 0.05)
    show_mask   = st.checkbox("Show anomaly mask overlay", value=True)

    st.divider()
    st.caption(
        "**Pipeline**\n"
        "1. Roboflow → crop concrete ROI\n"
        "2. ResNet-18 → classify crop\n"
        "3. U-Net → binary mask (if seg class)\n"
        "4. SI_ia, GAI → defect cluster\n"
        "5. Material cluster means lookup"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODELS (lazy — only on first use)
# ══════════════════════════════════════════════════════════════════════════════

seg_model = load_unet(unet_path)
cls_model = load_classifier(cls_path)
rf_model  = load_roboflow(rf_api_key, rf_model_id) if rf_api_key else None


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("🏗️ Concrete Anomaly Inspector")
st.markdown(
    "Upload a concrete surface image. The pipeline will **detect**, **classify**, "
    "and **characterise** any anomalies."
)

uploaded = st.file_uploader(
    "Upload image", type=["jpg", "jpeg", "png", "bmp", "tiff"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("👆 Upload an image to begin analysis.", icon="📷")
    st.stop()

# ── Load & display uploaded image ─────────────────────────────────────────────
file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
orig_bgr   = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if orig_bgr is None:
    st.error("Could not decode the uploaded image. Please try a different file.")
    st.stop()

orig_rgb   = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
image_name = os.path.splitext(uploaded.name)[0]

col_img, col_info = st.columns([1, 1])
with col_img:
    st.subheader("Uploaded image")
    st.image(orig_rgb, use_container_width=True)
with col_info:
    st.subheader("Image info")
    H, W = orig_rgb.shape[:2]
    st.markdown(f"- **File:** `{uploaded.name}`")
    st.markdown(f"- **Size:** {W} × {H} px")
    st.markdown(f"- **Device:** `{device}`")
    if rf_model is None:
        st.warning("Roboflow not connected — using full image as ROI.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

run_btn = st.button("▶ Run Inspection", type="primary", use_container_width=True)
if not run_btn:
    st.stop()

with st.spinner("Running pipeline…"):

    # ── Step 1 · Roboflow → ROI crops ─────────────────────────────────────────
    rois = get_concrete_rois(
        orig_rgb,
        rf_model=rf_model,
        conf_threshold=rf_conf,
    )

    st.success(f"✅  Roboflow detected **{len(rois)}** concrete region(s).")

    for det_idx, roi in enumerate(rois):
        crop_img = roi["crop"]       # RGB numpy array
        det_conf = roi["det_conf"]

        st.markdown(f"---\n### Detection {det_idx + 1} / {len(rois)}")

        # ── Step 2 · Classify the crop ────────────────────────────────────────
        cls_tensor = preprocess_for_inference(crop_img, size=224)[1]
        predicted_class, confidence, all_probs = classify_image(cls_model, cls_tensor, device)

        # ── Step 3 · Segment (only for segregation classes) ───────────────────
        pred_mask = None
        gai_score = 0.0
        si_ia     = 0.0
        features  = {}

        needs_segmentation = predicted_class in ("segregation", "crack_segregation")

        if needs_segmentation:
            _, seg_tensor = preprocess_for_seg(crop_img)
            mask_raw      = segment_image(seg_model, seg_tensor, device,
                                          threshold=seg_thresh, model_type="unet")

            # Edge-artifact removal (mirrors notebook Section 5)
            crop_h, crop_w = crop_img.shape[:2]
            aspect_ratio   = crop_h / max(crop_w, 1)
            if aspect_ratio > 1.4:
                pred_mask = apply_circular_mask(
                    remove_edge_artifacts(mask_raw, margin_pct=0.05), margin=0.03
                )
            else:
                pred_mask = remove_edge_artifacts(mask_raw, margin_pct=0.04)

            # Feature extraction → GAI + SI_ia
            features  = extract_features(pred_mask, grid_size=10)
            gai_score = features["anomaly_area_pct"]   # 0–100
            si_ia     = features["si_ia_score"]         # 0–100

        # ── Step 4 · Assign defect cluster ────────────────────────────────────
        defect_cluster              = assign_defect_cluster(gai_score, si_ia, predicted_class)
        stage_rank, damage_label, stage_label = get_stage_info(defect_cluster)

        # ─────────────────────────────────────────────────────────────────────
        #  DISPLAY RESULTS
        # ─────────────────────────────────────────────────────────────────────

        if needs_segmentation and defect_cluster >= 0:
            # ── Full inspection card ─────────────────────────────────────────
            mat      = MATERIAL_CLUSTER_MEANS[defect_cluster]
            color_hex = DAMAGE_COLORS[defect_cluster]

            # Top metrics row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Class",          predicted_class.replace("_", " ").title())
            m2.metric("Confidence",     f"{confidence * 100:.1f}%")
            m3.metric("GAI score",      f"{gai_score:.3f}")
            m4.metric("SI-IA score",    f"{si_ia:.3f}")
            m5.metric("Defect cluster", f"{defect_cluster} · {damage_label}")

            st.markdown(
                f'<span class="stage-badge" style="background:{color_hex};">'
                f"{stage_label}</span>",
                unsafe_allow_html=True,
            )

            # Render matplotlib inspection card
            fig = render_full_card(
                image_name     = f"{image_name} (det {det_idx + 1})",
                crop_img       = crop_img,
                defect_cluster = defect_cluster,
                stage_rank     = stage_rank,
                damage_label   = damage_label,
                confidence     = confidence,
                gai_score      = gai_score,
                si_ia_score    = si_ia,
                pred_mask      = pred_mask if show_mask else None,
            )
            st.pyplot(fig, use_container_width=True)
            fig.clear(); plt.close(fig)

            # Optional: mask viewer
            if show_mask and pred_mask is not None:
                with st.expander("🗺️ Anomaly mask details"):
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.caption("Binary mask")
                        mask_display = (pred_mask * 255).astype(np.uint8)
                        st.image(
                            cv2.applyColorMap(mask_display, cv2.COLORMAP_HOT),
                            channels="BGR", use_container_width=True,
                        )
                    with col_m2:
                        st.caption("Mask statistics")
                        st.json({
                            "anomaly_pixels":  features.get("anomaly_pixels", 0),
                            "total_pixels":    features.get("total_pixels", 0),
                            "anomaly_area_%":  f"{gai_score:.4f}",
                            "SI_ia_score":     f"{si_ia:.4f}",
                            "num_regions":     features.get("num_regions", 0),
                        })

            # Material composition table
            with st.expander("🧱 Full material composition"):
                import pandas as pd
                mat_df = pd.DataFrame([{
                    "Component":           "Cement (kg/m³)",
                    "Mean value":          mat["mat_cement"],
                }, {
                    "Component":           "Blast Furnace Slag (kg/m³)",
                    "Mean value":          mat["mat_blast_furnace_slag"],
                }, {
                    "Component":           "Fly Ash (kg/m³)",
                    "Mean value":          mat["mat_fly_ash"],
                }, {
                    "Component":           "Water (kg/m³)",
                    "Mean value":          mat["mat_water"],
                }, {
                    "Component":           "Superplasticizer (kg/m³)",
                    "Mean value":          mat["mat_superplasticizer"],
                }, {
                    "Component":           "Coarse Aggregate (kg/m³)",
                    "Mean value":          mat["mat_coarse_aggregate"],
                }, {
                    "Component":           "Fine Aggregate (kg/m³)",
                    "Mean value":          mat["mat_fine_aggregate"],
                }, {
                    "Component":           "Strength (MPa)",
                    "Mean value":          mat["mat_strength"],
                }, {
                    "Component":           "Age (days)",
                    "Mean value":          mat["mat_age"],
                }])
                st.dataframe(mat_df, use_container_width=True, hide_index=True)

        else:
            # ── Simple card (Normal / Crack) ──────────────────────────────────
            m1, m2 = st.columns(2)
            m1.metric("Class",      predicted_class.replace("_", " ").title())
            m2.metric("Confidence", f"{confidence * 100:.1f}%")

            fig = render_simple_card(
                image_name = f"{image_name} (det {det_idx + 1})",
                crop_img   = crop_img,
                cls_name   = predicted_class,
                confidence = confidence,
                all_probs  = all_probs,
                pred_mask  = pred_mask if show_mask else None,
            )
            st.pyplot(fig, use_container_width=True)
            fig.clear(); plt.close(fig)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Concrete Inspector · ResNet-18 classifier + U-Net segmentation + Roboflow detection · "
    "Material means from K-Means cluster analysis (`material_set_enriched.ipynb`)"
)
