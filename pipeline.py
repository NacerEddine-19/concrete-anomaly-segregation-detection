"""
Inference utilities — mirrors Section 4 of Concrete_Anomaly_Detection_v2.ipynb
exactly, plus the edge-artifact helpers from Section 5.

Functions
---------
preprocess_for_inference   load image → (rgb_np, classifier_tensor)
preprocess_for_seg         load image → (rgb_np, seg_tensor)
classify_image             ResNet-18 forward pass → (class_name, conf, all_probs)
segment_image              U-Net forward pass → binary mask np.uint8
extract_features           binary mask → GAI, SI_ia, region stats
remove_edge_artifacts      zero border band of margin_pct on all four sides
apply_circular_mask        zero outside inscribed ellipse (cylindrical specimens)
assign_defect_cluster      (gai_pct, si_ia, cls_name) → cluster_id [0–2]
get_stage_info             cluster_id → (stage_rank, damage_label, stage_label)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from constants import (
    CLASSES, IDX_TO_CLASS,
    SEG_IMG_SIZE, CLS_IMG_SIZE,
    GAI_MIN_THRESHOLD, GAI_LOW, GAI_HIGH, SI_LOW, SI_HIGH,
    DAMAGE_LABELS,
)

# ─── ImageNet normalisation (used for both classifier and U-Net input) ────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════════════════
#  PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_for_inference(image_input, size: int = 224):
    """
    Load an image from a file path, numpy array, or PIL Image and return:
      - orig_img : RGB numpy array  (H, W, 3)
      - tensor   : normalised torch tensor (1, 3, size, size)

    Matches the notebook's `preprocess_for_inference` exactly.
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("RGB"))
    else:
        img = image_input  # assume RGB numpy array

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    tensor = transform(Image.fromarray(img)).unsqueeze(0)
    return img, tensor


def preprocess_for_seg(image_input, size: int = SEG_IMG_SIZE):
    """Convenience wrapper that calls preprocess_for_inference with SEG_IMG_SIZE."""
    return preprocess_for_inference(image_input, size=size)


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_image(model, tensor: torch.Tensor, device: torch.device):
    """
    Run the ResNet-18 classifier.

    Returns
    -------
    class_name : str
    confidence : float  (0–1)
    all_probs  : np.ndarray of shape (4,) — softmax probabilities for each class
    """
    model.eval()
    with torch.no_grad():
        probs      = F.softmax(model(tensor.to(device)), dim=1)
        conf, pred = probs.max(1)
    return IDX_TO_CLASS[pred.item()], conf.item(), probs.squeeze().cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def segment_image(
    model,
    image_input,          # preprocessed tensor (1, 3, H, W) for unet/cnn
    device: torch.device,
    threshold: float = 0.5,
    model_type: str = "unet",
) -> np.ndarray:
    """
    Unified segmentation runner — handles unet and cnn backends.

    Returns
    -------
    binary mask : np.uint8 array (H, W), values 0 or 1
    """
    if model_type in ("unet", "cnn"):
        model.eval()
        with torch.no_grad():
            pred = model(image_input.to(device))
        return (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'unet' or 'cnn'.")


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (SI_ia, GAI, region stats)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(mask_np: np.ndarray, grid_size: int = 10) -> dict:
    """
    Extract geometric features and compute SI_ia (Segregation Index via Image Analysis).

    Steps (from the notebook):
      1. GAI = total anomaly pixels / total pixels
      2. Divide mask into N×N grid
      3. For each cell: LAI = anomaly pixels in cell / total pixels in cell
      4. For each cell: LAD = |LAI − GAI|
      5. LDC = mean of all LADs
      6. SI_ia = LDC / (2 × GAI × (1 − GAI)) × 100
    """
    binary = (mask_np * 255).astype(np.uint8) if mask_np.max() <= 1 else mask_np.astype(np.uint8)
    H, W   = binary.shape
    total  = H * W
    anom   = int(np.sum(binary > 0))

    # 1. Global Aggregate Index
    gai = anom / total if total > 0 else 0.0

    # 2–6. SI_ia via spatial grid
    if gai == 0 or gai == 1:
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
                lai = np.sum(cell > 0) / cell_total
                lads.append(abs(lai - gai))
        ldc   = float(np.mean(lads))
        si_ia = ldc / (2 * gai * (1 - gai))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = {
        "total_pixels":     total,
        "anomaly_pixels":   anom,
        "anomaly_area_pct": round(gai * 100, 4),
        "si_ia_score":      round(si_ia * 100, 4),
        "num_regions":      len(contours),
        "bounding_boxes":   [],
        "centroids":        [],
    }
    for cnt in contours:
        features["bounding_boxes"].append(cv2.boundingRect(cnt))
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            features["centroids"].append(
                (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            )
    return features


# ══════════════════════════════════════════════════════════════════════════════
#  EDGE-ARTIFACT REMOVAL  (from Section 5 helpers)
# ══════════════════════════════════════════════════════════════════════════════

def remove_edge_artifacts(mask: np.ndarray, margin_pct: float = 0.04) -> np.ndarray:
    """
    Zero out a border band of `margin_pct` × dimension on all four sides.
    Default 4 % is safe for flat walls.
    """
    H, W    = mask.shape
    mh      = max(1, int(H * margin_pct))
    mw      = max(1, int(W * margin_pct))
    cleaned = mask.copy()
    cleaned[:mh,  :]  = 0
    cleaned[-mh:, :]  = 0
    cleaned[:,  :mw]  = 0
    cleaned[:, -mw:]  = 0
    return cleaned


def apply_circular_mask(mask: np.ndarray, margin: float = 0.03) -> np.ndarray:
    """
    Zero everything outside an inscribed ellipse (shrunk by `margin`).
    For cylindrical / circular specimens.
    """
    H, W    = mask.shape
    cy_c, cx_c = H // 2, W // 2
    ry = max(1, int(H / 2 * (1 - margin)))
    rx = max(1, int(W / 2 * (1 - margin)))
    ellipse = np.zeros_like(mask)
    cv2.ellipse(ellipse, (cx_c, cy_c), (rx, ry), 0, 0, 360, 1, -1)
    return (mask * ellipse).astype(mask.dtype)


# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTER / STAGE ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def assign_defect_cluster(gai_pct: float, si_ia: float, cls_name: str) -> int:
    """
    Map (GAI %, SI_ia, class) → defect cluster [0, 1, 2].

    The notebook assigns clusters via KMeans trained on (confidence, gai, si_ia).
    Empirically, GAI is the dominant axis:
      cluster 0 → low GAI   (< GAI_LOW  = 15 %) → low damage
      cluster 1 → medium GAI (15–35 %)           → medium damage
      cluster 2 → high GAI   (≥ GAI_HIGH = 35 %) → high damage

    SI_ia refines the assignment at the boundary:
      - Very high SI at low GAI still maps to cluster 0 (damage is concentrated, not widespread)
      - Very low SI at medium GAI maps to cluster 1 (scattered, not yet severe)

    Example from screenshot: GAI = 10.37 %, SI = 49.007 → cluster 0 ✓
    """
    if cls_name in ("normal", "crack"):
        return -1  # no cluster for simple-card classes

    if gai_pct < GAI_MIN_THRESHOLD * 100:
        return 0   # unreliable — too small, treat as low

    if gai_pct < GAI_LOW * 100:
        # Low overall coverage — always cluster 0 regardless of SI
        return 0

    if gai_pct < GAI_HIGH * 100:
        # Medium coverage — SI decides between cluster 1 and 2
        return 1 if si_ia < SI_HIGH else 2

    # High coverage → cluster 2 (severe)
    return 2


def get_stage_info(cluster_id: int) -> tuple[int, str, str]:
    """
    Returns (stage_rank, damage_label, stage_label).
    stage_rank is 1-indexed to match the UI.
    """
    if cluster_id == -1:
        return -1, "N/A", "N/A"
    rank         = cluster_id + 1           # 1 / 2 / 3
    damage_label = DAMAGE_LABELS[cluster_id]
    stage_label  = f"STAGE {rank}"
    return rank, damage_label, stage_label


# ══════════════════════════════════════════════════════════════════════════════
#  CROP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def crop_with_padding(
    orig_img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    pad_frac: float = 0.10,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Return a padded crop and the (cx1, cy1, cx2, cy2) coordinates used.
    """
    H, W   = orig_img.shape[:2]
    pad_x  = int(pad_frac * max(x2 - x1, 1))
    pad_y  = int(pad_frac * max(y2 - y1, 1))
    cx1    = max(0, x1 - pad_x)
    cy1    = max(0, y1 - pad_y)
    cx2    = min(W, x2 + pad_x + 1)
    cy2    = min(H, y2 + pad_y + 1)
    return orig_img[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)
