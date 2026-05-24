"""
Roboflow concrete-segmentation wrapper.

Provides `get_concrete_rois(orig_img, rf_model, conf_threshold)` which:
  1. Runs the Roboflow model on `orig_img` (RGB numpy array)
  2. Returns a list of ROI dicts:
       { 'crop': np.ndarray (RGB), 'bbox': (x1,y1,x2,y2), 'det_conf': float }

If Roboflow returns no detections the full image is used as the single ROI.
"""

import cv2
import numpy as np
from typing import Optional


def get_concrete_rois(
    orig_img: np.ndarray,
    rf_model,                       # Roboflow model handle (inference)
    conf_threshold: float = 0.50,
    pad_frac: float = 0.10,
) -> list[dict]:
    """
    Run Roboflow segmentation and crop each detected concrete area.

    Parameters
    ----------
    orig_img       : RGB numpy array (H, W, 3)
    rf_model       : loaded Roboflow model  (get_model(...) from inference SDK)
    conf_threshold : skip detections below this confidence
    pad_frac       : padding added to each bounding box before cropping

    Returns
    -------
    list of dicts with keys: crop, bbox (cx1,cy1,cx2,cy2), roi_mask, det_conf
    """
    try:
        import supervision as sv

        bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        rf_results = rf_model.infer(bgr)[0]
        detections = sv.Detections.from_inference(rf_results)

        if len(detections) == 0:
            return [_full_image_roi(orig_img)]

        # Confidence filter
        if detections.confidence is not None and conf_threshold > 0:
            keep       = detections.confidence >= conf_threshold
            detections = detections[keep]

        if len(detections) == 0:
            return [_full_image_roi(orig_img)]

        rois = []
        H, W = orig_img.shape[:2]

        for i in range(len(detections)):
            # Build ROI mask
            roi_mask = np.zeros((H, W), dtype=np.uint8)

            if detections.mask is not None and len(detections.mask) > 0:
                m = cv2.resize(
                    detections.mask[i].astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST,
                )
                roi_mask = m
            else:
                bx1, by1, bx2, by2 = detections.xyxy[i].astype(int)
                roi_mask[by1:by2, bx1:bx2] = 1

            ys, xs = np.where(roi_mask > 0)
            if len(xs) == 0:
                continue

            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            crop, (cx1, cy1, cx2, cy2) = _padded_crop(orig_img, x1, y1, x2, y2, pad_frac)

            rois.append({
                "crop":     crop,
                "bbox":     (cx1, cy1, cx2, cy2),
                "roi_mask": roi_mask,
                "det_conf": float(detections.confidence[i])
                            if detections.confidence is not None else 1.0,
            })

        return rois if rois else [_full_image_roi(orig_img)]

    except Exception as e:
        # Roboflow unavailable or error → fall back to full image
        print(f"⚠️  Roboflow error: {e}. Using full image as ROI.")
        return [_full_image_roi(orig_img)]


# ─── helpers ──────────────────────────────────────────────────────────────────

def _padded_crop(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    pad_frac: float,
) -> tuple[np.ndarray, tuple]:
    H, W   = img.shape[:2]
    pad_x  = int(pad_frac * max(x2 - x1, 1))
    pad_y  = int(pad_frac * max(y2 - y1, 1))
    cx1    = max(0, x1 - pad_x)
    cy1    = max(0, y1 - pad_y)
    cx2    = min(W, x2 + pad_x + 1)
    cy2    = min(H, y2 + pad_y + 1)
    return img[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)


def _full_image_roi(orig_img: np.ndarray) -> dict:
    H, W = orig_img.shape[:2]
    return {
        "crop":     orig_img.copy(),
        "bbox":     (0, 0, W, H),
        "roi_mask": np.ones((H, W), dtype=np.uint8),
        "det_conf": 1.0,
    }
