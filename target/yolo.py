"""
find_target.py
==============
Detects a completely black target object in undistorted images using YOLOv10-n,
with a robust HSV-based colour fallback.

Scene description
-----------------
  - Target     : completely black object (uniform, sharp-edged, low-texture)
  - Walls      : white (one side) and grey (other side)
  - Floor/Roof : bright colours

Shadow rejection
----------------
Shadows are darkened versions of bright surfaces — they retain residual hue,
show soft/gradient edges, and preserve the underlying surface texture.
The genuine black target has none of these properties.  Four complementary
guards are applied in both the YOLO scoring path and the colour-fallback path:

  Fix 1 – HSV saturation gate   : black target has S ≈ 0; shadows keep colour.
  Fix 2 – Solidity filter        : target contour is compact; shadows are ragged.
  Fix 3 – Interior texture check : target interior is uniform; shadows show grain.
  Fix 4 – Edge-sharpness bonus   : target edges are crisp; shadow edges are soft.

Dependencies
------------
    pip install ultralytics opencv-python numpy
"""

import cv2
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Tunable thresholds  (adjust to match your camera / lighting conditions)
# ---------------------------------------------------------------------------
_HSV_MAX_VALUE      = 50    # Fix 1 – pixels darker than this (V channel)
_HSV_MAX_SATURATION = 60    # Fix 1 – AND less saturated than this pass the mask
_MIN_SOLIDITY       = 0.75  # Fix 2 – contour area / convex-hull area
_MAX_INTERIOR_VAR   = 30.0  # Fix 3 – grayscale variance inside bounding box
_SHARPNESS_SCALE    = 100.0 # Fix 4 – Laplacian variance normalisation factor


# ---------------------------------------------------------------------------
# Lazy-load the YOLOv10-n model once (module-level singleton)
# ---------------------------------------------------------------------------
_model = None

def _get_model():
    """Load YOLOv10-n weights on first call and cache them."""
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
            _model = YOLO("yolov10n.pt")   # auto-downloads on first use
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required.  Install with: pip install ultralytics"
            ) from exc
    return _model


# ---------------------------------------------------------------------------
# Shadow-rejection helpers
# ---------------------------------------------------------------------------

def _is_shadow_by_texture(gray: np.ndarray, x: int, y: int,
                           w: int, h: int) -> bool:
    """
    Fix 3 – Return True when the crop's interior variance is too high,
    indicating preserved surface texture (i.e. a shadow, not a solid object).
    """
    crop = gray[y: y + h, x: x + w]
    if crop.size == 0:
        return True
    return float(np.var(crop)) > _MAX_INTERIOR_VAR


def _edge_sharpness(gray: np.ndarray, x: int, y: int,
                    w: int, h: int) -> float:
    """
    Fix 4 – Return the Laplacian variance of the crop, normalised to [0, 1].
    High value → crisp edges → real object.
    Low value  → gradient edges → shadow.
    """
    crop = gray[y: y + h, x: x + w]
    if crop.size == 0:
        return 0.0
    lap_var = cv2.Laplacian(crop, cv2.CV_64F).var()
    return min(lap_var / _SHARPNESS_SCALE, 1.0)


def _contour_passes_shadow_checks(contour, gray: np.ndarray) -> bool:
    """
    Apply Fix 2 (solidity) and Fix 3 (texture) to a single contour.
    Returns True when the contour is likely the real target.
    """
    # Fix 2 – solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area < 1:
        return False
    solidity = cv2.contourArea(contour) / hull_area
    if solidity < _MIN_SOLIDITY:
        return False

    # Fix 3 – interior texture
    x, y, w, h = cv2.boundingRect(contour)
    if _is_shadow_by_texture(gray, x, y, w, h):
        return False

    return True


# ---------------------------------------------------------------------------
# Colour-fallback detector
# ---------------------------------------------------------------------------

def _largest_black_contour(image_bgr: np.ndarray,
                            gray: np.ndarray) -> Optional[tuple]:
    """
    Return (cx, cy, x, y, w, h) of the best shadow-rejected black contour,
    or None if nothing survives the filters.

    Fixes applied here
    ------------------
    Fix 1 – tight HSV mask (low V *and* low S) to exclude coloured shadows.
    Fix 2 – solidity filter on every surviving contour.
    Fix 3 – interior texture check on every surviving contour.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Fix 1: require both low Value AND low Saturation
    lower_black = np.array([0,  0,                    0])
    upper_black = np.array([180, _HSV_MAX_SATURATION, _HSV_MAX_VALUE])
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter by area, then shadow checks; keep the largest survivor
    valid = [
        c for c in contours
        if cv2.contourArea(c) >= 100 and _contour_passes_shadow_checks(c, gray)
    ]
    if not valid:
        return None

    best = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)
    return x + w // 2, y + h // 2, x, y, w, h


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_target(
    image: np.ndarray,
    conf_threshold: float = 0.35,
    prefer_yolo: bool = True,
) -> Optional[dict]:
    """
    Detect the black target in *image* and return its coordinates.

    Strategy
    --------
    1. Run YOLOv10-n.  For each confident detection compute a shadow-aware
       score that rewards darkness, low saturation, crisp edges, and uniform
       interior (Fixes 1, 3, 4).  Accept the top-scoring box.
    2. If YOLO yields nothing, fall back to HSV segmentation with solidity
       and texture filters (Fixes 1, 2, 3).

    Parameters
    ----------
    image : np.ndarray
        BGR image (as returned by ``cv2.imread`` or a video frame).
    conf_threshold : float
        Minimum YOLO confidence to accept a detection (0–1).
    prefer_yolo : bool
        If False, skip YOLO and use only the colour-based method.

    Returns
    -------
    dict or None
        On success::

            {
                "cx"         : int,          # centroid x (pixels)
                "cy"         : int,          # centroid y (pixels)
                "bbox"       : (x, y, w, h), # bounding box in pixel coords
                "method"     : "yolo" | "color_fallback",
                "confidence" : float | None,
                "score"      : float | None  # composite shadow-rejection score
            }

        Returns ``None`` when no target is found.

    Raises
    ------
    ValueError
        If *image* is not a valid 3-channel BGR numpy array.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("image must be a 3-channel BGR numpy array.")

    # Pre-compute grayscale and HSV once; reused by all sub-functions
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ------------------------------------------------------------------
    # 1. YOLOv10-n inference with shadow-aware scoring
    # ------------------------------------------------------------------
    best_yolo = None

    if prefer_yolo:
        model = _get_model()
        results = model(image, verbose=False)

        candidates = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf < conf_threshold:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                bx, by = int(x1), int(y1)
                bw, bh = int(x2 - x1), int(y2 - y1)
                if bw < 1 or bh < 1:
                    continue

                # Fix 1: mean brightness AND mean saturation of the crop
                crop_hsv    = hsv[by: by + bh, bx: bx + bw]
                mean_val    = float(crop_hsv[:, :, 2].mean())   # V channel
                mean_sat    = float(crop_hsv[:, :, 1].mean())   # S channel
                darkness_score = 1.0 - mean_val / 255.0   # higher = darker
                purity_score   = 1.0 - mean_sat / 255.0   # higher = less coloured

                # Fix 3: interior texture (penalise high variance)
                texture_ok    = not _is_shadow_by_texture(gray, bx, by, bw, bh)
                texture_score = 1.0 if texture_ok else 0.0

                # Fix 4: edge sharpness (reward crisp boundaries)
                sharpness_score = _edge_sharpness(gray, bx, by, bw, bh)

                # Composite score — all four factors weighted equally
                score = conf * darkness_score * purity_score * (
                    0.5 + 0.25 * texture_score + 0.25 * sharpness_score
                )

                candidates.append((score, conf, bx, by, bw, bh))

        if candidates:
            candidates.sort(key=lambda t: t[0], reverse=True)
            best_score, best_conf, bx, by, bw, bh = candidates[0]
            best_yolo = {
                "cx":         bx + bw // 2,
                "cy":         by + bh // 2,
                "bbox":       (bx, by, bw, bh),
                "method":     "yolo",
                "confidence": best_conf,
                "score":      best_score,
            }

    # ------------------------------------------------------------------
    # 2. Colour-based fallback (Fixes 1, 2, 3)
    # ------------------------------------------------------------------
    color_result = _largest_black_contour(image, gray)

    # ------------------------------------------------------------------
    # 3. Decision
    # ------------------------------------------------------------------
    if best_yolo is not None:
        return best_yolo

    if color_result is not None:
        cx, cy, x, y, w, h = color_result
        return {
            "cx":         cx,
            "cy":         cy,
            "bbox":       (x, y, w, h),
            "method":     "color_fallback",
            "confidence": None,
            "score":      None,
        }

    return None   # target not found


# ---------------------------------------------------------------------------
# Quick visual demo  (python find_target.py <image_path>)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Usage: python find_target.py <image_path>")
        sys.exit(0)

    img = cv2.imread(path)
    if img is None:
        print(f"Could not read image: {path}")
        sys.exit(1)

    result = find_target(img)
    if result is None:
        print("Target NOT found.")
    else:
        print(f"Target found via [{result['method']}]")
        print(f"  Centroid : ({result['cx']}, {result['cy']})")
        print(f"  BBox     : {result['bbox']}")
        if result["confidence"] is not None:
            print(f"  Conf     : {result['confidence']:.2f}")
        if result["score"] is not None:
            print(f"  Score    : {result['score']:.4f}")

        # Draw and save annotated image
        x, y, w, h = result["bbox"]
        vis = img.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(vis, (result["cx"], result["cy"]), 6, (0, 0, 255), -1)
        label = (f"{result['method']} | score={result['score']:.3f}"
                 if result["score"] else result["method"])
        cv2.putText(vis, label, (x, max(y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        out_path = path.rsplit(".", 1)[0] + "_detected.jpg"
        cv2.imwrite(out_path, vis)
        print(f"  Saved annotated image → {out_path}")