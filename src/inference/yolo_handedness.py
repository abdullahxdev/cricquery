"""Q3: Right- vs left-handed batsman.

Fallback chain:
  1. YOLOv8 detect person + bat -> compare bat-box center to person-box center.
     If bat is to the LEFT of the person centerline (from camera POV) and
     camera view is from the bowler's end -> LEFT-handed batsman; otherwise
     RIGHT-handed. Camera view flips the rule.
  2. MediaPipe Pose -> compare wrist landmarks; the wrist that is more
     forward (toward the bowler / lower in image y for a side view) is the
     bottom hand on the bat. For a right-hander, the left wrist is the top
     hand and is positioned higher on the bat handle.
  3. CLIP zero-shot ['right-handed batsman', 'left-handed batsman'].

We always record which tier fired in `result.via` for the report's
robustness analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# Lazy imports to keep startup fast
_yolo = None
_pose = None


@dataclass
class HandednessResult:
    label: str           # 'right-handed' or 'left-handed'
    confidence: float
    via: str             # 'yolo', 'pose', or 'clip'


def _ensure_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        from ..config import YOLO_WEIGHTS
        _yolo = YOLO(YOLO_WEIGHTS)
    return _yolo


def _ensure_pose():
    global _pose
    if _pose is None:
        import mediapipe as mp
        _pose = mp.solutions.pose.Pose(static_image_mode=True,
                                       model_complexity=1,
                                       min_detection_confidence=0.5)
    return _pose


def _yolo_bat_geometry(np_image: np.ndarray, camera_flip: bool = False) -> Optional[HandednessResult]:
    """Detect person + bat (COCO baseball bat) and compare centers."""
    yolo = _ensure_yolo()
    res = yolo(np_image, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    cls = res.boxes.cls.cpu().numpy().astype(int)
    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()

    # COCO: 0=person, 39=baseball bat (used as cricket bat proxy)
    person_idx = np.where(cls == 0)[0]
    bat_idx = np.where(cls == 39)[0]
    if len(person_idx) == 0 or len(bat_idx) == 0:
        return None
    # pick the highest-confidence person and bat
    p = xyxy[person_idx[conf[person_idx].argmax()]]
    b = xyxy[bat_idx[conf[bat_idx].argmax()]]
    person_cx = (p[0] + p[2]) / 2.0
    bat_cx = (b[0] + b[2]) / 2.0
    bat_left_of_person = bat_cx < person_cx

    # Default: bowler's-end view; bat on right of body == right-handed
    if not camera_flip:
        label = "left-handed" if bat_left_of_person else "right-handed"
    else:
        label = "right-handed" if bat_left_of_person else "left-handed"

    # Confidence from absolute horizontal offset normalised by person width
    person_w = max(p[2] - p[0], 1.0)
    offset = abs(bat_cx - person_cx) / person_w
    confidence = float(min(0.99, 0.55 + offset))  # 0.55 baseline rising to 0.99
    return HandednessResult(label=label, confidence=confidence, via="yolo")


def _pose_geometry(np_image: np.ndarray, camera_flip: bool = False) -> Optional[HandednessResult]:
    pose = _ensure_pose()
    if np_image.ndim == 2:
        np_image = np.stack([np_image] * 3, axis=-1)
    rgb = np_image[..., :3]
    out = pose.process(rgb)
    if not out.pose_landmarks:
        return None
    lm = out.pose_landmarks.landmark
    # 15=left wrist, 16=right wrist (MediaPipe indexing follows the SUBJECT's
    # body, mirrored relative to the camera). For a typical side-on shot,
    # the wrist with the smaller x (more to the left in the image) is the
    # bottom hand for a right-hander, top hand for a left-hander.
    lw, rw = lm[15], lm[16]
    # Use which wrist is higher in the image (smaller y == higher up).
    # In a right-hander grip, the LEFT hand is the TOP hand (higher up).
    if lw.y < rw.y:
        label = "right-handed"
    else:
        label = "left-handed"
    if camera_flip:
        label = "right-handed" if label == "left-handed" else "left-handed"
    confidence = float(min(0.95, 0.6 + abs(lw.y - rw.y) * 3))
    return HandednessResult(label=label, confidence=confidence, via="pose")


def predict_handedness(image: Image.Image | np.ndarray,
                       camera_flip: bool = False,
                       clip_fallback=None) -> HandednessResult:
    if isinstance(image, Image.Image):
        np_image = np.array(image.convert("RGB"))
    else:
        np_image = image

    # Tier 1: YOLO
    r = _yolo_bat_geometry(np_image, camera_flip=camera_flip)
    if r is not None and r.confidence > 0.65:
        return r
    # Tier 2: Pose
    r2 = _pose_geometry(np_image, camera_flip=camera_flip)
    if r2 is not None and r2.confidence > 0.65:
        return r2
    # Tier 3: CLIP fallback (caller supplies the CLIPZeroShot instance)
    if clip_fallback is not None:
        c = clip_fallback.handedness(image)
        return HandednessResult(label=c.label, confidence=c.confidence, via="clip")
    # Last resort
    return r or r2 or HandednessResult(label="right-handed", confidence=0.5, via="default")
