"""Pre-flight checks: is-this-cricket sanity + camera-view classifier."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from .inference.clip_qa import CLIPZeroShot


CAMERA_VIEW_PROMPTS = {
    "bowler-end": [
        "a cricket photo taken from behind the bowler looking at the batsman",
    ],
    "keeper-end": [
        "a cricket photo taken from behind the wicketkeeper looking at the batsman",
    ],
    "side": [
        "a side-on view of a cricket batsman, square-leg or point camera angle",
    ],
}


@dataclass
class PreflightResult:
    is_cricket: bool
    cricket_confidence: float
    camera_view: str         # 'bowler-end' | 'keeper-end' | 'side'
    camera_flip: bool        # True if handedness rule should flip


def run_preflight(image: Image.Image | np.ndarray, clip: CLIPZeroShot) -> PreflightResult:
    cricket = clip.is_cricket(image)
    is_cricket = cricket.label == "cricket" and cricket.confidence > 0.55
    view = clip.classify(image, CAMERA_VIEW_PROMPTS)
    return PreflightResult(
        is_cricket=is_cricket,
        cricket_confidence=cricket.confidence,
        camera_view=view.label,
        camera_flip=(view.label == "keeper-end"),
    )
