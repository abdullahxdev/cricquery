"""End-to-end CricQuery pipeline. Used by both app.py and evaluate.py."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Any, Dict

import numpy as np
import torch
from PIL import Image

from .config import VIT_WEIGHTS
from .question_router import QuestionRouter
from .preflight import run_preflight
from .visualize import preprocess_pil, render_answer_panel
from .inference.clip_qa import CLIPZeroShot
from .inference.shot_classifier import ShotClassifier
from .inference.yolo_handedness import predict_handedness
from .inference.intent_rule import infer_intent
from .inference.blip2_fallback import BLIP2Fallback


# Toggle preflight (is-cricket + camera-view CLIP calls) — adds ~5s/question on CPU.
# Off by default. Re-enable with CRICQUERY_PREFLIGHT=1.
ENABLE_PREFLIGHT = os.environ.get("CRICQUERY_PREFLIGHT", "0") == "1"


def _auto_device(override: Optional[str] = None) -> str:
    """Pick the best available device. MPS on Apple Silicon, CUDA on NVIDIA, else CPU."""
    if override:
        return override
    env = os.environ.get("CRICQUERY_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class VQAResult:
    answer: str
    confidence: float
    qtype: str
    via: str
    extras: Dict[str, Any] = field(default_factory=dict)


class CricQueryPipeline:
    """Loads all models once, routes each (image, question) to the right path."""

    def __init__(self, device: Optional[str] = None, enable_blip2: bool = True):
        self.device = _auto_device(device)
        print(f"[CricQuery] Loading models on {self.device}... (preflight={'on' if ENABLE_PREFLIGHT else 'off'})")
        # CLIP and ViT work fine on MPS; YOLO/MediaPipe stay on CPU internally.
        self.clip = CLIPZeroShot(device=self.device)
        self.router = QuestionRouter(device=self.device)
        self.shot = self._maybe_load_shot()
        # BLIP-2 doesn't work well on MPS — force CPU for it
        self.blip2 = BLIP2Fallback(device="cpu" if self.device == "mps" else self.device) if enable_blip2 else None
        print("[CricQuery] Ready.")

    def _maybe_load_shot(self) -> Optional[ShotClassifier]:
        if not VIT_WEIGHTS.exists():
            print(f"[CricQuery] WARN: {VIT_WEIGHTS} not found — Q1 (shot) and Q5 (intent) will fall back to CLIP/BLIP-2.")
            return None
        return ShotClassifier(device=self.device)

    def _answer_shot(self, img):
        if self.shot is None:
            ans = self.clip.classify(img, {  # crude CLIP fallback
                "drive": ["a cricket drive shot"],
                "pullshot": ["a cricket pull shot"],
                "sweep": ["a cricket sweep shot"],
                "legglance-flick": ["a cricket leg glance or flick"],
                "defensive": ["a defensive cricket block"],
            })
            return VQAResult(answer=ans.label, confidence=ans.confidence, qtype="shot",
                             via="clip-fallback", extras={"scores": ans.scores})
        p = self.shot.predict(img)
        return VQAResult(answer=p.top1_label, confidence=p.top1_conf, qtype="shot",
                         via="vit", extras={"top2": p.top2_label, "scores": p.all_scores})

    def answer(self, image: Image.Image, question: str, enhance: bool = True) -> VQAResult:
        t0 = time.time()

        # Step 1: OpenCV preprocessing
        img = preprocess_pil(image, enhance=enhance)
        t_prep = time.time()

        # Step 2: preflight (optional — off by default for speed)
        camera_view = "side"
        camera_flip = False
        if ENABLE_PREFLIGHT:
            pre = run_preflight(img, self.clip)
            if not pre.is_cricket:
                return VQAResult(
                    answer="This doesn't look like a cricket image. Please upload a cricket photo.",
                    confidence=pre.cricket_confidence, qtype="preflight", via="clip",
                    extras={"camera_view": pre.camera_view},
                )
            camera_view = pre.camera_view
            camera_flip = pre.camera_flip
        t_pre = time.time()

        # Step 3: route the question (regex first — usually instant)
        route = self.router.route(question)
        qtype = route.qtype
        t_route = time.time()

        # Step 4: dispatch
        if qtype == "shot":
            res = self._answer_shot(img)
        elif qtype == "role":
            a = self.clip.role(img)
            res = VQAResult(a.label, a.confidence, "role", "clip", {"scores": a.scores})
        elif qtype == "handedness":
            h = predict_handedness(img, camera_flip=camera_flip, clip_fallback=self.clip)
            res = VQAResult(h.label, h.confidence, "handedness", h.via)
        elif qtype == "foot":
            a = self.clip.foot(img)
            res = VQAResult(a.label, a.confidence, "foot", "clip", {"scores": a.scores})
        elif qtype == "intent":
            shot_res = self._answer_shot(img)
            it = infer_intent(shot_res.answer, shot_res.confidence)
            res = VQAResult(it.label, it.confidence, "intent", "rule+vit",
                            extras={"derived_from": it.derived_from})
        else:  # freeform
            if self.blip2 is None:
                res = VQAResult("(BLIP-2 fallback disabled)", 0.0, "freeform", "none")
            else:
                b = self.blip2.answer(img, question)
                res = VQAResult(b.text, 0.85, "freeform", "blip2")
        t_ans = time.time()

        res.extras["camera_view"] = camera_view
        res.extras["router_via"] = route.via
        res.extras["router_confidence"] = route.confidence
        res.extras["timing"] = {
            "preprocess_ms": int((t_prep - t0) * 1000),
            "preflight_ms": int((t_pre - t_prep) * 1000),
            "route_ms": int((t_route - t_pre) * 1000),
            "answer_ms": int((t_ans - t_route) * 1000),
            "total_ms": int((t_ans - t0) * 1000),
        }
        print(f"[CricQuery] '{question[:40]}...' -> {res.qtype}/{res.answer} "
              f"in {res.extras['timing']['total_ms']}ms")
        return res

    def visualize(self, image: Image.Image, result: VQAResult) -> np.ndarray:
        np_img = np.array(image.convert("RGB"))[:, :, ::-1].copy()  # to BGR
        annotated = render_answer_panel(np_img, result.answer, result.qtype, result.confidence)
        return annotated[:, :, ::-1]  # back to RGB
