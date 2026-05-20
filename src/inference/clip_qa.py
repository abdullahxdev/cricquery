"""Q2 (role) and Q4 (foot) — CLIP zero-shot classifier with templated prompts.

Generic enough to also serve as the last-resort fallback for Q3 (handedness).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ..config import CLIP_MODEL


ROLE_PROMPTS = {
    "batsman": [
        "a photo of a cricket batsman holding a bat at the crease",
        "a cricket player batting in a batting stance",
    ],
    "bowler": [
        "a photo of a cricket bowler in delivery stride",
        "a cricket player about to bowl the ball",
    ],
}

FOOT_PROMPTS = {
    "front foot": [
        "a cricket batsman on the front foot leaning forward to play a shot",
        "a batsman with front leg extended toward the ball",
    ],
    "back foot": [
        "a cricket batsman on the back foot leaning back to play a shot",
        "a batsman with weight on the back leg to play a pull or cut",
    ],
}

HAND_PROMPTS = {
    "right-handed": [
        "a right-handed cricket batsman in batting stance, bat held on the right side",
    ],
    "left-handed": [
        "a left-handed cricket batsman in batting stance, bat held on the left side",
    ],
}

CRICKET_PROMPTS = {
    "cricket": ["a photo of a cricket match", "a cricket player on the field"],
    "not-cricket": ["a photo not related to cricket", "an everyday scene without sports"],
}


@dataclass
class CLIPAnswer:
    label: str
    confidence: float
    scores: Dict[str, float]


class CLIPZeroShot:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self._text_cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _encode_text(self, prompts: List[str]) -> torch.Tensor:
        key = "||".join(prompts)
        if key in self._text_cache:
            return self._text_cache[key]
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        f = self.model.get_text_features(**inputs)
        f = f / f.norm(dim=-1, keepdim=True)
        self._text_cache[key] = f
        return f

    @torch.no_grad()
    def classify(self, image: Image.Image | np.ndarray, label_prompts: Dict[str, List[str]]) -> CLIPAnswer:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        img_f = self.model.get_image_features(**inputs)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)

        scores: Dict[str, float] = {}
        for label, prompts in label_prompts.items():
            txt_f = self._encode_text(prompts)
            sim = (img_f @ txt_f.T).mean(dim=-1)  # average over paraphrases
            scores[label] = float(sim.item())

        # softmax over the label means for a confidence value.
        # Using a softer temperature (20) than CLIP's default 100 — that one
        # over-saturates every answer to 99% which is misleading. Temp 20 means
        # a 5% cosine-sim gap maps to ~73% confidence, which matches reality.
        names = list(scores.keys())
        logits = torch.tensor([scores[n] for n in names]) * 20
        probs = logits.softmax(dim=-1)
        idx = int(probs.argmax().item())
        return CLIPAnswer(
            label=names[idx],
            confidence=float(probs[idx].item()),
            scores={n: float(p.item()) for n, p in zip(names, probs)},
        )

    # Convenience wrappers
    def role(self, image) -> CLIPAnswer:
        return self.classify(image, ROLE_PROMPTS)

    def foot(self, image) -> CLIPAnswer:
        return self.classify(image, FOOT_PROMPTS)

    def handedness(self, image) -> CLIPAnswer:
        return self.classify(image, HAND_PROMPTS)

    def is_cricket(self, image) -> CLIPAnswer:
        return self.classify(image, CRICKET_PROMPTS)
