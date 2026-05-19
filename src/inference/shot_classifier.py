"""Q1: Cricket shot type — fine-tuned ViT classifier."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from ..config import VIT_BACKBONE, VIT_WEIGHTS


@dataclass
class ShotPrediction:
    top1_label: str
    top1_conf: float
    top2_label: str
    top2_conf: float
    all_scores: dict   # class -> probability


class ShotClassifier:
    def __init__(self, device: str = "cpu"):
        self.device = device
        ckpt = torch.load(VIT_WEIGHTS, map_location=device)
        self.classes: List[str] = ckpt["classes"]
        self.model = ViTForImageClassification.from_pretrained(
            VIT_BACKBONE, num_labels=len(self.classes),
            ignore_mismatched_sizes=True,
        ).to(device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.processor = ViTImageProcessor.from_pretrained(VIT_BACKBONE)

    @torch.no_grad()
    def predict(self, image: Image.Image | np.ndarray) -> ShotPrediction:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0]
        probs = logits.softmax(dim=-1)
        order = probs.argsort(descending=True).tolist()
        all_scores = {self.classes[i]: float(probs[i].item()) for i in range(len(self.classes))}
        return ShotPrediction(
            top1_label=self.classes[order[0]],
            top1_conf=float(probs[order[0]].item()),
            top2_label=self.classes[order[1]],
            top2_conf=float(probs[order[1]].item()),
            all_scores=all_scores,
        )
