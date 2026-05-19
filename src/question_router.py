"""Two-layer question router.

Layer 1: regex keyword match → fast and deterministic.
Layer 2: CLIP text-encoder zero-shot against canonical templates → catches
         rephrasings, typos, and unusual phrasings.

Returns one of: 'shot', 'role', 'handedness', 'foot', 'intent', 'freeform'.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import CLIPModel, CLIPProcessor

from .config import CLIP_MODEL

QUESTION_TYPES = ["shot", "role", "handedness", "foot", "intent", "freeform"]

REGEX_PATTERNS = {
    "shot": [
        r"\b(what|which)\s+(kind|type)\s+of\s+(shot|stroke)\b",
        r"\bwhat\s+shot\b",
        r"\bwhich\s+shot\b",
        r"\bstroke\s+is\s+(being\s+)?played\b",
        r"\b(name|identify)\s+the\s+shot\b",
    ],
    "role": [
        r"\b(batsman|batter|bowler|fielder|keeper|wicket-?keeper)\b",
        r"\b(role|position)\s+of\s+the\s+player\b",
        r"\bwho\s+is\s+this\s+player\b",
    ],
    "handedness": [
        r"\b(right|left)[-\s]?hand(ed)?\b",
        r"\bhandedness\b",
        r"\bdominant\s+hand\b",
    ],
    "foot": [
        r"\b(front|back)[-\s]?foot\b",
        r"\bfoot\s+position\b",
        r"\b(on\s+the\s+)?front\s+foot\b|\bback\s+foot\b",
    ],
    "intent": [
        r"\b(attacking|aggressive|defensive|defending|defence)\b",
        r"\bintent\s+of\s+the\s+shot\b",
        r"\bis\s+(this|it)\s+a\s+(defensive|attacking)\b",
    ],
}

CANONICAL_TEMPLATES = {
    "shot": "a question about the type of cricket shot being played",
    "role": "a question about whether the player is a batsman or a bowler",
    "handedness": "a question about whether the batsman is right-handed or left-handed",
    "foot": "a question about whether the batsman is on the front foot or back foot",
    "intent": "a question about whether the shot is attacking or defensive",
    "freeform": "an open-ended question describing the scene or asking for details",
}


@dataclass
class RouterResult:
    qtype: str            # one of QUESTION_TYPES
    confidence: float     # 0-1, how sure the router is
    via: str              # 'regex' or 'clip'


class QuestionRouter:
    def __init__(self, device: str = "cpu", clip_threshold: float = 0.25):
        self.device = device
        self.clip_threshold = clip_threshold
        self._clip = None
        self._processor = None
        self._template_features = None

    def _lazy_load_clip(self):
        if self._clip is None:
            self._clip = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device).eval()
            self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
            with torch.no_grad():
                inputs = self._processor(
                    text=list(CANONICAL_TEMPLATES.values()),
                    return_tensors="pt", padding=True,
                ).to(self.device)
                feats = self._clip.get_text_features(**inputs)
                self._template_features = feats / feats.norm(dim=-1, keepdim=True)

    def _try_regex(self, q: str) -> Optional[str]:
        q_low = q.lower().strip()
        for qtype, patterns in REGEX_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, q_low):
                    return qtype
        return None

    def _clip_classify(self, q: str) -> RouterResult:
        self._lazy_load_clip()
        with torch.no_grad():
            inputs = self._processor(text=[q], return_tensors="pt", padding=True).to(self.device)
            qf = self._clip.get_text_features(**inputs)
            qf = qf / qf.norm(dim=-1, keepdim=True)
            sims = (qf @ self._template_features.T).softmax(dim=-1)[0]
        idx = int(sims.argmax().item())
        conf = float(sims[idx].item())
        qtype = list(CANONICAL_TEMPLATES.keys())[idx]
        # If even the best template is weak, treat as freeform
        if conf < self.clip_threshold:
            qtype = "freeform"
        return RouterResult(qtype=qtype, confidence=conf, via="clip")

    def route(self, question: str) -> RouterResult:
        if not question or not question.strip():
            return RouterResult(qtype="freeform", confidence=0.0, via="regex")
        # Layer 1: regex
        hit = self._try_regex(question)
        if hit is not None:
            return RouterResult(qtype=hit, confidence=1.0, via="regex")
        # Layer 2: CLIP zero-shot
        return self._clip_classify(question)


if __name__ == "__main__":
    r = QuestionRouter()
    for q in [
        "What shot is being played?",
        "Is the batsman right-handed?",
        "Is this player a batsman or a bowler?",
        "front foot or back foot?",
        "is this attacking or defensive?",
        "describe the scene",
        "shott type??",  # typo case
    ]:
        print(f"{q!r:55s} -> {r.route(q)}")
