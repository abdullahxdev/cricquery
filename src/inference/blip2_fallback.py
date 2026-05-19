"""Q6: BLIP-2 free-form fallback for out-of-scope questions.

Lazy-loaded so the main 5 question paths stay fast. Loads in 8-bit on CUDA
when bitsandbytes is available, otherwise falls back to fp16/fp32 on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

from ..config import BLIP2_MODEL


@dataclass
class BLIP2Answer:
    text: str
    via: str = "blip2"


class BLIP2Fallback:
    def __init__(self, device: Optional[str] = None, load_in_8bit: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None
        self._model = None
        self._load_in_8bit = load_in_8bit and self.device == "cuda"

    def _lazy_load(self):
        if self._model is not None:
            return
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        self._processor = Blip2Processor.from_pretrained(BLIP2_MODEL)
        kwargs = {}
        if self._load_in_8bit:
            kwargs["load_in_8bit"] = True
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
        self._model = Blip2ForConditionalGeneration.from_pretrained(BLIP2_MODEL, **kwargs)
        if not self._load_in_8bit:
            self._model = self._model.to(self.device)
        self._model.eval()

    @torch.no_grad()
    def answer(self, image: Image.Image, question: str) -> BLIP2Answer:
        self._lazy_load()
        prompt = f"Question: {question} Answer:"
        inputs = self._processor(image.convert("RGB"), prompt, return_tensors="pt").to(
            self.device, torch.float16 if self.device == "cuda" and not self._load_in_8bit else None
        )
        out = self._model.generate(**inputs, max_new_tokens=40)
        text = self._processor.decode(out[0], skip_special_tokens=True).strip()
        # Strip the echoed prompt if model returned it
        if text.lower().startswith(prompt.lower()):
            text = text[len(prompt):].strip()
        return BLIP2Answer(text=text or "(no answer)")
