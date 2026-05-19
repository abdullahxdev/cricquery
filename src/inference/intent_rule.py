"""Q5: Attacking vs defensive intent — rule-based on top of the shot classifier."""
from __future__ import annotations

from dataclasses import dataclass

from ..config import ATTACKING_SHOTS, DEFENSIVE_SHOTS


@dataclass
class IntentResult:
    label: str            # 'attacking' or 'defensive'
    confidence: float
    derived_from: str     # the shot label this was derived from


def infer_intent(shot_label: str, shot_confidence: float) -> IntentResult:
    s = shot_label.lower()
    if s in ATTACKING_SHOTS:
        label = "attacking"
    elif s in DEFENSIVE_SHOTS:
        label = "defensive"
    else:
        # Unknown shot label — bias toward attacking (most cricket shots are)
        label = "attacking"
    return IntentResult(label=label, confidence=shot_confidence, derived_from=shot_label)
