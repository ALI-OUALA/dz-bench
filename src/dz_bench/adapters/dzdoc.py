"""Public JSON adapter boundary for DzDoc predictions.

This module deliberately imports no DzDoc package. A producer may be DzDoc,
another OCR engine, or a hand-authored fixture as long as it emits the public
prediction schema.
"""

from __future__ import annotations

from pathlib import Path

from ..io import load_predictions
from ..models import Predictions, PredictionSample


class PublicPredictionAdapter:
    """Read versioned prediction JSON without importing private engine code."""

    name = "public-prediction-json"
    version = "1.0.0"

    def load(self, path: str | Path) -> Predictions:
        return load_predictions(path)

    def samples(self, path: str | Path) -> list[PredictionSample]:
        return self.load(path).samples


def load_dzdoc_predictions(path: str | Path) -> Predictions:
    """Compatibility-named loader for a DzDoc public prediction artifact."""

    return PublicPredictionAdapter().load(path)
