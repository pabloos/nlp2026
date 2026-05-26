from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from app.models.schemas import PredictionItem


class ScoringService:
    """Loads private labels once at startup and evaluates submitted predictions."""

    def __init__(self, labels_path: Path) -> None:
        df = pd.read_csv(labels_path)
        self._true_labels: Dict[int, str] = {
            int(row["id"]): str(row["label"]) for _, row in df.iterrows()
        }

    def calculate_metrics(self, predictions: List[PredictionItem]) -> Dict[str, float]:
        pred_ids = {p.id for p in predictions}
        unknown = pred_ids - set(self._true_labels)
        if unknown:
            raise ValueError(f"Unknown sample IDs: {sorted(unknown)[:10]}")

        if len(predictions) != len(self._true_labels):
            raise ValueError(
                f"Expected {len(self._true_labels)} predictions, got {len(predictions)}. "
                "Submit one prediction for every sample in public_test.csv."
            )

        # Sort predictions by id to ensure consistent ordering
        sorted_preds = sorted(predictions, key=lambda p: p.id)
        y_true = [self._true_labels[p.id] for p in sorted_preds]
        y_pred = [p.label for p in sorted_preds]

        return {
            "f1": round(float(f1_score(y_true, y_pred, pos_label="pos")), 4),
            "precision": round(float(precision_score(y_true, y_pred, pos_label="pos")), 4),
            "recall": round(float(recall_score(y_true, y_pred, pos_label="pos")), 4),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        }
