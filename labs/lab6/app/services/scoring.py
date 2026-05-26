from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import f1_score

from app.models.schemas import PredictionItem


class ScoringService:
    def __init__(self, labels_path: Path) -> None:
        df = pd.read_csv(labels_path)
        self._true: Dict[int, Tuple[str, str]] = {
            int(row["id"]): (str(row["quality"]), str(row["delivery"]))
            for _, row in df.iterrows()
        }

    def calculate_metrics(self, predictions: List[PredictionItem]) -> Dict[str, float]:
        pred_ids = {p.id for p in predictions}
        unknown = pred_ids - set(self._true)
        if unknown:
            raise ValueError(f"Unknown sample IDs: {sorted(unknown)[:10]}")

        if len(predictions) != len(self._true):
            raise ValueError(
                f"Expected {len(self._true)} predictions, got {len(predictions)}. "
                "Submit one prediction for every sample in public_test.csv."
            )

        sorted_preds = sorted(predictions, key=lambda p: p.id)
        y_true_q = [self._true[p.id][0] for p in sorted_preds]
        y_pred_q = [p.quality for p in sorted_preds]
        y_true_d = [self._true[p.id][1] for p in sorted_preds]
        y_pred_d = [p.delivery for p in sorted_preds]

        f1_q = float(f1_score(y_true_q, y_pred_q, pos_label="pos"))
        f1_d = float(f1_score(y_true_d, y_pred_d, pos_label="pos"))

        return {
            "f1": round((f1_q + f1_d) / 2, 4),
            "f1_quality": round(f1_q, 4),
            "f1_delivery": round(f1_d, 4),
        }
