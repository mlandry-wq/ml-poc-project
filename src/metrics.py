from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    recall_score,
    f1_score,
    fbeta_score,
    accuracy_score,
    precision_score,
)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return the metrics used to compare model performance.

    Expected return value:
        A dictionary mapping metric names to numeric values, for example:
        ``{"accuracy": 0.91, "f1": 0.88}``.

    Constraints:
    - Every value must be numeric and convertible to ``float``.
    - Use the same metric set for every model so results remain comparable.
    - Keep metric names stable because they are written to
      ``results/model_metrics.csv``.

    Contexte médical :
        Le Recall est la métrique PRIORITAIRE.
        Un Faux Négatif = bébé à risque non détecté → coût médical inacceptable.
        L'Accuracy est trompeuse avec ~9% de positifs (un modèle naïf atteint 91%).
    """

    metrics = {
        # PRIORITAIRE : minimiser les Faux Négatifs (bébés à risque non détectés)
        "recall"   : float(recall_score(y_true, y_pred, zero_division=0)),

        # F-beta β=2 : pondère le Recall 2× plus que la Précision (asymétrie médicale FN/FP)
        "fbeta_2"  : float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),

        # Équilibre Précision / Recall
        "f1_score" : float(f1_score(y_true, y_pred, zero_division=0)),

        # Taux de vrais positifs parmi toutes les prédictions positives
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),

        # Trompeuse sur données déséquilibrées — gardée pour référence uniquement
        "accuracy" : float(accuracy_score(y_true, y_pred)),
    }

    return metrics