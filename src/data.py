from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return the dataset split used for model evaluation.

    Expected return value:
        A tuple ``(X_train, X_test, y_train, y_test)``.

    Constraints:
    - ``X_train`` and ``X_test`` must contain feature data in a format accepted
      by the trained models stored in ``config.MODELS``.
    - ``y_train`` and ``y_test`` must contain the corresponding targets.
    - ``y_test`` must align with the predictions produced by each loaded model.

    Typical choices for the return types are ``pandas.DataFrame`` /
    ``pandas.Series`` or ``numpy.ndarray``.
    """

    # Dataset complet après imputation + scaling + OHE (sans PCA)
    # Utilisé par les 3 modèles : Logistic Regression, Random Forest, HistGradientBoosting
    PATH = "/Users/madeleine/Desktop/ml-poc-project/data/process/processed_data_full.csv"

    try:
        df = pd.read_csv(PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Le fichier {PATH} est introuvable.\n"
            "→ Lancez d'abord le notebook Preprocessing.ipynb pour générer les datasets."
        )

    # Séparation features / cible
    X = df.drop(columns=["Target_Risk"])
    y = df["Target_Risk"]

    # Split 80/20 avec stratification impérative (déséquilibre ~9% positifs)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
