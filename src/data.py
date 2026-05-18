from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

PATH = Path(__file__).resolve().parent.parent / "data" / "process" / "processed_data_full.csv"


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return the dataset split used for model evaluation.

    Returns (X_train, X_test, y_train, y_test) where X arrays are PCA(10)-transformed,
    matching the input format expected by all 3 trained models.
    """

    try:
        df = pd.read_csv(PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Le fichier {PATH} est introuvable.\n"
            "→ Lancez d'abord le notebook Preprocessing.ipynb pour générer les datasets."
        )

    X = df.drop(columns=["Target_Risk"])
    y = df["Target_Risk"]

    # Même split que dans les notebooks (stratifié, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # PCA(10) fitté sur X_train original (même protocole que les notebooks)
    pca = PCA(n_components=10, random_state=42)
    X_train_pca = pca.fit_transform(X_train.values)
    X_test_pca = pca.transform(X_test.values)

    return X_train_pca, X_test_pca, y_train, y_test
