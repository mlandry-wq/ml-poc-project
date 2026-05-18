# Rapport de Preprocessing et Feature Engineering

---

## 1. Nettoyage des données

### Gestion des Valeurs Sentinelles CDC
Le CDC n'utilise pas de `NaN` natifs pour les données manquantes. Il encode les non-réponses avec des **codes numériques sentinelles** (`9`, `99`, `99.9`) qui, sans traitement, seraient interprétés comme de vraies valeurs par le modèle (ex : 99 cigarettes/jour).

Ces valeurs ont été systématiquement remplacées par des `NaN` avant toute transformation :

| Variable | Valeur sentinelle | Signification |
|---|---|---|
| `IMC_Mere` | 99.9 | Inconnu |
| `Prise_Poids` | 99 | Inconnu |
| `Tabac_Avant`, `Tabac_Trim1`, `Tabac_Trim3` | 99 | Inconnu |
| `Mois_Debut_Suivi` | 99 | Inconnu |
| `Assurance` | 9 | Inconnu |
| `ATCD_Mort_Foetale` | 99 | Inconnu |

### Filtrage et Encodage de la Cible
La variable cible `ab_nicu` a été filtrée pour ne conserver que les entrées `Y` et `N`, puis encodée en binaire : **1** (Admis NICU) et **0** (Pas d'admission).

### Imputation des valeurs manquantes
- **Variables numériques** : imputation par la **médiane** — robuste aux distributions asymétriques et aux valeurs extrêmes (ex : mères > 45 ans, IMC extrêmes).
- **Variables catégorielles** : imputation par le **mode** (valeur la plus fréquente) pour préserver la distribution originale.

*Alternative rejetée — imputation par la moyenne* : trop sensible aux outliers médicaux (IMC > 50, âge > 45). La médiane donne une meilleure représentation du profil-type de la population.

---

## 2. Suppression de `Nb_Visites_Prenatales`

Cette variable a été **retirée du dataset** malgré une corrélation apparente avec le risque NICU. Deux raisons justifient cette décision :

**Raison 1 — Data Leakage temporel (argument principal)**
Le nombre total de visites prénatales se construit progressivement tout au long de la grossesse. En début de suivi, cette valeur n'est pas encore connue. Utiliser la valeur finale dans un modèle de prévention revient à regarder dans le futur — ce qui invalide toute utilisation en conditions réelles.

**Raison 2 — Confusion causalité / corrélation**
Une mère dont le bébé sera admis en NICU peut accumuler plus de visites *parce que* des complications ont été détectées en cours de suivi. La variable capture l'effet (complication déjà en cours), pas la cause.

> La forte importance attribuée par le Random Forest (~0.40 MDI) confirme que la variable est corrélée au risque — mais cette corrélation est post-prédictive, pas pré-prédictive. Un modèle de prévention prénatale ne peut pas l'utiliser.

---

## 3. Feature Engineering

### Variables dérivées créées (3)

**`Grossesse_Multiple`** : binarisation de la variable CDC `dplural` (pluralité de la grossesse). `1` si `dplural > 1` (jumeaux, triplets…), `0` sinon. C'est la variable avec le **signal le plus discriminant** du dataset : jumeaux → 37.5% NICU, triplets → 87.5% (×4 à ×10 vs base 8.9%). La valeur numérique brute (`dplural` = 1, 2, 3…) est remplacée par ce binaire car le modèle n'a pas besoin de distinguer jumeaux de triplets — le signal est capturé dès que la valeur est `> 1`.

**`Evol_Tabac`** : différence entre la consommation de tabac au premier trimestre et avant la grossesse (`Tabac_Trim1 - Tabac_Avant`). Une valeur négative indique une réduction (comportement positif), une valeur positive une augmentation (signal d'alerte). Cette variable capture un **changement de comportement** que les deux variables séparées ne capturent pas individuellement. Fixée à 0 dans la démo (93.7% des cas d'entraînement ont `Evol_Tabac = 0`).

**`Suivi_T1`** : variable binaire (0/1) indiquant si le suivi prénatal a débuté durant le premier trimestre (mois 1 à 3). Elle sert de proxy pour mesurer la **précocité de l'accès aux soins**, facteur de protection reconnu dans la littérature médicale.

**`ATCD_Mort_Foetale_bin`** : binarisation de la variable CDC `priordead` (nombre d'antécédents de mort fœtale, valeur numérique 0, 1, 2…). `1` si `priordead > 0` (au moins un antécédent), `0` sinon. La valeur brute (compte) est remplacée par un binaire car la distinction entre 1 et 2+ antécédents n'apporte pas de signal supplémentaire significatif au modèle.

### Nouvelles variables médicales à fort signal

L'analyse EDA a révélé que les variables de base (âge, IMC, tabac) ont des corrélations très faibles (< 0.05) avec la target. L'enrichissement avec des variables médicales disponibles dans le dataset CDC a permis d'intégrer des signaux beaucoup plus discriminants :

| Variable | Source CDC | Taux NICU | Signal vs base (8.9%) |
|---|---|---|---|
| `Grossesse_Multiple` | `dplural > 1` | jumeaux 37.5%, triplets 87.5% | **×4 à ×10** |
| `Eclampsie` | `rf_ehype` | 27.1% | ×3.0 |
| `PMA` | `rf_artec` | 23.9% | ×2.7 |
| `Traitement_Infertilite` | `rf_inftr` | 21.3% | ×2.4 |
| `ATCD_Premature` | `rf_ppterm` | 19.2% | ×2.2 |
| `HTA_Gestationnelle` | `rf_ghype` | 16.8% | ×1.9 |
| `ATCD_Mort_Foetale_bin` | `priordead > 0` | jusqu'à 16.1% | ×1.8 |
| `Diabete_Gestationnel` | `rf_gdiab` | 12.3% | ×1.4 |
| `Tabac_Trim3` | `cig_3` | corr +0.142 | signal > T1 (+0.016) |

> **Note sur le timing** : les conditions gestationnelles (éclampsie, HTA, diabète gestationnel) se développent pendant la grossesse. Ce modèle est applicable à partir du 2e/3e trimestre ou au moment de l'accouchement — pas en tout début de grossesse.

---

## 4. Transformations appliquées

Un pipeline de transformation automatisé (`ColumnTransformer`) a été mis en place via scikit-learn pour garantir la reproductibilité et éviter tout data leakage.

### Traitement des valeurs aberrantes — `Prise_Poids`
Les valeurs hors de la plage médicalement plausible `[0–80 lbs]` (recommandations IOM) ont été remplacées par `NaN` avant le pipeline. Ces valeurs ne sont pas des sentinelles CDC mais de vraies aberrations cliniques — elles sont ensuite prises en charge par l'imputation médiane.

### Robust Scaling
Toutes les variables numériques continues ont été scalées via `RobustScaler` (centré sur la médiane, échelle sur l'IQR). Ce choix est plus adapté que `StandardScaler` pour ce dataset car plusieurs variables présentent des distributions fortement asymétriques (tabac, prise de poids) et des outliers médicaux légitimes (IMC > 50, âge > 45). `RobustScaler` limite leur influence sur la normalisation sans les supprimer.

### One-Hot Encoding
Appliqué à toutes les variables catégorielles nominales avec `drop='first'` pour éviter la **multicolinéarité** (dummy trap) :

- `Assurance` (1=Medicaid, 2=Privée, 3=CHIP…) — nominale, pas d'ordre
- `Aide_WIC`, `H_Chronique`, `D_Chronique` — binaires Y/N/U
- `ATCD_Premature`, `PMA`, `Traitement_Infertilite` — Y/N/U ou Y/N/X/U
- `Eclampsie`, `HTA_Gestationnelle`, `Diabete_Gestationnel` — Y/N/U

*Alternative rejetée — OrdinalEncoder* : inadapté car les modalités sont nominales.
*Alternative rejetée — Target Encoding* : risque élevé de data leakage sur dataset déséquilibré (~9% positifs).

### PCA (Analyse en Composantes Principales)
Deux niveaux de réduction dimensionnelle ont été mis en place :

- **Preprocessing** : PCA conservant 95% de la variance (4 composantes sur 36 features) → `processed_data_pca.csv`, pour archivage et exploration.
- **Modeling** : PCA(10 composantes, ~99% variance) appliquée **dans le pipeline de modélisation**, après SMOTE. Le PCA est fitté sur `X_train` original (avant SMOTE) pour ne pas laisser les exemples synthétiques influencer l'espace latent.

**Justification de PCA(10) pour la modélisation** : le dataset contient 26 colonnes OHE issues de 10 variables catégorielles, créant une redondance importante. La PCA compresse cette information en 10 composantes denses, éliminant les colinéarités et améliorant la séparation des classes minoritaires. Impact mesuré : +10 pts de recall pour le Random Forest (0.662 → 0.766).

---

## 5. Justification des choix et alternatives

| Choix retenu | Alternative rejetée | Raison |
|---|---|---|
| Imputation médiane | Imputation moyenne | Robustesse aux outliers médicaux |
| `RobustScaler` | `StandardScaler` | Variables asymétriques (tabac, poids) |
| Cap `Prise_Poids` à `[0–80 lbs]` | Conservation | Aberrations cliniques, pas des sentinelles |
| `Assurance` → OHE | Numérique | Nominale (1=Medicaid, 2=Privée…) sans ordre |
| Nouvelles features médicales | Variables de base seules | Corrélations < 0.05 pour vars de base ; médicales jusqu'à ×10 |
| OneHotEncoder | OrdinalEncoder | Pas d'ordre naturel dans les catégories |
| OneHotEncoder | Target Encoding | Risque de data leakage sur classe minoritaire |
| Suppression `Nb_Visites` | Conservation | Data leakage temporel + corrélation post-prédictive |
| `drop='first'` dans OHE | Pas de drop | Évite la multicolinéarité (dummy trap) |
| SMOTE dans pipeline CV | SMOTE avant CV | Évite le data leakage dans la validation croisée Optuna |
| PCA(10) après SMOTE | Pas de PCA | +10 pts recall RF (0.662→0.766), élimine redondance OHE |
| PCA fitté sur X_train original | PCA fitté sur X_train_res (post-SMOTE) | Évite que les exemples synthétiques influencent l'espace latent |

---

## 6. Impact mesuré sur les modèles

| Modèle | Recall | F-beta β=2 | Seuil |
|---|---|---|---|
| Logistic Regression (baseline) | 0.354 | 0.259 | 0.10 |
| **Random Forest + PCA(10)** ✅ | **0.766** | **0.350** | 0.16 |
| HistGBM + Optuna + PCA(10) | 0.698 | 0.337 | 0.46 |

- **Signal fortement amélioré** : les nouvelles features médicales multiplient le taux NICU observé par 2 à 10x.
- **RF retenu** (meilleur F-beta β=2 = 0.350) : détecte 76.6% des bébés à risque. HGB préférable en pratique (seuil plus conservateur : 0.46 vs 0.16).
- **Meilleure convergence** grâce au `RobustScaler` (gradients stables, moins perturbés par les outliers médicaux).
- **CV plus fiable** : SMOTE appliqué à l'intérieur du pipeline Optuna élimine le data leakage qui gonflait artificiellement le recall CV de 0.86 à 0.33.

---

## 7. Fichiers Python modifiés

### `src/data.py`

Point d'entrée utilisé par `scripts/main.py` et `src/app.py` pour charger les données et les préparer pour l'évaluation des modèles.

**Modifications apportées :**

| Problème initial | Correction |
|---|---|
| Chemin hardcodé `/Users/madeleine/...` | Remplacé par `Path(__file__).resolve().parent.parent / "data/process/processed_data_full.csv"` — portable |
| `sys.path.insert` fragile | Supprimé |
| Retournait 36 features brutes | Ajout de `PCA(10)` fitté sur `X_train` — les modèles attendent 10 composantes |

**Fonction principale :**

```python
from src.data import load_dataset_split
X_train, X_test, y_train, y_test = load_dataset_split()
# X_train.shape = (79 920, 10)  — PCA(10) appliqué
# X_test.shape  = (19 980, 10)
```

Le PCA est refitté à chaque appel sur `X_train` original (avant SMOTE), ce qui reproduit exactement le protocole des notebooks. Le split est stratifié (`random_state=42`, `test_size=0.2`).

---

### `src/metrics.py`

Calcul des métriques d'évaluation appliquées à chaque modèle après prédiction sur `X_test`.

**Modification apportée :** ajout du **F-beta β=2** comme métrique prioritaire.

```python
from sklearn.metrics import recall_score, f1_score, fbeta_score, accuracy_score, precision_score

def compute_metrics(y_true, y_pred):
    return {
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "fbeta_2":   float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
    }
```

**Justification du F-beta β=2 :** dans un contexte médical, un faux négatif (bébé à risque non détecté) est cliniquement deux fois plus coûteux qu'un faux positif (fausse alerte). `β=2` pondère le recall deux fois plus que la précision, reflétant cette asymétrie.

---

## 8. Données et Notebooks

### Datasets générés

| Fichier | Description | Dimensions |
|---|---|---|
| `data/process/processed_data_full.csv` | Dataset preprocessé complet (après sentinelles, imputation, OHE, feature engineering) | 99 900 × 37 (36 features + target) |
| `data/process/processed_data_pca.csv` | Version réduite PCA(4 composantes, 95% variance) — exploration uniquement | 99 900 × 5 |
| `models/preprocessor.pkl` | Pipeline scikit-learn sérialisé (`ColumnTransformer`) | — |

### Comment obtenir les datasets

Les datasets preprocessés sont générés par `notebooks/Preprocessing.ipynb`. Pour les reproduire :

```bash
# 1. Ouvrir et exécuter toutes les cellules du notebook
jupyter notebook notebooks/Preprocessing.ipynb

# Les fichiers sont sauvegardés automatiquement dans data/process/
```

Le notebook applique dans l'ordre : neutralisation des sentinelles → feature engineering → cap Prise_Poids → imputation → RobustScaler → OHE → sauvegarde CSV.

### Comment charger et utiliser

```python
# Option 1 — via data.py (recommandé, avec PCA(10))
from src.data import load_dataset_split
X_train, X_test, y_train, y_test = load_dataset_split()

# Option 2 — chargement direct du CSV (36 features brutes)
import pandas as pd
df = pd.read_csv("data/process/processed_data_full.csv")
X = df.drop(columns=["Target_Risk"])
y = df["Target_Risk"]

# Option 3 — via le preprocessor sauvegardé (pour nouvelles données)
import joblib
preprocessor = joblib.load("models/preprocessor.pkl")
X_new_transformed = preprocessor.transform(X_new_raw)
```

**Note :** `load_dataset_split()` applique un `PCA(10)` refitté sur `X_train` à chaque appel — c'est le format attendu par les 3 modèles entraînés. Utiliser directement le CSV (Option 2) retourne 36 features et nécessite d'appliquer manuellement la PCA.
