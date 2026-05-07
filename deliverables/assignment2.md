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
| `Tabac_Avant`, `Tabac_Trim1` | 99 | Inconnu |
| `Mois_Debut_Suivi` | 99 | Inconnu |
| `Assurance` | 9 | Inconnu |

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
Une mère dont le bébé sera admis en NICU peut accumuler plus de visites *parce que* des complications ont été détectées en cours de suivi. La variable capture l'effet (complication déjà en cours), pas la cause. Elle ne permet donc pas de prédire le risque *a priori*, avant que les complications apparaissent.

> La forte importance attribuée par le Random Forest (~0.40 MDI) confirme que la variable est corrélée au risque — mais cette corrélation est post-prédictive, pas pré-prédictive. Un modèle de prévention prénatale ne peut pas l'utiliser.

---

## 3. Feature Engineering

Deux nouvelles variables métier ont été créées pour enrichir le signal disponible :

**`Evol_Tabac`** : différence entre la consommation de tabac au premier trimestre et avant la grossesse (`Tabac_Trim1 - Tabac_Avant`). Une valeur négative indique une réduction (comportement positif), une valeur positive une augmentation (signal d'alerte). Cette variable capture un **changement de comportement** que les deux variables séparées ne capturent pas individuellement.

**`Suivi_T1`** : variable binaire (0/1) indiquant si le suivi prénatal a débuté durant le premier trimestre (mois 1 à 3). Elle sert de proxy pour mesurer la **précocité de l'accès aux soins**, facteur de protection reconnu dans la littérature médicale.

---

## 4. Transformations appliquées

Un pipeline de transformation automatisé (`ColumnTransformer`) a été mis en place via scikit-learn pour garantir la reproductibilité et éviter tout data leakage.

### Standard Scaling
Toutes les variables numériques continues ont été centrées-réduites (moyenne = 0, écart-type = 1) via `StandardScaler`. Cette normalisation est indispensable pour la stabilité des algorithmes sensibles à l'échelle (Logistic Regression) et pour la PCA.

### One-Hot Encoding
Appliqué aux variables catégorielles nominales (`Assurance`, `Aide_WIC`, `H_Chronique`, `D_Chronique`) avec `drop='first'` pour éviter la **multicolinéarité** (dummy trap). Le OneHotEncoder a été choisi car ces variables n'ont **pas d'ordre naturel** — un OrdinalEncoder induirait une hiérarchie artificielle (ex : Medicaid < Privée < CHIP n'a aucun sens métier).

*Alternative rejetée — OrdinalEncoder* : inadapté pour `Assurance` car les modalités sont nominales. L'ordinalité n'est justifiable que si les catégories ont un ordre naturel et mesurable (ex : niveau d'éducation).

*Alternative rejetée — Target Encoding* : risque élevé de data leakage sur un dataset déséquilibré (~9% positifs). Sans validation croisée stricte, le target encoding sur la classe minoritaire sur-représente les patterns du train set.

### PCA (Analyse en Composantes Principales)
Réduction dimensionnelle configurée pour conserver **95% de la variance expliquée**. Utilisée pour produire une version compacte du dataset (`processed_data_pca.csv`) et supprimer les colinéarités résiduelles entre features.

---

## 5. Justification des choix et alternatives

| Choix retenu | Alternative rejetée | Raison |
|---|---|---|
| Imputation médiane | Imputation moyenne | Robustesse aux outliers médicaux |
| OneHotEncoder | OrdinalEncoder | Pas d'ordre naturel dans les catégories |
| OneHotEncoder | Target Encoding | Risque de data leakage sur classe minoritaire |
| Suppression `Nb_Visites` | Conservation | Data leakage temporel + corrélation post-prédictive |
| `drop='first'` dans OHE | Pas de drop | Évite la multicolinéarité (dummy trap) |

---

## 6. Impact attendu sur les modèles

- **Meilleure convergence** des algorithmes grâce au StandardScaler (gradients stables pour la Logistic Regression).
- **Réduction du bruit** via la PCA — suppression des colinéarités entre `Tabac_Avant` et `Tabac_Trim1`.
- **Signal métier additionnel** via `Evol_Tabac` et `Suivi_T1` — ces features capturent des dynamiques comportementales non présentes dans les variables brutes.
- **Absence de data leakage** : le preprocessor est fitté sur le train set uniquement (`preprocessor.fit_transform(X_train)`), puis appliqué au test set (`preprocessor.transform(X_test)`).

---

## 7. Données et Notebooks

- **Dataset full** (pour Random Forest & HistGradientBoosting) : `/data/process/processed_data_full.csv`
- **Dataset PCA** (archivage / référence) : `/data/process/processed_data_pca.csv`
- **Preprocessor sauvegardé** : `/models/preprocessor.pkl`
- **Notebook de référence** : `notebooks/Preprocessing.ipynb`
- **Méthode de chargement** : les datasets sont prêts via `pd.read_csv()` — plus de valeurs manquantes ni de variables catégorielles textuelles.
- **`data.py`** : mis à jour pour pointer vers `processed_data_full.csv` — compatible avec les 3 modèles chargés par `main.py`.
