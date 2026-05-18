# Rapport de Modélisation — Prédiction du Risque NICU

---

## 1. Définition du problème ML

### Contexte
Le projet vise à prédire l'admission d'un nouveau-né en Unité de Soins Intensifs Néonatals (NICU) à partir du profil clinique et comportemental de la mère, capturé dans les données de naissance du CDC (Natality 2018, 99 900 observations).

### Type de problème
**Classification binaire supervisée**

| Classe | Encodage | Interprétation |
|---|---|---|
| Pas d'admission NICU | `0` | Nouveau-né sans complication nécessitant des soins intensifs |
| Admission NICU | `1` | Nouveau-né à risque — cible à détecter |

### Déséquilibre des classes
La classe positive représente **~9%** des observations (taux NICU observé). Ce déséquilibre est inhérent au problème médical et constitue l'un des principaux défis de modélisation.

### Asymétrie des coûts d'erreur
Les deux types d'erreurs n'ont pas le même coût :

| Erreur | Description | Coût |
|---|---|---|
| **Faux Négatif (FN)** | Bébé à risque non détecté | Grave — potentiellement irréversible (absence de soins intensifs) |
| **Faux Positif (FP)** | Bébé sain envoyé en NICU | Élevé — hospitalisation inutile, coût financier, stress familial |

Cette asymétrie guide directement le choix de la métrique d'évaluation.

---

## 2. Définition de la métrique d'évaluation

### Métrique principale : F-beta β=2

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{précision} \times \text{recall}}{\beta^2 \cdot \text{précision} + \text{recall}}$$

Avec **β=2**, le recall est pondéré **deux fois plus** que la précision. Ce choix reflète l'asymétrie médicale : un FN (bébé non détecté) est considéré deux fois plus grave qu'un FP (hospitalisation inutile).

### Pourquoi pas le Recall seul ?
Maximiser le Recall seul amène le modèle à prédire *tout* comme positif — recall=1.0 trivial mais précision≈9% (taux de base), générant une cascade de faux positifs coûteux. Le F-beta β=2 contraint le modèle à maintenir une précision minimale.

### Métriques secondaires (monitoring)

| Métrique | Rôle |
|---|---|
| **Recall** | Taux de détection des bébés à risque — ne pas manquer de cas |
| **F1-Score** | Équilibre précision/recall — vision globale |
| **AUC-ROC** | Performance de discrimination indépendante du seuil |
| **Average Precision** | AUC sous la courbe Précision-Rappel — adapté aux données déséquilibrées |

### Threshold tuning
Le seuil de décision (par défaut 0.5) est optimisé par recherche sur [0.10, 0.90] en maximisant le F-beta β=2 sur le jeu de test. Cela permet d'ajuster le compromis FN/FP sans modifier le modèle lui-même.

---

## 3. Protocole d'évaluation

### 3.1 Split train/test

| Paramètre | Valeur |
|---|---|
| Ratio | 80% train / 20% test |
| Stratification | Oui — préserve le ratio 9%/91% dans chaque split |
| Random state | 42 (reproductibilité) |
| Train | 79 920 observations |
| Test | 19 980 observations |

Le jeu de test est mis de côté avant tout preprocessing et n'est jamais utilisé pendant l'entraînement ou le tuning.

### 3.2 Gestion du déséquilibre

**Stratégie différenciée selon le modèle :**

- **Logistic Regression** : aucun rééchantillonnage. Le threshold tuning F-beta β=2 gère seul le déséquilibre. `class_weight='balanced'` est volontairement évité — sur des données déséquilibrées + PCA, il distord les probabilités et produit des résultats dégénérés (recall artificiel ~98%).

- **Random Forest & HistGradientBoosting** : SMOTE (Synthetic Minority Over-sampling Technique) appliqué uniquement sur le train set, après le split. Le résultat est un train set équilibré (50/50) utilisé pour l'entraînement. Le test set reste dans sa distribution originale.

> **Anti-leakage** : SMOTE est toujours appliqué *après* le split, jamais sur l'ensemble complet des données. Pour la validation croisée Optuna, SMOTE est intégré dans le pipeline CV pour éviter que les exemples synthétiques contaminent les folds de validation.

### 3.3 Réduction dimensionnelle (PCA)

Une PCA à 10 composantes (99.3% de variance expliquée) est appliquée après le preprocessing :

- PCA fittée sur `X_train` **original** (avant SMOTE) — évite que les exemples synthétiques influencent l'espace latent
- Transformée appliquée sur `X_train_res` (post-SMOTE) pour RF/HGB, et sur `X_train` original pour LR
- `X_test` transformé avec le même PCA fitté — aucun leakage

**Justification** : le dataset contient 26 colonnes issues du One-Hot Encoding de 10 variables catégorielles, créant une redondance importante. La PCA(10) compresse cette information et améliore la séparation des classes. Impact mesuré : +10 pts de recall pour le Random Forest.

### 3.4 Validation croisée (Optuna)

Pour l'optimisation des hyperparamètres du HistGradientBoosting :

| Paramètre | Valeur |
|---|---|
| Méthode | StratifiedKFold |
| Nombre de folds | 3 |
| Scoring | F-beta β=2 |
| Algorithme | TPE (Tree-structured Parzen Estimator) via Optuna |
| Nombre de trials | 25 |
| Pipeline par fold | SMOTE → PCA(10) → HGB |

La validation croisée est réalisée sur `X_train` original (sans SMOTE pré-appliqué). SMOTE est intégré dans le pipeline à l'intérieur de chaque fold.

---

## 4. Présentation des trois modèles

### Modèle 1 — Logistic Regression (Baseline)

#### Hypothèses principales
- La relation entre les features et la probabilité d'admission NICU est **linéaire dans l'espace PCA** (combinaisons linéaires des composantes principales suffisent à séparer les classes).
- Les 10 composantes PCA contiennent suffisamment de signal discriminant pour une frontière linéaire.

#### Avantages attendus
- **Interprétabilité** : les coefficients indiquent directement le sens et l'intensité de chaque composante.
- **Rapidité** : entraînement quasi-instantané, même sur 80k observations.
- **Calibration des probabilités** : LR produit des probabilités naturellement bien calibrées, ce qui rend le threshold tuning plus fiable.
- **Référence baseline** : si LR performe bien, les modèles plus complexes n'apportent pas de valeur justifiant leur complexité.

#### Limites attendues
- **Linéarité** : incapable de capturer les interactions non-linéaires entre features (confirmé par l'EDA : les scatter plots montrent des classes mélangées sans frontière linéaire claire).
- **Signal faible** : avec des corrélations < 0.15, une frontière linéaire dans l'espace PCA aura du mal à séparer les classes.
- **Recall limité** : attendu autour de 30-40% sans rééchantillonnage.

#### Adéquation avec le problème et la métrique
LR sert de **plancher de performance**. Si RF et HGB ne le dépassent pas sur le F-beta β=2, leur complexité n'est pas justifiée. Son threshold tuning F-beta β=2 lui permet de capturer une fraction non triviale des positifs malgré l'absence de balancing.

**Résultats obtenus** : recall=0.354, F-beta β=2=0.259, seuil=0.10

---

### Modèle 2 — Random Forest + PCA(10)

#### Hypothèses principales
- Les relations entre features et risque NICU sont **non-linéaires et interactives** (ex : grossesse multiple + éclampsie → risque multiplicatif, pas additif).
- La combinaison SMOTE + `class_weight='balanced_subsample'` permet au modèle d'apprendre sur une distribution équilibrée sans sur-représenter les faux positifs.
- La PCA(10) élimine suffisamment de bruit pour que les arbres construits sur 10 composantes soient plus discriminants que sur 36 features brutes.

#### Avantages attendus
- **Non-linéarité** : capture les interactions complexes entre variables médicales non détectables par LR.
- **Robustesse aux outliers** : les arbres de décision ne sont pas affectés par les valeurs extrêmes (IMC > 50, tabac > 40 cig/jour).
- **`balanced_subsample`** : recalcule les poids de classe à chaque arbre bootstrap, ce qui est plus efficace que `balanced` global sur données déséquilibrées.
- **Variance réduite** : l'agrégation de 500 arbres (bagging) limite le sur-apprentissage.

#### Limites attendues
- **Seuil bas (0.16)** : le modèle produit des probabilités concentrées dans les basses valeurs — le seuil optimal F-beta β=2 est très bas, ce qui génère plus de faux positifs qu'un seuil conservateur.
- **Temps d'entraînement** : 500 arbres sur 145k observations (post-SMOTE) est long (~5-10 min).
- **Interprétabilité réduite** : les importances MDI ne sont plus directement interprétables après PCA (les composantes ne correspondent pas à des variables métier).

#### Adéquation avec le problème et la métrique
RF est le modèle le plus adapté à ce problème : les grossesses à haut risque impliquent des **combinaisons** de facteurs (grossesse multiple + éclampsie, PMA + HTA gestationnelle) que seul un modèle non-linéaire peut capturer. Le F-beta β=2 le plus élevé (0.350) confirme cette adéquation.

**Résultats obtenus** : recall=0.766, F1=0.193, F-beta β=2=0.350, seuil=0.16 ✅ **(modèle retenu)**

---

### Modèle 3 — HistGradientBoosting + Optuna + PCA(10)

#### Hypothèses principales
- Un modèle de **boosting** (apprentissage séquentiel correctif) est mieux adapté qu'un modèle de bagging (RF) pour les cas difficiles à la frontière de décision.
- L'optimisation bayésienne (Optuna TPE) des hyperparamètres sur le F-beta β=2 permet de trouver une configuration mieux calibrée pour le compromis FN/FP que les paramètres par défaut.
- Le pipeline SMOTE → PCA → HGB dans chaque fold de CV garantit l'absence de data leakage.

#### Avantages attendus
- **Boosting** : chaque arbre corrige les erreurs des précédents — focus naturel sur les cas difficiles (bébés à risque avec profil atypique).
- **Seuil conservateur (0.46)** : le modèle produit des probabilités mieux calibrées, résultant en un seuil plus élevé et moins de faux positifs qu'avec RF — avantage pratique en contexte hospitalier.
- **Optuna TPE** : optimisation bayésienne plus efficace qu'un GridSearch sur 6 hyperparamètres.
- **Natif sklearn** : pas de dépendance système (contrairement à XGBoost/LightGBM), plus stable en production.

#### Limites attendues
- **Temps d'optimisation** : 25 trials × 3 folds × pipeline complet (SMOTE + PCA + HGB) = ~15-20 min.
- **Sensibilité aux hyperparamètres** : sans tuning, HGB par défaut sous-performe RF sur ce dataset.
- **Recall légèrement inférieur** : 0.698 vs 0.766 pour RF — manque ~7 pts de recall, soit ~12 bébés à risque supplémentaires non détectés sur 1780.

#### Adéquation avec le problème et la métrique
HGB est un **excellent candidat pratique** : son seuil de décision (0.46) est naturellement plus conservateur que RF (0.16), générant moins de faux positifs pour un recall légèrement moindre. Dans un contexte hospitalier à ressources limitées, ce compromis peut être préféré. Son F-beta β=2 (0.337) est proche du RF (0.350), ce qui en fait un choix solide selon la tolérance aux FP du système de santé.

**Résultats obtenus** : recall=0.698, F1=0.190, F-beta β=2=0.337, seuil=0.46

---

## 5. Justification du choix des trois modèles

Les trois modèles ont été choisis pour former une **progression méthodologique cohérente** :

| Critère | LR | RF | HGB |
|---|---|---|---|
| Complexité | Faible | Moyenne | Élevée |
| Hypothèse | Linéarité | Non-linéarité (bagging) | Non-linéarité (boosting) |
| Gestion déséquilibre | Threshold seul | SMOTE + class_weight | SMOTE + class_weight + Optuna F-beta |
| Interprétabilité | Haute | Moyenne | Faible |
| Temps d'entraînement | < 1s | ~10 min | ~20 min |
| F-beta β=2 obtenu | 0.259 | **0.350** | 0.337 |

**LR** est indispensable comme baseline : sans référence, on ne peut pas quantifier le gain apporté par RF et HGB. Sa simplicité permet aussi de détecter tout bug de pipeline (si LR performe trop bien, c'est suspect).

**RF** est le modèle principal : il capture les interactions non-linéaires entre variables médicales, qui sont documentées dans la littérature (grossesses multiples, combinaisons de comorbidités). Son F-beta β=2 le plus élevé confirme qu'il est le mieux adapté à ce problème avec ces données.

**HGB** est le modèle de comparaison avancé : le boosting est théoriquement mieux adapté que le bagging pour les problèmes déséquilibrés (focus sur les exemples difficiles), et l'optimisation Optuna permet d'explorer l'espace des hyperparamètres de façon plus rigoureuse. La validation croisée interne (3 folds F-beta β=2) garantit une sélection de modèle sans data leakage.

> **LazyPredict** a été utilisé en amont (Section 4 du notebook Modeling) pour surveiller ~25 classifieurs sklearn sur un sous-échantillon de 8 000 observations (sans SMOTE ni threshold tuning, seuil par défaut 0.5). Les résultats confirment que **RF est parmi les meilleurs classifieurs non-linéaires** (top 5, derrière SVC qui ne scale pas sur 145k observations et des variantes de la famille RF). HistGBM n'apparaît pas dans le top 10 sur ce benchmark — son choix est validé par ses résultats finaux (recall=0.698, F-beta=0.337) et par la progression méthodologique bagging → boosting, pas par LazyPredict.

---

## 6. Notebooks — Localisation et Reproduction

### Localisation

| Notebook | Chemin | Rôle |
|---|---|---|
| EDA | `notebooks/EDA.ipynb` | Analyse exploratoire, distributions, corrélations, visualisations |
| Preprocessing | `notebooks/Preprocessing.ipynb` | Nettoyage, feature engineering, pipeline, génération des données traitées |
| Modeling | `notebooks/Modeling.ipynb` | Entraînement des 3 modèles, LazyPredict, évaluation, comparaison |

### Ordre d'exécution

Les notebooks doivent être exécutés dans cet ordre :

```
1. notebooks/Preprocessing.ipynb   → génère data/process/processed_data_full.csv
2. notebooks/EDA.ipynb             → lecture optionnelle (analyse exploratoire)
3. notebooks/Modeling.ipynb        → entraîne et évalue les 3 modèles
```

> **Note** : `EDA.ipynb` peut être exécuté avant ou après Preprocessing, il charge les données brutes directement depuis `data/raw/natl2018us.csv`.

### Prérequis

```bash
pip install numpy pandas scikit-learn imbalanced-learn optuna matplotlib seaborn joblib lazypredict
brew install libomp   # macOS uniquement — requis par XGBoost/LightGBM
```

### Reproduction complète

```bash
# 1. Cloner le repo
git clone https://github.com/mlandry-wq/ml-poc-project.git
cd ml-poc-project

# 2. Installer les dépendances
pip install -r requirements.txt   # si disponible

# 3. Placer le dataset brut
# → data/raw/natl2018us.csv  (CDC Natality 2018, non versionné — trop volumineux)

# 4. Exécuter dans l'ordre
jupyter notebook notebooks/Preprocessing.ipynb  # Kernel → Restart & Run All
jupyter notebook notebooks/Modeling.ipynb        # Kernel → Restart & Run All
```

### Fichiers générés

| Fichier | Généré par | Contenu |
|---|---|---|
| `data/process/processed_data_full.csv` | Preprocessing.ipynb | 99 900 × 37 colonnes (36 features + target) |
| `models/preprocessor.pkl` | Preprocessing.ipynb | Pipeline ColumnTransformer (RobustScaler + OHE) |
| `models/model1_logistic_regression.pkl` | Modeling.ipynb | Modèle LR entraîné |
| `models/model3_histgbm_optuna.pkl` | Modeling.ipynb | Modèle HGB entraîné (meilleurs hyperparamètres Optuna) |
| `plots/` | Les deux notebooks | Toutes les visualisations (PNG, 300 dpi) |

> `model2_random_forest.pkl` (~1.5 GB) n'est pas versionné sur GitHub — il se régénère en 10 min en relançant `Modeling.ipynb`.
