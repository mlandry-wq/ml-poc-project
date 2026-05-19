# NORA — Neonatal Outcome Risk Assessment

Modèle de machine learning pour la prédiction du risque d'admission en Unité de Soins Intensifs Néonatals (NICU), développé dans le cadre d'un projet de santé publique.

---

## Description du projet

NORA prédit, à partir du profil clinique et comportemental de la mère, la probabilité qu'un nouveau-né soit admis en NICU à la naissance.

**Données :** échantillon de 99 900 naissances issu de la base **Natality 2018** du CDC (Centers for Disease Control and Prevention, États-Unis).

**Problème ML :** classification binaire — `1` = admission NICU, `0` = pas d'admission. Le déséquilibre de classe (~9% positifs) est géré différemment selon le modèle : SMOTE + `class_weight` pour RF et HGB, threshold tuning seul pour LR.

**Pipeline :** données brutes CDC → préprocesseur (imputation + RobustScaler + OHE) → 36 features → PCA(10, ~99% variance) → modèle → probabilité.

**Métrique prioritaire :** F-beta β=2 — recall pondéré 2× plus que la précision (un faux négatif = bébé à risque non détecté est deux fois plus coûteux qu'un faux positif).

---

## Résultats des modèles

| Modèle | Recall | F-beta β=2 | Seuil optimal |
|---|---|---|---|
| Logistic Regression (baseline) | 0.354 | 0.259 | 0.10 |
| **Random Forest + PCA(10)** ✅ | **0.766** | **0.350** | 0.16 |
| HistGradientBoosting + Optuna | 0.698 | 0.337 | 0.46 |

Le Random Forest est le modèle retenu (meilleur F-beta β=2). Les métriques affichées dans `results/model_metrics.csv` après `python scripts/main.py` utilisent le seuil par défaut 0.50 — elles diffèrent des résultats notebooks qui utilisent les seuils optimisés.

---

## Installation

**Prérequis :** Python 3.10+

```bash
# Cloner le dépôt
git clone https://github.com/mlandry-wq/ml-poc-project.git
cd ml-poc-project

# Créer et activer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## Données et modèles

Les fichiers de données et certains modèles ne sont pas inclus dans le dépôt (trop volumineux).

1. Placer le dataset préprocessé dans : `data/process/processed_data_full.csv`
2. Placer les modèles entraînés dans `models/` :
   - `models/model1_logistic_regression.pkl`
   - `models/model2_random_forest.pkl` *(>1GB — non versionné)*
   - `models/model3_histgbm_optuna.pkl`

Pour régénérer les données et les modèles, exécuter dans l'ordre :
- `notebooks/EDA.ipynb`
- `notebooks/Preprocessing.ipynb`
- `notebooks/Modeling.ipynb`

---

## Exécution

```bash
python scripts/main.py
```

Cette commande :
1. Charge les 3 modèles enregistrés dans `src/config.py`
2. Appelle `load_dataset_split()` — données PCA(10)-transformées
3. Évalue chaque modèle sur le jeu de test (80/20, stratifié, seed=42)
4. Calcule recall, F-beta β=2, F1, précision, accuracy
5. Sauvegarde les résultats dans `results/model_metrics.csv`
6. Lance l'application Streamlit NORA sur `http://localhost:8501`

---

## Structure du dépôt

```
ml-poc-project/
├── data/
│   └── process/
│       ├── processed_data_full.csv   # 36 features post-preprocessing
│       └── processed_data_pca.csv    # 4 composantes (95% variance)
├── deliverables/
│   ├── assignment1.md                # Description du projet + EDA
│   ├── assignment2.md                # Preprocessing + Feature Engineering
│   ├── assignment3.md                # Modélisation
│   ├── assignment4.md                # Évaluation et visualisations
│   └── assignment5.md                # Intégration finale
├── models/
│   ├── model1_logistic_regression.pkl
│   ├── model2_random_forest.pkl      # Non versionné (>1GB)
│   ├── model3_histgbm_optuna.pkl
│   └── preprocessor.pkl
├── notebooks/
│   ├── EDA.ipynb
│   ├── Preprocessing.ipynb
│   └── Modeling.ipynb
├── plots/                            # Graphiques générés
├── results/
│   └── model_metrics.csv             # Généré par scripts/main.py
├── scripts/
│   └── main.py                       # Point d'entrée principal
├── src/
│   ├── app.py                        # Dashboard Streamlit NORA
│   ├── config.py                     # Chemins et registre des modèles
│   ├── data.py                       # Chargement et split du dataset
│   └── metrics.py                    # Calcul des métriques
├── requirements.txt
└── README.md
```

---

## Dashboard NORA

L'application Streamlit propose 6 onglets :

- **Accueil** — contexte médical, objectifs, périmètre du modèle
- **Données (EDA)** — distribution des variables, taux NICU par facteur de risque
- **Feature Engineering** — variables dérivées et médicales à fort signal
- **Modèles & Évaluation** — comparaison des 3 modèles, courbes ROC et Precision-Recall
- **Démo** — prédiction interactive du risque NICU à partir d'un profil maternel
- **Limites** — contraintes du dataset, périmètre d'utilisation, perspectives
