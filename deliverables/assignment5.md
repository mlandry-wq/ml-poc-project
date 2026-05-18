# Rapport d'Intégration — Application NORA

---

## 1. Description de l'application

**NORA** (Neonatal Outcome Risk Assessment) est un dashboard interactif développé avec **Streamlit** qui présente l'ensemble du projet de prédiction du risque NICU sous une interface unifiée. L'application couvre le cycle complet du projet : contextualisation médicale, exploration des données, justification des choix de feature engineering, comparaison des modèles, et démonstration en conditions quasi-réelles.

L'application est accessible localement via :

```bash
# Option 1 — pipeline complet (évaluation + dashboard)
python scripts/main.py

# Option 2 — dashboard seul
streamlit run src/app.py
```

Par défaut, le dashboard est accessible sur **http://localhost:8501**.

---

## 2. Objectifs de l'interface

L'interface poursuit trois objectifs distincts :

**Démonstration** — montrer de façon pédagogique que le projet est complet et cohérent, de la donnée brute jusqu'à la prédiction. Chaque décision technique (choix du scaler, de la métrique, du seuil) est expliquée et justifiée dans l'interface elle-même.

**Exploration** — permettre à un utilisateur non technique de comprendre les données, leurs distributions, leurs corrélations et les signaux prédictifs identifiés, sans avoir à ouvrir les notebooks.

**Prédiction** — offrir une démonstration interactive du modèle : l'utilisateur entre un profil clinique maternel et obtient en temps réel une probabilité d'admission NICU, accompagnée d'une interprétation des facteurs de risque détectés.

---

## 3. Structure de l'application

L'application est organisée en **5 onglets** accessibles depuis la barre de navigation latérale.

```
NORA
├── Accueil
├── Données (EDA)
├── Feature Engineering
├── Modèles & Évaluation
└── Démo — Prédiction NICU
```

### Code source

| Fichier | Rôle |
|---|---|
| `src/app.py` | Point d'entrée Streamlit — contient toutes les pages et la logique de prédiction |
| `src/config.py` | Chemins, registre des 3 modèles, configuration Streamlit |
| `src/data.py` | Chargement du dataset préprocessé + split PCA(10) |
| `src/metrics.py` | Calcul des métriques (recall, F-beta β=2, F1, précision, accuracy) |
| `src/model_io.py` | Chargement des modèles sérialisés (joblib/pkl) |
| `scripts/main.py` | Orchestration : évaluation → CSV → lancement Streamlit |

---

## 4. Description des pages

### Page 1 — Accueil

Contextualise le projet pour un public non technique. Présente :

- La problématique médicale (risque NICU, ~9% des naissances aux États-Unis)
- L'objectif du modèle et son périmètre d'utilisation (2e–3e trimestre de grossesse)
- Les données sources (CDC Natality 2018, 99 900 naissances)
- Un guide de navigation rapide vers chaque onglet
- Les sources et les limites de l'outil

### Page 2 — Données (EDA)

Présente les résultats de l'analyse exploratoire avec 9 visualisations commentées :

| Visualisation | Description |
|---|---|
| Donut chart (interactif) | Déséquilibre des classes — 91% / 9% |
| Taux NICU par facteur de risque (interactif) | Signal des 10 variables médicales (×1.4 à ×10 vs base) |
| Distributions des variables continues | Asymétrie des distributions, outliers légitimes |
| Matrice de corrélations | Corrélations faibles des variables de base (< 0.05) |
| Variables catégorielles vs taux NICU | Impact de chaque modalité (Y/N/U) sur le risque |
| Tabac vs risque | Analyse comportementale tabagique |
| Valeurs manquantes & sentinelles CDC | Taux réel après neutralisation des codes 9/99/99.9 |
| Distribution de la cible | Confirmation du déséquilibre en conditions réelles |
| Analyse multivariée | Scatter matrix — absence de séparation linéaire entre classes |

### Page 3 — Feature Engineering

Documente les transformations appliquées au dataset :

- Tableau des **8 variables médicales à fort signal** ajoutées (taux NICU et multiplicateur vs base)
- Description des **3 variables dérivées** créées manuellement (Evol_Tabac, Suivi_T1, Grossesse_Multiple)
- Pipeline de preprocessing en 5 étapes visuelles (sentinelles → imputation → RobustScaler → OHE → SMOTE + PCA)
- Graphique de **variance expliquée par la PCA** (10 composantes = 99.3%)
- Distribution de la prise de poids après nettoyage des aberrations cliniques

### Page 4 — Modèles & Évaluation

Compare les trois modèles entraînés :

- Explication de la **métrique F-beta β=2** et sa justification médicale (FN 2× plus pénalisants que FP)
- 3 cartes descriptives (principe, gestion du déséquilibre, atout/limite de chaque modèle)
- **Tableau comparatif** : recall, précision, F1, F-beta β=2 et seuil optimal pour les 3 modèles
- Graphique comparatif interactif (Plotly) et graphique notebook (4 métriques + F-beta β=2)
- Protocole d'évaluation anti-leakage (split stratifié, SMOTE dans CV, PCA avant SMOTE)
- **Triptyques de performance en onglets** — pour chacun des 3 modèles : matrice de confusion, courbe Précision-Rappel, courbe ROC
- Convergence Optuna (HistGBM) + Benchmark LazyPredict (25 classifieurs)

### Page 5 — Démo — Prédiction NICU

Interface de prédiction interactive organisée en **4 sections de saisie** :

| Section | Variables |
|---|---|
| Profil maternel | Âge, IMC, prise de poids, mois début suivi |
| Tabac | Cigarettes/jour (moyenne grossesse) |
| Antécédents & contexte | Type de grossesse, antécédent mort fœtale, ATCD prématurité, aide WIC |
| Pathologies chroniques | Diabète, hypertension, assurance, PMA, traitement infertilité |
| Pathologies gestationnelles | Éclampsie, HTA gestationnelle, diabète gestationnel |

---

## 5. Inputs utilisateur (page Démo)

Tous les inputs sont inline dans la page (pas dans la sidebar). Chaque variable correspond à une feature réelle du modèle.

| Variable | Type de widget | Plage / Options |
|---|---|---|
| Âge | Slider | 15–50 ans |
| IMC avant grossesse | Number input | 15.0–70.0 |
| Prise de poids | Slider | 0–80 lbs |
| Mois début suivi prénatal | Slider | 1–10 |
| Cigarettes/jour | Slider | 0–40 |
| Type de grossesse | Selectbox | Simple / Multiple (jumeaux+) |
| Antécédent mort fœtale | Selectbox | Non / Oui / Inconnu |
| ATCD prématurité | Selectbox | Non / Oui / Inconnu |
| Aide WIC | Selectbox | Non / Oui / Inconnu |
| Diabète chronique | Selectbox | Non / Oui / Inconnu |
| Hypertension chronique | Selectbox | Non / Oui / Inconnu |
| Assurance | Selectbox | Medicaid / Privée / CHIP / Autre... |
| PMA | Selectbox | Non / Oui / Inconnu / Non applicable |
| Traitement infertilité | Selectbox | Non / Oui / Inconnu |
| Éclampsie | Selectbox | Non / Oui / Inconnu |
| HTA gestationnelle | Selectbox | Non / Oui / Inconnu |
| Diabète gestationnel | Selectbox | Non / Oui / Inconnu |

**Note sur Evol_Tabac :** cette variable dérivée (Tabac_Trim1 − Tabac_Avant) est fixée à 0 dans la démo. L'utilisation de deux sliders distincts générait des valeurs hors distribution (93.7% des données d'entraînement ont Evol_Tabac = 0), ce qui poussait le modèle hors de son espace d'entraînement et produisait des probabilités contre-intuitives.

---

## 6. Outputs affichés (page Démo)

Après soumission du formulaire, l'application affiche :

- **Jauge animée** (Plotly Indicator) — probabilité NICU de 0 à 100%, avec le seuil de 0.16 matérialisé par une ligne verticale et les zones vertes/rouges
- **Verdict coloré** — bandeau vert (Risque faible) ou rouge (Risque élevé détecté), avec la probabilité exacte et la comparaison au seuil optimisé
- **Facteurs de risque identifiés** — liste des variables à forte valeur saisies par l'utilisateur, avec leur taux NICU observé dans les données d'entraînement
- **Avertissement médical** — rappel que l'outil est éducatif et ne remplace pas un avis clinique

---

## 7. Pipeline d'exécution (`scripts/main.py`)

```
python scripts/main.py
         │
         ├─ 1. Validation des modèles (config.MODELS)
         ├─ 2. Validation du point d'entrée Streamlit (build_app callable)
         ├─ 3. load_dataset_split() → (X_train, X_test, y_train, y_test) — PCA(10)
         ├─ 4. Pour chaque modèle :
         │       load_model(path) → model.predict(X_test) → compute_metrics()
         ├─ 5. write_metrics() → results/model_metrics.csv
         ├─ 6. Affichage terminal des métriques
         └─ 7. streamlit run src/app.py --server.address localhost --server.port 8501
```

**Note sur les métriques de `results/model_metrics.csv`** : le pipeline utilise `model.predict()` avec le seuil par défaut (0.50), pas les seuils optimisés (0.10 / 0.16 / 0.46). Les valeurs dans le CSV diffèrent donc des résultats notebooks — c'est une contrainte du framework, pas une dégradation du modèle.

---

## 8. Intégration et nettoyage réalisés

### Fichiers modifiés

| Fichier | Modification |
|---|---|
| `src/config.py` | Remplacement du placeholder `model_a` par les 3 modèles réels (LR, RF, HGB) |
| `src/data.py` | Ajout PCA(10) dans `load_dataset_split()` + chemin relatif portable (suppression du hardcode `/Users/madeleine/...`) |
| `src/metrics.py` | Ajout du F-beta β=2 (`fbeta_score(beta=2)`) comme métrique prioritaire |
| `src/model_io.py` | Correction du chargement `.pkl` : joblib en priorité (les modèles sont sérialisés avec joblib, non compatible avec `pickle.load`) |
| `src/app.py` | Dashboard NORA complet — 5 pages, palette pastel, 15+ graphiques commentés |
| `requirements.txt` | Ajout de `optuna` |
| `.gitignore` | Ajout de `.DS_Store` |
| `README.md` | Réécriture complète — description NORA, installation, exécution, structure |

### Fichiers supprimés

| Fichier | Raison |
|---|---|
| `models/model3_optuna_study.pkl` | Artifact intermédiaire Optuna — inutile une fois le modèle entraîné et les graphiques sauvegardés |

### Notebook mis à jour

| Fichier | Modification |
|---|---|
| `notebooks/Modeling.ipynb` | Suppression de `joblib.dump(study, ...)` — le study Optuna n'est plus sauvegardé |

---

## 9. Résultats finaux

| Modèle | Recall | F-beta β=2 | Seuil optimal |
|---|---|---|---|
| Logistic Regression (baseline) | 0.354 | 0.259 | 0.10 |
| **Random Forest + PCA(10)** ✅ | **0.766** | **0.350** | 0.16 |
| HistGradientBoosting + Optuna | 0.698 | 0.337 | 0.46 |

Le **Random Forest** est retenu comme modèle principal de NORA (meilleur F-beta β=2 = 0.350). Il détecte **76.6% des bébés à risque** sur le jeu de test, avec un seuil de décision à 0.16.
