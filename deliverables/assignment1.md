# Rapport de Projet : Prédiction du Risque Néonatal (NICU)
---

## 1. Description du projet
Ce projet consiste à développer un modèle de Machine Learning capable de prédire les risques de complications de santé chez le nouveau-né en se basant sur le profil clinique et les comportements de la mère durant la grossesse.

L'objectif est d'utiliser les données massives de santé publique pour identifier des signaux d'alarme précoces. Une détection proactive permet d'intensifier le suivi prénatal et de mettre en place des interventions ciblées (ex: sevrage tabagique, gestion du poids) afin d'améliorer la santé infantile et de réduire les coûts liés aux hospitalisations prolongées.

---

## 2. Définition du problème (Machine Learning)
Le projet s'inscrit dans une problématique de **Classification Binaire**.

*   **Entrées (X) :** Variables comportementales (tabagisme), physiologiques (âge, IMC), obstétricales (antécédents, pluralité) et socio-économiques.
*   **Cible (y) :** Admission du nouveau-né en Unité de Soins Intensifs Néonatals (NICU).  
    *   `0` : Pas d'admission.
    *   `1` : Admission (Risque identifié).

---

## 3. Description du dataset
Le dataset s'appuie sur la base de données **Natality 2018** du **CDC** (Centers for Disease Control and Prevention).

*   **Méthode de collecte :** Données extraites des certificats de naissance standardisés aux États-Unis, centralisées par le National Center for Health Statistics (NCHS).
*   **Volume utilisé :** Un échantillon représentatif de **99 900 lignes** (issu du dataset original de 3,8 millions de lignes).
*   **Justification du choix :** Ce dataset offre un réalisme "terrain" avec des classes déséquilibrées et des variables déclaratives, confrontant le modèle à de vrais défis de Data Science.

---

## 4. Description des Features (Variables explicatives)
Afin de maximiser le signal prédictif, **19 variables** ont été sélectionnées et traitées parmi les plus de 200 disponibles, combinant profil de base et antécédents médicaux à fort signal.

### Variables de base (profil maternel)

| Variable | Type | Description métier |
| :--- | :--- | :--- |
| **Age_Mere** | Numérique | Âge de la mère au moment de l'accouchement. |
| **IMC_Mere** | Continu | Indice de Masse Corporelle avant la grossesse. |
| **Tabac_Avant** | Discret | Consommation moyenne de cigarettes avant la conception. |
| **Tabac_Trim1** | Discret | Consommation de cigarettes durant le premier trimestre. |
| **Tabac_Trim3** | Discret | Consommation de cigarettes durant le troisième trimestre (signal plus fort). |
| **Mois_Debut_Suivi** | Discret | Mois de la grossesse où le suivi médical a commencé. |
| **Aide_WIC** | Catégorique | Bénéficiaire du programme d'aide alimentaire (Y/N/U). |
| **H_Chronique** | Catégorique | Hypertension préexistante à la grossesse (Y/N/U). |
| **D_Chronique** | Catégorique | Diabète préexistant à la grossesse (Y/N/U). |
| **Assurance** | Catégorique | Mode de prise en charge financière (Medicaid, Privée, CHIP…). |
| **Prise_Poids** | Continu | Gain de poids total durant la grossesse (en livres, cap [0–80]). |

### Nouvelles variables médicales (fort signal)

| Variable | Taux NICU si positif | vs Base (8.9%) | Type |
| :--- | :--- | :--- | :--- |
| **Grossesse_Multiple** | 37–87% (jumeaux/triplets) | **×4 à ×10** | Binaire (feat. eng.) |
| **Eclampsie** | 27.1% | ×3.0 | Catégorique (Y/N/U) |
| **Traitement_Infertilite** | 21.3% | ×2.4 | Catégorique (Y/N/U) |
| **PMA** (assist. médicale) | 23.9% | ×2.7 | Catégorique (Y/N/X/U) |
| **ATCD_Premature** | 19.2% | ×2.2 | Catégorique (Y/N/U) |
| **HTA_Gestationnelle** | 16.8% | ×1.9 | Catégorique (Y/N/U) |
| **ATCD_Mort_Foetale_bin** | jusqu'à 16.1% | ×1.8 | Binaire (feat. eng.) |
| **Diabete_Gestationnel** | 12.3% | ×1.4 | Catégorique (Y/N/U) |

### Variables créées par Feature Engineering

| Variable | Formule | Justification |
| :--- | :--- | :--- |
| **Evol_Tabac** | `Tabac_Trim1 − Tabac_Avant` | Capture le changement de comportement tabagique |
| **Suivi_T1** | `Mois_Debut_Suivi ≤ 3` | Proxy pour la précocité d'accès aux soins |

---

## 5. Objectifs Business et ML

*   **Objectif Business :** Créer un outil d'aide à la décision pour les services de santé publique afin d'allouer les ressources de prévention aux mères les plus vulnérables.
*   **Métriques d'évaluation :**
    *   **Recall (Sensibilité) :** Métrique prioritaire. Il est crucial de minimiser les Faux Négatifs (ne pas détecter un bébé en danger).
    *   **F-beta (β=2) :** Utilisé pour le tuning du seuil — pondère le Recall 2x plus que la Précision, reflétant l'asymétrie médicale des coûts FN/FP.
    *   **F1-Score & AUC-ROC :** Pour monitorer l'équilibre global face au déséquilibre des classes.
*   **Réduction dimensionnelle :** PCA(10 composantes, ~99% de variance) appliquée après SMOTE dans le pipeline de modélisation — élimine les redondances OHE et améliore la séparation des classes minoritaires.

---

## 6. Premières Analyses Exploratoires (EDA)

L'analyse exploratoire réalisée dans le notebook `EDA.ipynb` a mis en évidence :

1.  **Déséquilibre des classes :** Environ **9%** des naissances aboutissent à une admission en NICU. Le modèle gère ce déséquilibre via SMOTE (dans le pipeline CV pour éviter le leakage) et `class_weight='balanced'`.
2.  **Signal fort des grossesses multiples :** Jumeaux → 37.5% NICU, triplets → 87.5%. Variable la plus discriminante du dataset.
3.  **Conditions gestationnelles prédictives :** Éclampsie (27%), PMA (24%), traitement infertilité (21%), HTA gestationnelle (17%).
4.  **Corrélations faibles pour les variables de base :** Les variables initiales (âge, IMC, tabac) ont toutes des corrélations < 0.05 avec la target. Les nouvelles features médicales apportent un signal significativement plus fort.
5.  **Valeurs sentinelles CDC :** Le CDC encode les non-réponses avec des codes numériques (99, 99.9, 9) — identifiés et remplacés par NaN avant toute transformation.
6.  **PCA(10) : gain majeur en modélisation** : la réduction de 36 → 10 dimensions améliore le recall du Random Forest de 0.662 à **0.766** (+10 pts) en éliminant la redondance des 26 colonnes OHE. Meilleur F-beta β=2 = 0.350 (RF retenu).

---

## 7. Hypothèses, risques et limites

*   **Risque de Data Leakage :** Les données de l'accouchement (poids de naissance, APGAR, méthode d'accouchement) ont été exclues. Les conditions gestationnelles incluses (éclampsie, HTA, diabète gestationnel) sont disponibles pendant la grossesse — ce modèle est applicable à partir du 2e/3e trimestre.
*   **Intégrité et Valeurs Sentinelles :** Le CDC encode les non-réponses avec des valeurs sentinelles numériques — leur identification et neutralisation est une étape critique du preprocessing.
*   **Profil de la population :** Âge moyen 27.4 ans, IMC moyen 29.7. Présence d'outliers médicaux légitimes (IMC > 50, âge > 45) gérés via `RobustScaler`.
*   **Limite dataset :** Absence de données biologiques (bilans sanguins, marqueurs sériques) et génétiques qui représentent les facteurs les plus prédictifs cliniquement.

---

## 8. Données et Notebooks
*   **Localisation :**
    *   Notebook EDA : `notebooks/EDA.ipynb`
    *   Notebook Preprocessing : `notebooks/Preprocessing.ipynb`
    *   Graphiques : `plots/` (sauvegardés automatiquement).
*   **Reproductibilité :** Le code utilise une approche par dictionnaire pour renommer les colonnes brutes du CDC en noms explicites. Le chargement est optimisé avec `usecols` pour limiter l'empreinte mémoire.
