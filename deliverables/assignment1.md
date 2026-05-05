# Rapport de Projet : Prédiction du Risque Néonatal (NICU)
---

## 1. Description du projet
Ce projet consiste à développer un modèle de Machine Learning capable de prédire les risques de complications de santé chez le nouveau-né en se basant sur le profil clinique et les comportements de la mère durant la grossesse. 

L'objectif est d'utiliser les données massives de santé publique pour identifier des signaux d'alarme précoces. Une détection proactive permet d'intensifier le suivi prénatal et de mettre en place des interventions ciblées (ex: sevrage tabagique, gestion du poids) afin d'améliorer la santé infantile et de réduire les coûts liés aux hospitalisations prolongées.

---

## 2. Définition du problème (Machine Learning)
Le projet s'inscrit dans une problématique de **Classification Binaire**.

*   **Entrées (X) :** Variables comportementales (tabagisme), physiologiques (âge, IMC) et socio-économiques.
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
Afin de maximiser le signal prédictif, j'ai sélectionné 11 variables clés parmi les plus de 200 disponibles initialement. Les noms ont été rendus explicites pour l'analyse.

| Variable | Type | Description métier |
| :--- | :--- | :--- |
| **Age_Mere** | Numérique | Âge de la mère au moment de l'accouchement. |
| **IMC_Mere** | Continu | Indice de Masse Corporelle avant la grossesse. |
| **Tabac_Avant_Grossesse** | Discret | Consommation moyenne de cigarettes avant la conception. |
| **Tabac_Trim1** | Discret | Consommation de cigarettes durant le premier trimestre. |
| **Mois_Debut_Suivi** | Discret | Mois de la grossesse où le suivi médical a commencé. |
| **Nb_Visites_Prenatales**| Discret | Nombre total de consultations durant la grossesse. |
| **Aide_Sociale_WIC** | Catégorique | Bénéficiaire du programme d'aide alimentaire (Y/N). |
| **Hypertension_Chronique**| Catégorique | Hypertension préexistante à la grossesse (Y/N). |
| **Diabete_Chronique** | Catégorique | Diabète préexistant à la grossesse (Y/N). |
| **Type_Assurance** | Catégorique | Mode de prise en charge financière (Privé, Medicaid, etc.). |
| **Prise_Poids_Lbs** | Continu | Gain de poids total durant la grossesse (en livres). |

---

## 5. Objectifs Business et ML

*   **Objectif Business :** Créer un outil d'aide à la décision pour les services de santé publique afin d'allouer les ressources de prévention aux mères les plus vulnérables.
*   **Métriques d'évaluation :** 
    *   **Recall (Sensibilité) :** Métrique prioritaire. Il est crucial de minimiser les Faux Négatifs (ne pas détecter un bébé en danger).
    *   **AUC-ROC & F1-Score :** Utilisés pour monitorer la performance globale face au déséquilibre des classes.

---

## 6. Premières Analyses Exploratoires (EDA)

L'analyse exploratoire réalisée dans le notebook `EDA.ipynb` a mis en évidence :

1.  **Déséquilibre des classes :** Environ **9%** des naissances aboutissent à une admission en NICU. Le modèle devra gérer ce déséquilibre (via `class_weight` ou rééchantillonnage).
2.  **Corrélation et Biais :** La variable `Nb_Visites_Prenatales` montre la plus forte corrélation avec la cible ($r = -0.14$). Cependant, un biais de "durée de grossesse" est suspecté : un accouchement prématuré réduit mécaniquement le nombre de visites possibles.
3.  **Profils à risque :** Les visualisations (boxplots) indiquent qu'un IMC élevé et un âge maternel avancé sont des facteurs aggravants visibles dans la distribution des admissions NICU.

---

## 7. Hypothèses, risques et limites

*   **Risque de Data Leakage :** Nous avons exclu les données de l'accouchement (césarienne, poids de naissance) pour garantir que le modèle ne prédise que sur des éléments connus *pendant* la grossesse.
*   **Intégrité et Valeurs Sentinelles :** L'EDA a révélé que le CDC n'utilise pas de "NaN" classiques mais des valeurs sentinelles (ex: 99 ou 99.9 pour l'IMC et le tabagisme) pour coder l'inconnu. Leur identification est cruciale pour ne pas fausser les futures prédictions.  
*   **Profil de la population d'étude :** La population échantillonnée présente un âge moyen de 27,4 ans et un IMC moyen de 29,7. L'analyse des histogrammes montre une forte concentration de l'IMC vers le seuil de surpoids et la présence d'outliers extrêmes qui nécessiteront un traitement de mise à l'échelle (scaling)

---

## 8. Données et Notebooks
*   **Localisation :**
    *   Notebook : `notebooks/EDA.ipynb`
    *   Graphiques : `plots/` (sauvegardés automatiquement).
*   **Reproductibilité :** Le code utilise une approche par dictionnaire pour renommer les colonnes brutes du CDC en noms explicites. Le chargement est optimisé avec `usecols` pour limiter l'empreinte mémoire.