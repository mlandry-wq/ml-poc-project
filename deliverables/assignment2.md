# Rapport de Preprocessing et Feature Engineering
---

## 1. Nettoyage des données
*   **Gestion des Valeurs Sentinelles :** Comme identifié lors de l'EDA, le CDC utilise des codes numériques (**9**, **99**, **99.9**) pour désigner l'inconnu. Ces valeurs ont été systématiquement remplacées par des `NaN` pour éviter de biaiser le calcul de la moyenne et l'étape de standardisation.
*   **Filtrage et Encodage de la Cible :** La variable cible `ab_nicu` a été nettoyée pour ne conserver que les entrées "Y" et "N", puis transformée en format binaire : **1** (Risque) et **0** (Normal).
*   **Imputation :** 
    *   **Variables numériques :** Remplissage des valeurs manquantes par la **médiane**, stratégie choisie pour sa robustesse face aux valeurs aberrantes (outliers) présentes dans l'âge ou l'IMC.
    *   **Variables catégorielles :** Remplissage par la **valeur la plus fréquente** (mode) pour préserver la distribution originale.

---

## 2. Feature Engineering
Afin d'apporter une dimension métier supplémentaire au modèle, deux nouvelles variables ont été créées :

*   **`Evol_Tabac` :** Calcule la différence entre la consommation au premier trimestre et celle avant la grossesse. Elle permet au modèle de capter un changement de comportement (réduction ou arrêt), indicateur fort pour la santé du nouveau-né.
*   **`Suivi_T1` :** Variable binaire indiquant si le suivi prénatal a débuté durant le premier trimestre (mois 1 à 3). Elle sert de proxy pour mesurer la précocité de l'accès aux soins.

---

## 3. Transformations appliquées
Un pipeline de transformation automatisé (`ColumnTransformer`) a été mis en place pour garantir la reproductibilité du traitement.

*   **One-Hot Encoding (OHE) :** Appliqué aux variables catégorielles (Assurance, WIC, Diabète...) pour les convertir en vecteurs binaires exploitables par les modèles.
*   **Standard Scaling :** Toutes les variables numériques ont été centrées sur 0 avec un écart-type de 1. Cette normalisation est indispensable avant l'application de la PCA et pour la stabilité des algorithmes.
*   **PCA (Analyse en Composantes Principales) :** Réduction de dimension configurée pour conserver **$95\%$ de la variance expliquée**. Cela permet de condenser l'information des 14 colonnes initiales tout en supprimant le bruit et les colinéarités.

---

## 4. Justification des choix et Alternatives
*   **PCA vs Sélection Manuelle :** L'EDA montrait des corrélations fortes entre les variables de tabagisme. La PCA a été préférée car elle fusionne ces variables liées en axes mathématiques indépendants (orthogonaux) sans perte arbitraire d'information.
*   **Alternative rejetée (Imputation par la moyenne) :** La moyenne est trop sensible aux mères de plus de 45 ans ou aux IMC extrêmes. La médiane a été retenue comme alternative plus fidèle au "profil type" de la population.
*   **Encodage :** Le One-Hot Encoding a été choisi plutôt que le Label Encoding pour éviter d'induire une hiérarchie artificielle entre les types d'assurance.

---

## 5. Impact attendu et Modifications
*   **Impact Modèle :** On attend une meilleure convergence des algorithmes et une réduction du risque de sur-apprentissage (*overfitting*) grâce à la réduction du bruit par la PCA.
*   **Fichiers Python à modifier :**
    *   **`data.py` :** Mise à jour de la fonction de chargement pour pointer vers le nouveau fichier `processed_data_pca.csv`.
    *   **`metrics.py` :** Implémentation du focus sur le **Recall** pour maximiser la détection des admissions NICU.

---

## 6. Données et Notebooks
*   **Dataset transformé :** Localisé dans `/data/process/processed_data_pca.csv`.
*   **Notebook de référence :** `notebooks/Preprocessing.ipynb`.
*   **Méthode de chargement :** Le dataset est prêt à l'emploi via `pd.read_csv()`. Il ne contient plus de valeurs manquantes ni de variables catégorielles textuelles.