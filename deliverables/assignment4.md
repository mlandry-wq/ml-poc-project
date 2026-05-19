# Rapport de Visualisation — Prédiction du Risque NICU

---

## 1. Vue d'ensemble des visualisations produites

Le projet a généré **16 visualisations** réparties en trois catégories, sauvegardées dans le dossier `plots/` :

| Catégorie | Fichier(s) | Notebook source |
|---|---|---|
| **Données brutes (EDA)** | `0_missing_values.png`, `1_distribution_target.png`, `2_distributions_continues.png`, `3_tabac_vs_risque.png`, `4_categoriques_vs_risque.png`, `4b_nouvelles_vars_medicales.png`, `5_correlations.png`, `6_multivariate.png`, `poids_bins.png` | `EDA.ipynb` |
| **Après preprocessing** | `preprocessing_pca_variance.png` | `Preprocessing.ipynb` |
| **Performances des modèles** | `model1_logistic_regression.png`, `model2_random_forest.png`, `model3_histgbm_optuna.png`, `model3_optuna.png`, `comparaison_modeles.png`, `lazypredict_benchmark.png` | `Modeling.ipynb` |

---

## 2. Visualisations des données brutes

### 2.1 Distribution des valeurs manquantes (`0_missing_values.png`)

**Objectif** : identifier les colonnes avec des données manquantes et quantifier leur ampleur avant tout traitement. Le CDC encode les non-réponses avec des valeurs sentinelles numériques (99, 99.9, 9) — elles doivent être converties en NaN avant d'être comptabilisées comme manquantes.

**Type de graphique** : diagramme en barres horizontales, trié par taux de valeurs manquantes décroissant, avec annotation du pourcentage sur chaque barre.

Ce choix permet une lecture rapide des variables les plus incomplètes et oriente les décisions d'imputation. Un tableau serait moins lisible sur 20 variables.

**Interprétation** : `IMC_Mere` et `Prise_Poids` concentrent la majorité des valeurs manquantes (jusqu'à ~15-20% après remplacement des sentinelles). Les variables catégorielles médicales (`Eclampsie`, `HTA_Gestationnelle`, etc.) présentent des taux de manquants plus faibles (<5%). Aucune variable n'atteint un taux justifiant sa suppression — toutes sont conservées et imputées.

**Pertinence** : visualisation préalable indispensable — elle valide le choix de l'imputation par médiane (variables numériques) et par mode (variables catégorielles) plutôt que la suppression de lignes ou de colonnes.

---

### 2.2 Distribution de la variable cible (`1_distribution_target.png`)

**Objectif** : quantifier et visualiser le déséquilibre des classes, point de départ de toute la stratégie de modélisation.

**Type de graphique** : diagramme en barres avec annotation des effectifs et pourcentages, palette rouge/bleu pour distinguer les classes.

Un camembert aurait été illisible sur des proportions 91%/9%. Le bar chart avec annotations permet de lire les effectifs exacts (~ 91 000 négatifs, ~ 8 900 positifs).

**Interprétation** : la classe positive (admission NICU) représente **~8.9%** du dataset. Ce déséquilibre est réaliste — il reflète fidèlement la prévalence NICU dans la population générale américaine. Il justifie directement l'utilisation de SMOTE, de `class_weight`, du threshold tuning et du F-beta β=2 comme métrique.

**Pertinence** : cette visualisation est la justification centrale de toute la stratégie de gestion du déséquilibre. Elle est citée en référence dans les sections Preprocessing et Modeling.

---

### 2.3 Distributions des variables continues vs risque NICU (`2_distributions_continues.png`)

**Objectif** : comparer les distributions des variables numériques (âge, IMC, tabac, prise de poids) entre la classe positive (admission NICU) et la classe négative, pour identifier les variables les plus discriminantes visuellement.

**Type de graphique** : KDE plots (Kernel Density Estimation) superposés par classe, en grille multi-panneaux. Les KDE permettent de comparer des distributions continues sans biais de binage contrairement aux histogrammes.

**Interprétation** : les distributions des deux classes se chevauchent massivement pour toutes les variables de base — l'âge, l'IMC et le tabac seuls ne permettent pas de séparer les classes (corrélations < 0.05 confirmées numériquement). Ce résultat motive directement l'ajout de variables médicales à fort signal (`Grossesse_Multiple`, `Eclampsie`, etc.) dans le feature engineering.

**Pertinence** : cette visualisation démontre l'insuffisance des variables comportementales seules et justifie l'enrichissement du dataset avec des variables cliniques. Elle constitue le principal argument de l'EDA pour la phase de feature engineering.

---

### 2.4 Analyse comportementale — Tabac vs risque NICU (`3_tabac_vs_risque.png`)

**Objectif** : analyser si l'intensité du tabagisme maternel est corrélée au risque d'admission NICU, en segmentant la consommation en tranches (non-fumeur, 1–5, 6–10, 11–20, 20+ cig/jour).

**Type de graphique** : diagrammes en barres groupées par tranche de consommation, avec ligne de référence horizontale au taux moyen. Deux panneaux côte à côte : tabac avant grossesse et tabac au T1.

**Interprétation** : une légère tendance est observable (taux NICU légèrement plus élevé chez les fumeuses intensives), mais le signal reste faible — corrélation < 0.05, probablement atténué par la **sous-déclaration** des mères sur les certificats de naissance. Le tabac est conservé comme variable du modèle pour sa valeur comportementale cumulée (Evol_Tabac), mais ne constitue pas un signal discriminant fort.

**Pertinence** : cette visualisation est importante précisément parce qu'elle montre l'**absence de signal fort** — ce qui motive l'enrichissement du dataset avec des variables médicales à fort signal.

---

### 2.5 Taux NICU par variable catégorielle (`4_categoriques_vs_risque.png`)

**Objectif** : mesurer le pouvoir discriminant des variables catégorielles médicales en comparant le taux d'admission NICU par modalité.

**Type de graphique** : diagrammes en barres groupées par variable, avec une ligne de référence horizontale indiquant le taux de base (~8.9%). Chaque barre est annotée de son taux NICU.

Ce type de graphique est plus informatif qu'une matrice de corrélation pour les variables catégorielles — il révèle à la fois la direction et l'amplitude de l'effet par modalité.

**Interprétation** : les écarts par rapport au taux de base révèlent les signaux les plus forts :

| Variable | Modalité positive | Taux NICU | Ratio vs base |
|---|---|---|---|
| `Grossesse_Multiple` | Jumeaux | 37.5% | ×4.2 |
| `Grossesse_Multiple` | Triplets+ | 87.5% | ×9.8 |
| `Eclampsie` | Y | 27.1% | ×3.0 |
| `PMA` | Y | 23.9% | ×2.7 |
| `Traitement_Infertilite` | Y | 21.3% | ×2.4 |
| `ATCD_Premature` | Y | 19.2% | ×2.2 |
| `HTA_Gestationnelle` | Y | 16.8% | ×1.9 |

**Pertinence** : cette visualisation est la principale justification de l'enrichissement des features. Elle démontre que les variables médicales disponibles dans le dataset CDC portent un signal 2 à 10× supérieur aux variables comportementales de base.

---

### 2.6 Variables médicales à fort signal — Taux NICU par modalité (`4b_nouvelles_vars_medicales.png`)

**Objectif** : visualiser le signal prédictif des 6 variables médicales clés (éclampsie, PMA, traitement infertilité, ATCD prématuré, HTA gestationnelle, diabète gestationnel) en comparant le taux NICU par modalité (Y/N/U).

**Type de graphique** : grille 2×3 de diagrammes en barres, un par variable médicale, avec ligne de référence horizontale au taux de base (8.9%). Chaque barre est colorée selon qu'elle dépasse ou non la base.

**Interprétation** : pour toutes ces variables, la modalité `Y` (présence de la condition) produit systématiquement un taux NICU **supérieur à la base**, avec des multiplicateurs allant de ×1.4 (diabète gestationnel, 12.3%) à ×3.0 (éclampsie, 27.1%). La modalité `U` (inconnu) montre des taux intermédiaires — ces valeurs ne sont pas aléatoires et portent un signal résiduel.

**Pertinence** : ce graphique est le pendant de `4_categoriques_vs_risque.png` pour les variables médicales spécifiquement. Il justifie leur inclusion dans le modèle avec des chiffres concrets et valide *a posteriori* la stratégie de feature selection basée sur l'EDA.

---

### 2.7 Matrice de corrélations (`5_correlations.png`)

**Objectif** : quantifier les corrélations linéaires entre toutes les variables numériques et avec la cible, et détecter les colinéarités potentielles entre prédicteurs.

**Type de graphique** : heatmap de corrélation (Pearson) avec masque sur le triangle supérieur, colormap divergente (bleu/rouge), annotations des coefficients.

La heatmap est le standard pour visualiser une matrice de corrélation — elle permet d'identifier d'un coup d'œil les clusters de variables corrélées entre elles et les variables corrélées avec la cible.

**Interprétation** : deux observations majeures :
1. **Signal faible** : toutes les corrélations avec la cible sont inférieures à 0.15. Les variables comportementales (tabac, IMC, âge) ont des corrélations < 0.05.
2. **Colinéarités internes** : `Tabac_Avant`, `Tabac_Trim1` et `Tabac_Trim3` sont fortement corrélés entre eux (r > 0.7), justifiant la variable dérivée `Evol_Tabac` et motivant l'utilisation de la PCA pour réduire cette redondance.

**Pertinence** : cette visualisation justifie simultanément (1) l'ajout de nouvelles features médicales et (2) l'utilisation de la PCA pour compresser les colinéarités OHE/tabac en composantes denses.

---

### 2.8 Analyse multivariée — Scatter plots (`6_multivariate.png`)

**Objectif** : explorer les interactions entre paires de variables continues (âge/IMC et IMC/prise de poids) en superposant les deux classes pour détecter une éventuelle séparation visuelle.

**Type de graphique** : scatter plots avec sous-échantillonnage stratifié (2 500 points par classe), opacité différenciée pour mettre en évidence les cas NICU (rouge, plus opaque) sur les non-NICU (bleu, transparent).

**Interprétation** : les cas NICU (rouge) sont répartis **uniformément** dans l'espace des variables de base — aucun cluster isolé n'émerge. Les classes sont mélangées sans frontière linéaire ou non-linéaire évidente sur ces deux projections. Ce résultat confirme que les relations entre features et risque NICU sont complexes et non détectables par des approches simples.

**Pertinence** : cette visualisation justifie le choix de modèles non-linéaires (Random Forest, HistGBM) plutôt qu'une régression logistique simple, et valide l'utilisation de la PCA pour réorganiser l'espace des features en directions plus discriminantes.

---

## 3. Visualisations après feature engineering

### 3.1 Variance expliquée cumulée par la PCA (`preprocessing_pca_variance.png`)

**Objectif** : déterminer le nombre optimal de composantes principales à retenir et justifier le choix de PCA(10) pour la modélisation.

**Type de graphique** : courbe de variance cumulée (en %) en fonction du nombre de composantes, avec une ligne de référence horizontale à 95% et une ligne verticale au coude optimal. Ce graphique est le standard de visualisation pour le choix du nombre de composantes PCA.

**Interprétation** : la courbe montre que :
- **4 composantes** capturent déjà 95% de la variance (utilisé pour `processed_data_pca.csv`)
- **10 composantes** capturent 99.3% de la variance — seuil choisi pour la modélisation car il conserve les signaux des features médicales rares (grossesse multiple, éclampsie) qui contribuent peu à la variance totale mais sont très discriminants

Le genou de la courbe n'est pas au niveau des 4 composantes pour la modélisation : les composantes 5-10 encodent des patterns rares mais à fort signal prédictif (cas NICU liés à des comorbidités spécifiques).

**Pertinence** : cette visualisation est la justification quantitative du choix PCA(10). Elle valide que la réduction de 36 → 10 dimensions ne supprime que 0.7% de l'information totale tout en éliminant la redondance des 26 colonnes OHE.

---

### 3.2 Distribution de la prise de poids après traitement (`poids_bins.png`)

**Objectif** : valider le traitement des valeurs aberrantes de `Prise_Poids` — vérifier que le cap à [0–80 lbs] (recommandations IOM) a correctement éliminé les valeurs cliniquement impossibles sans distordre la distribution réelle.

**Type de graphique** : histogramme par intervalles (bins de 5 lbs) avec courbe KDE superposée, séparant les distributions avant et après traitement.

L'histogramme avec KDE est adapté pour visualiser l'impact d'une transformation sur une variable continue — il révèle à la fois la forme globale et les pics locaux.

**Interprétation** : avant traitement, des valeurs > 80 lbs (parfois > 200 lbs) créaient une longue queue droite — ces valeurs correspondent à des erreurs de saisie ou des codes sentinelles non documentés. Après cap, la distribution suit une forme proche de la normale centrée autour de 28-30 lbs, cohérente avec les recommandations médicales IOM pour une grossesse normale. Aucune valeur médicalement plausible n'a été supprimée.

**Pertinence** : cette visualisation prouve que le preprocessing n'a pas introduit de biais — les valeurs retirées étaient bien des aberrations et non des cas médicaux légitimes.

---

## 4. Visualisations des performances des modèles

### 4.1 Évaluation complète par modèle — Matrice de confusion, courbe PR, ROC (`model1_*.png`, `model2_*.png`, `model3_*.png`)

**Objectif** : évaluer chaque modèle sur trois dimensions complémentaires — qualité des prédictions binaires (confusion), discrimination probabiliste (PR curve, ROC) — pour avoir une vision exhaustive des forces et faiblesses de chaque modèle.

**Type de graphique** : triptyque (3 panneaux côte à côte) :
- **Matrice de confusion** : heatmap 2×2 annotée avec VN/FP/FN/VP, mise en évidence en rouge de la case FN (erreur prioritaire)
- **Courbe Précision-Rappel** : avec Average Precision (AP) et ligne de référence aléatoire
- **Courbe ROC** : avec AUC et diagonale aléatoire

Le triptyque permet de ne pas réduire l'évaluation à une seule métrique scalaire. La matrice de confusion contextualise les erreurs en termes médicaux concrets (nombre réel de bébés non détectés).

**Interprétation pour le Random Forest (modèle retenu)** :

- **Matrice de confusion** : sur 1 780 bébés à risque, **1 363 sont détectés (VP)**, 417 sont manqués (FN, case rouge). Les FP s'élèvent à ~9 400 — soit ~8 faux positifs par vrai positif.
- **Courbe PR** : AP significativement au-dessus de la baseline aléatoire (9%), montrant que le modèle discrimine bien malgré le faible signal.
- **Courbe ROC** : AUC > 0.80, confirmant une bonne discrimination globale.

**Pertinence** : la mise en évidence visuelle de la case FN (rouge) ancre l'évaluation dans le contexte médical. Un data scientist qui verrait uniquement accuracy=43% conclurait à un mauvais modèle — la matrice de confusion montre que 43% est le résultat d'un modèle agressif qui détecte 77% des cas à risque.

---

### 4.2 Convergence et importance des hyperparamètres Optuna (`model3_optuna.png`)

**Objectif** : visualiser le processus d'optimisation bayésienne — montrer que l'optimisation converge et identifier les hyperparamètres les plus impactants sur le F-beta β=2.

**Type de graphique** : deux panneaux :
- **Convergence** : graphe trial-par-trial avec valeur F-beta β=2 (gris) et meilleur cumulé (rouge)
- **Importance des hyperparamètres** : diagramme en barres horizontales (importance relative Optuna)

La courbe de convergence est le standard pour valider qu'Optuna a eu suffisamment de trials pour stabiliser vers un optimum. Le bar chart d'importance hyperparamètre oriente les futurs tunings.

**Interprétation** : la courbe de convergence montre une stabilisation après ~15 trials, validant que 25 trials sont suffisants sur ce dataset. L'importance révèle que `learning_rate` et `max_depth` sont les hyperparamètres les plus influents sur le F-beta β=2 — information actionnable pour un futur affinement du modèle.

**Pertinence** : cette visualisation prouve que l'optimisation Optuna n'est pas arbitraire — elle converge et identifie des leviers d'action clairs. C'est la validation méthodologique du choix de l'optimisation bayésienne vs GridSearch.

---

### 4.3 Comparaison finale des trois modèles (`comparaison_modeles.png`)

**Objectif** : comparer les trois modèles sur toutes les métriques clés (recall, F1, précision, accuracy) et identifier le meilleur modèle selon le F-beta β=2.

**Type de graphique** : deux panneaux :
- **Gauche** : diagramme en barres groupées, un groupe par métrique, une barre par modèle (LR/RF/HGB en bleu/vert/orange)
- **Droite** : diagramme en barres simple montrant le F-beta β=2 par modèle, avec valeurs annotées

Les barres groupées permettent de comparer simultanément plusieurs métriques entre modèles — plus lisible qu'un tableau pour identifier des patterns (ex : LR a le recall le plus bas mais la précision la plus haute).

**Interprétation** :

| Modèle | Recall | F1 | F-beta β=2 | Seuil |
|---|---|---|---|---|
| Logistic Regression | 0.354 | 0.185 | 0.259 | 0.10 |
| **Random Forest** ✅ | **0.766** | **0.193** | **0.350** | 0.16 |
| HistGBM + Optuna | 0.698 | 0.190 | 0.337 | 0.46 |

Le RF domine sur le F-beta β=2 (0.350), métrique de sélection. Le HGB obtient un score proche (0.337) avec un seuil plus conservateur (0.46 vs 0.16) — ce qui signifie moins de faux positifs en pratique. LR confirme son rôle de baseline faible, validant l'intérêt des modèles non-linéaires.

**Pertinence** : cette visualisation est la conclusion opérationnelle du projet — elle synthétise en un seul graphique l'ensemble de la démarche d'évaluation et justifie la sélection du Random Forest comme modèle final.

---

### 4.4 LazyPredict — Benchmark de 25 classifieurs (`lazypredict_benchmark.png`)

**Objectif** : valider *a posteriori* le choix des trois familles de modèles (LR, RF, boosting) en comparant leurs performances à ~25 classifieurs sklearn en quelques minutes, sans tuning approfondi.

**Type de graphique** : deux diagrammes en barres horizontales côte à côte — top 10 par recall (gauche) et top 10 par F1-score (droite), sur un sous-échantillon stratifié de 8 000 observations post-SMOTE.

**Interprétation** : les résultats LazyPredict (sans SMOTE ni threshold tuning, seuil par défaut 0.5) montrent que **RF est confirmé parmi les meilleurs classifieurs non-linéaires** (top 5 par recall). Le DummyClassifier arrive en tête — artefact dû au fait que le "recall" LazyPredict correspond à l'accuracy au seuil 0.5 (91% = classe majoritaire). SVC (#2) est écarté : ne scale pas sur 145k observations post-SMOTE. BaggingClassifier et ExtraTreesClassifier (#3–4) sont des variantes inférieures de RF. HistGBM n'apparaît pas dans le top 10 sur ce benchmark — son choix est validé par ses résultats finaux (recall=0.698, F-beta=0.337) et la progression méthodologique LR → RF → HGB.

**Pertinence** : LazyPredict confirme que les familles non-linéaires surpassent les modèles linéaires, validant le rejet de LR comme modèle unique. C'est un filet de sécurité méthodologique — pas un outil de sélection finale.

---

## 5. Justification globale de la pertinence des visualisations

Les visualisations ont été choisies pour former une **narration cohérente du pipeline ML** :

```
Données brutes              → EDA : comprendre le problème
    ↓ (signal faible détecté)
Feature engineering         → Preprocessing : transformer et valider
    ↓ (PCA justifiée)
Modélisation                → Modeling : évaluer et comparer
    ↓ (RF retenu)
Conclusion opérationnelle   → Comparaison : décision finale
```

Chaque visualisation répond à **une question précise** dans cette narration :

| Question | Visualisation |
|---|---|
| Y a-t-il des données manquantes ? | `0_missing_values.png` |
| Les classes sont-elles équilibrées ? | `1_distribution_target.png` |
| Les features de base discriminent-elles ? | `2_distributions_continues.png` + `5_correlations.png` + `6_multivariate.png` |
| Le tabac est-il un signal fort ? | `3_tabac_vs_risque.png` |
| Quelles nouvelles features apportent du signal ? | `4_categoriques_vs_risque.png` + `4b_nouvelles_vars_medicales.png` |
| Combien de composantes PCA retenir ? | `preprocessing_pca_variance.png` |
| Le traitement des outliers est-il valide ? | `poids_bins.png` |
| Quels modèles valent la peine d'être tunés ? | `lazypredict_benchmark.png` |
| Quel modèle commet le moins d'erreurs graves ? | `model*_*.png` (matrices de confusion) |
| L'optimisation Optuna a-t-elle convergé ? | `model3_optuna.png` |
| Quel modèle retenir au final ? | `comparaison_modeles.png` |

---

## 6. Notebooks — Localisation et génération des visualisations

### Localisation

| Notebook | Chemin | Visualisations générées |
|---|---|---|
| EDA | `notebooks/EDA.ipynb` | `plots/0_*` à `plots/6_*`, `plots/4b_nouvelles_vars_medicales.png`, `plots/poids_bins.png` |
| Preprocessing | `notebooks/Preprocessing.ipynb` | `plots/preprocessing_pca_variance.png` |
| Modeling | `notebooks/Modeling.ipynb` | `plots/model*`, `plots/comparaison_modeles.png`, `plots/lazypredict_benchmark.png` |

### Ordre d'exécution

```bash
# 1. Preprocessing (génère les données traitées)
jupyter notebook notebooks/Preprocessing.ipynb
# → Kernel → Restart & Run All (~2 min)

# 2. EDA (lecture des données brutes + traitées)
jupyter notebook notebooks/EDA.ipynb
# → Kernel → Restart & Run All (~3 min)

# 3. Modeling (génère les plots de performance)
jupyter notebook notebooks/Modeling.ipynb
# → Kernel → Restart & Run All (~20-25 min, dont ~15 min Optuna)
```

Toutes les visualisations sont sauvegardées automatiquement en PNG (300 dpi) dans le dossier `plots/` à chaque exécution — elles écrasent les versions précédentes, garantissant la cohérence entre résultats et graphiques.
