SESSION 1 
1. Description du projet
Développement d'un modèle de ML capable de prédire les risques de complications de santé et/ou de malformations chez le nouveau-né en se basant sur le profil et les comportements de la mère durant la grossesse. 
L'idée est d'utiliser les données de santé publique pour identifier les signaux d'alarme précoces et ainsi intensifier le suivi prénatal et mettre en place des interventions ciblées pour réduire la mortalité.


2. Définition de la problématique (Problématique ML)
Est-il possible d'identifier un risque de complications chez le nouveau né à partir de facteurs comportementaux et cliniques maternels ?

-> type de ML : Classification binaire.

Entrée (X) : Variables comportementales et médicales de la mère.

Sortie (y) : Présence d'une complication ou d'une anomalie (0 ou 1).


3. Description du dataset idéal 
Pas de dataset agrégé mais composé de microdonnées individuelles.

A. Les Features  :
Comportementales :
- Consommation de tabac (nombre de cigarettes par jour avant et pendant chaque trimestre).
- Consommation d'alcool (fréquence et quantité).
- Prise de compléments alimentaires (acide folique/vitamines).
- Qualité du suivi prénatal (mois du premier examen, nombre total de visites).

Physiologiques & Médicales :
- Âge de la mère et IMC pré-grossesse.
- Antécédents de maladies chroniques (Diabète, Hypertension).
- Prise de poids totale durant la grossesse.
- Socio-économiques (Biais importants à surveiller) :
- Niveau d'éducation, type d'assurance santé, zone de résidence (urbaine/rurale).

B. La Target :
Variable combinée ou une spécifique très documentée comme :
- Admission en USIN (Unité de Soins Intensifs Néonataux)
- Score d'APGAR à 5 minutes : Un score inférieur à 7 indique un nouveau-né en difficulté.
- Présence d'anomalies congénitales : (Variable binaire Oui/Non).

C. Volume et Qualité :
- Taille : Au moins 50 000 à 100 000 dossiers individuels pour capturer les cas rares (les malformations ne touchent qu'un petit pourcentage des naissances)
- Diversité : Des données provenant de différentes régions et horizons sociaux pour éviter que le modèle ne soit biaisé.

SESSION 2
A. Description du dataset
Le dataset utilisé s'appuie sur la base de données Natality 2018, issue du CDC (Centers for Disease Control and Prevention) américain. Le fichier original recense la quasi-totalité des naissances aux États-Unis sur une année, avec plus de 3,8 millions d'enregistrements et environ 200 variables par certificat de naissance.
Afin de rendre le traitement optimal et d'isoler le signal du bruit (Feature Selection), le projet se concentre sur un échantillon représentatif de 100 000 lignes. Le périmètre a été restreint à 10 variables explicatives clés (profil physiologique de la mère, habitudes de vie et données socio-économiques) pour prédire une variable cible.

B. Méthode de collecte
Les données primaires sont collectées par le National Center for Health Statistics (NCHS) directement via les certificats de naissance standardisés remplis par les hôpitaux. Pour ce projet, j'ai récupéré l'extraction brute Open Data via la plateforme Kaggle (dataset "US Births 2018") sous format CSV. Le chargement a été optimisé en Python (via Pandas avec ciblage strict des colonnes) pour manipuler efficacement ces microdonnées.

D. Justification du choix
Le choix de ce dataset repose sur deux critères majeurs :
- Qualité et réalisme : Contrairement aux datasets académiques "parfaits", on confronte ici le ML à de vrais enjeux avec des classes fortement déséquilibrées (la majorité des nouveau-nés sont en bonne santé) et des variables soumises au déclaratif (comme le tabagisme).
- Potentiel d'actionnabilité : J'ai volontairement exclu les données liées au déroulement de l'accouchement (pour éviter le data leakage). Le dataset final ne contient que des variables préexistantes à l'accouchement. C'est ce qui permet de construire un modèle réellement utile pour la prévention pendant la grossesse.

E. Objectif business et ML
- Objectif Business : S'inscrire dans une démarche de médecine préventive. En identifiant très tôt le niveau de risque néonatal à partir du profil maternel, les systèmes de santé peuvent mettre en place des interventions ciblées (aide au sevrage tabagique, renforcement du suivi prénatal). L'enjeu est d'améliorer la santé infantile tout en réduisant les coûts considérables liés aux hospitalisations prolongées.
- Objectif Machine Learning : Il s'agit d'un problème de classification binaire. Le modèle doit apprendre les relations entre les variables d'entrée (comportement, âge, IMC, etc.) pour prédire la variable cible : l'admission du nouveau-né en unité de soins intensifs (colonne ab_nicu, modélisée en 0 ou 1).

F. Métrique et objectif d’évaluation envisagés
Dans ce contexte médical, la simple "Accuracy" (précision globale) serait trompeuse à cause du fort déséquilibre des classes. Si le modèle prédit "aucun risque" pour tout le monde, il aura 95% de réussite, mais sera inutile médicalement.
La métrique prioritaire sera donc le Recall. La logique métier l'exige : le coût d'un Faux Négatif (rater un bébé réellement en danger) est infiniment plus grave que celui d'un Faux Positif (surveiller une mère de plus près par précaution).
Le F1-Score et l'AUC-ROC seront également monitorés pour s'assurer que le modèle maintient un bon équilibre de performance globale.