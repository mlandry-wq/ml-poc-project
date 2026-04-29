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

A. Les Features :
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