"""Streamlit dashboard — Prédiction du Risque NICU."""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from config import PLOTS_DIR, MODELS_DIR, PROJECT_ROOT

DATA_PATH    = PROJECT_ROOT / "data" / "process" / "processed_data_full.csv"
THRESHOLD_RF = 0.16  # seuil optimisé F-beta β=2


# ── Load resources (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement des modèles…")
def load_resources():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    model_rf     = joblib.load(MODELS_DIR / "model2_random_forest.pkl")

    # Refit PCA sur le même X_train (non sauvegardé depuis le notebook)
    df = pd.read_csv(DATA_PATH)
    X  = df.drop(columns=["Target_Risk"])
    y  = df["Target_Risk"]
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pca = PCA(n_components=10, random_state=42)
    pca.fit(X_train.values)

    return preprocessor, model_rf, pca


def build_app() -> None:
    """Entry point Streamlit — appelé par scripts/main.py."""

    preprocessor, model_rf, pca = load_resources()

    # ── Page config ────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Prédiction Risque NICU",
        page_icon="🏥",
        layout="wide",
    )

    # ── Navigation ─────────────────────────────────────────────────────────────
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio(
        "",
        ["Projet & EDA", "Modèles", "Démo"],
        label_visibility="collapsed",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — PROJET & EDA
    # ═══════════════════════════════════════════════════════════════════════════
    if menu == "Projet & EDA":
        st.title("Prédiction du Risque d'Admission en NICU")

        st.markdown("""
        **Contexte :** Chaque année aux États-Unis, environ **9% des nouveau-nés** sont admis
        en Unité de Soins Intensifs Néonatals (NICU). Cette admission est souvent prévisible
        dès la grossesse, mais les ressources de prévention sont limitées.

        **Objectif :** Développer un modèle de classification binaire capable d'identifier,
        **avant ou pendant la grossesse**, les bébés à risque d'admission en NICU —
        afin de concentrer le suivi prénatal sur les mères les plus vulnérables.
        """)

        st.divider()

        # ── Métriques clés ─────────────────────────────────────────────────────
        st.header("Application Business")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Taux NICU base", "~9%", help="Taux de base dans la population CDC 2018")
            st.markdown("**Coût moyen** d'une hospitalisation NICU : 3 000 à 100 000 USD.")
        with col2:
            st.metric("Jumeaux → NICU", "37.5%", "+28.5 pts")
            st.metric("Triplets → NICU", "87.5%", "+78.5 pts")
        with col3:
            st.metric("Dataset CDC Natality 2018", "99 900 naissances")
            st.markdown("**Cible :** `ab_nicu` — 0 = pas d'admission, 1 = admission NICU.")

        st.markdown("""
        Un **Faux Négatif** (bébé à risque non détecté) est plus coûteux qu'un Faux Positif.
        La métrique principale est le **Recall**, pondéré via **F-beta β=2**.
        """)

        st.divider()

        # ── EDA plots ──────────────────────────────────────────────────────────
        st.header("Analyse Exploratoire des Données (EDA)")

        st.subheader("Distribution de la variable cible")
        st.image(
            str(PLOTS_DIR / "1_distribution_target.png"),
            use_container_width=True,
            caption="Distribution de la cible — déséquilibre ~9% positifs (admission NICU)",
        )

        st.subheader("Variables catégorielles et taux de risque NICU")
        st.image(
            str(PLOTS_DIR / "4_categoriques_vs_risque.png"),
            use_container_width=True,
            caption="Taux NICU par modalité — signal fort identifié sur plusieurs variables",
        )

        st.subheader("Matrice de corrélations")
        st.image(
            str(PLOTS_DIR / "5_correlations.png"),
            use_container_width=True,
            caption="Corrélations entre features et la cible — variables de base < 0.05",
        )

        st.subheader("Distributions continues")
        st.image(
            str(PLOTS_DIR / "2_distributions_continues.png"),
            use_container_width=True,
            caption="Distributions numériques — présence d'outliers médicaux légitimes",
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — MODÈLES
    # ═══════════════════════════════════════════════════════════════════════════
    elif menu == "Modèles":
        st.title("Choix des Modèles et Évaluation")

        # ── Métrique ──────────────────────────────────────────────────────────
        st.header("Métrique : F-beta β=2")
        st.markdown(r"""
        Avec un déséquilibre fort (~9% positifs) et un coût asymétrique FN >> FP :

        $$F_\beta = (1 + \beta^2) \cdot \frac{\text{Précision} \times \text{Recall}}{(\beta^2 \times \text{Précision}) + \text{Recall}}$$

        Avec **β = 2**, le Recall est pondéré **2× plus que la Précision** — un Faux Négatif
        (bébé à risque non détecté) est 2 fois plus pénalisant qu'un Faux Positif.

        Les seuils de décision ont aussi été optimisés sur cette métrique (grid search [0.10–0.90]).
        """)

        st.divider()

        # ── Les 3 modèles ──────────────────────────────────────────────────────
        st.header("Les 3 Modèles Testés")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("1. Régression Logistique")
            st.markdown("""
            **Baseline** — modèle linéaire interprétable.
            - Pas de SMOTE (dégénérescence avec class_weight)
            - Seuil ajusté à 0.10 pour maximiser le recall
            - Hypothèse : séparabilité linéaire dans l'espace PCA
            """)
            st.metric("Recall", "0.354")
            st.metric("F-beta β=2", "0.259")

        with col2:
            st.subheader("2. Random Forest ✅")
            st.markdown("""
            **Modèle retenu** — meilleur F-beta β=2.
            - Entraîné sur données augmentées SMOTE
            - PCA(10) appliqué avant l'entraînement
            - 100 arbres, profondeur libre
            """)
            st.metric("Recall", "0.766", "+41 pts vs LR")
            st.metric("F-beta β=2", "0.350", delta="meilleur")

        with col3:
            st.subheader("3. HistGradientBoosting + Optuna")
            st.markdown("""
            **Modèle optimisé** — hyperparamètres tuné via Optuna (25 trials, 3-fold CV).
            - Objective Optuna : F-beta β=2
            - SMOTE + PCA dans le pipeline CV
            """)
            st.metric("Recall", "0.698")
            st.metric("F-beta β=2", "0.337")

        st.divider()

        # ── Tableau ────────────────────────────────────────────────────────────
        st.header("Comparaison des Performances")
        results_df = pd.DataFrame({
            "Modèle": ["Logistic Regression (baseline)", "Random Forest ✅", "HistGBM + Optuna"],
            "Recall": [0.354, 0.766, 0.698],
            "F1-Score": [0.185, 0.193, 0.190],
            "F-beta β=2": [0.259, 0.350, 0.337],
            "Seuil optimal": [0.10, 0.16, 0.46],
            "Gestion déséquilibre": ["Seuil seul", "SMOTE + Seuil", "SMOTE + Seuil (CV)"],
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Plots ──────────────────────────────────────────────────────────────
        st.header("Graphiques de Performance")

        st.subheader("Comparaison finale des modèles")
        st.image(
            str(PLOTS_DIR / "comparaison_modeles.png"),
            use_container_width=True,
            caption="Recall, F1, F-beta β=2 et AUC-ROC pour les 3 modèles",
        )

        st.subheader("Meilleur modèle — Random Forest (triptyque)")
        st.image(
            str(PLOTS_DIR / "model2_random_forest.png"),
            use_container_width=True,
            caption="Matrice de confusion, courbe Précision-Rappel et ROC — Random Forest",
        )

        st.subheader("Convergence Optuna (HGB)")
        st.image(
            str(PLOTS_DIR / "model3_optuna.png"),
            use_container_width=True,
            caption="Historique des 25 trials Optuna — optimisation F-beta β=2",
        )

        st.subheader("Benchmark LazyPredict (25 classifieurs)")
        st.image(
            str(PLOTS_DIR / "lazypredict_benchmark.png"),
            use_container_width=True,
            caption="Comparaison rapide de 25 algorithmes sur sous-échantillon stratifié",
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — DÉMO
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        st.title("Démo — Évaluation du Risque NICU")
        st.markdown("""
        Renseignez le profil clinique de la mère pour obtenir une **estimation du risque
        d'admission du nouveau-né en NICU**. Modèle utilisé : **Random Forest**
        (Recall = 76.6%, F-beta β=2 = 0.350, seuil = 0.16).

        > Ce prototype est à vocation éducative. Il ne remplace pas un avis médical.
        """)

        # ── Inputs sidebar ─────────────────────────────────────────────────────
        st.sidebar.header("Profil de la mère")

        age         = st.sidebar.slider("Âge de la mère (ans)", 15, 50, 28)
        imc         = st.sidebar.number_input("IMC avant grossesse", min_value=15.0, max_value=70.0, value=25.5, step=0.5)
        prise_poids = st.sidebar.slider("Prise de poids durant la grossesse (lbs)", 0, 80, 25)
        mois_suivi  = st.sidebar.slider("Mois de début du suivi prénatal", 1, 10, 2,
                                         help="1 = premier mois, 10 = tardif")

        st.sidebar.subheader("Tabac")
        tabac_avant = st.sidebar.slider("Cigarettes/jour avant la grossesse", 0, 40, 0)
        tabac_t1    = st.sidebar.slider("Cigarettes/jour au 1er trimestre", 0, 40, 0)
        tabac_t3    = st.sidebar.slider("Cigarettes/jour au 3e trimestre", 0, 40, 0)

        st.sidebar.subheader("Antécédents & Grossesse")
        grossesse_mult = st.sidebar.selectbox("Type de grossesse", ["Grossesse simple", "Grossesse multiple (jumeaux+)"])
        atcd_mort_foet = st.sidebar.selectbox("Antécédent de mort fœtale", ["Non", "Oui"])

        st.sidebar.subheader("Conditions médicales")
        aide_wic    = st.sidebar.selectbox("Bénéficiaire aide WIC", ["N", "Y", "U"])
        d_chronique = st.sidebar.selectbox("Diabète chronique préexistant", ["N", "Y", "U"])
        h_chronique = st.sidebar.selectbox("Hypertension chronique préexistante", ["N", "Y", "U"])
        assurance_map = {
            "Medicaid (1)": 1.0, "Privée (2)": 2.0, "CHIP (3)": 3.0,
            "Autre gouvernement (4)": 4.0, "Sans assurance (5)": 5.0,
            "Indian Health Service (6)": 6.0, "Autre (8)": 8.0,
        }
        assurance_label = st.sidebar.selectbox("Type d'assurance", list(assurance_map.keys()))
        assurance_val   = assurance_map[assurance_label]

        atcd_premature = st.sidebar.selectbox("Antécédent d'accouchement prématuré", ["N", "Y", "U"])
        pma            = st.sidebar.selectbox("Procréation médicalement assistée (PMA)", ["N", "Y", "U", "X"])
        trt_infert     = st.sidebar.selectbox("Traitement de l'infertilité", ["N", "Y", "U"])

        st.sidebar.subheader("Complications gestationnelles")
        eclampsie = st.sidebar.selectbox("Éclampsie / pré-éclampsie sévère", ["N", "Y", "U"])
        hta_gest  = st.sidebar.selectbox("HTA gestationnelle", ["N", "Y", "U"])
        diab_gest = st.sidebar.selectbox("Diabète gestationnel", ["N", "Y", "U"])

        # ── Bouton prédire ─────────────────────────────────────────────────────
        if st.sidebar.button("Prédire le risque", type="primary"):

            evol_tabac    = tabac_t1 - tabac_avant
            suivi_t1      = 1 if mois_suivi <= 3 else 0
            grossesse_v   = 1 if grossesse_mult == "Grossesse multiple (jumeaux+)" else 0
            atcd_mort_v   = 1 if atcd_mort_foet == "Oui" else 0

            raw = pd.DataFrame([{
                "Age_Mere":              age,
                "Mois_Debut_Suivi":      mois_suivi,
                "Tabac_Avant":           tabac_avant,
                "Tabac_Trim1":           tabac_t1,
                "Tabac_Trim3":           tabac_t3,
                "IMC_Mere":              imc,
                "Prise_Poids":           prise_poids,
                "Evol_Tabac":            evol_tabac,
                "Suivi_T1":              suivi_t1,
                "Grossesse_Multiple":    grossesse_v,
                "ATCD_Mort_Foetale_bin": atcd_mort_v,
                "Aide_WIC":              aide_wic,
                "D_Chronique":           d_chronique,
                "H_Chronique":           h_chronique,
                "Assurance":             assurance_val,
                "ATCD_Premature":        atcd_premature,
                "PMA":                   pma,
                "Traitement_Infertilite": trt_infert,
                "Eclampsie":             eclampsie,
                "HTA_Gestationnelle":    hta_gest,
                "Diabete_Gestationnel":  diab_gest,
            }])

            X_proc = preprocessor.transform(raw)
            X_pca  = pca.transform(X_proc)
            proba  = model_rf.predict_proba(X_pca)[0, 1]
            pred   = int(proba >= THRESHOLD_RF)

            st.divider()
            col_res, col_gauge = st.columns([2, 1])

            with col_res:
                if pred == 1:
                    st.error("⚠️ Risque NICU **ÉLEVÉ** détecté")
                    st.markdown(f"""
                    Probabilité estimée : **{proba*100:.1f}%** (seuil : {THRESHOLD_RF*100:.0f}%).

                    **Recommandation :** Intensifier le suivi prénatal et référer vers un service
                    spécialisé en médecine fœtale si ce n'est pas déjà le cas.
                    """)
                else:
                    st.success("Risque NICU **FAIBLE**")
                    st.markdown(f"""
                    Probabilité estimée : **{proba*100:.1f}%** (seuil : {THRESHOLD_RF*100:.0f}%).

                    Suivi prénatal standard recommandé.
                    """)

            with col_gauge:
                st.metric("Probabilité estimée", f"{proba*100:.1f}%")
                st.metric("Seuil de décision", f"{THRESHOLD_RF*100:.0f}%")
                st.metric("Décision", "NICU" if pred == 1 else "Standard")

            st.subheader("Facteurs de risque identifiés")
            factors = []
            if grossesse_v:
                factors.append(("🔴", "Grossesse multiple", "Taux NICU : 37–87%"))
            if eclampsie == "Y":
                factors.append(("🔴", "Éclampsie / pré-éclampsie", "Taux NICU : 27.1%"))
            if pma == "Y":
                factors.append(("🟠", "PMA (procréation médicale assistée)", "Taux NICU : 23.9%"))
            if trt_infert == "Y":
                factors.append(("🟠", "Traitement infertilité", "Taux NICU : 21.3%"))
            if atcd_premature == "Y":
                factors.append(("🟠", "Antécédent d'accouchement prématuré", "Taux NICU : 19.2%"))
            if hta_gest == "Y":
                factors.append(("🟡", "HTA gestationnelle", "Taux NICU : 16.8%"))
            if atcd_mort_v:
                factors.append(("🟡", "Antécédent de mort fœtale", "Taux NICU : 16.1%"))
            if diab_gest == "Y":
                factors.append(("🟡", "Diabète gestationnel", "Taux NICU : 12.3%"))
            if tabac_t3 > 0:
                factors.append(("🟡", "Tabagisme au 3e trimestre", f"{tabac_t3} cigarettes/jour"))
            if evol_tabac > 0:
                factors.append(("🟡", "Augmentation du tabagisme", f"+{evol_tabac} cig/jour"))

            if factors:
                for emoji, label, detail in factors:
                    st.markdown(f"{emoji} **{label}** — {detail}")
            else:
                st.markdown("Aucun facteur de risque majeur identifié dans le profil renseigné.")

        else:
            st.info(
                "Renseignez le profil clinique dans la barre latérale gauche, "
                "puis cliquez sur **Prédire le risque**."
            )
            st.markdown("""
            ### Comment utiliser cette démo ?
            1. **Profil de la mère** : âge, IMC, prise de poids, suivi prénatal
            2. **Tabac** : consommation avant et pendant la grossesse
            3. **Antécédents** : grossesse multiple, mort fœtale antérieure
            4. **Conditions médicales** : hypertension, diabète, assurance
            5. **Complications gestationnelles** : éclampsie, HTA, diabète gestationnel

            Le modèle **Random Forest** détecte **76.6% des bébés à risque**
            avec un seuil de décision optimisé à 0.16.
            """)


if __name__ == "__main__":
    build_app()
