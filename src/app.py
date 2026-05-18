"""Streamlit dashboard — Prédiction du Risque NICU."""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from config import PLOTS_DIR, MODELS_DIR, PROJECT_ROOT

DATA_PATH    = PROJECT_ROOT / "data" / "process" / "processed_data_full.csv"
THRESHOLD_RF = 0.16

# ── Palette pastel santé / maternité ──────────────────────────────────────────
ROSE      = "#E8A5B0"
BLUE      = "#A5C8E8"
MINT      = "#A5D9C5"
PEACH     = "#F5C8A5"
LAVENDER  = "#C8B5E8"
PLUM      = "#5C4B6B"
SLATE     = "#7A8B99"
BG        = "#FAF8F5"
CARD      = "#FFFFFF"
BORDER    = "#EDE8E3"
DANGER    = "#E57373"
WARNING_C = "#FFB74D"
SUCCESS   = "#81C784"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=PLUM),
    margin=dict(t=40, b=20, l=20, r=20),
)

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

[data-testid="stAppViewContainer"] { background-color: #FAF8F5; font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"]          { background-color: #F5EFF8; border-right: 1px solid #EDE8E3; }
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif; }

h1 { color: #5C4B6B !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h2 { color: #6B5B78 !important; font-weight: 600 !important; }
h3 { color: #7A6B88 !important; font-weight: 600 !important; }
p, li { color: #4A4058; line-height: 1.7; }

.hero {
    background: linear-gradient(135deg, #F5EEF8 0%, #EEF4FA 55%, #EBF8F3 100%);
    border-radius: 20px;
    padding: 36px 44px;
    margin-bottom: 28px;
    border: 1px solid #EDE8E3;
}
.hero h1 { font-size: 2rem; margin-bottom: 10px; }
.hero p  { font-size: 1.05rem; color: #6B5B78; max-width: 700px; }

.kpi-row { display: flex; gap: 16px; margin: 24px 0; flex-wrap: wrap; }
.kpi {
    flex: 1;
    min-width: 140px;
    background: white;
    border-radius: 16px;
    padding: 20px 24px;
    border: 1px solid #EDE8E3;
    box-shadow: 0 2px 10px rgba(92,75,107,0.06);
    text-align: center;
}
.kpi-val  { font-size: 1.9rem; font-weight: 700; color: #5C4B6B; }
.kpi-lbl  { font-size: 0.78rem; color: #7A8B99; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-delta { font-size: 0.82rem; font-weight: 600; margin-top: 6px; }
.kpi-delta.up   { color: #C0392B; }
.kpi-delta.down { color: #27AE60; }

.info-box {
    background: #F8F4FC;
    border-left: 4px solid #C8B5E8;
    border-radius: 0 12px 12px 0;
    padding: 14px 20px;
    margin: 16px 0;
    color: #5C4B6B;
    font-size: 0.95rem;
}

.model-grid { display: flex; gap: 16px; margin: 24px 0; flex-wrap: wrap; }
.model-card {
    flex: 1; min-width: 220px;
    background: white;
    border-radius: 16px;
    padding: 24px;
    border: 1px solid #EDE8E3;
    box-shadow: 0 2px 8px rgba(92,75,107,0.05);
}
.model-card.best     { border-top: 4px solid #A5D9C5; }
.model-card.baseline { border-top: 4px solid #C8B5E8; }
.model-card.boost    { border-top: 4px solid #F5C8A5; }
.model-title { font-size: 1rem; font-weight: 700; color: #5C4B6B; margin-bottom: 8px; }
.model-desc  { font-size: 0.84rem; color: #7A8B99; line-height: 1.6; }
.model-metric { margin-top: 14px; }
.model-metric span.val { font-size: 1.5rem; font-weight: 700; color: #5C4B6B; }
.model-metric span.lbl { font-size: 0.78rem; color: #A0909E; display: block; }

.factor-row {
    display: flex; align-items: center; gap: 12px;
    background: white; border-radius: 12px;
    padding: 12px 16px; margin: 8px 0;
    border: 1px solid #EDE8E3;
    box-shadow: 0 1px 4px rgba(92,75,107,0.04);
}
.factor-dot {
    width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0;
}
.factor-name { font-weight: 600; color: #4A4058; font-size: 0.92rem; }
.factor-rate { font-size: 0.82rem; color: #7A8B99; }

.result-box {
    border-radius: 16px; padding: 24px 28px; margin: 16px 0;
    border: 1px solid;
}
.result-box.high { background: #FDECEA; border-color: #FFCDD2; }
.result-box.low  { background: #E8F5E9; border-color: #C8E6C9; }
.result-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 8px; }
.result-box.high .result-title { color: #B71C1C; }
.result-box.low  .result-title { color: #1B5E20; }

.formula-box {
    background: white;
    border: 1px solid #EDE8E3;
    border-radius: 14px;
    padding: 20px 28px;
    margin: 16px 0;
    text-align: center;
}

hr.soft { border: none; border-top: 1px solid #EDE8E3; margin: 28px 0; }

[data-testid="metric-container"] {
    background: white !important;
    border: 1px solid #EDE8E3 !important;
    border-radius: 12px !important;
    padding: 14px !important;
}
</style>
"""


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement des modèles…")
def load_resources():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    model_rf     = joblib.load(MODELS_DIR / "model2_random_forest.pkl")
    df = pd.read_csv(DATA_PATH)
    X  = df.drop(columns=["Target_Risk"])
    y  = df["Target_Risk"]
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pca = PCA(n_components=10, random_state=42)
    pca.fit(X_train.values)
    return preprocessor, model_rf, pca


@st.cache_data(show_spinner=False)
def load_class_counts():
    df = pd.read_csv(DATA_PATH)
    counts = df["Target_Risk"].value_counts()
    return int(counts.get(0, 0)), int(counts.get(1, 0))


# ── Chart builders ────────────────────────────────────────────────────────────
def donut_chart(n_neg: int, n_pos: int) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Pas d'admission (0)", "Admission NICU (1)"],
        values=[n_neg, n_pos],
        hole=0.58,
        marker_colors=[BLUE, ROSE],
        textinfo="percent",
        textfont=dict(size=13, color="white"),
        hovertemplate="<b>%{label}</b><br>%{value:,} naissances<br>%{percent}<extra></extra>",
    ))
    total = n_neg + n_pos
    fig.add_annotation(
        text=f"<b>{total:,}</b><br>naissances",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=PLUM),
        align="center",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=320,
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"))
    return fig


def nicu_rates_chart() -> go.Figure:
    labels = [
        "Base (population)", "Diabète gestationnel", "ATCD mort fœtale",
        "HTA gestationnelle", "ATCD accouchement prématuré",
        "Traitement infertilité", "PMA", "Éclampsie",
        "Grossesse multiple (jumeaux)", "Grossesse multiple (triplets+)",
    ]
    rates = [8.9, 12.3, 16.1, 16.8, 19.2, 21.3, 23.9, 27.1, 37.5, 87.5]
    colors = [SLATE] + [LAVENDER, LAVENDER, PEACH, PEACH, PEACH, PEACH, ROSE, ROSE, DANGER]

    fig = go.Figure(go.Bar(
        x=rates, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{r}%" for r in rates],
        textposition="outside",
        textfont=dict(size=11, color=PLUM),
        hovertemplate="<b>%{y}</b><br>Taux NICU : %{x}%<extra></extra>",
    ))
    fig.add_vline(x=8.9, line_dash="dot", line_color=SLATE, line_width=1.5,
                  annotation_text="Base 8.9%", annotation_font_color=SLATE,
                  annotation_position="top right")
    fig.update_layout(**PLOTLY_LAYOUT, height=420, xaxis_title="Taux d'admission NICU (%)",
                      xaxis=dict(range=[0, 100], gridcolor="#F0ECE8"),
                      yaxis=dict(tickfont=dict(size=11)))
    return fig


def model_comparison_chart() -> go.Figure:
    models = ["Logistic\nRegression", "Random\nForest ✓", "HistGBM\n+ Optuna"]
    recall  = [0.354, 0.766, 0.698]
    fbeta   = [0.259, 0.350, 0.337]
    f1      = [0.185, 0.193, 0.190]

    fig = go.Figure()
    for vals, name, color in [
        (recall, "Recall", ROSE),
        (fbeta,  "F-beta β=2", MINT),
        (f1,     "F1-Score", PEACH),
    ]:
        fig.add_trace(go.Bar(
            name=name, x=models, y=vals,
            marker_color=color,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(size=11),
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT, height=380,
        barmode="group",
        yaxis=dict(range=[0, 0.95], gridcolor="#F0ECE8", tickformat=".2f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        bargap=0.2, bargroupgap=0.08,
    )
    return fig


def gauge_chart(proba: float) -> go.Figure:
    pct = proba * 100
    color = DANGER if proba >= THRESHOLD_RF else SUCCESS

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number=dict(suffix="%", font=dict(size=36, color=color)),
        delta=dict(reference=THRESHOLD_RF * 100, suffix="% vs seuil",
                   font=dict(size=13), increasing_color=DANGER, decreasing_color=SUCCESS),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=BORDER,
                      tickfont=dict(size=10, color=SLATE)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="white",
            borderwidth=0,
            steps=[
                dict(range=[0, THRESHOLD_RF * 100], color="#E8F5E9"),
                dict(range=[THRESHOLD_RF * 100, 100], color="#FDECEA"),
            ],
            threshold=dict(
                line=dict(color=PLUM, width=3),
                thickness=0.85,
                value=THRESHOLD_RF * 100,
            ),
        ),
        title=dict(text="Probabilité d'admission NICU", font=dict(size=14, color=PLUM)),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=280)
    return fig


# ── Pages ─────────────────────────────────────────────────────────────────────
def page_eda():
    n_neg, n_pos = load_class_counts()
    total = n_neg + n_pos
    rate  = n_pos / total * 100

    st.markdown("""
    <div class="hero">
      <h1>🩺 Prédiction du Risque d'Admission en NICU</h1>
      <p>Un modèle de machine learning entraîné sur <strong>99 900 naissances</strong>
      (CDC Natality 2018) pour identifier, dès la grossesse, les nouveau-nés
      susceptibles d'être admis en Unité de Soins Intensifs Néonatals.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi">
        <div class="kpi-val">{rate:.1f}%</div>
        <div class="kpi-lbl">Taux NICU base</div>
        <div class="kpi-delta up">1 bébé sur 11</div>
      </div>
      <div class="kpi">
        <div class="kpi-val">{n_pos:,}</div>
        <div class="kpi-lbl">Admissions NICU</div>
        <div class="kpi-delta">sur {total:,} naissances</div>
      </div>
      <div class="kpi">
        <div class="kpi-val">×10</div>
        <div class="kpi-lbl">Risque — grossesse triple</div>
        <div class="kpi-delta up">87.5% d'admission</div>
      </div>
      <div class="kpi">
        <div class="kpi-val">19</div>
        <div class="kpi-lbl">Variables retenues</div>
        <div class="kpi-delta down">sur 200+ disponibles</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="info-box">💡 <strong>Application business :</strong> '
                'concentrer les ressources de prévention prénatale sur les mères à risque élevé '
                '— réduire les hospitalisations longues et les coûts associés (3 000 à 100 000 USD/séjour).</div>',
                unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.subheader("Déséquilibre des classes")
        st.plotly_chart(donut_chart(n_neg, n_pos), use_container_width=True)
        st.markdown(
            f"Le dataset est **fortement déséquilibré** : seulement {rate:.1f}% de cas positifs. "
            "Le modèle doit être spécifiquement adapté pour ne pas ignorer la classe minoritaire."
        )
    with col2:
        st.subheader("Taux NICU par facteur de risque")
        st.plotly_chart(nicu_rates_chart(), use_container_width=True)
        st.markdown(
            "Les **grossesses multiples** et les **complications gestationnelles** sont de loin "
            "les signaux les plus forts. Les variables de base (âge, IMC) ont une corrélation < 0.05."
        )

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    st.subheader("Distributions et corrélations")
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(str(PLOTS_DIR / "2_distributions_continues.png"), use_container_width=True,
                 caption="Distributions des variables numériques — outliers médicaux légitimes traités par RobustScaler")
    with col_b:
        st.image(str(PLOTS_DIR / "5_correlations.png"), use_container_width=True,
                 caption="Matrice de corrélations — signal faible pour les variables de base, fort pour les variables médicales")


def page_models():
    st.title("Méthodologie & Comparaison des Modèles")

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    st.subheader("Pourquoi le Recall ne suffit pas")
    st.markdown("""
    Un modèle qui prédit **tout positif** obtiendrait un Recall de 100% — inutilisable en pratique.
    La métrique **F-beta β=2** équilibre précision et recall en pénalisant 2× plus les Faux Négatifs
    (bébés à risque non détectés) que les Faux Positifs (suivis inutiles).
    """)

    st.markdown("""
    <div class="formula-box">
      <p style="color:#7A8B99;font-size:0.85rem;margin-bottom:6px;">MÉTRIQUE PRINCIPALE</p>
      <p style="font-size:1.15rem;color:#5C4B6B;font-weight:600;margin:0;">
        F<sub>β=2</sub> &nbsp;=&nbsp; 5 · Précision · Recall &nbsp;/&nbsp; (4 · Précision + Recall)
      </p>
      <p style="font-size:0.82rem;color:#A0909E;margin-top:8px;">
        β=2 → le Recall est pondéré 2× plus que la Précision
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    st.subheader("Les trois modèles")

    st.markdown("""
    <div class="model-grid">
      <div class="model-card baseline">
        <div class="model-title">1. Régression Logistique &nbsp;<span style="font-size:0.75rem;color:#A0909E;font-weight:400;">Baseline</span></div>
        <div class="model-desc">
          Frontière de décision linéaire dans l'espace PCA(10).
          Sans SMOTE ni class_weight — le déséquilibre est géré uniquement
          par le seuil de classification (optimisé à 0.10).
        </div>
        <div class="model-metric">
          <span class="val">0.354</span>
          <span class="lbl">Recall · F-beta β=2 : 0.259 · Seuil : 0.10</span>
        </div>
      </div>
      <div class="model-card best">
        <div class="model-title">2. Random Forest &nbsp;<span style="background:#E8F5E9;color:#2E7D32;padding:2px 8px;border-radius:10px;font-size:0.72rem;">✓ Retenu</span></div>
        <div class="model-desc">
          100 arbres de décision parallèles entraînés sur données augmentées (SMOTE).
          Non-linéaire, robuste aux outliers médicaux. PCA(10) réduit la dimensionnalité
          de 36 → 10 composantes avant l'entraînement.
        </div>
        <div class="model-metric">
          <span class="val">0.766</span>
          <span class="lbl">Recall · F-beta β=2 : 0.350 · Seuil : 0.16</span>
        </div>
      </div>
      <div class="model-card boost">
        <div class="model-title">3. HistGradientBoosting + Optuna</div>
        <div class="model-desc">
          Gradient boosting séquentiel optimisé par Optuna (25 trials, 3-fold CV).
          L'objectif d'optimisation est directement F-beta β=2 — pas le recall pur.
          Seuil conservateur (0.46) moins adapté au contexte médical.
        </div>
        <div class="model-metric">
          <span class="val">0.698</span>
          <span class="lbl">Recall · F-beta β=2 : 0.337 · Seuil : 0.46</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Comparaison des métriques")
        st.plotly_chart(model_comparison_chart(), use_container_width=True)
    with col2:
        st.subheader("Protocole d'évaluation")
        st.markdown("""
        **Split :** 80% train / 20% test, stratifié (seed 42)

        **SMOTE :** appliqué uniquement pour RF et HGB, jamais sur le test set

        **PCA(10) :** fitté sur X_train original (avant SMOTE) pour éviter
        que les exemples synthétiques influencent l'espace latent

        **Optuna :** 25 trials, 3-fold StratifiedKFold, pipeline SMOTE→PCA→HGB
        à l'intérieur de chaque fold

        **Seuil :** optimisé par grid search sur F-beta β=2
        """)
        st.markdown("""
        <div class="info-box">
        🎯 Le Random Forest obtient le meilleur F-beta β=2 = <strong>0.350</strong>
        et détecte <strong>76.6% des bébés à risque</strong>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    st.subheader("Performance détaillée — Random Forest (modèle retenu)")
    st.image(str(PLOTS_DIR / "model2_random_forest.png"), use_container_width=True,
             caption="Matrice de confusion · Courbe Précision-Rappel · Courbe ROC")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Convergence Optuna")
        st.image(str(PLOTS_DIR / "model3_optuna.png"), use_container_width=True,
                 caption="Évolution du F-beta β=2 sur 25 trials")
    with col4:
        st.subheader("Benchmark — 25 classifieurs (LazyPredict)")
        st.image(str(PLOTS_DIR / "lazypredict_benchmark.png"), use_container_width=True,
                 caption="Scores sur sous-échantillon stratifié de 8 000 lignes")


def page_demo(preprocessor, model_rf, pca):
    st.title("Évaluation du Risque NICU")
    st.markdown(
        "Renseignez le profil clinique dans la barre latérale. "
        "Le modèle **Random Forest** (Recall 76.6%, seuil 0.16) estime la probabilité "
        "d'admission en Unité de Soins Intensifs Néonatals.",
    )
    st.markdown(
        '<div class="info-box">⚕️ Cet outil est à vocation <strong>éducative</strong>. '
        'Il ne remplace pas un avis médical ni un diagnostic clinique.</div>',
        unsafe_allow_html=True,
    )

    # ── Inputs ────────────────────────────────────────────────────────────────
    st.sidebar.header("Profil de la mère")

    age         = st.sidebar.slider("Âge (ans)", 15, 50, 28)
    imc         = st.sidebar.number_input("IMC avant grossesse", 15.0, 70.0, 25.5, 0.5)
    prise_poids = st.sidebar.slider("Prise de poids (lbs)", 0, 80, 25)
    mois_suivi  = st.sidebar.slider("Début suivi prénatal (mois)", 1, 10, 2,
                                     help="1 = dès le 1er mois, 10 = tardif")

    st.sidebar.subheader("Tabac")
    tabac_avant = st.sidebar.slider("Cigarettes/jour avant grossesse", 0, 40, 0)
    tabac_t1    = st.sidebar.slider("Cigarettes/jour — 1er trimestre", 0, 40, 0)
    tabac_t3    = st.sidebar.slider("Cigarettes/jour — 3e trimestre", 0, 40, 0)

    st.sidebar.subheader("Grossesse & Antécédents")
    gross_mult  = st.sidebar.selectbox("Type de grossesse", ["Simple", "Multiple (jumeaux+)"])
    atcd_mort   = st.sidebar.selectbox("Antécédent mort fœtale", ["Non", "Oui"])
    atcd_prem   = st.sidebar.selectbox("Antécédent accouchement prématuré", ["N", "Y", "U"])

    st.sidebar.subheader("Conditions médicales")
    aide_wic    = st.sidebar.selectbox("Aide alimentaire WIC", ["N", "Y", "U"])
    d_chron     = st.sidebar.selectbox("Diabète chronique", ["N", "Y", "U"])
    h_chron     = st.sidebar.selectbox("Hypertension chronique", ["N", "Y", "U"])
    assur_map   = {"Medicaid": 1.0, "Privée": 2.0, "CHIP": 3.0,
                   "Autre gouvernement": 4.0, "Sans assurance": 5.0,
                   "Indian Health Service": 6.0, "Autre": 8.0}
    assurance   = assur_map[st.sidebar.selectbox("Assurance", list(assur_map.keys()))]
    pma         = st.sidebar.selectbox("PMA (procréation médicale)", ["N", "Y", "U", "X"])
    trt_infert  = st.sidebar.selectbox("Traitement infertilité", ["N", "Y", "U"])

    st.sidebar.subheader("Complications gestationnelles")
    eclampsie   = st.sidebar.selectbox("Éclampsie / pré-éclampsie", ["N", "Y", "U"])
    hta_gest    = st.sidebar.selectbox("HTA gestationnelle", ["N", "Y", "U"])
    diab_gest   = st.sidebar.selectbox("Diabète gestationnel", ["N", "Y", "U"])

    if st.sidebar.button("Évaluer le risque", type="primary", use_container_width=True):

        evol_tabac  = tabac_t1 - tabac_avant
        suivi_t1    = 1 if mois_suivi <= 3 else 0
        gross_v     = 1 if gross_mult == "Multiple (jumeaux+)" else 0
        mort_v      = 1 if atcd_mort == "Oui" else 0

        raw = pd.DataFrame([{
            "Age_Mere": age, "Mois_Debut_Suivi": mois_suivi,
            "Tabac_Avant": tabac_avant, "Tabac_Trim1": tabac_t1, "Tabac_Trim3": tabac_t3,
            "IMC_Mere": imc, "Prise_Poids": prise_poids,
            "Evol_Tabac": evol_tabac, "Suivi_T1": suivi_t1,
            "Grossesse_Multiple": gross_v, "ATCD_Mort_Foetale_bin": mort_v,
            "Aide_WIC": aide_wic, "D_Chronique": d_chron, "H_Chronique": h_chron,
            "Assurance": assurance, "ATCD_Premature": atcd_prem,
            "PMA": pma, "Traitement_Infertilite": trt_infert,
            "Eclampsie": eclampsie, "HTA_Gestationnelle": hta_gest,
            "Diabete_Gestationnel": diab_gest,
        }])

        X_proc = preprocessor.transform(raw)
        X_pca  = pca.transform(X_proc)
        proba  = model_rf.predict_proba(X_pca)[0, 1]
        pred   = proba >= THRESHOLD_RF

        # ── Résultat ──────────────────────────────────────────────────────────
        col_gauge, col_result = st.columns([1, 1.4])
        with col_gauge:
            st.plotly_chart(gauge_chart(proba), use_container_width=True)

        with col_result:
            if pred:
                st.markdown(f"""
                <div class="result-box high">
                  <div class="result-title">⚠️ Risque élevé d'admission NICU</div>
                  <p>La probabilité estimée (<strong>{proba*100:.1f}%</strong>) dépasse le seuil
                  clinique de {THRESHOLD_RF*100:.0f}%.</p>
                  <p><strong>Recommandation :</strong> intensifier le suivi prénatal
                  et orienter vers un service de médecine fœtale.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box low">
                  <div class="result-title">Risque faible d'admission NICU</div>
                  <p>La probabilité estimée (<strong>{proba*100:.1f}%</strong>) est
                  inférieure au seuil clinique de {THRESHOLD_RF*100:.0f}%.</p>
                  <p>Suivi prénatal standard recommandé.</p>
                </div>
                """, unsafe_allow_html=True)

        # ── Facteurs identifiés ────────────────────────────────────────────────
        st.markdown('<hr class="soft">', unsafe_allow_html=True)
        st.subheader("Facteurs de risque identifiés dans ce profil")

        FACTORS = [
            (gross_v == 1,        "#E57373", "Grossesse multiple",               "Taux NICU : 37–87%"),
            (eclampsie == "Y",    "#E57373", "Éclampsie / pré-éclampsie",         "Taux NICU : 27.1%"),
            (pma == "Y",          "#FFB74D", "PMA",                               "Taux NICU : 23.9%"),
            (trt_infert == "Y",   "#FFB74D", "Traitement infertilité",            "Taux NICU : 21.3%"),
            (atcd_prem == "Y",    "#FFB74D", "Antécédent accouchement prématuré", "Taux NICU : 19.2%"),
            (hta_gest == "Y",     "#FFD54F", "HTA gestationnelle",                "Taux NICU : 16.8%"),
            (mort_v == 1,         "#FFD54F", "Antécédent mort fœtale",            "Taux NICU : 16.1%"),
            (diab_gest == "Y",    "#FFD54F", "Diabète gestationnel",              "Taux NICU : 12.3%"),
            (tabac_t3 > 0,        "#A5D9C5", "Tabagisme au 3e trimestre",         f"{tabac_t3} cig/jour"),
            (evol_tabac > 0,      "#A5D9C5", "Augmentation tabac en grossesse",   f"+{evol_tabac} cig/jour"),
        ]

        active = [(c, n, d) for cond, c, n, d in FACTORS if cond]

        if active:
            html = ""
            for color, name, detail in active:
                html += f"""
                <div class="factor-row">
                  <div class="factor-dot" style="background:{color};"></div>
                  <div>
                    <div class="factor-name">{name}</div>
                    <div class="factor-rate">{detail}</div>
                  </div>
                </div>"""
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="info-box" style="border-left-color:#A5D9C5;">'
                '✓ Aucun facteur de risque majeur identifié dans ce profil.</div>',
                unsafe_allow_html=True,
            )

    else:
        # ── Écran d'accueil démo ───────────────────────────────────────────────
        st.markdown('<hr class="soft">', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Comment utiliser la démo")
            st.markdown("""
            1. **Profil maternel** — âge, IMC, prise de poids
            2. **Suivi prénatal** — mois de début du suivi
            3. **Tabac** — consommation avant et pendant la grossesse
            4. **Antécédents** — type de grossesse, antécédents obstétricaux
            5. **Conditions médicales** — hypertension, diabète, assurance
            6. **Complications** — éclampsie, HTA, diabète gestationnel

            Puis cliquez sur **Évaluer le risque**.
            """)
        with col_b:
            st.markdown("### À propos du modèle")
            st.markdown(f"""
            Le **Random Forest** a été entraîné sur 99 900 naissances américaines
            (CDC Natality 2018, 19 variables cliniques).

            | Métrique | Valeur |
            |---|---|
            | Recall | **76.6%** |
            | F-beta β=2 | **0.350** |
            | Seuil de décision | **{THRESHOLD_RF*100:.0f}%** |
            | Faux positifs évités | ✓ équilibre FN/FP |

            Le modèle détecte environ **3 bébés à risque sur 4**.
            """)


# ── Entry point ────────────────────────────────────────────────────────────────
def build_app() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    preprocessor, model_rf, pca = load_resources()

    st.sidebar.markdown(
        "<div style='text-align:center;padding:16px 0 8px;'>"
        "<span style='font-size:2rem;'>👶</span><br>"
        "<span style='font-weight:700;color:#5C4B6B;font-size:1.05rem;'>NICU Risk</span><br>"
        "<span style='font-size:0.75rem;color:#A0909E;'>Prédiction néonatale</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    menu = st.sidebar.radio(
        "Navigation",
        ["Projet & EDA", "Modèles", "Démo"],
        label_visibility="collapsed",
    )

    if menu == "Projet & EDA":
        page_eda()
    elif menu == "Modèles":
        page_models()
    else:
        page_demo(preprocessor, model_rf, pca)


if __name__ == "__main__":
    build_app()
