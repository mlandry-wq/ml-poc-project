"""NORA — Neonatal Outcome Risk Assessment · Streamlit dashboard."""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from config import PLOTS_DIR, MODELS_DIR, PROJECT_ROOT

DATA_PATH    = PROJECT_ROOT / "data" / "process" / "processed_data_full.csv"
THRESHOLD_RF = 0.16

# ── Palette ───────────────────────────────────────────────────────────────────
ROSE     = "#E8A5B0"
BLUE     = "#A5C8E8"
MINT     = "#A5D9C5"
PEACH    = "#F5C8A5"
LAVENDER = "#C8B5E8"
PLUM     = "#5C4B6B"
SLATE    = "#7A8B99"
BORDER   = "#EDE8E3"
DANGER   = "#E57373"
SUCCESS  = "#81C784"

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=PLUM),
    margin=dict(t=40, b=20, l=20, r=20),
)

YN_MAP  = {"Non": "N", "Oui": "Y", "Inconnu": "U"}
PMA_MAP = {"Non": "N", "Oui": "Y", "Inconnu": "U", "Non applicable": "X"}
ASSUR_MAP = {
    "Medicaid": 1.0, "Assurance privée": 2.0, "CHIP": 3.0,
    "Autre gouvernement": 4.0, "Sans assurance": 5.0,
    "Indian Health Service": 6.0, "Autre": 8.0,
}

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

[data-testid="stAppViewContainer"] { background:#FAF8F5; font-family:'Inter',sans-serif; }
[data-testid="stSidebar"] { background:#F5EFF8; border-right:1px solid #EDE8E3; }
[data-testid="stSidebar"] * { font-family:'Inter',sans-serif; }
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"] { display:none !important; }

h1 { color:#5C4B6B !important; font-weight:700 !important; letter-spacing:-0.5px; }
h2 { color:#6B5B78 !important; font-weight:600 !important; }
h3 { color:#7A6B88 !important; font-weight:600 !important; }
p,li { color:#4A4058; line-height:1.75; }

/* ── Logo ── */
.nora-brand {
    display:flex; align-items:center; gap:11px;
    padding:20px 16px 12px;
}
.nora-mark {
    width:38px; height:38px; flex-shrink:0;
    background:linear-gradient(135deg,#E8A5B0 0%,#C8B5E8 100%);
    border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    font-weight:800; font-size:1.15rem; color:white; letter-spacing:-1px;
}
.nora-name  { font-weight:800; color:#5C4B6B; font-size:1.05rem; letter-spacing:2px; }
.nora-sub   { font-size:0.62rem; color:#A0909E; letter-spacing:0.8px; text-transform:uppercase; line-height:1.3; }

/* ── Cards ── */
.hero {
    background:linear-gradient(135deg,#F5EEF8 0%,#EEF4FA 55%,#EBF8F3 100%);
    border-radius:20px; padding:36px 44px; margin-bottom:24px;
    border:1px solid #EDE8E3;
}
.hero-title { font-size:2rem; font-weight:700; color:#5C4B6B; margin:0 0 10px; }
.hero-sub   { font-size:1.05rem; color:#6B5B78; max-width:680px; margin:0; }

.kpi-row  { display:flex; gap:14px; margin:20px 0; flex-wrap:wrap; }
.kpi {
    flex:1; min-width:130px; background:white;
    border-radius:14px; padding:18px 22px;
    border:1px solid #EDE8E3;
    box-shadow:0 2px 8px rgba(92,75,107,0.06);
    text-align:center;
}
.kpi-val   { font-size:1.8rem; font-weight:700; color:#5C4B6B; }
.kpi-lbl   { font-size:0.75rem; color:#7A8B99; margin-top:4px; text-transform:uppercase; letter-spacing:0.5px; }
.kpi-delta { font-size:0.8rem; font-weight:600; margin-top:5px; }
.kpi-delta.up   { color:#C0392B; }
.kpi-delta.down { color:#27AE60; }

.info-box {
    background:#F8F4FC; border-left:4px solid #C8B5E8;
    border-radius:0 12px 12px 0; padding:13px 18px; margin:14px 0;
    color:#5C4B6B; font-size:0.94rem;
}
.info-box.green { background:#F0FBF4; border-color:#A5D9C5; }

/* ── Model cards ── */
.model-grid { display:flex; gap:14px; flex-wrap:wrap; }
.model-card {
    flex:1; min-width:210px; background:white;
    border-radius:14px; padding:22px;
    border:1px solid #EDE8E3;
    box-shadow:0 2px 8px rgba(92,75,107,0.05);
}
.model-card.best     { border-top:4px solid #A5D9C5; }
.model-card.baseline { border-top:4px solid #C8B5E8; }
.model-card.boost    { border-top:4px solid #F5C8A5; }
.model-title  { font-size:0.97rem; font-weight:700; color:#5C4B6B; margin-bottom:7px; }
.model-desc   { font-size:0.83rem; color:#7A8B99; line-height:1.6; }
.model-metric { margin-top:13px; }
.model-metric .val { font-size:1.45rem; font-weight:700; color:#5C4B6B; }
.model-metric .lbl { font-size:0.75rem; color:#A0909E; display:block; }

/* ── Feature table ── */
.feat-table { width:100%; border-collapse:collapse; font-size:0.88rem; }
.feat-table th { background:#F5EFF8; color:#5C4B6B; font-weight:600;
    padding:10px 14px; text-align:left; border-bottom:2px solid #EDE8E3; }
.feat-table td { padding:9px 14px; border-bottom:1px solid #F0ECE8; color:#4A4058; }
.feat-table tr:hover td { background:#FAFAF8; }
.badge {
    display:inline-block; border-radius:20px; padding:2px 10px;
    font-size:0.75rem; font-weight:600;
}
.badge.red    { background:#FDECEA; color:#C62828; }
.badge.orange { background:#FFF3E0; color:#E65100; }
.badge.yellow { background:#FFFDE7; color:#F57F17; }
.badge.gray   { background:#F5F5F5; color:#757575; }

/* ── Demo form ── */
.demo-section {
    background:white; border-radius:14px; padding:20px 24px;
    border:1px solid #EDE8E3; margin-bottom:16px;
    box-shadow:0 1px 5px rgba(92,75,107,0.04);
}
.demo-section-title {
    font-size:0.8rem; font-weight:700; color:#A0909E;
    text-transform:uppercase; letter-spacing:1px; margin-bottom:14px;
    padding-bottom:10px; border-bottom:1px solid #F0ECE8;
}

/* ── Result ── */
.result-box { border-radius:16px; padding:22px 26px; margin:14px 0; border:1px solid; }
.result-box.high { background:#FDECEA; border-color:#FFCDD2; }
.result-box.low  { background:#E8F5E9; border-color:#C8E6C9; }
.result-title { font-size:1.25rem; font-weight:700; margin-bottom:7px; }
.result-box.high .result-title { color:#B71C1C; }
.result-box.low  .result-title  { color:#1B5E20; }

.factor-row {
    display:flex; align-items:center; gap:11px; background:white;
    border-radius:10px; padding:10px 14px; margin:6px 0;
    border:1px solid #EDE8E3;
}
.factor-dot { width:11px; height:11px; border-radius:50%; flex-shrink:0; }
.factor-name { font-weight:600; color:#4A4058; font-size:0.9rem; }
.factor-rate { font-size:0.8rem; color:#7A8B99; }

.formula-box {
    background:white; border:1px solid #EDE8E3; border-radius:14px;
    padding:18px 26px; margin:14px 0; text-align:center;
}

hr.soft { border:none; border-top:1px solid #EDE8E3; margin:26px 0; }

[data-testid="metric-container"] {
    background:white !important; border:1px solid #EDE8E3 !important;
    border-radius:12px !important; padding:14px !important;
}
</style>
"""

LOGO_HTML = """
<div class="nora-brand">
  <div class="nora-mark">N</div>
  <div>
    <div class="nora-name">NORA</div>
    <div class="nora-sub">Neonatal Outcome<br>Risk Assessment</div>
  </div>
</div>
<hr style="border:none;border-top:1px solid #EDE8E3;margin:0 16px 8px;">
"""


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement des modèles…")
def load_resources():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    model_rf     = joblib.load(MODELS_DIR / "model2_random_forest.pkl")
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop(columns=["Target_Risk"]), df["Target_Risk"]
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pca = PCA(n_components=10, random_state=42)
    pca.fit(X_train.values)
    return preprocessor, model_rf, pca

@st.cache_data(show_spinner=False)
def load_class_counts():
    df = pd.read_csv(DATA_PATH)
    c  = df["Target_Risk"].value_counts()
    return int(c.get(0, 0)), int(c.get(1, 0))


# ── Plotly charts ─────────────────────────────────────────────────────────────
def donut_chart(n_neg, n_pos):
    total = n_neg + n_pos
    fig = go.Figure(go.Pie(
        labels=["Pas d'admission", "Admission NICU"],
        values=[n_neg, n_pos],
        hole=0.6,
        marker_colors=[BLUE, ROSE],
        textinfo="percent",
        textfont=dict(size=12, color="white"),
        hovertemplate="<b>%{label}</b><br>%{value:,} · %{percent}<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{total:,}</b><br><span style='font-size:11px'>naissances</span>",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color=PLUM), align="center")
    fig.update_layout(**PLOTLY_BASE, height=300,
                      legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"))
    return fig

def nicu_rates_chart():
    labels = ["Population générale","Diabète gestationnel","ATCD mort fœtale",
              "HTA gestationnelle","ATCD accouchement prématuré","Traitement infertilité",
              "PMA","Éclampsie","Grossesse multiple (jumeaux)","Grossesse multiple (triplets+)"]
    rates  = [8.9, 12.3, 16.1, 16.8, 19.2, 21.3, 23.9, 27.1, 37.5, 87.5]
    colors = [SLATE, LAVENDER, LAVENDER, PEACH, PEACH, PEACH, ROSE, ROSE, DANGER, DANGER]
    fig = go.Figure(go.Bar(
        x=rates, y=labels, orientation="h",
        marker_color=colors,
        text=[f"  {r}%" for r in rates], textposition="outside",
        textfont=dict(size=10, color=PLUM),
        hovertemplate="<b>%{y}</b> — %{x}%<extra></extra>",
    ))
    fig.add_vline(x=8.9, line_dash="dot", line_color=SLATE, line_width=1.5)
    fig.update_layout(**PLOTLY_BASE, height=400,
                      xaxis=dict(range=[0, 100], gridcolor="#F0ECE8", title="Taux NICU (%)"),
                      yaxis=dict(tickfont=dict(size=10)))
    return fig

def model_comparison_chart():
    models = ["Logistic\nRegression", "Random\nForest ✓", "HistGBM\n+ Optuna"]
    data = [
        ("Recall",     [0.354, 0.766, 0.698], ROSE),
        ("F-beta β=2", [0.259, 0.350, 0.337], MINT),
        ("F1-Score",   [0.185, 0.193, 0.190], PEACH),
    ]
    fig = go.Figure()
    for name, vals, color in data:
        fig.add_trace(go.Bar(name=name, x=models, y=vals, marker_color=color,
                             text=[f"{v:.3f}" for v in vals],
                             textposition="outside", textfont=dict(size=10)))
    fig.update_layout(**PLOTLY_BASE, height=360, barmode="group",
                      yaxis=dict(range=[0, 0.95], gridcolor="#F0ECE8"),
                      legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                      bargap=0.18, bargroupgap=0.06)
    return fig

def gauge_chart(proba):
    color = DANGER if proba >= THRESHOLD_RF else SUCCESS
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=proba * 100,
        number=dict(suffix="%", font=dict(size=38, color=color)),
        delta=dict(reference=THRESHOLD_RF * 100, suffix="% vs seuil",
                   font=dict(size=12), increasing_color=DANGER, decreasing_color=SUCCESS),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickfont=dict(size=10, color=SLATE)),
            bar=dict(color=color, thickness=0.22),
            bgcolor="white", borderwidth=0,
            steps=[dict(range=[0, THRESHOLD_RF * 100], color="#E8F5E9"),
                   dict(range=[THRESHOLD_RF * 100, 100], color="#FDECEA")],
            threshold=dict(line=dict(color=PLUM, width=3),
                           thickness=0.82, value=THRESHOLD_RF * 100),
        ),
        title=dict(text="Probabilité NICU", font=dict(size=14, color=PLUM)),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(**PLOTLY_BASE, height=270)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE : ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
def page_accueil():
    n_neg, n_pos = load_class_counts()
    total = n_neg + n_pos
    rate  = n_pos / total * 100

    st.markdown("""
    <div class="hero">
      <p class="hero-title">Bienvenue sur NORA</p>
      <p class="hero-sub">
        NORA est un outil d'aide à la décision clinique qui prédit, à partir du profil
        de santé de la mère, la probabilité qu'un nouveau-né soit admis en
        <strong>Unité de Soins Intensifs Néonatals (NICU)</strong> — avant ou pendant la grossesse.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.subheader("Qu'est-ce que le NICU ?")
        st.markdown("""
        Le NICU *(Neonatal Intensive Care Unit)* accueille les nouveau-nés en détresse : prématurité, complications obstétricales, pathologies maternelles.
        Un séjour représente un coût de **3 000 à 100 000 USD** et mobilise des ressources hospitalières critiques.
        """)
        st.subheader("À quoi sert NORA ?")
        st.markdown("""
        Identifier **dès la grossesse** les mères à risque élevé pour :
        - Intensifier le suivi prénatal
        - Planifier les ressources néonatales
        - Prioriser les interventions de prévention
        """)

    with col2:
        st.subheader("Le défi")
        st.markdown(f"""
        Seulement **{rate:.1f}%** des naissances aboutissent à une admission NICU —
        un modèle naïf atteindrait 91% d'accuracy en prédisant toujours "non", sans jamais détecter un cas réel.
        NORA utilise **SMOTE + PCA + Random Forest** pour détecter **76.6% des bébés à risque**.
        """)
        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi">
            <div class="kpi-val">{rate:.1f}%</div>
            <div class="kpi-lbl">Taux NICU réel</div>
          </div>
          <div class="kpi">
            <div class="kpi-val">76.6%</div>
            <div class="kpi-lbl">Recall NORA</div>
            <div class="kpi-delta down">bébés à risque détectés</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    st.subheader("Sources et périmètre")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="demo-section">
          <div class="demo-section-title">Données</div>
          <p><strong>CDC Natality 2018</strong> — certificats de naissance américains, centralisés par le NCHS.</p>
          <p>Échantillon de <strong>99 900 naissances</strong> sur 3,8 millions disponibles.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="demo-section">
          <div class="demo-section-title">Quand utiliser NORA ?</div>
          <p>À partir du <strong>2e ou 3e trimestre</strong>, une fois les complications gestationnelles diagnostiquées (éclampsie, HTA, diabète).</p>
          <p>Non applicable en tout début de grossesse.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="demo-section">
          <div class="demo-section-title">Limites connues</div>
          <p>Absence de <strong>données biologiques</strong> (bilans sanguins, marqueurs sériques) — facteurs parmi les plus prédictifs cliniquement.</p>
          <p>Les performances sont donc <strong>sous-estimées</strong> par rapport à des données hospitalières complètes.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    ⚕️ NORA est un outil à vocation <strong>éducative et de recherche</strong>.
    Il ne remplace pas un diagnostic médical ni l'avis d'un professionnel de santé.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    st.subheader("Comment naviguer")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="demo-section" style="border-top:3px solid #C8B5E8;">
          <div class="demo-section-title">Données (EDA)</div>
          <p>Déséquilibre des classes, taux NICU par facteur de risque, distributions, corrélations.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="demo-section" style="border-top:3px solid #F5C8A5;">
          <div class="demo-section-title">Feature Engineering</div>
          <p>Variables médicales à <strong>fort signal</strong> (×2 à ×10), variables dérivées, pipeline complet.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="demo-section" style="border-top:3px solid #A5D9C5;">
          <div class="demo-section-title">Modèles</div>
          <p>3 modèles comparés sur F-beta β=2 — courbes ROC, matrices de confusion, Optuna.</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="demo-section" style="border-top:3px solid #E8A5B0;">
          <div class="demo-section-title">Démo</div>
          <p>Saisissez un profil maternel — NORA calcule la probabilité NICU en temps réel.</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE : EDA
# ══════════════════════════════════════════════════════════════════════════════
def page_eda():
    n_neg, n_pos = load_class_counts()
    total = n_neg + n_pos
    rate  = n_pos / total * 100

    st.title("Analyse Exploratoire des Données")
    st.markdown(f"Dataset CDC Natality 2018 — **{total:,} naissances**, "
                f"**{n_pos:,} admissions NICU** ({rate:.1f}%)")

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── 1. Déséquilibre des classes ──────────────────────────────────────────
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("Déséquilibre des classes")
        st.plotly_chart(donut_chart(n_neg, n_pos), use_container_width=True)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        Seulement **{n_pos:,} ({rate:.1f}%)** admissions NICU sur {total:,} naissances — un modèle naïf atteint 91% d'accuracy en prédisant toujours "non", sans détecter un seul cas.

        **NORA contourne ce biais via :**
        - **SMOTE** — génère des exemples synthétiques pour équilibrer les classes
        - **Seuil 0.16** — abaissé depuis 0.50 pour favoriser la détection
        - **F-beta β=2** — pénalise 2× plus les bébés à risque non détectés
        """)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── 2. Taux NICU par facteur ─────────────────────────────────────────────
    st.subheader("Taux NICU par facteur de risque")
    st.markdown("Taux d'admission observé par variable — la ligne pointillée est le taux de base (8.9%). Une grossesse triple atteint **×10** la moyenne.")
    st.plotly_chart(nicu_rates_chart(), use_container_width=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── 3. Distributions & corrélations ─────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distributions des variables continues")
        st.markdown("Distributions asymétriques avec outliers médicaux légitimes (IMC > 50, âge > 45) — justifie le choix du **RobustScaler**.")
        st.image(str(PLOTS_DIR / "2_distributions_continues.png"), use_container_width=True)
    with col_b:
        st.subheader("Matrice de corrélations")
        st.markdown("Corrélations des variables de base toutes < 0.05 avec la cible — justifie un modèle **non-linéaire** et l'ajout de variables médicales à plus fort signal.")
        st.image(str(PLOTS_DIR / "5_correlations.png"), use_container_width=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── 4. Variables catégorielles ───────────────────────────────────────────
    st.subheader("Variables catégorielles vs taux NICU")
    st.markdown("Taux NICU par modalité (Oui/Non/Inconnu) — les complications (éclampsie, HTA, PMA) montrent systématiquement un taux bien supérieur à la base.")
    st.image(str(PLOTS_DIR / "4_categoriques_vs_risque.png"), use_container_width=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── 5. Tabac vs risque ───────────────────────────────────────────────────
    st.subheader("Tabac & comportement maternel")
    st.markdown("Impact du tabac avant grossesse, au T1 et au T3 — signal faible (< 0.15) comparé aux variables médicales, mais pertinent comme indicateur comportemental cumulatif.")
    st.image(str(PLOTS_DIR / "3_tabac_vs_risque.png"), use_container_width=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── 6. Qualité des données ───────────────────────────────────────────────
    col_mv, col_mv2 = st.columns([1, 1])
    with col_mv:
        st.subheader("Valeurs manquantes & sentinelles CDC")
        st.markdown("Le CDC encode les non-réponses par des codes numériques (`9`, `99`, `99.9`) — neutralisés en `NaN` avant toute transformation. Ce graphique montre les taux réels après traitement.")
        st.image(str(PLOTS_DIR / "0_missing_values.png"), use_container_width=True)
    with col_mv2:
        st.subheader("Distribution de la cible")
        st.markdown("~91% de cas négatifs / ~9% positifs — déséquilibre réel conservé dans le jeu de test pour évaluer les modèles en conditions cliniques.")
        st.image(str(PLOTS_DIR / "1_distribution_target.png"), use_container_width=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── 7. Analyse multivariée ───────────────────────────────────────────────
    st.subheader("Analyse multivariée")
    st.markdown("Scatter matrix des variables colorées par classe NICU — les deux classes se mélangent dans toutes les projections, confirmant l'**absence de séparation linéaire** et la nécessité d'un modèle non-linéaire.")
    st.image(str(PLOTS_DIR / "6_multivariate.png"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE : FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def page_feature_engineering():
    st.title("Feature Engineering & Preprocessing")
    st.markdown("Sélection et transformation des 19 variables retenues parmi les 200+ disponibles du dataset CDC.")

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    st.subheader("Variables à fort signal médical")
    st.markdown("Les variables de base (âge, IMC, tabac) ont des corrélations < 0.05 avec la cible. Les variables médicales CDC multiplient le signal de **×1.4 à ×10**.")
    st.markdown("""
    <table class="feat-table">
    <tr><th>Variable</th><th>Source CDC</th><th>Taux NICU</th><th>Signal vs base (8.9%)</th></tr>
    <tr><td>Grossesse Multiple</td><td>dplural &gt; 1</td><td>37–87%</td>
        <td><span class="badge red">×4 à ×10</span></td></tr>
    <tr><td>Éclampsie</td><td>rf_ehype</td><td>27.1%</td>
        <td><span class="badge red">×3.0</span></td></tr>
    <tr><td>PMA</td><td>rf_artec</td><td>23.9%</td>
        <td><span class="badge orange">×2.7</span></td></tr>
    <tr><td>Traitement infertilité</td><td>rf_inftr</td><td>21.3%</td>
        <td><span class="badge orange">×2.4</span></td></tr>
    <tr><td>ATCD accouchement prématuré</td><td>rf_ppterm</td><td>19.2%</td>
        <td><span class="badge orange">×2.2</span></td></tr>
    <tr><td>HTA gestationnelle</td><td>rf_ghype</td><td>16.8%</td>
        <td><span class="badge orange">×1.9</span></td></tr>
    <tr><td>ATCD mort fœtale</td><td>priordead &gt; 0</td><td>16.1%</td>
        <td><span class="badge yellow">×1.8</span></td></tr>
    <tr><td>Diabète gestationnel</td><td>rf_gdiab</td><td>12.3%</td>
        <td><span class="badge yellow">×1.4</span></td></tr>
    <tr><td>Tabac 3e trimestre</td><td>cig_3</td><td>—</td>
        <td><span class="badge gray">corr +0.142</span></td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    st.subheader("Variables créées par Feature Engineering")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="demo-section" style="border-top:3px solid #E8A5B0;">
          <div class="demo-section-title">Evol_Tabac</div>
          <p style="font-family:monospace;font-size:0.85rem;color:#5C4B6B;background:#F8F4FC;padding:6px 10px;border-radius:6px;">Tabac_Trim1 − Tabac_Avant</p>
          <p>Capture le changement de comportement tabagique entre avant et pendant la grossesse.<br>
          <strong>−</strong> réduction · <strong>+</strong> aggravation · <strong>0</strong> stable (93.7% des cas)</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="demo-section" style="border-top:3px solid #A5D9C5;">
          <div class="demo-section-title">Suivi_T1</div>
          <p style="font-family:monospace;font-size:0.85rem;color:#5C4B6B;background:#F8F4FC;padding:6px 10px;border-radius:6px;">1 si Mois_Debut_Suivi ≤ 3</p>
          <p>Proxy de la <strong>précocité d'accès aux soins</strong> — un suivi dès le 1er trimestre est un facteur de protection reconnu.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="demo-section" style="border-top:3px solid #C8B5E8;">
          <div class="demo-section-title">Grossesse_Multiple</div>
          <p style="font-family:monospace;font-size:0.85rem;color:#5C4B6B;background:#F8F4FC;padding:6px 10px;border-radius:6px;">1 si dplural &gt; 1</p>
          <p><strong>Signal le plus discriminant</strong> du dataset — jumeaux : 37.5% NICU, triplets : 87.5% (×4 à ×10 vs base).</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    st.subheader("Pipeline de preprocessing")
    st.markdown("Chaque étape est appliquée dans l'ordre indiqué, via un `ColumnTransformer` scikit-learn sauvegardé.")

    c1, c2, c3, c4, c5 = st.columns(5)
    steps = [
        ("#E8A5B0", "1 · Sentinelles CDC",
         "Le CDC encode les non-réponses avec <code>9</code>, <code>99</code>, <code>99.9</code>. "
         "Ces valeurs sont remplacées par <code>NaN</code> avant toute transformation "
         "pour éviter qu'elles soient interprétées comme de vraies données."),
        ("#F5C8A5", "2 · Imputation",
         "— Variables <strong>numériques</strong> → médiane (robuste aux outliers)<br>"
         "— Variables <strong>catégorielles</strong> → mode (valeur la plus fréquente)<br>"
         "<em>La moyenne est rejetée : trop sensible aux IMC > 50 et âges > 45.</em>"),
        ("#A5D9C5", "3 · RobustScaler",
         "Centré sur la <strong>médiane</strong>, mis à l'échelle sur l'<strong>IQR</strong>. "
         "Choisi à la place du StandardScaler car les distributions de tabac et poids "
         "sont fortement asymétriques avec outliers légitimes."),
        ("#C8B5E8", "4 · One-Hot Encoding",
         "Variables catégorielles → colonnes binaires. <code>drop='first'</code> "
         "évite la <strong>dummy trap</strong> (multicolinéarité). "
         "Résultat : <strong>36 colonnes</strong> après encodage."),
        ("#A5C8E8", "5 · SMOTE + PCA(10)",
         "SMOTE équilibre les classes pour RF/HGB (pas pour LR). "
         "PCA réduit 36 → <strong>10 composantes</strong> (99.3% variance). "
         "Le PCA est fitté <em>avant</em> SMOTE pour éviter le data leakage."),
    ]
    for col, (color, title, desc) in zip([c1, c2, c3, c4, c5], steps):
        with col:
            st.markdown(f"""
            <div class="demo-section" style="border-top:3px solid {color};min-height:180px;">
              <div class="demo-section-title">{title}</div>
              <p style="font-size:0.83rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    col_pca, col_poids = st.columns(2)
    with col_pca:
        st.subheader("Variance expliquée par la PCA")
        st.markdown("**10 composantes capturent 99.3%** de la variance des 36 colonnes originales — la redondance OHE est éliminée efficacement.")
        st.image(str(PLOTS_DIR / "preprocessing_pca_variance.png"), use_container_width=True)
    with col_poids:
        st.subheader("Prise de poids — après nettoyage")
        st.markdown("Les valeurs hors plage [0–80 lbs] (erreurs cliniques, pas des sentinelles) sont remplacées par `NaN` et imputées — distribution finale conforme aux recommandations IOM.")
        st.image(str(PLOTS_DIR / "poids_bins.png"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE : MODÈLES
# ══════════════════════════════════════════════════════════════════════════════
def page_models():
    st.title("Modèles & Évaluation")

    st.subheader("Métrique — F-beta β=2")
    st.markdown("L'Accuracy est trompeuse à 9% de positifs. Le **F-beta β=2** pondère le Recall 2× plus que la Précision — un bébé non détecté coûte plus qu'une fausse alerte.")
    st.markdown("""
    <div class="formula-box">
      <span style="color:#7A8B99;font-size:0.82rem;">OBJECTIF D'OPTIMISATION</span><br><br>
      <span style="font-size:1.1rem;font-weight:600;color:#5C4B6B;">
        F<sub>β=2</sub> &nbsp;=&nbsp; 5 · Précision · Recall &nbsp;/&nbsp; (4 · Précision + Recall)
      </span><br>
      <span style="font-size:0.8rem;color:#A0909E;">β=2 → un Faux Négatif est 2× plus pénalisant qu'un Faux Positif</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── Les 3 modèles ────────────────────────────────────────────────────────
    st.subheader("Les trois modèles testés")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="demo-section" style="border-top:4px solid #C8B5E8;">
          <div class="demo-section-title">1 · Régression Logistique &nbsp; Baseline</div>
          <p>Frontière linéaire dans PCA(10). Pas de SMOTE — le seuil seul gère le déséquilibre.</p>
          <p><strong>Limite :</strong> incapable de capturer les interactions non-linéaires.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="demo-section" style="border-top:4px solid #A5D9C5;">
          <div class="demo-section-title">2 · Random Forest &nbsp;
            <span style="background:#E8F5E9;color:#2E7D32;padding:2px 8px;border-radius:8px;font-size:0.7rem;font-weight:700;">Retenu</span>
          </div>
          <p>100 arbres en parallèle sur sous-échantillons. SMOTE + PCA(10) + seuil optimisé F-beta β=2.</p>
          <p><strong>Atout :</strong> robuste aux outliers, capture les interactions complexes.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="demo-section" style="border-top:4px solid #F5C8A5;">
          <div class="demo-section-title">3 · HistGradientBoosting + Optuna</div>
          <p>Boosting séquentiel, hyperparamètres optimisés via Optuna (25 trials, CV=3).</p>
          <p><strong>Note :</strong> seuil conservateur (0.46) — moins de FP, recall légèrement inférieur.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── Tableau comparatif ───────────────────────────────────────────────────
    st.subheader("Tableau comparatif des résultats")
    st.markdown("Métriques sur le **jeu de test** (20%, jamais vu pendant l'entraînement). Seuil optimisé indépendamment par F-beta β=2.")
    st.markdown("""
    <table class="feat-table" style="text-align:center;">
      <tr>
        <th style="text-align:left;">Modèle</th>
        <th>Recall</th>
        <th>Précision</th>
        <th>F1-Score</th>
        <th>F-beta β=2</th>
        <th>Seuil</th>
        <th>SMOTE</th>
      </tr>
      <tr>
        <td style="text-align:left;color:#7A8B99;">Régression Logistique</td>
        <td>0.354</td><td>0.109</td><td>0.185</td><td>0.259</td>
        <td>0.10</td><td style="color:#A0909E;">Non</td>
      </tr>
      <tr style="background:#F3FBF7;">
        <td style="text-align:left;font-weight:700;color:#5C4B6B;">
          Random Forest ✓
        </td>
        <td style="font-weight:700;color:#2E7D32;">0.766</td>
        <td>0.106</td>
        <td style="font-weight:700;">0.193</td>
        <td style="font-weight:700;color:#2E7D32;">0.350</td>
        <td>0.16</td>
        <td style="color:#2E7D32;font-weight:600;">Oui</td>
      </tr>
      <tr>
        <td style="text-align:left;color:#7A8B99;">HistGBM + Optuna</td>
        <td>0.698</td><td>0.105</td><td>0.190</td><td>0.337</td>
        <td>0.46</td><td style="color:#2E7D32;">Oui</td>
      </tr>
    </table>
    <p style="font-size:0.78rem;color:#A0909E;margin-top:8px;">
      La précision est faible sur les trois modèles — conséquence du déséquilibre des classes
      et du seuil bas choisi pour maximiser le recall. Dans ce contexte médical, c'est un compromis acceptable.
    </p>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── Graphique + protocole ────────────────────────────────────────────────
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.subheader("Comparaison visuelle des métriques")
        st.markdown("Recall (rose) en priorité — le Random Forest domine à **0.766**, soit **+41 pts** vs la baseline.")
        st.plotly_chart(model_comparison_chart(), use_container_width=True)
    with col2:
        st.subheader("Protocole d'évaluation")
        st.markdown("""
        <div class="demo-section">
          <div class="demo-section-title">Protocole anti-leakage</div>
          <p><strong>Split 80/20 stratifié</strong> — jeu de test mis de côté dès le début.</p>
          <p><strong>SMOTE dans le pipeline CV</strong> — jamais avant le split.</p>
          <p><strong>PCA fitté sur X_train original</strong> — avant SMOTE, pour ne pas biaiser les axes.</p>
          <p><strong>Seuil optimisé sur F-beta β=2</strong> — grid search [0.10–0.90] sur test uniquement.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box green">
          Le Random Forest obtient le meilleur F-beta β=2 (<strong>0.350</strong>)
          et détecte <strong>76.6%</strong> des bébés à risque — il est retenu
          comme modèle principal de NORA.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── Comparaison notebook ─────────────────────────────────────────────────
    st.subheader("Vue d'ensemble — comparaison notebook")
    st.markdown("Quatre métriques côte à côte (gauche) et F-beta β=2 seul (droite) — le Random Forest domine sur les deux panels.")
    st.image(str(PLOTS_DIR / "comparaison_modeles.png"), use_container_width=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── Triptyques des 3 modèles ─────────────────────────────────────────────
    st.subheader("Analyse de performance — les 3 modèles")
    st.markdown("Matrice de confusion · Courbe PR · Courbe ROC — les Faux Négatifs (rouge) sont la case à minimiser.")

    tab1, tab2, tab3 = st.tabs([
        "1 · Régression Logistique — Baseline",
        "2 · Random Forest ✓ Retenu",
        "3 · HistGradientBoosting + Optuna",
    ])
    with tab1:
        st.markdown("""
        <div class="info-box" style="margin-bottom:12px;">
        Seuil optimisé : <strong>0.10</strong> — Recall : <strong>0.354</strong> — F-beta β=2 : <strong>0.259</strong>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Frontière linéaire — seuil 0.10 nécessaire pour capturer des cas positifs, mais discrimination faible confirmée par la courbe PR.")
        st.image(str(PLOTS_DIR / "model1_logistic_regression.png"), use_container_width=True)
    with tab2:
        st.markdown("""
        <div class="info-box green" style="margin-bottom:12px;">
        Seuil optimisé : <strong>0.16</strong> — Recall : <strong>0.766</strong> — F-beta β=2 : <strong>0.350</strong> ✓ Meilleur
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Sur 100 bébés à risque, **77 sont détectés**. Seuil 0.16 volontairement bas — manquer un cas réel est le coût le plus grave.")
        st.image(str(PLOTS_DIR / "model2_random_forest.png"), use_container_width=True)
    with tab3:
        st.markdown("""
        <div class="info-box" style="margin-bottom:12px;">
        Seuil optimisé : <strong>0.46</strong> — Recall : <strong>0.698</strong> — F-beta β=2 : <strong>0.337</strong>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Recall 69.8% — seuil plus conservateur (0.46) qui réduit les faux positifs. Aire sous la courbe PR comparable au Random Forest.")
        st.image(str(PLOTS_DIR / "model3_histgbm_optuna.png"), use_container_width=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    st.subheader("Visualisations complémentaires")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Convergence Optuna (HistGBM)**")
        st.markdown("Évolution du F-beta β=2 sur 25 trials — convergence rapide, la majorité du gain est obtenu dès les 10 premiers essais.")
        st.image(str(PLOTS_DIR / "model3_optuna.png"), use_container_width=True)
    with col4:
        st.markdown("**Benchmark LazyPredict — 25 classifieurs**")
        st.markdown("25 algorithmes entraînés sans tuning sur 8 000 lignes — confirme que RF et GBM dominent, validant le choix de ces deux modèles.")
        st.image(str(PLOTS_DIR / "lazypredict_benchmark.png"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE : DÉMO
# ══════════════════════════════════════════════════════════════════════════════
def page_demo(preprocessor, model_rf, pca):
    st.title("Évaluation du Risque NICU")
    st.markdown(
        "Renseignez le profil clinique ci-dessous. NORA estimera la probabilité "
        "d'admission en NICU via le **Random Forest** (Recall 76.6%, seuil 0.16)."
    )
    st.markdown(
        '<div class="info-box">⚕️ Outil éducatif — ne remplace pas un avis médical.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── Section 1 : Profil maternel ──────────────────────────────────────────
    st.markdown('<div class="demo-section"><div class="demo-section-title">Profil maternel</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: age         = st.slider("Âge (ans)", 15, 50, 28)
    with c2: imc         = st.number_input("IMC avant grossesse", 15.0, 70.0, 25.5, 0.5)
    with c3: prise_poids = st.slider("Prise de poids (lbs)", 0, 80, 25)
    with c4: mois_suivi  = st.slider("Début suivi prénatal (mois)", 1, 10, 2,
                                      help="1 = dès le début, 10 = très tardif")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 2 : Tabac ────────────────────────────────────────────────────
    st.markdown('<div class="demo-section"><div class="demo-section-title">Tabac</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1: tabac_cig = st.slider("Cigarettes/jour (moyenne grossesse)", 0, 40, 0,
                                    help="93% des mères dans le dataset : 0 cigarette")
    with c2:
        if tabac_cig == 0:
            st.markdown('<div class="info-box green" style="margin-top:28px;">Non-fumeur — facteur protecteur.</div>',
                        unsafe_allow_html=True)
        elif tabac_cig <= 10:
            st.markdown(f'<div class="info-box" style="margin-top:28px;">{tabac_cig} cig/jour — risque modéré.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="info-box" style="border-color:#E8A5B0;background:#FDF4F6;margin-top:28px;">'
                        f'{tabac_cig} cig/jour — signal d\'alerte tabagique.</div>',
                        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 3 : Grossesse & antécédents ─────────────────────────────────
    st.markdown('<div class="demo-section"><div class="demo-section-title">Grossesse & antécédents</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: gross_mult  = st.selectbox("Type de grossesse", ["Simple", "Multiple (jumeaux+)"])
    with c2: atcd_mort   = st.selectbox("Antécédent mort fœtale", list(YN_MAP.keys()), index=0)
    with c3: atcd_prem   = st.selectbox("Antécédent accouchement prématuré", list(YN_MAP.keys()), index=0)
    with c4: aide_wic    = st.selectbox("Aide alimentaire WIC", list(YN_MAP.keys()), index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 4 : Conditions médicales ────────────────────────────────────
    st.markdown('<div class="demo-section"><div class="demo-section-title">Conditions médicales préexistantes</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: d_chron  = st.selectbox("Diabète chronique", list(YN_MAP.keys()), index=0)
    with c2: h_chron  = st.selectbox("Hypertension chronique", list(YN_MAP.keys()), index=0)
    with c3: assurance = st.selectbox("Assurance", list(ASSUR_MAP.keys()), index=0)
    with c4: pma       = st.selectbox("PMA", list(PMA_MAP.keys()), index=0)
    with c5: trt_inf   = st.selectbox("Traitement infertilité", list(YN_MAP.keys()), index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 5 : Complications gestationnelles ────────────────────────────
    st.markdown('<div class="demo-section"><div class="demo-section-title">Complications gestationnelles</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: eclampsie = st.selectbox("Éclampsie / pré-éclampsie sévère", list(YN_MAP.keys()), index=0)
    with c2: hta_gest  = st.selectbox("HTA gestationnelle", list(YN_MAP.keys()), index=0)
    with c3: diab_gest = st.selectbox("Diabète gestationnel", list(YN_MAP.keys()), index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Bouton ───────────────────────────────────────────────────────────────
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        run = st.button("Évaluer le risque NICU", type="primary", use_container_width=True)

    if run:
        suivi_t1   = 1 if mois_suivi <= 3 else 0
        gross_v    = 1 if gross_mult == "Multiple (jumeaux+)" else 0
        mort_v     = 1 if YN_MAP[atcd_mort] == "Y" else 0

        raw = pd.DataFrame([{
            "Age_Mere": age, "Mois_Debut_Suivi": mois_suivi,
            "Tabac_Avant": tabac_cig, "Tabac_Trim1": tabac_cig, "Tabac_Trim3": tabac_cig,
            "IMC_Mere": imc, "Prise_Poids": prise_poids,
            "Evol_Tabac": 0, "Suivi_T1": suivi_t1,
            "Grossesse_Multiple": gross_v, "ATCD_Mort_Foetale_bin": mort_v,
            "Aide_WIC": YN_MAP[aide_wic], "D_Chronique": YN_MAP[d_chron],
            "H_Chronique": YN_MAP[h_chron], "Assurance": ASSUR_MAP[assurance],
            "ATCD_Premature": YN_MAP[atcd_prem], "PMA": PMA_MAP[pma],
            "Traitement_Infertilite": YN_MAP[trt_inf],
            "Eclampsie": YN_MAP[eclampsie], "HTA_Gestationnelle": YN_MAP[hta_gest],
            "Diabete_Gestationnel": YN_MAP[diab_gest],
        }])

        X_proc = preprocessor.transform(raw)
        X_pca  = pca.transform(X_proc)
        proba  = model_rf.predict_proba(X_pca)[0, 1]
        pred   = proba >= THRESHOLD_RF

        st.markdown('<hr class="soft">', unsafe_allow_html=True)
        col_g, col_r = st.columns([1, 1.4])
        with col_g:
            st.plotly_chart(gauge_chart(proba), use_container_width=True)
        with col_r:
            if pred:
                st.markdown(f"""
                <div class="result-box high">
                  <div class="result-title">Risque élevé d'admission NICU</div>
                  <p>Probabilité estimée : <strong>{proba*100:.1f}%</strong>
                  (seuil clinique : {THRESHOLD_RF*100:.0f}%)</p>
                  <p><strong>Recommandation :</strong> intensifier le suivi prénatal
                  et orienter vers une consultation en médecine fœtale.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box low">
                  <div class="result-title">Risque faible d'admission NICU</div>
                  <p>Probabilité estimée : <strong>{proba*100:.1f}%</strong>
                  (seuil clinique : {THRESHOLD_RF*100:.0f}%)</p>
                  <p>Suivi prénatal standard recommandé selon les protocoles habituels.</p>
                </div>
                """, unsafe_allow_html=True)

        # ── Facteurs identifiés ────────────────────────────────────────────
        FACTORS = [
            (gross_v,                  "#E57373", "Grossesse multiple",               "Taux NICU : 37–87%"),
            (YN_MAP[eclampsie]=="Y",   "#E57373", "Éclampsie / pré-éclampsie",         "Taux NICU : 27.1%"),
            (PMA_MAP[pma]=="Y",        "#FFB74D", "PMA",                               "Taux NICU : 23.9%"),
            (YN_MAP[trt_inf]=="Y",     "#FFB74D", "Traitement infertilité",            "Taux NICU : 21.3%"),
            (YN_MAP[atcd_prem]=="Y",   "#FFB74D", "Antécédent accouchement prématuré","Taux NICU : 19.2%"),
            (YN_MAP[hta_gest]=="Y",    "#FFD54F", "HTA gestationnelle",                "Taux NICU : 16.8%"),
            (mort_v,                   "#FFD54F", "Antécédent mort fœtale",            "Taux NICU : 16.1%"),
            (YN_MAP[diab_gest]=="Y",   "#FFD54F", "Diabète gestationnel",              "Taux NICU : 12.3%"),
            (tabac_cig > 0,            "#A5D9C5", "Tabagisme durant la grossesse",     f"{tabac_cig} cig/jour"),
        ]
        active = [(c, n, d) for cond, c, n, d in FACTORS if cond]

        st.markdown('<hr class="soft">', unsafe_allow_html=True)
        st.subheader("Facteurs de risque dans ce profil")
        if active:
            html = ""
            for color, name, detail in active:
                html += (f'<div class="factor-row"><div class="factor-dot" style="background:{color};"></div>'
                         f'<div><div class="factor-name">{name}</div>'
                         f'<div class="factor-rate">{detail}</div></div></div>')
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box green">Aucun facteur de risque majeur identifié.</div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE : LIMITES
# ══════════════════════════════════════════════════════════════════════════════
def page_limites():
    st.title("Limites & Perspectives")
    st.markdown("Recensement honnête des contraintes du projet — données, modèle et périmètre d'utilisation.")

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── 1. Limites du dataset ────────────────────────────────────────────────
    st.subheader("Limites du dataset")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="demo-section" style="border-top:4px solid #E8A5B0;">
          <div class="demo-section-title">Données manquantes — biologiques</div>
          <p>Le dataset CDC ne contient <strong>aucune donnée biologique</strong> :
          bilans sanguins, marqueurs sériques, échographies, données génétiques.</p>
          <p>Ces facteurs sont cliniquement parmi les plus prédictifs du risque NICU —
          leur absence constitue le <strong>plafond de performance</strong> du modèle (~76% recall).</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="demo-section" style="border-top:4px solid #F5C8A5;">
          <div class="demo-section-title">Population — biais géographique</div>
          <p>Le dataset couvre exclusivement les <strong>naissances américaines en 2018</strong>
          (certificats CDC). Les résultats peuvent ne pas se généraliser à d'autres
          systèmes de santé, pratiques obstétricales ou populations.</p>
          <p>Le profil moyen : mère de 27.4 ans, IMC 29.7 — tout profil très éloigné
          de cette distribution est moins bien couvert.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── 2. Limites du modèle ─────────────────────────────────────────────────
    st.subheader("Limites du modèle")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="demo-section" style="border-top:4px solid #C8B5E8;">
          <div class="demo-section-title">Interprétabilité réduite</div>
          <p>La <strong>PCA</strong> transforme les 36 features en 10 composantes sans
          correspondance directe avec des variables médicales.</p>
          <p>Il est impossible d'expliquer la prédiction en termes de "cette variable
          a contribué de X%" — un outil SHAP post-PCA serait nécessaire en production.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="demo-section" style="border-top:4px solid #A5C8E8;">
          <div class="demo-section-title">Faux positifs élevés</div>
          <p>Le seuil bas (<strong>0.16</strong>) génère ~8 faux positifs pour chaque
          vrai positif détecté — soit ~9 400 alertes inutiles sur 19 980 cas de test.</p>
          <p>Ce compromis est volontaire (contexte médical), mais il limiterait
          l'adoption en milieu hospitalier à ressources restreintes.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="demo-section" style="border-top:4px solid #A5D9C5;">
          <div class="demo-section-title">SMOTE — données synthétiques</div>
          <p><strong>SMOTE</strong> génère des exemples positifs artificiels par
          interpolation entre voisins réels. Ces exemples n'existent pas dans la
          réalité clinique.</p>
          <p>Les performances en production sur de nouvelles données réelles
          pourraient être légèrement inférieures aux métriques mesurées sur le jeu de test.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── 3. Périmètre d'utilisation ───────────────────────────────────────────
    st.subheader("Périmètre d'utilisation")
    st.markdown("""
    <div class="info-box" style="margin-bottom:16px;">
    ⚕️ NORA est un <strong>outil éducatif et de recherche</strong> — il ne remplace pas un diagnostic médical ni l'avis d'un professionnel de santé.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="demo-section" style="border-top:3px solid #F5C8A5;">
          <div class="demo-section-title">Fenêtre temporelle</div>
          <p>Le modèle est applicable à partir du <strong>2e–3e trimestre</strong> uniquement,
          une fois que les complications gestationnelles sont diagnostiquées
          (éclampsie, HTA gestationnelle, diabète gestationnel).</p>
          <p>En début de grossesse, ces variables sont inconnues — le modèle
          ne peut pas être utilisé de façon fiable.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="demo-section" style="border-top:3px solid #C8B5E8;">
          <div class="demo-section-title">Variables déclaratives</div>
          <p>Plusieurs variables (tabac, aide WIC, antécédents) sont
          <strong>auto-déclarées</strong> sur les certificats de naissance —
          elles peuvent être sous-déclarées, notamment pour le tabac.</p>
          <p>Un biais de déclaration systématique peut réduire la précision
          du modèle sur certains profils sociaux.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── 4. Perspectives ──────────────────────────────────────────────────────
    st.subheader("Perspectives d'amélioration")
    st.markdown("""
    | Axe | Action envisageable |
    |---|---|
    | **Signal** | Intégrer des données biologiques (marqueurs sériques, bilans) |
    | **Interprétabilité** | Ajouter SHAP values post-PCA pour expliquer chaque prédiction |
    | **Seuil** | Adapter dynamiquement le seuil selon le contexte hospitalier (ressources disponibles) |
    | **Généralisation** | Valider sur des datasets non-américains (Europe, autres années) |
    | **Données** | Récupérer les données post-2018 pour évaluer la stabilité temporelle du modèle |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def build_app() -> None:
    st.set_page_config(page_title="NORA — Neonatal Risk", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    preprocessor, model_rf, pca = load_resources()

    st.sidebar.markdown(LOGO_HTML, unsafe_allow_html=True)
    menu = st.sidebar.radio(
        "Navigation", ["Accueil", "Données (EDA)", "Feature Engineering", "Modèles", "Démo", "Limites"],
        label_visibility="collapsed",
    )

    if   menu == "Accueil":             page_accueil()
    elif menu == "Données (EDA)":       page_eda()
    elif menu == "Feature Engineering": page_feature_engineering()
    elif menu == "Modèles":             page_models()
    elif menu == "Démo":                page_demo(preprocessor, model_rf, pca)
    else:                               page_limites()


if __name__ == "__main__":
    build_app()
