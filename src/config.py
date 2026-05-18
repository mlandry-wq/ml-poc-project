from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

for dir in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    dir.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

MODELS = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "description": "Baseline linéaire — PCA(10), seuil par défaut 0.50.",
        "path": MODELS_DIR / "model1_logistic_regression.pkl",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "100 arbres — SMOTE + PCA(10). Modèle retenu (meilleur F-beta β=2).",
        "path": MODELS_DIR / "model2_random_forest.pkl",
    },
    "histgbm_optuna": {
        "name": "HistGradientBoosting + Optuna",
        "description": "Boosting séquentiel optimisé Optuna — PCA(10), seuil optimisé 0.46.",
        "path": MODELS_DIR / "model3_histgbm_optuna.pkl",
    },
}
