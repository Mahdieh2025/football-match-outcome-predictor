import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================
# Paths (robust)
# =========================
APP_DIR = Path(__file__).resolve().parent          # .../project_repo/app
PROJECT_ROOT = APP_DIR.parent                     # .../project_repo
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"


# =========================
# Helpers
# =========================
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    return model, metadata


LABEL_MAP = {
    0: "Away Win",
    1: "Draw",
    2: "Home Win",
    "A": "Away Win",
    "D": "Draw",
    "H": "Home Win",
}


def decode_prediction(pred):
    if isinstance(pred, str):
        return LABEL_MAP.get(pred, pred)
    return LABEL_MAP.get(int(pred), str(pred))


def decode_classes(classes):
    return [LABEL_MAP.get(c, str(c)) for c in classes]


def validate_inputs(home_team: str, away_team: str):
    if not home_team or not away_team:
        return False, "Please provide both Home Team and Away Team."
    if home_team.strip().lower() == away_team.strip().lower():
        return False, "Home Team and Away Team must be different."
    return True, ""


# =========================
# UI (attractive centered layout)
# =========================
st.set_page_config(page_title="Football Match Outcome Predictor", layout="centered")
st.title("⚽ Football Match Outcome Predictor")
st.write("Predict match outcome: **Home win (H)**, **Draw (D)**, **Away win (A)**")

with st.expander("How it works"):
    st.markdown(
        """
This app predicts a match outcome using a trained machine learning classifier.

**Inputs**
- League division (**Div**)
- Home and away teams
- Season and date information (**season, year, month**)

**Outputs**
- The predicted result (**Home Win / Draw / Away Win**)
- The probability of each possible outcome
"""
    )

# Load model safely
try:
    model, metadata = load_artifacts()
except Exception as e:
    st.error("Failed to load model files. Check `project_repo/models/`.")
    st.exception(e)
    st.stop()

# Inputs
divisions = metadata.get("divisions", ["E0", "E1", "SP1"])
teams = metadata.get("teams", None)

div = st.selectbox("Division (Div)", divisions)

# ✅ Improvement: Teams in two columns (cleaner UI)
col1, col2 = st.columns(2)

if isinstance(teams, list) and len(teams) > 0:
    def_idx_home = teams.index("Arsenal") if "Arsenal" in teams else 0
    def_idx_away = teams.index("Chelsea") if "Chelsea" in teams else min(1, len(teams) - 1)

    with col1:
        home = st.selectbox("Home Team", teams, index=def_idx_home)
    with col2:
        away = st.selectbox("Away Team", teams, index=def_idx_away)
else:
    with col1:
        home = st.text_input("Home Team", "Arsenal")
    with col2:
        away = st.text_input("Away Team", "Chelsea")

# ✅ Improvement: Season/Year/Month in three columns
c1, c2, c3 = st.columns(3)
with c1:
    season = st.number_input("Season", min_value=2015, max_value=2025, value=2023, step=1)
with c2:
    year = st.number_input("Year", min_value=2015, max_value=2025, value=2023, step=1)
with c3:
    month = st.number_input("Month", min_value=1, max_value=12, value=5, step=1)

# Predict
if st.button("Predict"):
    ok, msg = validate_inputs(home, away)
    if not ok:
        st.warning(msg)
        st.stop()

    X_input = pd.DataFrame([{
        "Div": div,
        "HomeTeam": home,
        "AwayTeam": away,
        "season": season,
        "year": year,
        "month": month,
    }])

    try:
        pred = model.predict(X_input)[0]
        label = decode_prediction(pred)
        st.success(f"🏆 Predicted Outcome: **{label}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            classes = list(getattr(model, "classes_", ["A", "D", "H"]))
            class_labels = decode_classes(classes)

            proba_df = (
                pd.DataFrame({"Outcome": class_labels, "Probability": proba})
                .sort_values("Probability", ascending=False)
                .reset_index(drop=True)
            )
            proba_df["Probability %"] = (proba_df["Probability"] * 100).round(1)

            st.subheader("Prediction Probabilities")
            for _, row in proba_df.iterrows():
                st.write(f"**{row['Outcome']}** — {row['Probability %']}%")
                st.progress(float(row["Probability"]))

            with st.expander("Show probability table"):
                st.dataframe(
                    proba_df[["Outcome", "Probability %"]],
                    use_container_width=True,
                    hide_index=True,
                )

    except Exception as e:
        st.error("Prediction failed. Most likely input columns do not match training features.")
        st.exception(e)

# Footer
st.markdown("---")
st.caption("Built with Python • scikit-learn • Streamlit")

# Debug info
with st.expander("Debug info"):
    st.write("PROJECT_ROOT:", str(PROJECT_ROOT))
    st.write("MODEL_PATH:", str(MODEL_PATH))
    st.write("MODEL_EXISTS:", MODEL_PATH.exists())
    st.write("METADATA_PATH:", str(METADATA_PATH))
    st.write("METADATA_EXISTS:", METADATA_PATH.exists())
    if metadata:
        st.write("METADATA_KEYS:", list(metadata.keys()))