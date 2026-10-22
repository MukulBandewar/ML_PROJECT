from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "credit_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DATA_PATH = ROOT / "credit_data.csv"

st.set_page_config(
    page_title="CreditScore AI",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0b1220,#172943,#103744);
    color: #f4f7fb;
}
.block-container {
    max-width: 1200px;
    padding: 1.6rem 1.2rem;
}
h1,h2,h3,p,label {
    color: #f4f7fb;
}
[data-testid="stMetric"] {
    background: #ffffff10;
    border: 1px solid #ffffff25;
    border-radius: 14px;
    padding: 14px;
}
.stButton button,.stDownloadButton button {
    width: 100%;
    min-height: 44px;
    border-radius: 10px;
}
@media(max-width:640px) {
    .block-container {
        padding: 1rem .7rem;
    }
    h1 {
        font-size: 1.8rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "annual_income_usd",
    "existing_debts_usd",
    "payment_history_score",
    "credit_utilization_ratio",
    "employment_years"
]

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)

st.title("💳 CreditScore AI")
st.caption("Machine Learning Creditworthiness Prediction")
st.info(
    "Educational project only. Predictions are not official "
    "credit scores or real lending decisions."
)

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.warning("Model files not found. First run: python train_model.py")
    st.stop()

model = load_model()
metrics = load_metrics()

predict_tab, metrics_tab, data_tab = st.tabs([
    "🔍 Credit Prediction",
    "📊 Model Performance",
    "📁 Dataset"
])

with predict_tab:
    st.subheader("Enter Financial Details")

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            income = st.number_input(
                "Annual Income (USD)",
                min_value=1, value=60000
            )
            debt = st.number_input(
                "Existing Debts (USD)",
                min_value=0, value=12000
            )
            payment = st.slider(
                "Payment History Score", 0, 100, 75
            )

        with right:
            utilization = st.slider(
                "Credit Utilization Ratio",
                0.0, 1.0, 0.30, step=0.01,
                help="Enter a ratio from 0 to 1. For example, 0.37 means 37%."
            )
            employment = st.slider(
                "Employment Years", 0, 50, 5
            )

        submitted = st.form_submit_button(
            "Predict Creditworthiness",
            type="primary"
        )

    if submitted:
        row = pd.DataFrame([{
            "annual_income_usd": income,
            "existing_debts_usd": debt,
            "payment_history_score": payment,
            "credit_utilization_ratio": utilization,
            "employment_years": employment
        }], columns=FEATURES)

        probability = float(model.predict_proba(row)[0][1])
        prediction = int(probability >= 0.5)

        st.divider()
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Category",
                "Creditworthy" if prediction == 1
                else "Not Creditworthy"
            )

        with col2:
            st.metric(
                "Estimated Creditworthy Probability",
                f"{probability:.1%}"
            )

        st.progress(probability)

        st.caption(
            "This probability is the model's estimate for class 1, "
            "not a verified probability of loan repayment."
        )

        if prediction == 1:
            st.success("The model predicts class 1: Creditworthy.")
        else:
            st.warning("The model predicts class 0: Not Creditworthy.")

with metrics_tab:
    st.subheader("Test Set Evaluation")
    best = metrics["best_model"]
    score = metrics["results"][best]

    st.write(f"Selected model: **{best}**")

    cols = st.columns(5)
    names = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1_score"),
        ("ROC-AUC", "roc_auc")
    ]

    for col, (label, key) in zip(cols, names):
        value = score[key]
        text = f"{value:.1%}" if key != "roc_auc" else f"{value:.3f}"
        col.metric(label, text)

    st.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots()
    display = ConfusionMatrixDisplay(
        confusion_matrix=score["confusion_matrix"],
        display_labels=["Class 0", "Class 1"]
    )
    display.plot(ax=ax, colorbar=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    comparison = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": value["accuracy"],
            "Precision": value["precision"],
            "Recall": value["recall"],
            "F1-score": value["f1_score"],
            "ROC-AUC": value["roc_auc"]
        }
        for name, value in metrics["results"].items()
    ])

    st.markdown("#### Algorithm Comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

with data_tab:
    st.subheader("Uploaded Dataset")

    data = pd.read_csv(DATA_PATH)
    st.write(f"Total records: {len(data)}")
    st.dataframe(data, use_container_width=True, hide_index=True)

    st.download_button(
        "Download CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="credit_data.csv",
        mime="text/csv"
    )

    st.caption(
        "The customer_id column is displayed for reference but is "
        "not used as a model feature."
    )

st.divider()
st.caption("Built with Python · Scikit-learn · Pandas · Streamlit")
