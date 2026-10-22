from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "annual_income_usd",
    "existing_debts_usd",
    "payment_history_score",
    "credit_utilization_ratio",
    "employment_years"
]

def main():
    data = pd.read_csv(ROOT / "credit_data.csv")

    required = FEATURES + ["creditworthy"]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    data = data.dropna(subset=["creditworthy"])
    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(
        data["creditworthy"], errors="coerce"
    )

    valid = y.isin([0, 1])
    X, y = X.loc[valid], y.loc[valid].astype(int)

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError(
            "The dataset needs enough records from both classes (0 and 1)."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced"
            )
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42
            )
        )
    }

    results = {}
    best_model, best_name, best_f1 = None, "", -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        scores = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(
                y_test, pred, zero_division=0
            )),
            "recall": float(recall_score(
                y_test, pred, zero_division=0
            )),
            "f1_score": float(f1_score(
                y_test, pred, zero_division=0
            )),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "confusion_matrix": confusion_matrix(
                y_test, pred, labels=[0, 1]
            ).tolist()
        }

        results[name] = scores
        print(f"\n{name}")

        for metric, value in scores.items():
            if metric != "confusion_matrix":
                print(f"{metric}: {value:.3f}")

        if scores["f1_score"] > best_f1:
            best_f1 = scores["f1_score"]
            best_model, best_name = model, name

    joblib.dump(
        best_model, MODEL_DIR / "credit_model.joblib"
    )

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "results": results
        }, f, indent=2)

    print(f"\nBest model: {best_name}")
    print("Model and evaluation metrics saved.")

if __name__ == "__main__":
    main()
