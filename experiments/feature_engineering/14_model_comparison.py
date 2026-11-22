
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

PROJECT_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab"
)

DATA_FILE = (
    PROJECT_ROOT / "Feature_Engineering"
    / "09_motor_features_binary.csv"
)

FEATURES = [
    "rms",
    "crest_factor",
    "envelope_dominant_frequency"
]

df = pd.read_csv(DATA_FILE)

data = (
    df.groupby("recording_id")
    .agg(
        rms=("rms", "median"),
        crest_factor=("crest_factor", "median"),
        envelope_dominant_frequency=(
            "envelope_dominant_frequency", "median"
        ),
        label=("label", "first")
    )
    .reset_index()
)

X = data[FEATURES]
y = data["label"].to_numpy()

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            C=1.0
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )
}

loo = LeaveOneOut()
all_results = []

print("\nMODEL COMPARISON")
print("=" * 65)
print("Independent recordings:", len(data))
print("Features:", FEATURES)

for name, model in models.items():

    predictions = []
    probabilities = []

    for train_idx, test_idx in loo.split(X, y):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]

        model.fit(X_train, y_train)

        predictions.append(model.predict(X_test)[0])
        probabilities.append(
            model.predict_proba(X_test)[0, 1]
        )

    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0

    result = {
        "model": name,
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y, predictions
        ),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "sensitivity": recall_score(
            y, predictions, zero_division=0
        ),
        "specificity": specificity,
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities)
    }

    all_results.append(result)

    print(f"\n{name}")
    print("-" * 50)
    for metric, value in result.items():
        if metric != "model":
            print(f"{metric}: {value:.4f}")

    print("Confusion matrix:")
    print(confusion_matrix(y, predictions, labels=[0, 1]))

results_df = pd.DataFrame(all_results)

OUTPUT_FILE = (
    PROJECT_ROOT / "Feature_Engineering"
    / "14_model_comparison_results.csv"
)

results_df.to_csv(OUTPUT_FILE, index=False)

print("\nSUMMARY")
print("=" * 65)
print(results_df.round(4).to_string(index=False))
print("\nSaved:", OUTPUT_FILE)
print("\nMODEL COMPARISON COMPLETE")