
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

PROJECT_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab"
)

DATA_FILE = (
    PROJECT_ROOT / "Feature_Engineering"
    / "09_motor_features_binary.csv"
)

FEATURE_SETS = {
    "RMS only": ["rms"],
    "Crest factor only": ["crest_factor"],
    "Envelope frequency only": [
        "envelope_dominant_frequency"
    ],
    "All three": [
        "rms",
        "crest_factor",
        "envelope_dominant_frequency"
    ]
}

df = pd.read_csv(DATA_FILE)

recording_data = (
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

X_all = recording_data
y = recording_data["label"].to_numpy()

loo = LeaveOneOut()

results = []

print("\nFEATURE COMPARISON EXPERIMENT")
print("=" * 65)
print("Independent recordings:", len(recording_data))

for name, features in FEATURE_SETS.items():

    X = X_all[features]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            C=1.0
        ))
    ])

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

    results.append({
        "experiment": name,
        "features": len(features),
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y, predictions
        ),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y, probabilities)
    })

results_df = pd.DataFrame(results)

print("\nRESULTS")
print("=" * 65)
print(results_df.round(4).to_string(index=False))

OUTPUT_FILE = (
    PROJECT_ROOT / "Feature_Engineering"
    / "13_feature_comparison_results.csv"
)

results_df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
print("\nFEATURE COMPARISON COMPLETE")