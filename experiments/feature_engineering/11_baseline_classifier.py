import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab"
)

DATA_FILE = (
    PROJECT_ROOT
    / "Feature_Engineering"
    / "09_motor_features_binary.csv"
)

FEATURES = [
    "rms",
    "crest_factor",
    "envelope_dominant_frequency",
]

TARGET = "label"
GROUP = "recording_id"


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\nBASELINE CLASSIFICATION")
print("=" * 65)

print("Segment observations:", len(df))
print("Source recordings:", df[GROUP].nunique())


# ============================================================
# 3. AGGREGATE SEGMENTS BY RECORDING
# ============================================================

recording_data = (
    df.groupby(GROUP)
    .agg(
        rms=("rms", "median"),
        crest_factor=("crest_factor", "median"),
        envelope_dominant_frequency=(
            "envelope_dominant_frequency", "median"
        ),
        label=("label", "first"),
        fault_type=("fault_type", "first"),
    )
    .reset_index()
)

X = recording_data[FEATURES]
y = recording_data[TARGET]

print("\nRECORDING-LEVEL DATASET")
print("=" * 65)

print("Observations:", len(recording_data))
print("Features:", len(FEATURES))

print("\nClass distribution:")
print(recording_data["fault_type"].value_counts())


# ============================================================
# 4. LEAVE-ONE-RECORDING-OUT VALIDATION
# ============================================================

loo = LeaveOneOut()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        C=1.0
    ))
])

dummy = DummyClassifier(strategy="prior")

predictions = []
probabilities = []
dummy_predictions = []

for train_idx, test_idx in loo.split(X, y):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)[0]

    probability = model.predict_proba(X_test)[0, 1]

    predictions.append(prediction)
    probabilities.append(probability)

    dummy.fit(X_train, y_train)

    dummy_predictions.append(
        dummy.predict(X_test)[0]
    )


# ============================================================
# 5. EVALUATION
# ============================================================

y_true = y.to_numpy()
y_pred = np.asarray(predictions)
y_probability = np.asarray(probabilities)

dummy_pred = np.asarray(dummy_predictions)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0 else 0
)

print("\nLOGISTIC REGRESSION RESULTS")
print("=" * 65)

print("Accuracy:",
      accuracy_score(y_true, y_pred))

print("Balanced accuracy:",
      balanced_accuracy_score(y_true, y_pred))

print("Precision:",
      precision_score(y_true, y_pred, zero_division=0))

print("Recall / Sensitivity:",
      recall_score(y_true, y_pred, zero_division=0))

print("Specificity:", specificity)

print("F1-score:",
      f1_score(y_true, y_pred, zero_division=0))

print("ROC-AUC:",
      roc_auc_score(y_true, y_probability))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Healthy", "Inner Race"],
    zero_division=0
))


# ============================================================
# 6. COMPARE AGAINST A SIMPLE BASELINE
# ============================================================

print("\nDUMMY CLASSIFIER COMPARISON")
print("=" * 65)

print("Dummy accuracy:",
      accuracy_score(y_true, dummy_pred))

print("Dummy balanced accuracy:",
      balanced_accuracy_score(y_true, dummy_pred))


# ============================================================
# 7. RECORDING-LEVEL PREDICTIONS
# ============================================================

recording_results = recording_data[
    [GROUP, "fault_type", TARGET]
].copy()

recording_results["predicted_label"] = y_pred

recording_results["fault_probability"] = y_probability

recording_results["correct"] = (
    recording_results[TARGET]
    == recording_results["predicted_label"]
)

print("\nINDIVIDUAL RECORDING PREDICTIONS")
print("=" * 65)

print(
    recording_results.to_string(index=False)
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "Feature_Engineering"
    / "11_baseline_predictions.csv"
)

recording_results.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved predictions:", OUTPUT_FILE)

print("\nBASELINE EXPERIMENT COMPLETE")