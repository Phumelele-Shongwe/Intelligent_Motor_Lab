# Import Dependancies
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from src.models import BearingCNN

#Load all trained bearing diagnosis models and CNN normalization parameters. 
def load_models(model_dir, device=None):

    model_dir = Path(model_dir)

    if device is None:
        device = torch.device("cpu")

    # Load traditional machine learning models
    logistic_regression = joblib.load(
        model_dir / "logistic_regression.joblib"
    )

    random_forest = joblib.load(
        model_dir / "random_forest.joblib"
    )

    # Load CNN normalization parameters
    with open(
        model_dir / "cnn_normalization.json",
        "r",
        encoding="utf-8"
    ) as file:
        cnn_normalization = json.load(file)

    # Load CNN architecture and trained weights
    cnn = BearingCNN().to(device)

    checkpoint = torch.load(
        model_dir / "bearing_cnn.pth",
        map_location=device,
        weights_only=True
    )

    cnn.load_state_dict(checkpoint["model_state_dict"])
    cnn.eval()

    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
        "cnn": cnn,
        "cnn_normalization": cnn_normalization,
        "device": device
    }

# Generate the probability of an Inner Race fault using a traditional machine learning model.

def predict_traditional(model, recording_features):
    """
    Generate the probability of an Inner Race fault
    using a traditional machine learning model.
    """

    # Models were trained using NumPy arrays without
    # feature-name metadata. Maintain that input format.
    if hasattr(recording_features, "to_numpy"):
        recording_features = recording_features.to_numpy(
            dtype=np.float64
        )
    else:
        recording_features = np.asarray(
            recording_features,
            dtype=np.float64
        )

    if recording_features.shape != (1, 3):
        raise ValueError(
            "Expected one recording with exactly three features."
        )

    probabilities = model.predict_proba(recording_features)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "The model does not contain the expected "
            "Inner Race class (1)."
        )

    ir_index = classes.index(1)

    return float(probabilities[0, ir_index])


#Predict Inner Race fault probabilities for raw vibration segments.
def predict_cnn(model, segments, device=None, batch_size=32):

    if device is None:
        device = torch.device("cpu")

    segments = np.asarray(segments, dtype=np.float32)

    if segments.ndim != 2 or segments.shape[0] == 0:
        raise ValueError(
            "Expected a non-empty array of shape "
            "(number_of_segments, 12000)."
        )

    model.eval()
    predictions = []

    with torch.no_grad():

        for start in range(0, len(segments), batch_size):

            batch = segments[start:start + batch_size]

            batch_tensor = torch.from_numpy(batch)
            batch_tensor = batch_tensor.unsqueeze(1).to(device)

            logits = model(batch_tensor)

            probabilities = torch.sigmoid(logits)

            predictions.extend(
                probabilities.cpu().numpy().tolist()
            )

    segment_probabilities = np.asarray(
        predictions,
        dtype=np.float32
    )

    recording_probability = float(
        np.mean(segment_probabilities)
    )

    return recording_probability, segment_probabilities