# Intelligent Motor Laboratory

## CWRU Bearing Fault Diagnosis Using Signal Processing and Machine Learning

An experimental bearing-condition monitoring system that investigates the use of vibration signal processing, traditional machine learning, and one-dimensional convolutional neural networks (1D CNNs) to distinguish healthy bearings from inner-race faults.

The project combines classical vibration analysis with data-driven classification and provides a Streamlit application for exploring signals and evaluating trained models.

## 1. Research Objective

The objective is to investigate how vibration-based signal features and learned representations can support automated bearing fault diagnosis.

The current binary classification task considers:

- **Healthy (0):** Normal bearing condition.
- **Inner Race (1):** Inner-race bearing fault.

Three classifiers are evaluated:

1. Logistic Regression
2. Random Forest
3. One-dimensional Convolutional Neural Network (1D CNN)

The study compares their classification performance, model storage requirements, and inference latency.

## 2. Dataset

The project uses the Case Western Reserve University (CWRU) Bearing Data Center vibration dataset.

The local dataset is organised under:

`dataset/12k_Drive_End_Bearing_Fault_Data/`

The experimental workflow uses 12 kHz drive-end vibration recordings.

The selected binary experiment contains 20 recordings:

- 15 training recordings.
- 5 held-out test recordings.

The fixed test set contains one healthy recording and four inner-race fault recordings.

The recording-level split is important: segments from a recording are not distributed across both training and test sets in the final experiment.

**Dataset note:** The original CWRU recordings are not included in version control because of their size. Consult `dataset/README.md` for the dataset information and place the required recordings in the expected directory before running the experiments.

## 3. Signal Processing and Feature Engineering

The signal-processing workflow uses:

- Sampling frequency: 12,000 Hz.
- Fourth-order Butterworth bandpass filter.
- Passband: 2,000–5,000 Hz.
- Zero-phase filtering using `sosfiltfilt`.
- Hilbert transform for envelope analysis.
- Removal of 3,000 samples at each signal boundary.
- Non-overlapping windows of 12,000 samples (one second).

The traditional feature set includes:

- Root mean square (RMS).
- Crest factor.
- Envelope dominant frequency, evaluated over 0–500 Hz.

The project also contains earlier exploratory feature-engineering experiments and additional signal-analysis scripts.

## 4. Model Development

### Traditional machine learning

Two traditional classifiers are evaluated:

- Logistic Regression.
- Random Forest.

Traditional features are aggregated at recording level using the median of the segment-level feature values.

### Deep learning

The CNN operates on raw, preprocessed vibration segments of 12,000 samples.

The network uses three convolutional stages, batch normalisation, ReLU activations, max pooling, adaptive average pooling, and a binary classification head.

CNN normalisation statistics are calculated exclusively from training segments and reused during inference.

Segment probabilities are averaged to obtain a recording-level prediction.

The classification threshold is 0.5.

## 5. Experimental Methodology

The final fixed holdout experiment uses 20 recordings:

| Dataset partition | Recordings |
|---|---:|
| Training | 15 |
| Held-out testing | 5 |
| Total | 20 |

The final test set consists of:

- Healthy: `100_Normal_3.mat`
- Inner race: `105_0.mat`, `170_1.mat`, `211_2.mat`, `3004_3.mat`

The test recordings are excluded from fitting the final models in the fixed holdout experiment.

**Important evaluation limitation:** These recordings were also involved in earlier exploratory analyses. Therefore, this should not be described as a completely untouched test set across the entire project history. Furthermore, five test recordings provide limited evidence of generalisation. Perfect test metrics do not establish perfect real-world diagnostic performance.

## 6. Final Results

The final evaluation produced the following results on the five-recording holdout set:

| Metric | Logistic Regression | Random Forest | CNN |
|---|---:|---:|---:|
| Accuracy | 1.00 | 1.00 | 1.00 |
| Precision | 1.00 | 1.00 | 1.00 |
| Recall | 1.00 | 1.00 | 1.00 |
| Specificity | 1.00 | 1.00 | 1.00 |
| F1-score | 1.00 | 1.00 | 1.00 |
| ROC-AUC | 1.00 | 1.00 | 1.00 |

These results describe the selected holdout recordings only and should be interpreted in light of the small test sample.

The project also includes CPU inference benchmarking and model-size comparisons. The corresponding results are available in `results/`.

## 7. Repository Structure

```text
Intelligent_Motor_Lab/
│
├── app/
│   ├── app.py
│   └── dashboard/
│
├── dataset/
│   └── README.md
│
├── experiments/
│   ├── data_inspection/
│   ├── feature_engineering/
│   ├── dataset_preparation/
│   ├── model_training/
│   └── evaluation/
│
├── models/
│   ├── bearing_cnn.pth
│   ├── cnn_normalization.json
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── holdout_experiment_metadata.json
│
├── results/
│   ├── dataset_audit/
│   ├── exploratory/
│   ├── figures/
│   ├── metrics/
│   ├── predictions/
│   ├── prepared_data/
│   ├── recording_split.csv
│   └── final_experimental_comparison.md
│
├── src/
│   ├── feature_extraction.py
│   ├── inference.py
│   ├── models.py
│   └── preprocessing.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 8. Installation

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation should be compatible with the selected Python version and operating system.

## 9. Running the Application

From the project root:

```bash
streamlit run app/app.py
```

The application uses the trained model artifacts stored in `models/`.

Ensure the required model files and local dataset are available before using the relevant application functionality.

## 10. Reproducing the Experiments

The experiment scripts are grouped by research activity.

| Directory | Purpose |
|---|---|
| `experiments/data_inspection/` | Dataset inspection, signal analysis and recording audits |
| `experiments/feature_engineering/` | Feature extraction, analysis and traditional model experiments |
| `experiments/dataset_preparation/` | Recording-level splitting and preparation of training/test data |
| `experiments/model_training/` | Model development and training |
| `experiments/evaluation/` | Model verification, holdout evaluation, benchmarking and final comparisons |

The final evaluation can be reproduced using the existing trained models:

```bash
python experiments/evaluation/23_evaluate_existing_holdout.py
```

The comparison figures can be regenerated using:

```bash
python experiments/evaluation/24_generate_comparison_figures.py
```

The inference benchmark can be run using:

```bash
python experiments/evaluation/25_benchmark_inference.py
```

The final comparison report and figure can be regenerated using:

```bash
python experiments/evaluation/26_generate_final_comparison.py
```

**Training caution:** `experiments/model_training/21_train_and_evaluate_holdout.py` performs model training and can overwrite the saved model artifacts and associated metadata. Do not execute it merely to reproduce the existing evaluation.

## 11. Technologies

- Python
- NumPy and pandas
- SciPy
- scikit-learn
- PyTorch
- Matplotlib
- Streamlit
- Plotly

## 12. Research Scope and Future Work

The current work establishes a reproducible experimental pipeline for binary bearing fault classification.

Potential extensions include:

- Expanding the evaluation to additional fault classes.
- Testing on more independent recordings and operating conditions.
- Investigating domain generalisation across loads and rotational speeds.
- Evaluating robustness to noise and changing operating conditions.
- Exploring model compression and edge deployment.
- Investigating physics-informed features and bearing fault-frequency relationships.

The long-term objective is to move beyond controlled dataset classification towards reliable, interpretable and deployable condition-monitoring systems.