# Intelligent Motor Laboratory
## Final Experimental Model Comparison

### 1. Experiment overview

This experiment compares three approaches to bearing fault
classification using vibration signals from the CWRU dataset:

- Logistic Regression using engineered statistical features.
- Random Forest using engineered statistical features.
- A one-dimensional Convolutional Neural Network (1D CNN)
  operating directly on raw vibration segments.

The experiment used 15 training recordings
and 5 held-out test recordings, containing
75 test segments.

### 2. Classification performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1D CNN | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

All three models achieved perfect classification metrics on
this particular held-out test set.

**Important limitation:** The test set contains only five
recordings, comprising one healthy recording and four faulty
recordings. These results are encouraging but do not establish
100% real-world diagnostic accuracy or generalization.

### 3. Computational efficiency

| Model | Mean latency (ms) | P95 latency (ms) | Recordings/s |
| --- | --- | --- | --- |
| Logistic Regression | 0.4843 | 1.0059 | 2065.03 |
| Random Forest | 31.9876 | 50.3542 | 31.26 |
| 1D CNN | 25.7014 | 70.7134 | 38.91 |

Inference was measured on the CPU, with one recording processed
per call. Model loading and signal preprocessing were excluded
from the timed inference operation.

The CNN processes all available segments belonging to a recording
and averages their predicted fault probabilities. Traditional
models operate on one median-aggregated feature vector per
recording.

Throughput represents the reciprocal of mean sequential
per-recording latency. It is not a parallel-processing benchmark.

### 4. Model storage

| Model | Storage (KB) |
| --- | --- |
| Logistic Regression | 1.41 |
| Random Forest | 189.06 |
| 1D CNN | 85.80 |

The CNN storage figure includes both its saved checkpoint and
the normalization parameters required for inference.

### 5. Engineering interpretation

**Logistic Regression:** Provides the lowest inference latency
and smallest storage footprint. Its input is a compact set of
engineered features, making it computationally efficient.

**Random Forest:** Captures nonlinear relationships between
engineered features but has the highest mean inference latency
and largest saved model size in this experiment.

**1D CNN:** Learns representations directly from vibration
segments rather than relying on manually selected features.
It achieved the same observed classification metrics and had
lower mean inference latency than Random Forest in this benchmark.
Its inference latency is more variable and depends on the number
of segments processed per recording.

### 6. Overall conclusion

The experiment demonstrates three viable approaches to
bearing fault classification, each with different engineering
trade-offs involving feature representation, model complexity,
storage, and inference efficiency.

Logistic Regression is attractive where computational efficiency
is the priority. The CNN is particularly interesting where
learning directly from vibration signals is desirable. Random
Forest provides a nonlinear traditional machine-learning
baseline.

Further evaluation on a larger and more diverse set of
independent recordings is required before drawing conclusions
about real-world deployment performance.

---

### Generated artifacts

- `results/metrics/final_model_comparison.csv`
- `results/figures/final_model_comparison.png`
- `results/final_experimental_comparison.md`
