
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "09_motor_features.csv"

OUTPUT_DIR = BASE_DIR / "exercise_10_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# Explicitly identify the 20 engineered features.
# Metadata and labels must NOT be included here.

FEATURES = [
    "mean",
    "std",
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "skewness",
    "envelope_rms",
    "envelope_kurtosis",
    "envelope_peak",
    "bpfi_h1_amplitude",
    "bpfi_h2_amplitude",
    "bpfi_h3_amplitude",
    "envelope_dominant_frequency",
    "envelope_spectral_centroid",
    "envelope_band_0_100_energy",
    "envelope_band_100_200_energy",
    "envelope_band_200_350_energy",
    "envelope_band_350_500_energy",
]


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset():

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df = pd.read_csv(DATA_FILE)

    required = FEATURES + ["recording", "label"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Feature dataset contains invalid values.")

    return df, X


# ============================================================
# 3. FEATURE DISTRIBUTIONS
# ============================================================

def plot_feature_distributions(df):

    fig, axes = plt.subplots(
        4, 5,
        figsize=(18, 14)
    )

    axes = axes.flatten()

    for i, feature in enumerate(FEATURES):

        ax = axes[i]

        healthy = df.loc[
            df["label"] == 0, feature
        ]

        faulty = df.loc[
            df["label"] == 1, feature
        ]

        ax.boxplot(
            [healthy, faulty],
            tick_labels=["Healthy", "Faulty"],
            showmeans=True
        )

        ax.set_title(feature, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        "Feature Distributions: Healthy vs Inner-Race Fault",
        fontsize=15
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "10_feature_distributions.png",
        dpi=200
    )

    plt.show()


# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================

def calculate_feature_statistics(df):

    summary = df.groupby("recording")[FEATURES].agg(
        ["mean", "median", "std"]
    )

    summary.to_csv(
        OUTPUT_DIR / "10_feature_statistics.csv"
    )

    # Calculate the difference between class means.
    means = df.groupby("label")[FEATURES].mean()

    comparison = pd.DataFrame({
        "healthy_mean": means.loc[0],
        "faulty_mean": means.loc[1]
    })

    comparison["faulty_minus_healthy"] = (
        comparison["faulty_mean"]
        - comparison["healthy_mean"]
    )

    comparison.to_csv(
        OUTPUT_DIR / "10_feature_mean_comparison.csv"
    )

    print("\nFEATURE STATISTICS")
    print("=" * 70)

    print(summary.to_string())

    print("\nCLASS MEAN COMPARISON")
    print("=" * 70)

    print(comparison.to_string())


# ============================================================
# 5. POOLED FEATURE CORRELATION
# ============================================================

def plot_correlation_matrix(X, filename, title):

    corr = X.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(15, 12))

    image = ax.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))

    ax.set_xticklabels(
        FEATURES,
        rotation=90,
        fontsize=8
    )

    ax.set_yticklabels(
        FEATURES,
        fontsize=8
    )

    ax.set_title(title)

    fig.colorbar(
        image,
        ax=ax,
        label="Pearson correlation"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=200
    )

    plt.show()

    return corr


# ============================================================
# 6. IDENTIFY HIGHLY CORRELATED FEATURE PAIRS
# ============================================================

def find_correlated_features(corr, threshold=0.85):

    pairs = []

    for i in range(len(FEATURES)):

        for j in range(i + 1, len(FEATURES)):

            value = corr.iloc[i, j]

            if abs(value) >= threshold:

                pairs.append({
                    "feature_1": FEATURES[i],
                    "feature_2": FEATURES[j],
                    "correlation": value,
                    "absolute_correlation": abs(value)
                })

    result = pd.DataFrame(pairs)

    if not result.empty:

        result = result.sort_values(
            "absolute_correlation",
            ascending=False
        )

    result.to_csv(
        OUTPUT_DIR / "10_high_correlation_pairs.csv",
        index=False
    )

    print("\nHIGHLY CORRELATED FEATURE PAIRS")
    print("=" * 70)

    if result.empty:
        print("No feature pairs exceeded the threshold.")
    else:
        print(result.to_string(index=False))

    return result


# ============================================================
# 7. WITHIN-CLASS CORRELATION
# ============================================================

def calculate_within_class_correlation(df, X):

    # Remove each class's feature mean.
    # This helps separate within-class variation
    # from differences between the two classes.

    centered = X - X.groupby(
        df["label"].to_numpy()
    ).transform("mean")

    corr = centered.corr(method="pearson")

    plot_correlation_matrix(
        centered,
        "10_within_class_correlation.png",
        "Within-Class Feature Correlation"
    )

    return corr


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("Loading feature dataset...")

    df, X = load_dataset()

    print("\nDATASET INFORMATION")
    print("=" * 70)

    print("Number of observations:", len(df))
    print("Number of features:", len(FEATURES))

    print("\nClass distribution:")
    print(df["recording"].value_counts())

    # A. Feature distributions
    plot_feature_distributions(df)

    # B. Descriptive statistics
    calculate_feature_statistics(df)

    # C. Pooled correlation
    pooled_corr = plot_correlation_matrix(
        X,
        "10_pooled_correlation.png",
        "Pooled Feature Correlation"
    )

    # D. Identify highly correlated pairs
    find_correlated_features(
        pooled_corr,
        threshold=0.85
    )

    # E. Within-class correlation
    within_corr = calculate_within_class_correlation(
        df, X
    )

    find_correlated_features(
        within_corr,
        threshold=0.85
    )

    print("\nEXERCISE 10 COMPLETED")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()