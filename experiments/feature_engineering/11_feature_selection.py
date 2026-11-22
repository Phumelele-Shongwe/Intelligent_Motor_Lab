
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "09_motor_features.csv"

OUTPUT_DIR = BASE_DIR / "exercise_11_results"
OUTPUT_DIR.mkdir(exist_ok=True)

HEALTHY_LABEL = 0
FAULTY_LABEL = 1

CORRELATION_THRESHOLD = 0.95

# These columns describe the recording or operating conditions.
# They are not vibration-derived predictor features.
METADATA_COLUMNS = [
    "recording",
    "segment",
    "label",
    "rpm",
    "shaft_frequency",
    "bpfi_order",
]


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    print("\nLoading feature dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df = pd.read_csv(DATA_FILE)

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a label column.")

    print("\nDATASET INFORMATION")
    print("=" * 65)

    print(f"Observations: {len(df)}")

    feature_columns = [
        col for col in df.select_dtypes(include=np.number).columns
        if col not in METADATA_COLUMNS
    ]

    print(f"Predictor features: {len(feature_columns)}")

    print("\nCLASS DISTRIBUTION")
    print(df["recording"].value_counts())

    return df, feature_columns


# ============================================================
# 3. CLASS-WISE STATISTICS
# ============================================================

def calculate_class_statistics(df, features):

    print("\nCLASS-WISE FEATURE STATISTICS")
    print("=" * 65)

    healthy = df[df["label"] == HEALTHY_LABEL]
    faulty = df[df["label"] == FAULTY_LABEL]

    statistics = []

    for feature in features:

        h_mean = healthy[feature].mean()
        f_mean = faulty[feature].mean()

        h_std = healthy[feature].std()
        f_std = faulty[feature].std()

        statistics.append({
            "feature": feature,
            "healthy_mean": h_mean,
            "faulty_mean": f_mean,
            "healthy_std": h_std,
            "faulty_std": f_std,
            "mean_difference": f_mean - h_mean,
        })

    result = pd.DataFrame(statistics)

    print(result.to_string(index=False))

    result.to_csv(
        OUTPUT_DIR / "class_statistics.csv",
        index=False
    )

    return result


# ============================================================
# 4. STANDARDIZED MEAN DIFFERENCE (COHEN'S D)
# ============================================================

def calculate_effect_sizes(df, features):

    print("\nFEATURE DISCRIMINATION: COHEN'S D")
    print("=" * 65)

    healthy = df[df["label"] == HEALTHY_LABEL]
    faulty = df[df["label"] == FAULTY_LABEL]

    n_h = len(healthy)
    n_f = len(faulty)

    results = []

    for feature in features:

        h = healthy[feature].dropna()
        f = faulty[feature].dropna()

        nh = len(h)
        nf = len(f)

        if nh < 2 or nf < 2:
            continue

        h_mean = h.mean()
        f_mean = f.mean()

        h_std = h.std(ddof=1)
        f_std = f.std(ddof=1)

        pooled_std = np.sqrt(
            (
                (nh - 1) * h_std**2
                + (nf - 1) * f_std**2
            ) / (nh + nf - 2)
        )

        if pooled_std == 0:
            cohens_d = np.nan
        else:
            cohens_d = (f_mean - h_mean) / pooled_std

        results.append({
            "feature": feature,
            "cohens_d": cohens_d,
            "absolute_effect": abs(cohens_d),
        })

    result = pd.DataFrame(results)

    result = result.sort_values(
        "absolute_effect",
        ascending=False
    )

    print(result.to_string(index=False))

    result.to_csv(
        OUTPUT_DIR / "feature_effect_sizes.csv",
        index=False
    )

    return result


# ============================================================
# 5. CORRELATION ANALYSIS
# ============================================================

def calculate_correlations(df, features):

    print("\nCORRELATION ANALYSIS")
    print("=" * 65)

    datasets = {
        "Overall": df,
        "Healthy": df[df["label"] == HEALTHY_LABEL],
        "Faulty": df[df["label"] == FAULTY_LABEL],
    }

    correlation_matrices = {}

    for name, subset in datasets.items():

        corr = subset[features].corr(method="pearson")

        correlation_matrices[name] = corr

        corr.to_csv(
            OUTPUT_DIR / f"correlation_{name.lower()}.csv"
        )

        print(f"\n{name.upper()}")

        pairs = (
            corr.where(
                np.triu(
                    np.ones(corr.shape),
                    k=1
                ).astype(bool)
            )
            .stack()
            .rename("correlation")
            .reset_index()
        )

        pairs.columns = [
            "feature_1",
            "feature_2",
            "correlation"
        ]

        pairs["absolute_correlation"] = (
            pairs["correlation"].abs()
        )

        pairs = pairs.sort_values(
            "absolute_correlation",
            ascending=False
        )

        highly_correlated = pairs[
            pairs["absolute_correlation"]
            >= CORRELATION_THRESHOLD
        ]

        print(
            f"Pairs above threshold "
            f"({CORRELATION_THRESHOLD}): "
            f"{len(highly_correlated)}"
        )

        print(highly_correlated.head(15).to_string(index=False))

        pairs.to_csv(
            OUTPUT_DIR / f"correlation_pairs_{name.lower()}.csv",
            index=False
        )

    return correlation_matrices


# ============================================================
# 6. CORRELATION HEATMAPS
# ============================================================

def plot_correlation_heatmaps(matrices):

    print("\nGenerating correlation heatmaps...")

    for name, corr in matrices.items():

        fig, ax = plt.subplots(figsize=(15, 12))

        image = ax.imshow(
            corr,
            cmap="coolwarm",
            vmin=-1,
            vmax=1
        )

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))

        ax.set_xticklabels(
            corr.columns,
            rotation=90,
            fontsize=8
        )

        ax.set_yticklabels(
            corr.index,
            fontsize=8
        )

        ax.set_title(
            f"{name}: Feature Correlation Heatmap"
        )

        fig.colorbar(
            image,
            ax=ax,
            label="Pearson Correlation"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"heatmap_{name.lower()}.png",
            dpi=200
        )

        plt.close()


# ============================================================
# 7. EFFECT SIZE VISUALIZATION
# ============================================================

def plot_effect_sizes(effect_sizes):

    print("\nGenerating feature discrimination plot...")

    plot_data = effect_sizes.sort_values(
        "absolute_effect",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(11, 9))

    ax.barh(
        plot_data["feature"],
        plot_data["absolute_effect"]
    )

    ax.set_xlabel("Absolute Cohen's d")
    ax.set_ylabel("Feature")

    ax.set_title(
        "Feature Discrimination: Healthy vs Faulty"
    )

    ax.grid(
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "feature_discrimination.png",
        dpi=200
    )

    plt.close()


# ============================================================
# 8. MAIN
# ============================================================

def main():

    df, features = load_data()

    statistics = calculate_class_statistics(
        df, features
    )

    effect_sizes = calculate_effect_sizes(
        df, features
    )

    correlations = calculate_correlations(
        df, features
    )

    plot_correlation_heatmaps(
        correlations
    )

    plot_effect_sizes(
        effect_sizes
    )

    print("\n" + "=" * 65)
    print("EXERCISE 11 COMPLETED")
    print("=" * 65)

    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()