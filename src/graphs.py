"""
This file will take the datasets created in the pipeline and train the models.
It will compute many graphs and metrics to evaluate the models on the val set.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_pearson_correlation(
    features_csv: str = "../datasets/processed/djmix_dataset_partition_features.csv",
):
    """
    Plot and display a Pearson correlation heatmap from features.csv.
    """
    csv_path = Path(features_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find features file at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Keep only numeric columns to compute Pearson correlation.
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        raise RuntimeError("No numeric columns found in features.csv for correlation.")

    corr = numeric_df.corr(method="pearson")

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.title("Pearson Correlation Heatmap")
    plt.tight_layout()

    plt.show()
    return corr


if __name__ == "__main__":
    plot_pearson_correlation()
    print("Pearson correlation heatmap displayed successfully.")