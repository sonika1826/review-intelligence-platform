"""
Exploratory Data Analysis utilities for SentiScope AI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logger import logger


class ReviewEDA:
    """
    Perform exploratory analysis on canonical review data.
    """

    def __init__(
        self,
        output_dir: str | Path = Path("artifacts/eda"),
    ) -> None:
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def summary(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
    ) -> dict:
        """
        Generate a basic dataset summary.
        """

        if dataframe.empty:
            raise ValueError("Cannot analyze an empty dataframe.")

        review_lengths = dataframe["review_text"].astype(str).str.len()

        result = {
            "dataset": dataset_name,
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "positive": int((dataframe["sentiment"] == "positive").sum()),
            "negative": int((dataframe["sentiment"] == "negative").sum()),
            "neutral": int((dataframe["sentiment"] == "neutral").sum()),
            "average_review_length": float(review_lengths.mean()),
            "median_review_length": float(review_lengths.median()),
            "minimum_review_length": int(review_lengths.min()),
            "maximum_review_length": int(review_lengths.max()),
        }

        logger.info(
            "EDA summary for %s: %s",
            dataset_name,
            result,
        )

        return result

    def sentiment_distribution(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
    ) -> Path:
        """
        Create a sentiment distribution chart.
        """

        counts = dataframe["sentiment"].value_counts()

        output_path = self.output_dir / f"{dataset_name.lower()}_sentiment.png"

        plt.figure(figsize=(8, 5))

        counts.plot(kind="bar")

        plt.title(f"{dataset_name} Sentiment Distribution")

        plt.xlabel("Sentiment")
        plt.ylabel("Number of Reviews")

        plt.tight_layout()

        plt.savefig(output_path)

        plt.close()

        logger.info(
            "Saved sentiment chart: %s",
            output_path,
        )

        return output_path

    def review_length_distribution(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
    ) -> Path:
        """
        Create a review-length distribution chart.
        """

        lengths = dataframe["review_text"].astype(str).str.len()

        output_path = self.output_dir / f"{dataset_name.lower()}_review_length.png"

        plt.figure(figsize=(8, 5))

        lengths.plot(
            kind="hist",
            bins=50,
        )

        plt.title(f"{dataset_name} Review Length Distribution")

        plt.xlabel("Review Length")
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(output_path)

        plt.close()

        logger.info(
            "Saved review length chart: %s",
            output_path,
        )

        return output_path
