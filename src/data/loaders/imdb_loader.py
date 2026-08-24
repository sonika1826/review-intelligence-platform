"""
Balanced and optimized IMDB dataset loader for SentiScope AI.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from src.data.loaders.base_loader import BaseLoader
from src.utils.logger import logger


class IMDBLoader(BaseLoader):
    """
    Loader for the IMDB Large Movie Review Dataset.

    Features:
        - Balanced positive/negative sampling
        - Globally unique review IDs
        - Reproducible sampling
        - Configurable max_reviews
    """

    RANDOM_STATE = 42

    def load(self) -> pd.DataFrame:
        """Load a balanced IMDB sample."""

        logger.info("=" * 60)
        logger.info("Loading IMDB dataset...")

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        # --------------------------------------------------
        # Determine target sample size
        # --------------------------------------------------

        if self.max_reviews is None:
            target_per_sentiment = None
        else:
            target_per_sentiment = self.max_reviews // 2

        # --------------------------------------------------
        # Collect review files
        # --------------------------------------------------

        positive_files: list[Path] = []
        negative_files: list[Path] = []

        for split in ("train", "test"):

            split_path = self.dataset_path / split

            if not split_path.exists():
                raise FileNotFoundError(f"IMDB split not found: {split_path}")

            positive_files.extend((split_path / "pos").glob("*.txt"))

            negative_files.extend((split_path / "neg").glob("*.txt"))

        # --------------------------------------------------
        # Reproducible shuffle
        # --------------------------------------------------

        rng = random.Random(self.RANDOM_STATE)

        rng.shuffle(positive_files)
        rng.shuffle(negative_files)

        # --------------------------------------------------
        # Select balanced sample
        # --------------------------------------------------

        if target_per_sentiment is None:

            selected_positive = positive_files
            selected_negative = negative_files

        else:

            selected_positive = positive_files[:target_per_sentiment]

            selected_negative = negative_files[:target_per_sentiment]

        logger.info(
            "Selected %d positive reviews.",
            len(selected_positive),
        )

        logger.info(
            "Selected %d negative reviews.",
            len(selected_negative),
        )

        # --------------------------------------------------
        # Read selected reviews
        # --------------------------------------------------

        records: list[dict] = []

        for review_file in selected_positive:

            records.append(
                self._read_review(
                    review_file,
                    sentiment="positive",
                )
            )

        for review_file in selected_negative:

            records.append(
                self._read_review(
                    review_file,
                    sentiment="negative",
                )
            )

        dataframe = pd.DataFrame(records)

        # --------------------------------------------------
        # Shuffle final dataset
        # --------------------------------------------------

        dataframe = dataframe.sample(
            frac=1,
            random_state=self.RANDOM_STATE,
        ).reset_index(drop=True)

        logger.info("IMDB dataset loaded successfully.")

        logger.info(
            "Rows: %d | Columns: %d",
            dataframe.shape[0],
            dataframe.shape[1],
        )

        logger.info(
            "Sentiment distribution:\n%s",
            dataframe["sentiment"].value_counts(),
        )

        logger.info("=" * 60)

        return dataframe

    @staticmethod
    def _read_review(
        review_file: Path,
        sentiment: str,
    ) -> dict:
        """Read one IMDB review and create a unique ID."""

        # --------------------------------------------------
        # Extract rating from filename
        #
        # Example:
        # 12345_8.txt
        # --------------------------------------------------

        filename = review_file.stem

        try:
            review_number, rating_text = filename.rsplit(
                "_",
                1,
            )

            rating = int(rating_text)

        except ValueError:
            review_number = filename
            rating = None

        # --------------------------------------------------
        # Create globally unique review ID
        # --------------------------------------------------

        split = review_file.parent.parent.name
        sentiment_folder = review_file.parent.name

        review_id = f"imdb_{split}_" f"{sentiment_folder}_" f"{review_number}"

        # --------------------------------------------------
        # Read review text
        # --------------------------------------------------

        review_text = review_file.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

        return {
            "review_id": review_id,
            "review_text": review_text,
            "rating": rating,
            "sentiment": sentiment,
        }
