"""
Fast and balanced IMDB dataset loader for SentiScope AI.
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
        - Reproducible sampling
        - Configurable max_reviews
        - Support for train/test directories
    """

    RANDOM_STATE = 42

    def load(self) -> pd.DataFrame:
        """
        Load a balanced IMDB sample.
        """

        logger.info("=" * 60)
        logger.info("Loading IMDB dataset...")

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        # --------------------------------------------------
        # Determine target sample size
        # --------------------------------------------------

        if self.max_reviews is None:
            target_total = None
        else:
            target_total = self.max_reviews

        # --------------------------------------------------
        # Collect files without reading their contents
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
        # Reproducible random sampling
        # --------------------------------------------------

        rng = random.Random(self.RANDOM_STATE)

        rng.shuffle(positive_files)
        rng.shuffle(negative_files)

        if target_total is None:

            selected_positive = positive_files
            selected_negative = negative_files

        else:

            positive_count = target_total // 2
            negative_count = target_total // 2

            selected_positive = positive_files[:positive_count]

            selected_negative = negative_files[:negative_count]

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
                    "positive",
                )
            )

        for review_file in selected_negative:

            records.append(
                self._read_review(
                    review_file,
                    "negative",
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
        """
        Read one IMDB review file.
        """

        filename = review_file.stem

        review_id, rating_text = filename.rsplit(
            "_",
            1,
        )

        try:
            rating = int(rating_text)
        except ValueError:
            rating = None

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
