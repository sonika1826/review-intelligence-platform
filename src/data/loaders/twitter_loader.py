"""
Balanced Twitter/Sentiment140 dataset loader for SentiScope AI.
"""

from __future__ import annotations

import pandas as pd

from src.data.loaders.base_loader import BaseLoader
from src.utils.logger import logger


class TwitterLoader(BaseLoader):
    """
    Loader for the Sentiment140 Twitter dataset.

    Creates a balanced positive/negative development sample.
    """

    RANDOM_STATE = 42

    COLUMNS = [
        "sentiment",
        "review_id",
        "created_at",
        "query",
        "user",
        "review_text",
    ]

    def load(self) -> pd.DataFrame:
        """
        Load a balanced Twitter dataset.
        """

        logger.info("=" * 60)
        logger.info("Loading Twitter dataset...")

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        # --------------------------------------------------
        # Determine sample size
        # --------------------------------------------------

        if self.max_reviews is None:
            sample_size = None
        else:
            sample_size = self.max_reviews // 2

        # --------------------------------------------------
        # Read dataset
        # --------------------------------------------------

        dataframe = pd.read_csv(
            self.dataset_path,
            encoding="latin-1",
            header=None,
            names=self.COLUMNS,
            usecols=range(6),
        )

        # --------------------------------------------------
        # Keep only positive and negative tweets
        #
        # Sentiment140:
        # 0 = negative
        # 4 = positive
        # --------------------------------------------------

        negative = dataframe[dataframe["sentiment"] == 0].copy()

        positive = dataframe[dataframe["sentiment"] == 4].copy()

        # --------------------------------------------------
        # Balanced sampling
        # --------------------------------------------------

        if sample_size is not None:

            negative = negative.sample(
                n=min(
                    sample_size,
                    len(negative),
                ),
                random_state=self.RANDOM_STATE,
            )

            positive = positive.sample(
                n=min(
                    sample_size,
                    len(positive),
                ),
                random_state=self.RANDOM_STATE,
            )

        # --------------------------------------------------
        # Combine samples
        # --------------------------------------------------

        dataframe = pd.concat(
            [negative, positive],
            ignore_index=True,
        )

        # --------------------------------------------------
        # Shuffle final dataset
        # --------------------------------------------------

        dataframe = dataframe.sample(
            frac=1,
            random_state=self.RANDOM_STATE,
        ).reset_index(drop=True)

        logger.info("Twitter dataset loaded successfully.")

        logger.info(
            "Rows: %d | Columns: %d",
            dataframe.shape[0],
            dataframe.shape[1],
        )

        logger.info(
            "Raw sentiment distribution:\n%s",
            dataframe["sentiment"].value_counts(),
        )

        logger.info("=" * 60)

        return dataframe
