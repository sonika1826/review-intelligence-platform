"""
Data cleaning module for SentiScope AI.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import logger


class DataCleaner:
    """
    Cleans canonical review data before validation.
    """

    def clean(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the canonical review dataframe.

        Cleaning operations:
        - Remove completely duplicated rows.
        - Remove duplicate review IDs.
        - Remove missing review text.
        - Remove empty review text.
        - Remove whitespace-only review text.
        - Normalize review text to strings.
        - Remove invalid sentiment values.
        - Reset the dataframe index.
        """

        logger.info("=" * 60)
        logger.info("Starting data cleaning...")

        dataframe = dataframe.copy()

        initial_rows = len(dataframe)

        # --------------------------------------------------
        # 1. Remove completely duplicated rows
        # --------------------------------------------------

        dataframe = dataframe.drop_duplicates()

        # --------------------------------------------------
        # 2. Remove rows with missing review text
        # --------------------------------------------------

        dataframe = dataframe.dropna(subset=["review_text"])

        # --------------------------------------------------
        # 3. Convert review text to string
        # --------------------------------------------------

        dataframe["review_text"] = dataframe["review_text"].astype(str).str.strip()

        # --------------------------------------------------
        # 4. Remove empty / whitespace-only reviews
        # --------------------------------------------------

        dataframe = dataframe[dataframe["review_text"].ne("")]

        # --------------------------------------------------
        # 5. Remove duplicate review IDs
        # --------------------------------------------------

        if "review_id" in dataframe.columns:

            dataframe = dataframe.drop_duplicates(
                subset=["review_id"],
                keep="first",
            )

        # --------------------------------------------------
        # 6. Keep only supported sentiment values
        # --------------------------------------------------

        valid_sentiments = {
            "positive",
            "negative",
            "neutral",
        }

        if "sentiment" in dataframe.columns:

            dataframe = dataframe[dataframe["sentiment"].isin(valid_sentiments)]

        # --------------------------------------------------
        # 7. Reset index
        # --------------------------------------------------

        dataframe = dataframe.reset_index(drop=True)

        final_rows = len(dataframe)

        rows_removed = initial_rows - final_rows

        logger.info(
            "Removed %d rows.",
            rows_removed,
        )

        logger.info(
            "Final dataset shape: %s",
            dataframe.shape,
        )

        logger.info("Data cleaning completed.")

        logger.info("=" * 60)

        return dataframe
