"""
Data validation utilities for SentiScope AI.
"""

from typing import Iterable

import pandas as pd

from src.utils.logger import logger


class DataValidator:
    """
    Validates datasets before preprocessing.
    """

    def __init__(self, required_columns: Iterable[str]) -> None:
        self.required_columns = list(required_columns)

    def validate(self, dataframe: pd.DataFrame) -> None:
        """
        Validate the input dataset.

        Raises:
            ValueError: If validation fails.
        """

        logger.info("Starting dataset validation.")

        if dataframe.empty:
            raise ValueError("Dataset is empty.")

        missing_columns = [
            column
            for column in self.required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        null_columns = dataframe[self.required_columns].isnull().sum()

        invalid_columns = null_columns[null_columns > 0]

        if not invalid_columns.empty:
            raise ValueError(f"Missing values found:\n{invalid_columns}")

        if "review_id" in dataframe.columns:
            duplicate_count = dataframe["review_id"].duplicated().sum()

            if duplicate_count:
                raise ValueError(f"Found {duplicate_count} duplicate review_id values.")

        logger.info("Dataset validation completed successfully.")
