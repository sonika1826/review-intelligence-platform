"""
Data validation module for SentiScope AI.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.logger import logger


@dataclass
class ValidationReport:
    """
    Result of dataset validation.
    """

    is_valid: bool
    rows: int
    columns: int
    duplicate_ids: int
    short_reviews: int
    long_reviews: int
    missing_values: dict[str, int]


class DataValidator:
    """
    Validate canonical SentiScope AI datasets.

    Required columns must exist.

    Only core fields are required to be non-null.

    Optional fields:
        - rating
        - created_at

    These can legitimately be missing depending
    on the source dataset.
    """

    REQUIRED_COLUMNS = [
        "review_id",
        "review_text",
        "rating",
        "sentiment",
        "source",
        "domain",
        "language",
        "created_at",
    ]

    NON_NULL_COLUMNS = [
        "review_id",
        "review_text",
        "sentiment",
        "source",
        "domain",
        "language",
    ]

    VALID_SENTIMENTS = {
        "positive",
        "negative",
        "neutral",
    }

    MIN_REVIEW_LENGTH = 3

    MAX_REVIEW_LENGTH = 10000

    def __init__(
        self,
        required_columns: list[str] | None = None,
    ) -> None:

        self.required_columns = (
            required_columns if required_columns is not None else self.REQUIRED_COLUMNS
        )

    def validate(
        self,
        dataframe: pd.DataFrame,
    ) -> ValidationReport:
        """
        Validate a canonical dataframe.
        """

        logger.info("=" * 60)
        logger.info("Starting dataset validation...")

        # --------------------------------------------------
        # 1. Validate dataframe type
        # --------------------------------------------------

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError("Expected pandas DataFrame.")

        # --------------------------------------------------
        # 2. Validate required columns
        # --------------------------------------------------

        missing_columns = [
            column
            for column in self.required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:

            logger.error(
                "Missing required columns: %s",
                missing_columns,
            )

            raise ValueError("Missing required columns: " f"{missing_columns}")

        # --------------------------------------------------
        # 3. Check missing values
        # --------------------------------------------------

        missing_values = {
            column: int(dataframe[column].isna().sum())
            for column in self.required_columns
        }

        invalid_columns = {
            column: count
            for column, count in missing_values.items()
            if column in self.NON_NULL_COLUMNS and count > 0
        }

        if invalid_columns:

            logger.error(
                "Missing values detected: %s",
                invalid_columns,
            )

            raise ValueError("Missing values found: " f"{invalid_columns}")

        # --------------------------------------------------
        # 4. Check duplicate review IDs
        # --------------------------------------------------

        duplicate_ids = int(dataframe["review_id"].duplicated().sum())

        # --------------------------------------------------
        # 5. Check review lengths
        # --------------------------------------------------

        review_lengths = dataframe["review_text"].astype(str).str.len()

        short_reviews = int((review_lengths < self.MIN_REVIEW_LENGTH).sum())

        long_reviews = int((review_lengths > self.MAX_REVIEW_LENGTH).sum())

        # --------------------------------------------------
        # 6. Check sentiment values
        # --------------------------------------------------

        invalid_sentiments = (
            set(dataframe["sentiment"].dropna().unique()) - self.VALID_SENTIMENTS
        )

        if invalid_sentiments:

            logger.error(
                "Invalid sentiment values: %s",
                invalid_sentiments,
            )

            raise ValueError("Invalid sentiment values found: " f"{invalid_sentiments}")

        # --------------------------------------------------
        # 7. Check duplicate IDs
        # --------------------------------------------------

        if duplicate_ids > 0:

            logger.warning(
                "Duplicate review IDs detected: %d",
                duplicate_ids,
            )

        # --------------------------------------------------
        # 8. Log validation summary
        # --------------------------------------------------

        logger.info("Validation Summary")

        logger.info(
            "Rows             : %d",
            len(dataframe),
        )

        logger.info(
            "Columns          : %d",
            len(dataframe.columns),
        )

        logger.info(
            "Duplicate IDs    : %d",
            duplicate_ids,
        )

        logger.info(
            "Short Reviews    : %d",
            short_reviews,
        )

        logger.info(
            "Long Reviews     : %d",
            long_reviews,
        )

        logger.info(
            "Missing Values   : %s",
            missing_values,
        )

        # --------------------------------------------------
        # 9. Determine validation status
        # --------------------------------------------------

        is_valid = len(dataframe) > 0 and not invalid_columns and not invalid_sentiments

        logger.info(
            "Validation Passed: %s",
            is_valid,
        )

        logger.info("=" * 60)

        return ValidationReport(
            is_valid=is_valid,
            rows=len(dataframe),
            columns=len(dataframe.columns),
            duplicate_ids=duplicate_ids,
            short_reviews=short_reviews,
            long_reviews=long_reviews,
            missing_values=missing_values,
        )
