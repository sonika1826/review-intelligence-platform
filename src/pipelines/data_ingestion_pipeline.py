"""
Universal Data Ingestion Pipeline for SentiScope AI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.cleaners.data_cleaner import DataCleaner
from src.data.loaders.amazon_loader import AmazonLoader
from src.data.loaders.imdb_loader import IMDBLoader
from src.data.loaders.twitter_loader import TwitterLoader
from src.data.transformers.canonical_transformer import (
    CanonicalTransformer,
)
from src.data.validators.data_validator import DataValidator
from src.utils.logger import logger


class DataIngestionPipeline:
    """
    Universal data ingestion pipeline.

    Supports:
        - Amazon
        - IMDB
        - Twitter
    """

    def __init__(
        self,
        dataset_type: str,
        dataset_path: Path,
        max_reviews: int | None = None,
    ) -> None:

        self.dataset_type = dataset_type.lower()
        self.dataset_path = dataset_path
        self.max_reviews = max_reviews

        self.loader = self._get_loader()
        self.transformer = CanonicalTransformer()
        self.cleaner = DataCleaner()

        self.validator = DataValidator(
            required_columns=[
                "review_id",
                "review_text",
                "rating",
                "sentiment",
                "source",
                "domain",
                "language",
                "created_at",
            ]
        )

    def _get_loader(self):
        """
        Select the correct loader based on dataset type.
        """

        if self.dataset_type == "amazon":
            return AmazonLoader(
                self.dataset_path,
                self.max_reviews,
            )

        if self.dataset_type == "imdb":
            return IMDBLoader(
                self.dataset_path,
                self.max_reviews,
            )

        if self.dataset_type == "twitter":
            return TwitterLoader(
                self.dataset_path,
                self.max_reviews,
            )

        raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def run(self) -> pd.DataFrame:
        """
        Execute the complete ingestion pipeline.

        Flow:

        Load
            ↓
        Transform
            ↓
        Clean
            ↓
        Validate
            ↓
        Return canonical DataFrame
        """

        logger.info("=" * 60)

        logger.info(
            "Starting %s ingestion pipeline...",
            self.dataset_type.upper(),
        )

        # 1. Load raw dataset
        dataframe = self.loader.load()

        # 2. Transform to canonical schema
        dataframe = self.transformer.transform(
            dataframe,
            self.dataset_type,
        )

        # 3. Clean data
        dataframe = self.cleaner.clean(dataframe)

        # 4. Validate canonical data
        report = self.validator.validate(dataframe)

        logger.info(
            "Validation Passed: %s",
            report.is_valid,
        )

        logger.info(
            "Final dataset shape: %s",
            dataframe.shape,
        )

        logger.info("Pipeline completed successfully.")

        logger.info("=" * 60)

        return dataframe
