"""
Data ingestion pipeline for SentiScope AI.
"""

from pathlib import Path

import pandas as pd

from src.data.loaders.amazon_loader import AmazonLoader
from src.data.transformers.canonical_transformer import CanonicalTransformer
from src.data.validators.data_validator import DataValidator
from src.utils.logger import logger


class DataIngestionPipeline:
    """
    End-to-end pipeline for data ingestion.
    """

    def __init__(self, dataset_path: Path) -> None:
        self.loader = AmazonLoader(dataset_path)

        self.validator = DataValidator(
            required_columns=[
                "review_id",
                "review_text",
                "rating",
            ]
        )

        self.transformer = CanonicalTransformer()

    def run(self) -> pd.DataFrame:
        """
        Execute the complete ingestion pipeline.
        """

        logger.info("Starting data ingestion pipeline.")

        dataframe = self.loader.load()

        self.validator.validate(dataframe)

        dataframe = self.transformer.transform_amazon(dataframe)

        logger.info("Data ingestion pipeline completed successfully.")

        return dataframe
