"""
Amazon dataset loader.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders.base_loader import BaseLoader
from src.utils.logger import logger


class AmazonLoader(BaseLoader):
    """
    Loader for Amazon Review datasets.
    """

    def __init__(
        self,
        dataset_path: Path,
        max_reviews: int | None = None,
    ) -> None:

        super().__init__(dataset_path, max_reviews)

    def load(self) -> pd.DataFrame:

        logger.info("=" * 60)
        logger.info("Loading Amazon dataset...")

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        dataframe = pd.read_json(
            self.dataset_path,
            lines=True,
        )

        if self.max_reviews is not None:
            dataframe = dataframe.head(self.max_reviews)

        logger.info("Amazon dataset loaded successfully.")

        logger.info(
            "Rows: %d | Columns: %d",
            dataframe.shape[0],
            dataframe.shape[1],
        )

        logger.info("=" * 60)

        return dataframe
