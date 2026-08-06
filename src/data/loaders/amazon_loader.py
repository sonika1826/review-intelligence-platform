"""
Amazon dataset loader.
"""

from pathlib import Path

import pandas as pd

from src.data.loaders.base_loader import BaseLoader
from src.utils.logger import logger


class AmazonLoader(BaseLoader):
    """
    Loader for Amazon review datasets.
    """

    def __init__(self, dataset_path: Path) -> None:
        super().__init__(dataset_path)

    def load(self) -> pd.DataFrame:
        """
        Load the Amazon dataset.

        Returns:
            Loaded pandas DataFrame.

        Raises:
            FileNotFoundError:
                If the dataset file does not exist.
        """

        logger.info("Loading Amazon dataset.")

        if not self.dataset_path.exists():
            logger.error("Dataset not found: %s", self.dataset_path)
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        dataframe = pd.read_csv(self.dataset_path)

        logger.info(
            "Amazon dataset loaded successfully. Shape: %s",
            dataframe.shape,
        )

        return dataframe
