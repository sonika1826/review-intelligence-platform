"""
Abstract base class for dataset loaders.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseLoader(ABC):
    """
    Abstract base class for dataset loaders.
    """

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load the dataset.

        Returns:
            Loaded pandas DataFrame.
        """
        raise NotImplementedError
