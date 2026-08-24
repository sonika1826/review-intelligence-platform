"""
Abstract base class for dataset loaders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseLoader(ABC):
    """
    Abstract base class for all dataset loaders.
    """

    def __init__(
        self,
        dataset_path: Path,
        max_reviews: int | None = None,
    ) -> None:

        self.dataset_path = dataset_path
        self.max_reviews = max_reviews

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load dataset.

        Returns:
            pandas DataFrame.
        """
        raise NotImplementedError
