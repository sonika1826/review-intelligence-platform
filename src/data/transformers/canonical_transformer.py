"""
Canonical data transformer for SentiScope AI.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import logger


class CanonicalTransformer:
    """
    Transform datasets into the canonical schema.
    """

    def transform_amazon(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Transform an Amazon dataset into the canonical schema.
        """

        logger.info("Transforming Amazon dataset to canonical schema.")

        transformed = dataframe.rename(
            columns={
                "rating": "rating",
                "review_text": "review_text",
                "review_id": "review_id",
            }
        ).copy()

        transformed["source"] = "amazon"
        transformed["domain"] = "ecommerce"

        transformed["language"] = "en"
        transformed["created_at"] = pd.NaT

        canonical_columns = [
            "review_id",
            "source",
            "domain",
            "review_text",
            "rating",
            "language",
            "created_at",
        ]

        transformed = transformed[canonical_columns]

        logger.info("Amazon dataset transformed successfully.")

        return transformed
