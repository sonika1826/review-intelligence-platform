"""
Canonical data transformer for SentiScope AI.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import logger


class CanonicalTransformer:
    """
    Transform source-specific datasets into the
    SentiScope AI canonical schema.
    """

    CANONICAL_COLUMNS = [
        "review_id",
        "review_text",
        "rating",
        "sentiment",
        "source",
        "domain",
        "language",
        "created_at",
    ]

    def transform(
        self,
        dataframe: pd.DataFrame,
        dataset_type: str,
    ) -> pd.DataFrame:
        """
        Transform a source dataframe into the
        canonical SentiScope AI schema.
        """

        dataset_type = dataset_type.lower()

        logger.info("=" * 60)
        logger.info(
            "Transforming %s dataset...",
            dataset_type,
        )

        if dataset_type == "amazon":
            transformed = self._transform_amazon(dataframe)

        elif dataset_type == "imdb":
            transformed = self._transform_imdb(dataframe)

        elif dataset_type == "twitter":
            transformed = self._transform_twitter(dataframe)

        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        transformed = transformed[self.CANONICAL_COLUMNS]

        logger.info(
            "%s dataset transformed successfully.",
            dataset_type.capitalize(),
        )

        logger.info("=" * 60)

        return transformed

    def _transform_amazon(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform Amazon reviews."""

        transformed = pd.DataFrame()

        transformed["review_id"] = dataframe["reviewerID"].astype(str)

        transformed["review_text"] = dataframe["reviewText"]

        transformed["rating"] = pd.to_numeric(
            dataframe["overall"],
            errors="coerce",
        )

        transformed["sentiment"] = transformed["rating"].apply(
            self._rating_to_sentiment
        )

        transformed["source"] = "amazon"

        transformed["domain"] = "beauty"

        transformed["language"] = "en"

        transformed["created_at"] = pd.to_datetime(
            dataframe["reviewTime"],
            format="%m %d, %Y",
            errors="coerce",
        )

        return transformed

    def _transform_imdb(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform IMDB reviews."""

        transformed = pd.DataFrame()

        transformed["review_id"] = dataframe["review_id"].astype(str)

        transformed["review_text"] = dataframe["review_text"]

        transformed["rating"] = pd.to_numeric(
            dataframe["rating"],
            errors="coerce",
        )

        transformed["sentiment"] = dataframe["sentiment"].astype(str).str.lower()

        transformed["source"] = "imdb"

        transformed["domain"] = "movies"

        transformed["language"] = "en"

        # IMDB dataset does not provide review dates.
        transformed["created_at"] = pd.NaT

        return transformed

    def _transform_twitter(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform Sentiment140 Twitter data."""

        transformed = pd.DataFrame()

        transformed["review_id"] = dataframe["review_id"].astype(str)

        transformed["review_text"] = dataframe["review_text"]

        # Twitter/Sentiment140 does not contain
        # star ratings.
        transformed["rating"] = pd.NA

        transformed["sentiment"] = dataframe["sentiment"].map(
            {
                0: "negative",
                4: "positive",
            }
        )

        transformed["source"] = "twitter"

        transformed["domain"] = "social_media"

        transformed["language"] = "en"

        # Sentiment140 date format:
        # "Mon Apr 06 22:19:45 PDT 2009"
        transformed["created_at"] = pd.to_datetime(
            dataframe["created_at"],
            format="%a %b %d %H:%M:%S %Z %Y",
            errors="coerce",
            utc=True,
        )

        return transformed

    @staticmethod
    def _rating_to_sentiment(
        rating: float | int | None,
    ) -> str:
        """Convert star rating to sentiment."""

        if pd.isna(rating):
            return "neutral"

        if rating >= 4:
            return "positive"

        if rating <= 2:
            return "negative"

        return "neutral"
