import pandas as pd

from src.data.cleaners.data_cleaner import DataCleaner


def test_cleaner_removes_missing_reviews():

    dataframe = pd.DataFrame(
        {
            "review_id": ["1", "2", "3"],
            "review_text": [
                "Good product",
                None,
                "Excellent product",
            ],
            "rating": [5, 3, 5],
            "sentiment": [
                "positive",
                "neutral",
                "positive",
            ],
            "source": [
                "amazon",
                "amazon",
                "amazon",
            ],
            "domain": [
                "beauty",
                "beauty",
                "beauty",
            ],
            "language": [
                "en",
                "en",
                "en",
            ],
            "created_at": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
            ],
        }
    )

    cleaner = DataCleaner()

    cleaned = cleaner.clean(dataframe)

    assert len(cleaned) == 2
    assert cleaned["review_text"].isna().sum() == 0


def test_cleaner_removes_duplicate_ids():

    dataframe = pd.DataFrame(
        {
            "review_id": ["1", "1", "2"],
            "review_text": [
                "Good product",
                "Good product again",
                "Bad product",
            ],
            "rating": [5, 4, 1],
            "sentiment": [
                "positive",
                "positive",
                "negative",
            ],
            "source": [
                "amazon",
                "amazon",
                "amazon",
            ],
            "domain": [
                "beauty",
                "beauty",
                "beauty",
            ],
            "language": [
                "en",
                "en",
                "en",
            ],
            "created_at": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
            ],
        }
    )

    cleaner = DataCleaner()

    cleaned = cleaner.clean(dataframe)

    assert len(cleaned) == 2
    assert cleaned["review_id"].nunique() == 2
