import pandas as pd

from src.analysis.eda import ReviewEDA


def test_eda_summary():

    dataframe = pd.DataFrame(
        {
            "review_id": ["1", "2", "3"],
            "review_text": [
                "Excellent product",
                "Very bad product",
                "It is okay",
            ],
            "rating": [5, 1, 3],
            "sentiment": [
                "positive",
                "negative",
                "neutral",
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
            "created_at": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
        }
    )

    eda = ReviewEDA(output_dir="artifacts/eda_test")

    result = eda.summary(
        dataframe,
        "Amazon",
    )

    assert result["rows"] == 3
    assert result["positive"] == 1
    assert result["negative"] == 1
    assert result["neutral"] == 1


def test_sentiment_distribution_creates_file():

    dataframe = pd.DataFrame(
        {
            "review_text": [
                "Excellent",
                "Bad",
            ],
            "sentiment": [
                "positive",
                "negative",
            ],
        }
    )

    eda = ReviewEDA(output_dir="artifacts/eda_test")

    output = eda.sentiment_distribution(
        dataframe,
        "Amazon",
    )

    assert output.exists()
