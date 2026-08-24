import pandas as pd
import pytest

from src.data.validators.data_validator import DataValidator


def create_valid_dataframe():
    return pd.DataFrame(
        {
            "review_id": ["1", "2"],
            "review_text": [
                "Excellent product",
                "Very bad product",
            ],
            "rating": [5, 1],
            "sentiment": [
                "positive",
                "negative",
            ],
            "source": ["amazon", "amazon"],
            "domain": ["beauty", "beauty"],
            "language": ["en", "en"],
            "created_at": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )


def test_valid_dataframe_passes():
    validator = DataValidator()

    dataframe = create_valid_dataframe()

    report = validator.validate(dataframe)

    assert report.is_valid is True
    assert report.rows == 2
    assert report.columns == 8


def test_missing_review_text_fails():
    validator = DataValidator()

    dataframe = create_valid_dataframe()

    dataframe.loc[0, "review_text"] = None

    with pytest.raises(ValueError):
        validator.validate(dataframe)


def test_invalid_sentiment_fails():
    validator = DataValidator()

    dataframe = create_valid_dataframe()

    dataframe.loc[0, "sentiment"] = "unknown"

    with pytest.raises(ValueError):
        validator.validate(dataframe)


def test_optional_created_at_can_be_missing():
    validator = DataValidator()

    dataframe = create_valid_dataframe()

    dataframe["created_at"] = pd.NaT

    report = validator.validate(dataframe)

    assert report.is_valid is True
