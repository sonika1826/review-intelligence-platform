from pathlib import Path

from src.pipelines.data_ingestion_pipeline import (
    DataIngestionPipeline,
)


def test_amazon_pipeline():

    pipeline = DataIngestionPipeline(
        dataset_type="amazon",
        dataset_path=Path("data/raw/amazon/All_Beauty.json"),
        max_reviews=20,
    )

    dataframe = pipeline.run()

    assert len(dataframe) > 0

    assert list(dataframe.columns) == [
        "review_id",
        "review_text",
        "rating",
        "sentiment",
        "source",
        "domain",
        "language",
        "created_at",
    ]

    assert dataframe["source"].eq("amazon").all()
