from pathlib import Path

from src.data.loaders.amazon_loader import AmazonLoader

AMAZON_PATH = Path("data/raw/amazon/All_Beauty.json")


def test_amazon_loader_loads_sample():

    loader = AmazonLoader(
        AMAZON_PATH,
        max_reviews=10,
    )

    dataframe = loader.load()

    assert len(dataframe) == 10
    assert "reviewText" in dataframe.columns
    assert "overall" in dataframe.columns
