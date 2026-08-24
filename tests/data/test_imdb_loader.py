from pathlib import Path

from src.data.loaders.imdb_loader import IMDBLoader

IMDB_PATH = Path("data/raw/imdb/aclImdb")


def test_imdb_loader_is_balanced():

    loader = IMDBLoader(
        IMDB_PATH,
        max_reviews=20,
    )

    dataframe = loader.load()

    assert len(dataframe) == 20

    sentiment_counts = dataframe["sentiment"].value_counts()

    assert sentiment_counts["positive"] == 10
    assert sentiment_counts["negative"] == 10
