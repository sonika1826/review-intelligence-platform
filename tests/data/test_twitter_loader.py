from pathlib import Path

from src.data.loaders.twitter_loader import TwitterLoader

TWITTER_PATH = Path("data/raw/twitter/" "training.1600000.processed.noemoticon.csv")


def test_twitter_loader_is_balanced():

    loader = TwitterLoader(
        TWITTER_PATH,
        max_reviews=20,
    )

    dataframe = loader.load()

    assert len(dataframe) == 20

    sentiment_counts = dataframe["sentiment"].value_counts()

    assert sentiment_counts[0] == 10
    assert sentiment_counts[4] == 10
