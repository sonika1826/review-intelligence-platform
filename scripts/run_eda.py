"""
Run exploratory data analysis for SentiScope AI.
"""

from pathlib import Path
import sys


# Add project root to Python path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.analysis.eda import ReviewEDA
from src.pipelines.data_ingestion_pipeline import (
    DataIngestionPipeline,
)


DATASETS = {
    "amazon": Path(
        "data/raw/amazon/All_Beauty.json"
    ),
    "imdb": Path(
        "data/raw/imdb/aclImdb"
    ),
    "twitter": Path(
        "data/raw/twitter/"
        "training.1600000.processed.noemoticon.csv"
    ),
}

MAX_REVIEWS = {
    "amazon": 50000,
    "imdb": 10000,
    "twitter": 100000,
}


def main() -> None:
    """Run EDA for all supported datasets."""

    eda = ReviewEDA(
        output_dir=Path("artifacts/eda")
    )

    for dataset_name, dataset_path in DATASETS.items():

        print()
        print("=" * 60)
        print(
            f"{dataset_name.upper()} EDA"
        )
        print("=" * 60)

        pipeline = DataIngestionPipeline(
            dataset_type=dataset_name,
            dataset_path=dataset_path,
            max_reviews=MAX_REVIEWS[
                dataset_name
            ],
        )

        dataframe = pipeline.run()

        summary = eda.summary(
            dataframe,
            dataset_name,
        )

        print()
        print("Dataset Summary")
        print("-" * 40)

        for key, value in summary.items():
            print(
                f"{key}: {value}"
            )

        eda.sentiment_distribution(
            dataframe,
            dataset_name,
        )

        eda.review_length_distribution(
            dataframe,
            dataset_name,
        )

        print()
        print(
            f"{dataset_name.capitalize()} EDA completed."
        )


if __name__ == "__main__":
    main()