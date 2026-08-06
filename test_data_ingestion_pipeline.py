from pathlib import Path

from src.pipelines.data_ingestion_pipeline import (
    DataIngestionPipeline,
)

pipeline = DataIngestionPipeline(Path("data/raw/amazon/sample_reviews.csv"))

canonical_df = pipeline.run()

print(canonical_df)
