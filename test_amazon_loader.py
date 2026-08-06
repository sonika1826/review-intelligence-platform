from pathlib import Path

from src.data.loaders.amazon_loader import AmazonLoader

dataset_path = Path("data/raw/amazon/sample_reviews.csv")

loader = AmazonLoader(dataset_path)

dataframe = loader.load()

print(dataframe.head())
print(dataframe.shape)
