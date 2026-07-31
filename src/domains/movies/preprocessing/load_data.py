#src/domains/movies/preprocessing/load_data.py

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]

DATA_DIR = ROOT / "data" / "raw" / "movies"


def load_raw_data():

    movies = pd.read_csv(DATA_DIR / "movies.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    tags = pd.read_csv(DATA_DIR / "tags.csv")

    return movies, ratings, tags