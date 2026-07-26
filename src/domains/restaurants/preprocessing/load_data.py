#src/domains/restaurants/preprocessing/load_data.py


from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]

DATA_DIR = ROOT / "data" / "raw" / "restaurants"


def load_raw_data():

    businesses = pd.read_json(
        DATA_DIR / "business.json",
        lines=True
    )

    reviews = pd.read_json(
        DATA_DIR / "review.json",
        lines=True
    )

    tips = pd.read_json(
        DATA_DIR / "tip.json",
        lines=True
    )

    return businesses, reviews, tips