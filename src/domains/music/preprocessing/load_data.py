#src/domains/music/preprocessing/load_data.py

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]

DATA_DIR = ROOT / "data" / "raw" / "music"


def load_raw_data():

    music = pd.read_csv(
        DATA_DIR / "artists.csv",
        low_memory=False
    )

    return music