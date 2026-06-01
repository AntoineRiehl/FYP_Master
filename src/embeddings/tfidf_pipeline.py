from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "embeddings"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


# =========================================================
# BUILD OR LOAD TF-IDF
# =========================================================

def get_tfidf_embeddings(df):

    # -----------------------------------------------------
    # LOAD EXISTING VECTORIZER
    # -----------------------------------------------------

    if TFIDF_PATH.exists():

        print("Loading existing TF-IDF vectorizer...")

        vectorizer = joblib.load(TFIDF_PATH)

    else:

        print("Training new TF-IDF vectorizer...")

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        vectorizer.fit(df["tags_text"])

        joblib.dump(vectorizer, TFIDF_PATH)

        print("TF-IDF vectorizer saved.")

    # -----------------------------------------------------
    # TRANSFORM MOVIES
    # -----------------------------------------------------

    tfidf_matrix = vectorizer.transform(
        df["tags_text"]
    )

    return tfidf_matrix, vectorizer