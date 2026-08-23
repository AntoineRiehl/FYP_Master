# src/domains/movies/embed_movie_reviews.py

# ============================================================
# MOVIE REVIEW EMBEDDINGS
#
# Thin domain-specific wrapper around the reusable review
# embedding engine.
#
# ============================================================


from pathlib import Path

from src.reviews.embeddings.review_embeddings import (
    build_review_embeddings
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "reviews"
    / "movies_reviews_prepared.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "movies"
)

OUTPUT_NPZ = (
    OUTPUT_DIR
    / "movie_review_embeddings.npz"
)

OUTPUT_INDEX = (
    OUTPUT_DIR
    / "movie_review_embedding_index.csv"
)

OUTPUT_METADATA = (
    OUTPUT_DIR
    / "movie_review_embeddings_metadata.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

MAX_REVIEWS_PER_MOVIE = 50

RANDOM_SEED = 42

BATCH_SIZE = 64

SELECTION_CHUNK_SIZE = 100_000

ENCODING_CHUNK_SIZE = 10_000

# None allows SentenceTransformer / PyTorch to choose
# the available device automatically.
DEVICE = None


# ============================================================
# MAIN
# ============================================================


def main():

    build_review_embeddings(

        input_file=
            INPUT_FILE,

        output_npz=
            OUTPUT_NPZ,

        output_index_csv=
            OUTPUT_INDEX,

        output_metadata_json=
            OUTPUT_METADATA,

        entity_id_column=
            "movieId",

        review_id_column=
            "review_id",

        text_column=
            "review_text",

        model_name=
            MODEL_NAME,

        max_reviews_per_entity=
            MAX_REVIEWS_PER_MOVIE,

        random_seed=
            RANDOM_SEED,

        batch_size=
            BATCH_SIZE,

        selection_chunk_size=
            SELECTION_CHUNK_SIZE,

        encoding_chunk_size=
            ENCODING_CHUNK_SIZE,

        device=
            DEVICE

    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()