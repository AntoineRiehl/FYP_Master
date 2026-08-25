# src/domains/music/embed_music_reviews.py

# ============================================================
# MUSIC REVIEW EMBEDDINGS
#
# Thin domain-specific wrapper around the reusable review
# embedding engine.
#
# Input:
#
#   data/processed/reviews/
#       music_reviews_prepared.csv
#
# Output:
#
#   data/processed/embeddings/music/
#       music_review_embeddings.npz
#       music_review_embedding_index.csv
#       music_review_embeddings_metadata.json
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
    / "music_reviews_prepared.csv"
)


OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "music"
)


OUTPUT_NPZ = (
    OUTPUT_DIR
    / "music_review_embeddings.npz"
)


OUTPUT_INDEX = (
    OUTPUT_DIR
    / "music_review_embedding_index.csv"
)


OUTPUT_METADATA = (
    OUTPUT_DIR
    / "music_review_embeddings_metadata.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


# Keep the same maximum as Movies for consistency.
#
# In practice this barely affects Music because almost
# every artist has far fewer than 50 reviews.
MAX_REVIEWS_PER_ARTIST = 50


RANDOM_SEED = 42


# The RTX laptop GPU handled this configuration for Movies,
# so we can safely reuse it here.
BATCH_SIZE = 64


SELECTION_CHUNK_SIZE = 100_000

ENCODING_CHUNK_SIZE = 10_000


# Explicitly use the CUDA-enabled GPU environment that
# was already validated during the movie embedding run.
DEVICE = "cuda"


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

        # ----------------------------------------------------
        # Music-specific identifier
        # ----------------------------------------------------

        entity_id_column=
            "artist_mbid",

        review_id_column=
            "review_id",

        text_column=
            "review_text",

        # ----------------------------------------------------
        # Shared SBERT methodology
        # ----------------------------------------------------

        model_name=
            MODEL_NAME,

        max_reviews_per_entity=
            MAX_REVIEWS_PER_ARTIST,

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