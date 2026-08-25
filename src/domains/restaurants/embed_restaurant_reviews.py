# src/domains/restaurants/embed_restaurant_reviews.py

# ============================================================
# RESTAURANT REVIEW EMBEDDINGS
#
# Thin domain-specific wrapper around the reusable review
# embedding engine.
#
#
# Input:
#
#   data/processed/reviews/
#       restaurant_reviews_prepared.csv
#
#
# Outputs:
#
#   data/processed/embeddings/restaurants/
#       restaurant_review_embeddings.npz
#       restaurant_review_embedding_index.csv
#       restaurant_review_embeddings_metadata.json
#
#
# Method:
#
#   - entity identifier:
#         business_id
#
#   - review identifier:
#         review_id
#
#   - text:
#         review_text
#
#   - model:
#         sentence-transformers/all-MiniLM-L6-v2
#
#   - maximum reviews per restaurant:
#         50
#
#   - deterministic review sampling:
#         random seed 42
#
#   - individual review embeddings are L2-normalized
#
#   - restaurant vectors are obtained through mean pooling
#
#   - final restaurant embeddings are L2-normalized
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


# ------------------------------------------------------------
# Prepared Yelp review dataset
# ------------------------------------------------------------

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "reviews"
    / "restaurant_reviews_prepared.csv"
)


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "restaurants"
)


# ------------------------------------------------------------
# Restaurant embeddings
# ------------------------------------------------------------

OUTPUT_NPZ = (
    OUTPUT_DIR
    / "restaurant_review_embeddings.npz"
)


# ------------------------------------------------------------
# Embedding index
# ------------------------------------------------------------

OUTPUT_INDEX = (
    OUTPUT_DIR
    / "restaurant_review_embedding_index.csv"
)


# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------

OUTPUT_METADATA = (
    OUTPUT_DIR
    / "restaurant_review_embeddings_metadata.json"
)


# ============================================================
# CONFIGURATION
# ============================================================


# ------------------------------------------------------------
# Sentence-BERT model
#
# Same model used for Movies and Music.
#
# Keeping the same model across domains is particularly
# important later when we construct cross-domain semantic
# representations.
# ------------------------------------------------------------

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


# ------------------------------------------------------------
# Maximum reviews per restaurant
#
# Yelp review coverage is very dense:
#
#   33,941 restaurants
#   4,522,570 reviews
#
# Many restaurants contain hundreds or thousands of reviews.
#
# Limiting each restaurant to 50 reviews:
#
#   - prevents highly reviewed restaurants from dominating
#     computational cost
#
#   - gives each restaurant a more comparable amount of
#     textual evidence
#
#   - keeps methodology consistent with Movies and Music
#
# Review selection is deterministic.
# ------------------------------------------------------------

MAX_REVIEWS_PER_RESTAURANT = 50


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

RANDOM_SEED = 42


# ------------------------------------------------------------
# GPU encoding batch size
#
# This configuration already worked successfully for the
# Movie and Music review embedding runs.
# ------------------------------------------------------------

BATCH_SIZE = 64


# ------------------------------------------------------------
# File-processing chunk sizes
#
# restaurant_reviews_prepared.csv contains more than
# 4.5 million reviews, so the generic embedding pipeline
# processes it incrementally rather than loading everything
# into memory.
# ------------------------------------------------------------

SELECTION_CHUNK_SIZE = 100_000

ENCODING_CHUNK_SIZE = 10_000


# ------------------------------------------------------------
# CUDA
#
# CUDA was already validated successfully with:
#
#   NVIDIA GeForce RTX 4050 Laptop GPU
#
# and the Movie / Music embedding pipelines.
# ------------------------------------------------------------

DEVICE = "cuda"


# ============================================================
# VALIDATE INPUT
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(

        "Prepared restaurant review dataset was not found:\n"
        f"{INPUT_FILE}\n\n"

        "Run:\n"
        "python -m src.domains.restaurants."
        "prepare_restaurant_reviews\n"

        "before generating restaurant review embeddings."

    )


# ============================================================
# MAIN
# ============================================================


def main():

    build_review_embeddings(

        # ----------------------------------------------------
        # Input / outputs
        # ----------------------------------------------------

        input_file=
            INPUT_FILE,

        output_npz=
            OUTPUT_NPZ,

        output_index_csv=
            OUTPUT_INDEX,

        output_metadata_json=
            OUTPUT_METADATA,


        # ----------------------------------------------------
        # Restaurant-specific columns
        # ----------------------------------------------------

        entity_id_column=
            "business_id",

        review_id_column=
            "review_id",

        text_column=
            "review_text",


        # ----------------------------------------------------
        # Shared Sentence-BERT methodology
        # ----------------------------------------------------

        model_name=
            MODEL_NAME,

        max_reviews_per_entity=
            MAX_REVIEWS_PER_RESTAURANT,

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