# src/domains/restaurants/embed_restaurant_semantics.py

# ============================================================
# RESTAURANT BASE SEMANTIC EMBEDDINGS
# ============================================================
#
# Purpose
# -------
#
# Build one Sentence-BERT base semantic embedding per
# restaurant using the restaurant's existing non-review
# semantic representation.
#
#
# This is part of Method B:
#
#     Shared Experiential / Feel Space
#
#
# Representation
# --------------
#
# Yelp business data
#      ↓
# filter_restaurants()
#      ↓
# categories + tips
#      ↓
# create_tags_text()
#      ↓
# tags_text
#      ↓
# sentence-transformers/all-MiniLM-L6-v2
#      ↓
# 384D base semantic restaurant embedding
#
#
# These embeddings inhabit the SAME semantic space as:
#
#     - Yelp review embeddings
#     - Movie base embeddings
#     - Music base embeddings
#     - Feel-anchor embeddings
#
#
# Later:
#
#     base semantic embedding
#              +
#     review semantic embedding
#              ↓
#     shared restaurant semantic vector
#              ↓
#        feel-anchor projection
#
#
# IMPORTANT
# ---------
#
# Full Yelp reviews are NOT used in this script.
#
# Tips remain part of the base semantic representation,
# exactly as they were in the existing Restaurant atlas.
#
# ============================================================


from pathlib import Path


# ============================================================
# RESTAURANT DATA
# ============================================================

from src.domains.restaurants.load_data import (
    load_raw_data
)

from src.domains.restaurants.feature_engineering import (
    filter_restaurants,
    create_tags_text,
    compute_popularity_score
)


# ============================================================
# GENERIC FEEL-SPACE EMBEDDING ENGINE
# ============================================================

from src.atlas.cross_domain.feel_space.base_embeddings import (
    build_base_semantic_embeddings
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "feel_space"
    / "restaurants"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Base semantic embedding matrix
# ------------------------------------------------------------

OUTPUT_NPZ = (
    OUTPUT_DIR
    / "restaurant_base_semantic_embeddings.npz"
)


# ------------------------------------------------------------
# Human-readable embedding index
# ------------------------------------------------------------

OUTPUT_INDEX = (
    OUTPUT_DIR
    / "restaurant_base_semantic_embedding_index.csv"
)


# ------------------------------------------------------------
# Methodological metadata
# ------------------------------------------------------------

OUTPUT_METADATA = (
    OUTPUT_DIR
    / "restaurant_base_semantic_embeddings_metadata.json"
)


# ============================================================
# CONFIGURATION
# ============================================================


# ------------------------------------------------------------
# Same Restaurant population threshold used by the current
# Restaurant mono-domain and cross-domain pipelines.
# ------------------------------------------------------------

MIN_REVIEWS = 20


# ------------------------------------------------------------
# Sentence-BERT encoding settings
# ------------------------------------------------------------

BATCH_SIZE = 64


ENCODING_CHUNK_SIZE = 10_000


DEVICE = "cuda"


# ============================================================
# LOAD AND PREPARE RESTAURANTS
# ============================================================


def prepare_restaurants():
    """
    Reconstruct the same Restaurant semantic population used
    by the current Restaurant atlas pipeline.

    Returns
    -------
    pandas.DataFrame

        Contains at minimum:

            business_id
            tags_text
    """

    print()

    print(
        "=" * 65
    )

    print(
        "PREPARING RESTAURANT BASE SEMANTIC DATA"
    )

    print(
        "=" * 65
    )

    print()


    # ========================================================
    # LOAD RAW YELP DATA
    # ========================================================

    businesses, reviews, tips = (
        load_raw_data()
    )


    print(
        f"Raw businesses: "
        f"{len(businesses):,}"
    )


    print(
        f"Raw reviews:    "
        f"{len(reviews):,}"
    )


    print(
        f"Raw tips:       "
        f"{len(tips):,}"
    )


    # ========================================================
    # FILTER RESTAURANTS
    # ========================================================

    restaurants = filter_restaurants(

        businesses,

        min_reviews=
            MIN_REVIEWS

    )


    print(
        f"Restaurants after filtering: "
        f"{len(restaurants):,}"
    )


    # ========================================================
    # CREATE EXISTING RESTAURANT SEMANTIC TEXT
    # ========================================================

    restaurants = create_tags_text(

        restaurants,

        tips

    )


    restaurants["tags_text"] = (

        restaurants["tags_text"]
        .fillna("")

    )


    # --------------------------------------------------------
    # Popularity itself is not part of the semantic embedding,
    # but keeping this preprocessing step aligned with the
    # current Restaurant pipeline helps preserve consistency
    # with the established domain workflow.
    # --------------------------------------------------------

    restaurants = compute_popularity_score(
        restaurants
    )


    # ========================================================
    # STANDARDIZE BUSINESS IDS
    # ========================================================

    restaurants["business_id"] = (

        restaurants["business_id"]
        .astype(str)
        .str.strip()

    )


    # ========================================================
    # VALIDATION
    # ========================================================

    invalid_ids = (

        restaurants[
            "business_id"
        ]
        ==
        ""

    )


    if invalid_ids.any():

        raise ValueError(

            f"{int(invalid_ids.sum()):,} restaurants contain "
            "empty business_id values."

        )


    if restaurants["business_id"].duplicated().any():

        duplicate_count = int(

            restaurants[
                "business_id"
            ]
            .duplicated()
            .sum()

        )


        raise ValueError(

            f"Prepared Restaurant dataframe contains "
            f"{duplicate_count:,} duplicate business IDs."

        )


    restaurants_with_text = int(

        (
            restaurants[
                "tags_text"
            ]
            .str.strip()
            !=
            ""
        ).sum()

    )


    restaurants_without_text = (

        len(
            restaurants
        )

        -

        restaurants_with_text

    )


    print()

    print(
        f"Prepared restaurants:       "
        f"{len(restaurants):,}"
    )


    print(
        f"Restaurants with base text: "
        f"{restaurants_with_text:,}"
    )


    print(
        f"Restaurants without text:   "
        f"{restaurants_without_text:,}"
    )


    if len(
        restaurants
    ) > 0:

        coverage = (

            restaurants_with_text

            /

            len(
                restaurants
            )

            *

            100

        )


        print(
            f"Base text coverage:          "
            f"{coverage:.2f}%"
        )


    print()


    return restaurants


# ============================================================
# MAIN
# ============================================================


def main():

    print()

    print(
        "=" * 65
    )

    print(
        "RESTAURANT BASE SEMANTIC EMBEDDINGS"
    )

    print(
        "=" * 65
    )

    print()


    # ========================================================
    # 1. RECONSTRUCT RESTAURANT SEMANTIC DATA
    # ========================================================

    restaurants = (
        prepare_restaurants()
    )


    # ========================================================
    # 2. BUILD BASE SENTENCE-BERT EMBEDDINGS
    # ========================================================

    (
        embeddings,
        entity_data,
        metadata

    ) = build_base_semantic_embeddings(

        # ----------------------------------------------------
        # Entity dataframe
        # ----------------------------------------------------

        df=
            restaurants,


        # ----------------------------------------------------
        # Canonical Yelp entity identifier
        # ----------------------------------------------------

        entity_id_column=
            "business_id",


        # ----------------------------------------------------
        # Existing non-review semantic representation
        #
        # Includes:
        #
        #     categories
        #     tips
        #
        # through the Restaurant create_tags_text() pipeline.
        # ----------------------------------------------------

        text_column=
            "tags_text",


        # ----------------------------------------------------
        # Outputs
        # ----------------------------------------------------

        output_npz=
            OUTPUT_NPZ,

        output_index_csv=
            OUTPUT_INDEX,

        output_metadata_json=
            OUTPUT_METADATA,


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        source_name=
            "restaurants",


        # ----------------------------------------------------
        # Encoding
        # ----------------------------------------------------

        batch_size=
            BATCH_SIZE,

        encoding_chunk_size=
            ENCODING_CHUNK_SIZE,

        device=
            DEVICE

    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=" * 65
    )

    print(
        "RESTAURANT BASE SEMANTIC EMBEDDING COMPLETE"
    )

    print(
        "=" * 65
    )

    print()


    print(
        f"Restaurants represented:       "
        f"{len(entity_data):,}"
    )


    print(
        f"Restaurants with base text:    "
        f"{int(entity_data['has_semantic_text'].sum()):,}"
    )


    print(
        f"Restaurants without base text: "
        f"{int((~entity_data['has_semantic_text']).sum()):,}"
    )


    print(
        f"Embedding shape:               "
        f"{embeddings.shape}"
    )


    print()


    print(
        "Embedding output:"
    )

    print(
        OUTPUT_NPZ
    )


    print()


    print(
        "Index output:"
    )

    print(
        OUTPUT_INDEX
    )


    print()


    print(
        "Metadata output:"
    )

    print(
        OUTPUT_METADATA
    )


    print()


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()