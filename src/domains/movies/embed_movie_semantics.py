# src/domains/movies/embed_movie_semantics.py

# ============================================================
# MOVIE BASE SEMANTIC EMBEDDINGS
# ============================================================
#
# Purpose
# -------
#
# Build one Sentence-BERT base semantic embedding per movie
# using the movie's existing MovieLens tag representation.
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
# MovieLens tags
#      ↓
# concatenate_tags()
#      ↓
# tags_text
#      ↓
# sentence-transformers/all-MiniLM-L6-v2
#      ↓
# 384D base semantic movie embedding
#
#
# These base embeddings inhabit the SAME embedding space as
# the previously generated IMDb review embeddings.
#
# Later:
#
#     base semantic embedding
#              +
#     review semantic embedding
#              ↓
#     shared movie semantic vector
#              ↓
#        feel-anchor projection
#
#
# IMPORTANT
# ---------
#
# Reviews are NOT used in this script.
#
# This script only builds the non-review semantic component.
#
# ============================================================


from pathlib import Path


# ============================================================
# MOVIE DATA
# ============================================================

from src.domains.movies.load_data import (
    load_raw_data
)

from src.domains.movies.merge_data import (
    compute_movie_stats,
    merge_movies
)

from src.domains.movies.feature_engineering import (
    compute_weighted_rating,
    concatenate_tags
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
    / "movies"
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
    / "movie_base_semantic_embeddings.npz"
)


# ------------------------------------------------------------
# Human-readable embedding index
# ------------------------------------------------------------

OUTPUT_INDEX = (
    OUTPUT_DIR
    / "movie_base_semantic_embedding_index.csv"
)


# ------------------------------------------------------------
# Methodological metadata
# ------------------------------------------------------------

OUTPUT_METADATA = (
    OUTPUT_DIR
    / "movie_base_semantic_embeddings_metadata.json"
)


# ============================================================
# CONFIGURATION
# ============================================================


# ------------------------------------------------------------
# Same MiniLM model used for:
#
#     Movie reviews
#     Music reviews
#     Restaurant reviews
#     Feel anchors
#
# The generic helper obtains the model name from
# feel_anchors.py by default.
# ------------------------------------------------------------

BATCH_SIZE = 64


# ------------------------------------------------------------
# Number of entity texts processed in each encoding chunk.
#
# Movie base semantic encoding is considerably lighter than
# review embedding because there is only ONE text per movie.
# ------------------------------------------------------------

ENCODING_CHUNK_SIZE = 10_000


# ------------------------------------------------------------
# GPU
# ------------------------------------------------------------

DEVICE = "cuda"


# ============================================================
# LOAD AND PREPARE MOVIES
# ============================================================


def prepare_movies():
    """
    Reconstruct the same Movie semantic population used by
    the existing Movie atlas pipeline.

    Returns
    -------
    pandas.DataFrame

        Contains at minimum:

            movieId
            tags_text
    """

    print()

    print(
        "=" * 65
    )

    print(
        "PREPARING MOVIE BASE SEMANTIC DATA"
    )

    print(
        "=" * 65
    )

    print()


    # ========================================================
    # LOAD RAW MOVIELENS DATA
    # ========================================================

    movies, ratings, tags = (
        load_raw_data()
    )


    print(
        f"Raw movies:   "
        f"{len(movies):,}"
    )


    print(
        f"Ratings:      "
        f"{len(ratings):,}"
    )


    print(
        f"Tag rows:     "
        f"{len(tags):,}"
    )


    # ========================================================
    # MOVIE STATISTICS
    # ========================================================

    movie_stats = compute_movie_stats(
        ratings
    )


    movies = merge_movies(

        movie_stats,

        movies

    )


    # --------------------------------------------------------
    # Keep preprocessing consistent with the normal Movie
    # pipeline.
    #
    # Weighted rating itself is not part of semantic text,
    # but applying the same preprocessing keeps the domain
    # dataframe aligned with the atlas workflow.
    # --------------------------------------------------------

    movies = compute_weighted_rating(
        movies
    )


    # ========================================================
    # MOVIE TAGS
    # ========================================================

    movie_tags = concatenate_tags(
        tags
    )


    movies = movies.merge(

        movie_tags,

        on="movieId",

        how="left"

    )


    movies["tags_text"] = (

        movies["tags_text"]
        .fillna("")

    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if movies["movieId"].duplicated().any():

        duplicate_count = int(

            movies[
                "movieId"
            ]
            .duplicated()
            .sum()

        )


        raise ValueError(

            f"Prepared Movie dataframe contains "
            f"{duplicate_count:,} duplicate movieId values."

        )


    movies_with_tags = int(

        (
            movies[
                "tags_text"
            ]
            .str.strip()
            !=
            ""
        ).sum()

    )


    movies_without_tags = (

        len(
            movies
        )

        -

        movies_with_tags

    )


    print()

    print(
        f"Prepared movies:       "
        f"{len(movies):,}"
    )


    print(
        f"Movies with tags:      "
        f"{movies_with_tags:,}"
    )


    print(
        f"Movies without tags:   "
        f"{movies_without_tags:,}"
    )


    if len(
        movies
    ) > 0:

        coverage = (

            movies_with_tags

            /

            len(
                movies
            )

            *

            100

        )


        print(
            f"Tag coverage:          "
            f"{coverage:.2f}%"
        )


    print()


    return movies


# ============================================================
# MAIN
# ============================================================


def main():

    print()

    print(
        "=" * 65
    )

    print(
        "MOVIE BASE SEMANTIC EMBEDDINGS"
    )

    print(
        "=" * 65
    )

    print()


    # ========================================================
    # 1. RECONSTRUCT MOVIE SEMANTIC DATA
    # ========================================================

    movies = prepare_movies()


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
            movies,


        # ----------------------------------------------------
        # Canonical MovieLens entity ID
        # ----------------------------------------------------

        entity_id_column=
            "movieId",


        # ----------------------------------------------------
        # Existing non-review semantic representation
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
            "movies",


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
        "MOVIE BASE SEMANTIC EMBEDDING COMPLETE"
    )

    print(
        "=" * 65
    )

    print()


    print(
        f"Movies represented:       "
        f"{len(entity_data):,}"
    )


    print(
        f"Movies with base text:    "
        f"{int(entity_data['has_semantic_text'].sum()):,}"
    )


    print(
        f"Movies without base text: "
        f"{int((~entity_data['has_semantic_text']).sum()):,}"
    )


    print(
        f"Embedding shape:          "
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