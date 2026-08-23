#src/domains/movies/prepare_movies.py

# ============================================================
# Prepare MovieLens Movie Dataset
#
# Purpose:
#   Create a clean canonical movie-level dataset from the
#   raw MovieLens files.
#
# This script prepares the base movie table.
#
# It does NOT:
#   - create embeddings
#   - perform PCA
#   - perform UMAP
#   - perform clustering
#   - process IMDb review text
#
# Those remain separate pipeline stages.
#
# Output:
#
#   data/processed/movies/movies_prepared.csv
#
# ============================================================


from pathlib import Path
import re

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

RAW_MOVIES_DIR = (
    ROOT
    / "data"
    / "raw"
    / "movies"
)

PROCESSED_MOVIES_DIR = (
    ROOT
    / "data"
    / "processed"
    / "movies"
)

PROCESSED_MOVIES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    PROCESSED_MOVIES_DIR
    / "movies_prepared.csv"
)


# ============================================================
# HELPERS
# ============================================================


def extract_year(title):

    """
    Extract the four-digit release year from a
    MovieLens title.

    Example:
        Toy Story (1995)
        -> 1995
    """

    if pd.isna(title):

        return pd.NA

    match = re.search(
        r"\((\d{4})\)",
        str(title)
    )

    if match:

        return int(
            match.group(1)
        )

    return pd.NA


def clean_title(title):

    """
    Keep the MovieLens title but remove the trailing
    year from the canonical title field.

    Example:

        Toy Story (1995)
        -> Toy Story
    """

    if pd.isna(title):

        return ""

    title = str(title).strip()

    title = re.sub(
        r"\s*\(\d{4}\)\s*$",
        "",
        title
    )

    return title.strip()


# ============================================================
# LOAD DATA
# ============================================================


def load_movies():

    movies_path = (
        RAW_MOVIES_DIR
        / "movies.csv"
    )

    if not movies_path.exists():

        raise FileNotFoundError(
            f"MovieLens movies.csv not found:\n"
            f"{movies_path}"
        )

    movies = pd.read_csv(
        movies_path
    )

    return movies


def load_ratings():

    ratings_path = (
        RAW_MOVIES_DIR
        / "ratings.csv"
    )

    if not ratings_path.exists():

        raise FileNotFoundError(
            f"MovieLens ratings.csv not found:\n"
            f"{ratings_path}"
        )

    ratings = pd.read_csv(
        ratings_path
    )

    return ratings


def load_tags():

    tags_path = (
        RAW_MOVIES_DIR
        / "tags.csv"
    )

    if not tags_path.exists():

        raise FileNotFoundError(
            f"MovieLens tags.csv not found:\n"
            f"{tags_path}"
        )

    tags = pd.read_csv(
        tags_path
    )

    return tags


# ============================================================
# RATING STATISTICS
# ============================================================


def compute_rating_statistics(
    ratings
):

    stats = (
        ratings
        .groupby("movieId")
        .agg(
            avg_rating=(
                "rating",
                "mean"
            ),
            rating_count=(
                "rating",
                "count"
            ),
            rating_std=(
                "rating",
                "std"
            )
        )
        .reset_index()
    )

    return stats


# ============================================================
# TAG FEATURES
# ============================================================


def compute_tag_features(
    tags
):

    tags_clean = (
        tags
        .dropna(
            subset=["tag"]
        )
        .copy()
    )

    tags_clean["tag"] = (
        tags_clean["tag"]
        .astype(str)
        .str.strip()
    )

    tags_clean = tags_clean[
        tags_clean["tag"] != ""
    ]

    tag_features = (
        tags_clean
        .groupby("movieId")
        .agg(
            tag_count=(
                "tag",
                "count"
            ),
            tags_text=(
                "tag",
                lambda x:
                    " ".join(x)
            )
        )
        .reset_index()
    )

    return tag_features


# ============================================================
# PREPARE MOVIE TABLE
# ============================================================


def prepare_movies():

    print()
    print("=" * 60)
    print("PREPARING MOVIE DATA")
    print("=" * 60)

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    print()
    print("Loading MovieLens data...")

    movies = load_movies()

    ratings = load_ratings()

    tags = load_tags()

    print(
        f"Movies:  {len(movies):,}"
    )

    print(
        f"Ratings: {len(ratings):,}"
    )

    print(
        f"Tags:    {len(tags):,}"
    )

    # --------------------------------------------------------
    # Validate movie IDs
    # --------------------------------------------------------

    if movies["movieId"].duplicated().any():

        raise ValueError(
            "Duplicate movieId values found "
            "in movies.csv."
        )

    # --------------------------------------------------------
    # Create title/year fields
    # --------------------------------------------------------

    movies["year"] = (
        movies["title"]
        .apply(extract_year)
    )

    movies["title_clean"] = (
        movies["title"]
        .apply(clean_title)
    )

    # --------------------------------------------------------
    # Rating statistics
    # --------------------------------------------------------

    print(
        "\nComputing rating statistics..."
    )

    rating_stats = (
        compute_rating_statistics(
            ratings
        )
    )

    # --------------------------------------------------------
    # Tag features
    # --------------------------------------------------------

    print(
        "Computing tag features..."
    )

    tag_features = (
        compute_tag_features(
            tags
        )
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print(
        "Merging movie features..."
    )

    prepared = (
        movies
        .merge(
            rating_stats,
            on="movieId",
            how="left"
        )
        .merge(
            tag_features,
            on="movieId",
            how="left"
        )
    )

    # --------------------------------------------------------
    # Fill missing values
    #
    # A movie without ratings has no rating statistics.
    # A movie without tags has no tag information.
    #
    # We keep those movies rather than dropping them.
    # --------------------------------------------------------

    prepared["rating_count"] = (
        prepared["rating_count"]
        .fillna(0)
        .astype(int)
    )

    prepared["tag_count"] = (
        prepared["tag_count"]
        .fillna(0)
        .astype(int)
    )

    prepared["tags_text"] = (
        prepared["tags_text"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    prepared = (
        prepared
        .sort_values(
            "movieId"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Column ordering
    # --------------------------------------------------------

    preferred_columns = [

        "movieId",

        "title",

        "title_clean",

        "year",

        "genres",

        "avg_rating",

        "rating_count",

        "rating_std",

        "tag_count",

        "tags_text"

    ]

    remaining_columns = [
        column
        for column in prepared.columns
        if column not in preferred_columns
    ]

    prepared = prepared[
        preferred_columns
        + remaining_columns
    ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    prepared.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Movies prepared: "
        f"{len(prepared):,}"
    )

    print(
        f"Movies with ratings: "
        f"{(prepared['rating_count'] > 0).sum():,}"
    )

    print(
        f"Movies with tags: "
        f"{(prepared['tag_count'] > 0).sum():,}"
    )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_FILE
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    prepare_movies()