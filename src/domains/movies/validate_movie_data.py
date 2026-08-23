#src/domains/movies/validate_movie_data.py

# ============================================================
# Movie Data Validation
#
# Purpose:
#   1. Validate the raw MovieLens movie dataset
#   2. Inspect rating coverage
#   3. Inspect tag coverage
#   4. Inspect IMDb review coverage
#   5. Identify potential data-quality issues
#
# This script DOES NOT modify any data.
#
# ============================================================


from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

MOVIES_DIR = (
    ROOT
    / "data"
    / "raw"
    / "movies"
)

REVIEWS_DIR = (
    ROOT
    / "data"
    / "processed"
    / "reviews"
)

MATCHED_REVIEWS = (
    REVIEWS_DIR
    / "movies_reviews_matched.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_REVIEW_THRESHOLDS = [
    1,
    5,
    10,
    25,
    50,
    100,
    500,
    1000
]


# ============================================================
# HELPERS
# ============================================================


def print_section(title):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_coverage(
    series,
    total,
    label
):

    count = series.sum()

    percentage = (
        count / total * 100
        if total > 0
        else 0
    )

    print(
        f"{label:<35}"
        f"{count:>10,}"
        f"  ({percentage:>6.2f}%)"
    )


# ============================================================
# LOAD RAW MOVIELENS DATA
# ============================================================


def load_movie_data():

    print_section(
        "LOADING MOVIELENS DATA"
    )

    movies_path = (
        MOVIES_DIR
        / "movies.csv"
    )

    ratings_path = (
        MOVIES_DIR
        / "ratings.csv"
    )

    tags_path = (
        MOVIES_DIR
        / "tags.csv"
    )

    links_path = (
        MOVIES_DIR
        / "links.csv"
    )

    for path in [
        movies_path,
        ratings_path,
        tags_path,
        links_path
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    movies = pd.read_csv(
        movies_path
    )

    ratings = pd.read_csv(
        ratings_path
    )

    tags = pd.read_csv(
        tags_path
    )

    links = pd.read_csv(
        links_path
    )

    print(
        f"Movies:  {len(movies):,}"
    )

    print(
        f"Ratings: {len(ratings):,}"
    )

    print(
        f"Tags:    {len(tags):,}"
    )

    print(
        f"Links:   {len(links):,}"
    )

    return (
        movies,
        ratings,
        tags,
        links
    )


# ============================================================
# BASIC MOVIE VALIDATION
# ============================================================


def validate_movies(
    movies
):

    print_section(
        "MOVIE DATA VALIDATION"
    )

    print(
        f"Total movie rows: "
        f"{len(movies):,}"
    )

    print(
        f"Unique movieIds: "
        f"{movies['movieId'].nunique():,}"
    )

    duplicate_ids = (
        movies["movieId"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate movieIds: "
        f"{duplicate_ids:,}"
    )

    missing_ids = (
        movies["movieId"]
        .isna()
        .sum()
    )

    missing_titles = (
        movies["title"]
        .isna()
        .sum()
    )

    missing_genres = (
        movies["genres"]
        .isna()
        .sum()
    )

    print(
        f"Missing movieIds: "
        f"{missing_ids:,}"
    )

    print(
        f"Missing titles: "
        f"{missing_titles:,}"
    )

    print(
        f"Missing genres: "
        f"{missing_genres:,}"
    )


# ============================================================
# RATING COVERAGE
# ============================================================


def validate_ratings(
    movies,
    ratings
):

    print_section(
        "RATING COVERAGE"
    )

    movie_ids_with_ratings = (
        ratings["movieId"]
        .nunique()
    )

    print(
        f"Movies with at least one rating: "
        f"{movie_ids_with_ratings:,}"
    )

    movie_rating_counts = (
        ratings
        .groupby("movieId")
        .size()
    )

    print()

    print(
        "Rating-count distribution:"
    )

    for threshold in MIN_REVIEW_THRESHOLDS:

        count = (
            movie_rating_counts
            .ge(threshold)
            .sum()
        )

        percentage = (
            count
            / len(movies)
            * 100
        )

        print(
            f"  >= {threshold:<5} ratings: "
            f"{count:>8,}"
            f" ({percentage:>6.2f}% of movies)"
        )


# ============================================================
# TAG COVERAGE
# ============================================================


def validate_tags(
    movies,
    tags
):

    print_section(
        "TAG COVERAGE"
    )

    movie_ids_with_tags = (
        tags["movieId"]
        .nunique()
    )

    print(
        f"Movies with at least one tag: "
        f"{movie_ids_with_tags:,}"
    )

    percentage = (
        movie_ids_with_tags
        / len(movies)
        * 100
    )

    print(
        f"Tag coverage: "
        f"{percentage:.2f}%"
    )


# ============================================================
# IMDb REVIEW COVERAGE
# ============================================================


def validate_reviews(
    movies
):

    print_section(
        "IMDb REVIEW COVERAGE"
    )

    if not MATCHED_REVIEWS.exists():

        print(
            "Matched review file not found:"
        )

        print(
            MATCHED_REVIEWS
        )

        return

    print(
        "Loading matched IMDb reviews..."
    )

    reviews = pd.read_csv(
        MATCHED_REVIEWS,
        usecols=[
            "movieId",
            "review_id"
        ]
    )

    print(
        f"Matched reviews: "
        f"{len(reviews):,}"
    )

    unique_review_ids = (
        reviews["review_id"]
        .nunique()
    )

    print(
        f"Unique review IDs: "
        f"{unique_review_ids:,}"
    )

    movies_with_reviews = (
        reviews["movieId"]
        .nunique()
    )

    print(
        f"Movies with IMDb reviews: "
        f"{movies_with_reviews:,}"
    )

    coverage = (
        movies_with_reviews
        / len(movies)
        * 100
    )

    print(
        f"Movie review coverage: "
        f"{coverage:.2f}%"
    )

    review_counts = (
        reviews
        .groupby("movieId")
        .size()
    )

    print()

    print(
        "IMDb review-count distribution:"
    )

    for threshold in MIN_REVIEW_THRESHOLDS:

        count = (
            review_counts
            .ge(threshold)
            .sum()
        )

        percentage = (
            count
            / len(movies)
            * 100
        )

        print(
            f"  >= {threshold:<5} reviews: "
            f"{count:>8,}"
            f" ({percentage:>6.2f}% of movies)"
        )

    # --------------------------------------------------------
    # Reviews matched to MovieLens IDs that do not exist
    # --------------------------------------------------------

    movie_id_set = set(
        movies["movieId"]
    )

    invalid_movie_ids = (
        ~reviews["movieId"]
        .isin(movie_id_set)
    )

    invalid_count = (
        invalid_movie_ids.sum()
    )

    print()

    print(
        f"Reviews pointing to unknown "
        f"MovieLens IDs: "
        f"{invalid_count:,}"
    )


# ============================================================
# OVERLAP ANALYSIS
# ============================================================


def validate_overlap(
    movies,
    ratings,
    tags
):

    print_section(
        "DATA SOURCE OVERLAP"
    )

    movie_ids = set(
        movies["movieId"]
    )

    rating_ids = set(
        ratings["movieId"]
    )

    tag_ids = set(
        tags["movieId"]
    )

    print(
        f"Movies with ratings: "
        f"{len(movie_ids & rating_ids):,}"
    )

    print(
        f"Movies with tags: "
        f"{len(movie_ids & tag_ids):,}"
    )

    print(
        f"Movies with both ratings "
        f"and tags: "
        f"{len(movie_ids & rating_ids & tag_ids):,}"
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 60)
    print("MOVIE DATA VALIDATION")
    print("=" * 60)

    (
        movies,
        ratings,
        tags,
        links
    ) = load_movie_data()

    validate_movies(
        movies
    )

    validate_ratings(
        movies,
        ratings
    )

    validate_tags(
        movies,
        tags
    )

    validate_reviews(
        movies
    )

    validate_overlap(
        movies,
        ratings,
        tags
    )

    print_section(
        "VALIDATION COMPLETE"
    )

    print(
        "No files were modified."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()