# src/domains/movies/prepare_movie_reviews.py

# ============================================================
# PREPARE IMDb MOVIE REVIEWS
#
# Purpose:
#   1. Load the IMDb reviews matched to MovieLens movies
#   2. Remove duplicate review IDs
#   3. Remove invalid / empty reviews
#   4. Standardize ratings, spoiler flags and helpfulness
#   5. Preserve one row per review for future BERT embeddings
#   6. Create a compact movie-level review summary
#
#
# Input:
#
#   data/processed/reviews/
#       movies_reviews_matched.csv
#
#
# Outputs:
#
#   data/processed/reviews/
#       movies_reviews_prepared.csv
#
#       movie_review_summary.csv
#
#
# Important:
#
#   This script DOES NOT:
#
#       - create BERT embeddings
#       - concatenate all reviews into one huge text
#       - modify the MovieLens movie preprocessing
#       - modify build_movie_map.py
#
#   Individual reviews are deliberately preserved so that
#   review embeddings can later be created review-by-review
#   and aggregated at movie level.
#
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

REVIEWS_DIR = (
    ROOT
    / "data"
    / "processed"
    / "reviews"
)

INPUT_FILE = (
    REVIEWS_DIR
    / "movies_reviews_matched.csv"
)

OUTPUT_REVIEWS = (
    REVIEWS_DIR
    / "movies_reviews_prepared.csv"
)

OUTPUT_SUMMARY = (
    REVIEWS_DIR
    / "movie_review_summary.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Reviews contain potentially long text.
#
# Processing the file in chunks avoids loading the entire
# 2.5M+ review dataset into memory at once.
CHUNK_SIZE = 50_000


# ============================================================
# COLUMNS
# ============================================================

# Columns required from the matched review dataset.
REQUIRED_COLUMNS = [

    "movieId",

    "review_id",

    "review_text",

    "rating",

    "review_date",

    "spoiler_tag",

    "helpful",

    "source",

    "match_method",

    "match_score",

    "match_status",

]


# Columns retained in the clean review-level output.
OUTPUT_COLUMNS = [

    "movieId",

    "review_id",

    "review_text",

    "rating",

    "review_date",

    "spoiler_tag",

    "helpful_yes",

    "helpful_total",

    "helpful_ratio",

    "review_char_count",

    "review_word_count",

    "source",

    "match_method",

    "match_score",

    "match_status",

]


# ============================================================
# VALIDATE INPUT FILE
# ============================================================


def validate_input_file():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "Matched movie review file not found:\n"
            f"{INPUT_FILE}"
        )

    # Read header only.
    columns = (
        pd.read_csv(
            INPUT_FILE,
            nrows=0
        )
        .columns
        .tolist()
    )

    missing_columns = [

        column

        for column in REQUIRED_COLUMNS

        if column not in columns

    ]

    if missing_columns:

        raise ValueError(
            "The matched review file is missing "
            "required columns:\n"
            f"{missing_columns}"
        )


# ============================================================
# HELPFULNESS PARSING
# ============================================================


def parse_helpfulness(helpful_series):
    """
    Convert IMDb helpfulness values into numeric columns.

    IMDb values look approximately like:

        ["26", "41"]

    which means:

        26 users found the review helpful
        out of 41 votes.

    Returns:

        helpful_yes
        helpful_total
        helpful_ratio
    """

    helpful_text = (
        helpful_series
        .astype("string")
        .fillna("")
    )

    # Extract the two integers from strings such as:
    #
    # ["26", "41"]
    #
    # The pattern is deliberately tolerant of quotes/spaces.

    extracted = helpful_text.str.extract(

        r'["\']?(\d+)["\']?\s*,\s*["\']?(\d+)'

    )

    helpful_yes = pd.to_numeric(
        extracted[0],
        errors="coerce"
    )

    helpful_total = pd.to_numeric(
        extracted[1],
        errors="coerce"
    )

    helpful_ratio = np.where(

        helpful_total > 0,

        helpful_yes / helpful_total,

        np.nan

    )

    return (
        helpful_yes,
        helpful_total,
        helpful_ratio
    )


# ============================================================
# CLEAN ONE CHUNK
# ============================================================


def clean_review_chunk(
    chunk,
    seen_review_ids
):
    """
    Clean one chunk of matched IMDb reviews.

    Returns:

        cleaned_chunk
        statistics
    """

    statistics = {

        "input_rows": len(chunk),

        "duplicates_removed": 0,

        "missing_review_ids": 0,

        "invalid_movie_ids": 0,

        "empty_review_text": 0,

    }

    chunk = chunk.copy()

    # ========================================================
    # MOVIE ID
    # ========================================================

    chunk["movieId"] = pd.to_numeric(
        chunk["movieId"],
        errors="coerce"
    )

    invalid_movie_mask = (
        chunk["movieId"].isna()
    )

    statistics[
        "invalid_movie_ids"
    ] = int(
        invalid_movie_mask.sum()
    )

    chunk = chunk[
        ~invalid_movie_mask
    ].copy()

    chunk["movieId"] = (
        chunk["movieId"]
        .astype(int)
    )

    # ========================================================
    # REVIEW ID
    # ========================================================

    chunk["review_id"] = (
        chunk["review_id"]
        .astype("string")
        .str.strip()
    )

    missing_review_id_mask = (

        chunk["review_id"].isna()

        |

        (
            chunk["review_id"]
            == ""
        )

    )

    statistics[
        "missing_review_ids"
    ] = int(
        missing_review_id_mask.sum()
    )

    chunk = chunk[
        ~missing_review_id_mask
    ].copy()

    # --------------------------------------------------------
    # Remove duplicates inside the current chunk first.
    # --------------------------------------------------------

    before = len(chunk)

    chunk = chunk.drop_duplicates(
        subset=[
            "review_id"
        ],
        keep="first"
    )

    statistics[
        "duplicates_removed"
    ] += (
        before - len(chunk)
    )

    # --------------------------------------------------------
    # Remove review IDs already seen in previous chunks.
    #
    # This handles duplicates occurring in different portions
    # of the source file.
    # --------------------------------------------------------

    duplicate_previous_mask = (
        chunk["review_id"]
        .isin(
            seen_review_ids
        )
    )

    statistics[
        "duplicates_removed"
    ] += int(
        duplicate_previous_mask.sum()
    )

    chunk = chunk[
        ~duplicate_previous_mask
    ].copy()

    # Add newly accepted IDs to global set.
    seen_review_ids.update(
        chunk["review_id"]
        .astype(str)
        .tolist()
    )

    # ========================================================
    # REVIEW TEXT
    # ========================================================

    chunk["review_text"] = (
        chunk["review_text"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    empty_text_mask = (
        chunk["review_text"]
        == ""
    )

    statistics[
        "empty_review_text"
    ] = int(
        empty_text_mask.sum()
    )

    # Empty review text cannot contribute to the future
    # semantic review embedding, so remove it.
    chunk = chunk[
        ~empty_text_mask
    ].copy()

    # ========================================================
    # RATING
    # ========================================================

    chunk["rating"] = pd.to_numeric(
        chunk["rating"],
        errors="coerce"
    )

    # IMDb user ratings should lie between 1 and 10.
    #
    # Invalid values become missing instead of removing
    # the review itself, because the text is still useful.
    invalid_rating_mask = (

        chunk["rating"].notna()

        &

        (
            (chunk["rating"] < 1)

            |

            (chunk["rating"] > 10)
        )

    )

    chunk.loc[
        invalid_rating_mask,
        "rating"
    ] = np.nan

    # ========================================================
    # SPOILER FLAG
    # ========================================================

    spoiler_numeric = pd.to_numeric(
        chunk["spoiler_tag"],
        errors="coerce"
    )

    chunk["spoiler_tag"] = (

        spoiler_numeric
        .fillna(0)
        .gt(0)
        .astype(int)

    )

    # IMPORTANT:
    #
    # Spoiler reviews are NOT removed here.
    #
    # We preserve the information so that later we can test
    # whether spoiler reviews should be included or excluded
    # from the semantic representation.

    # ========================================================
    # HELPFULNESS
    # ========================================================

    (
        helpful_yes,
        helpful_total,
        helpful_ratio

    ) = parse_helpfulness(
        chunk["helpful"]
    )

    chunk["helpful_yes"] = (
        helpful_yes
    )

    chunk["helpful_total"] = (
        helpful_total
    )

    chunk["helpful_ratio"] = (
        helpful_ratio
    )

    # ========================================================
    # TEXT LENGTH FEATURES
    # ========================================================

    chunk["review_char_count"] = (

        chunk["review_text"]
        .str.len()

    )

    chunk["review_word_count"] = (

        chunk["review_text"]
        .str.split()
        .str.len()

    )

    # ========================================================
    # MATCH SCORE
    # ========================================================

    chunk["match_score"] = pd.to_numeric(
        chunk["match_score"],
        errors="coerce"
    )

    # ========================================================
    # OUTPUT COLUMNS
    # ========================================================

    chunk = chunk[
        OUTPUT_COLUMNS
    ].copy()

    return (
        chunk,
        statistics
    )


# ============================================================
# PREPARE REVIEW-LEVEL DATASET
# ============================================================


def prepare_review_dataset():

    print()
    print("=" * 60)
    print("PREPARING IMDb MOVIE REVIEWS")
    print("=" * 60)
    print()

    validate_input_file()

    print(
        "Input:"
    )

    print(
        INPUT_FILE
    )

    print()

    # --------------------------------------------------------
    # Ensure a clean output on every complete run.
    # --------------------------------------------------------

    if OUTPUT_REVIEWS.exists():

        OUTPUT_REVIEWS.unlink()

    # --------------------------------------------------------
    # Track review IDs globally so duplicates across chunks
    # are removed.
    # --------------------------------------------------------

    seen_review_ids = set()

    total_input_rows = 0

    total_output_rows = 0

    total_duplicates = 0

    total_missing_ids = 0

    total_invalid_movie_ids = 0

    total_empty_text = 0

    first_output_chunk = True

    # --------------------------------------------------------
    # Process source CSV in chunks.
    # --------------------------------------------------------

    reader = pd.read_csv(
        INPUT_FILE,
        usecols=REQUIRED_COLUMNS,
        chunksize=CHUNK_SIZE
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1
    ):

        (
            cleaned,
            stats
        ) = clean_review_chunk(

            chunk,

            seen_review_ids

        )

        total_input_rows += (
            stats["input_rows"]
        )

        total_duplicates += (
            stats["duplicates_removed"]
        )

        total_missing_ids += (
            stats["missing_review_ids"]
        )

        total_invalid_movie_ids += (
            stats["invalid_movie_ids"]
        )

        total_empty_text += (
            stats["empty_review_text"]
        )

        total_output_rows += len(
            cleaned
        )

        # ----------------------------------------------------
        # Write incrementally.
        # ----------------------------------------------------

        cleaned.to_csv(

            OUTPUT_REVIEWS,

            mode=(
                "w"
                if first_output_chunk
                else "a"
            ),

            header=first_output_chunk,

            index=False

        )

        first_output_chunk = False

        print(

            f"\r"
            f"Chunks processed: "
            f"{chunk_number:,} | "

            f"Rows read: "
            f"{total_input_rows:,} | "

            f"Rows kept: "
            f"{total_output_rows:,} | "

            f"Duplicates removed: "
            f"{total_duplicates:,}",

            end=""

        )

    print()
    print()

    print(
        "Review-level preparation complete."
    )

    print()

    print(
        f"Input rows:              "
        f"{total_input_rows:,}"
    )

    print(
        f"Prepared reviews:        "
        f"{total_output_rows:,}"
    )

    print(
        f"Duplicate IDs removed:   "
        f"{total_duplicates:,}"
    )

    print(
        f"Missing review IDs:      "
        f"{total_missing_ids:,}"
    )

    print(
        f"Invalid movie IDs:       "
        f"{total_invalid_movie_ids:,}"
    )

    print(
        f"Empty review text:       "
        f"{total_empty_text:,}"
    )

    print()

    print(
        "Prepared review output:"
    )

    print(
        OUTPUT_REVIEWS
    )


# ============================================================
# CREATE MOVIE-LEVEL SUMMARY
# ============================================================


def create_movie_review_summary():
    """
    Create one summary row per MovieLens movie that has
    prepared IMDb review data.

    The summary is deliberately compact.

    It contains descriptive review metadata only.

    Individual review text remains in
    movies_reviews_prepared.csv for the later BERT step.
    """

    print()
    print("=" * 60)
    print("CREATING MOVIE REVIEW SUMMARY")
    print("=" * 60)
    print()

    if not OUTPUT_REVIEWS.exists():

        raise FileNotFoundError(
            "Prepared review dataset does not exist:\n"
            f"{OUTPUT_REVIEWS}"
        )

    # --------------------------------------------------------
    # We do a second chunked pass over the clean output.
    #
    # Importantly, review_text itself is NOT loaded here.
    # Only the compact columns needed for statistics are read.
    # --------------------------------------------------------

    summary_columns = [

        "movieId",

        "review_id",

        "rating",

        "spoiler_tag",

        "helpful_ratio",

        "review_char_count",

        "review_word_count",

    ]

    accumulator = None

    reader = pd.read_csv(

        OUTPUT_REVIEWS,

        usecols=summary_columns,

        chunksize=CHUNK_SIZE

    )

    total_rows = 0

    for chunk_number, chunk in enumerate(
        reader,
        start=1
    ):

        total_rows += len(
            chunk
        )

        # ----------------------------------------------------
        # Components needed for later mean / standard
        # deviation calculation.
        # ----------------------------------------------------

        chunk["rating_squared"] = (
            chunk["rating"] ** 2
        )

        chunk["helpful_ratio_valid"] = (
            chunk["helpful_ratio"]
            .notna()
            .astype(int)
        )

        chunk["helpful_ratio_filled"] = (
            chunk["helpful_ratio"]
            .fillna(0)
        )

        # ----------------------------------------------------
        # Aggregate current chunk.
        # ----------------------------------------------------

        grouped = (

            chunk
            .groupby(
                "movieId",
                as_index=False
            )
            .agg(

                review_count=(
                    "review_id",
                    "size"
                ),

                review_rating_count=(
                    "rating",
                    "count"
                ),

                review_rating_sum=(
                    "rating",
                    "sum"
                ),

                review_rating_squared_sum=(
                    "rating_squared",
                    "sum"
                ),

                spoiler_review_count=(
                    "spoiler_tag",
                    "sum"
                ),

                helpful_ratio_sum=(
                    "helpful_ratio_filled",
                    "sum"
                ),

                helpful_ratio_count=(
                    "helpful_ratio_valid",
                    "sum"
                ),

                review_char_count_sum=(
                    "review_char_count",
                    "sum"
                ),

                review_word_count_sum=(
                    "review_word_count",
                    "sum"
                )

            )

        )

        # ----------------------------------------------------
        # Combine with summaries from previous chunks.
        #
        # We only retain one row per movieId in memory,
        # so memory use remains small.
        # ----------------------------------------------------

        if accumulator is None:

            accumulator = grouped

        else:

            accumulator = (

                pd.concat(
                    [
                        accumulator,
                        grouped
                    ],
                    ignore_index=True
                )

                .groupby(
                    "movieId",
                    as_index=False
                )
                .sum()

            )

        print(

            f"\r"
            f"Summary chunks processed: "
            f"{chunk_number:,} | "

            f"Reviews summarized: "
            f"{total_rows:,}",

            end=""

        )

    print()
    print()

    if accumulator is None:

        raise ValueError(
            "No prepared reviews were available "
            "to create the movie summary."
        )

    # ========================================================
    # DERIVED STATISTICS
    # ========================================================

    summary = accumulator.copy()

    # --------------------------------------------------------
    # Mean IMDb review rating
    # --------------------------------------------------------

    summary["review_rating_mean"] = np.where(

        summary["review_rating_count"] > 0,

        (
            summary["review_rating_sum"]
            /
            summary["review_rating_count"]
        ),

        np.nan

    )

    # --------------------------------------------------------
    # Sample standard deviation
    #
    # Equivalent to pandas std(ddof=1).
    # --------------------------------------------------------

    rating_count = (
        summary["review_rating_count"]
    )

    numerator = (

        summary[
            "review_rating_squared_sum"
        ]

        -

        (
            summary["review_rating_sum"] ** 2

            /

            rating_count.replace(
                0,
                np.nan
            )
        )

    )

    variance = np.where(

        rating_count > 1,

        numerator
        /
        (
            rating_count - 1
        ),

        np.nan

    )

    # Numerical rounding can occasionally produce a tiny
    # negative value such as -1e-15.
    variance = np.where(

        np.isnan(variance),

        np.nan,

        np.maximum(
            variance,
            0
        )

    )

    summary["review_rating_std"] = (
        np.sqrt(
            variance
        )
    )

    # --------------------------------------------------------
    # Spoiler ratio
    # --------------------------------------------------------

    summary["spoiler_ratio"] = (

        summary["spoiler_review_count"]

        /

        summary["review_count"]

    )

    # --------------------------------------------------------
    # Mean helpfulness ratio
    # --------------------------------------------------------

    summary["avg_helpful_ratio"] = np.where(

        summary["helpful_ratio_count"] > 0,

        (
            summary["helpful_ratio_sum"]
            /
            summary["helpful_ratio_count"]
        ),

        np.nan

    )

    # --------------------------------------------------------
    # Average review lengths
    # --------------------------------------------------------

    summary["avg_review_char_count"] = (

        summary["review_char_count_sum"]

        /

        summary["review_count"]

    )

    summary["avg_review_word_count"] = (

        summary["review_word_count_sum"]

        /

        summary["review_count"]

    )

    # ========================================================
    # FINAL COLUMN SELECTION
    # ========================================================

    summary = summary[

        [

            "movieId",

            "review_count",

            "review_rating_count",

            "review_rating_mean",

            "review_rating_std",

            "spoiler_review_count",

            "spoiler_ratio",

            "avg_helpful_ratio",

            "avg_review_char_count",

            "avg_review_word_count",

        ]

    ].copy()

    # --------------------------------------------------------
    # Integer columns
    # --------------------------------------------------------

    summary["movieId"] = (
        summary["movieId"]
        .astype(int)
    )

    summary["review_count"] = (
        summary["review_count"]
        .astype(int)
    )

    summary["review_rating_count"] = (
        summary["review_rating_count"]
        .astype(int)
    )

    summary["spoiler_review_count"] = (
        summary["spoiler_review_count"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Sort by MovieLens ID
    # --------------------------------------------------------

    summary = (

        summary
        .sort_values(
            "movieId"
        )
        .reset_index(
            drop=True
        )

    )

    # ========================================================
    # SAVE
    # ========================================================

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        f"Movies with prepared reviews: "
        f"{len(summary):,}"
    )

    print(
        f"Total reviews represented:    "
        f"{summary['review_count'].sum():,}"
    )

    print()

    print(
        "Movie-level summary output:"
    )

    print(
        OUTPUT_SUMMARY
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 60)
    print("MOVIE REVIEW PREPARATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Step 1:
    # Clean review-level dataset
    # --------------------------------------------------------

    prepare_review_dataset()

    # --------------------------------------------------------
    # Step 2:
    # Create movie-level metadata summary
    # --------------------------------------------------------

    create_movie_review_summary()

    print()
    print("=" * 60)
    print("MOVIE REVIEW PREPARATION COMPLETE")
    print("=" * 60)

    print()

    print(
        "Review-level dataset:"
    )

    print(
        OUTPUT_REVIEWS
    )

    print()

    print(
        "Movie-level summary:"
    )

    print(
        OUTPUT_SUMMARY
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()