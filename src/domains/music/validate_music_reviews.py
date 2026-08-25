# src/domains/music/validate_music_reviews.py

# ============================================================
# VALIDATE CRITIQUEBRAINZ MUSIC REVIEWS
#
# Purpose:
#
#   Inspect the CritiqueBrainz reviews that were already
#   resolved and matched to artists using MusicBrainz MBIDs.
#
#   This script DOES NOT modify the data.
#
# We want to understand:
#
#   - total matched review rows
#   - unique review IDs
#   - duplicate review IDs
#   - conflicting duplicate review IDs
#   - unique matched artists
#   - missing artist MBIDs
#   - missing / empty review text
#   - language distribution
#   - rating availability and distribution
#   - direct artist vs release/release-group reviews
#   - reviews-per-artist distribution
#   - whether artist MBIDs still exist in the current
#     music dataset
#
# The results will determine the configuration used in:
#
#   prepare_music_reviews.py
#   embed_music_reviews.py
#   evaluate_fusion_weights.py
#
# ============================================================


from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------
# The original scraper writes relative paths.
#
# Depending on where it was executed from, the generated
# files may therefore currently live in slightly different
# locations.
#
# The validator searches the most likely locations and prints
# which one it selected.
# ------------------------------------------------------------

REVIEW_FILE_CANDIDATES = [

    ROOT
    / "data"
    / "data_scraping"
    / "critiquebrainz_matched_reviews.csv",

    ROOT
    / "data"
    / "raw"
    / "music"
    / "critiquebrainz_matched_reviews.csv",

    ROOT
    / "data"
    / "processed"
    / "reviews"
    / "critiquebrainz_matched_reviews.csv",

    ROOT
    / "critiquebrainz_matched_reviews.csv",

]


MUSIC_FILE_CANDIDATES = [

    ROOT
    / "data"
    / "data_scraping"
    / "music_data.csv",

    ROOT
    / "data"
    / "raw"
    / "music"
    / "music_data.csv",

    ROOT
    / "data"
    / "processed"
    / "music"
    / "music_data.csv",

    ROOT
    / "music_data.csv",

]


# ============================================================
# EXPECTED REVIEW COLUMNS
# ============================================================

EXPECTED_REVIEW_COLUMNS = [

    "review_id",

    "original_entity_id",

    "original_entity_type",

    "artist_mbid",

    "artist_name_musicbrainz",

    "artist_name_dataset",

    "matched",

    "language",

    "created",

    "rating",

    "text",

    "info_url",

]


# ============================================================
# PATH RESOLUTION
# ============================================================


def find_existing_file(
    candidates,
    description
):
    """
    Return the first existing file from a list of candidate
    locations.
    """

    for path in candidates:

        if path.exists():

            return path

    searched = "\n".join(
        f"  - {path}"
        for path in candidates
    )

    raise FileNotFoundError(

        f"Could not find {description}.\n\n"
        f"Searched:\n"
        f"{searched}\n\n"
        f"If your file is elsewhere, add its path to the "
        f"candidate list at the top of this script."

    )


# ============================================================
# BOOLEAN NORMALIZATION
# ============================================================


def normalize_boolean(
    series
):
    """
    Convert common CSV boolean representations to True/False.

    Handles values such as:

        True
        False
        "true"
        "false"
        1
        0
    """

    return (

        series
        .astype("string")
        .str.strip()
        .str.lower()
        .map({

            "true": True,

            "false": False,

            "1": True,

            "0": False,

            "yes": True,

            "no": False,

        })

    )


# ============================================================
# PRINT SECTION
# ============================================================


def section(
    title
):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    print()


# ============================================================
# MAIN
# ============================================================


def main():

    section(
        "MUSIC REVIEW VALIDATION"
    )


    # ========================================================
    # LOCATE FILES
    # ========================================================

    reviews_path = find_existing_file(

        REVIEW_FILE_CANDIDATES,

        "CritiqueBrainz matched review file"

    )

    music_path = find_existing_file(

        MUSIC_FILE_CANDIDATES,

        "music dataset"

    )


    print(
        "Review file:"
    )

    print(
        reviews_path
    )

    print()

    print(
        "Music dataset:"
    )

    print(
        music_path
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    section(
        "LOADING DATA"
    )

    reviews = pd.read_csv(
        reviews_path,
        low_memory=False
    )

    music = pd.read_csv(
        music_path,
        low_memory=False
    )


    print(
        f"Review rows:    "
        f"{len(reviews):,}"
    )

    print(
        f"Music entities: "
        f"{len(music):,}"
    )


    # ========================================================
    # COLUMN VALIDATION
    # ========================================================

    section(
        "COLUMN VALIDATION"
    )

    missing_review_columns = [

        column

        for column
        in EXPECTED_REVIEW_COLUMNS

        if column not in reviews.columns

    ]


    if missing_review_columns:

        print(
            "WARNING — missing expected review columns:"
        )

        for column in missing_review_columns:

            print(
                f"  - {column}"
            )

    else:

        print(
            "All expected review columns are present."
        )


    if "mbid" not in music.columns:

        raise ValueError(

            "The music dataset does not contain an "
            "'mbid' column.\n\n"
            f"Available columns:\n"
            f"{list(music.columns)}"

        )


    # ========================================================
    # BASIC MATCH INFORMATION
    # ========================================================

    section(
        "MATCH INFORMATION"
    )

    if "matched" in reviews.columns:

        matched_values = normalize_boolean(
            reviews["matched"]
        )

        matched_count = int(
            (matched_values == True).sum()
        )

        unmatched_count = int(
            (matched_values == False).sum()
        )

        unknown_match_count = int(
            matched_values.isna().sum()
        )


        print(
            f"Rows marked matched:    "
            f"{matched_count:,}"
        )

        print(
            f"Rows marked unmatched:  "
            f"{unmatched_count:,}"
        )

        print(
            f"Unknown match status:   "
            f"{unknown_match_count:,}"
        )

    else:

        print(
            "No 'matched' column available."
        )


    # ========================================================
    # REVIEW IDs
    # ========================================================

    section(
        "REVIEW ID VALIDATION"
    )

    review_ids = (

        reviews["review_id"]
        .astype("string")
        .str.strip()

    )

    missing_review_id_mask = (

        review_ids.isna()

        |

        (
            review_ids == ""
        )

    )


    print(
        f"Total review rows:       "
        f"{len(reviews):,}"
    )

    print(
        f"Unique review IDs:       "
        f"{review_ids.nunique(dropna=True):,}"
    )

    print(
        f"Missing review IDs:      "
        f"{int(missing_review_id_mask.sum()):,}"
    )


    duplicate_mask = (

        review_ids.notna()

        &

        review_ids.duplicated(
            keep=False
        )

    )

    duplicate_rows = reviews[
        duplicate_mask
    ].copy()


    duplicate_id_count = (

        duplicate_rows[
            "review_id"
        ]
        .nunique()

        if len(duplicate_rows) > 0

        else 0

    )


    duplicate_extra_rows = (

        len(reviews)

        -

        review_ids.nunique(
            dropna=True
        )

        -

        int(
            missing_review_id_mask.sum()
        )

    )


    print(
        f"Duplicate review IDs:    "
        f"{duplicate_id_count:,}"
    )

    print(
        f"Extra duplicate rows:    "
        f"{max(duplicate_extra_rows, 0):,}"
    )


    # --------------------------------------------------------
    # Check whether the same review ID points to more than
    # one artist.
    #
    # This is much more important than harmless duplicate
    # copies of the exact same review.
    # --------------------------------------------------------

    conflicting_review_ids = 0

    if (
        len(duplicate_rows) > 0

        and

        "artist_mbid"
        in duplicate_rows.columns
    ):

        conflicts = (

            duplicate_rows
            .groupby(
                "review_id"
            )["artist_mbid"]
            .nunique(
                dropna=True
            )

        )

        conflicting_review_ids = int(
            (
                conflicts > 1
            ).sum()
        )


    print(
        f"Duplicate IDs linked to "
        f"multiple artists: "
        f"{conflicting_review_ids:,}"
    )


    # ========================================================
    # ARTIST MBIDS
    # ========================================================

    section(
        "ARTIST MBID VALIDATION"
    )

    artist_mbids = (

        reviews["artist_mbid"]
        .astype("string")
        .str.strip()

    )

    missing_artist_mask = (

        artist_mbids.isna()

        |

        (
            artist_mbids == ""
        )

    )


    print(
        f"Missing artist MBIDs:    "
        f"{int(missing_artist_mask.sum()):,}"
    )

    print(
        f"Unique reviewed artists: "
        f"{artist_mbids.nunique(dropna=True):,}"
    )


    # ========================================================
    # CHECK AGAINST CURRENT MUSIC DATASET
    # ========================================================

    section(
        "CURRENT MUSIC DATASET ALIGNMENT"
    )

    music_mbids = (

        music["mbid"]
        .astype("string")
        .str.strip()

    )

    music_mbid_set = set(

        music_mbids[
            music_mbids.notna()
            &
            (
                music_mbids != ""
            )
        ]

    )

    reviewed_mbid_set = set(

        artist_mbids[
            artist_mbids.notna()
            &
            (
                artist_mbids != ""
            )
        ]

    )

    known_reviewed_artists = (
        reviewed_mbid_set
        &
        music_mbid_set
    )

    unknown_reviewed_artists = (
        reviewed_mbid_set
        -
        music_mbid_set
    )


    print(
        f"Unique MBIDs in music dataset: "
        f"{len(music_mbid_set):,}"
    )

    print(
        f"Reviewed artist MBIDs:         "
        f"{len(reviewed_mbid_set):,}"
    )

    print(
        f"Reviewed artists still present "
        f"in music dataset:              "
        f"{len(known_reviewed_artists):,}"
    )

    print(
        f"Reviewed artists NOT present "
        f"in music dataset:              "
        f"{len(unknown_reviewed_artists):,}"
    )


    if len(
        music_mbid_set
    ) > 0:

        coverage = (

            len(
                known_reviewed_artists
            )

            /

            len(
                music_mbid_set
            )

            *

            100

        )

        print(
            f"Artist review coverage:         "
            f"{coverage:.2f}%"
        )


    # ========================================================
    # REVIEW TEXT
    # ========================================================

    section(
        "REVIEW TEXT"
    )

    review_text = (

        reviews["text"]
        .astype("string")
        .fillna("")
        .str.strip()

    )

    empty_text_mask = (
        review_text == ""
    )


    print(
        f"Empty review texts:     "
        f"{int(empty_text_mask.sum()):,}"
    )

    print(
        f"Usable review texts:    "
        f"{int((~empty_text_mask).sum()):,}"
    )


    if (
        (~empty_text_mask).any()
    ):

        text_lengths = (

            review_text[
                ~empty_text_mask
            ]
            .str.len()

        )

        word_counts = (

            review_text[
                ~empty_text_mask
            ]
            .str.split()
            .str.len()

        )


        print()
        print(
            "Review character length:"
        )

        print(
            text_lengths.describe(
                percentiles=[
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                    0.99
                ]
            ).to_string()
        )


        print()
        print(
            "Review word count:"
        )

        print(
            word_counts.describe(
                percentiles=[
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                    0.99
                ]
            ).to_string()
        )


    # ========================================================
    # LANGUAGE
    # ========================================================

    section(
        "LANGUAGE DISTRIBUTION"
    )

    if "language" in reviews.columns:

        language_counts = (

            reviews["language"]
            .fillna("<missing>")
            .astype(str)
            .value_counts()

        )


        print(
            language_counts.to_string()
        )

    else:

        print(
            "No language column available."
        )


    # ========================================================
    # ENTITY TYPE
    # ========================================================

    section(
        "REVIEW ENTITY TYPES"
    )

    if (
        "original_entity_type"
        in reviews.columns
    ):

        entity_counts = (

            reviews[
                "original_entity_type"
            ]
            .fillna("<missing>")
            .astype(str)
            .value_counts()

        )


        print(
            entity_counts.to_string()
        )

    else:

        print(
            "No original_entity_type "
            "column available."
        )


    # ========================================================
    # RATINGS
    # ========================================================

    section(
        "REVIEW RATINGS"
    )

    ratings = pd.to_numeric(

        reviews["rating"],

        errors="coerce"

    )


    print(
        f"Reviews with rating:     "
        f"{int(ratings.notna().sum()):,}"
    )

    print(
        f"Reviews without rating:  "
        f"{int(ratings.isna().sum()):,}"
    )


    if ratings.notna().any():

        print()
        print(
            "Rating distribution:"
        )

        print(
            ratings.describe().to_string()
        )


        print()
        print(
            "Exact rating values:"
        )

        print(

            ratings
            .dropna()
            .value_counts()
            .sort_index()
            .to_string()

        )


    # ========================================================
    # REVIEWS PER ARTIST
    # ========================================================

    section(
        "REVIEWS PER ARTIST"
    )

    valid_reviews = reviews.copy()

    valid_reviews[
        "artist_mbid_clean"
    ] = artist_mbids

    valid_reviews[
        "review_text_clean"
    ] = review_text


    valid_reviews = valid_reviews[

        valid_reviews[
            "artist_mbid_clean"
        ].notna()

        &

        (
            valid_reviews[
                "artist_mbid_clean"
            ]
            != ""
        )

        &

        (
            valid_reviews[
                "review_text_clean"
            ]
            != ""
        )

    ].copy()


    reviews_per_artist = (

        valid_reviews
        .groupby(
            "artist_mbid_clean"
        )
        .size()

    )


    print(
        f"Artists with usable reviews: "
        f"{len(reviews_per_artist):,}"
    )

    print()

    print(
        "Reviews-per-artist distribution:"
    )

    print(
        reviews_per_artist.describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        ).to_string()
    )


    # --------------------------------------------------------
    # Thresholds particularly useful for choosing:
    #
    #   MAX_REVIEWS_PER_ARTIST
    #   MIN_REVIEWS during fusion evaluation
    # --------------------------------------------------------

    thresholds = [

        1,

        2,

        3,

        5,

        10,

        25,

        50,

        100,

    ]


    print()
    print(
        "Artist review coverage by threshold:"
    )


    for threshold in thresholds:

        count = int(

            (
                reviews_per_artist
                >= threshold
            ).sum()

        )

        percentage = (

            count
            /
            len(reviews_per_artist)
            *
            100

            if len(
                reviews_per_artist
            ) > 0

            else 0.0

        )


        print(

            f">= {threshold:>3}: "
            f"{count:>6,} artists "
            f"({percentage:6.2f}%)"

        )


    # ========================================================
    # REVIEW SOURCES PER ARTIST
    # ========================================================

    section(
        "ARTIST VS RELEASE REVIEW COVERAGE"
    )

    if (
        "original_entity_type"
        in valid_reviews.columns
    ):

        source_artist_counts = (

            valid_reviews
            .groupby(
                "artist_mbid_clean"
            )[
                "original_entity_type"
            ]
            .agg(
                lambda values:
                    set(
                        values.dropna()
                    )
            )

        )


        direct_only = 0
        release_only = 0
        mixed = 0


        for source_types in (
            source_artist_counts
        ):

            has_artist = (
                "artist"
                in source_types
            )

            has_release = bool(

                {
                    "release",
                    "release_group"
                }

                &

                source_types

            )


            if (
                has_artist
                and
                has_release
            ):

                mixed += 1


            elif has_artist:

                direct_only += 1


            elif has_release:

                release_only += 1


        print(
            f"Artists with direct artist reviews only: "
            f"{direct_only:,}"
        )

        print(
            f"Artists with release/album reviews only: "
            f"{release_only:,}"
        )

        print(
            f"Artists with both sources:               "
            f"{mixed:,}"
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    section(
        "VALIDATION SUMMARY"
    )

    print(
        f"Matched review rows:             "
        f"{len(reviews):,}"
    )

    print(
        f"Unique review IDs:               "
        f"{review_ids.nunique(dropna=True):,}"
    )

    print(
        f"Duplicate review IDs:            "
        f"{duplicate_id_count:,}"
    )

    print(
        f"Conflicting duplicate IDs:       "
        f"{conflicting_review_ids:,}"
    )

    print(
        f"Unique reviewed artists:         "
        f"{len(reviewed_mbid_set):,}"
    )

    print(
        f"Artists present in music data:   "
        f"{len(known_reviewed_artists):,}"
    )

    print(
        f"Empty review texts:              "
        f"{int(empty_text_mask.sum()):,}"
    )

    print()

    print(
        "No files were modified."
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()