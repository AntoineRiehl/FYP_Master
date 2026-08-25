# src/domains/restaurants/prepare_restaurant_reviews.py

# ============================================================
# PREPARE YELP RESTAURANT REVIEWS
#
# Purpose:
#
#   1. Load the business IDs currently present in the
#      Restaurant atlas
#
#   2. Stream Yelp review.json
#
#   3. Keep only reviews belonging to atlas restaurants
#
#   4. Clean and standardize review-level fields
#
#   5. Remove duplicate review IDs
#
#   6. Save a clean review-level dataset for Sentence-BERT
#
#   7. Build a restaurant-level review summary
#
#
# Input:
#
#   data/raw/restaurants/review.json
#
#   frontend/public/data/restaurants/atlas.json
#
#
# Outputs:
#
#   data/processed/reviews/
#       restaurant_reviews_prepared.csv
#
#       restaurant_review_summary.csv
#
#
# Important:
#
#   - business_id is the canonical restaurant identifier
#
#   - tips are NOT included here because they already
#     contribute to the existing TF-IDF representation
#
#   - review text is NOT lowercased, stemmed, lemmatized,
#     or stripped of punctuation
#
#   - all atlas restaurants are retained regardless of
#     review count
#
# ============================================================


from pathlib import Path

import csv
import json
import math
import re

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]


REVIEW_FILE = (
    ROOT
    / "data"
    / "raw"
    / "restaurants"
    / "review.json"
)


ATLAS_FILE = (
    ROOT
    / "frontend"
    / "public"
    / "data"
    / "restaurants"
    / "atlas.json"
)


OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "reviews"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_REVIEWS = (
    OUTPUT_DIR
    / "restaurant_reviews_prepared.csv"
)


OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "restaurant_review_summary.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

PROGRESS_EVERY = 250_000


# ============================================================
# REVIEW-LEVEL OUTPUT COLUMNS
# ============================================================

OUTPUT_COLUMNS = [

    "business_id",

    "review_id",

    "review_text",

    "rating",

    "review_date",

    "useful",

    "funny",

    "cool",

    "review_char_count",

    "review_word_count",

    "source",

]


# ============================================================
# PRINT SECTION
# ============================================================


def section(
    title
):

    print()

    print(
        "=" * 65
    )

    print(
        title
    )

    print(
        "=" * 65
    )

    print()


# ============================================================
# FILE VALIDATION
# ============================================================


def validate_files():

    missing = [

        path

        for path in [
            REVIEW_FILE,
            ATLAS_FILE
        ]

        if not path.exists()

    ]


    if missing:

        formatted = "\n".join(

            f"  - {path}"

            for path in missing

        )


        raise FileNotFoundError(

            "Required files were not found:\n\n"
            f"{formatted}"

        )


# ============================================================
# LOAD CURRENT RESTAURANT IDS
# ============================================================


def load_atlas_business_ids():
    """
    Load the business IDs currently represented in the
    Restaurant atlas.

    This guarantees that review preprocessing uses exactly
    the same restaurant population as the existing atlas.
    """

    section(
        "LOADING CURRENT RESTAURANT ATLAS"
    )


    with open(
        ATLAS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(
            file
        )


    if isinstance(
        data,
        list
    ):

        nodes = data


    elif (
        isinstance(
            data,
            dict
        )
        and
        isinstance(
            data.get("atlas"),
            list
        )
    ):

        nodes = data[
            "atlas"
        ]


    else:

        raise ValueError(

            "Unexpected Restaurant atlas JSON structure."

        )


    business_ids = []


    for node in nodes:

        business_id = (

            node.get(
                "source_id"
            )

            or

            node.get(
                "id"
            )

        )


        if business_id is None:

            continue


        business_id = str(
            business_id
        ).strip()


        if business_id:

            business_ids.append(
                business_id
            )


    duplicate_count = (

        len(
            business_ids
        )

        -

        len(
            set(
                business_ids
            )
        )

    )


    if duplicate_count > 0:

        raise ValueError(

            f"Restaurant atlas contains "
            f"{duplicate_count:,} duplicate business IDs."

        )


    business_id_set = set(
        business_ids
    )


    print(
        f"Atlas restaurants: "
        f"{len(business_id_set):,}"
    )


    return business_id_set


# ============================================================
# CLEAN TEXT
# ============================================================


def clean_review_text(
    text
):
    """
    Conservatively clean Yelp review text for Sentence-BERT.

    We deliberately do NOT:

        - lowercase
        - remove stop words
        - stem
        - lemmatize
        - remove punctuation

    Only whitespace is normalized.
    """

    if text is None:

        return ""


    text = str(
        text
    )


    text = re.sub(

        r"\s+",

        " ",

        text

    )


    return text.strip()


# ============================================================
# SAFE NUMERIC HELPERS
# ============================================================


def safe_float(
    value
):

    try:

        number = float(
            value
        )


        if math.isfinite(
            number
        ):

            return number


    except (
        TypeError,
        ValueError
    ):

        pass


    return None


def safe_int(
    value,
    default=0
):

    try:

        return int(
            value
        )


    except (
        TypeError,
        ValueError
    ):

        return default


# ============================================================
# INITIALIZE SUMMARY RECORD
# ============================================================


def create_summary_record():
    """
    Running aggregation values for one restaurant.
    """

    return {

        "review_count":
            0,

        "review_rating_count":
            0,

        "rating_sum":
            0.0,

        "rating_sum_squared":
            0.0,

        "useful_sum":
            0,

        "funny_sum":
            0,

        "cool_sum":
            0,

        "review_char_sum":
            0,

        "review_word_sum":
            0,

    }


# ============================================================
# PREPARE REVIEWS
# ============================================================


def prepare_reviews(
    atlas_business_ids
):
    """
    Stream Yelp review.json once.

    Prepared review rows are written directly to CSV.

    Only compact restaurant-level aggregation state is kept
    in memory.
    """

    section(
        "PREPARING YELP RESTAURANT REVIEWS"
    )


    print(
        "Input:"
    )

    print(
        REVIEW_FILE
    )

    print()


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    reviews_scanned = 0

    atlas_reviews_seen = 0

    prepared_reviews = 0

    missing_review_ids = 0

    missing_business_ids = 0

    empty_review_texts = 0

    duplicate_review_ids = 0

    invalid_ratings = 0


    # --------------------------------------------------------
    # Deduplication
    #
    # Only review IDs belonging to atlas restaurants are
    # stored.
    #
    # This avoids keeping IDs for the entire 6.99M-review
    # Yelp dataset.
    # --------------------------------------------------------

    seen_review_ids = set()


    # --------------------------------------------------------
    # One small running summary per restaurant.
    # --------------------------------------------------------

    restaurant_stats = {}


    # --------------------------------------------------------
    # Stream prepared rows directly to disk.
    # --------------------------------------------------------

    with open(

        OUTPUT_REVIEWS,

        "w",

        encoding="utf-8",

        newline=""

    ) as output_file:


        writer = csv.DictWriter(

            output_file,

            fieldnames=
                OUTPUT_COLUMNS,

            quoting=
                csv.QUOTE_MINIMAL

        )


        writer.writeheader()


        # ====================================================
        # STREAM RAW YELP REVIEWS
        # ====================================================

        with open(

            REVIEW_FILE,

            "r",

            encoding="utf-8"

        ) as input_file:


            for line in input_file:


                line = line.strip()


                if not line:

                    continue


                review = json.loads(
                    line
                )


                reviews_scanned += 1


                # =============================================
                # BUSINESS ID
                # =============================================

                business_id = review.get(
                    "business_id"
                )


                if (
                    business_id is None
                    or
                    str(
                        business_id
                    ).strip() == ""
                ):

                    missing_business_ids += 1

                    continue


                business_id = str(
                    business_id
                ).strip()


                # -------------------------------------------------
                # Ignore reviews for businesses not currently
                # represented in the Restaurant atlas.
                # -------------------------------------------------

                if (
                    business_id
                    not in
                    atlas_business_ids
                ):

                    if (
                        reviews_scanned
                        %
                        PROGRESS_EVERY
                        ==
                        0
                    ):

                        print(

                            f"\rReviews scanned: "
                            f"{reviews_scanned:,} | "
                            f"Atlas reviews prepared: "
                            f"{prepared_reviews:,}",

                            end=""

                        )


                    continue


                atlas_reviews_seen += 1


                # =============================================
                # REVIEW ID
                # =============================================

                review_id = review.get(
                    "review_id"
                )


                if (
                    review_id is None
                    or
                    str(
                        review_id
                    ).strip() == ""
                ):

                    missing_review_ids += 1

                    continue


                review_id = str(
                    review_id
                ).strip()


                # =============================================
                # DUPLICATE REVIEW ID
                # =============================================

                if (
                    review_id
                    in
                    seen_review_ids
                ):

                    duplicate_review_ids += 1

                    continue


                seen_review_ids.add(
                    review_id
                )


                # =============================================
                # REVIEW TEXT
                # =============================================

                review_text = clean_review_text(

                    review.get(
                        "text"
                    )

                )


                if not review_text:

                    empty_review_texts += 1

                    continue


                # =============================================
                # RATING
                # =============================================

                rating = safe_float(

                    review.get(
                        "stars"
                    )

                )


                if (
                    rating is not None
                    and
                    not (
                        1.0
                        <=
                        rating
                        <=
                        5.0
                    )
                ):

                    invalid_ratings += 1

                    rating = None


                # =============================================
                # DATE
                # =============================================

                review_date = review.get(
                    "date"
                )


                if review_date is not None:

                    review_date = str(
                        review_date
                    ).strip()


                    if not review_date:

                        review_date = None


                # =============================================
                # ENGAGEMENT
                # =============================================

                useful = safe_int(

                    review.get(
                        "useful"
                    ),

                    0

                )


                funny = safe_int(

                    review.get(
                        "funny"
                    ),

                    0

                )


                cool = safe_int(

                    review.get(
                        "cool"
                    ),

                    0

                )


                # =============================================
                # TEXT LENGTH
                # =============================================

                review_char_count = len(
                    review_text
                )


                review_word_count = len(

                    review_text.split()

                )


                # =============================================
                # WRITE PREPARED REVIEW
                # =============================================

                writer.writerow({

                    "business_id":
                        business_id,

                    "review_id":
                        review_id,

                    "review_text":
                        review_text,

                    "rating":
                        (
                            rating
                            if rating is not None
                            else ""
                        ),

                    "review_date":
                        (
                            review_date
                            if review_date is not None
                            else ""
                        ),

                    "useful":
                        useful,

                    "funny":
                        funny,

                    "cool":
                        cool,

                    "review_char_count":
                        review_char_count,

                    "review_word_count":
                        review_word_count,

                    "source":
                        "Yelp",

                })


                prepared_reviews += 1


                # =============================================
                # UPDATE RESTAURANT SUMMARY
                # =============================================

                stats = restaurant_stats.get(
                    business_id
                )


                if stats is None:

                    stats = (
                        create_summary_record()
                    )

                    restaurant_stats[
                        business_id
                    ] = stats


                stats[
                    "review_count"
                ] += 1


                stats[
                    "review_char_sum"
                ] += review_char_count


                stats[
                    "review_word_sum"
                ] += review_word_count


                stats[
                    "useful_sum"
                ] += useful


                stats[
                    "funny_sum"
                ] += funny


                stats[
                    "cool_sum"
                ] += cool


                if rating is not None:

                    stats[
                        "review_rating_count"
                    ] += 1


                    stats[
                        "rating_sum"
                    ] += rating


                    stats[
                        "rating_sum_squared"
                    ] += (

                        rating
                        *
                        rating

                    )


                # =============================================
                # PROGRESS
                # =============================================

                if (
                    reviews_scanned
                    %
                    PROGRESS_EVERY
                    ==
                    0
                ):

                    print(

                        f"\rReviews scanned: "
                        f"{reviews_scanned:,} | "
                        f"Atlas reviews prepared: "
                        f"{prepared_reviews:,}",

                        end=""

                    )


    print()


    # ========================================================
    # PREPARATION SUMMARY
    # ========================================================

    print()

    print(
        f"Reviews scanned:               "
        f"{reviews_scanned:,}"
    )


    print(
        f"Atlas reviews encountered:     "
        f"{atlas_reviews_seen:,}"
    )


    print(
        f"Prepared reviews:              "
        f"{prepared_reviews:,}"
    )


    print(
        f"Duplicate review IDs removed:  "
        f"{duplicate_review_ids:,}"
    )


    print(
        f"Missing review IDs removed:    "
        f"{missing_review_ids:,}"
    )


    print(
        f"Missing business IDs removed:  "
        f"{missing_business_ids:,}"
    )


    print(
        f"Empty review texts removed:    "
        f"{empty_review_texts:,}"
    )


    print(
        f"Invalid ratings set missing:   "
        f"{invalid_ratings:,}"
    )


    print(
        f"Restaurants represented:       "
        f"{len(restaurant_stats):,}"
    )


    print()

    print(
        "Prepared review output:"
    )

    print(
        OUTPUT_REVIEWS
    )


    return (
        restaurant_stats,
        prepared_reviews
    )


# ============================================================
# CREATE RESTAURANT SUMMARY
# ============================================================


def create_restaurant_summary(

    restaurant_stats,
    atlas_business_ids

):
    """
    Convert the running restaurant statistics into a clean
    restaurant-level review summary.
    """

    section(
        "CREATING RESTAURANT REVIEW SUMMARY"
    )


    summary_rows = []


    # --------------------------------------------------------
    # Iterate over every atlas restaurant.
    #
    # In our validated dataset all restaurants have reviews,
    # but including every atlas ID makes the output robust.
    # --------------------------------------------------------

    for business_id in sorted(
        atlas_business_ids
    ):


        stats = restaurant_stats.get(
            business_id
        )


        if stats is None:

            summary_rows.append({

                "business_id":
                    business_id,

                "review_count":
                    0,

                "review_rating_count":
                    0,

                "review_rating_mean":
                    None,

                "review_rating_std":
                    None,

                "avg_useful":
                    None,

                "avg_funny":
                    None,

                "avg_cool":
                    None,

                "avg_review_char_count":
                    None,

                "avg_review_word_count":
                    None,

            })


            continue


        review_count = stats[
            "review_count"
        ]


        rating_count = stats[
            "review_rating_count"
        ]


        # ====================================================
        # RATING MEAN / STANDARD DEVIATION
        # ====================================================

        if rating_count > 0:

            rating_mean = (

                stats[
                    "rating_sum"
                ]

                /

                rating_count

            )


        else:

            rating_mean = None


        # ----------------------------------------------------
        # Sample standard deviation (ddof=1), matching the
        # default behaviour of pandas Series.std().
        # ----------------------------------------------------

        if rating_count > 1:

            numerator = (

                stats[
                    "rating_sum_squared"
                ]

                -

                (
                    stats[
                        "rating_sum"
                    ]
                    ** 2
                )

                /

                rating_count

            )


            # Numerical rounding can theoretically create an
            # extremely small negative value.
            numerator = max(
                numerator,
                0.0
            )


            rating_std = math.sqrt(

                numerator

                /

                (
                    rating_count
                    -
                    1
                )

            )


        else:

            rating_std = None


        # ====================================================
        # AVERAGES
        # ====================================================

        avg_useful = (

            stats[
                "useful_sum"
            ]

            /

            review_count

        )


        avg_funny = (

            stats[
                "funny_sum"
            ]

            /

            review_count

        )


        avg_cool = (

            stats[
                "cool_sum"
            ]

            /

            review_count

        )


        avg_review_char_count = (

            stats[
                "review_char_sum"
            ]

            /

            review_count

        )


        avg_review_word_count = (

            stats[
                "review_word_sum"
            ]

            /

            review_count

        )


        summary_rows.append({

            "business_id":
                business_id,

            "review_count":
                review_count,

            "review_rating_count":
                rating_count,

            "review_rating_mean":
                rating_mean,

            "review_rating_std":
                rating_std,

            "avg_useful":
                avg_useful,

            "avg_funny":
                avg_funny,

            "avg_cool":
                avg_cool,

            "avg_review_char_count":
                avg_review_char_count,

            "avg_review_word_count":
                avg_review_word_count,

        })


    summary = pd.DataFrame(
        summary_rows
    )


    # ========================================================
    # SAVE
    # ========================================================

    summary.to_csv(

        OUTPUT_SUMMARY,

        index=False,

        encoding="utf-8"

    )


    # ========================================================
    # INFORMATION
    # ========================================================

    print(
        f"Atlas restaurants:              "
        f"{len(summary):,}"
    )


    print(
        f"Restaurants with reviews:       "
        f"{int((summary['review_count'] > 0).sum()):,}"
    )


    print(
        f"Restaurants without reviews:    "
        f"{int((summary['review_count'] == 0).sum()):,}"
    )


    print(
        f"Total reviews represented:      "
        f"{int(summary['review_count'].sum()):,}"
    )


    print()

    print(
        "Review-count distribution:"
    )


    print(

        summary[
            "review_count"
        ]
        .describe(

            percentiles=[

                0.25,

                0.50,

                0.75,

                0.90,

                0.95,

                0.99,

            ]

        )
        .to_string()

    )


    print()

    print(
        "Restaurants by review threshold:"
    )


    for threshold in [

        1,

        10,

        20,

        25,

        50,

        100,

        250,

        500,

        1000,

    ]:

        count = int(

            (
                summary[
                    "review_count"
                ]
                >=
                threshold
            ).sum()

        )


        print(

            f">= {threshold:>4}: "
            f"{count:>7,} restaurants"

        )


    print()

    print(
        "Restaurant-level summary output:"
    )

    print(
        OUTPUT_SUMMARY
    )


    return summary


# ============================================================
# MAIN
# ============================================================


def main():

    section(
        "RESTAURANT REVIEW PREPARATION"
    )


    validate_files()


    # ========================================================
    # 1. CURRENT RESTAURANT POPULATION
    # ========================================================

    atlas_business_ids = (
        load_atlas_business_ids()
    )


    # ========================================================
    # 2. PREPARE REVIEW-LEVEL DATA
    # ========================================================

    (
        restaurant_stats,
        prepared_reviews

    ) = prepare_reviews(

        atlas_business_ids

    )


    # ========================================================
    # 3. CREATE RESTAURANT SUMMARY
    # ========================================================

    summary = (
        create_restaurant_summary(

            restaurant_stats,

            atlas_business_ids

        )
    )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    section(
        "FINAL VALIDATION"
    )


    represented_reviews = int(

        summary[
            "review_count"
        ].sum()

    )


    print(
        f"Prepared review rows:       "
        f"{prepared_reviews:,}"
    )


    print(
        f"Reviews in summary:         "
        f"{represented_reviews:,}"
    )


    if (
        prepared_reviews
        !=
        represented_reviews
    ):

        raise ValueError(

            "Prepared review count does not match "
            "restaurant summary count."

        )


    print(
        f"Restaurants in atlas:       "
        f"{len(atlas_business_ids):,}"
    )


    print(
        f"Restaurants in summary:     "
        f"{len(summary):,}"
    )


    if (
        len(
            atlas_business_ids
        )
        !=
        len(
            summary
        )
    ):

        raise ValueError(

            "Restaurant summary does not contain exactly "
            "one row per atlas restaurant."

        )


    section(
        "RESTAURANT REVIEW PREPARATION COMPLETE"
    )


    print(
        "Review-level dataset:"
    )

    print(
        OUTPUT_REVIEWS
    )


    print()

    print(
        "Restaurant-level summary:"
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