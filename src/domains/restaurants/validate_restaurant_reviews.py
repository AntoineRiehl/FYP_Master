# src/domains/restaurants/validate_restaurant_reviews.py

# ============================================================
# VALIDATE YELP RESTAURANT REVIEWS
#
# Purpose:
#
#   Inspect the raw Yelp review dataset before creating the
#   Restaurant Sentence-BERT review enrichment.
#
#
# Raw inputs:
#
#   data/raw/restaurants/
#       business.json
#       review.json
#       tip.json
#
#
# Current atlas:
#
#   frontend/public/data/restaurants/atlas.json
#
#
# Important design decision:
#
#   The CURRENT Restaurant atlas is used to define the
#   restaurant population.
#
#   This avoids introducing a second, potentially different,
#   restaurant-filtering rule during review preprocessing.
#
#
# This script DOES NOT modify any files.
#
#
# It reports:
#
#   - raw Yelp business count
#   - current atlas restaurant count
#   - alignment between atlas IDs and business.json
#
#   - total Yelp review rows
#   - reviews linked to current atlas restaurants
#   - restaurants with reviews
#   - missing IDs / empty review text
#   - stars distribution
#   - review length distribution
#   - reviews-per-restaurant distribution
#
#   - tip coverage for reference only
#
#
# NOTE:
#
#   Tips are NOT candidates for Sentence-BERT review
#   enrichment because they already contribute to the
#   existing TF-IDF representation.
#
# ============================================================


from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]


RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "restaurants"
)


BUSINESS_FILE = (
    RAW_DIR
    / "business.json"
)


REVIEW_FILE = (
    RAW_DIR
    / "review.json"
)


TIP_FILE = (
    RAW_DIR
    / "tip.json"
)


ATLAS_FILE = (
    ROOT
    / "frontend"
    / "public"
    / "data"
    / "restaurants"
    / "atlas.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Print progress every N review rows.
PROGRESS_EVERY = 500_000


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
# VALIDATE FILES
# ============================================================


def validate_files():

    required_files = [

        BUSINESS_FILE,

        REVIEW_FILE,

        ATLAS_FILE,

    ]


    missing = [

        path

        for path in required_files

        if not path.exists()

    ]


    if missing:

        formatted = "\n".join(

            f"  - {path}"

            for path in missing

        )


        raise FileNotFoundError(

            "Required Restaurant files were not found:\n\n"

            f"{formatted}"

        )


# ============================================================
# LOAD CURRENT ATLAS IDS
# ============================================================


def load_atlas_business_ids():
    """
    Load business IDs from the currently exported Restaurant
    atlas.

    These IDs define the restaurant population that we want
    to enrich with Yelp reviews.
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


    # --------------------------------------------------------
    # Support:
    #
    #   [node, node, ...]
    #
    # or:
    #
    #   {"atlas": [...]}
    # --------------------------------------------------------

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
# VALIDATE BUSINESS FILE
# ============================================================


def inspect_business_file(
    atlas_business_ids
):
    """
    Stream business.json and verify that the businesses
    currently displayed in the Restaurant atlas still exist
    in the raw Yelp dataset.
    """

    section(
        "YELP BUSINESS DATA"
    )


    total_businesses = 0

    raw_business_ids = set()

    atlas_businesses_found = set()

    businesses_with_categories = 0

    businesses_category_restaurants = 0


    with open(
        BUSINESS_FILE,
        "r",
        encoding="utf-8"
    ) as file:


        for line in file:

            line = line.strip()


            if not line:

                continue


            business = json.loads(
                line
            )


            total_businesses += 1


            business_id = business.get(
                "business_id"
            )


            if business_id:

                business_id = str(
                    business_id
                ).strip()

                raw_business_ids.add(
                    business_id
                )


                if (
                    business_id
                    in
                    atlas_business_ids
                ):

                    atlas_businesses_found.add(
                        business_id
                    )


            categories = business.get(
                "categories"
            )


            if categories:

                businesses_with_categories += 1


                category_tokens = {

                    category.strip().lower()

                    for category
                    in str(
                        categories
                    ).split(",")

                }


                if (
                    "restaurants"
                    in category_tokens
                ):

                    businesses_category_restaurants += 1


    missing_atlas_businesses = (

        atlas_business_ids

        -

        atlas_businesses_found

    )


    print(
        f"Raw Yelp businesses:                    "
        f"{total_businesses:,}"
    )


    print(
        f"Unique raw business IDs:               "
        f"{len(raw_business_ids):,}"
    )


    print(
        f"Businesses with categories:            "
        f"{businesses_with_categories:,}"
    )


    print(
        f"Businesses explicitly categorised "
        f"'Restaurants':                       "
        f"{businesses_category_restaurants:,}"
    )


    print()

    print(
        f"Atlas restaurants:                     "
        f"{len(atlas_business_ids):,}"
    )


    print(
        f"Atlas restaurants found in business.json: "
        f"{len(atlas_businesses_found):,}"
    )


    print(
        f"Atlas IDs missing from business.json:  "
        f"{len(missing_atlas_businesses):,}"
    )


    if missing_atlas_businesses:

        print()

        print(
            "Example missing atlas IDs:"
        )


        for business_id in list(
            missing_atlas_businesses
        )[:10]:

            print(
                f"  - {business_id}"
            )


    return raw_business_ids


# ============================================================
# ONLINE SUMMARY STATS
# ============================================================


class RunningStats:
    """
    Numerically stable running mean / variance calculation.

    This avoids storing millions of review lengths in memory.
    """

    def __init__(
        self
    ):

        self.count = 0

        self.mean = 0.0

        self.m2 = 0.0

        self.minimum = None

        self.maximum = None


    def update(
        self,
        value
    ):

        value = float(
            value
        )


        self.count += 1


        if (
            self.minimum is None
            or
            value < self.minimum
        ):

            self.minimum = value


        if (
            self.maximum is None
            or
            value > self.maximum
        ):

            self.maximum = value


        delta = (
            value
            -
            self.mean
        )


        self.mean += (
            delta
            /
            self.count
        )


        delta2 = (
            value
            -
            self.mean
        )


        self.m2 += (
            delta
            *
            delta2
        )


    @property
    def std(
        self
    ):

        if self.count < 2:

            return 0.0


        return (

            self.m2
            /
            (
                self.count
                -
                1
            )

        ) ** 0.5


# ============================================================
# INSPECT REVIEWS
# ============================================================


def inspect_reviews(
    atlas_business_ids,
    raw_business_ids
):
    """
    Stream Yelp review.json and inspect review coverage for
    the current Restaurant atlas.

    Only lightweight counts are retained in memory.
    """

    section(
        "YELP REVIEW DATA"
    )


    # --------------------------------------------------------
    # Global counts
    # --------------------------------------------------------

    total_reviews = 0

    missing_review_ids = 0

    missing_business_ids = 0

    unknown_business_ids = 0

    empty_review_texts_global = 0


    # --------------------------------------------------------
    # Atlas-linked counts
    # --------------------------------------------------------

    atlas_reviews = 0

    atlas_empty_review_texts = 0

    atlas_missing_review_ids = 0


    reviews_per_business = {}


    # --------------------------------------------------------
    # Ratings
    # --------------------------------------------------------

    star_counts = {

        1: 0,

        2: 0,

        3: 0,

        4: 0,

        5: 0,

    }


    missing_stars = 0


    # --------------------------------------------------------
    # Review-length stats
    # --------------------------------------------------------

    char_stats = RunningStats()

    word_stats = RunningStats()


    # --------------------------------------------------------
    # Helpfulness / engagement fields
    # --------------------------------------------------------

    useful_total = 0

    funny_total = 0

    cool_total = 0


    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    earliest_date = None

    latest_date = None


    with open(
        REVIEW_FILE,
        "r",
        encoding="utf-8"
    ) as file:


        for line in file:

            line = line.strip()


            if not line:

                continue


            review = json.loads(
                line
            )


            total_reviews += 1


            # =================================================
            # REVIEW ID
            # =================================================

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


            # =================================================
            # BUSINESS ID
            # =================================================

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


            if (
                business_id
                not in
                raw_business_ids
            ):

                unknown_business_ids += 1


            # =================================================
            # TEXT
            # =================================================

            text = review.get(
                "text"
            )


            if text is None:

                cleaned_text = ""

            else:

                cleaned_text = str(
                    text
                ).strip()


            if not cleaned_text:

                empty_review_texts_global += 1


            # =================================================
            # ONLY CONTINUE DETAILED ANALYSIS FOR CURRENT
            # ATLAS RESTAURANTS
            # =================================================

            if (
                business_id
                not in
                atlas_business_ids
            ):

                if (
                    total_reviews
                    %
                    PROGRESS_EVERY
                    ==
                    0
                ):

                    print(

                        f"\rReviews scanned: "
                        f"{total_reviews:,} | "
                        f"Atlas reviews: "
                        f"{atlas_reviews:,}",

                        end=""

                    )


                continue


            atlas_reviews += 1


            # =================================================
            # ATLAS REVIEW ID
            # =================================================

            if (
                review_id is None
                or
                str(
                    review_id
                ).strip() == ""
            ):

                atlas_missing_review_ids += 1


            # =================================================
            # ATLAS TEXT
            # =================================================

            if not cleaned_text:

                atlas_empty_review_texts += 1


            else:

                char_count = len(
                    cleaned_text
                )


                word_count = len(
                    cleaned_text.split()
                )


                char_stats.update(
                    char_count
                )


                word_stats.update(
                    word_count
                )


            # =================================================
            # REVIEWS PER RESTAURANT
            # =================================================

            reviews_per_business[
                business_id
            ] = (

                reviews_per_business.get(
                    business_id,
                    0
                )

                +

                1

            )


            # =================================================
            # STARS
            # =================================================

            stars = review.get(
                "stars"
            )


            try:

                stars_value = int(
                    float(
                        stars
                    )
                )


                if (
                    stars_value
                    in
                    star_counts
                ):

                    star_counts[
                        stars_value
                    ] += 1


                else:

                    missing_stars += 1


            except (
                TypeError,
                ValueError
            ):

                missing_stars += 1


            # =================================================
            # ENGAGEMENT
            # =================================================

            try:

                useful_total += int(
                    review.get(
                        "useful",
                        0
                    )
                    or
                    0
                )

            except (
                TypeError,
                ValueError
            ):

                pass


            try:

                funny_total += int(
                    review.get(
                        "funny",
                        0
                    )
                    or
                    0
                )

            except (
                TypeError,
                ValueError
            ):

                pass


            try:

                cool_total += int(
                    review.get(
                        "cool",
                        0
                    )
                    or
                    0
                )

            except (
                TypeError,
                ValueError
            ):

                pass


            # =================================================
            # DATE
            # =================================================

            review_date = review.get(
                "date"
            )


            if review_date:

                review_date = str(
                    review_date
                )


                if (
                    earliest_date is None
                    or
                    review_date
                    <
                    earliest_date
                ):

                    earliest_date = (
                        review_date
                    )


                if (
                    latest_date is None
                    or
                    review_date
                    >
                    latest_date
                ):

                    latest_date = (
                        review_date
                    )


            # =================================================
            # PROGRESS
            # =================================================

            if (
                total_reviews
                %
                PROGRESS_EVERY
                ==
                0
            ):

                print(

                    f"\rReviews scanned: "
                    f"{total_reviews:,} | "
                    f"Atlas reviews: "
                    f"{atlas_reviews:,}",

                    end=""

                )


    print()


    # ========================================================
    # BASIC OUTPUT
    # ========================================================

    print(
        f"Total Yelp reviews:                 "
        f"{total_reviews:,}"
    )


    print(
        f"Missing review IDs:                 "
        f"{missing_review_ids:,}"
    )


    print(
        f"Missing business IDs:               "
        f"{missing_business_ids:,}"
    )


    print(
        f"Unknown business IDs:               "
        f"{unknown_business_ids:,}"
    )


    print(
        f"Empty review text globally:         "
        f"{empty_review_texts_global:,}"
    )


    print()

    print(
        f"Reviews linked to atlas restaurants:"
        f" {atlas_reviews:,}"
    )


    print(
        f"Atlas reviews missing review ID:    "
        f"{atlas_missing_review_ids:,}"
    )


    print(
        f"Atlas reviews with empty text:      "
        f"{atlas_empty_review_texts:,}"
    )


    print(
        f"Atlas restaurants with reviews:     "
        f"{len(reviews_per_business):,}"
    )


    print(
        f"Atlas restaurants without reviews:  "
        f"{len(atlas_business_ids) - len(reviews_per_business):,}"
    )


    # ========================================================
    # COVERAGE
    # ========================================================

    if atlas_business_ids:

        restaurant_coverage = (

            len(
                reviews_per_business
            )

            /

            len(
                atlas_business_ids
            )

            *

            100

        )


        print(
            f"Atlas restaurant review coverage:   "
            f"{restaurant_coverage:.2f}%"
        )


    # ========================================================
    # RATINGS
    # ========================================================

    print()

    print(
        "Atlas review stars:"
    )


    for stars in sorted(
        star_counts
    ):

        print(

            f"  {stars} stars: "
            f"{star_counts[stars]:,}"

        )


    print(
        f"  Missing / invalid: "
        f"{missing_stars:,}"
    )


    # ========================================================
    # LENGTHS
    # ========================================================

    print()

    print(
        "Atlas review character length:"
    )


    print(
        f"  count: {char_stats.count:,}"
    )

    print(
        f"  mean:  {char_stats.mean:.2f}"
    )

    print(
        f"  std:   {char_stats.std:.2f}"
    )

    print(
        f"  min:   {char_stats.minimum}"
    )

    print(
        f"  max:   {char_stats.maximum}"
    )


    print()

    print(
        "Atlas review word count:"
    )


    print(
        f"  count: {word_stats.count:,}"
    )

    print(
        f"  mean:  {word_stats.mean:.2f}"
    )

    print(
        f"  std:   {word_stats.std:.2f}"
    )

    print(
        f"  min:   {word_stats.minimum}"
    )

    print(
        f"  max:   {word_stats.maximum}"
    )


    # ========================================================
    # DATES
    # ========================================================

    print()

    print(
        f"Earliest atlas review date: "
        f"{earliest_date}"
    )


    print(
        f"Latest atlas review date:   "
        f"{latest_date}"
    )


    # ========================================================
    # REVIEWS PER RESTAURANT
    # ========================================================

    section(
        "REVIEWS PER RESTAURANT"
    )


    review_counts = pd.Series(

        list(
            reviews_per_business.values()
        ),

        dtype="int64"

    )


    if len(
        review_counts
    ) > 0:

        print(

            review_counts.describe(

                percentiles=[

                    0.25,

                    0.50,

                    0.75,

                    0.90,

                    0.95,

                    0.99,

                ]

            ).to_string()

        )


        thresholds = [

            1,

            2,

            3,

            5,

            10,

            25,

            50,

            100,

            250,

            500,

            1000,

        ]


        print()

        print(
            "Restaurant coverage by review threshold:"
        )


        for threshold in thresholds:

            count = int(

                (
                    review_counts
                    >=
                    threshold
                ).sum()

            )


            percentage = (

                count

                /

                len(
                    atlas_business_ids
                )

                *

                100

            )


            print(

                f">= {threshold:>4}: "
                f"{count:>7,} restaurants "
                f"({percentage:6.2f}% of atlas)"

            )


    # ========================================================
    # ENGAGEMENT
    # ========================================================

    section(
        "REVIEW ENGAGEMENT"
    )


    if atlas_reviews > 0:

        print(
            f"Average useful votes / review: "
            f"{useful_total / atlas_reviews:.3f}"
        )


        print(
            f"Average funny votes / review:  "
            f"{funny_total / atlas_reviews:.3f}"
        )


        print(
            f"Average cool votes / review:   "
            f"{cool_total / atlas_reviews:.3f}"
        )


    return {

        "total_reviews":
            total_reviews,

        "atlas_reviews":
            atlas_reviews,

        "restaurants_with_reviews":
            len(
                reviews_per_business
            ),

        "reviews_per_business":
            reviews_per_business,

    }


# ============================================================
# INSPECT TIPS
# ============================================================


def inspect_tips(
    atlas_business_ids
):
    """
    Inspect tip coverage for context only.

    Tips already contribute to the original Restaurant
    TF-IDF representation and will NOT be reused as
    Sentence-BERT review input.
    """


    section(
        "YELP TIP DATA"
    )


    if not TIP_FILE.exists():

        print(
            "tip.json not found."
        )

        print(
            "Skipping tip diagnostics."
        )

        return


    total_tips = 0

    atlas_tips = 0

    atlas_businesses_with_tips = set()

    empty_tip_texts = 0


    with open(
        TIP_FILE,
        "r",
        encoding="utf-8"
    ) as file:


        for line in file:

            line = line.strip()


            if not line:

                continue


            tip = json.loads(
                line
            )


            total_tips += 1


            business_id = tip.get(
                "business_id"
            )


            if business_id is None:

                continue


            business_id = str(
                business_id
            ).strip()


            if (
                business_id
                not in
                atlas_business_ids
            ):

                continue


            atlas_tips += 1

            atlas_businesses_with_tips.add(
                business_id
            )


            text = tip.get(
                "text"
            )


            if (
                text is None
                or
                str(
                    text
                ).strip()
                ==
                ""
            ):

                empty_tip_texts += 1


    print(
        f"Total Yelp tips:                 "
        f"{total_tips:,}"
    )


    print(
        f"Tips linked to atlas restaurants:"
        f" {atlas_tips:,}"
    )


    print(
        f"Atlas restaurants with tips:     "
        f"{len(atlas_businesses_with_tips):,}"
    )


    print(
        f"Empty atlas tip texts:           "
        f"{empty_tip_texts:,}"
    )


    print()

    print(
        "Reminder: tips are already part of the existing "
        "TF-IDF semantics and will NOT be included in "
        "Sentence-BERT review embeddings."
    )


# ============================================================
# MAIN
# ============================================================


def main():

    section(
        "RESTAURANT REVIEW VALIDATION"
    )


    print(
        "Business file:"
    )

    print(
        BUSINESS_FILE
    )


    print()

    print(
        "Review file:"
    )

    print(
        REVIEW_FILE
    )


    print()

    print(
        "Tip file:"
    )

    print(
        TIP_FILE
    )


    print()

    print(
        "Current atlas:"
    )

    print(
        ATLAS_FILE
    )


    # ========================================================
    # FILE CHECK
    # ========================================================

    validate_files()


    # ========================================================
    # CURRENT ATLAS
    # ========================================================

    atlas_business_ids = (
        load_atlas_business_ids()
    )


    # ========================================================
    # BUSINESS.JSON
    # ========================================================

    raw_business_ids = (
        inspect_business_file(
            atlas_business_ids
        )
    )


    # ========================================================
    # REVIEW.JSON
    # ========================================================

    review_stats = (
        inspect_reviews(

            atlas_business_ids,

            raw_business_ids

        )
    )


    # ========================================================
    # TIP.JSON
    # ========================================================

    inspect_tips(
        atlas_business_ids
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    section(
        "VALIDATION SUMMARY"
    )


    print(
        f"Atlas restaurants:              "
        f"{len(atlas_business_ids):,}"
    )


    print(
        f"Yelp reviews scanned:           "
        f"{review_stats['total_reviews']:,}"
    )


    print(
        f"Reviews for atlas restaurants:  "
        f"{review_stats['atlas_reviews']:,}"
    )


    print(
        f"Restaurants with reviews:       "
        f"{review_stats['restaurants_with_reviews']:,}"
    )


    if len(
        atlas_business_ids
    ) > 0:

        coverage = (

            review_stats[
                "restaurants_with_reviews"
            ]

            /

            len(
                atlas_business_ids
            )

            *

            100

        )


        print(
            f"Restaurant review coverage:     "
            f"{coverage:.2f}%"
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