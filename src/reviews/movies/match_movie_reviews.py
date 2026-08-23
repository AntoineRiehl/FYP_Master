#src/reviews/movies/match_movie_reviews.py

# ============================================================
# IMDb -> MovieLens Review Matching
#
# Purpose:
#   1. Read IMDb review JSON files
#   2. Match each IMDb review to a MovieLens movieId
#   3. Use conservative title + year matching
#   4. Reject ambiguous / uncertain matches
#   5. Process large JSON files without loading everything
#      into memory
#   6. Save checkpoints so the script can safely resume
#
# Input:
#
#   data/raw/reviews/movies/
#       part-01.json
#       part-02.json
#       ...
#       part-06.json
#
#   data/raw/movies/
#       movies.csv
#       links.csv
#
# Output:
#
#   data/processed/reviews/
#       movies_reviews_matched.csv
#       movie_review_matching_report.csv
#
# ============================================================


import json
import re
import csv
from pathlib import Path
from collections import defaultdict


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
    / "raw"
    / "reviews"
    / "movies"
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
    / "movies_reviews_matched.csv"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "movie_review_matching_report.csv"
)

CHECKPOINT_FILE = (
    OUTPUT_DIR
    / "movie_review_matching_checkpoint.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

REVIEW_FILES = [
    REVIEWS_DIR / "part-01.json",
    REVIEWS_DIR / "part-02.json",
    REVIEWS_DIR / "part-03.json",
    REVIEWS_DIR / "part-04.json",
    REVIEWS_DIR / "part-05.json",
    REVIEWS_DIR / "part-06.json",
]


# ------------------------------------------------------------
# Matching policy
# ------------------------------------------------------------

# Exact normalized title + exact year is the primary
# and preferred matching method.
ACCEPT_EXACT_TITLE_YEAR = True

# A title-only match is allowed ONLY when the IMDb review
# does not provide a year AND MovieLens contains exactly
# one movie with that normalized title.
#
# If IMDb provides a year, we NEVER ignore that year.
ACCEPT_UNIQUE_EXACT_TITLE_WITHOUT_YEAR = True

# We deliberately do NOT perform fuzzy matching automatically.
#
# A fuzzy match could silently attach reviews to the wrong movie.
#
# Fuzzy matching can be investigated separately later if needed.
USE_FUZZY_MATCHING = False


# ============================================================
# OUTPUT COLUMNS
# ============================================================

OUTPUT_COLUMNS = [

    "movieId",

    "review_id",

    "reviewer",

    "movie",

    "rating",

    "review_summary",

    "review_detail",

    "review_text",

    "review_date",

    "spoiler_tag",

    "helpful",

    "source",

    "match_method",

    "match_score",

    "match_status",

    "match_title",

    "match_year"

]


# ============================================================
# CHECKPOINT STRUCTURE
# ============================================================

DEFAULT_CHECKPOINT = {

    "file_index": 0,

    "review_index": 0,

    "total_processed": 0,

    "accepted": 0,

    "ambiguous": 0,

    "unmatched": 0,

    "invalid": 0

}


# ============================================================
# CHECKPOINT HELPERS
# ============================================================


def load_checkpoint():

    if not CHECKPOINT_FILE.exists():

        return DEFAULT_CHECKPOINT.copy()

    try:

        with open(
            CHECKPOINT_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            checkpoint = json.load(f)

        print(
            "\nCheckpoint found."
        )

        print(
            f"Resuming from file index "
            f"{checkpoint.get('file_index', 0)}"
        )

        print(
            f"Review index: "
            f"{checkpoint.get('review_index', 0):,}"
        )

        return checkpoint

    except Exception as e:

        print(
            f"\nCould not load checkpoint: {e}"
        )

        print(
            "Starting from the beginning."
        )

        return DEFAULT_CHECKPOINT.copy()


def save_checkpoint(checkpoint):

    temporary_path = (
        str(CHECKPOINT_FILE)
        + ".tmp"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            checkpoint,
            f,
            indent=2
        )

    # Atomic replacement
    Path(temporary_path).replace(
        CHECKPOINT_FILE
    )


# ============================================================
# TITLE NORMALISATION
# ============================================================


def normalize_title(title):
    """
    Normalize a title for conservative exact matching.

    Operations:
        - convert to string
        - lowercase
        - normalize unicode-like punctuation
        - remove IMDb-style disambiguation markers
        - remove punctuation
        - collapse whitespace

    We intentionally do NOT aggressively remove words.
    """

    if title is None:

        return ""

    title = str(title).strip()

    if not title:

        return ""

    # --------------------------------------------------------
    # Lowercase
    # --------------------------------------------------------

    title = title.lower()

    # --------------------------------------------------------
    # Normalize common punctuation
    # --------------------------------------------------------

    replacements = {

        "–": "-",
        "—": "-",
        "−": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "&": "and"

    }

    for old, new in replacements.items():

        title = title.replace(
            old,
            new
        )

    # --------------------------------------------------------
    # Remove IMDb-style disambiguation markers
    #
    # Examples:
    #
    #   The Half of It (I) (2020)
    #   Movie Title (II) (1998)
    #
    # The marker is usually a Roman numeral.
    # --------------------------------------------------------

    title = re.sub(

        r"\s+\([ivxlcdm]+\)\s*$",

        "",

        title,

        flags=re.IGNORECASE

    )

    # --------------------------------------------------------
    # Remove punctuation
    # --------------------------------------------------------

    title = re.sub(
        r"[^a-z0-9\s]",
        " ",
        title
    )

    # --------------------------------------------------------
    # Collapse whitespace
    # --------------------------------------------------------

    title = re.sub(
        r"\s+",
        " ",
        title
    ).strip()

    return title


# ============================================================
# YEAR EXTRACTION
# ============================================================


def extract_year_from_movielens_title(title):
    """
    Extract the first four-digit year from a MovieLens title.

    Example:

        "Toy Story (1995)"
        -> 1995
    """

    if title is None:

        return None

    match = re.search(
        r"\((\d{4})\)",
        str(title)
    )

    if match:

        return int(
            match.group(1)
        )

    return None


def extract_year_from_imdb_title(title):
    """
    Extract the first four-digit year from an IMDb title.

    Handles examples such as:

        After Life (2019– )
        This Is Us (2016– )
        The Half of It (I) (2020)
        The Droving (2020)

    Returns the first four-digit year.
    """

    if title is None:

        return None

    matches = re.findall(
        r"(\d{4})",
        str(title)
    )

    if matches:

        return int(
            matches[0]
        )

    return None


# ============================================================
# CLEAN IMDb TITLE
# ============================================================


def clean_imdb_title(title):
    """
    Remove the year and trailing IMDb disambiguation marker
    from an IMDb title before normalization.

    Examples:

        "The Half of It (I) (2020)"
            -> "The Half of It"

        "After Life (2019– )"
            -> "After Life"

        "All About Eve (1950)"
            -> "All About Eve"
    """

    if title is None:

        return ""

    title = str(title).strip()

    # --------------------------------------------------------
    # Remove year / year range at the end
    # --------------------------------------------------------

    title = re.sub(

        r"\s*\(\d{4}(?:\s*[-–—]\s*\d{0,4})?\)\s*$",

        "",

        title

    )

    # --------------------------------------------------------
    # Remove IMDb disambiguation marker
    #
    # e.g. "(I)", "(II)", "(III)"
    # --------------------------------------------------------

    title = re.sub(

        r"\s+\([ivxlcdm]+\)\s*$",

        "",

        title,

        flags=re.IGNORECASE

    )

    return title.strip()


# ============================================================
# MOVIELENS TITLE INDEX
# ============================================================


def build_movielens_index():

    print(
        "\n=============================================="
    )

    print(
        "LOADING MOVIELENS MOVIES"
    )

    print(
        "==============================================\n"
    )

    movies_path = (
        MOVIES_DIR
        / "movies.csv"
    )

    if not movies_path.exists():

        raise FileNotFoundError(
            f"MovieLens movies.csv not found:\n"
            f"{movies_path}"
        )

    # --------------------------------------------------------
    # Read CSV manually.
    #
    # This avoids needing pandas for this standalone matching
    # script and keeps memory usage relatively low.
    # --------------------------------------------------------

    title_year_index = defaultdict(list)

    title_index = defaultdict(list)

    total_movies = 0

    with open(
        movies_path,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            movie_id = row.get(
                "movieId"
            )

            title = row.get(
                "title"
            )

            if not movie_id or not title:

                continue

            total_movies += 1

            clean_title = clean_imdb_title(
                title
            )

            normalized_title = normalize_title(
                clean_title
            )

            year = extract_year_from_movielens_title(
                title
            )

            if not normalized_title:

                continue

            candidate = {

                "movieId": movie_id,

                "title": title,

                "year": year

            }

            title_index[
                normalized_title
            ].append(
                candidate
            )

            if year is not None:

                title_year_index[
                    (
                        normalized_title,
                        year
                    )
                ].append(
                    candidate
                )

    print(
        f"MovieLens movies loaded: "
        f"{total_movies:,}"
    )

    print(
        f"Unique normalized titles: "
        f"{len(title_index):,}"
    )

    print(
        f"Unique title/year combinations: "
        f"{len(title_year_index):,}"
    )

    return (
        title_index,
        title_year_index
    )


# ============================================================
# MATCH SINGLE REVIEW
# ============================================================


def match_review(
    review,
    title_index,
    title_year_index
):
    """
    Attempt to match one IMDb review to a MovieLens movie.

    Conservative matching hierarchy:

        1. Exact normalized title + exact year
        2. If IMDb has NO year:
              accept a unique exact title
        3. Otherwise reject

    Important rule:

        If IMDb provides a year, that year is NEVER ignored.

        Therefore:

            IMDb: "The Master (2012)"
            MovieLens: "The Master (1980)"

        -> rejected

        Even if "The Master" is otherwise unique in MovieLens.

    This prevents reviews from being silently attached to
    the wrong movie when multiple movies share the same title
    across different years.

    Returns a result dictionary.
    """

    imdb_title = review.get(
        "movie"
    )

    # --------------------------------------------------------
    # Validate title
    # --------------------------------------------------------

    if not imdb_title:

        return {

            "match_status": "invalid",

            "match_method": "missing_title",

            "match_score": 0.0,

            "candidate": None

        }

    # --------------------------------------------------------
    # Extract year from IMDb title
    # --------------------------------------------------------

    imdb_year = extract_year_from_imdb_title(
        imdb_title
    )

    # --------------------------------------------------------
    # Clean and normalize title
    # --------------------------------------------------------

    clean_title = clean_imdb_title(
        imdb_title
    )

    normalized_title = normalize_title(
        clean_title
    )

    if not normalized_title:

        return {

            "match_status": "invalid",

            "match_method": "empty_normalized_title",

            "match_score": 0.0,

            "candidate": None

        }

    # ========================================================
    # RULE 1
    # EXACT TITLE + EXACT YEAR
    # ========================================================

    if (
        ACCEPT_EXACT_TITLE_YEAR
        and imdb_year is not None
    ):

        candidates = title_year_index.get(

            (
                normalized_title,
                imdb_year
            ),

            []

        )

        # ----------------------------------------------------
        # Exactly one title/year candidate
        # ----------------------------------------------------

        if len(candidates) == 1:

            return {

                "match_status": "accepted",

                "match_method":
                    "exact_title_year",

                "match_score": 1.0,

                "candidate":
                    candidates[0]

            }

        # ----------------------------------------------------
        # Multiple candidates with same title AND year
        #
        # Extremely unusual, but never guess.
        # ----------------------------------------------------

        if len(candidates) > 1:

            return {

                "match_status": "ambiguous",

                "match_method":
                    "exact_title_year_multiple_candidates",

                "match_score": 1.0,

                "candidate": None

            }

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # IMDb supplied a year, but there is no MovieLens
        # movie with the same title AND year.
        #
        # We DO NOT fall back to title-only matching.
        #
        # This prevents:
        #
        #   IMDb:      The Master (2012)
        #   MovieLens: The Master (1980)
        #
        # from being incorrectly accepted.
        # ----------------------------------------------------

        title_candidates = title_index.get(

            normalized_title,

            []

        )

        if len(title_candidates) > 0:

            return {

                "match_status": "unmatched",

                "match_method":
                    "title_found_but_year_mismatch",

                "match_score": 0.0,

                "candidate": None

            }

        return {

            "match_status": "unmatched",

            "match_method":
                "no_exact_title_year_match",

            "match_score": 0.0,

            "candidate": None

        }

    # ========================================================
    # RULE 2
    # EXACT TITLE WITHOUT YEAR
    # ========================================================

    # This rule is ONLY reached when IMDb does not provide
    # a year.
    #
    # If there is no year, a unique title is sufficiently
    # deterministic for our purposes.

    candidates = title_index.get(

        normalized_title,

        []

    )

    # --------------------------------------------------------
    # Unique title
    # --------------------------------------------------------

    if (
        ACCEPT_UNIQUE_EXACT_TITLE_WITHOUT_YEAR
        and len(candidates) == 1
    ):

        return {

            "match_status": "accepted",

            "match_method":
                "unique_exact_title_without_year",

            "match_score": 0.99,

            "candidate":
                candidates[0]

        }

    # --------------------------------------------------------
    # Multiple movies share the same title
    # --------------------------------------------------------

    if len(candidates) > 1:

        return {

            "match_status": "ambiguous",

            "match_method":
                "exact_title_multiple_candidates_without_year",

            "match_score": 0.0,

            "candidate": None

        }

    # --------------------------------------------------------
    # No candidate
    # --------------------------------------------------------

    return {

        "match_status": "unmatched",

        "match_method":
            "no_exact_title_match",

        "match_score": 0.0,

        "candidate": None

    }

# ============================================================
# REVIEW TEXT
# ============================================================


def create_review_text(review):
    """
    Combine review summary and detailed review text.

    The original fields remain available separately.
    """

    summary = review.get(
        "review_summary"
    )

    detail = review.get(
        "review_detail"
    )

    parts = []

    if summary:

        parts.append(
            str(summary).strip()
        )

    if detail:

        parts.append(
            str(detail).strip()
        )

    return "\n\n".join(
        part
        for part in parts
        if part
    )


# ============================================================
# JSON REVIEW ITERATOR
# ============================================================


def iterate_reviews(path):
    """
    Iterate through a JSON file without intentionally keeping
    all reviews in memory.

    The IMDb dataset is expected to contain a JSON array:

        [
            {...},
            {...},
            ...
        ]

    Python's standard json module does not provide true
    streaming JSON-array parsing, so we use a small incremental
    parser based on JSONDecoder.

    This allows us to process one review at a time.
    """

    decoder = json.JSONDecoder()

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        buffer = ""

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:

                break

            buffer += chunk

            while True:

                buffer = buffer.lstrip()

                # ------------------------------------------------
                # Opening JSON array
                # ------------------------------------------------

                if buffer.startswith("["):

                    buffer = buffer[1:]

                    continue

                # ------------------------------------------------
                # End of JSON array
                # ------------------------------------------------

                if buffer.startswith("]"):

                    return

                # ------------------------------------------------
                # Remove commas between objects
                # ------------------------------------------------

                if buffer.startswith(","):

                    buffer = buffer[1:]

                    continue

                # ------------------------------------------------
                # Try to decode one JSON object
                # ------------------------------------------------

                try:

                    obj, index = (
                        decoder.raw_decode(
                            buffer
                        )
                    )

                except json.JSONDecodeError:

                    # We need more data.
                    break

                buffer = buffer[index:]

                yield obj


# ============================================================
# CSV WRITER
# ============================================================


def open_output_writer(
    append=False
):

    file_exists = (
        OUTPUT_REVIEWS.exists()
        and OUTPUT_REVIEWS.stat().st_size > 0
    )

    mode = "a" if append else "w"

    f = open(
        OUTPUT_REVIEWS,
        mode,
        encoding="utf-8",
        newline=""
    )

    writer = csv.DictWriter(
        f,
        fieldnames=OUTPUT_COLUMNS
    )

    if not file_exists or not append:

        writer.writeheader()

    return f, writer


# ============================================================
# PROCESS REVIEWS
# ============================================================


def process_reviews(
    title_index,
    title_year_index
):

    checkpoint = load_checkpoint()

    start_file_index = checkpoint[
        "file_index"
    ]

    start_review_index = checkpoint[
        "review_index"
    ]

    total_processed = checkpoint[
        "total_processed"
    ]

    accepted = checkpoint[
        "accepted"
    ]

    ambiguous = checkpoint[
        "ambiguous"
    ]

    unmatched = checkpoint[
        "unmatched"
    ]

    invalid = checkpoint[
        "invalid"
    ]

    # --------------------------------------------------------
    # Determine whether output already contains data.
    # --------------------------------------------------------

    append_output = (
        OUTPUT_REVIEWS.exists()
        and OUTPUT_REVIEWS.stat().st_size > 0
    )

    output_file, writer = open_output_writer(
        append=append_output
    )

    try:

        for file_index in range(
            start_file_index,
            len(REVIEW_FILES)
        ):

            review_file = REVIEW_FILES[
                file_index
            ]

            if not review_file.exists():

                print(
                    f"\nWARNING: Review file not found:"
                )

                print(
                    review_file
                )

                continue

            print(
                "\n=============================================="
            )

            print(
                f"PROCESSING FILE "
                f"{file_index + 1}/{len(REVIEW_FILES)}"
            )

            print(
                review_file.name
            )

            print(
                "==============================================\n"
            )

            # ------------------------------------------------
            # Resume within current file
            # ------------------------------------------------

            skip_until = (
                start_review_index
                if file_index
                == start_file_index
                else 0
            )

            current_index = 0

            for review in iterate_reviews(
                review_file
            ):

                # ------------------------------------------------
                # Resume
                # ------------------------------------------------

                if current_index < skip_until:

                    current_index += 1

                    continue

                # ------------------------------------------------
                # Match
                # ------------------------------------------------

                result = match_review(

                    review,

                    title_index,

                    title_year_index

                )

                status = result[
                    "match_status"
                ]

                # ------------------------------------------------
                # Statistics
                # ------------------------------------------------

                total_processed += 1

                if status == "accepted":

                    accepted += 1

                elif status == "ambiguous":

                    ambiguous += 1

                elif status == "unmatched":

                    unmatched += 1

                elif status == "invalid":

                    invalid += 1

                # ------------------------------------------------
                # Only write accepted matches
                # ------------------------------------------------

                if status == "accepted":

                    candidate = result[
                        "candidate"
                    ]

                    review_text = (
                        create_review_text(
                            review
                        )
                    )

                    output_row = {

                        "movieId":
                            candidate[
                                "movieId"
                            ],

                        "review_id":
                            review.get(
                                "review_id"
                            ),

                        "reviewer":
                            review.get(
                                "reviewer"
                            ),

                        "movie":
                            review.get(
                                "movie"
                            ),

                        "rating":
                            review.get(
                                "rating"
                            ),

                        "review_summary":
                            review.get(
                                "review_summary"
                            ),

                        "review_detail":
                            review.get(
                                "review_detail"
                            ),

                        "review_text":
                            review_text,

                        "review_date":
                            review.get(
                                "review_date"
                            ),

                        "spoiler_tag":
                            review.get(
                                "spoiler_tag"
                            ),

                        "helpful":
                            json.dumps(
                                review.get(
                                    "helpful"
                                )
                            ),

                        "source":
                            "IMDb",

                        "match_method":
                            result[
                                "match_method"
                            ],

                        "match_score":
                            result[
                                "match_score"
                            ],

                        "match_status":
                            status,

                        "match_title":
                            candidate[
                                "title"
                            ],

                        "match_year":
                            candidate[
                                "year"
                            ]

                    }

                    writer.writerow(
                        output_row
                    )

                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                current_index += 1

                if (
                    current_index % 10000
                    == 0
                ):

                    print(

                        f"\r"
                        f"Reviews processed: "
                        f"{current_index:,} | "
                        f"Accepted: "
                        f"{accepted:,} | "
                        f"Ambiguous: "
                        f"{ambiguous:,} | "
                        f"Unmatched: "
                        f"{unmatched:,}",

                        end=""

                    )

                # ------------------------------------------------
                # Checkpoint every 10,000 reviews
                # ------------------------------------------------

                if (
                    current_index % 10000
                    == 0
                ):

                    output_file.flush()

                    save_checkpoint({

                        "file_index":
                            file_index,

                        "review_index":
                            current_index,

                        "total_processed":
                            total_processed,

                        "accepted":
                            accepted,

                        "ambiguous":
                            ambiguous,

                        "unmatched":
                            unmatched,

                        "invalid":
                            invalid

                    })

            print(
                "\n\nFile complete."
            )

            print(
                f"Reviews processed: "
                f"{current_index:,}"
            )

            print(
                f"Accepted: "
                f"{accepted:,}"
            )

            print(
                f"Ambiguous: "
                f"{ambiguous:,}"
            )

            print(
                f"Unmatched: "
                f"{unmatched:,}"
            )

            # ----------------------------------------------------
            # Move checkpoint to next file
            # ----------------------------------------------------

            save_checkpoint({

                "file_index":
                    file_index + 1,

                "review_index":
                    0,

                "total_processed":
                    total_processed,

                "accepted":
                    accepted,

                "ambiguous":
                    ambiguous,

                "unmatched":
                    unmatched,

                "invalid":
                    invalid

            })

    finally:

        output_file.flush()

        output_file.close()

    return {

        "total_processed":
            total_processed,

        "accepted":
            accepted,

        "ambiguous":
            ambiguous,

        "unmatched":
            unmatched,

        "invalid":
            invalid

    }


# ============================================================
# MATCHING REPORT
# ============================================================


def create_report(
    statistics
):

    print(
        "\n=============================================="
    )

    print(
        "CREATING MATCHING REPORT"
    )

    print(
        "==============================================\n"
    )

    total = statistics[
        "total_processed"
    ]

    accepted = statistics[
        "accepted"
    ]

    ambiguous = statistics[
        "ambiguous"
    ]

    unmatched = statistics[
        "unmatched"
    ]

    invalid = statistics[
        "invalid"
    ]

    if total > 0:

        accepted_rate = (
            accepted / total * 100
        )

        ambiguous_rate = (
            ambiguous / total * 100
        )

        unmatched_rate = (
            unmatched / total * 100
        )

    else:

        accepted_rate = 0
        ambiguous_rate = 0
        unmatched_rate = 0

    rows = [

        {
            "metric":
                "Total reviews processed",

            "value":
                total
        },

        {
            "metric":
                "Accepted matches",

            "value":
                accepted
        },

        {
            "metric":
                "Ambiguous matches",

            "value":
                ambiguous
        },

        {
            "metric":
                "Unmatched reviews",

            "value":
                unmatched
        },

        {
            "metric":
                "Invalid reviews",

            "value":
                invalid
        },

        {
            "metric":
                "Accepted match rate (%)",

            "value":
                round(
                    accepted_rate,
                    3
                )
        },

        {
            "metric":
                "Ambiguous rate (%)",

            "value":
                round(
                    ambiguous_rate,
                    3
                )
        },

        {
            "metric":
                "Unmatched rate (%)",

            "value":
                round(
                    unmatched_rate,
                    3
                )
        }

    ]

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=[
                "metric",
                "value"
            ]

        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        f"Report saved to:\n"
        f"{OUTPUT_REPORT}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================


def print_summary(
    statistics
):

    print(
        "\n\n=============================================="
    )

    print(
        "MOVIE REVIEW MATCHING COMPLETE"
    )

    print(
        "==============================================\n"
    )

    print(
        f"Reviews processed:     "
        f"{statistics['total_processed']:,}"
    )

    print(
        f"Accepted matches:      "
        f"{statistics['accepted']:,}"
    )

    print(
        f"Ambiguous matches:     "
        f"{statistics['ambiguous']:,}"
    )

    print(
        f"Unmatched reviews:     "
        f"{statistics['unmatched']:,}"
    )

    print(
        f"Invalid reviews:       "
        f"{statistics['invalid']:,}"
    )

    total = statistics[
        "total_processed"
    ]

    if total > 0:

        print(
            f"\nAccepted match rate:   "
            f"{statistics['accepted'] / total * 100:.2f}%"
        )

    print(
        "\nOutput:"
    )

    print(
        f"  Reviews:"
    )

    print(
        f"    {OUTPUT_REVIEWS}"
    )

    print(
        f"\n  Report:"
    )

    print(
        f"    {OUTPUT_REPORT}"
    )

    print(
        f"\n  Checkpoint:"
    )

    print(
        f"    {CHECKPOINT_FILE}"
    )

    print(
        "\nDone."
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print(
        "\n=================================================="
    )

    print(
        "IMDb -> MovieLens REVIEW MATCHING"
    )

    print(
        "=================================================="
    )

    print(
        "\nMatching policy:"
    )

    print(
        "  1. Exact normalized title + exact year"
    )

    print(
        "  2. If no year is available: unique exact title"
    )

    print(
        "  3. Year mismatches are rejected"
    )

    print(
        "  4. Ambiguous matches are rejected"
    )

    print(
        "  5. No fuzzy matching"
    )

    print(
        "\nReview files:"
    )

    for path in REVIEW_FILES:

        print(
            f"  - {path}"
        )

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    missing_files = [

        path
        for path in REVIEW_FILES
        if not path.exists()

    ]

    if missing_files:

        print(
            "\nWARNING:"
        )

        print(
            "The following review files were not found:"
        )

        for path in missing_files:

            print(
                f"  - {path}"
            )

        print(
            "\nThe script will process the files that exist."
        )

    # --------------------------------------------------------
    # Build MovieLens index
    # --------------------------------------------------------

    (
        title_index,
        title_year_index
    ) = build_movielens_index()

    # --------------------------------------------------------
    # Process reviews
    # --------------------------------------------------------

    statistics = process_reviews(

        title_index,

        title_year_index

    )

    # --------------------------------------------------------
    # Create report
    # --------------------------------------------------------

    create_report(
        statistics
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_summary(
        statistics
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()