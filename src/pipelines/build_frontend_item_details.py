# src/pipelines/build_frontend_item_details.py

import argparse
import json
import shutil

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# =========================================================
# PROJECT PATHS
# =========================================================

FEEL_SCORES_PATH = Path(
    "data/processed/feel_space/"
    "movies_music_restaurants/"
    "feel_space_scores.csv"
)


REVIEW_FILES = {

    "movies": {
        "path": Path(
            "data/processed/reviews/"
            "movies_reviews_prepared.csv"
        ),
        "id_column": "movieId",
    },

    "music": {
        "path": Path(
            "data/processed/reviews/"
            "music_reviews_prepared.csv"
        ),
        "id_column": "artist_mbid",
    },

    "restaurants": {
        "path": Path(
            "data/processed/reviews/"
            "restaurant_reviews_prepared.csv"
        ),
        "id_column": "business_id",
    },

}


OUTPUT_ROOT = Path(
    "frontend/public/data/item_details"
)


# =========================================================
# FEEL DIMENSIONS
# =========================================================

FEEL_DIMENSIONS = [

    "valence",
    "activation",
    "potency",
    "tension",
    "warmth",
    "scale",
    "tone",
    "familiarity",
    "refinement",
    "complexity",
    "nostalgia",
    "wonder",
    "tenderness",

]


# =========================================================
# DEFAULT CONFIGURATION
# =========================================================

DEFAULT_SHARD_COUNT = 256

DEFAULT_REVIEW_SAMPLE_SIZE = 5

DEFAULT_REVIEW_CHUNK_SIZE = 100_000


# =========================================================
# BASIC VALUE HELPERS
# =========================================================

def clean_string(value):
    """
    Convert a scalar into a clean string.

    Missing values become None.
    """

    if value is None:

        return None


    try:

        if pd.isna(value):

            return None

    except TypeError:

        pass


    value = str(value).strip()


    if not value:

        return None


    return value


# =========================================================
# SOURCE ID
# =========================================================

def normalize_source_id(value):
    """
    Standardise entity IDs for JSON lookup.

    IDs remain strings because:

        Movies:
            numeric MovieLens IDs

        Music:
            MusicBrainz UUIDs

        Restaurants:
            Yelp business IDs

    Using strings gives us one consistent frontend
    lookup contract across all domains.
    """

    value = clean_string(
        value
    )


    if value is None:

        return None


    return value


# =========================================================
# BOOLEAN
# =========================================================

def clean_bool(value):
    """
    Safely interpret booleans that may have been
    written to CSV as bools, integers or strings.
    """

    if value is None:

        return False


    try:

        if pd.isna(value):

            return False

    except TypeError:

        pass


    if isinstance(
        value,
        (bool, np.bool_)
    ):

        return bool(
            value
        )


    if isinstance(
        value,
        (int, np.integer)
    ):

        return value != 0


    if isinstance(
        value,
        (float, np.floating)
    ):

        return value != 0


    text = (
        str(value)
        .strip()
        .lower()
    )


    return text in {

        "true",
        "1",
        "yes",
        "y",
        "t",

    }


# =========================================================
# INTEGER
# =========================================================

def clean_int(value):
    """
    Convert a scalar to int where possible.
    """

    if value is None:

        return None


    try:

        if pd.isna(value):

            return None

    except TypeError:

        pass


    try:

        return int(
            float(value)
        )

    except (
        TypeError,
        ValueError
    ):

        return None


# =========================================================
# FLOAT
# =========================================================

def clean_float(value):
    """
    Convert to a finite Python float.

    Non-finite values become None.
    """

    if value is None:

        return None


    try:

        if pd.isna(value):

            return None

    except TypeError:

        pass


    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return None


    if not np.isfinite(
        result
    ):

        return None


    return result


# =========================================================
# SHARD HASH
# =========================================================

def fnv1a_32(
    value: str
):
    """
    Stable 32-bit FNV-1a hash.

    We deliberately avoid Python's built-in hash()
    because it is randomised between processes.

    The React loader can reproduce this exact algorithm
    using JavaScript integer arithmetic.
    """

    hash_value = 2166136261


    for character in value:

        hash_value ^= ord(
            character
        )

        hash_value = (
            hash_value
            *
            16777619
        ) & 0xFFFFFFFF


    return hash_value


# =========================================================
# SHARD NUMBER
# =========================================================

def get_shard_index(
    source_id: str,
    shard_count: int,
):
    """
    Determine which static JSON shard stores an entity.
    """

    return (
        fnv1a_32(
            source_id
        )
        %
        shard_count
    )


# =========================================================
# SHARD FILE NAME
# =========================================================

def get_shard_filename(
    shard_index: int
):
    return (
        f"shard_"
        f"{shard_index:03d}"
        f".json"
    )


# =========================================================
# WRITE JSON
# =========================================================

def write_json(
    path: Path,
    payload,
):
    """
    Write compact standard JSON.

    Compact output matters because these files are loaded
    by the browser.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        path,
        "w",
        encoding="utf8"
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            allow_nan=False,
            separators=(
                ",",
                ":"
            )
        )


# =========================================================
# VALIDATE INPUT FILES
# =========================================================

def validate_input_files():
    """
    Fail early with useful paths if an expected source
    file is missing.
    """

    paths = [

        FEEL_SCORES_PATH,

        *[
            config["path"]
            for config
            in REVIEW_FILES.values()
        ]

    ]


    missing = [

        path
        for path
        in paths
        if not path.exists()

    ]


    if missing:

        formatted = "\n".join(

            f"  - {path}"
            for path
            in missing

        )


        raise FileNotFoundError(

            "Missing required frontend-detail "
            "source files:\n"
            f"{formatted}"

        )


# =========================================================
# RESET OUTPUT
# =========================================================

def reset_output_directory():
    """
    Remove stale shards from previous runs.

    This pipeline owns only:

        frontend/public/data/item_details/
    """

    if OUTPUT_ROOT.exists():

        shutil.rmtree(
            OUTPUT_ROOT
        )


    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# LOAD FEEL SCORES
# =========================================================

def load_feel_scores():
    """
    Load the frozen shared experiential-space scores.
    """

    print(
        "\nLoading Feel-space scores..."
    )


    df = pd.read_csv(

        FEEL_SCORES_PATH,

        dtype={
            "domain": "string",
            "source_id": "string",
        }

    )


    required_columns = [

        "domain",
        "source_id",
        "is_semantically_defined",
        "has_base_semantics",
        "has_review_semantics",
        "reviews_used_for_embedding",
        "semantic_source",

        *FEEL_DIMENSIONS,

    ]


    missing_columns = [

        column
        for column
        in required_columns
        if column not in df.columns

    ]


    if missing_columns:

        raise ValueError(

            "feel_space_scores.csv is missing "
            "required columns:\n"
            f"{missing_columns}"

        )


    # -----------------------------------------------------
    # NORMALISE DOMAIN
    # -----------------------------------------------------

    df["domain"] = (

        df["domain"]
        .astype("string")
        .str.strip()
        .str.lower()

    )


    # -----------------------------------------------------
    # NORMALISE SOURCE ID
    # -----------------------------------------------------

    df["source_id"] = (

        df["source_id"]
        .astype("string")
        .str.strip()

    )


    # -----------------------------------------------------
    # VALIDATE DOMAINS
    # -----------------------------------------------------

    valid_domains = set(
        REVIEW_FILES.keys()
    )


    unexpected_domains = (

        set(
            df["domain"]
            .dropna()
            .unique()
            .tolist()
        )

        -

        valid_domains

    )


    if unexpected_domains:

        raise ValueError(

            "Unexpected domains in Feel-space file: "
            f"{sorted(unexpected_domains)}"

        )


    # -----------------------------------------------------
    # NUMERIC FEEL VALUES
    # -----------------------------------------------------

    for dimension in FEEL_DIMENSIONS:

        df[dimension] = pd.to_numeric(

            df[dimension],

            errors="coerce"

        )


    # -----------------------------------------------------
    # DEFINED FLAG
    # -----------------------------------------------------

    df["_feel_defined"] = (

        df[
            "is_semantically_defined"
        ]
        .map(
            clean_bool
        )

    )


    # -----------------------------------------------------
    # ENSURE DEFINED ITEMS REALLY HAVE 13 VALUES
    # -----------------------------------------------------

    defined_df = df[
        df["_feel_defined"]
    ]


    missing_defined_values = (

        defined_df[
            FEEL_DIMENSIONS
        ]
        .isna()
        .any(
            axis=1
        )

    )


    if missing_defined_values.any():

        bad_count = int(

            missing_defined_values
            .sum()

        )


        raise ValueError(

            f"{bad_count:,} entities are marked "
            "semantically defined but have missing "
            "Feel-dimension values."

        )


    print(
        f"Loaded {len(df):,} entities."
    )

    print(
        "Semantically defined: "
        f"{len(defined_df):,}"
    )

    print(
        "Semantically undefined: "
        f"{len(df) - len(defined_df):,}"
    )


    return df


# =========================================================
# VALID SOURCE IDS BY DOMAIN
# =========================================================

def build_valid_source_ids(
    feel_df: pd.DataFrame
):
    """
    Build the entity universe represented in the atlas.

    Reviews for entities outside this universe are ignored.
    """

    result = {}


    for domain in REVIEW_FILES:

        domain_ids = (

            feel_df.loc[
                feel_df["domain"] == domain,
                "source_id"
            ]
            .dropna()
            .astype(str)
            .tolist()

        )


        result[domain] = set(
            domain_ids
        )


        print(

            f"{domain}: "
            f"{len(result[domain]):,} "
            "valid atlas entities"

        )


    return result


# =========================================================
# FIT GLOBAL FEEL SCALER
# =========================================================

def standardize_feel_scores(
    feel_df: pd.DataFrame
):
    """
    Reproduce the frozen Method-B scaling:

        ONE StandardScaler
        across all semantically defined
        Movies + Music + Restaurants.

    The original 13 dimensions are NOT altered.

    We simply compute the z-score representation needed
    by the frontend profile.
    """

    print(
        "\nReconstructing global Feel scaling..."
    )


    defined_mask = (

        feel_df[
            "_feel_defined"
        ]
        .to_numpy(
            dtype=bool
        )

    )


    raw_defined = (

        feel_df.loc[
            defined_mask,
            FEEL_DIMENSIONS
        ]
        .to_numpy(
            dtype=np.float64
        )

    )


    scaler = StandardScaler()


    scaled_defined = scaler.fit_transform(
        raw_defined
    )


    scaled_all = np.full(

        (
            len(feel_df),
            len(FEEL_DIMENSIONS)
        ),

        np.nan,

        dtype=np.float64

    )


    scaled_all[
        defined_mask,
        :
    ] = scaled_defined


    print(
        "Global Feel StandardScaler fitted."
    )


    return (
        scaled_all,
        scaler
    )


# =========================================================
# REVIEW RECORD
# =========================================================

def make_review_record(
    row
):
    """
    Keep only fields useful for frontend inspection.

    We intentionally do not expose all modelling /
    preprocessing columns.
    """

    return {

        "review_id":
            clean_string(
                row.get(
                    "review_id"
                )
            ),

        "text":
            clean_string(
                row.get(
                    "review_text"
                )
            )
            or
            "",

        "rating":
            clean_float(
                row.get(
                    "rating"
                )
            ),

        "date":
            clean_string(
                row.get(
                    "review_date"
                )
            ),

        "source":
            clean_string(
                row.get(
                    "source"
                )
            ),

    }


# =========================================================
# PROCESS DOMAIN REVIEWS
# =========================================================

def process_domain_reviews(
    domain: str,
    config: dict,
    valid_source_ids: set,
    shard_count: int,
    sample_size: int,
    chunk_size: int,
):
    """
    Count all available prepared reviews while keeping
    only a small deterministic sample per entity.

    Sampling strategy
    -----------------

    Each review receives a stable uint64 pandas hash
    based on:

        domain | source_id | review_id

    For each entity we keep the `sample_size` reviews
    with the smallest hash values.

    Advantages:

        - deterministic;
        - independent of source-file row order;
        - no random seed management;
        - only K review texts need to be retained
          per entity;
        - suitable for chunked processing.
    """

    path = config[
        "path"
    ]

    id_column = config[
        "id_column"
    ]


    print(
        "\n"
        "=================================================="
    )

    print(
        f"Processing reviews: {domain}"
    )

    print(
        f"Source: {path}"
    )

    print(
        "=================================================="
    )


    # -----------------------------------------------------
    # EXACT AVAILABLE REVIEW COUNTS
    # -----------------------------------------------------

    review_counts = Counter()


    # -----------------------------------------------------
    # SAMPLES ALREADY ORGANISED BY OUTPUT SHARD
    # -----------------------------------------------------

    samples_by_shard = [

        {}

        for _ in range(
            shard_count
        )

    ]


    # -----------------------------------------------------
    # READ ONLY REQUIRED COLUMNS
    # -----------------------------------------------------

    use_columns = [

        id_column,
        "review_id",
        "review_text",
        "rating",
        "review_date",
        "source",

    ]


    dtype = {

        id_column:
            "string",

        "review_id":
            "string",

    }


    reader = pd.read_csv(

        path,

        usecols=use_columns,

        dtype=dtype,

        chunksize=chunk_size

    )


    total_valid_reviews = 0


    # =====================================================
    # PROCESS CHUNKS
    # =====================================================

    for chunk_number, chunk in enumerate(
        reader,
        start=1
    ):


        # -------------------------------------------------
        # STANDARDISE SOURCE ID
        # -------------------------------------------------

        chunk[id_column] = (

            chunk[id_column]
            .astype("string")
            .str.strip()

        )


        # -------------------------------------------------
        # REVIEW ID
        # -------------------------------------------------

        chunk["review_id"] = (

            chunk["review_id"]
            .astype("string")
            .str.strip()

        )


        # -------------------------------------------------
        # REVIEW TEXT
        # -------------------------------------------------

        chunk["review_text"] = (

            chunk["review_text"]
            .fillna("")
            .astype(str)

        )


        # -------------------------------------------------
        # KEEP ATLAS ENTITIES ONLY
        # -------------------------------------------------

        chunk = chunk[

            chunk[id_column]
            .isin(
                valid_source_ids
            )

        ]


        # -------------------------------------------------
        # REQUIRE ACTUAL TEXT
        # -------------------------------------------------

        chunk = chunk[

            chunk["review_text"]
            .str.strip()
            .ne("")

        ]


        if chunk.empty:

            continue


        # -------------------------------------------------
        # EXACT REVIEW COUNTS
        # -------------------------------------------------

        counts = (

            chunk[id_column]
            .value_counts(
                sort=False
            )

        )


        for (
            source_id,
            count
        ) in counts.items():

            review_counts[
                str(source_id)
            ] += int(
                count
            )


        total_valid_reviews += len(
            chunk
        )


        # -------------------------------------------------
        # STABLE SAMPLING HASH
        # -------------------------------------------------

        sample_keys = (

            domain
            +
            "|"
            +
            chunk[id_column]
            .astype(str)
            +
            "|"
            +
            chunk["review_id"]
            .astype(str)

        )


        chunk["_sample_hash"] = (

            pd.util
            .hash_pandas_object(

                sample_keys,

                index=False

            )
            .astype(
                "uint64"
            )

        )


        # -------------------------------------------------
        # KEEP AT MOST K CANDIDATES PER ENTITY
        # FROM THIS CHUNK
        # -------------------------------------------------

        candidates = (

            chunk

            .sort_values(

                [
                    id_column,
                    "_sample_hash"
                ]

            )

            .groupby(

                id_column,

                sort=False

            )

            .head(
                sample_size
            )

        )


        # -------------------------------------------------
        # MERGE CHUNK CANDIDATES INTO GLOBAL TOP-K
        # -------------------------------------------------

        candidate_records = (

            candidates[
                [
                    id_column,
                    "review_id",
                    "review_text",
                    "rating",
                    "review_date",
                    "source",
                    "_sample_hash",
                ]
            ]
            .to_dict(
                orient="records"
            )

        )


        for row in candidate_records:

            source_id = (
                str(
                    row[
                        id_column
                    ]
                )
                .strip()
            )


            sample_hash = int(
                row[
                    "_sample_hash"
                ]
            )


            shard_index = (

                get_shard_index(

                    source_id,

                    shard_count

                )

            )


            shard = samples_by_shard[
                shard_index
            ]


            existing = shard.get(
                source_id
            )


            if existing is None:

                existing = []

                shard[
                    source_id
                ] = existing


            existing.append(

                (
                    sample_hash,
                    make_review_record(
                        row
                    )
                )

            )


            # ---------------------------------------------
            # KEEP GLOBAL SMALLEST K HASHES
            # ---------------------------------------------

            existing.sort(

                key=lambda item:
                    item[0]

            )


            if len(
                existing
            ) > sample_size:

                del existing[
                    sample_size:
                ]


        # -------------------------------------------------
        # PROGRESS
        # -------------------------------------------------

        if (
            chunk_number == 1
            or
            chunk_number % 10 == 0
        ):

            print(

                f"  chunks: {chunk_number:,} | "
                f"valid reviews: "
                f"{total_valid_reviews:,} | "
                f"entities with reviews: "
                f"{len(review_counts):,}"

            )


    # =====================================================
    # WRITE REVIEW SHARDS
    # =====================================================

    output_directory = (

        OUTPUT_ROOT
        /
        "reviews"
        /
        domain

    )


    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    sampled_entity_count = 0


    for shard_index in range(
        shard_count
    ):


        shard_samples = (

            samples_by_shard[
                shard_index
            ]

        )


        payload = {}


        for (
            source_id,
            samples
        ) in shard_samples.items():


            ordered_samples = sorted(

                samples,

                key=lambda item:
                    item[0]

            )


            reviews = [

                review

                for (
                    _,
                    review
                )
                in ordered_samples

            ]


            payload[
                source_id
            ] = {

                "available_count":
                    int(
                        review_counts.get(
                            source_id,
                            0
                        )
                    ),

                "sampled_count":
                    len(
                        reviews
                    ),

                "reviews":
                    reviews,

            }


        sampled_entity_count += len(
            payload
        )


        write_json(

            output_directory
            /
            get_shard_filename(
                shard_index
            ),

            payload

        )


        # Free one shard as soon as it is written.

        samples_by_shard[
            shard_index
        ].clear()


    print(
        f"\n{domain} review export complete."
    )

    print(
        f"Available prepared reviews: "
        f"{total_valid_reviews:,}"
    )

    print(
        f"Entities with reviews: "
        f"{len(review_counts):,}"
    )

    print(
        f"Entities with exported samples: "
        f"{sampled_entity_count:,}"
    )


    return dict(
        review_counts
    )


# =========================================================
# PROCESS ALL REVIEWS
# =========================================================

def process_all_reviews(
    valid_source_ids,
    shard_count: int,
    sample_size: int,
    chunk_size: int,
):
    """
    Export review shards domain by domain.

    Only compact review counts are retained after each
    domain has been written.
    """

    counts_by_domain = {}


    for (
        domain,
        config
    ) in REVIEW_FILES.items():

        counts_by_domain[
            domain
        ] = process_domain_reviews(

            domain=domain,

            config=config,

            valid_source_ids=
                valid_source_ids[
                    domain
                ],

            shard_count=
                shard_count,

            sample_size=
                sample_size,

            chunk_size=
                chunk_size,

        )


    return counts_by_domain


# =========================================================
# EXPORT PROFILE SHARDS
# =========================================================

def export_profile_shards(
    feel_df: pd.DataFrame,
    scaled_feel: np.ndarray,
    review_counts_by_domain,
    shard_count: int,
    review_sample_size: int,
):
    """
    Export lightweight, selection-time item profiles.

    Each profile contains:

        - Feel availability;
        - standardized 13D Feel values;
        - semantic-source information;
        - base/review semantic availability;
        - review counts.

    Raw review text is deliberately NOT stored here.
    """

    print(
        "\n"
        "=================================================="
    )

    print(
        "Exporting item profiles"
    )

    print(
        "=================================================="
    )


    # -----------------------------------------------------
    # PREPARE SHARDS PER DOMAIN
    # -----------------------------------------------------

    profile_shards = {

        domain: [

            {}

            for _ in range(
                shard_count
            )

        ]

        for domain
        in REVIEW_FILES

    }


    # -----------------------------------------------------
    # ARRAYS FOR FASTER ROW ACCESS
    # -----------------------------------------------------

    domains = (

        feel_df[
            "domain"
        ]
        .astype(str)
        .to_numpy()

    )


    source_ids = (

        feel_df[
            "source_id"
        ]
        .astype(str)
        .to_numpy()

    )


    feel_defined = (

        feel_df[
            "_feel_defined"
        ]
        .to_numpy(
            dtype=bool
        )

    )


    has_base_semantics = (

        feel_df[
            "has_base_semantics"
        ]
        .map(
            clean_bool
        )
        .to_numpy(
            dtype=bool
        )

    )


    has_review_semantics = (

        feel_df[
            "has_review_semantics"
        ]
        .map(
            clean_bool
        )
        .to_numpy(
            dtype=bool
        )

    )


    semantic_sources = (

        feel_df[
            "semantic_source"
        ]
        .to_numpy()

    )


    reviews_used = (

        feel_df[
            "reviews_used_for_embedding"
        ]
        .to_numpy()

    )


    # =====================================================
    # BUILD PROFILE RECORDS
    # =====================================================

    for index in range(
        len(feel_df)
    ):


        domain = domains[
            index
        ]


        source_id = source_ids[
            index
        ]


        if (
            domain not in profile_shards
            or
            not source_id
        ):

            continue


        # -------------------------------------------------
        # FEEL PROFILE
        # -------------------------------------------------

        if feel_defined[
            index
        ]:


            feel_profile = {

                dimension:
                    round(
                        float(
                            scaled_feel[
                                index,
                                dimension_index
                            ]
                        ),
                        5
                    )

                for (
                    dimension_index,
                    dimension
                )
                in enumerate(
                    FEEL_DIMENSIONS
                )

            }


        else:

            feel_profile = None


        # -------------------------------------------------
        # REVIEW SUMMARY
        # -------------------------------------------------

        available_review_count = int(

            review_counts_by_domain
                .get(
                    domain,
                    {}
                )
                .get(
                    source_id,
                    0
                )

        )


        reviews_used_for_embedding = clean_int(

            reviews_used[
                index
            ]

        )


        # -------------------------------------------------
        # PROFILE
        # -------------------------------------------------

        profile = {

            "feel_defined":
                bool(
                    feel_defined[
                        index
                    ]
                ),

        
        }


        # Python does not support JS-style comments inside
        # dictionaries, so add the actual values separately.

        profile[
            "feel"
        ] = feel_profile


        profile[
            "semantic"
        ] = {

            "source":
                clean_string(
                    semantic_sources[
                        index
                    ]
                ),

            "has_base_semantics":
                bool(
                    has_base_semantics[
                        index
                    ]
                ),

            "has_review_semantics":
                bool(
                    has_review_semantics[
                        index
                    ]
                ),

        }


        profile[
            "reviews"
        ] = {

            "available_count":
                available_review_count,

            "used_for_embedding":
                reviews_used_for_embedding
                or
                0,

            "sampled_count":
                min(

                    available_review_count,

                    review_sample_size

                ),

        }


        # -------------------------------------------------
        # ASSIGN SHARD
        # -------------------------------------------------

        shard_index = get_shard_index(

            source_id,

            shard_count

        )


        profile_shards[
            domain
        ][
            shard_index
        ][
            source_id
        ] = profile


    # =====================================================
    # WRITE PROFILE SHARDS
    # =====================================================

    for domain in REVIEW_FILES:


        output_directory = (

            OUTPUT_ROOT
            /
            "profiles"
            /
            domain

        )


        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )


        exported_count = 0


        for shard_index in range(
            shard_count
        ):


            payload = (

                profile_shards[
                    domain
                ][
                    shard_index
                ]

            )


            exported_count += len(
                payload
            )


            write_json(

                output_directory
                /
                get_shard_filename(
                    shard_index
                ),

                payload

            )


            profile_shards[
                domain
            ][
                shard_index
            ] = {}


        print(

            f"{domain}: "
            f"{exported_count:,} "
            "profiles exported"

        )


# =========================================================
# EXPORT MANIFEST
# =========================================================

def export_manifest(
    scaler: StandardScaler,
    feel_df: pd.DataFrame,
    review_counts_by_domain,
    shard_count: int,
    review_sample_size: int,
):
    """
    Store enough metadata for the frontend and report to
    understand exactly how the detail layer was produced.
    """

    domain_summary = {}


    for domain in REVIEW_FILES:


        domain_mask = (

            feel_df[
                "domain"
            ]
            ==
            domain

        )


        domain_defined_mask = (

            domain_mask

            &

            feel_df[
                "_feel_defined"
            ]

        )


        domain_summary[
            domain
        ] = {

            "entity_count":
                int(
                    domain_mask.sum()
                ),

            "feel_defined_count":
                int(
                    domain_defined_mask.sum()
                ),

            "entities_with_reviews":
                int(

                    len(

                        review_counts_by_domain
                            .get(
                                domain,
                                {}
                            )

                    )

                ),

            "prepared_review_count":
                int(

                    sum(

                        review_counts_by_domain
                            .get(
                                domain,
                                {}
                            )
                            .values()

                    )

                ),

        }


    manifest = {

        "schema_version":
            1,

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "sharding": {

            "shard_count":
                shard_count,

            "algorithm":
                "fnv1a_32(source_id) % shard_count",

            "profile_pattern":
                (
                    "profiles/"
                    "{domain}/"
                    "shard_{index:03d}.json"
                ),

            "review_pattern":
                (
                    "reviews/"
                    "{domain}/"
                    "shard_{index:03d}.json"
                ),

        },

        "feel": {

            "available_on_all_atlases":
                True,

            "representation":
                (
                    "global_standardized_"
                    "anchor_derived_"
                    "experiential_scores"
                ),

            "dimensions":
                FEEL_DIMENSIONS,

            "scaling": {

                "method":
                    "StandardScaler",

                "fit_population":
                    (
                        "all semantically defined "
                        "movies + music + restaurants"
                    ),

                "mean": {

                    dimension:
                        round(
                            float(
                                scaler.mean_[
                                    index
                                ]
                            ),
                            8
                        )

                    for (
                        index,
                        dimension
                    )
                    in enumerate(
                        FEEL_DIMENSIONS
                    )

                },

                "scale": {

                    dimension:
                        round(
                            float(
                                scaler.scale_[
                                    index
                                ]
                            ),
                            8
                        )

                    for (
                        index,
                        dimension
                    )
                    in enumerate(
                        FEEL_DIMENSIONS
                    )

                },

            },

        },

        "reviews": {

            "sample_size_per_entity":
                review_sample_size,

            "sampling":
                (
                    "deterministic smallest stable "
                    "hash values by "
                    "domain|source_id|review_id"
                ),

            "note":
                (
                    "Review samples are provided for "
                    "inspection and are not claimed to "
                    "be exactly the subset used when "
                    "creating pooled review embeddings."
                ),

        },

        "domains":
            domain_summary,

    }


    write_json(

        OUTPUT_ROOT
        /
        "manifest.json",

        manifest

    )


# =========================================================
# BUILD FRONTEND ITEM DETAILS
# =========================================================

def build_frontend_item_details(
    shard_count: int,
    review_sample_size: int,
    review_chunk_size: int,
):
    """
    Complete frontend-detail export.

    This pipeline does NOT:

        - retrain embeddings;
        - rerun UMAP;
        - change clustering;
        - change atlas coordinates;
        - change semantic fusion.

    It only packages already-computed information for
    frontend inspection.
    """

    print(
        "\n"
        "=================================================="
    )

    print(
        "BUILD FRONTEND ITEM DETAILS"
    )

    print(
        "=================================================="
    )


    print(
        f"Shard count: {shard_count}"
    )

    print(
        "Reviews shown per entity: "
        f"{review_sample_size}"
    )

    print(
        "Review CSV chunk size: "
        f"{review_chunk_size:,}"
    )


    # =====================================================
    # VALIDATION
    # =====================================================

    validate_input_files()


    if shard_count <= 0:

        raise ValueError(
            "shard_count must be > 0."
        )


    if review_sample_size <= 0:

        raise ValueError(
            "review_sample_size must be > 0."
        )


    if review_chunk_size <= 0:

        raise ValueError(
            "review_chunk_size must be > 0."
        )


    # =====================================================
    # RESET OUTPUT
    # =====================================================

    reset_output_directory()


    # =====================================================
    # FEEL DATA
    # =====================================================

    feel_df = load_feel_scores()


    valid_source_ids = (

        build_valid_source_ids(
            feel_df
        )

    )


    scaled_feel, scaler = (

        standardize_feel_scores(
            feel_df
        )

    )


    # =====================================================
    # REVIEWS
    # =====================================================

    review_counts_by_domain = (

        process_all_reviews(

            valid_source_ids=
                valid_source_ids,

            shard_count=
                shard_count,

            sample_size=
                review_sample_size,

            chunk_size=
                review_chunk_size,

        )

    )


    # =====================================================
    # PROFILES
    # =====================================================

    export_profile_shards(

        feel_df=
            feel_df,

        scaled_feel=
            scaled_feel,

        review_counts_by_domain=
            review_counts_by_domain,

        shard_count=
            shard_count,

        review_sample_size=
            review_sample_size,

    )


    # =====================================================
    # MANIFEST
    # =====================================================

    export_manifest(

        scaler=
            scaler,

        feel_df=
            feel_df,

        review_counts_by_domain=
            review_counts_by_domain,

        shard_count=
            shard_count,

        review_sample_size=
            review_sample_size,

    )


    # =====================================================
    # COMPLETE
    # =====================================================

    print(
        "\n"
        "=================================================="
    )

    print(
        "FRONTEND ITEM DETAILS COMPLETE"
    )

    print(
        "=================================================="
    )


    print(
        f"\nOutput:\n{OUTPUT_ROOT}"
    )


    print(
        "\nGenerated structure:"
    )


    print(
        """
item_details/
├── manifest.json
├── profiles/
│   ├── movies/
│   ├── music/
│   └── restaurants/
└── reviews/
    ├── movies/
    ├── music/
    └── restaurants/
"""
    )


# =========================================================
# CLI
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser(

        description=(
            "Build lightweight frontend item-detail "
            "profiles and lazy-loadable review samples."
        )

    )


    parser.add_argument(

        "--shards",

        type=int,

        default=
            DEFAULT_SHARD_COUNT,

        help=(
            "Number of deterministic JSON shards "
            f"(default: {DEFAULT_SHARD_COUNT})."
        )

    )


    parser.add_argument(

        "--reviews-per-item",

        type=int,

        default=
            DEFAULT_REVIEW_SAMPLE_SIZE,

        help=(
            "Maximum review samples exported per entity "
            f"(default: {DEFAULT_REVIEW_SAMPLE_SIZE})."
        )

    )


    parser.add_argument(

        "--review-chunk-size",

        type=int,

        default=
            DEFAULT_REVIEW_CHUNK_SIZE,

        help=(
            "Number of review rows read from CSV at once "
            f"(default: {DEFAULT_REVIEW_CHUNK_SIZE:,})."
        )

    )


    return parser.parse_args()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    args = parse_args()


    build_frontend_item_details(

        shard_count=
            args.shards,

        review_sample_size=
            args.reviews_per_item,

        review_chunk_size=
            args.review_chunk_size,

    )