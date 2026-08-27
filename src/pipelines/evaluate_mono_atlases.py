# src/pipelines/evaluate_mono_atlases.py

# ============================================================
# MONO-DOMAIN ATLAS EVALUATION
# ============================================================
#
# Purpose
# -------
#
# Quantitatively evaluate whether local neighbourhoods in the
# three mono-domain atlases are semantically meaningful.
#
#
# ATLASES
# -------
#
# Movies
# Music
# Restaurants
#
#
# MAIN QUESTION
# -------------
#
# Are items that are close together in the 2D atlas more
# semantically similar than randomly paired items?
#
#
# DOMAIN-SPECIFIC SEMANTIC SIGNAL
# --------------------------------
#
# Movies:
#     genre overlap
#
# Music:
#     tag/token overlap
#
# Restaurants:
#     category overlap
#
#
# METRICS
# -------
#
# For every domain:
#
#     mean local Jaccard similarity
#     mean random Jaccard similarity
#     absolute improvement
#     similarity lift
#
# For Movies additionally:
#
#     macro-genre neighbour agreement
#     random macro-genre agreement
#
# Existing structural metrics are also reused:
#
#     cluster count
#     largest cluster share
#     cluster entropy
#     cluster silhouette
#
# Review-status artefact diagnostics are also reused:
#
#     review-status kNN balanced accuracy
#     same review-status neighbour share
#     review-status silhouette
#
#
# IMPORTANT
# ---------
#
# This script DOES NOT:
#
#     rebuild embeddings
#     rebuild UMAP
#     rebuild clusters
#     modify frontend files
#     modify any atlas
#
#
# OUTPUT
# ------
#
# data/processed/evaluation/mono_domain/
#
#     mono_atlas_summary.csv
#     movies_semantic_coherence.csv
#     music_semantic_coherence.csv
#     restaurants_semantic_coherence.csv
#
# ============================================================


from pathlib import Path
import ast
import math
import re

import numpy as np
import pandas as pd


from sklearn.neighbors import (
    NearestNeighbors
)


from src.evaluation.atlas_metrics import (
    evaluate_atlas
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)


OUTPUT_DIR = (
    PROCESSED_DIR
    / "evaluation"
    / "mono_domain"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "mono_atlas_summary.csv"
)


# ============================================================
# INPUT ATLASES
# ============================================================

MOVIE_ATLAS = (
    PROCESSED_DIR
    / "movie_map_v1.csv"
)


MUSIC_ATLAS = (
    PROCESSED_DIR
    / "music_map_v1.csv"
)


RESTAURANT_ATLAS = (
    PROCESSED_DIR
    / "restaurant_map_v1.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

K_NEIGHBORS = 15

RANDOM_STATE = 42


# ------------------------------------------------------------
# We do not need to evaluate every item to obtain a robust
# semantic-coherence estimate.
#
# 5,000 query items per atlas gives us:
#
#     5,000 × 15 = 75,000 local neighbour comparisons
#
# which is already substantial.
# ------------------------------------------------------------

MAX_QUERY_ITEMS = 5_000


# ------------------------------------------------------------
# Random baseline comparisons.
#
# We use the same number of random comparisons as local
# neighbour comparisons where practical.
# ------------------------------------------------------------

MAX_RANDOM_PAIRS = (
    MAX_QUERY_ITEMS
    *
    K_NEIGHBORS
)


# ============================================================
# ATLAS CONFIGURATION
# ============================================================


ATLAS_CONFIGS = [

    {
        "id":
            "movies",

        "label":
            "Movies",

        "path":
            MOVIE_ATLAS,

        "semantic_type":
            "movies",

        "detail_output":
            OUTPUT_DIR
            / "movies_semantic_coherence.csv",
    },


    {
        "id":
            "music",

        "label":
            "Music",

        "path":
            MUSIC_ATLAS,

        "semantic_type":
            "music",

        "detail_output":
            OUTPUT_DIR
            / "music_semantic_coherence.csv",
    },


    {
        "id":
            "restaurants",

        "label":
            "Restaurants",

        "path":
            RESTAURANT_ATLAS,

        "semantic_type":
            "restaurants",

        "detail_output":
            OUTPUT_DIR
            / "restaurants_semantic_coherence.csv",
    },

]


# ============================================================
# PRINTING
# ============================================================


def section(
    title
):

    print()

    print(
        "=" * 78
    )

    print(
        title
    )

    print(
        "=" * 78
    )

    print()


# ============================================================
# COLUMN HELPERS
# ============================================================


def first_available_column(
    columns,
    candidates
):
    """
    Return the first candidate column that exists.
    """

    columns = set(
        columns
    )


    for candidate in candidates:

        if candidate in columns:

            return candidate


    return None


# ============================================================
# COORDINATE DETECTION
# ============================================================


def detect_coordinate_columns(
    columns
):

    x_column = first_available_column(

        columns,

        [
            "umap_x",
            "x",
            "position_x"
        ]

    )


    y_column = first_available_column(

        columns,

        [
            "umap_y",
            "y",
            "position_y"
        ]

    )


    if (

        x_column is None

        or

        y_column is None

    ):

        raise ValueError(

            "Could not detect atlas coordinate columns.\n\n"

            f"Available columns:\n"
            f"{list(columns)}"

        )


    return (
        x_column,
        y_column
    )


# ============================================================
# REVIEW FLAG DETECTION
# ============================================================


def detect_review_flag_column(
    columns
):

    return first_available_column(

        columns,

        [
            "has_review_embedding",
            "has_review_semantics"
        ]

    )


# ============================================================
# SEMANTIC COLUMN DETECTION
# ============================================================


def detect_movie_columns(
    columns
):
    """
    Identify Movie semantic columns.
    """

    genre_column = first_available_column(

        columns,

        [
            "genres",
            "genre",
            "genres_text"
        ]

    )


    macro_genre_column = first_available_column(

        columns,

        [
            "macro_genre",
            "macro_genres",
            "macro_genre_label"
        ]

    )


    title_column = first_available_column(

        columns,

        [
            "title",
            "name"
        ]

    )


    return {

        "semantic_column":
            genre_column,

        "macro_column":
            macro_genre_column,

        "name_column":
            title_column,

    }


def detect_music_columns(
    columns
):
    """
    Identify Music semantic columns.

    Prefer tags_text because it corresponds to the semantic
    representation used by the atlas.
    """

    tag_column = first_available_column(

        columns,

        [
            "tags_text",
            "tags_lastfm",
            "tags_mb",
            "tags"
        ]

    )


    name_column = first_available_column(

        columns,

        [
            "title",
            "artist_lastfm",
            "artist_mb",
            "artist_name",
            "name"
        ]

    )


    return {

        "semantic_column":
            tag_column,

        "macro_column":
            None,

        "name_column":
            name_column,

    }


def detect_restaurant_columns(
    columns
):
    """
    Identify Restaurant semantic columns.

    Prefer categories because they are interpretable,
    domain-specific labels.
    """

    category_column = first_available_column(

        columns,

        [
            "categories",
            "category",
            "categories_text"
        ]

    )


    name_column = first_available_column(

        columns,

        [
            "title",
            "name"
        ]

    )


    return {

        "semantic_column":
            category_column,

        "macro_column":
            None,

        "name_column":
            name_column,

    }


# ============================================================
# TOKEN CLEANING
# ============================================================


def clean_token(
    token
):
    """
    Conservative normalization used only for evaluation.
    """

    token = str(
        token
    ).strip().lower()


    token = re.sub(

        r"\s+",

        " ",

        token

    )


    token = token.strip(

        " '\"[](){}"

    )


    return token


# ============================================================
# LIST-LIKE VALUE PARSING
# ============================================================


def parse_list_like(
    value,
    semantic_type
):
    """
    Convert heterogeneous semantic metadata into a set.

    Movies
    ------
    MovieLens genres are usually pipe-separated:

        Action|Adventure|Sci-Fi

    Restaurants
    -----------
    Yelp categories are normally comma-separated:

        Restaurants, Italian, Pizza

    Music
    -----
    tags_text may be a free token string generated during
    preprocessing. We therefore fall back to word-level tokens
    if no clear delimiter is available.

    This parser is deliberately evaluation-only and does not
    affect the semantic model.
    """

    if pd.isna(
        value
    ):

        return set()


    # ========================================================
    # NATIVE COLLECTION
    # ========================================================

    if isinstance(

        value,

        (
            list,
            tuple,
            set
        )

    ):

        tokens = list(
            value
        )


    else:

        text = str(
            value
        ).strip()


        if not text:

            return set()


        # ====================================================
        # PYTHON LIST-LIKE STRING
        # ====================================================

        parsed_collection = None


        if (

            text.startswith("[")
            and
            text.endswith("]")

        ):

            try:

                candidate = ast.literal_eval(
                    text
                )


                if isinstance(

                    candidate,

                    (
                        list,
                        tuple,
                        set
                    )

                ):

                    parsed_collection = list(
                        candidate
                    )


            except (
                ValueError,
                SyntaxError
            ):

                parsed_collection = None


        if parsed_collection is not None:

            tokens = parsed_collection


        # ====================================================
        # MOVIES
        # ====================================================

        elif semantic_type == "movies":

            if "|" in text:

                tokens = text.split(
                    "|"
                )


            elif "," in text:

                tokens = text.split(
                    ","
                )


            else:

                tokens = [
                    text
                ]


        # ====================================================
        # RESTAURANTS
        # ====================================================

        elif semantic_type == "restaurants":

            if "," in text:

                tokens = text.split(
                    ","
                )


            elif "|" in text:

                tokens = text.split(
                    "|"
                )


            else:

                tokens = [
                    text
                ]


        # ====================================================
        # MUSIC
        # ====================================================

        elif semantic_type == "music":

            if "|" in text:

                tokens = text.split(
                    "|"
                )


            elif "," in text:

                tokens = text.split(
                    ","
                )


            else:

                # --------------------------------------------
                # tags_text in the current music pipeline is
                # often a natural text/token representation.
                #
                # Word-level overlap is therefore used as a
                # conservative proxy for semantic tag overlap.
                # --------------------------------------------

                tokens = re.findall(

                    r"[a-zA-Z0-9][a-zA-Z0-9_\-']+",

                    text

                )


        else:

            tokens = [
                text
            ]


    # ========================================================
    # CLEAN + DEDUPLICATE
    # ========================================================

    cleaned = {

        clean_token(
            token
        )

        for token in tokens

        if clean_token(
            token
        )

    }


    # --------------------------------------------------------
    # Remove common missing-value strings.
    # --------------------------------------------------------

    cleaned -= {

        "nan",

        "none",

        "null",

        "(no genres listed)",

        "no genres listed"

    }


    return cleaned


# ============================================================
# JACCARD
# ============================================================


def jaccard_similarity(
    a,
    b
):
    """
    Jaccard similarity between two sets.

    Empty/empty comparisons are not semantically meaningful
    and therefore return NaN rather than 1.
    """

    if (

        not a

        or

        not b

    ):

        return np.nan


    intersection = len(

        a.intersection(
            b
        )

    )


    union = len(

        a.union(
            b
        )

    )


    if union == 0:

        return np.nan


    return (

        intersection

        /

        union

    )


# ============================================================
# RANDOM SAMPLE
# ============================================================


def sample_indices(

    n_items,

    max_items,

    random_state

):
    """
    Deterministic uniform sample.
    """

    if n_items <= max_items:

        return np.arange(
            n_items
        )


    rng = np.random.default_rng(
        random_state
    )


    return np.sort(

        rng.choice(

            n_items,

            size=
                max_items,

            replace=
                False

        )

    )


# ============================================================
# LOCAL NEIGHBOURS
# ============================================================


def compute_local_neighbors(

    positions,

    query_indices,

    k=
        K_NEIGHBORS

):
    """
    Find 2D nearest neighbours using the complete atlas as the
    reference population.
    """

    positions = np.asarray(

        positions,

        dtype=np.float32

    )


    actual_k = min(

        int(
            k
        ),

        len(
            positions
        )
        -
        1

    )


    if actual_k < 1:

        raise ValueError(

            "Atlas contains too few items for kNN evaluation."

        )


    model = NearestNeighbors(

        n_neighbors=
            actual_k
            +
            1,

        metric=
            "euclidean",

        algorithm=
            "auto"

    )


    model.fit(
        positions
    )


    distances, indices = model.kneighbors(

        positions[
            query_indices
        ]

    )


    cleaned_indices = []

    cleaned_distances = []


    for row_index, original_index in enumerate(

        query_indices

    ):


        row_neighbors = indices[
            row_index
        ]


        row_distances = distances[
            row_index
        ]


        mask = (

            row_neighbors
            !=
            original_index

        )


        row_neighbors = row_neighbors[
            mask
        ][
            :actual_k
        ]


        row_distances = row_distances[
            mask
        ][
            :actual_k
        ]


        cleaned_indices.append(
            row_neighbors
        )


        cleaned_distances.append(
            row_distances
        )


    return (

        np.asarray(
            cleaned_indices,
            dtype=np.int64
        ),

        np.asarray(
            cleaned_distances,
            dtype=np.float32
        )

    )


# ============================================================
# RANDOM PAIRS
# ============================================================


def create_random_pairs(

    n_items,

    n_pairs,

    random_state

):
    """
    Generate random non-self pairs.
    """

    rng = np.random.default_rng(
        random_state
    )


    left = rng.integers(

        0,

        n_items,

        size=
            n_pairs

    )


    right = rng.integers(

        0,

        n_items,

        size=
            n_pairs

    )


    # --------------------------------------------------------
    # Resample accidental self-pairs.
    # --------------------------------------------------------

    same = (

        left
        ==
        right

    )


    while np.any(
        same
    ):

        right[
            same
        ] = rng.integers(

            0,

            n_items,

            size=
                int(
                    same.sum()
                )

        )


        same = (

            left
            ==
            right

        )


    return (
        left,
        right
    )


# ============================================================
# LOCAL SEMANTIC COHERENCE
# ============================================================


def evaluate_semantic_coherence(

    df,

    semantic_column,

    semantic_type,

    name_column=None,

    macro_column=None,

    k=
        K_NEIGHBORS,

    max_query_items=
        MAX_QUERY_ITEMS,

    random_state=
        RANDOM_STATE

):
    """
    Compare semantic overlap between local atlas neighbours and
    randomly paired items.
    """

    # ========================================================
    # COORDINATES
    # ========================================================

    x_column, y_column = (
        detect_coordinate_columns(
            df.columns
        )
    )


    working = df.copy()


    working[
        x_column
    ] = pd.to_numeric(

        working[
            x_column
        ],

        errors=
            "coerce"

    )


    working[
        y_column
    ] = pd.to_numeric(

        working[
            y_column
        ],

        errors=
            "coerce"

    )


    working = working.loc[

        working[
            [
                x_column,
                y_column
            ]
        ]
        .notna()
        .all(
            axis=1
        )

    ].reset_index(
        drop=True
    )


    # ========================================================
    # PARSE SEMANTIC SETS
    # ========================================================

    print(
        f"Parsing semantic field: "
        f"{semantic_column}"
    )


    semantic_sets = [

        parse_list_like(

            value,

            semantic_type=
                semantic_type

        )

        for value in working[
            semantic_column
        ]

    ]


    has_semantics = np.asarray(

        [

            len(
                item
            )
            >
            0

            for item in semantic_sets

        ],

        dtype=bool

    )


    semantic_coverage = float(

        has_semantics.mean()

    )


    print(
        f"Semantic metadata coverage: "
        f"{semantic_coverage * 100:.2f}%"
    )


    # ========================================================
    # QUERY ITEMS
    #
    # Only entities with interpretable semantic labels are used
    # as queries.
    # ========================================================

    valid_query_candidates = np.flatnonzero(
        has_semantics
    )


    if len(
        valid_query_candidates
    ) == 0:

        raise ValueError(

            f"No usable semantic metadata found in "
            f"'{semantic_column}'."

        )


    if (

        len(
            valid_query_candidates
        )
        >
        max_query_items

    ):

        rng = np.random.default_rng(
            random_state
        )


        query_indices = np.sort(

            rng.choice(

                valid_query_candidates,

                size=
                    max_query_items,

                replace=
                    False

            )

        )


    else:

        query_indices = (
            valid_query_candidates
        )


    print(
        f"Query items: "
        f"{len(query_indices):,}"
    )


    # ========================================================
    # NEIGHBOURS
    # ========================================================

    positions = (

        working[
            [
                x_column,
                y_column
            ]
        ]
        .to_numpy(
            dtype=np.float32
        )

    )


    (

        neighbor_indices,

        neighbor_distances

    ) = compute_local_neighbors(

        positions,

        query_indices,

        k=
            k

    )


    # ========================================================
    # LOCAL SEMANTIC SIMILARITY
    # ========================================================

    local_scores = []

    detail_records = []


    macro_matches = []


    for query_row, query_index in enumerate(

        query_indices

    ):


        query_set = semantic_sets[
            query_index
        ]


        query_local_scores = []


        for neighbor_rank, neighbor_index in enumerate(

            neighbor_indices[
                query_row
            ],

            start=1

        ):


            neighbor_set = semantic_sets[
                neighbor_index
            ]


            score = jaccard_similarity(

                query_set,

                neighbor_set

            )


            if np.isfinite(
                score
            ):

                local_scores.append(
                    score
                )


                query_local_scores.append(
                    score
                )


            # =================================================
            # MACRO-GENRE AGREEMENT
            # =================================================

            if (

                macro_column is not None

                and

                macro_column
                in working.columns

            ):


                query_macro = working.iloc[
                    query_index
                ][
                    macro_column
                ]


                neighbor_macro = working.iloc[
                    neighbor_index
                ][
                    macro_column
                ]


                if (

                    pd.notna(
                        query_macro
                    )

                    and

                    pd.notna(
                        neighbor_macro
                    )

                ):


                    query_macro = str(
                        query_macro
                    ).strip()


                    neighbor_macro = str(
                        neighbor_macro
                    ).strip()


                    if (

                        query_macro

                        and

                        neighbor_macro

                    ):

                        macro_matches.append(

                            float(

                                query_macro
                                ==
                                neighbor_macro

                            )

                        )


        # ====================================================
        # DETAIL RECORD
        # ====================================================

        record = {

            "query_index":
                int(
                    query_index
                ),

            "mean_neighbor_jaccard":
                (
                    float(
                        np.mean(
                            query_local_scores
                        )
                    )

                    if query_local_scores

                    else np.nan
                ),

            "valid_neighbor_comparisons":
                int(
                    len(
                        query_local_scores
                    )
                ),

            "mean_2d_neighbor_distance":
                float(

                    np.mean(

                        neighbor_distances[
                            query_row
                        ]

                    )

                ),

        }


        if (

            name_column is not None

            and

            name_column
            in working.columns

        ):

            record[
                "name"
            ] = working.iloc[
                query_index
            ][
                name_column
            ]


        detail_records.append(
            record
        )


    # ========================================================
    # RANDOM BASELINE
    # ========================================================

    n_random_pairs = min(

        MAX_RANDOM_PAIRS,

        max(

            1,

            len(
                local_scores
            )

        )

    )


    random_left, random_right = (
        create_random_pairs(

            n_items=
                len(
                    working
                ),

            n_pairs=
                n_random_pairs,

            random_state=
                random_state
                +
                100

        )
    )


    random_scores = []


    random_macro_matches = []


    for left, right in zip(

        random_left,

        random_right

    ):


        score = jaccard_similarity(

            semantic_sets[
                left
            ],

            semantic_sets[
                right
            ]

        )


        if np.isfinite(
            score
        ):

            random_scores.append(
                score
            )


        # ====================================================
        # RANDOM MACRO-GENRE BASELINE
        # ====================================================

        if (

            macro_column is not None

            and

            macro_column
            in working.columns

        ):


            left_macro = working.iloc[
                left
            ][
                macro_column
            ]


            right_macro = working.iloc[
                right
            ][
                macro_column
            ]


            if (

                pd.notna(
                    left_macro
                )

                and

                pd.notna(
                    right_macro
                )

            ):


                left_macro = str(
                    left_macro
                ).strip()


                right_macro = str(
                    right_macro
                ).strip()


                if (

                    left_macro

                    and

                    right_macro

                ):

                    random_macro_matches.append(

                        float(

                            left_macro
                            ==
                            right_macro

                        )

                    )


    # ========================================================
    # AGGREGATE METRICS
    # ========================================================

    mean_local = (

        float(
            np.mean(
                local_scores
            )
        )

        if local_scores

        else np.nan

    )


    mean_random = (

        float(
            np.mean(
                random_scores
            )
        )

        if random_scores

        else np.nan

    )


    absolute_improvement = (

        mean_local
        -
        mean_random

        if (

            np.isfinite(
                mean_local
            )

            and

            np.isfinite(
                mean_random
            )

        )

        else np.nan

    )


    if (

        np.isfinite(
            mean_random
        )

        and

        mean_random > 0

    ):

        similarity_lift = (

            mean_local

            /

            mean_random

        )


    else:

        similarity_lift = np.nan


    metrics = {

        "semantic_metadata_coverage":
            semantic_coverage,

        "semantic_query_items":
            int(
                len(
                    query_indices
                )
            ),

        "local_semantic_comparisons":
            int(
                len(
                    local_scores
                )
            ),

        "random_semantic_comparisons":
            int(
                len(
                    random_scores
                )
            ),

        "mean_local_jaccard":
            mean_local,

        "mean_random_jaccard":
            mean_random,

        "jaccard_absolute_improvement":
            absolute_improvement,

        "jaccard_similarity_lift":
            similarity_lift,

    }


    # ========================================================
    # MOVIE MACRO GENRE
    # ========================================================

    if macro_matches:

        local_macro = float(

            np.mean(
                macro_matches
            )

        )


        random_macro = (

            float(

                np.mean(
                    random_macro_matches
                )

            )

            if random_macro_matches

            else np.nan

        )


        metrics.update({

            "local_macro_genre_agreement":
                local_macro,

            "random_macro_genre_agreement":
                random_macro,

            "macro_genre_absolute_improvement":
                (
                    local_macro
                    -
                    random_macro

                    if np.isfinite(
                        random_macro
                    )

                    else np.nan
                ),

            "macro_genre_agreement_lift":
                (
                    local_macro
                    /
                    random_macro

                    if (

                        np.isfinite(
                            random_macro
                        )

                        and

                        random_macro > 0

                    )

                    else np.nan
                ),

        })


    else:

        metrics.update({

            "local_macro_genre_agreement":
                np.nan,

            "random_macro_genre_agreement":
                np.nan,

            "macro_genre_absolute_improvement":
                np.nan,

            "macro_genre_agreement_lift":
                np.nan,

        })


    details = pd.DataFrame(
        detail_records
    )


    return (
        metrics,
        details
    )


# ============================================================
# LOAD ATLAS
# ============================================================


def load_atlas(
    config
):
    """
    Load one mono-domain analysis CSV.
    """

    path = config[
        "path"
    ]


    if not path.exists():

        raise FileNotFoundError(

            f"Atlas file not found:\n"
            f"{path}"

        )


    print(
        f"Source:"
    )


    print(
        path
    )


    print()


    df = pd.read_csv(

        path,

        low_memory=
            False

    )


    print(
        f"Rows: "
        f"{len(df):,}"
    )


    return df


# ============================================================
# DETECT DOMAIN SEMANTICS
# ============================================================


def detect_semantic_configuration(

    df,

    semantic_type

):
    """
    Determine which interpretable semantic fields are
    available in one atlas.
    """

    if semantic_type == "movies":

        detected = detect_movie_columns(
            df.columns
        )


    elif semantic_type == "music":

        detected = detect_music_columns(
            df.columns
        )


    elif semantic_type == "restaurants":

        detected = detect_restaurant_columns(
            df.columns
        )


    else:

        raise ValueError(

            f"Unsupported semantic type: "
            f"{semantic_type}"

        )


    semantic_column = detected[
        "semantic_column"
    ]


    if semantic_column is None:

        raise ValueError(

            f"Could not find an interpretable semantic "
            f"metadata field for {semantic_type}.\n\n"

            f"Available columns:\n"
            f"{list(df.columns)}"

        )


    return detected


# ============================================================
# EXISTING STRUCTURAL METRICS
# ============================================================


def evaluate_existing_atlas_metrics(

    df,

    atlas_id,

    atlas_label

):
    """
    Reuse the generic atlas evaluator for cluster and
    review-availability diagnostics.
    """

    review_flag = detect_review_flag_column(
        df.columns
    )


    # --------------------------------------------------------
    # evaluate_atlas only needs the atlas geometry for the
    # structural metrics used here.
    # --------------------------------------------------------

    metrics, _ = evaluate_atlas(

        df=
            df,

        atlas_id=
            atlas_id,

        atlas_label=
            atlas_label,

        review_flag_column=
            review_flag,

        feature_columns=
            None,

        source_feature_name=
            None,

        source_metric=
            "cosine",

        k=
            K_NEIGHBORS,

        random_state=
            RANDOM_STATE

    )


    return metrics


# ============================================================
# PRINT ONE RESULT
# ============================================================


def print_domain_result(

    label,

    semantic_column,

    semantic_metrics,

    structural_metrics

):

    print()

    print(
        f"Semantic field: "
        f"{semantic_column}"
    )


    print()

    print(
        "Local semantic coherence"
    )


    print(

        f"  Local Jaccard:        "
        f"{semantic_metrics['mean_local_jaccard']:.4f}"

    )


    print(

        f"  Random Jaccard:       "
        f"{semantic_metrics['mean_random_jaccard']:.4f}"

    )


    print(

        f"  Absolute improvement: "
        f"{semantic_metrics['jaccard_absolute_improvement']:.4f}"

    )


    if np.isfinite(

        semantic_metrics[
            "jaccard_similarity_lift"
        ]

    ):

        print(

            f"  Similarity lift:      "
            f"{semantic_metrics['jaccard_similarity_lift']:.2f}x"

        )


    # ========================================================
    # MOVIE MACRO GENRES
    # ========================================================

    if np.isfinite(

        semantic_metrics[
            "local_macro_genre_agreement"
        ]

    ):

        print()


        print(
            "Macro-genre agreement"
        )


        print(

            f"  Local agreement:      "
            f"{semantic_metrics['local_macro_genre_agreement']:.4f}"

        )


        print(

            f"  Random agreement:     "
            f"{semantic_metrics['random_macro_genre_agreement']:.4f}"

        )


        if np.isfinite(

            semantic_metrics[
                "macro_genre_agreement_lift"
            ]

        ):

            print(

                f"  Agreement lift:       "
                f"{semantic_metrics['macro_genre_agreement_lift']:.2f}x"

            )


    # ========================================================
    # CLUSTERS
    # ========================================================

    print()


    print(
        "Cluster structure"
    )


    print(

        f"  Clusters:             "
        f"{structural_metrics['cluster_count']}"

    )


    print(

        f"  Largest cluster:      "
        f"{structural_metrics['largest_cluster_share'] * 100:.2f}%"

    )


    print(

        f"  Cluster entropy:      "
        f"{structural_metrics['cluster_size_entropy']:.4f}"

    )


    print(

        f"  Cluster silhouette:   "
        f"{structural_metrics['cluster_silhouette_2d']:.4f}"

    )


    # ========================================================
    # REVIEW ARTEFACT
    # ========================================================

    review_accuracy = structural_metrics.get(

        "review_status_knn_balanced_accuracy_2d",

        np.nan

    )


    if np.isfinite(
        review_accuracy
    ):

        print()


        print(
            "Review-status geometry"
        )


        print(

            f"  kNN balanced accuracy: "
            f"{review_accuracy:.4f}"

        )


        same_share = structural_metrics.get(

            "review_status_same_neighbor_share_2d",

            np.nan

        )


        if np.isfinite(
            same_share
        ):

            print(

                f"  Same-status neighbours: "
                f"{same_share:.4f}"

            )


        review_silhouette = structural_metrics.get(

            "review_status_silhouette_2d",

            np.nan

        )


        if np.isfinite(
            review_silhouette
        ):

            print(

                f"  Review silhouette:      "
                f"{review_silhouette:.4f}"

            )


# ============================================================
# MAIN
# ============================================================


def main():

    section(
        "MONO-DOMAIN ATLAS EVALUATION"
    )


    print(
        f"k-neighbourhood size: "
        f"{K_NEIGHBORS}"
    )


    print(
        f"Maximum query items per atlas: "
        f"{MAX_QUERY_ITEMS:,}"
    )


    print(
        f"Random state: "
        f"{RANDOM_STATE}"
    )


    summary_records = []


    # ========================================================
    # EACH DOMAIN
    # ========================================================

    for config in ATLAS_CONFIGS:


        section(

            f"EVALUATING: "
            f"{config['label']}"

        )


        # ====================================================
        # LOAD
        # ====================================================

        df = load_atlas(
            config
        )


        # ====================================================
        # DETECT SEMANTIC METADATA
        # ====================================================

        semantic_config = (
            detect_semantic_configuration(

                df,

                config[
                    "semantic_type"
                ]

            )
        )


        semantic_column = (
            semantic_config[
                "semantic_column"
            ]
        )


        macro_column = (
            semantic_config[
                "macro_column"
            ]
        )


        name_column = (
            semantic_config[
                "name_column"
            ]
        )


        print()


        print(
            f"Detected semantic field: "
            f"{semantic_column}"
        )


        if macro_column is not None:

            print(

                f"Detected macro field:    "
                f"{macro_column}"

            )


        # ====================================================
        # LOCAL SEMANTIC COHERENCE
        # ====================================================

        (

            semantic_metrics,

            detail_df

        ) = evaluate_semantic_coherence(

            df=
                df,

            semantic_column=
                semantic_column,

            semantic_type=
                config[
                    "semantic_type"
                ],

            name_column=
                name_column,

            macro_column=
                macro_column,

            k=
                K_NEIGHBORS,

            max_query_items=
                MAX_QUERY_ITEMS,

            random_state=
                RANDOM_STATE

        )


        # ====================================================
        # STRUCTURAL + REVIEW METRICS
        # ====================================================

        structural_metrics = (
            evaluate_existing_atlas_metrics(

                df=
                    df,

                atlas_id=
                    config[
                        "id"
                    ],

                atlas_label=
                    config[
                        "label"
                    ]

            )
        )


        # ====================================================
        # SAVE DETAILED LOCAL RESULTS
        # ====================================================

        detail_df.to_csv(

            config[
                "detail_output"
            ],

            index=False

        )


        # ====================================================
        # SUMMARY RECORD
        # ====================================================

        record = {

            "atlas_id":
                config[
                    "id"
                ],

            "atlas_label":
                config[
                    "label"
                ],

            "n_items":
                len(
                    df
                ),

            "semantic_field":
                semantic_column,

            **semantic_metrics,

            "cluster_count":
                structural_metrics.get(

                    "cluster_count",

                    np.nan

                ),

            "largest_cluster_share":
                structural_metrics.get(

                    "largest_cluster_share",

                    np.nan

                ),

            "cluster_size_entropy":
                structural_metrics.get(

                    "cluster_size_entropy",

                    np.nan

                ),

            "cluster_silhouette_2d":
                structural_metrics.get(

                    "cluster_silhouette_2d",

                    np.nan

                ),

            "review_status_knn_balanced_accuracy_2d":
                structural_metrics.get(

                    "review_status_knn_balanced_accuracy_2d",

                    np.nan

                ),

            "review_status_same_neighbor_share_2d":
                structural_metrics.get(

                    "review_status_same_neighbor_share_2d",

                    np.nan

                ),

            "review_status_silhouette_2d":
                structural_metrics.get(

                    "review_status_silhouette_2d",

                    np.nan

                ),

        }


        summary_records.append(
            record
        )


        # ====================================================
        # PRINT
        # ====================================================

        print_domain_result(

            label=
                config[
                    "label"
                ],

            semantic_column=
                semantic_column,

            semantic_metrics=
                semantic_metrics,

            structural_metrics=
                structural_metrics

        )


        print()


        print(
            f"Detailed output:"
        )


        print(
            config[
                "detail_output"
            ]
        )


    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    summary_df = pd.DataFrame(
        summary_records
    )


    summary_df.to_csv(

        SUMMARY_OUTPUT,

        index=False

    )


    section(
        "MONO-DOMAIN COMPARISON"
    )


    display_columns = [

        "atlas_label",

        "n_items",

        "semantic_metadata_coverage",

        "mean_local_jaccard",

        "mean_random_jaccard",

        "jaccard_similarity_lift",

        "local_macro_genre_agreement",

        "random_macro_genre_agreement",

        "cluster_silhouette_2d",

        "review_status_knn_balanced_accuracy_2d",

    ]


    display_columns = [

        column

        for column in display_columns

        if column in summary_df.columns

    ]


    display_df = summary_df[
        display_columns
    ].copy()


    numeric_columns = (
        display_df
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .columns
    )


    display_df[
        numeric_columns
    ] = (

        display_df[
            numeric_columns
        ]
        .round(
            4
        )

    )


    print(

        display_df.to_string(

            index=False

        )

    )


    # ========================================================
    # INTERPRETATION GUIDE
    # ========================================================

    section(
        "INTERPRETATION GUIDE"
    )


    print(
        "mean_local_jaccard"
    )


    print(
        "  Average semantic overlap between an item and its"
    )


    print(
        "  nearest neighbours in the 2D atlas."
    )


    print()


    print(
        "mean_random_jaccard"
    )


    print(
        "  Expected overlap between randomly paired items."
    )


    print()


    print(
        "jaccard_similarity_lift"
    )


    print(
        "  local similarity / random similarity"
    )


    print()


    print(
        "  > 1.0:"
    )


    print(
        "      atlas neighbours are more semantically similar"
    )


    print(
        "      than random items."
    )


    print()


    print(
        "  The larger the lift, the stronger the evidence that"
    )


    print(
        "  local atlas proximity carries domain meaning."
    )


    print()


    print(
        "Movie macro-genre agreement"
    )


    print(
        "  Tests whether neighbouring Movies share the same"
    )


    print(
        "  broad genre more often than random Movie pairs."
    )


    print()


    print(
        "review_status_knn_balanced_accuracy_2d"
    )


    print(
        "  Around 0.5:"
    )


    print(
        "      review availability is difficult to recover from"
    )


    print(
        "      atlas position."
    )


    print()


    print(
        "  Much greater than 0.5:"
    )


    print(
        "      reviewed and non-reviewed entities occupy"
    )


    print(
        "      systematically different neighbourhoods."
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    section(
        "OUTPUTS"
    )


    print(
        "Summary:"
    )


    print(
        SUMMARY_OUTPUT
    )


    print()


    print(
        "Detailed domain results:"
    )


    for config in ATLAS_CONFIGS:

        print(

            f"  {config['detail_output']}"

        )


    section(
        "MONO-DOMAIN EVALUATION COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()