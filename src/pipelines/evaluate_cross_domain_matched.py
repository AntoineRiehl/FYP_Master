# src/pipelines/evaluate_cross_domain_matched.py

# ============================================================
# MATCHED-POPULATION CROSS-DOMAIN EVALUATION
# ============================================================
#
# Purpose
# -------
#
# Compare Method A (General Semantic) and Method B
# (Shared Experiential / Feel) fairly by evaluating them on
# EXACTLY the same entities.
#
#
# WHY THIS IS NEEDED
# ------------------
#
# The complete Method A atlases contain:
#
#     Movies + Music:
#         274,380 entities
#
#     Movies + Music + Restaurants:
#         308,321 entities
#
#
# Method B excludes entities that have neither usable base
# semantics nor review semantics:
#
#     Movies + Music:
#         220,062 entities
#
#     Movies + Music + Restaurants:
#         254,003 entities
#
#
# Therefore, comparing the complete maps alone introduces a
# population difference.
#
# This script computes:
#
#     intersection(Method A, Method B)
#
# and evaluates both geometries on that identical population.
#
#
# MAIN METRICS
# ------------
#
# Cross-domain integration:
#
#     domain kNN balanced accuracy
#     local domain entropy
#     cross-domain neighbour share
#     domain silhouette
#
#
# Structural diagnostics:
#
#     cluster count
#     largest cluster share
#     cluster-size entropy
#     cluster silhouette
#
#
# Review-status artefact diagnostics:
#
#     review-status kNN balanced accuracy
#     same-review-status neighbour share
#     review-status silhouette
#
#
# IMPORTANT
# ---------
#
# This script does NOT:
#
#     rebuild either atlas
#     rerun UMAP
#     rerun clustering
#     modify any representation
#
#
# It evaluates the existing 2D geometries only.
#
# ============================================================


from pathlib import Path
import re

import numpy as np
import pandas as pd


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
    / "cross_domain_matched"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CLUSTER_OUTPUT_DIR = (
    OUTPUT_DIR
    / "cluster_composition"
)


CLUSTER_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


METRICS_OUTPUT = (
    OUTPUT_DIR
    / "matched_atlas_metrics.csv"
)


COMPARISON_OUTPUT = (
    OUTPUT_DIR
    / "matched_method_comparison.csv"
)


POPULATION_OUTPUT = (
    OUTPUT_DIR
    / "matched_population_summary.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

K_NEIGHBORS = 15

RANDOM_STATE = 42


# ============================================================
# PAIRS TO COMPARE
# ============================================================

PAIR_CONFIGS = [

    {

        "pair_id":
            "movies_music",

        "label":
            "Movies + Music",

        "method_a_path":
            PROCESSED_DIR
            / "movies_music_atlas_v1.csv",

        "method_b_path":
            PROCESSED_DIR
            / "movies_music_feel_atlas.csv",

        "expected_matched_items":
            220_062,

        "expected_domains": [
            "movies",
            "music"
        ],

    },


    {

        "pair_id":
            "movies_music_restaurants",

        "label":
            "Movies + Music + Restaurants",

        "method_a_path":
            PROCESSED_DIR
            / "movies_music_restaurants_atlas_v1.csv",

        "method_b_path":
            PROCESSED_DIR
            / "movies_music_restaurants_feel_atlas.csv",

        "expected_matched_items":
            254_003,

        "expected_domains": [
            "movies",
            "music",
            "restaurants"
        ],

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

    columns = set(
        columns
    )


    for candidate in candidates:

        if candidate in columns:

            return candidate


    return None


# ============================================================
# ID NORMALIZATION
# ============================================================


def normalize_source_id(
    value
):
    """
    Normalize heterogeneous identifiers.

    Examples:

        123
        123.0
        "123"

    all become:

        "123"

    UUIDs remain unchanged.
    """

    if pd.isna(
        value
    ):

        return ""


    text = str(
        value
    ).strip()


    if re.fullmatch(

        r"[+-]?\d+\.0+",

        text

    ):

        return text.split(
            "."
        )[0]


    return text


# ============================================================
# LOAD ONE CROSS-DOMAIN ATLAS
# ============================================================


def load_cross_domain_atlas(
    path,
    method_label
):
    """
    Load only the fields needed for matched evaluation.

    Returns standardized columns:

        domain
        source_id
        entity_key
        umap_x
        umap_y
        cluster
        has_review_embedding   (when available)
    """

    if not path.exists():

        raise FileNotFoundError(

            f"{method_label} atlas not found:\n"
            f"{path}"

        )


    header = pd.read_csv(

        path,

        nrows=0

    )


    columns = list(
        header.columns
    )


    # ========================================================
    # DOMAIN
    # ========================================================

    domain_column = first_available_column(

        columns,

        [
            "domain",
            "source_domain"
        ]

    )


    if domain_column is None:

        raise ValueError(

            f"Could not identify domain column in:\n"
            f"{path}\n\n"

            f"Available columns:\n"
            f"{columns}"

        )


    # ========================================================
    # SOURCE ID
    # ========================================================

    source_id_column = first_available_column(

        columns,

        [
            "source_id",

            # Fallbacks, mainly defensive.
            "movieId",
            "movie_id",
            "mbid",
            "business_id"
        ]

    )


    # --------------------------------------------------------
    # Cross-domain builders normally include source_id.
    #
    # If not, a global id such as:
    #
    #     movies:123
    #
    # can be used instead.
    # --------------------------------------------------------

    global_id_column = first_available_column(

        columns,

        [
            "id",
            "global_id"
        ]

    )


    if (

        source_id_column is None

        and

        global_id_column is None

    ):

        raise ValueError(

            f"Could not identify entity IDs in:\n"
            f"{path}\n\n"

            f"Available columns:\n"
            f"{columns}"

        )


    # ========================================================
    # COORDINATES
    # ========================================================

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

            f"Could not identify UMAP coordinates in:\n"
            f"{path}"

        )


    # ========================================================
    # CLUSTER
    # ========================================================

    cluster_column = first_available_column(

        columns,

        [
            "cluster",
            "cluster_id"
        ]

    )


    # ========================================================
    # REVIEW STATUS
    # ========================================================

    review_column = first_available_column(

        columns,

        [
            "has_review_embedding",
            "has_review_semantics"
        ]

    )


    # ========================================================
    # LOAD ONLY REQUIRED COLUMNS
    # ========================================================

    usecols = {

        domain_column,
        x_column,
        y_column

    }


    if source_id_column is not None:

        usecols.add(
            source_id_column
        )


    if global_id_column is not None:

        usecols.add(
            global_id_column
        )


    if cluster_column is not None:

        usecols.add(
            cluster_column
        )


    if review_column is not None:

        usecols.add(
            review_column
        )


    df = pd.read_csv(

        path,

        usecols=
            list(
                usecols
            ),

        low_memory=
            False

    )


    # ========================================================
    # STANDARDIZE DOMAIN
    # ========================================================

    df[
        "domain"
    ] = (

        df[
            domain_column
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()

    )


    # ========================================================
    # STANDARDIZE SOURCE ID
    # ========================================================

    if source_id_column is not None:

        df[
            "source_id"
        ] = (

            df[
                source_id_column
            ]
            .apply(
                normalize_source_id
            )

        )


        df[
            "entity_key"
        ] = (

            df[
                "domain"
            ]

            +

            ":"

            +

            df[
                "source_id"
            ]

        )


    else:

        # ----------------------------------------------------
        # Fallback to global ID.
        # ----------------------------------------------------

        global_ids = (

            df[
                global_id_column
            ]
            .fillna("")
            .astype(str)
            .str.strip()

        )


        keys = []


        for domain, global_id in zip(

            df[
                "domain"
            ],

            global_ids

        ):


            expected_prefix = (

                domain

                +

                ":"

            )


            if global_id.startswith(
                expected_prefix
            ):

                key = global_id


            else:

                key = (

                    expected_prefix

                    +

                    normalize_source_id(
                        global_id
                    )

                )


            keys.append(
                key
            )


        df[
            "entity_key"
        ] = keys


        df[
            "source_id"
        ] = (

            df[
                "entity_key"
            ]
            .str.split(
                ":",
                n=1
            )
            .str[
                1
            ]

        )


    # ========================================================
    # STANDARDIZE COORDINATES
    # ========================================================

    df[
        "umap_x"
    ] = pd.to_numeric(

        df[
            x_column
        ],

        errors=
            "coerce"

    )


    df[
        "umap_y"
    ] = pd.to_numeric(

        df[
            y_column
        ],

        errors=
            "coerce"

    )


    # ========================================================
    # STANDARDIZE CLUSTER
    # ========================================================

    if cluster_column is not None:

        df[
            "cluster"
        ] = df[
            cluster_column
        ]


    # ========================================================
    # STANDARDIZE REVIEW FLAG
    # ========================================================

    if review_column is not None:

        if review_column != "has_review_embedding":

            df[
                "has_review_embedding"
            ] = df[
                review_column
            ]


    # ========================================================
    # KEEP STANDARD FIELDS ONLY
    # ========================================================

    keep_columns = [

        "domain",
        "source_id",
        "entity_key",
        "umap_x",
        "umap_y"

    ]


    if "cluster" in df.columns:

        keep_columns.append(
            "cluster"
        )


    if "has_review_embedding" in df.columns:

        keep_columns.append(
            "has_review_embedding"
        )


    df = df[
        keep_columns
    ].copy()


    # ========================================================
    # VALIDATE
    # ========================================================

    if df[
        "entity_key"
    ].duplicated().any():

        duplicates = (

            df.loc[

                df[
                    "entity_key"
                ].duplicated(
                    keep=False
                ),

                "entity_key"

            ]
            .head(
                10
            )
            .tolist()

        )


        raise ValueError(

            f"Duplicate entity keys detected in "
            f"{method_label}.\n\n"

            f"Examples:\n"
            f"{duplicates}"

        )


    invalid_coordinates = (

        df[
            [
                "umap_x",
                "umap_y"
            ]
        ]
        .isna()
        .any(
            axis=1
        )

    )


    if invalid_coordinates.any():

        raise ValueError(

            f"{method_label} contains "
            f"{int(invalid_coordinates.sum()):,} entities "
            f"with invalid UMAP coordinates."

        )


    print(
        f"{method_label} entities: "
        f"{len(df):,}"
    )


    print(
        f"{method_label} domains:"
    )


    print(

        df[
            "domain"
        ]
        .value_counts()
        .to_string()

    )


    return df


# ============================================================
# CREATE MATCHED POPULATION
# ============================================================


def create_matched_population(

    method_a,

    method_b,

    expected_items=None

):
    """
    Match Method A and Method B by global entity identity.

    Both returned DataFrames are ordered by exactly the same
    entity_key sequence.
    """

    method_a_keys = set(

        method_a[
            "entity_key"
        ]

    )


    method_b_keys = set(

        method_b[
            "entity_key"
        ]

    )


    matched_keys = sorted(

        method_a_keys
        &
        method_b_keys

    )


    only_a = (

        method_a_keys

        -

        method_b_keys

    )


    only_b = (

        method_b_keys

        -

        method_a_keys

    )


    print()


    print(
        f"Matched entities:      "
        f"{len(matched_keys):,}"
    )


    print(
        f"Method A only:         "
        f"{len(only_a):,}"
    )


    print(
        f"Method B only:         "
        f"{len(only_b):,}"
    )


    if expected_items is not None:

        if len(
            matched_keys
        ) != expected_items:

            print()


            print(
                "WARNING:"
            )


            print(

                f"Expected approximately "
                f"{expected_items:,} matched entities, "
                f"but found {len(matched_keys):,}."

            )


    # ========================================================
    # INDEX BY KEY
    # ========================================================

    method_a_indexed = (

        method_a
        .set_index(
            "entity_key",
            drop=False
        )

    )


    method_b_indexed = (

        method_b
        .set_index(
            "entity_key",
            drop=False
        )

    )


    matched_a = (

        method_a_indexed
        .loc[
            matched_keys
        ]
        .reset_index(
            drop=True
        )

    )


    matched_b = (

        method_b_indexed
        .loc[
            matched_keys
        ]
        .reset_index(
            drop=True
        )

    )


    # ========================================================
    # STRONG ALIGNMENT CHECK
    # ========================================================

    if not np.array_equal(

        matched_a[
            "entity_key"
        ].to_numpy(),

        matched_b[
            "entity_key"
        ].to_numpy()

    ):

        raise RuntimeError(

            "Matched Method A and Method B populations are "
            "not aligned."

        )


    if not np.array_equal(

        matched_a[
            "domain"
        ].to_numpy(),

        matched_b[
            "domain"
        ].to_numpy()

    ):

        raise RuntimeError(

            "Domain labels disagree between Method A and "
            "Method B for matched entities."

        )


    return (

        matched_a,

        matched_b,

        len(
            only_a
        ),

        len(
            only_b
        )

    )


# ============================================================
# EVALUATE ONE METHOD
# ============================================================


def evaluate_matched_atlas(

    df,

    atlas_id,

    atlas_label

):
    """
    Evaluate one matched-population 2D atlas.
    """

    review_flag_column = (

        "has_review_embedding"

        if "has_review_embedding"
        in df.columns

        else None

    )


    metrics, cluster_composition = (
        evaluate_atlas(

            df=
                df,

            atlas_id=
                atlas_id,

            atlas_label=
                atlas_label,

            x_column=
                "umap_x",

            y_column=
                "umap_y",

            cluster_column=
                "cluster",

            domain_column=
                "domain",

            review_flag_column=
                review_flag_column,

            feature_columns=
                None,

            source_feature_name=
                None,

            source_metric=
                "euclidean",

            k=
                K_NEIGHBORS,

            random_state=
                RANDOM_STATE

        )
    )


    return (

        metrics,

        cluster_composition

    )


# ============================================================
# BUILD METHOD COMPARISON
# ============================================================


def build_comparison_record(

    pair_id,

    pair_label,

    method_a_metrics,

    method_b_metrics

):
    """
    Build a compact Method A vs Method B comparison.

    Positive values for the explicitly named "improvement"
    fields indicate stronger cross-domain mixing under
    Method B.
    """

    a_accuracy = method_a_metrics[
        "domain_knn_balanced_accuracy_2d"
    ]


    b_accuracy = method_b_metrics[
        "domain_knn_balanced_accuracy_2d"
    ]


    a_entropy = method_a_metrics[
        "local_domain_entropy_2d"
    ]


    b_entropy = method_b_metrics[
        "local_domain_entropy_2d"
    ]


    a_cross_share = method_a_metrics[
        "cross_domain_neighbor_share_2d"
    ]


    b_cross_share = method_b_metrics[
        "cross_domain_neighbor_share_2d"
    ]


    a_silhouette = method_a_metrics[
        "domain_silhouette_2d"
    ]


    b_silhouette = method_b_metrics[
        "domain_silhouette_2d"
    ]


    return {

        "pair_id":
            pair_id,

        "pair_label":
            pair_label,

        "matched_items":
            method_a_metrics[
                "n_items"
            ],


        # ====================================================
        # DOMAIN KNN RECOVERABILITY
        # ====================================================

        "method_a_domain_knn_balanced_accuracy":
            a_accuracy,

        "method_b_domain_knn_balanced_accuracy":
            b_accuracy,

        # Lower accuracy = stronger mixing.
        "domain_knn_accuracy_reduction":
            (
                a_accuracy
                -
                b_accuracy
            ),


        # ====================================================
        # LOCAL DOMAIN ENTROPY
        # ====================================================

        "method_a_local_domain_entropy":
            a_entropy,

        "method_b_local_domain_entropy":
            b_entropy,

        # Higher entropy = stronger mixing.
        "local_domain_entropy_increase":
            (
                b_entropy
                -
                a_entropy
            ),


        # ====================================================
        # CROSS-DOMAIN NEIGHBOURS
        # ====================================================

        "method_a_cross_domain_neighbor_share":
            a_cross_share,

        "method_b_cross_domain_neighbor_share":
            b_cross_share,

        "cross_domain_neighbor_share_increase":
            (
                b_cross_share
                -
                a_cross_share
            ),


        # ====================================================
        # DOMAIN SILHOUETTE
        # ====================================================

        "method_a_domain_silhouette":
            a_silhouette,

        "method_b_domain_silhouette":
            b_silhouette,

        "domain_silhouette_change":
            (
                b_silhouette
                -
                a_silhouette
            ),


        # ====================================================
        # CLUSTER STRUCTURE
        # ====================================================

        "method_a_cluster_count":
            method_a_metrics[
                "cluster_count"
            ],

        "method_b_cluster_count":
            method_b_metrics[
                "cluster_count"
            ],

        "method_a_largest_cluster_share":
            method_a_metrics[
                "largest_cluster_share"
            ],

        "method_b_largest_cluster_share":
            method_b_metrics[
                "largest_cluster_share"
            ],

        "method_a_cluster_silhouette":
            method_a_metrics[
                "cluster_silhouette_2d"
            ],

        "method_b_cluster_silhouette":
            method_b_metrics[
                "cluster_silhouette_2d"
            ],


        # ====================================================
        # REVIEW-STATUS ARTEFACT
        # ====================================================

        "method_a_review_status_knn_accuracy":
            method_a_metrics.get(

                "review_status_knn_balanced_accuracy_2d",

                np.nan

            ),

        "method_b_review_status_knn_accuracy":
            method_b_metrics.get(

                "review_status_knn_balanced_accuracy_2d",

                np.nan

            ),

        "method_a_review_status_silhouette":
            method_a_metrics.get(

                "review_status_silhouette_2d",

                np.nan

            ),

        "method_b_review_status_silhouette":
            method_b_metrics.get(

                "review_status_silhouette_2d",

                np.nan

            ),

    }


# ============================================================
# PRINT METHOD RESULT
# ============================================================


def print_method_metrics(

    method_name,

    metrics

):

    print()


    print(
        method_name
    )


    print(
        "-" * len(
            method_name
        )
    )


    print(

        f"Domain kNN balanced accuracy: "
        f"{metrics['domain_knn_balanced_accuracy_2d']:.4f}"

    )


    print(

        f"Chance accuracy:              "
        f"{metrics['domain_chance_accuracy']:.4f}"

    )


    print(

        f"Local domain entropy:         "
        f"{metrics['local_domain_entropy_2d']:.4f}"

    )


    print(

        f"Cross-domain neighbour share: "
        f"{metrics['cross_domain_neighbor_share_2d']:.4f}"

    )


    print(

        f"Domain silhouette:            "
        f"{metrics['domain_silhouette_2d']:.4f}"

    )


    print()


    print(

        f"Clusters:                     "
        f"{metrics['cluster_count']}"

    )


    print(

        f"Largest cluster share:        "
        f"{metrics['largest_cluster_share']:.4f}"

    )


    print(

        f"Cluster silhouette:           "
        f"{metrics['cluster_silhouette_2d']:.4f}"

    )


    review_accuracy = metrics.get(

        "review_status_knn_balanced_accuracy_2d",

        np.nan

    )


    if np.isfinite(
        review_accuracy
    ):

        print()


        print(

            f"Review-status kNN accuracy:   "
            f"{review_accuracy:.4f}"

        )


        review_silhouette = metrics.get(

            "review_status_silhouette_2d",

            np.nan

        )


        if np.isfinite(
            review_silhouette
        ):

            print(

                f"Review-status silhouette:     "
                f"{review_silhouette:.4f}"

            )


# ============================================================
# MAIN
# ============================================================


def main():

    section(
        "MATCHED-POPULATION CROSS-DOMAIN EVALUATION"
    )


    print(
        f"k-neighbourhood size: "
        f"{K_NEIGHBORS}"
    )


    print(
        f"Random state: "
        f"{RANDOM_STATE}"
    )


    print()


    print(
        "Method A and Method B will be evaluated on exactly "
        "the same entity population."
    )


    all_metrics = []

    comparison_records = []

    population_records = []


    # ========================================================
    # EACH CROSS-DOMAIN PAIR
    # ========================================================

    for config in PAIR_CONFIGS:


        section(

            f"MATCHING: "
            f"{config['label']}"

        )


        # ====================================================
        # LOAD METHOD A
        # ====================================================

        method_a = load_cross_domain_atlas(

            path=
                config[
                    "method_a_path"
                ],

            method_label=
                "Method A"

        )


        print()


        # ====================================================
        # LOAD METHOD B
        # ====================================================

        method_b = load_cross_domain_atlas(

            path=
                config[
                    "method_b_path"
                ],

            method_label=
                "Method B"

        )


        # ====================================================
        # MATCH
        # ====================================================

        section(
            "CREATING IDENTICAL ENTITY POPULATION"
        )


        (

            matched_a,

            matched_b,

            only_a_count,

            only_b_count

        ) = create_matched_population(

            method_a=
                method_a,

            method_b=
                method_b,

            expected_items=
                config[
                    "expected_matched_items"
                ]

        )


        # ====================================================
        # DOMAIN COUNTS
        # ====================================================

        domain_counts = (

            matched_a[
                "domain"
            ]
            .value_counts()
            .sort_index()

        )


        print()


        print(
            "Matched entities by domain:"
        )


        print(
            domain_counts.to_string()
        )


        population_record = {

            "pair_id":
                config[
                    "pair_id"
                ],

            "pair_label":
                config[
                    "label"
                ],

            "method_a_full_items":
                int(
                    len(
                        method_a
                    )
                ),

            "method_b_full_items":
                int(
                    len(
                        method_b
                    )
                ),

            "matched_items":
                int(
                    len(
                        matched_a
                    )
                ),

            "method_a_only_items":
                int(
                    only_a_count
                ),

            "method_b_only_items":
                int(
                    only_b_count
                ),

        }


        for domain, count in (
            domain_counts.items()
        ):

            population_record[
                f"matched_{domain}"
            ] = int(
                count
            )


        population_records.append(
            population_record
        )


        # ====================================================
        # EVALUATE METHOD A
        # ====================================================

        section(
            "EVALUATING METHOD A — GENERAL SEMANTIC"
        )


        (

            method_a_metrics,

            method_a_clusters

        ) = evaluate_matched_atlas(

            df=
                matched_a,

            atlas_id=
                (
                    config[
                        "pair_id"
                    ]
                    +
                    "_method_a_matched"
                ),

            atlas_label=
                (
                    config[
                        "label"
                    ]
                    +
                    " — General Semantic"
                )

        )


        method_a_metrics[
            "pair_id"
        ] = config[
            "pair_id"
        ]


        method_a_metrics[
            "method"
        ] = "Method A"


        method_a_metrics[
            "population"
        ] = "matched"


        all_metrics.append(
            method_a_metrics
        )


        print_method_metrics(

            "Method A",

            method_a_metrics

        )


        # ====================================================
        # SAVE METHOD A CLUSTER COMPOSITION
        # ====================================================

        if not method_a_clusters.empty:

            method_a_clusters.to_csv(

                CLUSTER_OUTPUT_DIR

                /

                (
                    config[
                        "pair_id"
                    ]
                    +
                    "_method_a.csv"
                ),

                index=False

            )


        # ====================================================
        # EVALUATE METHOD B
        # ====================================================

        section(
            "EVALUATING METHOD B — SHARED FEEL"
        )


        (

            method_b_metrics,

            method_b_clusters

        ) = evaluate_matched_atlas(

            df=
                matched_b,

            atlas_id=
                (
                    config[
                        "pair_id"
                    ]
                    +
                    "_method_b_matched"
                ),

            atlas_label=
                (
                    config[
                        "label"
                    ]
                    +
                    " — Feel"
                )

        )


        method_b_metrics[
            "pair_id"
        ] = config[
            "pair_id"
        ]


        method_b_metrics[
            "method"
        ] = "Method B"


        method_b_metrics[
            "population"
        ] = "matched"


        all_metrics.append(
            method_b_metrics
        )


        print_method_metrics(

            "Method B",

            method_b_metrics

        )


        # ====================================================
        # SAVE METHOD B CLUSTER COMPOSITION
        # ====================================================

        if not method_b_clusters.empty:

            method_b_clusters.to_csv(

                CLUSTER_OUTPUT_DIR

                /

                (
                    config[
                        "pair_id"
                    ]
                    +
                    "_method_b.csv"
                ),

                index=False

            )


        # ====================================================
        # DIRECT COMPARISON
        # ====================================================

        comparison = build_comparison_record(

            pair_id=
                config[
                    "pair_id"
                ],

            pair_label=
                config[
                    "label"
                ],

            method_a_metrics=
                method_a_metrics,

            method_b_metrics=
                method_b_metrics

        )


        comparison_records.append(
            comparison
        )


        section(
            "METHOD B CHANGE RELATIVE TO METHOD A"
        )


        print(

            f"Domain kNN accuracy reduction:     "
            f"{comparison['domain_knn_accuracy_reduction']:+.4f}"

        )


        print(

            f"Local domain entropy increase:     "
            f"{comparison['local_domain_entropy_increase']:+.4f}"

        )


        print(

            f"Cross-domain neighbour increase:   "
            f"{comparison['cross_domain_neighbor_share_increase']:+.4f}"

        )


        print(

            f"Domain silhouette change:          "
            f"{comparison['domain_silhouette_change']:+.4f}"

        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    metrics_df = pd.DataFrame(
        all_metrics
    )


    comparison_df = pd.DataFrame(
        comparison_records
    )


    population_df = pd.DataFrame(
        population_records
    )


    metrics_df.to_csv(

        METRICS_OUTPUT,

        index=False

    )


    comparison_df.to_csv(

        COMPARISON_OUTPUT,

        index=False

    )


    population_df.to_csv(

        POPULATION_OUTPUT,

        index=False

    )


    # ========================================================
    # FINAL COMPARISON TABLE
    # ========================================================

    section(
        "MATCHED METHOD A VS METHOD B"
    )


    display_columns = [

        "pair_label",

        "matched_items",

        "method_a_domain_knn_balanced_accuracy",

        "method_b_domain_knn_balanced_accuracy",

        "method_a_local_domain_entropy",

        "method_b_local_domain_entropy",

        "method_a_cross_domain_neighbor_share",

        "method_b_cross_domain_neighbor_share",

        "method_a_domain_silhouette",

        "method_b_domain_silhouette",

    ]


    display = comparison_df[
        display_columns
    ].copy()


    numeric_columns = (

        display
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .columns
    )


    display[
        numeric_columns
    ] = (

        display[
            numeric_columns
        ]
        .round(
            4
        )

    )


    print(

        display.to_string(
            index=False
        )

    )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    section(
        "INTERPRETATION GUIDE"
    )


    print(
        "The matched population removes entity-coverage as a "
        "possible explanation for Method A / Method B "
        "differences."
    )


    print()


    print(
        "Evidence for stronger Method B mixing is:"
    )


    print(
        "  lower domain kNN balanced accuracy"
    )


    print(
        "  higher local domain entropy"
    )


    print(
        "  higher cross-domain neighbour share"
    )


    print(
        "  lower domain silhouette, where applicable"
    )


    print()


    print(
        "These metrics describe domain integration only."
    )


    print(
        "They do NOT by themselves prove that cross-domain "
        "neighbours are semantically or experientially "
        "meaningful."
    )


    print()


    print(
        "That final question is intentionally handled by the "
        "small qualitative nearest-neighbour sanity check, "
        "rather than by adding many more generic metrics."
    )


    # ========================================================
    # OUTPUTS
    # ========================================================

    section(
        "OUTPUTS"
    )


    print(
        "Matched metrics:"
    )


    print(
        METRICS_OUTPUT
    )


    print()


    print(
        "Direct Method A / Method B comparison:"
    )


    print(
        COMPARISON_OUTPUT
    )


    print()


    print(
        "Population matching summary:"
    )


    print(
        POPULATION_OUTPUT
    )


    print()


    print(
        "Cluster/domain composition:"
    )


    print(
        CLUSTER_OUTPUT_DIR
    )


    section(
        "MATCHED CROSS-DOMAIN EVALUATION COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()