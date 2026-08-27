# src/pipelines/evaluate_atlases.py

# ============================================================
# EVALUATE ALL ATLAS VARIANTS
# ============================================================
#
# Purpose
# -------
#
# Produce one comparable quantitative evaluation table for:
#
#     Movies
#     Music
#     Restaurants
#
#     Movies + Music
#         Method A — General Semantic
#
#     Movies + Music
#         Method B — Shared Feel
#
#     Movies + Music + Restaurants
#         Method A — General Semantic
#
#     Movies + Music + Restaurants
#         Method B — Shared Feel
#
#
# OUTPUT
# ------
#
# data/processed/evaluation/atlas_comparison/
#
#     atlas_metrics_summary.csv
#     cross_domain_comparison.csv
#
#     cluster_composition/
#         movies.csv
#         music.csv
#         ...
#
#
# IMPORTANT
# ---------
#
# If an older atlas CSV does not persist its original
# high-dimensional feature representation, trustworthiness
# and kNN-preservation are left blank.
#
# We do NOT reconstruct an approximate representation merely
# to obtain a number.
#
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd


from src.evaluation.atlas_metrics import (
    evaluate_atlas
)

from src.atlas.cross_domain.feel_space.feel_anchors import (
    FEEL_DIMENSIONS
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
    / "atlas_comparison"
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


SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "atlas_metrics_summary.csv"
)


CROSS_DOMAIN_OUTPUT = (
    OUTPUT_DIR
    / "cross_domain_comparison.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

K_NEIGHBORS = 15

RANDOM_STATE = 42


# ------------------------------------------------------------
# Standardized Method B feature columns.
#
# These were explicitly stored by the Feel atlas builders.
# ------------------------------------------------------------

FEEL_Z_COLUMNS = [

    f"feel_z_{dimension}"

    for dimension in FEEL_DIMENSIONS

]


# ============================================================
# ATLAS REGISTRY
# ============================================================
#
# Several candidate paths are supplied for older pipelines
# because project outputs have evolved over time.
#
# The first existing file is used.
#
# If your exact old CSV filenames differ, only this registry
# needs changing.
# ============================================================


ATLAS_CONFIGS = [


    # ============================================================
    # MONO-DOMAIN
    # ============================================================

    {

        "id":
            "movies",

        "label":
            "Movies",

        "method":
            "Mono-domain",

        "default_domain":
            "movies",

        "candidate_paths": [

            PROCESSED_DIR
            / "movie_map_v1.csv",

        ],

        "search_filenames": [

            "movie_map_v1.csv",

        ],

        "source_feature_columns":
            None,

        "source_feature_name":
            None,

        "source_metric":
            "cosine",

    },


    {

        "id":
            "music",

        "label":
            "Music",

        "method":
            "Mono-domain",

        "default_domain":
            "music",

        "candidate_paths": [

            PROCESSED_DIR
            / "music_map_v1.csv",

        ],

        "search_filenames": [

            "music_map_v1.csv",

        ],

        "source_feature_columns":
            None,

        "source_feature_name":
            None,

        "source_metric":
            "cosine",

    },


    {

        "id":
            "restaurants",

        "label":
            "Restaurants",

        "method":
            "Mono-domain",

        "default_domain":
            "restaurants",

        "candidate_paths": [

            PROCESSED_DIR
            / "restaurant_map_v1.csv",

        ],

        "search_filenames": [

            "restaurant_map_v1.csv",

        ],

        "source_feature_columns":
            None,

        "source_feature_name":
            None,

        "source_metric":
            "cosine",

    },


    # ============================================================
    # METHOD A — GENERAL SEMANTIC
    # ============================================================

    {

        "id":
            "movies_music",

        "label":
            "Movies + Music — General Semantic",

        "method":
            "Method A",

        "default_domain":
            None,

        "candidate_paths": [

            PROCESSED_DIR
            / "movies_music_atlas_v1.csv",

        ],

        "search_filenames": [

            "movies_music_atlas_v1.csv",

        ],

        "source_feature_columns":
            None,

        "source_feature_name":
            None,

        "source_metric":
            "cosine",

    },


    {

        "id":
            "movies_music_restaurants",

        "label":
            "Movies + Music + Restaurants — General Semantic",

        "method":
            "Method A",

        "default_domain":
            None,

        "candidate_paths": [

            PROCESSED_DIR
            / "movies_music_restaurants_atlas_v1.csv",

        ],

        "search_filenames": [

            "movies_music_restaurants_atlas_v1.csv",

        ],

        "source_feature_columns":
            None,

        "source_feature_name":
            None,

        "source_metric":
            "cosine",

    },

    # ========================================================
    # METHOD B — SHARED FEEL
    # ========================================================

    {

        "id":
            "movies_music_feel",

        "label":
            "Movies + Music — Feel",

        "method":
            "Method B",

        "default_domain":
            None,

        "candidate_paths": [

            PROCESSED_DIR
            / "movies_music_feel_atlas.csv",

        ],

        "search_filenames": [

            "movies_music_feel_atlas.csv",

        ],

        "source_feature_columns":
            FEEL_Z_COLUMNS,

        "source_feature_name":
            "13D globally-standardized Feel Space",

        "source_metric":
            "euclidean",

    },


    {

        "id":
            "movies_music_restaurants_feel",

        "label":
            "Movies + Music + Restaurants — Feel",

        "method":
            "Method B",

        "default_domain":
            None,

        "candidate_paths": [

            PROCESSED_DIR
            / "movies_music_restaurants_feel_atlas.csv",

        ],

        "search_filenames": [

            "movies_music_restaurants_feel_atlas.csv",

        ],

        "source_feature_columns":
            FEEL_Z_COLUMNS,

        "source_feature_name":
            "13D globally-standardized Feel Space",

        "source_metric":
            "euclidean",

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
# RESOLVE CSV
# ============================================================


def resolve_atlas_csv(
    config
):
    """
    Find the analysis CSV for one atlas.
    """

    # ========================================================
    # EXPLICIT CANDIDATES
    # ========================================================

    for path in config[
        "candidate_paths"
    ]:


        if path.exists():

            return path


    # ========================================================
    # RECURSIVE EXACT-FILENAME SEARCH
    # ========================================================

    matches = []


    for filename in config.get(

        "search_filenames",

        []

    ):


        matches.extend(

            PROCESSED_DIR.rglob(
                filename
            )

        )


    # --------------------------------------------------------
    # Avoid evaluation outputs accidentally matching.
    # --------------------------------------------------------

    matches = [

        path

        for path in matches

        if "evaluation"
        not in path.parts

    ]


    matches = list(
        dict.fromkeys(
            matches
        )
    )


    if len(
        matches
    ) == 1:

        return matches[
            0
        ]


    if len(
        matches
    ) > 1:

        # ----------------------------------------------------
        # Prefer the shortest / least nested path.
        # ----------------------------------------------------

        matches = sorted(

            matches,

            key=lambda path:
                (
                    len(
                        path.parts
                    ),

                    str(
                        path
                    )

                )

        )


        return matches[
            0
        ]


    return None


# ============================================================
# COLUMN DETECTION
# ============================================================


def first_available_column(

    columns,

    candidates

):
    """
    Return the first available column name.
    """

    columns = set(
        columns
    )


    for candidate in candidates:

        if candidate in columns:

            return candidate


    return None


# ============================================================
# AUTO-DETECT PERSISTED SOURCE FEATURES
# ============================================================


def auto_detect_feature_columns(
    columns
):
    """
    Detect persisted vector dimensions when older pipelines
    happened to save them.

    This is intentionally conservative.
    """

    prefixes = [

        "feel_z_",

        "embedding_",

        "fused_",

        "semantic_dim_",

        "svd_",

        "tfidf_svd_",

        "feature_",

    ]


    detected = []


    for column in columns:


        if any(

            str(
                column
            ).startswith(
                prefix
            )

            for prefix in prefixes

        ):

            detected.append(
                column
            )


    return detected


# ============================================================
# LOAD ONE ATLAS
# ============================================================


def load_atlas_dataframe(

    path,

    config

):
    """
    Read only the columns needed for evaluation.

    This avoids unnecessarily loading large semantic-text
    columns such as tags or Yelp tips.
    """

    header = pd.read_csv(

        path,

        nrows=0

    )


    columns = list(
        header.columns
    )


    # ========================================================
    # COORDINATES
    # ========================================================

    x_column = first_available_column(

        columns,

        [

            "umap_x",

            "x",

            "position_x",

        ]

    )


    y_column = first_available_column(

        columns,

        [

            "umap_y",

            "y",

            "position_y",

        ]

    )


    if (

        x_column is None

        or

        y_column is None

    ):

        raise ValueError(

            f"Could not identify 2D coordinate columns in:\n"
            f"{path}\n\n"

            f"Available columns:\n"
            f"{columns}"

        )


    # ========================================================
    # CLUSTER
    # ========================================================

    cluster_column = first_available_column(

        columns,

        [

            "cluster",

            "cluster_id",

        ]

    )


    # ========================================================
    # DOMAIN
    # ========================================================

    domain_column = first_available_column(

        columns,

        [

            "domain",

            "source_domain",

        ]

    )


    # ========================================================
    # REVIEW FLAG
    # ========================================================

    review_flag_column = first_available_column(

        columns,

        [

            "has_review_embedding",

            "has_review_semantics",

        ]

    )


    # ========================================================
    # SOURCE FEATURES
    # ========================================================

    requested_features = config.get(

        "source_feature_columns"

    )


    if requested_features is not None:

        if all(

            column in columns

            for column in requested_features

        ):

            feature_columns = list(
                requested_features
            )


        else:

            feature_columns = []


    else:

        feature_columns = (
            auto_detect_feature_columns(
                columns
            )
        )


    # ========================================================
    # LOAD ONLY REQUIRED COLUMNS
    # ========================================================

    use_columns = {

        x_column,

        y_column,

    }


    if cluster_column is not None:

        use_columns.add(
            cluster_column
        )


    if domain_column is not None:

        use_columns.add(
            domain_column
        )


    if review_flag_column is not None:

        use_columns.add(
            review_flag_column
        )


    use_columns.update(
        feature_columns
    )


    df = pd.read_csv(

        path,

        usecols=
            list(
                use_columns
            ),

        low_memory=
            False

    )


    # ========================================================
    # STANDARD COLUMN NAMES
    # ========================================================

    rename_map = {

        x_column:
            "umap_x",

        y_column:
            "umap_y",

    }


    if (

        cluster_column is not None

        and

        cluster_column
        !=
        "cluster"

    ):

        rename_map[
            cluster_column
        ] = "cluster"


    if (

        domain_column is not None

        and

        domain_column
        !=
        "domain"

    ):

        rename_map[
            domain_column
        ] = "domain"


    df = df.rename(
        columns=rename_map
    )


    # ========================================================
    # DEFAULT DOMAIN FOR MONO-DOMAIN ATLASES
    # ========================================================

    if "domain" not in df.columns:

        default_domain = config.get(
            "default_domain"
        )


        if default_domain is not None:

            df[
                "domain"
            ] = default_domain


    # ========================================================
    # FEATURE COLUMN NAMES AFTER RENAME
    # ========================================================

    actual_feature_columns = [

        column

        for column in feature_columns

        if column in df.columns

    ]


    # ========================================================
    # REVIEW COLUMN NAME
    # ========================================================

    actual_review_flag = (
        review_flag_column

        if review_flag_column
        in df.columns

        else None
    )


    return (

        df,

        actual_feature_columns,

        actual_review_flag

    )


# ============================================================
# FORMAT TABLE
# ============================================================


def print_comparison_table(
    results
):
    """
    Print the most useful metrics without flooding terminal.
    """

    display_columns = [

        "atlas_label",

        "n_items",

        "cluster_count",

        "largest_cluster_share",

        "cluster_silhouette_2d",

        "domain_knn_balanced_accuracy_2d",

        "local_domain_entropy_2d",

        "cross_domain_neighbor_share_2d",

        "domain_silhouette_2d",

        "trustworthiness",

        "knn_preservation",

    ]


    available = [

        column

        for column in display_columns

        if column in results.columns

    ]


    display = results[
        available
    ].copy()


    numeric_columns = display.select_dtypes(

        include=[
            np.number
        ]

    ).columns


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


# ============================================================
# MAIN
# ============================================================


def main():

    section(
        "EVALUATING ALL SEMANTIC ATLASES"
    )


    print(
        f"k-neighbourhood size: "
        f"{K_NEIGHBORS}"
    )


    print(
        f"Random state: "
        f"{RANDOM_STATE}"
    )


    results = []

    resolved_atlases = []

    missing_atlases = []


    # ========================================================
    # EVALUATE EACH ATLAS
    # ========================================================

    for config in ATLAS_CONFIGS:


        section(

            f"EVALUATING: "
            f"{config['label']}"

        )


        path = resolve_atlas_csv(
            config
        )


        if path is None:

            print(
                "Analysis CSV could not be located."
            )


            print(
                "Skipping this atlas for now."
            )


            missing_atlases.append(

                config[
                    "id"
                ]

            )


            continue


        print(
            f"Source:"
        )


        print(
            path
        )


        print()


        # ====================================================
        # LOAD
        # ====================================================

        (

            df,

            feature_columns,

            review_flag_column

        ) = load_atlas_dataframe(

            path,

            config

        )


        print(
            f"Rows loaded: "
            f"{len(df):,}"
        )


        print(
            f"Domains: "
            f"{df['domain'].nunique() if 'domain' in df.columns else 1}"
        )


        if feature_columns:

            print(

                f"Persisted source feature space: "
                f"{len(feature_columns)} dimensions"

            )


        else:

            print(

                "Persisted source feature space: "
                "not available"

            )


        if review_flag_column is not None:

            print(

                f"Review-status field: "
                f"{review_flag_column}"

            )


        # ====================================================
        # EVALUATE
        # ====================================================

        (

            metrics,

            cluster_composition

        ) = evaluate_atlas(

            df=
                df,

            atlas_id=
                config[
                    "id"
                ],

            atlas_label=
                config[
                    "label"
                ],

            review_flag_column=
                review_flag_column,

            feature_columns=
                (
                    feature_columns

                    if feature_columns

                    else None
                ),

            source_feature_name=
                config.get(
                    "source_feature_name"
                ),

            source_metric=
                config.get(

                    "source_metric",

                    "euclidean"

                ),

            k=
                K_NEIGHBORS,

            random_state=
                RANDOM_STATE

        )


        metrics[
            "method"
        ] = config[
            "method"
        ]


        metrics[
            "source_csv"
        ] = str(
            path
        )


        results.append(
            metrics
        )


        resolved_atlases.append(

            config[
                "id"
            ]

        )


        # ====================================================
        # SAVE CLUSTER COMPOSITION
        # ====================================================

        if not cluster_composition.empty:

            cluster_output = (

                CLUSTER_OUTPUT_DIR

                /

                f"{config['id']}.csv"

            )


            cluster_composition.to_csv(

                cluster_output,

                index=False

            )


        # ====================================================
        # SHORT RESULT
        # ====================================================

        print()


        print(
            f"Items: "
            f"{metrics['n_items']:,}"
        )


        print(
            f"Clusters: "
            f"{metrics['cluster_count']}"
        )


        if np.isfinite(

            metrics[
                "largest_cluster_share"
            ]

        ):

            print(

                f"Largest cluster: "
                f"{metrics['largest_cluster_share'] * 100:.2f}%"

            )


        if np.isfinite(

            metrics[
                "domain_knn_balanced_accuracy_2d"
            ]

        ):

            print(

                f"2D domain kNN balanced accuracy: "
                f"{metrics['domain_knn_balanced_accuracy_2d']:.4f}"

            )


            print(

                f"Chance: "
                f"{metrics['domain_chance_accuracy']:.4f}"

            )


            print(

                f"Cross-domain neighbour share: "
                f"{metrics['cross_domain_neighbor_share_2d']:.4f}"

            )


            print(

                f"Local domain entropy: "
                f"{metrics['local_domain_entropy_2d']:.4f}"

            )


        if np.isfinite(

            metrics[
                "trustworthiness"
            ]

        ):

            print(

                f"Trustworthiness: "
                f"{metrics['trustworthiness']:.4f}"

            )


            print(

                f"kNN preservation: "
                f"{metrics['knn_preservation']:.4f}"

            )


    # ========================================================
    # BUILD MASTER TABLE
    # ========================================================

    if not results:

        raise RuntimeError(

            "No atlas CSVs could be evaluated."

        )


    results_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # Preserve registry ordering.
    # --------------------------------------------------------

    atlas_order = {

        config[
            "id"
        ]:
            index

        for index, config
        in enumerate(
            ATLAS_CONFIGS
        )

    }


    results_df[
        "_order"
    ] = (

        results_df[
            "atlas_id"
        ]
        .map(
            atlas_order
        )

    )


    results_df = (

        results_df
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )

    )


    results_df.to_csv(

        SUMMARY_OUTPUT,

        index=False

    )


    # ========================================================
    # CROSS-DOMAIN COMPARISON
    # ========================================================

    cross_domain_ids = [

        "movies_music",

        "movies_music_feel",

        "movies_music_restaurants",

        "movies_music_restaurants_feel",

    ]


    cross_domain_df = results_df.loc[

        results_df[
            "atlas_id"
        ]
        .isin(
            cross_domain_ids
        )

    ].copy()


    cross_domain_df.to_csv(

        CROSS_DOMAIN_OUTPUT,

        index=False

    )


    # ========================================================
    # FINAL TABLE
    # ========================================================

    section(
        "ATLAS METRICS COMPARISON"
    )


    print_comparison_table(
        results_df
    )


    # ========================================================
    # INTERPRETATION GUIDE
    # ========================================================

    section(
        "INTERPRETATION GUIDE"
    )


    print(
        "Cluster metrics"
    )


    print(
        "  largest_cluster_share:"
    )


    print(
        "      Higher = more cluster imbalance."
    )


    print(
        "  cluster_size_entropy:"
    )


    print(
        "      Higher = cluster sizes are more evenly distributed."
    )


    print(
        "  cluster_silhouette_2d:"
    )


    print(
        "      Higher = more discrete separation between clusters."
    )


    print()


    print(
        "Cross-domain metrics"
    )


    print(
        "  domain_knn_balanced_accuracy_2d:"
    )


    print(
        "      Lower / closer to chance = stronger domain mixing."
    )


    print(
        "  local_domain_entropy_2d:"
    )


    print(
        "      Higher = locally more mixed neighbourhoods."
    )


    print(
        "  cross_domain_neighbor_share_2d:"
    )


    print(
        "      Higher = more neighbours come from other domains."
    )


    print(
        "  domain_silhouette_2d:"
    )


    print(
        "      Closer to 0 or negative = weaker domain separation."
    )


    print()


    print(
        "Projection metrics"
    )


    print(
        "  trustworthiness:"
    )


    print(
        "      Higher = fewer false neighbours introduced by 2D."
    )


    print(
        "  knn_preservation:"
    )


    print(
        "      Higher = more source-space neighbours survive in 2D."
    )


    print()


    print(
        "Review artefact metrics"
    )


    print(
        "  review_status_knn_balanced_accuracy_2d:"
    )


    print(
        "      Close to 0.5 = review availability is hard to recover."
    )


    print(
        "      High values = reviewed/non-reviewed entities occupy"
    )


    print(
        "      systematically different parts of the map."
    )


    # ========================================================
    # MISSING FILES
    # ========================================================

    if missing_atlases:

        section(
            "ATLASES NOT LOCATED"
        )


        print(
            "The following atlas CSVs were not found:"
        )


        for atlas_id in missing_atlases:

            print(
                f"  - {atlas_id}"
            )


        print()


        print(
            "Nothing is wrong with the metric code. "
            "Add the exact CSV path to that atlas's "
            "candidate_paths entry in ATLAS_CONFIGS."
        )


    # ========================================================
    # OUTPUTS
    # ========================================================

    section(
        "EVALUATION OUTPUTS"
    )


    print(
        f"Master table:"
    )


    print(
        SUMMARY_OUTPUT
    )


    print()


    print(
        f"Cross-domain comparison:"
    )


    print(
        CROSS_DOMAIN_OUTPUT
    )


    print()


    print(
        f"Cluster composition:"
    )


    print(
        CLUSTER_OUTPUT_DIR
    )


    section(
        "ATLAS EVALUATION COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()