# src/pipelines/evaluate_feel_space.py

# ============================================================
# EVALUATE SHARED EXPERIENTIAL / FEEL SPACE
# ============================================================
#
# Purpose
# -------
#
# Run the complete Method B semantic representation BEFORE
# any dimensionality reduction or atlas construction.
#
#
# Pipeline
# --------
#
# Movies:
#     base SBERT + review SBERT
#
# Music:
#     base SBERT + review SBERT
#
# Restaurants:
#     base SBERT + review SBERT
#
#             ↓
#
# shared fused 384D MiniLM representations
#
#             ↓
#
# 23 controlled experiential anchors
#
#             ↓
#
# 13D Shared Experiential / Feel Space
#
#             ↓
#
# diagnostics
#
#
# IMPORTANT
# ---------
#
# This script does NOT run UMAP.
#
# We first want to determine whether:
#
#     - the dimensions have useful variance
#     - dimensions are excessively correlated
#     - dimensions behave similarly across domains
#     - some dimensions accidentally encode domain identity
#     - semantic coverage is sufficient
#     - missing-modality patterns are reasonable
#
#
# Outputs
# -------
#
# data/processed/feel_space/
#     movies_music_restaurants/
#
#         feel_space_scores.npz
#         feel_space_scores.csv
#         feel_space_metadata.json
#
#
# data/processed/evaluation/feel_space/
#
#     global_dimension_summary.csv
#     domain_dimension_summary.csv
#     correlation_matrix.csv
#     high_correlations.csv
#     domain_effects.csv
#     modality_summary.csv
#     dimension_extremes.csv
#
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd


from src.atlas.cross_domain.feel_space.feel_anchors import (
    FEEL_DIMENSIONS,
    BIPOLAR_AXES,
    UNIPOLAR_AXES,
    print_feel_space_summary,
    validate_feel_anchors
)

from src.atlas.cross_domain.feel_space.feel_fusion import (
    fuse_multiple_domains
)

from src.atlas.cross_domain.feel_space.feel_projection import (
    project_to_feel_space
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# MOVIE EMBEDDINGS
# ============================================================

MOVIE_BASE_EMBEDDINGS = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "feel_space"
    / "movies"
    / "movie_base_semantic_embeddings.npz"
)


MOVIE_REVIEW_EMBEDDINGS = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "movies"
    / "movie_review_embeddings.npz"
)


# ============================================================
# MUSIC EMBEDDINGS
# ============================================================

MUSIC_BASE_EMBEDDINGS = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "feel_space"
    / "music"
    / "music_base_semantic_embeddings.npz"
)


MUSIC_REVIEW_EMBEDDINGS = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "music"
    / "music_review_embeddings.npz"
)


# ============================================================
# RESTAURANT EMBEDDINGS
# ============================================================

RESTAURANT_BASE_EMBEDDINGS = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "feel_space"
    / "restaurants"
    / "restaurant_base_semantic_embeddings.npz"
)


RESTAURANT_REVIEW_EMBEDDINGS = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "restaurants"
    / "restaurant_review_embeddings.npz"
)


# ============================================================
# FEEL-SPACE OUTPUTS
# ============================================================

FEEL_OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "feel_space"
    / "movies_music_restaurants"
)


FEEL_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FEEL_OUTPUT_NPZ = (
    FEEL_OUTPUT_DIR
    / "feel_space_scores.npz"
)


FEEL_OUTPUT_CSV = (
    FEEL_OUTPUT_DIR
    / "feel_space_scores.csv"
)


FEEL_OUTPUT_METADATA = (
    FEEL_OUTPUT_DIR
    / "feel_space_metadata.json"
)


# ============================================================
# EVALUATION OUTPUTS
# ============================================================

EVALUATION_DIR = (
    ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "feel_space"
)


EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


GLOBAL_SUMMARY_FILE = (
    EVALUATION_DIR
    / "global_dimension_summary.csv"
)


DOMAIN_SUMMARY_FILE = (
    EVALUATION_DIR
    / "domain_dimension_summary.csv"
)


CORRELATION_FILE = (
    EVALUATION_DIR
    / "correlation_matrix.csv"
)


HIGH_CORRELATIONS_FILE = (
    EVALUATION_DIR
    / "high_correlations.csv"
)


DOMAIN_EFFECTS_FILE = (
    EVALUATION_DIR
    / "domain_effects.csv"
)


MODALITY_SUMMARY_FILE = (
    EVALUATION_DIR
    / "modality_summary.csv"
)


EXTREMES_FILE = (
    EVALUATION_DIR
    / "dimension_extremes.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

REVIEW_SHARE = 0.50


DEVICE = "cuda"


ANCHOR_BATCH_SIZE = 32


# ------------------------------------------------------------
# Correlations above this absolute value are highlighted.
#
# This is a diagnostic threshold only.
#
# It does NOT automatically mean that one dimension should
# be removed.
# ------------------------------------------------------------

HIGH_CORRELATION_THRESHOLD = 0.85


# ------------------------------------------------------------
# Number of highest / lowest scoring entities saved for each
# dimension.
#
# These are useful for qualitative inspection.
# ------------------------------------------------------------

N_EXTREME_EXAMPLES = 5


# ============================================================
# PRINT SECTION
# ============================================================


def section(
    title
):

    print()

    print(
        "=" * 72
    )

    print(
        title
    )

    print(
        "=" * 72
    )

    print()


# ============================================================
# VALIDATE FILES
# ============================================================


def validate_required_files():

    required_files = [

        MOVIE_BASE_EMBEDDINGS,
        MOVIE_REVIEW_EMBEDDINGS,

        MUSIC_BASE_EMBEDDINGS,
        MUSIC_REVIEW_EMBEDDINGS,

        RESTAURANT_BASE_EMBEDDINGS,
        RESTAURANT_REVIEW_EMBEDDINGS,

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

            "Required Method B embedding files were not found:\n\n"
            f"{formatted}"

        )


# ============================================================
# MODALITY SUMMARY
# ============================================================


def build_modality_summary(
    entity_index
):
    """
    Summarize base/review modality availability by domain.
    """

    section(
        "SEMANTIC MODALITY COVERAGE"
    )


    records = []


    for domain in sorted(

        entity_index[
            "domain"
        ].unique()

    ):


        domain_df = entity_index.loc[

            entity_index[
                "domain"
            ]
            ==
            domain

        ]


        total = len(
            domain_df
        )


        for source in [

            "base_and_reviews",

            "base_only",

            "review_only",

            "neither",

        ]:


            count = int(

                (
                    domain_df[
                        "semantic_source"
                    ]
                    ==
                    source
                ).sum()

            )


            percentage = (

                (
                    count
                    /
                    total
                    *
                    100
                )

                if total > 0

                else 0.0

            )


            records.append({

                "domain":
                    domain,

                "semantic_source":
                    source,

                "count":
                    count,

                "percentage":
                    percentage,

            })


    summary = pd.DataFrame(
        records
    )


    pivot = summary.pivot(

        index=
            "domain",

        columns=
            "semantic_source",

        values=
            "count"

    ).fillna(0)


    print(
        pivot.to_string()
    )


    print()


    coverage = (

        entity_index
        .groupby(
            "domain"
        )[
            "is_semantically_defined"
        ]
        .agg(
            [
                "count",
                "sum"
            ]
        )

    )


    coverage[
        "coverage_pct"
    ] = (

        coverage[
            "sum"
        ]

        /

        coverage[
            "count"
        ]

        *

        100

    )


    print(
        "Semantic coverage:"
    )


    print(
        coverage.to_string()
    )


    summary.to_csv(

        MODALITY_SUMMARY_FILE,

        index=False

    )


    return summary


# ============================================================
# GLOBAL DIMENSION SUMMARY
# ============================================================


def build_global_dimension_summary(
    defined_df
):
    """
    Describe the natural distributions of all 13 dimensions.
    """

    section(
        "GLOBAL FEEL-DIMENSION DISTRIBUTIONS"
    )


    rows = []


    for dimension in FEEL_DIMENSIONS:


        values = pd.to_numeric(

            defined_df[
                dimension
            ],

            errors="coerce"

        ).dropna()


        rows.append({

            "dimension":
                dimension,

            "count":
                int(
                    values.count()
                ),

            "mean":
                float(
                    values.mean()
                ),

            "std":
                float(
                    values.std()
                ),

            "min":
                float(
                    values.min()
                ),

            "q25":
                float(
                    values.quantile(
                        0.25
                    )
                ),

            "median":
                float(
                    values.median()
                ),

            "q75":
                float(
                    values.quantile(
                        0.75
                    )
                ),

            "max":
                float(
                    values.max()
                ),

            "range":
                float(
                    values.max()
                    -
                    values.min()
                ),

        })


    summary = pd.DataFrame(
        rows
    )


    print(

        summary[
            [
                "dimension",
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
        ]
        .to_string(
            index=False
        )

    )


    summary.to_csv(

        GLOBAL_SUMMARY_FILE,

        index=False

    )


    # --------------------------------------------------------
    # Show dimensions with weakest / strongest variance.
    # --------------------------------------------------------

    print()

    print(
        "Lowest-variance dimensions:"
    )


    print(

        summary
        .sort_values(
            "std"
        )
        .head(5)[

            [
                "dimension",
                "std",
                "range"
            ]

        ]
        .to_string(
            index=False
        )

    )


    print()

    print(
        "Highest-variance dimensions:"
    )


    print(

        summary
        .sort_values(
            "std",
            ascending=False
        )
        .head(5)[

            [
                "dimension",
                "std",
                "range"
            ]

        ]
        .to_string(
            index=False
        )

    )


    return summary


# ============================================================
# DOMAIN-SPECIFIC DIMENSION SUMMARY
# ============================================================


def build_domain_dimension_summary(
    defined_df
):
    """
    Compare feel-score distributions between Movies, Music
    and Restaurants.
    """

    section(
        "FEEL-DIMENSION DISTRIBUTIONS BY DOMAIN"
    )


    rows = []


    for domain in sorted(

        defined_df[
            "domain"
        ].unique()

    ):


        domain_df = defined_df.loc[

            defined_df[
                "domain"
            ]
            ==
            domain

        ]


        for dimension in FEEL_DIMENSIONS:


            values = pd.to_numeric(

                domain_df[
                    dimension
                ],

                errors="coerce"

            ).dropna()


            rows.append({

                "domain":
                    domain,

                "dimension":
                    dimension,

                "count":
                    int(
                        values.count()
                    ),

                "mean":
                    float(
                        values.mean()
                    ),

                "std":
                    float(
                        values.std()
                    ),

                "min":
                    float(
                        values.min()
                    ),

                "q25":
                    float(
                        values.quantile(
                            0.25
                        )
                    ),

                "median":
                    float(
                        values.median()
                    ),

                "q75":
                    float(
                        values.quantile(
                            0.75
                        )
                    ),

                "max":
                    float(
                        values.max()
                    ),

            })


    summary = pd.DataFrame(
        rows
    )


    summary.to_csv(

        DOMAIN_SUMMARY_FILE,

        index=False

    )


    # ========================================================
    # PRINT DOMAIN MEANS
    # ========================================================

    means = summary.pivot(

        index=
            "dimension",

        columns=
            "domain",

        values=
            "mean"

    )


    print(
        "Mean score by domain:"
    )


    print(

        means
        .round(4)
        .to_string()

    )


    return summary


# ============================================================
# CORRELATION ANALYSIS
# ============================================================


def evaluate_dimension_correlations(
    defined_df
):
    """
    Measure redundancy between experiential dimensions.
    """

    section(
        "DIMENSION CORRELATION ANALYSIS"
    )


    correlation = (

        defined_df[
            FEEL_DIMENSIONS
        ]
        .corr(
            method="pearson"
        )

    )


    correlation.to_csv(
        CORRELATION_FILE
    )


    # ========================================================
    # FIND HIGH ABSOLUTE CORRELATIONS
    # ========================================================

    records = []


    for i, dimension_a in enumerate(

        FEEL_DIMENSIONS

    ):


        for j in range(

            i + 1,

            len(
                FEEL_DIMENSIONS
            )

        ):


            dimension_b = (
                FEEL_DIMENSIONS[
                    j
                ]
            )


            value = float(

                correlation.loc[

                    dimension_a,

                    dimension_b

                ]

            )


            if (

                abs(
                    value
                )

                >=

                HIGH_CORRELATION_THRESHOLD

            ):


                records.append({

                    "dimension_a":
                        dimension_a,

                    "dimension_b":
                        dimension_b,

                    "correlation":
                        value,

                    "absolute_correlation":
                        abs(
                            value
                        ),

                })


    high_correlations = pd.DataFrame(

        records,

        columns=[

            "dimension_a",

            "dimension_b",

            "correlation",

            "absolute_correlation",

        ]

    )


    if not high_correlations.empty:

        high_correlations = (

            high_correlations
            .sort_values(

                "absolute_correlation",

                ascending=False

            )

        )


    high_correlations.to_csv(

        HIGH_CORRELATIONS_FILE,

        index=False

    )


    print(
        f"High-correlation threshold: "
        f"|r| >= {HIGH_CORRELATION_THRESHOLD:.2f}"
    )


    print()


    if high_correlations.empty:

        print(
            "No dimension pairs exceeded the threshold."
        )


    else:

        print(
            high_correlations.to_string(
                index=False
            )
        )


    # ========================================================
    # STRONGEST CORRELATIONS REGARDLESS OF THRESHOLD
    # ========================================================

    all_pairs = []


    for i, dimension_a in enumerate(

        FEEL_DIMENSIONS

    ):


        for j in range(

            i + 1,

            len(
                FEEL_DIMENSIONS
            )

        ):


            dimension_b = (
                FEEL_DIMENSIONS[
                    j
                ]
            )


            value = float(

                correlation.loc[

                    dimension_a,

                    dimension_b

                ]

            )


            all_pairs.append({

                "dimension_a":
                    dimension_a,

                "dimension_b":
                    dimension_b,

                "correlation":
                    value,

                "absolute_correlation":
                    abs(
                        value
                    ),

            })


    all_pairs = pd.DataFrame(
        all_pairs
    )


    print()

    print(
        "10 strongest absolute correlations:"
    )


    print(

        all_pairs
        .sort_values(

            "absolute_correlation",

            ascending=False

        )
        .head(10)
        .to_string(
            index=False
        )

    )


    return (

        correlation,

        high_correlations

    )


# ============================================================
# DOMAIN EFFECT SIZE
# ============================================================


def compute_eta_squared(

    values,

    domains

):
    """
    Compute one-way eta squared:

        eta² = SS_between / SS_total

    This measures how much of a dimension's variance can be
    attributed to domain membership.

    Interpretation here is diagnostic only.

    Higher values mean that Movies / Music / Restaurants
    occupy systematically different regions of that axis.

    We do NOT treat this as a significance test.
    """

    values = np.asarray(

        values,

        dtype=np.float64

    )


    domains = np.asarray(
        domains
    )


    valid = np.isfinite(
        values
    )


    values = values[
        valid
    ]


    domains = domains[
        valid
    ]


    if len(
        values
    ) == 0:

        return np.nan


    overall_mean = float(
        values.mean()
    )


    ss_total = float(

        np.sum(

            (
                values
                -
                overall_mean
            )
            ** 2

        )

    )


    if ss_total <= 0:

        return 0.0


    ss_between = 0.0


    for domain in np.unique(
        domains
    ):


        group = values[

            domains
            ==
            domain

        ]


        if len(
            group
        ) == 0:

            continue


        group_mean = float(
            group.mean()
        )


        ss_between += (

            len(
                group
            )

            *

            (
                group_mean
                -
                overall_mean
            )
            ** 2

        )


    return float(

        ss_between
        /
        ss_total

    )


# ============================================================
# DOMAIN-SEPARATION DIAGNOSTICS
# ============================================================


def evaluate_domain_effects(
    defined_df
):
    """
    Determine whether individual feel dimensions strongly
    differ between domains.

    Metrics
    -------
    eta_squared:
        Fraction of score variance associated with domain.

    domain_mean_range:
        max(domain mean) - min(domain mean)

    standardized_mean_range:
        domain mean range divided by global standard deviation.
    """

    section(
        "DOMAIN EFFECTS WITHIN FEEL DIMENSIONS"
    )


    rows = []


    domains = (

        defined_df[
            "domain"
        ]
        .astype(str)
        .to_numpy()

    )


    for dimension in FEEL_DIMENSIONS:


        values = (

            pd.to_numeric(

                defined_df[
                    dimension
                ],

                errors="coerce"

            )
            .to_numpy(
                dtype=np.float64
            )

        )


        eta_squared = compute_eta_squared(

            values,

            domains

        )


        domain_means = (

            defined_df
            .groupby(
                "domain"
            )[
                dimension
            ]
            .mean()

        )


        domain_mean_range = float(

            domain_means.max()

            -

            domain_means.min()

        )


        global_std = float(

            np.nanstd(

                values,

                ddof=1

            )

        )


        standardized_mean_range = (

            domain_mean_range
            /
            global_std

            if global_std > 0

            else 0.0

        )


        row = {

            "dimension":
                dimension,

            "eta_squared":
                eta_squared,

            "domain_mean_range":
                domain_mean_range,

            "global_std":
                global_std,

            "standardized_mean_range":
                standardized_mean_range,

        }


        for domain, mean_value in (
            domain_means.items()
        ):

            row[
                f"mean_{domain}"
            ] = float(
                mean_value
            )


        rows.append(
            row
        )


    effects = pd.DataFrame(
        rows
    )


    effects = effects.sort_values(

        "eta_squared",

        ascending=False

    )


    effects.to_csv(

        DOMAIN_EFFECTS_FILE,

        index=False

    )


    print(

        effects[
            [
                "dimension",
                "eta_squared",
                "domain_mean_range",
                "standardized_mean_range",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )

    )


    print()

    print(
        "Dimensions with larger eta-squared values should be "
        "inspected carefully because they may encode domain "
        "identity rather than purely experiential character."
    )


    return effects


# ============================================================
# EXTREME ENTITY EXAMPLES
# ============================================================


def build_dimension_extremes(
    defined_df
):
    """
    Save the highest and lowest scoring entities for every
    experiential dimension.

    At this stage we have domain + source_id rather than
    human-readable titles.

    These IDs are primarily useful for later targeted
    qualitative inspection.
    """

    section(
        "DIMENSION EXTREMES"
    )


    rows = []


    metadata_columns = [

        column

        for column in [

            "domain",

            "source_id",

            "id",

            "semantic_source",

            "has_base_semantics",

            "has_review_semantics",

            "reviews_used_for_embedding",

        ]

        if column
        in defined_df.columns

    ]


    for dimension in FEEL_DIMENSIONS:


        ordered = (

            defined_df
            .sort_values(

                dimension,

                ascending=True

            )

        )


        lowest = ordered.head(
            N_EXTREME_EXAMPLES
        )


        highest = ordered.tail(
            N_EXTREME_EXAMPLES
        ).sort_values(

            dimension,

            ascending=False

        )


        # ====================================================
        # LOWEST
        # ====================================================

        for rank, (_, row) in enumerate(

            lowest.iterrows(),

            start=1

        ):


            record = {

                "dimension":
                    dimension,

                "direction":
                    "lowest",

                "rank":
                    rank,

                "score":
                    float(
                        row[
                            dimension
                        ]
                    ),

            }


            for column in metadata_columns:

                record[
                    column
                ] = row[
                    column
                ]


            rows.append(
                record
            )


        # ====================================================
        # HIGHEST
        # ====================================================

        for rank, (_, row) in enumerate(

            highest.iterrows(),

            start=1

        ):


            record = {

                "dimension":
                    dimension,

                "direction":
                    "highest",

                "rank":
                    rank,

                "score":
                    float(
                        row[
                            dimension
                        ]
                    ),

            }


            for column in metadata_columns:

                record[
                    column
                ] = row[
                    column
                ]


            rows.append(
                record
            )


    extremes = pd.DataFrame(
        rows
    )


    extremes.to_csv(

        EXTREMES_FILE,

        index=False

    )


    print(

        f"Saved top/bottom "
        f"{N_EXTREME_EXAMPLES} entities "
        f"for each of the "
        f"{len(FEEL_DIMENSIONS)} dimensions."

    )


    print()

    print(
        "Output:"
    )

    print(
        EXTREMES_FILE
    )


    return extremes


# ============================================================
# AXIS DEFINITION TABLE
# ============================================================


def print_axis_definitions():

    section(
        "EXPERIENTIAL AXIS DEFINITIONS"
    )


    for dimension in FEEL_DIMENSIONS:


        if dimension in BIPOLAR_AXES:

            axis = BIPOLAR_AXES[
                dimension
            ]


            print(

                f"{dimension:<12} : "
                f"{axis['high_label']} "
                f"<-> "
                f"{axis['low_label']}"

            )


        elif dimension in UNIPOLAR_AXES:

            axis = UNIPOLAR_AXES[
                dimension
            ]


            print(

                f"{dimension:<12} : "
                f"{axis['label']} "
                f"(unipolar)"

            )


# ============================================================
# FINAL DIAGNOSTIC SUMMARY
# ============================================================


def print_final_diagnostic_summary(

    feel_df,

    global_summary,

    high_correlations,

    domain_effects

):

    section(
        "FINAL FEEL-SPACE DIAGNOSTIC SUMMARY"
    )


    total = len(
        feel_df
    )


    defined = int(

        feel_df[
            "is_semantically_defined"
        ].sum()

    )


    undefined = (

        total
        -
        defined

    )


    print(
        f"Total entities:          "
        f"{total:,}"
    )


    print(
        f"Semantically defined:    "
        f"{defined:,}"
    )


    print(
        f"Semantically undefined:  "
        f"{undefined:,}"
    )


    print(
        f"Semantic coverage:       "
        f"{defined / total * 100:.2f}%"
    )


    print()


    print(
        f"Feel dimensions:         "
        f"{len(FEEL_DIMENSIONS)}"
    )


    print(
        f"High correlations "
        f"(|r| >= {HIGH_CORRELATION_THRESHOLD:.2f}): "
        f"{len(high_correlations):,}"
    )


    print()


    lowest_variance = (

        global_summary
        .sort_values(
            "std"
        )
        .head(3)

    )


    print(
        "Lowest-variance dimensions:"
    )


    for _, row in (
        lowest_variance.iterrows()
    ):

        print(

            f"  {row['dimension']:<12} "
            f"std={row['std']:.4f}"

        )


    print()


    strongest_domain_effects = (

        domain_effects
        .sort_values(

            "eta_squared",

            ascending=False

        )
        .head(5)

    )


    print(
        "Strongest domain-associated dimensions:"
    )


    for _, row in (
        strongest_domain_effects.iterrows()
    ):

        print(

            f"  {row['dimension']:<12} "
            f"eta²={row['eta_squared']:.4f} | "
            f"mean spread/std="
            f"{row['standardized_mean_range']:.3f}"

        )


    print()

    print(
        "IMPORTANT:"
    )


    print(

        "No dimension should be removed automatically from "
        "these diagnostics alone. The results should be "
        "interpreted together with semantic meaning and "
        "qualitative inspection."

    )


# ============================================================
# MAIN
# ============================================================


def main():

    section(
        "SHARED EXPERIENTIAL / FEEL SPACE EVALUATION"
    )


    # ========================================================
    # 1. CONFIGURATION VALIDATION
    # ========================================================

    validate_feel_anchors()


    validate_required_files()


    print_feel_space_summary()


    print_axis_definitions()


    print()

    print(
        f"Review share when base + reviews exist: "
        f"{REVIEW_SHARE:.2f}"
    )


    # ========================================================
    # 2. FUSE ALL THREE DOMAINS IN SHARED SBERT SPACE
    # ========================================================

    section(
        "BUILDING SHARED 384D ENTITY SEMANTICS"
    )


    (

        entity_embeddings,

        entity_index,

        domain_info

    ) = fuse_multiple_domains(

        domain_paths={

            "movies": {

                "base":
                    MOVIE_BASE_EMBEDDINGS,

                "reviews":
                    MOVIE_REVIEW_EMBEDDINGS,

            },


            "music": {

                "base":
                    MUSIC_BASE_EMBEDDINGS,

                "reviews":
                    MUSIC_REVIEW_EMBEDDINGS,

            },


            "restaurants": {

                "base":
                    RESTAURANT_BASE_EMBEDDINGS,

                "reviews":
                    RESTAURANT_REVIEW_EMBEDDINGS,

            },

        },

        review_share=
            REVIEW_SHARE

    )


    print(
        f"Combined semantic shape: "
        f"{entity_embeddings.shape}"
    )


    # ========================================================
    # 3. MODALITY ANALYSIS
    # ========================================================

    modality_summary = (
        build_modality_summary(

            entity_index

        )
    )


    # ========================================================
    # 4. PROJECT ONTO 23 ANCHORS -> 13 DIMENSIONS
    # ========================================================

    (

        feel_scores,

        feel_df,

        anchor_embeddings,

        anchor_df,

        projection_info

    ) = project_to_feel_space(

        entity_embeddings=
            entity_embeddings,

        entity_index=
            entity_index,

        device=
            DEVICE,

        batch_size=
            ANCHOR_BATCH_SIZE,

        output_npz=
            FEEL_OUTPUT_NPZ,

        output_csv=
            FEEL_OUTPUT_CSV,

        output_metadata_json=
            FEEL_OUTPUT_METADATA,

        source_name=
            "movies_music_restaurants"

    )


    # ========================================================
    # 5. KEEP ONLY SEMANTICALLY DEFINED ENTITIES
    # ========================================================

    defined_df = feel_df.loc[

        feel_df[
            "is_semantically_defined"
        ]

    ].copy()


    print()

    print(
        f"Defined entities used for diagnostics: "
        f"{len(defined_df):,}"
    )


    print(
        f"Undefined entities excluded from diagnostics: "
        f"{len(feel_df) - len(defined_df):,}"
    )


    # ========================================================
    # 6. GLOBAL DISTRIBUTIONS
    # ========================================================

    global_summary = (
        build_global_dimension_summary(

            defined_df

        )
    )


    # ========================================================
    # 7. DOMAIN DISTRIBUTIONS
    # ========================================================

    domain_summary = (
        build_domain_dimension_summary(

            defined_df

        )
    )


    # ========================================================
    # 8. CORRELATIONS
    # ========================================================

    (

        correlation,

        high_correlations

    ) = evaluate_dimension_correlations(

        defined_df

    )


    # ========================================================
    # 9. DOMAIN EFFECTS
    # ========================================================

    domain_effects = (
        evaluate_domain_effects(

            defined_df

        )
    )


    # ========================================================
    # 10. EXTREME EXAMPLES
    # ========================================================

    extremes = (
        build_dimension_extremes(

            defined_df

        )
    )


    # ========================================================
    # 11. FINAL SUMMARY
    # ========================================================

    print_final_diagnostic_summary(

        feel_df=
            feel_df,

        global_summary=
            global_summary,

        high_correlations=
            high_correlations,

        domain_effects=
            domain_effects

    )


    # ========================================================
    # OUTPUT LOCATIONS
    # ========================================================

    section(
        "OUTPUT FILES"
    )


    print(
        "Reusable Feel Space:"
    )


    print(
        f"  {FEEL_OUTPUT_NPZ}"
    )


    print(
        f"  {FEEL_OUTPUT_CSV}"
    )


    print(
        f"  {FEEL_OUTPUT_METADATA}"
    )


    print()


    print(
        "Diagnostics:"
    )


    for path in [

        GLOBAL_SUMMARY_FILE,

        DOMAIN_SUMMARY_FILE,

        CORRELATION_FILE,

        HIGH_CORRELATIONS_FILE,

        DOMAIN_EFFECTS_FILE,

        MODALITY_SUMMARY_FILE,

        EXTREMES_FILE,

    ]:

        print(
            f"  {path}"
        )


    section(
        "FEEL-SPACE EVALUATION COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()