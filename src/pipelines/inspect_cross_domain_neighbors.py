# src/pipelines/inspect_cross_domain_neighbors.py

# ============================================================
# QUALITATIVE CROSS-DOMAIN NEIGHBOUR INSPECTION
# ============================================================
#
# Purpose
# -------
#
# Inspect whether cross-domain proximity in the final atlases
# is actually meaningful.
#
#
# This script compares:
#
#     Method A — General Semantic
#     Method B — Shared Experiential / Feel
#
#
# for:
#
#     Movies + Music
#
#     Movies + Music + Restaurants
#
#
# MAIN QUESTION
# -------------
#
# When an item is close to an item from another domain,
# does that pairing appear semantically / experientially
# plausible?
#
#
# For Method B, the script additionally reports:
#
#     - 13D Feel-space distance
#     - strongest shared Feel characteristics
#
#
# IMPORTANT
# ---------
#
# This is intentionally a QUALITATIVE sanity check.
#
# It is NOT another benchmark.
#
# The quantitative evaluation has already established:
#
#     - mono-domain semantic coherence
#     - projection fidelity
#     - domain mixing
#     - matched Method A / Method B differences
#
# This script simply makes those results interpretable.
#
#
# OUTPUT
# ------
#
# data/processed/evaluation/cross_domain_qualitative/
#
#     cross_domain_neighbor_examples.csv
#     cross_domain_query_summary.csv
#
# ============================================================


from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


from src.atlas.cross_domain.feel_space.feel_anchors import (
    FEEL_DIMENSIONS,
    BIPOLAR_AXES,
    UNIPOLAR_AXES
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
    / "cross_domain_qualitative"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


NEIGHBOR_OUTPUT = (
    OUTPUT_DIR
    / "cross_domain_neighbor_examples.csv"
)


SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "cross_domain_query_summary.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Movies + Music:
#
# There is only one possible other domain, so show 5.
# ------------------------------------------------------------

MM_NEIGHBORS_PER_OTHER_DOMAIN = 5


# ------------------------------------------------------------
# Three domains:
#
# Show 3 neighbours from EACH other domain.
#
# Example:
#
#     Movie query:
#         3 Music
#         3 Restaurants
#
# This prevents the much larger Music population from
# completely dominating the qualitative output.
# ------------------------------------------------------------

MMR_NEIGHBORS_PER_OTHER_DOMAIN = 3


# ============================================================
# RECOGNISABLE QUERIES
# ============================================================
#
# These are not labels or ground truth.
#
# They are simply convenient examples for manual inspection.
#
# Queries are first resolved in Method B, then the exact same
# entity ID is used in Method A.
#
# ============================================================


MOVIES_MUSIC_QUERIES = {

    "movies": [

        "Toy Story",

        "Titanic",

        "La La Land",

        "Mad Max Fury Road",

        "Schindler's List",

    ],

    "music": [

        "Coldplay",

        "Radiohead",

        "Eminem",

        "Metallica",

        "Daft Punk",

    ],

}


MOVIES_MUSIC_RESTAURANT_QUERIES = {

    "movies": [

        "Toy Story",

        "La La Land",

        "Mad Max Fury Road",

    ],

    "music": [

        "Radiohead",

        "Eminem",

        "Daft Punk",

    ],

    "restaurants": [

        "Starbucks",

        "McDonald's",

        "The Cheesecake Factory",

        "Panda Express",

    ],

}


# ============================================================
# ATLAS PAIRS
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

        "queries":
            MOVIES_MUSIC_QUERIES,

        "neighbors_per_other_domain":
            MM_NEIGHBORS_PER_OTHER_DOMAIN,

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

        "queries":
            MOVIES_MUSIC_RESTAURANT_QUERIES,

        "neighbors_per_other_domain":
            MMR_NEIGHBORS_PER_OTHER_DOMAIN,

    },

]


# ============================================================
# PRINT SECTION
# ============================================================


def section(
    title
):

    print()

    print(
        "=" * 82
    )

    print(
        title
    )

    print(
        "=" * 82
    )

    print()


# ============================================================
# ID NORMALISATION
# ============================================================


def normalise_source_id(
    value
):

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
# NAME NORMALISATION
# ============================================================


def normalise_name(
    value
):
    """
    Forgiving matching representation used only to find the
    recognisable query examples.
    """

    if pd.isna(
        value
    ):

        return ""


    text = str(
        value
    ).strip()


    # --------------------------------------------------------
    # Remove trailing MovieLens-style year.
    #
    # Toy Story (1995)
    #     ->
    # Toy Story
    # --------------------------------------------------------

    text = re.sub(

        r"\s*\(\d{4}\)\s*$",

        "",

        text

    )


    text = unicodedata.normalize(

        "NFKD",

        text

    )


    text = "".join(

        character

        for character in text

        if not unicodedata.combining(
            character
        )

    )


    text = text.lower()


    text = re.sub(

        r"[^a-z0-9]+",

        " ",

        text

    )


    text = re.sub(

        r"\s+",

        " ",

        text

    )


    return text.strip()


# ============================================================
# COLUMN DETECTION
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
# LOAD CROSS-DOMAIN ATLAS
# ============================================================


def load_atlas(
    path,
    method_name
):
    """
    Load only the fields needed for qualitative neighbour
    inspection.

    Standard output fields:

        domain
        source_id
        entity_key
        display_name
        umap_x
        umap_y
        cluster

    Method B additionally keeps:

        feel_z_<dimension>
    """

    if not path.exists():

        raise FileNotFoundError(

            f"{method_name} atlas not found:\n"
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
    # BASIC COLUMNS
    # ========================================================

    domain_column = first_available_column(

        columns,

        [
            "domain",
            "source_domain"
        ]

    )


    source_id_column = first_available_column(

        columns,

        [
            "source_id",
            "movieId",
            "movie_id",
            "mbid",
            "business_id"
        ]

    )


    global_id_column = first_available_column(

        columns,

        [
            "id",
            "global_id"
        ]

    )


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


    cluster_column = first_available_column(

        columns,

        [
            "cluster",
            "cluster_id"
        ]

    )


    # ========================================================
    # POSSIBLE HUMAN-READABLE NAME FIELDS
    # ========================================================

    possible_name_columns = [

        column

        for column in [

            "title",

            "name",

            "artist_lastfm",

            "artist_mb",

            "artist_name",

        ]

        if column in columns

    ]


    # ========================================================
    # VALIDATE
    # ========================================================

    if domain_column is None:

        raise ValueError(

            f"No domain column found in:\n"
            f"{path}"

        )


    if (

        source_id_column is None

        and

        global_id_column is None

    ):

        raise ValueError(

            f"No usable source/global ID column found in:\n"
            f"{path}"

        )


    if (

        x_column is None

        or

        y_column is None

    ):

        raise ValueError(

            f"No UMAP coordinate columns found in:\n"
            f"{path}"

        )


    # ========================================================
    # FEEL COLUMNS
    # ========================================================

    feel_z_columns = [

        f"feel_z_{dimension}"

        for dimension in FEEL_DIMENSIONS

        if f"feel_z_{dimension}"
        in columns

    ]


    # ========================================================
    # LOAD ONLY NECESSARY COLUMNS
    # ========================================================

    usecols = {

        domain_column,

        x_column,

        y_column,

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


    for column in possible_name_columns:

        usecols.add(
            column
        )


    for column in feel_z_columns:

        usecols.add(
            column
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
    # DOMAIN
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
    # SOURCE ID + GLOBAL ENTITY KEY
    # ========================================================

    if source_id_column is not None:

        df[
            "source_id"
        ] = (

            df[
                source_id_column
            ]
            .apply(
                normalise_source_id
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

        global_ids = (

            df[
                global_id_column
            ]
            .fillna("")
            .astype(str)
            .str.strip()

        )


        keys = []


        source_ids = []


        for domain, global_id in zip(

            df[
                "domain"
            ],

            global_ids

        ):


            prefix = (

                domain
                +
                ":"

            )


            if global_id.startswith(
                prefix
            ):

                key = global_id


                source_id = global_id.split(

                    ":",

                    1

                )[
                    1
                ]


            else:

                source_id = normalise_source_id(
                    global_id
                )


                key = (

                    prefix
                    +
                    source_id

                )


            keys.append(
                key
            )


            source_ids.append(
                source_id
            )


        df[
            "source_id"
        ] = source_ids


        df[
            "entity_key"
        ] = keys


    # ========================================================
    # DISPLAY NAME
    # ========================================================

    display_names = []


    for _, row in df.iterrows():


        domain = row[
            "domain"
        ]


        if domain == "movies":

            candidates = [

                "title",

                "name"

            ]


        elif domain == "music":

            candidates = [

                "title",

                "artist_lastfm",

                "artist_mb",

                "artist_name",

                "name"

            ]


        elif domain == "restaurants":

            candidates = [

                "title",

                "name"

            ]


        else:

            candidates = (
                possible_name_columns
            )


        selected = ""


        for column in candidates:


            if column not in row.index:

                continue


            value = row[
                column
            ]


            if pd.isna(
                value
            ):

                continue


            value = str(
                value
            ).strip()


            if (

                value

                and

                value.lower()
                not in
                {
                    "nan",
                    "none"
                }

            ):

                selected = value

                break


        if not selected:

            selected = row[
                "entity_key"
            ]


        display_names.append(
            selected
        )


    df[
        "display_name"
    ] = display_names


    # ========================================================
    # COORDINATES
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
    # CLUSTER
    # ========================================================

    if cluster_column is not None:

        df[
            "cluster"
        ] = df[
            cluster_column
        ]


    else:

        df[
            "cluster"
        ] = np.nan


    # ========================================================
    # VALIDATE FEEL VALUES
    # ========================================================

    for column in feel_z_columns:

        df[
            column
        ] = pd.to_numeric(

            df[
                column
            ],

            errors=
                "coerce"

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

            f"{method_name} contains "
            f"{int(invalid_coordinates.sum()):,} invalid "
            f"coordinate rows."

        )


    if df[
        "entity_key"
    ].duplicated().any():

        raise ValueError(

            f"{method_name} contains duplicate entity keys."

        )


    print(

        f"{method_name}: "
        f"{len(df):,} entities"

    )


    print(

        df[
            "domain"
        ]
        .value_counts()
        .to_string()

    )


    return (
        df,
        feel_z_columns
    )


# ============================================================
# FIND RECOGNISABLE QUERY
# ============================================================


def resolve_query(
    df,
    domain,
    query
):
    """
    Resolve a human-friendly query.

    Preference:
        exact normalized name
        then partial normalized name
    """

    domain_df = df.loc[

        df[
            "domain"
        ]
        ==
        domain

    ].copy()


    domain_df[
        "_name_key"
    ] = (

        domain_df[
            "display_name"
        ]
        .apply(
            normalise_name
        )

    )


    query_key = normalise_name(
        query
    )


    # ========================================================
    # EXACT
    # ========================================================

    exact = domain_df.loc[

        domain_df[
            "_name_key"
        ]
        ==
        query_key

    ]


    if not exact.empty:

        return exact.iloc[
            0
        ]


    # ========================================================
    # PARTIAL
    # ========================================================

    partial = domain_df.loc[

        domain_df[
            "_name_key"
        ]
        .str.contains(

            query_key,

            regex=False,

            na=False

        )

    ]


    if not partial.empty:

        return partial.iloc[
            0
        ]


    return None


# ============================================================
# FEEL TRAIT FORMATTING
# ============================================================


def describe_feel_dimension(
    dimension,
    z_value
):
    """
    Translate a standardized Feel coordinate into a readable
    direction.
    """

    z_value = float(
        z_value
    )


    if dimension in BIPOLAR_AXES:

        axis = BIPOLAR_AXES[
            dimension
        ]


        if z_value >= 0:

            label = axis[
                "high_label"
            ]


        else:

            label = axis[
                "low_label"
            ]


        return (
            f"{label} ({z_value:+.2f}σ)"
        )


    # ========================================================
    # UNIPOLAR
    # ========================================================

    axis = UNIPOLAR_AXES[
        dimension
    ]


    label = axis.get(

        "label",

        dimension

    )


    if z_value >= 0:

        return (
            f"high {label} ({z_value:+.2f}σ)"
        )


    return (
        f"low {label} ({z_value:+.2f}σ)"
    )


# ============================================================
# TOP INDIVIDUAL FEEL TRAITS
# ============================================================


def top_feel_traits(
    row,
    n=4
):
    """
    Return the dimensions on which an item is furthest from
    the global Method-B average.
    """

    available = []


    for dimension in FEEL_DIMENSIONS:


        column = (
            f"feel_z_{dimension}"
        )


        if column not in row.index:

            continue


        value = row[
            column
        ]


        if not np.isfinite(
            value
        ):

            continue


        available.append(

            (

                dimension,

                float(
                    value
                )

            )

        )


    available.sort(

        key=lambda item:
            abs(
                item[1]
            ),

        reverse=True

    )


    selected = available[
        :n
    ]


    return "; ".join(

        describe_feel_dimension(

            dimension,

            value

        )

        for dimension, value
        in selected

    )


# ============================================================
# SHARED FEEL TRAITS
# ============================================================


def shared_feel_traits(
    query_row,
    neighbor_row,
    n=3
):
    """
    Identify Feel dimensions where both entities deviate in
    the same direction from the global average.

    A shared-trait strength is based on the smaller absolute
    deviation of the pair.

    This is a simple interpretability aid, NOT an evaluation
    metric.
    """

    candidates = []


    for dimension in FEEL_DIMENSIONS:


        column = (
            f"feel_z_{dimension}"
        )


        if (

            column not in query_row.index

            or

            column not in neighbor_row.index

        ):

            continue


        query_value = query_row[
            column
        ]


        neighbor_value = neighbor_row[
            column
        ]


        if (

            not np.isfinite(
                query_value
            )

            or

            not np.isfinite(
                neighbor_value
            )

        ):

            continue


        # ----------------------------------------------------
        # Same direction relative to global mean.
        # ----------------------------------------------------

        if (

            query_value == 0

            or

            neighbor_value == 0

            or

            np.sign(
                query_value
            )
            !=
            np.sign(
                neighbor_value
            )

        ):

            continue


        strength = min(

            abs(
                float(
                    query_value
                )
            ),

            abs(
                float(
                    neighbor_value
                )
            )

        )


        candidates.append(

            (

                dimension,

                float(
                    query_value
                ),

                float(
                    neighbor_value
                ),

                strength

            )

        )


    candidates.sort(

        key=lambda item:
            item[
                3
            ],

        reverse=True

    )


    selected = candidates[
        :n
    ]


    descriptions = []


    for (

        dimension,

        query_value,

        neighbor_value,

        _

    ) in selected:


        if dimension in BIPOLAR_AXES:

            axis = BIPOLAR_AXES[
                dimension
            ]


            if query_value >= 0:

                label = axis[
                    "high_label"
                ]


            else:

                label = axis[
                    "low_label"
                ]


        else:

            axis = UNIPOLAR_AXES[
                dimension
            ]


            base_label = axis.get(

                "label",

                dimension

            )


            if query_value >= 0:

                label = (
                    f"high {base_label}"
                )


            else:

                label = (
                    f"low {base_label}"
                )


        descriptions.append(

            f"{label} "
            f"({query_value:+.2f}σ / "
            f"{neighbor_value:+.2f}σ)"

        )


    if not descriptions:

        return ""


    return "; ".join(
        descriptions
    )


# ============================================================
# FEEL-SPACE DISTANCE
# ============================================================


def feel_distance(
    query_row,
    neighbor_row
):
    """
    Euclidean distance in the globally standardized 13D Feel
    representation used as input to Method-B UMAP.
    """

    columns = [

        f"feel_z_{dimension}"

        for dimension in FEEL_DIMENSIONS

    ]


    if not all(

        column in query_row.index

        and

        column in neighbor_row.index

        for column in columns

    ):

        return np.nan


    query_vector = np.asarray(

        [

            query_row[
                column
            ]

            for column in columns

        ],

        dtype=np.float32

    )


    neighbor_vector = np.asarray(

        [

            neighbor_row[
                column
            ]

            for column in columns

        ],

        dtype=np.float32

    )


    if (

        not np.isfinite(
            query_vector
        ).all()

        or

        not np.isfinite(
            neighbor_vector
        ).all()

    ):

        return np.nan


    return float(

        np.linalg.norm(

            query_vector
            -
            neighbor_vector

        )

    )


# ============================================================
# FIND CROSS-DOMAIN NEIGHBOURS
# ============================================================


def find_cross_domain_neighbors(

    df,

    query_row,

    neighbors_per_other_domain

):
    """
    Find nearest 2D neighbours separately for every OTHER
    domain.

    This is intentionally domain-balanced for inspection.

    2D distance values should NOT be compared numerically
    between Method A and Method B because separate UMAP runs
    have arbitrary coordinate scales.

    Rankings within each atlas are what matter.
    """

    query_domain = query_row[
        "domain"
    ]


    other_domains = sorted(

        [

            domain

            for domain in df[
                "domain"
            ].unique()

            if domain
            !=
            query_domain

        ]

    )


    results = []


    query_x = float(
        query_row[
            "umap_x"
        ]
    )


    query_y = float(
        query_row[
            "umap_y"
        ]
    )


    for target_domain in other_domains:


        candidates = df.loc[

            df[
                "domain"
            ]
            ==
            target_domain

        ].copy()


        delta_x = (

            candidates[
                "umap_x"
            ].to_numpy(
                dtype=np.float32
            )

            -

            query_x

        )


        delta_y = (

            candidates[
                "umap_y"
            ].to_numpy(
                dtype=np.float32
            )

            -

            query_y

        )


        distances = np.sqrt(

            delta_x ** 2

            +

            delta_y ** 2

        )


        n_select = min(

            neighbors_per_other_domain,

            len(
                candidates
            )

        )


        if n_select == 0:

            continue


        # ----------------------------------------------------
        # Efficient top-N selection without sorting every item.
        # ----------------------------------------------------

        if n_select < len(
            candidates
        ):

            selected_positions = np.argpartition(

                distances,

                n_select
                -
                1

            )[
                :n_select
            ]


            selected_positions = selected_positions[

                np.argsort(

                    distances[
                        selected_positions
                    ]

                )

            ]


        else:

            selected_positions = np.argsort(
                distances
            )


        for rank, candidate_position in enumerate(

            selected_positions,

            start=1

        ):


            neighbor_row = candidates.iloc[
                int(
                    candidate_position
                )
            ]


            results.append(

                (

                    target_domain,

                    rank,

                    float(
                        distances[
                            candidate_position
                        ]
                    ),

                    neighbor_row

                )

            )


    return results


# ============================================================
# PROCESS ONE QUERY
# ============================================================


def inspect_query(

    pair_config,

    method_a,

    method_b,

    query_domain,

    query_text

):
    """
    Resolve one query in Method B, then inspect exactly the
    same entity in Method A.
    """

    resolved_b = resolve_query(

        method_b,

        domain=
            query_domain,

        query=
            query_text

    )


    if resolved_b is None:

        print(

            f"[NOT FOUND] "
            f"{query_domain:<12} | "
            f"{query_text}"

        )


        return (
            [],
            None
        )


    entity_key = resolved_b[
        "entity_key"
    ]


    matched_a = method_a.loc[

        method_a[
            "entity_key"
        ]
        ==
        entity_key

    ]


    if matched_a.empty:

        print(

            f"[NOT IN METHOD A] "
            f"{query_domain:<12} | "
            f"{resolved_b['display_name']}"

        )


        return (
            [],
            None
        )


    resolved_a = matched_a.iloc[
        0
    ]


    # ========================================================
    # FIND NEIGHBOURS
    # ========================================================

    method_a_neighbors = (
        find_cross_domain_neighbors(

            method_a,

            query_row=
                resolved_a,

            neighbors_per_other_domain=
                pair_config[
                    "neighbors_per_other_domain"
                ]

        )
    )


    method_b_neighbors = (
        find_cross_domain_neighbors(

            method_b,

            query_row=
                resolved_b,

            neighbors_per_other_domain=
                pair_config[
                    "neighbors_per_other_domain"
                ]

        )
    )


    records = []


    # ========================================================
    # METHOD A RECORDS
    # ========================================================

    for (

        target_domain,

        rank,

        distance_2d,

        neighbor

    ) in method_a_neighbors:


        records.append({

            "pair_id":
                pair_config[
                    "pair_id"
                ],

            "pair_label":
                pair_config[
                    "label"
                ],

            "method":
                "Method A",

            "query_requested":
                query_text,

            "query_domain":
                query_domain,

            "query_name":
                resolved_a[
                    "display_name"
                ],

            "query_source_id":
                resolved_a[
                    "source_id"
                ],

            "query_entity_key":
                entity_key,

            "query_cluster":
                resolved_a[
                    "cluster"
                ],

            "neighbor_domain":
                target_domain,

            "neighbor_rank_within_domain":
                rank,

            "neighbor_name":
                neighbor[
                    "display_name"
                ],

            "neighbor_source_id":
                neighbor[
                    "source_id"
                ],

            "neighbor_entity_key":
                neighbor[
                    "entity_key"
                ],

            "neighbor_cluster":
                neighbor[
                    "cluster"
                ],

            "distance_2d":
                distance_2d,

            "feel_distance_13d":
                np.nan,

            "query_top_feel_traits":
                "",

            "neighbor_top_feel_traits":
                "",

            "shared_feel_traits":
                "",

        })


    # ========================================================
    # METHOD B RECORDS
    # ========================================================

    for (

        target_domain,

        rank,

        distance_2d,

        neighbor

    ) in method_b_neighbors:


        records.append({

            "pair_id":
                pair_config[
                    "pair_id"
                ],

            "pair_label":
                pair_config[
                    "label"
                ],

            "method":
                "Method B",

            "query_requested":
                query_text,

            "query_domain":
                query_domain,

            "query_name":
                resolved_b[
                    "display_name"
                ],

            "query_source_id":
                resolved_b[
                    "source_id"
                ],

            "query_entity_key":
                entity_key,

            "query_cluster":
                resolved_b[
                    "cluster"
                ],

            "neighbor_domain":
                target_domain,

            "neighbor_rank_within_domain":
                rank,

            "neighbor_name":
                neighbor[
                    "display_name"
                ],

            "neighbor_source_id":
                neighbor[
                    "source_id"
                ],

            "neighbor_entity_key":
                neighbor[
                    "entity_key"
                ],

            "neighbor_cluster":
                neighbor[
                    "cluster"
                ],

            "distance_2d":
                distance_2d,

            "feel_distance_13d":
                feel_distance(

                    resolved_b,

                    neighbor

                ),

            "query_top_feel_traits":
                top_feel_traits(
                    resolved_b
                ),

            "neighbor_top_feel_traits":
                top_feel_traits(
                    neighbor
                ),

            "shared_feel_traits":
                shared_feel_traits(

                    resolved_b,

                    neighbor

                ),

        })


    # ========================================================
    # METHOD A/B NEIGHBOUR OVERLAP
    # ========================================================

    method_a_keys = {

        (

            target_domain,

            neighbor[
                "entity_key"
            ]

        )

        for (

            target_domain,

            _,

            _,

            neighbor

        ) in method_a_neighbors

    }


    method_b_keys = {

        (

            target_domain,

            neighbor[
                "entity_key"
            ]

        )

        for (

            target_domain,

            _,

            _,

            neighbor

        ) in method_b_neighbors

    }


    overlap = len(

        method_a_keys

        &

        method_b_keys

    )


    union = len(

        method_a_keys

        |

        method_b_keys

    )


    overlap_jaccard = (

        overlap

        /

        union

        if union > 0

        else np.nan

    )


    summary = {

        "pair_id":
            pair_config[
                "pair_id"
            ],

        "pair_label":
            pair_config[
                "label"
            ],

        "query_requested":
            query_text,

        "query_domain":
            query_domain,

        "query_name":
            resolved_b[
                "display_name"
            ],

        "query_source_id":
            resolved_b[
                "source_id"
            ],

        "query_entity_key":
            entity_key,

        "method_a_neighbors":
            len(
                method_a_neighbors
            ),

        "method_b_neighbors":
            len(
                method_b_neighbors
            ),

        "method_a_b_neighbor_overlap":
            overlap,

        "method_a_b_neighbor_union":
            union,

        "method_a_b_neighbor_jaccard":
            overlap_jaccard,

        "method_b_query_top_feel_traits":
            top_feel_traits(
                resolved_b
            ),

    }


    return (
        records,
        summary
    )


# ============================================================
# PRINT ONE QUERY
# ============================================================


def print_query_results(
    query_records,
    query_summary
):

    if query_summary is None:

        return


    print()

    print(
        "-" * 82
    )


    print(

        f"{query_summary['query_name']} "
        f"[{query_summary['query_domain']}]"

    )


    print(
        "-" * 82
    )


    print()


    print(
        "Method B Feel profile:"
    )


    print(

        f"  "
        f"{query_summary['method_b_query_top_feel_traits']}"

    )


    for method in [

        "Method A",

        "Method B"

    ]:


        print()


        print(
            method
        )


        print(
            "~" * len(
                method
            )
        )


        method_records = [

            record

            for record in query_records

            if record[
                "method"
            ]
            ==
            method

        ]


        target_domains = sorted(

            {

                record[
                    "neighbor_domain"
                ]

                for record
                in method_records

            }

        )


        for target_domain in target_domains:


            print()


            print(

                f"  Nearest "
                f"{target_domain}:"

            )


            domain_records = [

                record

                for record
                in method_records

                if record[
                    "neighbor_domain"
                ]
                ==
                target_domain

            ]


            domain_records.sort(

                key=lambda record:
                    record[
                        "neighbor_rank_within_domain"
                    ]

            )


            for record in domain_records:


                line = (

                    f"    "
                    f"{record['neighbor_rank_within_domain']}. "
                    f"{record['neighbor_name']}"

                )


                print(
                    line
                )


                # ------------------------------------------------
                # For Method B, explain why the pairing may make
                # experiential sense.
                # ------------------------------------------------

                if (

                    method == "Method B"

                    and

                    record[
                        "shared_feel_traits"
                    ]

                ):

                    print(

                        f"       shared: "
                        f"{record['shared_feel_traits']}"

                    )


                    if np.isfinite(

                        record[
                            "feel_distance_13d"
                        ]

                    ):

                        print(

                            f"       13D Feel distance: "
                            f"{record['feel_distance_13d']:.3f}"

                        )


    print()


    print(

        f"Method A/B neighbour overlap: "
        f"{query_summary['method_a_b_neighbor_overlap']} "
        f"(Jaccard "
        f"{query_summary['method_a_b_neighbor_jaccard']:.3f})"

    )


# ============================================================
# MAIN
# ============================================================


def main():

    section(
        "QUALITATIVE CROSS-DOMAIN NEIGHBOUR INSPECTION"
    )


    print(
        "This is the final qualitative evaluation step."
    )


    print()


    print(
        "The goal is not to assign another performance score."
    )


    print(
        "The goal is to inspect whether actual cross-domain "
        "neighbourhoods make conceptual sense."
    )


    all_records = []

    all_summaries = []


    # ========================================================
    # EACH ATLAS PAIR
    # ========================================================

    for pair_config in PAIR_CONFIGS:


        section(

            f"LOADING: "
            f"{pair_config['label']}"

        )


        method_a, _ = load_atlas(

            pair_config[
                "method_a_path"
            ],

            method_name=
                "Method A"

        )


        print()


        method_b, feel_columns = load_atlas(

            pair_config[
                "method_b_path"
            ],

            method_name=
                "Method B"

        )


        expected_feel_columns = len(
            FEEL_DIMENSIONS
        )


        if len(
            feel_columns
        ) != expected_feel_columns:

            raise ValueError(

                "Method B atlas does not contain the complete "
                "standardized 13D Feel representation.\n\n"

                f"Expected: {expected_feel_columns}\n"
                f"Found:    {len(feel_columns)}"

            )


        # ====================================================
        # QUERIES
        # ====================================================

        section(

            f"INSPECTING: "
            f"{pair_config['label']}"

        )


        for domain, queries in (

            pair_config[
                "queries"
            ].items()

        ):


            for query_text in queries:


                (

                    query_records,

                    query_summary

                ) = inspect_query(

                    pair_config=
                        pair_config,

                    method_a=
                        method_a,

                    method_b=
                        method_b,

                    query_domain=
                        domain,

                    query_text=
                        query_text

                )


                if query_summary is None:

                    continue


                all_records.extend(
                    query_records
                )


                all_summaries.append(
                    query_summary
                )


                print_query_results(

                    query_records,

                    query_summary

                )


    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    neighbor_df = pd.DataFrame(
        all_records
    )


    summary_df = pd.DataFrame(
        all_summaries
    )


    neighbor_df.to_csv(

        NEIGHBOR_OUTPUT,

        index=False,

        encoding=
            "utf-8"

    )


    summary_df.to_csv(

        SUMMARY_OUTPUT,

        index=False,

        encoding=
            "utf-8"

    )


    # ========================================================
    # SUMMARY
    # ========================================================

    section(
        "QUALITATIVE INSPECTION SUMMARY"
    )


    print(
        f"Queries inspected: "
        f"{len(summary_df):,}"
    )


    print(
        f"Neighbour examples saved: "
        f"{len(neighbor_df):,}"
    )


    if not summary_df.empty:

        mean_overlap = (

            summary_df[
                "method_a_b_neighbor_jaccard"
            ]
            .mean()

        )


        print()


        print(

            f"Mean Method A/B neighbour-set Jaccard: "
            f"{mean_overlap:.4f}"

        )


        print()


        print(
            "Low overlap is not automatically good or bad."
        )


        print(
            "It simply indicates that the two representations "
            "produce meaningfully different cross-domain "
            "neighbourhoods."
        )


    # ========================================================
    # INTERPRETATION GUIDE
    # ========================================================

    section(
        "HOW TO INTERPRET THE OUTPUT"
    )


    print(
        "Method A"
    )


    print(
        "  Ask whether the cross-domain neighbours make sense "
        "under broad/general semantic similarity."
    )


    print()


    print(
        "Method B"
    )


    print(
        "  Ask whether the cross-domain neighbours share an "
        "experiential character even when their literal "
        "subject/category differs."
    )


    print()


    print(
        "Examples of plausible Method-B relationships might "
        "share characteristics such as:"
    )


    print(
        "  grand + serious + powerful"
    )


    print(
        "  warm + comforting + familiar"
    )


    print(
        "  playful + energetic + positive"
    )


    print(
        "  cold + unsettling + raw"
    )


    print()


    print(
        "The 13D Feel distance is useful for checking that a "
        "visually close Method-B pair is also reasonably close "
        "in the source Feel representation."
    )


    print()


    print(
        "Do NOT compare raw 2D distances numerically between "
        "Method A and Method B."
    )


    print(
        "Separate UMAP runs have arbitrary coordinate scales."
    )


    print()


    print(
        "The useful comparison is the neighbour identities and "
        "their conceptual plausibility."
    )


    # ========================================================
    # OUTPUTS
    # ========================================================

    section(
        "OUTPUTS"
    )


    print(
        "Detailed neighbour examples:"
    )


    print(
        NEIGHBOR_OUTPUT
    )


    print()


    print(
        "Query-level summary:"
    )


    print(
        SUMMARY_OUTPUT
    )


    section(
        "CROSS-DOMAIN QUALITATIVE EVALUATION COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()