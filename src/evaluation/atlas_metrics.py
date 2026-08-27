# src/evaluation/atlas_metrics.py

# ============================================================
# GENERIC ATLAS EVALUATION METRICS
# ============================================================
#
# This module contains reusable quantitative diagnostics for
# mono-domain and cross-domain atlases.
#
#
# METRIC GROUPS
# -------------
#
# 1. Cluster structure
#
#       number of clusters
#       largest-cluster share
#       normalized cluster-size entropy
#       2D cluster silhouette
#
#
# 2. Cross-domain mixing
#
#       kNN domain classification accuracy
#       balanced kNN domain accuracy
#       local domain entropy
#       cross-domain neighbour proportion
#       domain silhouette
#
#
# 3. Cluster/domain composition
#
#       cluster domain purity
#       cluster domain entropy
#
#
# 4. Review-status artefacts
#
#       kNN review-status recoverability
#       same-status neighbour share
#       review-status silhouette
#
#
# 5. Projection quality
#
#       trustworthiness
#       kNN neighbourhood preservation
#
#
# IMPORTANT
# ---------
#
# Projection-quality metrics require the original source
# feature representation.
#
# For the Feel atlases this is immediately available through:
#
#       feel_z_*
#
# For older atlases, if the original high-dimensional
# representation was not persisted in the analysis CSV,
# projection-quality metrics are returned as NaN rather than
# reconstructed approximately.
#
# ============================================================


import math

import numpy as np
import pandas as pd


from sklearn.manifold import (
    trustworthiness
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    silhouette_score
)

from sklearn.neighbors import (
    NearestNeighbors
)


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_RANDOM_STATE = 42

DEFAULT_K = 15


# ============================================================
# BASIC HELPERS
# ============================================================


def _normalise_boolean(
    series
):
    """
    Convert common CSV boolean representations to bool.
    """

    if series.dtype == bool:

        return series.to_numpy(
            dtype=bool
        )


    return (

        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes"
            ]
        )
        .to_numpy(
            dtype=bool
        )

    )


def _safe_float(
    value
):
    """
    Convert metric values to ordinary Python floats.
    """

    if value is None:

        return np.nan


    try:

        value = float(
            value
        )


    except (
        TypeError,
        ValueError
    ):

        return np.nan


    if not np.isfinite(
        value
    ):

        return np.nan


    return value


# ============================================================
# ENTROPY
# ============================================================


def normalized_entropy_from_counts(
    counts
):
    """
    Normalized Shannon entropy.

    Returns
    -------
    float

        0:
            all observations belong to one category

        1:
            categories are perfectly evenly represented
    """

    counts = np.asarray(

        counts,

        dtype=np.float64

    )


    counts = counts[
        counts > 0
    ]


    if len(
        counts
    ) <= 1:

        return 0.0


    probabilities = (

        counts

        /

        counts.sum()

    )


    entropy = float(

        -np.sum(

            probabilities

            *

            np.log(
                probabilities
            )

        )

    )


    maximum_entropy = math.log(

        len(
            counts
        )

    )


    if maximum_entropy <= 0:

        return 0.0


    return float(

        entropy

        /

        maximum_entropy

    )


# ============================================================
# SAMPLING
# ============================================================


def uniform_sample_indices(

    n_items,

    max_items,

    random_state=
        DEFAULT_RANDOM_STATE

):
    """
    Uniform sample without replacement.
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


def balanced_sample_indices(

    labels,

    max_per_class,

    random_state=
        DEFAULT_RANDOM_STATE

):
    """
    Sample up to max_per_class observations from each class.

    Used heavily for cross-domain evaluation because domain
    populations are very unequal.
    """

    labels = np.asarray(
        labels
    )


    rng = np.random.default_rng(
        random_state
    )


    selected = []


    for label in np.unique(
        labels
    ):


        candidates = np.flatnonzero(

            labels == label

        )


        n_select = min(

            len(
                candidates
            ),

            max_per_class

        )


        if n_select <= 0:

            continue


        if len(
            candidates
        ) == n_select:

            sample = candidates


        else:

            sample = rng.choice(

                candidates,

                size=
                    n_select,

                replace=
                    False

            )


        selected.extend(
            sample.tolist()
        )


    return np.asarray(

        selected,

        dtype=np.int64

    )


# ============================================================
# MAJORITY VOTE
# ============================================================


def majority_vote(
    labels
):
    """
    Return the most common label.
    """

    values, counts = np.unique(

        labels,

        return_counts=True

    )


    return values[

        np.argmax(
            counts
        )

    ]


# ============================================================
# GENERIC LOCAL LABEL MIXING
# ============================================================


def evaluate_label_mixing_matrix(

    matrix,

    labels,

    k=
        DEFAULT_K,

    metric=
        "euclidean",

    max_reference_size=
        30_000,

    query_per_class=
        1_500,

    silhouette_per_class=
        750,

    random_state=
        DEFAULT_RANDOM_STATE

):
    """
    Evaluate how recoverable a categorical label is from
    local geometry.

    The reference sample is balanced across labels.

    This is important for the current project because:

        Music       >> Movies >> Restaurants

    in population size.

    Metrics
    -------
    knn_accuracy:
        majority-neighbour label prediction accuracy.

    knn_balanced_accuracy:
        balanced version of the above.

    local_entropy:
        average normalized label entropy in local
        neighbourhoods.

    cross_label_neighbor_share:
        proportion of neighbours with a different label.

    silhouette:
        silhouette of the labels in the supplied space.

    chance_accuracy:
        1 / number_of_labels
    """

    matrix = np.asarray(

        matrix,

        dtype=np.float32

    )


    labels = np.asarray(

        labels

    ).astype(str)


    # ========================================================
    # VALID ROWS
    # ========================================================

    valid = (

        np.isfinite(
            matrix
        )
        .all(
            axis=1
        )

        &

        (
            labels != ""
        )

    )


    matrix = matrix[
        valid
    ]


    labels = labels[
        valid
    ]


    unique_labels = np.unique(
        labels
    )


    n_classes = len(
        unique_labels
    )


    if n_classes < 2:

        return {

            "knn_accuracy":
                np.nan,

            "knn_balanced_accuracy":
                np.nan,

            "chance_accuracy":
                np.nan,

            "local_entropy":
                np.nan,

            "cross_label_neighbor_share":
                np.nan,

            "silhouette":
                np.nan,

            "reference_items":
                len(
                    matrix
                ),

            "query_items":
                0,

        }


    # ========================================================
    # BALANCED REFERENCE SAMPLE
    # ========================================================

    reference_per_class = max(

        2,

        max_reference_size
        //
        n_classes

    )


    reference_indices = (
        balanced_sample_indices(

            labels,

            max_per_class=
                reference_per_class,

            random_state=
                random_state

        )
    )


    reference_matrix = matrix[
        reference_indices
    ]


    reference_labels = labels[
        reference_indices
    ]


    # ========================================================
    # BALANCED QUERY SAMPLE
    # ========================================================

    query_indices = (
        balanced_sample_indices(

            reference_labels,

            max_per_class=
                query_per_class,

            random_state=
                random_state
                +
                1

        )
    )


    if len(
        query_indices
    ) == 0:

        raise ValueError(

            "No valid query observations were available."

        )


    actual_k = min(

        int(
            k
        ),

        len(
            reference_matrix
        )
        -
        1

    )


    if actual_k < 1:

        raise ValueError(

            "Not enough observations for nearest-neighbour "
            "evaluation."

        )


    # ========================================================
    # NEAREST NEIGHBOURS
    # ========================================================

    nearest = NearestNeighbors(

        n_neighbors=
            actual_k
            +
            1,

        metric=
            metric,

        algorithm=
            "auto"

    )


    nearest.fit(
        reference_matrix
    )


    _, neighbor_indices = nearest.kneighbors(

        reference_matrix[
            query_indices
        ]

    )


    predictions = []

    true_labels = []

    local_entropies = []

    cross_label_shares = []


    # ========================================================
    # LOCAL LABEL METRICS
    # ========================================================

    for query_row, query_reference_index in enumerate(

        query_indices

    ):


        neighbors = neighbor_indices[
            query_row
        ]


        # ----------------------------------------------------
        # Remove the item itself.
        # ----------------------------------------------------

        neighbors = neighbors[

            neighbors
            !=
            query_reference_index

        ]


        neighbors = neighbors[
            :actual_k
        ]


        neighbor_labels = reference_labels[
            neighbors
        ]


        query_label = reference_labels[
            query_reference_index
        ]


        predictions.append(

            majority_vote(
                neighbor_labels
            )

        )


        true_labels.append(
            query_label
        )


        counts = [

            np.sum(
                neighbor_labels
                ==
                label
            )

            for label in unique_labels

        ]


        # ----------------------------------------------------
        # Normalize against the number of possible labels,
        # not only labels that happened to appear locally.
        # ----------------------------------------------------

        probabilities = (

            np.asarray(
                counts,
                dtype=np.float64
            )

            /

            len(
                neighbor_labels
            )

        )


        nonzero = probabilities[
            probabilities > 0
        ]


        entropy = float(

            -np.sum(

                nonzero

                *

                np.log(
                    nonzero
                )

            )

        )


        maximum_entropy = math.log(
            n_classes
        )


        local_entropies.append(

            entropy

            /

            maximum_entropy

        )


        cross_label_shares.append(

            float(

                np.mean(

                    neighbor_labels
                    !=
                    query_label

                )

            )

        )


    # ========================================================
    # CLASSIFICATION-LIKE METRICS
    # ========================================================

    true_labels = np.asarray(
        true_labels
    )


    predictions = np.asarray(
        predictions
    )


    knn_accuracy = accuracy_score(

        true_labels,

        predictions

    )


    knn_balanced_accuracy = (
        balanced_accuracy_score(

            true_labels,

            predictions

        )
    )


    # ========================================================
    # SILHOUETTE
    # ========================================================

    silhouette_indices = (
        balanced_sample_indices(

            reference_labels,

            max_per_class=
                silhouette_per_class,

            random_state=
                random_state
                +
                2

        )
    )


    silhouette_value = np.nan


    if (

        len(
            silhouette_indices
        )
        >
        n_classes

        and

        len(
            np.unique(

                reference_labels[
                    silhouette_indices
                ]

            )
        )
        >
        1

    ):


        silhouette_value = silhouette_score(

            reference_matrix[
                silhouette_indices
            ],

            reference_labels[
                silhouette_indices
            ],

            metric=
                metric

        )


    return {

        "knn_accuracy":
            _safe_float(
                knn_accuracy
            ),

        "knn_balanced_accuracy":
            _safe_float(
                knn_balanced_accuracy
            ),

        "chance_accuracy":
            float(
                1.0
                /
                n_classes
            ),

        "local_entropy":
            _safe_float(
                np.mean(
                    local_entropies
                )
            ),

        "cross_label_neighbor_share":
            _safe_float(
                np.mean(
                    cross_label_shares
                )
            ),

        "silhouette":
            _safe_float(
                silhouette_value
            ),

        "reference_items":
            int(
                len(
                    reference_matrix
                )
            ),

        "query_items":
            int(
                len(
                    query_indices
                )
            ),

    }


# ============================================================
# CLUSTER STRUCTURE
# ============================================================


def evaluate_cluster_structure(

    df,

    x_column=
        "umap_x",

    y_column=
        "umap_y",

    cluster_column=
        "cluster",

    silhouette_sample_size=
        5_000,

    random_state=
        DEFAULT_RANDOM_STATE

):
    """
    Evaluate how discrete the current cluster partition is.
    """

    if cluster_column not in df.columns:

        return {

            "cluster_count":
                np.nan,

            "largest_cluster_share":
                np.nan,

            "cluster_size_entropy":
                np.nan,

            "cluster_silhouette_2d":
                np.nan,

        }


    cluster_labels = (

        df[
            cluster_column
        ]
        .astype(str)
        .to_numpy()

    )


    counts = (

        pd.Series(
            cluster_labels
        )
        .value_counts()

    )


    n_clusters = len(
        counts
    )


    largest_cluster_share = float(

        counts.iloc[
            0
        ]

        /

        counts.sum()

    )


    cluster_entropy = (
        normalized_entropy_from_counts(

            counts.to_numpy()

        )
    )


    silhouette_value = np.nan


    if n_clusters >= 2:

        positions = (

            df[
                [
                    x_column,
                    y_column
                ]
            ]
            .to_numpy(
                dtype=np.float32
            )

        )


        sample_indices = (
            uniform_sample_indices(

                len(
                    positions
                ),

                max_items=
                    silhouette_sample_size,

                random_state=
                    random_state

            )
        )


        sample_positions = positions[
            sample_indices
        ]


        sample_labels = cluster_labels[
            sample_indices
        ]


        if (

            len(
                np.unique(
                    sample_labels
                )
            )
            >=
            2

            and

            len(
                np.unique(
                    sample_labels
                )
            )
            <
            len(
                sample_labels
            )

        ):


            silhouette_value = silhouette_score(

                sample_positions,

                sample_labels,

                metric=
                    "euclidean"

            )


    return {

        "cluster_count":
            int(
                n_clusters
            ),

        "largest_cluster_share":
            _safe_float(
                largest_cluster_share
            ),

        "cluster_size_entropy":
            _safe_float(
                cluster_entropy
            ),

        "cluster_silhouette_2d":
            _safe_float(
                silhouette_value
            ),

    }


# ============================================================
# CLUSTER / DOMAIN COMPOSITION
# ============================================================


def build_cluster_domain_composition(

    df,

    cluster_column=
        "cluster",

    domain_column=
        "domain"

):
    """
    Detailed domain composition of each cluster.
    """

    if (

        cluster_column not in df.columns

        or

        domain_column not in df.columns

    ):

        return pd.DataFrame()


    domains = sorted(

        df[
            domain_column
        ]
        .astype(str)
        .unique()

    )


    if len(
        domains
    ) < 2:

        return pd.DataFrame()


    records = []


    for cluster, cluster_df in (
        df.groupby(
            cluster_column
        )
    ):


        counts = (

            cluster_df[
                domain_column
            ]
            .astype(str)
            .value_counts()

        )


        total = len(
            cluster_df
        )


        probabilities = np.asarray(

            [

                counts.get(
                    domain,
                    0
                )

                /
                total

                for domain in domains

            ],

            dtype=np.float64

        )


        nonzero = probabilities[
            probabilities > 0
        ]


        entropy = float(

            -np.sum(

                nonzero

                *

                np.log(
                    nonzero
                )

            )

        )


        entropy /= math.log(
            len(
                domains
            )
        )


        purity = float(

            counts.max()

            /

            total

        )


        record = {

            "cluster":
                cluster,

            "cluster_size":
                int(
                    total
                ),

            "domain_purity":
                purity,

            "domain_entropy":
                entropy,

        }


        for domain in domains:

            count = int(

                counts.get(
                    domain,
                    0
                )

            )


            record[
                f"count_{domain}"
            ] = count


            record[
                f"share_{domain}"
            ] = (

                count

                /

                total

            )


        records.append(
            record
        )


    return pd.DataFrame(
        records
    )


def summarize_cluster_domain_composition(
    composition
):
    """
    Population-weighted cluster/domain statistics.
    """

    if composition.empty:

        return {

            "cluster_domain_purity":
                np.nan,

            "cluster_domain_entropy":
                np.nan,

        }


    weights = (

        composition[
            "cluster_size"
        ]
        .to_numpy(
            dtype=np.float64
        )

    )


    weights /= weights.sum()


    purity = float(

        np.sum(

            weights

            *

            composition[
                "domain_purity"
            ]
            .to_numpy(
                dtype=np.float64
            )

        )

    )


    entropy = float(

        np.sum(

            weights

            *

            composition[
                "domain_entropy"
            ]
            .to_numpy(
                dtype=np.float64
            )

        )

    )


    return {

        "cluster_domain_purity":
            purity,

        "cluster_domain_entropy":
            entropy,

    }


# ============================================================
# PROJECTION QUALITY
# ============================================================


def evaluate_projection_quality(

    source_matrix,

    low_dimensional_matrix,

    k=
        DEFAULT_K,

    source_metric=
        "euclidean",

    sample_size=
        3_000,

    random_state=
        DEFAULT_RANDOM_STATE

):
    """
    Compare a high-dimensional source representation with its
    2D atlas projection.

    Metrics
    -------
    trustworthiness:
        Measures whether 2D introduces false neighbours.

    knn_preservation:
        Average fraction of source-space kNNs retained in 2D.

    Notes
    -----
    For computational practicality this uses a deterministic
    sample and computes neighbourhoods inside that sample.
    """

    source_matrix = np.asarray(

        source_matrix,

        dtype=np.float32

    )


    low_dimensional_matrix = np.asarray(

        low_dimensional_matrix,

        dtype=np.float32

    )


    if (

        len(
            source_matrix
        )

        !=

        len(
            low_dimensional_matrix
        )

    ):

        raise ValueError(

            "Source and 2D representations are not aligned."

        )


    valid = (

        np.isfinite(
            source_matrix
        )
        .all(
            axis=1
        )

        &

        np.isfinite(
            low_dimensional_matrix
        )
        .all(
            axis=1
        )

    )


    source_matrix = source_matrix[
        valid
    ]


    low_dimensional_matrix = (
        low_dimensional_matrix[
            valid
        ]
    )


    if len(
        source_matrix
    ) < 3:

        return {

            "trustworthiness":
                np.nan,

            "knn_preservation":
                np.nan,

            "projection_sample_size":
                len(
                    source_matrix
                ),

        }


    indices = uniform_sample_indices(

        len(
            source_matrix
        ),

        max_items=
            sample_size,

        random_state=
            random_state

    )


    source_sample = source_matrix[
        indices
    ]


    low_sample = low_dimensional_matrix[
        indices
    ]


    actual_k = min(

        int(
            k
        ),

        max(

            1,

            (
                len(
                    source_sample
                )
                -
                1
            )
            //
            2

        )

    )


    # ========================================================
    # TRUSTWORTHINESS
    # ========================================================

    trust = trustworthiness(

        source_sample,

        low_sample,

        n_neighbors=
            actual_k,

        metric=
            source_metric

    )


    # ========================================================
    # KNN PRESERVATION
    # ========================================================

    source_nn = NearestNeighbors(

        n_neighbors=
            actual_k
            +
            1,

        metric=
            source_metric

    )


    low_nn = NearestNeighbors(

        n_neighbors=
            actual_k
            +
            1,

        metric=
            "euclidean"

    )


    source_nn.fit(
        source_sample
    )


    low_nn.fit(
        low_sample
    )


    source_neighbors = (

        source_nn
        .kneighbors(

            source_sample,

            return_distance=False

        )[
            :,
            1:
        ]

    )


    low_neighbors = (

        low_nn
        .kneighbors(

            low_sample,

            return_distance=False

        )[
            :,
            1:
        ]

    )


    preservation = []


    for index in range(

        len(
            source_sample
        )

    ):


        source_set = set(

            source_neighbors[
                index
            ].tolist()

        )


        low_set = set(

            low_neighbors[
                index
            ].tolist()

        )


        preservation.append(

            len(

                source_set
                &
                low_set

            )

            /

            actual_k

        )


    return {

        "trustworthiness":
            _safe_float(
                trust
            ),

        "knn_preservation":
            _safe_float(
                np.mean(
                    preservation
                )
            ),

        "projection_sample_size":
            int(
                len(
                    source_sample
                )
            ),

    }


# ============================================================
# COMPLETE ATLAS EVALUATION
# ============================================================


def evaluate_atlas(

    df,

    atlas_id,

    atlas_label=None,

    x_column=
        "umap_x",

    y_column=
        "umap_y",

    cluster_column=
        "cluster",

    domain_column=
        "domain",

    review_flag_column=None,

    feature_columns=None,

    source_feature_name=None,

    source_metric=
        "euclidean",

    k=
        DEFAULT_K,

    random_state=
        DEFAULT_RANDOM_STATE

):
    """
    Evaluate one atlas.

    Returns
    -------
    metrics:
        dict

    cluster_domain_composition:
        DataFrame
    """

    if atlas_label is None:

        atlas_label = atlas_id


    # ========================================================
    # VALIDATE POSITIONS
    # ========================================================

    for column in [

        x_column,

        y_column

    ]:

        if column not in df.columns:

            raise ValueError(

                f"Atlas '{atlas_id}' is missing coordinate "
                f"column '{column}'."

            )


    working = df.copy()


    working[
        x_column
    ] = pd.to_numeric(

        working[
            x_column
        ],

        errors="coerce"

    )


    working[
        y_column
    ] = pd.to_numeric(

        working[
            y_column
        ],

        errors="coerce"

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


    # ========================================================
    # BASE RESULT
    # ========================================================

    metrics = {

        "atlas_id":
            atlas_id,

        "atlas_label":
            atlas_label,

        "n_items":
            int(
                len(
                    working
                )
            ),

        "n_domains":
            (
                int(

                    working[
                        domain_column
                    ]
                    .astype(str)
                    .nunique()

                )

                if domain_column
                in working.columns

                else 1
            ),

        "knn_k":
            int(
                k
            ),

    }


    # ========================================================
    # CLUSTER STRUCTURE
    # ========================================================

    metrics.update(

        evaluate_cluster_structure(

            working,

            x_column=
                x_column,

            y_column=
                y_column,

            cluster_column=
                cluster_column,

            random_state=
                random_state

        )

    )


    # ========================================================
    # DOMAIN MIXING IN 2D
    # ========================================================

    if (

        domain_column in working.columns

        and

        working[
            domain_column
        ]
        .astype(str)
        .nunique()
        >=
        2

    ):


        domain_metrics = (
            evaluate_label_mixing_matrix(

                positions,

                working[
                    domain_column
                ]
                .astype(str)
                .to_numpy(),

                k=
                    k,

                metric=
                    "euclidean",

                max_reference_size=
                    30_000,

                query_per_class=
                    1_500,

                silhouette_per_class=
                    750,

                random_state=
                    random_state

            )
        )


        metrics.update({

            "domain_knn_accuracy_2d":
                domain_metrics[
                    "knn_accuracy"
                ],

            "domain_knn_balanced_accuracy_2d":
                domain_metrics[
                    "knn_balanced_accuracy"
                ],

            "domain_chance_accuracy":
                domain_metrics[
                    "chance_accuracy"
                ],

            "local_domain_entropy_2d":
                domain_metrics[
                    "local_entropy"
                ],

            "cross_domain_neighbor_share_2d":
                domain_metrics[
                    "cross_label_neighbor_share"
                ],

            "domain_silhouette_2d":
                domain_metrics[
                    "silhouette"
                ],

        })


    else:

        metrics.update({

            "domain_knn_accuracy_2d":
                np.nan,

            "domain_knn_balanced_accuracy_2d":
                np.nan,

            "domain_chance_accuracy":
                np.nan,

            "local_domain_entropy_2d":
                np.nan,

            "cross_domain_neighbor_share_2d":
                np.nan,

            "domain_silhouette_2d":
                np.nan,

        })


    # ========================================================
    # CLUSTER / DOMAIN COMPOSITION
    # ========================================================

    cluster_domain_composition = (
        build_cluster_domain_composition(

            working,

            cluster_column=
                cluster_column,

            domain_column=
                domain_column

        )
    )


    metrics.update(

        summarize_cluster_domain_composition(

            cluster_domain_composition

        )

    )


    # ========================================================
    # REVIEW-STATUS GEOMETRY
    # ========================================================

    if (

        review_flag_column is not None

        and

        review_flag_column
        in working.columns

    ):


        review_flags = _normalise_boolean(

            working[
                review_flag_column
            ]

        )


        if len(
            np.unique(
                review_flags
            )
        ) >= 2:


            review_labels = np.where(

                review_flags,

                "review",

                "no_review"

            )


            review_metrics = (
                evaluate_label_mixing_matrix(

                    positions,

                    review_labels,

                    k=
                        k,

                    metric=
                        "euclidean",

                    max_reference_size=
                        20_000,

                    query_per_class=
                        1_000,

                    silhouette_per_class=
                        750,

                    random_state=
                        random_state
                        +
                        10

                )
            )


            metrics.update({

                "review_status_knn_balanced_accuracy_2d":
                    review_metrics[
                        "knn_balanced_accuracy"
                    ],

                "review_status_same_neighbor_share_2d":
                    (
                        1.0

                        -

                        review_metrics[
                            "cross_label_neighbor_share"
                        ]
                    ),

                "review_status_silhouette_2d":
                    review_metrics[
                        "silhouette"
                    ],

            })


        else:

            metrics.update({

                "review_status_knn_balanced_accuracy_2d":
                    np.nan,

                "review_status_same_neighbor_share_2d":
                    np.nan,

                "review_status_silhouette_2d":
                    np.nan,

            })


    else:

        metrics.update({

            "review_status_knn_balanced_accuracy_2d":
                np.nan,

            "review_status_same_neighbor_share_2d":
                np.nan,

            "review_status_silhouette_2d":
                np.nan,

        })


    # ========================================================
    # SOURCE-SPACE / PROJECTION METRICS
    # ========================================================

    usable_feature_columns = []


    if feature_columns is not None:

        usable_feature_columns = [

            column

            for column in feature_columns

            if column in working.columns

        ]


    if (

        feature_columns is not None

        and

        len(
            usable_feature_columns
        )
        ==
        len(
            feature_columns
        )

        and

        len(
            usable_feature_columns
        )
        >=
        2

    ):


        source_matrix = (

            working[
                usable_feature_columns
            ]
            .apply(

                pd.to_numeric,

                errors="coerce"

            )
            .to_numpy(
                dtype=np.float32
            )

        )


        projection_metrics = (
            evaluate_projection_quality(

                source_matrix,

                positions,

                k=
                    k,

                source_metric=
                    source_metric,

                random_state=
                    random_state

            )
        )


        metrics.update({

            "source_feature_space":
                (
                    source_feature_name

                    or

                    "persisted_features"
                ),

            "source_feature_dimensions":
                int(
                    source_matrix.shape[1]
                ),

            "trustworthiness":
                projection_metrics[
                    "trustworthiness"
                ],

            "knn_preservation":
                projection_metrics[
                    "knn_preservation"
                ],

            "projection_sample_size":
                projection_metrics[
                    "projection_sample_size"
                ],

        })


        # ====================================================
        # DOMAIN MIXING BEFORE UMAP
        # ====================================================

        if (

            domain_column in working.columns

            and

            working[
                domain_column
            ]
            .astype(str)
            .nunique()
            >=
            2

        ):


            source_domain_metrics = (
                evaluate_label_mixing_matrix(

                    source_matrix,

                    working[
                        domain_column
                    ]
                    .astype(str)
                    .to_numpy(),

                    k=
                        k,

                    metric=
                        source_metric,

                    max_reference_size=
                        30_000,

                    query_per_class=
                        1_000,

                    silhouette_per_class=
                        500,

                    random_state=
                        random_state
                        +
                        20

                )
            )


            metrics.update({

                "source_domain_knn_balanced_accuracy":
                    source_domain_metrics[
                        "knn_balanced_accuracy"
                    ],

                "source_local_domain_entropy":
                    source_domain_metrics[
                        "local_entropy"
                    ],

                "source_cross_domain_neighbor_share":
                    source_domain_metrics[
                        "cross_label_neighbor_share"
                    ],

                "source_domain_silhouette":
                    source_domain_metrics[
                        "silhouette"
                    ],

            })


        else:

            metrics.update({

                "source_domain_knn_balanced_accuracy":
                    np.nan,

                "source_local_domain_entropy":
                    np.nan,

                "source_cross_domain_neighbor_share":
                    np.nan,

                "source_domain_silhouette":
                    np.nan,

            })


    else:

        metrics.update({

            "source_feature_space":
                None,

            "source_feature_dimensions":
                np.nan,

            "trustworthiness":
                np.nan,

            "knn_preservation":
                np.nan,

            "projection_sample_size":
                np.nan,

            "source_domain_knn_balanced_accuracy":
                np.nan,

            "source_local_domain_entropy":
                np.nan,

            "source_cross_domain_neighbor_share":
                np.nan,

            "source_domain_silhouette":
                np.nan,

        })


    return (

        metrics,

        cluster_domain_composition

    )