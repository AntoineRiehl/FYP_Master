# src/atlas/cross_domain/review_fusion.py

# ============================================================
# CROSS-DOMAIN REVIEW SEMANTIC FUSION
#
# Purpose
# -------
#
# Fuse:
#
#     shared cross-domain TF-IDF semantics
#
# with:
#
#     domain-specific Sentence-BERT review embeddings
#
# inside a single cross-domain representation.
#
#
# Example:
#
#     Movies + Music
#
#     shared semantic_text
#             ↓
#          TF-IDF
#             ↓
#        TruncatedSVD
#           256D
#             \
#              \
#               50 / 50 fusion
#              /
#             /
#     Movie review SBERT
#     Music review SBERT
#           384D
#
#              ↓
#
#         fused 640D
#
#
# IMPORTANT
# ---------
#
# Review embeddings from Movies, Music and Restaurants are
# compatible because they were all generated using the same
# Sentence-BERT model:
#
#     sentence-transformers/all-MiniLM-L6-v2
#
#
# Missing reviews
# ---------------
#
# If an entity has no review embedding, its review block is
# zero.
#
# After final L2 normalisation, such an entity effectively
# retains its TF-IDF semantic direction.
#
# This reproduces the general semantic fusion methodology
# used by the mono-domain atlases.
#
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd


from sklearn.decomposition import (
    TruncatedSVD
)

from sklearn.preprocessing import (
    normalize
)


# ============================================================
# INTERNAL HELPERS
# ============================================================


def _normalise_id(
    value
) -> str:
    """
    Convert an entity identifier to the canonical string form
    used during embedding alignment.
    """

    if pd.isna(value):

        return ""

    return str(
        value
    ).strip()


# ============================================================
# LOAD REVIEW EMBEDDINGS
# ============================================================


def _load_review_embedding_file(
    path
):
    """
    Load one domain-specific review embedding file.

    Expected arrays
    ---------------
        entity_ids
        embeddings
        review_counts

    Returns
    -------
    tuple
        entity_ids
        embeddings
        review_counts
    """

    path = Path(
        path
    )


    if not path.exists():

        raise FileNotFoundError(

            "Review embedding file was not found:\n"
            f"{path}"

        )


    data = np.load(

        path,

        allow_pickle=True

    )


    required_keys = {

        "entity_ids",

        "embeddings",

        "review_counts",

    }


    missing_keys = (

        required_keys

        -

        set(
            data.files
        )

    )


    if missing_keys:

        raise ValueError(

            f"Review embedding file:\n"
            f"{path}\n\n"
            f"is missing required arrays:\n"
            f"{sorted(missing_keys)}"

        )


    entity_ids = [

        _normalise_id(
            value
        )

        for value in data[
            "entity_ids"
        ]

    ]


    embeddings = np.asarray(

        data[
            "embeddings"
        ],

        dtype=np.float32

    )


    review_counts = np.asarray(

        data[
            "review_counts"
        ],

        dtype=np.int32

    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if embeddings.ndim != 2:

        raise ValueError(

            f"Embeddings in {path} must be a 2D matrix."

        )


    if (

        len(
            entity_ids
        )

        !=

        embeddings.shape[0]

    ):

        raise ValueError(

            f"entity_ids and embeddings are not aligned "
            f"in:\n{path}"

        )


    if (

        len(
            review_counts
        )

        !=

        embeddings.shape[0]

    ):

        raise ValueError(

            f"review_counts and embeddings are not aligned "
            f"in:\n{path}"

        )


    # ========================================================
    # DUPLICATE IDS
    # ========================================================

    if (

        len(
            entity_ids
        )

        !=

        len(
            set(
                entity_ids
            )
        )

    ):

        raise ValueError(

            f"Duplicate entity IDs detected in review "
            f"embedding file:\n{path}"

        )


    # ========================================================
    # NORMALISE DEFENSIVELY
    # ========================================================

    embeddings = normalize(

        embeddings,

        norm="l2",

        axis=1

    ).astype(
        np.float32
    )


    return (
        entity_ids,
        embeddings,
        review_counts
    )


# ============================================================
# ALIGN CROSS-DOMAIN REVIEW EMBEDDINGS
# ============================================================


def align_cross_domain_review_embeddings(

    combined_df: pd.DataFrame,

    review_embedding_paths: dict

):
    """
    Align domain-specific review embeddings with a combined
    cross-domain dataframe.

    Parameters
    ----------
    combined_df:
        Cross-domain dataframe created through
        combine_domain_data().

        Must contain:

            domain
            source_id

    review_embedding_paths:
        Dictionary mapping domain names to review embedding
        files.

        Example:

            {
                "movies":
                    Path("movie_review_embeddings.npz"),

                "music":
                    Path("music_review_embeddings.npz"),

                "restaurants":
                    Path(
                        "restaurant_review_embeddings.npz"
                    )
            }

    Returns
    -------
    aligned_embeddings:
        np.ndarray
        Shape:

            (number_of_items, review_dimensions)

    aligned_counts:
        np.ndarray
        Number of reviews actually used to generate each
        entity review embedding.

    alignment_info:
        dict
        Diagnostic metadata.
    """

    # ========================================================
    # VALIDATE DATAFRAME
    # ========================================================

    if not isinstance(
        combined_df,
        pd.DataFrame
    ):

        raise TypeError(

            "combined_df must be a pandas DataFrame."

        )


    required_columns = {

        "domain",

        "source_id",

    }


    missing_columns = (

        required_columns

        -

        set(
            combined_df.columns
        )

    )


    if missing_columns:

        raise ValueError(

            "combined_df is missing required columns: "
            f"{sorted(missing_columns)}"

        )


    if not review_embedding_paths:

        raise ValueError(

            "review_embedding_paths cannot be empty."

        )


    # ========================================================
    # LOAD DOMAIN EMBEDDINGS
    # ========================================================

    domain_data = {}

    embedding_dimension = None


    print()

    print(
        "Loading cross-domain review embeddings..."
    )


    for domain, path in (
        review_embedding_paths.items()
    ):


        domain = str(
            domain
        ).strip().lower()


        (
            entity_ids,
            embeddings,
            review_counts

        ) = _load_review_embedding_file(

            path

        )


        current_dimension = (
            embeddings.shape[1]
        )


        # ----------------------------------------------------
        # Every participating domain must use the same review
        # embedding space.
        # ----------------------------------------------------

        if embedding_dimension is None:

            embedding_dimension = (
                current_dimension
            )


        elif (

            current_dimension
            !=
            embedding_dimension

        ):

            raise ValueError(

                "Review embedding dimensions differ across "
                "domains.\n\n"
                f"Expected: {embedding_dimension}\n"
                f"{domain}: {current_dimension}\n\n"
                "Cross-domain review fusion requires all "
                "domains to use the same embedding model."

            )


        lookup = {

            entity_id:
                index

            for index, entity_id
            in enumerate(
                entity_ids
            )

        }


        domain_data[
            domain
        ] = {

            "embeddings":
                embeddings,

            "review_counts":
                review_counts,

            "lookup":
                lookup,

            "available":
                len(
                    entity_ids
                ),

        }


        print(

            f"  {domain:<15} "
            f"{len(entity_ids):>8,} entities | "
            f"{current_dimension}D"

        )


    # ========================================================
    # CREATE ALIGNED MATRICES
    # ========================================================

    n_items = len(
        combined_df
    )


    aligned_embeddings = np.zeros(

        (
            n_items,
            embedding_dimension
        ),

        dtype=np.float32

    )


    aligned_counts = np.zeros(

        n_items,

        dtype=np.int32

    )


    # --------------------------------------------------------
    # Diagnostic counts
    # --------------------------------------------------------

    aligned_by_domain = {

        domain:
            0

        for domain
        in domain_data

    }


    items_by_domain = (

        combined_df[
            "domain"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .to_dict()

    )


    # ========================================================
    # ALIGN ROW BY ROW
    # ========================================================

    domains = (

        combined_df[
            "domain"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .to_numpy()

    )


    source_ids = (

        combined_df[
            "source_id"
        ]
        .apply(
            _normalise_id
        )
        .to_numpy()

    )


    for row_index, (
        domain,
        source_id

    ) in enumerate(

        zip(
            domains,
            source_ids
        )

    ):


        # ----------------------------------------------------
        # A domain may theoretically participate in the atlas
        # without having review embeddings.
        #
        # In that case the review block simply stays zero.
        # ----------------------------------------------------

        if domain not in domain_data:

            continue


        domain_review_data = (
            domain_data[
                domain
            ]
        )


        review_index = (

            domain_review_data[
                "lookup"
            ]
            .get(
                source_id
            )

        )


        if review_index is None:

            continue


        aligned_embeddings[
            row_index
        ] = (

            domain_review_data[
                "embeddings"
            ][
                review_index
            ]

        )


        aligned_counts[
            row_index
        ] = (

            domain_review_data[
                "review_counts"
            ][
                review_index
            ]

        )


        aligned_by_domain[
            domain
        ] += 1


    # ========================================================
    # ALIGNMENT SUMMARY
    # ========================================================

    entities_with_reviews = int(

        (
            aligned_counts > 0
        )
        .sum()

    )


    print()

    print(
        "Cross-domain review alignment:"
    )


    for domain in sorted(
        items_by_domain
    ):


        total = int(

            items_by_domain.get(
                domain,
                0
            )

        )


        aligned = int(

            aligned_by_domain.get(
                domain,
                0
            )

        )


        percentage = (

            (
                aligned
                /
                total
                *
                100
            )

            if total > 0

            else 0.0

        )


        print(

            f"  {domain:<15} "
            f"{aligned:>8,} / "
            f"{total:>8,} "
            f"({percentage:6.2f}%)"

        )


    print()

    print(

        f"  Total with review embeddings: "
        f"{entities_with_reviews:,} / "
        f"{n_items:,}"

    )


    alignment_info = {

        "review_dimensions":
            int(
                embedding_dimension
            ),

        "entities_with_reviews":
            entities_with_reviews,

        "entities_without_reviews":
            int(
                n_items
                -
                entities_with_reviews
            ),

        "aligned_by_domain":
            {
                domain:
                    int(
                        count
                    )

                for domain, count
                in aligned_by_domain.items()
            },

        "items_by_domain":
            {
                domain:
                    int(
                        count
                    )

                for domain, count
                in items_by_domain.items()
            },

    }


    return (
        aligned_embeddings,
        aligned_counts,
        alignment_info
    )


# ============================================================
# REDUCE TF-IDF
# ============================================================


def reduce_cross_domain_tfidf(

    tfidf_matrix,

    n_components=256,

    random_state=42

):
    """
    Reduce the shared cross-domain TF-IDF matrix using
    TruncatedSVD.

    The reduced vectors are then L2-normalized.

    Returns
    -------
    reduced:
        np.ndarray

    info:
        dict
    """

    if tfidf_matrix.ndim != 2:

        raise ValueError(

            "tfidf_matrix must be a 2D matrix."

        )


    # --------------------------------------------------------
    # Ensure the requested dimensionality is valid.
    # --------------------------------------------------------

    max_components = min(

        int(
            n_components
        ),

        tfidf_matrix.shape[0] - 1,

        tfidf_matrix.shape[1] - 1,

    )


    if max_components < 2:

        raise ValueError(

            "TF-IDF matrix is too small for SVD reduction."

        )


    print()

    print(
        "Reducing shared cross-domain TF-IDF:"
    )


    print(

        f"  {tfidf_matrix.shape[1]:,}D "
        f"-> "
        f"{max_components:,}D"

    )


    svd = TruncatedSVD(

        n_components=
            max_components,

        random_state=
            random_state

    )


    reduced = svd.fit_transform(

        tfidf_matrix

    )


    explained_variance = float(

        svd
        .explained_variance_ratio_
        .sum()

    )


    reduced = normalize(

        reduced,

        norm="l2",

        axis=1

    ).astype(
        np.float32
    )


    print(

        f"  SVD explained variance: "
        f"{explained_variance:.4f}"

    )


    info = {

        "tfidf_components":
            int(
                max_components
            ),

        "svd_explained_variance":
            explained_variance,

    }


    return (
        reduced,
        info
    )


# ============================================================
# COMBINE SEMANTIC COMPONENTS
# ============================================================


def combine_cross_domain_semantics(

    tfidf_embeddings,

    review_embeddings,

    review_share=0.50

):
    """
    Combine normalized TF-IDF and review embeddings.

    Fusion
    ------

        TF-IDF block weight:

            sqrt(1 - review_share)

        Review block weight:

            sqrt(review_share)

    Because both blocks are normalized before concatenation,
    this produces approximately:

        fused cosine

            =

        (1-r) * TF-IDF cosine

            +

        r * review cosine

    for entities having both modalities.


    Entities without reviews
    ------------------------

    Their review vector is zero.

    After final normalization they retain their TF-IDF
    semantic direction.
    """

    review_share = float(
        review_share
    )


    if not (
        0.0
        <=
        review_share
        <=
        1.0
    ):

        raise ValueError(

            "review_share must be between 0 and 1."

        )


    tfidf_embeddings = np.asarray(

        tfidf_embeddings,

        dtype=np.float32

    )


    review_embeddings = np.asarray(

        review_embeddings,

        dtype=np.float32

    )


    if (

        tfidf_embeddings.shape[0]

        !=

        review_embeddings.shape[0]

    ):

        raise ValueError(

            "TF-IDF and review matrices contain "
            "different numbers of entities."

        )


    # ========================================================
    # NORMALIZE EACH COMPONENT DEFENSIVELY
    # ========================================================

    tfidf_embeddings = normalize(

        tfidf_embeddings,

        norm="l2",

        axis=1

    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # sklearn normalize safely leaves zero rows as zero.
    # --------------------------------------------------------

    review_embeddings = normalize(

        review_embeddings,

        norm="l2",

        axis=1

    ).astype(
        np.float32
    )


    # ========================================================
    # BLOCK WEIGHTS
    # ========================================================

    tfidf_share = (
        1.0
        -
        review_share
    )


    tfidf_weight = np.sqrt(
        tfidf_share
    )


    review_weight = np.sqrt(
        review_share
    )


    weighted_tfidf = (

        tfidf_embeddings

        *

        tfidf_weight

    )


    weighted_reviews = (

        review_embeddings

        *

        review_weight

    )


    # ========================================================
    # CONCATENATE
    # ========================================================

    fused = np.concatenate(

        [

            weighted_tfidf,

            weighted_reviews,

        ],

        axis=1

    )


    # ========================================================
    # FINAL NORMALIZATION
    # ========================================================

    fused = normalize(

        fused,

        norm="l2",

        axis=1

    ).astype(
        np.float32
    )


    return fused


# ============================================================
# COMPLETE CROSS-DOMAIN FUSION
# ============================================================


def fuse_cross_domain_semantics(

    tfidf_matrix,

    combined_df: pd.DataFrame,

    review_embedding_paths: dict,

    review_share=0.50,

    tfidf_components=256,

    random_state=42,

    return_info=True

):
    """
    Complete cross-domain semantic fusion pipeline.

    Parameters
    ----------
    tfidf_matrix:
        Shared cross-domain TF-IDF matrix.

        This should already have been generated from the
        domain-neutral semantic_text column.

    combined_df:
        Cross-domain dataframe created by
        combine_domain_data().

        Must contain:

            domain
            source_id

    review_embedding_paths:
        Mapping from domain name to review embedding NPZ.

        Example:

            {
                "movies":
                    MOVIE_REVIEW_EMBEDDINGS,

                "music":
                    MUSIC_REVIEW_EMBEDDINGS,
            }

    review_share:
        Contribution of the review semantic block.

        Example:

            0.50

        means equal TF-IDF / review semantic contribution.

    tfidf_components:
        Number of TruncatedSVD dimensions used for the
        shared TF-IDF representation.

    random_state:
        Reproducibility seed.

    return_info:
        If True:

            return (
                semantic_matrix,
                info,
                review_counts
            )

        Otherwise:

            return semantic_matrix


    Returns
    -------
    semantic_matrix:
        Fused semantic representation.

        With:

            TF-IDF = 256 dimensions
            Reviews = 384 dimensions

        this produces:

            640 dimensions

    info:
        Diagnostic fusion metadata.

    review_counts:
        Number of reviews actually used for each entity.
    """

    print()

    print(
        "=" * 65
    )

    print(
        "CROSS-DOMAIN SEMANTIC FUSION"
    )

    print(
        "=" * 65
    )

    print()


    print(
        f"Items:                "
        f"{len(combined_df):,}"
    )


    print(
        f"TF-IDF input shape:   "
        f"{tfidf_matrix.shape}"
    )


    print(
        f"TF-IDF components:    "
        f"{tfidf_components}"
    )


    print(
        f"Review share:         "
        f"{review_share:.2f}"
    )


    print(
        f"TF-IDF share:         "
        f"{1.0 - review_share:.2f}"
    )


    # ========================================================
    # 1. REDUCE SHARED TF-IDF
    # ========================================================

    (
        tfidf_reduced,
        tfidf_info

    ) = reduce_cross_domain_tfidf(

        tfidf_matrix,

        n_components=
            tfidf_components,

        random_state=
            random_state

    )


    # ========================================================
    # 2. ALIGN REVIEW EMBEDDINGS
    # ========================================================

    (
        review_embeddings,
        review_counts,
        alignment_info

    ) = align_cross_domain_review_embeddings(

        combined_df,

        review_embedding_paths

    )


    # ========================================================
    # 3. FUSE
    # ========================================================

    print()

    print(
        "Combining semantic components..."
    )


    semantic_matrix = (
        combine_cross_domain_semantics(

            tfidf_reduced,

            review_embeddings,

            review_share=
                review_share

        )
    )


    # ========================================================
    # 4. INFO
    # ========================================================

    info = {

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        "tfidf_components":
            int(
                tfidf_reduced.shape[1]
            ),

        "svd_explained_variance":
            float(
                tfidf_info[
                    "svd_explained_variance"
                ]
            ),


        # ----------------------------------------------------
        # Reviews
        # ----------------------------------------------------

        "review_components":
            int(
                review_embeddings.shape[1]
            ),

        "entities_with_reviews":
            int(
                alignment_info[
                    "entities_with_reviews"
                ]
            ),

        "entities_without_reviews":
            int(
                alignment_info[
                    "entities_without_reviews"
                ]
            ),

        "review_alignment_by_domain":
            alignment_info[
                "aligned_by_domain"
            ],


        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        "review_share":
            float(
                review_share
            ),

        "tfidf_share":
            float(
                1.0
                -
                review_share
            ),

        "fused_dimensions":
            int(
                semantic_matrix.shape[1]
            ),

    }


    # ========================================================
    # 5. SUMMARY
    # ========================================================

    print()

    print(
        "Cross-domain fusion complete:"
    )


    print(

        f"  TF-IDF dimensions:        "
        f"{info['tfidf_components']}"

    )


    print(

        f"  Review dimensions:        "
        f"{info['review_components']}"

    )


    print(

        f"  Fused dimensions:         "
        f"{info['fused_dimensions']}"

    )


    print(

        f"  Entities with reviews:    "
        f"{info['entities_with_reviews']:,}"

    )


    print(

        f"  Entities without reviews: "
        f"{info['entities_without_reviews']:,}"

    )


    print(

        f"  TF-IDF / review share:    "
        f"{info['tfidf_share']:.2f} / "
        f"{info['review_share']:.2f}"

    )


    print()


    # ========================================================
    # RETURN
    # ========================================================

    if return_info:

        return (

            semantic_matrix,

            info,

            review_counts

        )


    return semantic_matrix