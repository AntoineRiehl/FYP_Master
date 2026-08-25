# src/atlas/cross_domain/feel_space/feel_fusion.py

# ============================================================
# FEEL-SPACE BASE + REVIEW SEMANTIC FUSION
# ============================================================
#
# Purpose
# -------
#
# Combine:
#
#     base semantic Sentence-BERT embeddings
#
# with:
#
#     review Sentence-BERT embeddings
#
# before projecting entities onto the experiential anchors
# defined in:
#
#     feel_anchors.py
#
#
# IMPORTANT DIFFERENCE FROM METHOD A
# ----------------------------------
#
# Method A combines:
#
#     TF-IDF / SVD 256D
#          +
#     Review SBERT 384D
#
# through block concatenation:
#
#     [TF-IDF | Reviews] -> 640D
#
#
# Method B instead has:
#
#     Base SBERT    384D
#     Review SBERT  384D
#
# Both representations use:
#
#     sentence-transformers/all-MiniLM-L6-v2
#
# and therefore inhabit the SAME semantic space.
#
# This allows direct vector fusion:
#
#     E = normalize(
#         (1-r) * base
#         +
#         r * reviews
#     )
#
#
# MISSING MODALITIES
# ------------------
#
# The fusion explicitly handles four cases:
#
#     Base     Reviews      Representation
#     ----------------------------------------
#     yes      yes          weighted fusion
#     yes      no           base only
#     no       yes          review only
#     no       no           zero / undefined
#
#
# This is important because we do NOT want missing reviews
# to create an artificial zero block as happened in the
# concatenated Method A representation.
#
#
# ENTITY POPULATION
# -----------------
#
# The BASE embedding file defines the population.
#
# This is intentional:
#
#     Movie base file       -> current Movie atlas population
#     Music base file       -> current Music atlas population
#     Restaurant base file  -> current Restaurant population
#
# Review embeddings are aligned onto that population.
#
# ============================================================


from pathlib import Path

import json
import re

import numpy as np
import pandas as pd


from sklearn.preprocessing import (
    normalize
)


# ============================================================
# SEMANTIC SOURCE CODES
# ============================================================
#
# Stored both numerically and as readable strings.
# ============================================================

SOURCE_NONE = 0

SOURCE_BASE_ONLY = 1

SOURCE_REVIEW_ONLY = 2

SOURCE_BOTH = 3


SOURCE_LABELS = {

    SOURCE_NONE:
        "neither",

    SOURCE_BASE_ONLY:
        "base_only",

    SOURCE_REVIEW_ONLY:
        "review_only",

    SOURCE_BOTH:
        "base_and_reviews",

}


# ============================================================
# ID NORMALISATION
# ============================================================


def _entity_id_key(
    value
) -> str:
    """
    Convert identifiers to a stable string representation.

    This is deliberately conservative because domains use
    different identifier types:

        Movies       -> numeric MovieLens IDs
        Music        -> MusicBrainz UUID strings
        Restaurants  -> Yelp business ID strings

    Integer-valued floating IDs such as:

        123.0

    are normalized to:

        123
    """

    if pd.isna(
        value
    ):

        return ""


    # --------------------------------------------------------
    # Native / numpy integers
    # --------------------------------------------------------

    if isinstance(
        value,
        (
            int,
            np.integer
        )
    ):

        return str(
            int(value)
        )


    # --------------------------------------------------------
    # Native / numpy floats
    # --------------------------------------------------------

    if isinstance(
        value,
        (
            float,
            np.floating
        )
    ):

        if np.isfinite(
            value
        ):

            if float(
                value
            ).is_integer():

                return str(
                    int(value)
                )


        return str(
            value
        ).strip()


    # --------------------------------------------------------
    # Strings / everything else
    # --------------------------------------------------------

    text = str(
        value
    ).strip()


    # --------------------------------------------------------
    # Handle string forms such as:
    #
    #     "123.0"
    #
    # without modifying UUID-like identifiers.
    # --------------------------------------------------------

    if re.fullmatch(

        r"[+-]?\d+\.0+",

        text

    ):

        return text.split(
            "."
        )[0]


    return text


# ============================================================
# PRINT SECTION
# ============================================================


def _section(
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
# LOAD BASE SEMANTIC EMBEDDINGS
# ============================================================


def load_base_semantic_embeddings(
    path
):
    """
    Load one base semantic embedding file.

    Expected arrays
    ---------------
        entity_ids
        embeddings
        has_semantic_text

    Returns
    -------
    entity_ids
    embeddings
    has_base
    """

    path = Path(
        path
    )


    if not path.exists():

        raise FileNotFoundError(

            "Base semantic embedding file was not found:\n"
            f"{path}"

        )


    data = np.load(

        path,

        allow_pickle=True

    )


    required = {

        "entity_ids",

        "embeddings",

        "has_semantic_text",

    }


    missing = (

        required

        -

        set(
            data.files
        )

    )


    if missing:

        raise ValueError(

            f"Base semantic embedding file:\n"
            f"{path}\n\n"

            f"is missing required arrays:\n"
            f"{sorted(missing)}"

        )


    entity_ids = np.asarray(

        [

            _entity_id_key(
                value
            )

            for value
            in data[
                "entity_ids"
            ]

        ],

        dtype=str

    )


    embeddings = np.asarray(

        data[
            "embeddings"
        ],

        dtype=np.float32

    )


    has_base = np.asarray(

        data[
            "has_semantic_text"
        ],

        dtype=bool

    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if embeddings.ndim != 2:

        raise ValueError(

            "Base embeddings must be a 2D matrix."

        )


    n_entities = len(
        entity_ids
    )


    if embeddings.shape[0] != n_entities:

        raise ValueError(

            "Base entity IDs and embeddings are not aligned."

        )


    if len(
        has_base
    ) != n_entities:

        raise ValueError(

            "Base semantic availability flags are not aligned "
            "with the embeddings."

        )


    if len(
        set(
            entity_ids
        )
    ) != n_entities:

        raise ValueError(

            "Duplicate entity IDs detected in base semantic "
            f"embedding file:\n{path}"

        )


    # ========================================================
    # NORMALIZE DEFENSIVELY
    # ========================================================

    embeddings = normalize(

        embeddings,

        norm="l2",

        axis=1

    ).astype(
        np.float32
    )


    norms = np.linalg.norm(

        embeddings,

        axis=1

    )


    # --------------------------------------------------------
    # Text available -> vector must be non-zero.
    # --------------------------------------------------------

    invalid_present = (

        has_base

        &

        (
            norms <= 0
        )

    )


    if invalid_present.any():

        raise ValueError(

            f"{int(invalid_present.sum()):,} entities are "
            "marked as having base semantic text but contain "
            "zero base embeddings."

        )


    # --------------------------------------------------------
    # No text -> vector should be zero.
    # --------------------------------------------------------

    invalid_missing = (

        (~has_base)

        &

        (
            norms > 0
        )

    )


    if invalid_missing.any():

        raise ValueError(

            f"{int(invalid_missing.sum()):,} entities are "
            "marked as missing base semantic text but contain "
            "non-zero base embeddings."

        )


    return (

        entity_ids,

        embeddings,

        has_base

    )


# ============================================================
# LOAD REVIEW EMBEDDINGS
# ============================================================


def load_review_semantic_embeddings(
    path
):
    """
    Load one review embedding file.

    Expected arrays
    ---------------
        entity_ids
        embeddings
        review_counts

    Returns
    -------
    entity_ids
    embeddings
    review_counts
    """

    path = Path(
        path
    )


    if not path.exists():

        raise FileNotFoundError(

            "Review semantic embedding file was not found:\n"
            f"{path}"

        )


    data = np.load(

        path,

        allow_pickle=True

    )


    required = {

        "entity_ids",

        "embeddings",

        "review_counts",

    }


    missing = (

        required

        -

        set(
            data.files
        )

    )


    if missing:

        raise ValueError(

            f"Review semantic embedding file:\n"
            f"{path}\n\n"

            f"is missing required arrays:\n"
            f"{sorted(missing)}"

        )


    entity_ids = np.asarray(

        [

            _entity_id_key(
                value
            )

            for value
            in data[
                "entity_ids"
            ]

        ],

        dtype=str

    )


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

            "Review embeddings must be a 2D matrix."

        )


    n_entities = len(
        entity_ids
    )


    if embeddings.shape[0] != n_entities:

        raise ValueError(

            "Review entity IDs and embeddings are not aligned."

        )


    if len(
        review_counts
    ) != n_entities:

        raise ValueError(

            "Review counts and review embeddings are not "
            "aligned."

        )


    if len(
        set(
            entity_ids
        )
    ) != n_entities:

        raise ValueError(

            "Duplicate entity IDs detected in review "
            f"embedding file:\n{path}"

        )


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
# ALIGN REVIEWS TO BASE POPULATION
# ============================================================


def align_review_embeddings_to_base(

    base_entity_ids,

    review_entity_ids,

    review_embeddings,

    review_counts

):
    """
    Align review embeddings with the entity order defined by
    the base semantic embedding file.

    The base file defines the final population.
    """

    review_lookup = {

        entity_id:
            index

        for index, entity_id
        in enumerate(
            review_entity_ids
        )

    }


    n_entities = len(
        base_entity_ids
    )


    review_dimensions = (
        review_embeddings.shape[1]
    )


    aligned_embeddings = np.zeros(

        (
            n_entities,
            review_dimensions
        ),

        dtype=np.float32

    )


    aligned_counts = np.zeros(

        n_entities,

        dtype=np.int32

    )


    aligned = 0


    for row_index, entity_id in enumerate(

        base_entity_ids

    ):

        review_index = (
            review_lookup.get(
                entity_id
            )
        )


        if review_index is None:

            continue


        aligned_embeddings[
            row_index
        ] = review_embeddings[
            review_index
        ]


        aligned_counts[
            row_index
        ] = review_counts[
            review_index
        ]


        aligned += 1


    has_reviews = (

        aligned_counts > 0

    )


    return (

        aligned_embeddings,

        aligned_counts,

        has_reviews,

        aligned

    )


# ============================================================
# COMBINE BASE + REVIEW VECTORS
# ============================================================


def combine_base_review_embeddings(

    base_embeddings,

    review_embeddings,

    has_base,

    has_reviews,

    review_share=0.50

):
    """
    Fuse base and review Sentence-BERT vectors.

    Both semantic sources inhabit the same 384D MiniLM
    coordinate system.

    Parameters
    ----------
    review_share:
        Review contribution when BOTH modalities exist.

        Example:

            0.50

        gives:

            0.5 * base
            +
            0.5 * reviews

    Missing modality behaviour
    --------------------------

    Both:
        weighted average

    Base only:
        base vector

    Review only:
        review vector

    Neither:
        zero vector
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


    base_embeddings = np.asarray(

        base_embeddings,

        dtype=np.float32

    )


    review_embeddings = np.asarray(

        review_embeddings,

        dtype=np.float32

    )


    has_base = np.asarray(

        has_base,

        dtype=bool

    )


    has_reviews = np.asarray(

        has_reviews,

        dtype=bool

    )


    if base_embeddings.shape != review_embeddings.shape:

        raise ValueError(

            "Base and review embeddings must have exactly "
            "the same shape for Method B fusion.\n\n"

            f"Base:    {base_embeddings.shape}\n"
            f"Reviews: {review_embeddings.shape}"

        )


    n_entities = (
        base_embeddings.shape[0]
    )


    if (

        len(
            has_base
        )
        !=
        n_entities

        or

        len(
            has_reviews
        )
        !=
        n_entities

    ):

        raise ValueError(

            "Semantic availability flags are not aligned with "
            "the embedding matrices."

        )


    # ========================================================
    # NORMALIZE INPUTS
    # ========================================================

    base_embeddings = normalize(

        base_embeddings,

        norm="l2",

        axis=1

    ).astype(
        np.float32
    )


    review_embeddings = normalize(

        review_embeddings,

        norm="l2",

        axis=1

    ).astype(
        np.float32
    )


    # ========================================================
    # MODALITY MASKS
    # ========================================================

    both = (

        has_base

        &

        has_reviews

    )


    base_only = (

        has_base

        &

        (~has_reviews)

    )


    review_only = (

        (~has_base)

        &

        has_reviews

    )


    neither = (

        (~has_base)

        &

        (~has_reviews)

    )


    # ========================================================
    # OUTPUT MATRIX
    # ========================================================

    fused = np.zeros_like(

        base_embeddings,

        dtype=np.float32

    )


    # ========================================================
    # BOTH MODALITIES
    # ========================================================

    base_share = (

        1.0

        -

        review_share

    )


    fused[
        both
    ] = (

        base_share

        *

        base_embeddings[
            both
        ]

        +

        review_share

        *

        review_embeddings[
            both
        ]

    )


    # ========================================================
    # BASE ONLY
    # ========================================================

    fused[
        base_only
    ] = base_embeddings[
        base_only
    ]


    # ========================================================
    # REVIEW ONLY
    # ========================================================

    fused[
        review_only
    ] = review_embeddings[
        review_only
    ]


    # ========================================================
    # NEITHER
    #
    # Already zero from matrix initialization.
    # ========================================================


    # ========================================================
    # VALIDATE FUSED BOTH-MODALITY VECTORS
    # ========================================================

    pre_normalization_norms = np.linalg.norm(

        fused,

        axis=1

    )


    invalid_fused = (

        (

            both
            |
            base_only
            |
            review_only

        )

        &

        (
            pre_normalization_norms
            <=
            0
        )

    )


    if invalid_fused.any():

        raise ValueError(

            f"{int(invalid_fused.sum()):,} entities have at "
            "least one semantic modality but produced a zero "
            "fused vector."

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


    # ========================================================
    # SOURCE CODES
    # ========================================================

    source_codes = np.full(

        n_entities,

        SOURCE_NONE,

        dtype=np.int8

    )


    source_codes[
        base_only
    ] = SOURCE_BASE_ONLY


    source_codes[
        review_only
    ] = SOURCE_REVIEW_ONLY


    source_codes[
        both
    ] = SOURCE_BOTH


    return (

        fused,

        source_codes

    )


# ============================================================
# CREATE MODALITY DATAFRAME
# ============================================================


def create_modality_dataframe(

    entity_ids,

    has_base,

    has_reviews,

    review_counts,

    source_codes

):
    """
    Build a human-readable entity-level modality table.
    """

    source_labels = [

        SOURCE_LABELS[
            int(code)
        ]

        for code in source_codes

    ]


    return pd.DataFrame({

        "entity_id":
            entity_ids,

        "has_base_semantics":
            has_base,

        "has_review_semantics":
            has_reviews,

        "reviews_used_for_embedding":
            review_counts,

        "semantic_source":
            source_labels,

        "is_semantically_defined":
            (
                has_base
                |
                has_reviews
            ),

    })


# ============================================================
# SAVE FUSED OUTPUTS
# ============================================================


def save_fused_semantics(

    output_npz,

    output_index_csv,

    output_metadata_json,

    entity_ids,

    fused_embeddings,

    modality_df,

    source_name,

    review_share,

    base_embeddings_path,

    review_embeddings_path

):
    """
    Save fused Method B semantic representations.

    This function is optional.

    Pipelines may alternatively use the returned matrices
    directly.
    """

    output_npz = Path(
        output_npz
    )


    output_index_csv = Path(
        output_index_csv
    )


    output_metadata_json = Path(
        output_metadata_json
    )


    for path in [

        output_npz,

        output_index_csv,

        output_metadata_json,

    ]:

        path.parent.mkdir(

            parents=True,

            exist_ok=True

        )


    # ========================================================
    # NPZ
    # ========================================================

    np.savez_compressed(

        output_npz,

        entity_ids=
            np.asarray(
                entity_ids,
                dtype=str
            ),

        embeddings=
            fused_embeddings,

        has_base_semantics=
            modality_df[
                "has_base_semantics"
            ].to_numpy(
                dtype=bool
            ),

        has_review_semantics=
            modality_df[
                "has_review_semantics"
            ].to_numpy(
                dtype=bool
            ),

        review_counts=
            modality_df[
                "reviews_used_for_embedding"
            ].to_numpy(
                dtype=np.int32
            ),

        semantic_source=
            modality_df[
                "semantic_source"
            ].to_numpy(
                dtype=str
            ),

        is_semantically_defined=
            modality_df[
                "is_semantically_defined"
            ].to_numpy(
                dtype=bool
            )

    )


    # ========================================================
    # INDEX CSV
    # ========================================================

    index_df = modality_df.copy()


    index_df.insert(

        0,

        "embedding_row",

        np.arange(

            len(
                index_df
            ),

            dtype=np.int32

        )

    )


    index_df.to_csv(

        output_index_csv,

        index=False,

        encoding="utf-8"

    )


    # ========================================================
    # COUNTS
    # ========================================================

    source_counts = (

        modality_df[
            "semantic_source"
        ]
        .value_counts()
        .to_dict()

    )


    total = len(
        modality_df
    )


    defined = int(

        modality_df[
            "is_semantically_defined"
        ].sum()

    )


    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "source_name":
            source_name,

        "method":
            "shared_sbert_base_review_fusion",

        "embedding_dimensions":
            int(
                fused_embeddings.shape[1]
            ),

        "review_share_when_both_available":
            float(
                review_share
            ),

        "base_share_when_both_available":
            float(
                1.0
                -
                review_share
            ),

        "missing_modality_policy":
            (
                "use available modality without "
                "zero-block concatenation"
            ),

        "entities":
            int(
                total
            ),

        "semantically_defined":
            defined,

        "semantically_undefined":
            int(
                total
                -
                defined
            ),

        "semantic_coverage":
            (
                float(
                    defined
                    /
                    total
                )

                if total > 0

                else 0.0
            ),

        "base_and_reviews":
            int(
                source_counts.get(
                    "base_and_reviews",
                    0
                )
            ),

        "base_only":
            int(
                source_counts.get(
                    "base_only",
                    0
                )
            ),

        "review_only":
            int(
                source_counts.get(
                    "review_only",
                    0
                )
            ),

        "neither":
            int(
                source_counts.get(
                    "neither",
                    0
                )
            ),

        "base_embeddings_path":
            str(
                base_embeddings_path
            ),

        "review_embeddings_path":
            str(
                review_embeddings_path
            ),

    }


    with open(

        output_metadata_json,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            metadata,

            file,

            indent=2,

            ensure_ascii=False

        )


    return metadata


# ============================================================
# COMPLETE DOMAIN FUSION
# ============================================================


def fuse_domain_semantics(

    base_embeddings_path,

    review_embeddings_path,

    review_share=0.50,

    source_name="unknown",

    output_npz=None,

    output_index_csv=None,

    output_metadata_json=None

):
    """
    Complete Method B base + review fusion for one domain.

    Parameters
    ----------
    base_embeddings_path:
        NPZ generated by base_embeddings.py.

    review_embeddings_path:
        NPZ generated by the review embedding pipeline.

    review_share:
        Review contribution when BOTH modalities exist.

        Default:

            0.50

    source_name:
        Domain name used in diagnostics / metadata.

    output_npz:
        Optional fused embedding output.

    output_index_csv:
        Optional human-readable modality index.

    output_metadata_json:
        Optional metadata output.

    Returns
    -------
    fused_embeddings:
        np.ndarray

        Shape:

            (n_entities, 384)

    entity_ids:
        np.ndarray

    modality_df:
        pd.DataFrame

    fusion_info:
        dict
    """

    _section(
        f"FUSING {str(source_name).upper()} SEMANTICS"
    )


    print(
        f"Base embeddings:"
    )

    print(
        base_embeddings_path
    )


    print()

    print(
        f"Review embeddings:"
    )

    print(
        review_embeddings_path
    )


    print()

    print(
        f"Review share when both available: "
        f"{review_share:.2f}"
    )


    # ========================================================
    # 1. LOAD BASE
    # ========================================================

    (

        entity_ids,

        base_embeddings,

        has_base

    ) = load_base_semantic_embeddings(

        base_embeddings_path

    )


    # ========================================================
    # 2. LOAD REVIEWS
    # ========================================================

    (

        review_entity_ids,

        review_embeddings,

        review_counts

    ) = load_review_semantic_embeddings(

        review_embeddings_path

    )


    # ========================================================
    # DIMENSION CHECK
    # ========================================================

    if (

        base_embeddings.shape[1]

        !=

        review_embeddings.shape[1]

    ):

        raise ValueError(

            "Base and review embedding dimensions differ.\n\n"

            f"Base:    {base_embeddings.shape[1]}\n"
            f"Reviews: {review_embeddings.shape[1]}\n\n"

            "Method B requires both modalities to use the "
            "same Sentence-BERT model."

        )


    print(
        f"Base population:     "
        f"{len(entity_ids):,}"
    )


    print(
        f"Base dimensions:     "
        f"{base_embeddings.shape[1]}"
    )


    print(
        f"Review embeddings:   "
        f"{len(review_entity_ids):,}"
    )


    print(
        f"Review dimensions:   "
        f"{review_embeddings.shape[1]}"
    )


    # ========================================================
    # 3. ALIGN REVIEWS
    # ========================================================

    _section(
        "ALIGNING REVIEW SEMANTICS"
    )


    (

        aligned_reviews,

        aligned_review_counts,

        has_reviews,

        aligned_count

    ) = align_review_embeddings_to_base(

        base_entity_ids=
            entity_ids,

        review_entity_ids=
            review_entity_ids,

        review_embeddings=
            review_embeddings,

        review_counts=
            review_counts

    )


    print(
        f"Entities in base population: "
        f"{len(entity_ids):,}"
    )


    print(
        f"Review embeddings aligned:   "
        f"{aligned_count:,}"
    )


    print(
        f"Entities with reviews:       "
        f"{int(has_reviews.sum()):,}"
    )


    # ========================================================
    # 4. FUSE
    # ========================================================

    _section(
        "COMBINING BASE + REVIEW SEMANTICS"
    )


    (

        fused_embeddings,

        source_codes

    ) = combine_base_review_embeddings(

        base_embeddings=
            base_embeddings,

        review_embeddings=
            aligned_reviews,

        has_base=
            has_base,

        has_reviews=
            has_reviews,

        review_share=
            review_share

    )


    # ========================================================
    # 5. MODALITY TABLE
    # ========================================================

    modality_df = create_modality_dataframe(

        entity_ids=
            entity_ids,

        has_base=
            has_base,

        has_reviews=
            has_reviews,

        review_counts=
            aligned_review_counts,

        source_codes=
            source_codes

    )


    # ========================================================
    # 6. DIAGNOSTICS
    # ========================================================

    source_counts = (

        modality_df[
            "semantic_source"
        ]
        .value_counts()

    )


    both_count = int(

        source_counts.get(
            "base_and_reviews",
            0
        )

    )


    base_only_count = int(

        source_counts.get(
            "base_only",
            0
        )

    )


    review_only_count = int(

        source_counts.get(
            "review_only",
            0
        )

    )


    neither_count = int(

        source_counts.get(
            "neither",
            0
        )

    )


    defined_count = (

        len(
            entity_ids
        )

        -

        neither_count

    )


    print(
        f"Base + reviews:    "
        f"{both_count:,}"
    )


    print(
        f"Base only:         "
        f"{base_only_count:,}"
    )


    print(
        f"Reviews only:      "
        f"{review_only_count:,}"
    )


    print(
        f"Neither:           "
        f"{neither_count:,}"
    )


    print()


    print(
        f"Semantically defined: "
        f"{defined_count:,} / "
        f"{len(entity_ids):,}"
    )


    if len(
        entity_ids
    ) > 0:

        print(

            f"Semantic coverage:    "
            f"{(
                defined_count
                /
                len(entity_ids)
                *
                100
            ):.2f}%"

        )


    # ========================================================
    # 7. VALIDATE OUTPUT
    # ========================================================

    fused_norms = np.linalg.norm(

        fused_embeddings,

        axis=1

    )


    defined_mask = (

        modality_df[
            "is_semantically_defined"
        ]
        .to_numpy(
            dtype=bool
        )

    )


    invalid_defined = (

        defined_mask

        &

        (
            fused_norms <= 0
        )

    )


    if invalid_defined.any():

        raise ValueError(

            f"{int(invalid_defined.sum()):,} semantically "
            "defined entities have zero fused embeddings."

        )


    invalid_undefined = (

        (~defined_mask)

        &

        (
            fused_norms > 0
        )

    )


    if invalid_undefined.any():

        raise ValueError(

            f"{int(invalid_undefined.sum()):,} semantically "
            "undefined entities have non-zero fused embeddings."

        )


    # ========================================================
    # 8. FUSION INFO
    # ========================================================

    fusion_info = {

        "source_name":
            source_name,

        "entities":
            int(
                len(
                    entity_ids
                )
            ),

        "embedding_dimensions":
            int(
                fused_embeddings.shape[1]
            ),

        "review_share":
            float(
                review_share
            ),

        "base_share":
            float(
                1.0
                -
                review_share
            ),

        "base_and_reviews":
            both_count,

        "base_only":
            base_only_count,

        "review_only":
            review_only_count,

        "neither":
            neither_count,

        "semantically_defined":
            defined_count,

        "semantically_undefined":
            neither_count,

        "semantic_coverage":
            (
                float(
                    defined_count
                    /
                    len(entity_ids)
                )

                if len(
                    entity_ids
                ) > 0

                else 0.0
            ),

    }


    # ========================================================
    # 9. OPTIONAL SAVE
    # ========================================================

    outputs_requested = [

        output_npz is not None,

        output_index_csv is not None,

        output_metadata_json is not None,

    ]


    if any(
        outputs_requested
    ):

        if not all(
            outputs_requested
        ):

            raise ValueError(

                "If fused outputs are requested, all three "
                "paths must be supplied:\n\n"

                "output_npz\n"
                "output_index_csv\n"
                "output_metadata_json"

            )


        _section(
            "SAVING FUSED SEMANTICS"
        )


        save_fused_semantics(

            output_npz=
                output_npz,

            output_index_csv=
                output_index_csv,

            output_metadata_json=
                output_metadata_json,

            entity_ids=
                entity_ids,

            fused_embeddings=
                fused_embeddings,

            modality_df=
                modality_df,

            source_name=
                source_name,

            review_share=
                review_share,

            base_embeddings_path=
                base_embeddings_path,

            review_embeddings_path=
                review_embeddings_path

        )


        print(
            "Fused embeddings:"
        )

        print(
            output_npz
        )


        print()


        print(
            "Modality index:"
        )

        print(
            output_index_csv
        )


        print()


        print(
            "Metadata:"
        )

        print(
            output_metadata_json
        )


    # ========================================================
    # COMPLETE
    # ========================================================

    _section(
        "FEEL-SPACE SEMANTIC FUSION COMPLETE"
    )


    print(
        f"Source:                "
        f"{source_name}"
    )


    print(
        f"Entities:              "
        f"{len(entity_ids):,}"
    )


    print(
        f"Embedding dimensions:  "
        f"{fused_embeddings.shape[1]}"
    )


    print(
        f"Base + reviews:        "
        f"{both_count:,}"
    )


    print(
        f"Base only:             "
        f"{base_only_count:,}"
    )


    print(
        f"Reviews only:          "
        f"{review_only_count:,}"
    )


    print(
        f"Neither:               "
        f"{neither_count:,}"
    )


    print(
        f"Semantic coverage:     "
        f"{fusion_info['semantic_coverage'] * 100:.2f}%"
    )


    print()


    return (

        fused_embeddings,

        entity_ids,

        modality_df,

        fusion_info

    )


# ============================================================
# MULTI-DOMAIN FUSION
# ============================================================


def fuse_multiple_domains(

    domain_paths,

    review_share=0.50

):
    """
    Fuse base + review semantics for multiple domains and
    combine them into one shared semantic matrix.

    Parameters
    ----------
    domain_paths:
        Example:

        {
            "movies": {
                "base": Path(...),
                "reviews": Path(...)
            },

            "music": {
                "base": Path(...),
                "reviews": Path(...)
            },

            "restaurants": {
                "base": Path(...),
                "reviews": Path(...)
            }
        }

    review_share:
        Review contribution when both modalities exist.

    Returns
    -------
    combined_embeddings:
        np.ndarray

    combined_index:
        pd.DataFrame containing:

            domain
            source_id
            has_base_semantics
            has_review_semantics
            reviews_used_for_embedding
            semantic_source
            is_semantically_defined

    domain_info:
        dict
    """

    if not isinstance(
        domain_paths,
        dict
    ):

        raise TypeError(

            "domain_paths must be a dictionary."

        )


    if not domain_paths:

        raise ValueError(

            "domain_paths cannot be empty."

        )


    _section(
        "MULTI-DOMAIN FEEL-SPACE SEMANTIC FUSION"
    )


    embedding_blocks = []

    index_blocks = []

    domain_info = {}


    embedding_dimension = None


    for domain, paths in (
        domain_paths.items()
    ):


        if (

            "base"
            not in paths

            or

            "reviews"
            not in paths

        ):

            raise ValueError(

                f"Domain '{domain}' must provide both "
                "'base' and 'reviews' embedding paths."

            )


        (

            embeddings,

            entity_ids,

            modality_df,

            fusion_info

        ) = fuse_domain_semantics(

            base_embeddings_path=
                paths[
                    "base"
                ],

            review_embeddings_path=
                paths[
                    "reviews"
                ],

            review_share=
                review_share,

            source_name=
                domain

        )


        if embedding_dimension is None:

            embedding_dimension = (
                embeddings.shape[1]
            )


        elif (

            embeddings.shape[1]

            !=

            embedding_dimension

        ):

            raise ValueError(

                "Fused semantic dimensions differ across "
                "domains."

            )


        embedding_blocks.append(
            embeddings
        )


        domain_index = (
            modality_df.copy()
        )


        domain_index.insert(

            0,

            "domain",

            domain

        )


        domain_index = (
            domain_index.rename(

                columns={

                    "entity_id":
                        "source_id"

                }

            )
        )


        index_blocks.append(
            domain_index
        )


        domain_info[
            domain
        ] = fusion_info


    # ========================================================
    # COMBINE DOMAINS
    # ========================================================

    combined_embeddings = np.concatenate(

        embedding_blocks,

        axis=0

    )


    combined_index = pd.concat(

        index_blocks,

        axis=0,

        ignore_index=True

    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if (

        len(
            combined_index
        )

        !=

        combined_embeddings.shape[0]

    ):

        raise ValueError(

            "Combined feel-space index and embeddings are "
            "not aligned."

        )


    # --------------------------------------------------------
    # Global IDs are useful for later analysis.
    # --------------------------------------------------------

    combined_index[
        "id"
    ] = (

        combined_index[
            "domain"
        ].astype(str)

        +

        ":"

        +

        combined_index[
            "source_id"
        ].astype(str)

    )


    if combined_index[
        "id"
    ].duplicated().any():

        raise ValueError(

            "Duplicate global IDs detected after multi-domain "
            "semantic fusion."

        )


    _section(
        "MULTI-DOMAIN FUSION COMPLETE"
    )


    print(
        f"Combined entities:     "
        f"{len(combined_index):,}"
    )


    print(
        f"Embedding dimensions:  "
        f"{combined_embeddings.shape[1]}"
    )


    print()


    print(
        "Entities by domain:"
    )


    print(

        combined_index[
            "domain"
        ]
        .value_counts()
        .to_string()

    )


    print()


    print(
        "Semantic source distribution:"
    )


    print(

        pd.crosstab(

            combined_index[
                "domain"
            ],

            combined_index[
                "semantic_source"
            ]

        ).to_string()

    )


    print()


    return (

        combined_embeddings,

        combined_index,

        domain_info

    )