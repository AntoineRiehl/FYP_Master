# src/atlas/cross_domain/feel_space/feel_projection.py

# ============================================================
# FEEL-SPACE ANCHOR PROJECTION
# ============================================================
#
# Purpose
# -------
#
# Project shared 384D Sentence-BERT entity representations
# onto the explicitly defined experiential dimensions from:
#
#     feel_anchors.py
#
#
# INPUT
# -----
#
# Entity semantic representation:
#
#     shape = (n_entities, 384)
#
# produced by:
#
#     base semantics
#          +
#     review semantics
#          ↓
#     feel_fusion.py
#
#
# ANCHORS
# -------
#
# The feel space contains:
#
#     10 bipolar dimensions
#      3 unipolar dimensions
#
# represented using:
#
#     23 controlled natural-language anchor sentences
#
#
# BIPOLAR SCORE
# --------------
#
# For a bipolar axis:
#
#     score =
#         cosine(entity, high_anchor)
#         -
#         cosine(entity, low_anchor)
#
#
# Example:
#
#     activation =
#         similarity(energetic)
#         -
#         similarity(calm)
#
#
# Therefore:
#
#     positive activation -> more energetic
#     negative activation -> more calm
#
#
# UNIPOLAR SCORE
# --------------
#
# For a unipolar dimension:
#
#     score =
#         cosine(entity, anchor)
#
#
# Example:
#
#     nostalgia =
#         similarity(entity, nostalgia_anchor)
#
#
# OUTPUT
# ------
#
# Final representation:
#
#     shape = (n_entities, 13)
#
# Columns:
#
#     valence
#     activation
#     potency
#     tension
#     warmth
#     scale
#     tone
#     familiarity
#     refinement
#     complexity
#     nostalgia
#     wonder
#     tenderness
#
#
# IMPORTANT
# ---------
#
# This module deliberately does NOT:
#
#     - standardize feel dimensions
#     - run PCA
#     - run UMAP
#     - remove correlated dimensions
#
# Those decisions should only be made AFTER inspecting the
# empirical distributions in evaluate_feel_space.py.
#
# ============================================================


from pathlib import Path

import json

import numpy as np
import pandas as pd


from sklearn.preprocessing import (
    normalize
)


from sentence_transformers import (
    SentenceTransformer
)


from src.atlas.cross_domain.feel_space.feel_anchors import (

    FEEL_EMBEDDING_MODEL,

    FEEL_DIMENSIONS,

    BIPOLAR_AXES,

    UNIPOLAR_AXES,

    get_named_anchors,

    validate_feel_anchors,

)


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
# LOAD SENTENCE TRANSFORMER
# ============================================================


def load_feel_model(

    model_name=
        FEEL_EMBEDDING_MODEL,

    device=
        "cuda"

):
    """
    Load the Sentence-BERT model used for feel anchors.

    The model MUST be identical to the one used for:

        - base semantic embeddings
        - review embeddings
        - fused entity representations

    Returns
    -------
    model
    embedding_dimension
    """

    _section(
        "LOADING FEEL-SPACE SENTENCE TRANSFORMER"
    )


    model = SentenceTransformer(

        model_name,

        device=
            device

    )


    if hasattr(
        model,
        "get_embedding_dimension"
    ):

        embedding_dimension = int(

            model.get_embedding_dimension()

        )


    else:

        embedding_dimension = int(

            model.get_sentence_embedding_dimension()

        )


    print(
        f"Model:      "
        f"{model_name}"
    )


    print(
        f"Device:     "
        f"{model.device}"
    )


    print(
        f"Dimensions: "
        f"{embedding_dimension}"
    )


    return (

        model,

        embedding_dimension

    )


# ============================================================
# EMBED FEEL ANCHORS
# ============================================================


def embed_feel_anchors(

    model=None,

    model_name=
        FEEL_EMBEDDING_MODEL,

    device=
        "cuda",

    batch_size=
        32

):
    """
    Embed all 23 controlled experiential anchor sentences.

    Returns
    -------
    anchor_embeddings:
        np.ndarray

        Shape:

            (23, 384)

    anchor_df:
        pd.DataFrame

        Contains:

            anchor_index
            anchor_name
            dimension
            type
            label
            text
    """

    validate_feel_anchors()


    _section(
        "EMBEDDING FEEL ANCHORS"
    )


    named_anchors = (
        get_named_anchors()
    )


    anchor_df = pd.DataFrame(
        named_anchors
    )


    anchor_df.insert(

        0,

        "anchor_index",

        np.arange(

            len(
                anchor_df
            ),

            dtype=np.int32

        )

    )


    print(
        f"Feel dimensions: "
        f"{len(FEEL_DIMENSIONS)}"
    )


    print(
        f"Anchor sentences: "
        f"{len(anchor_df)}"
    )


    # ========================================================
    # LOAD MODEL IF NOT ALREADY PROVIDED
    # ========================================================

    if model is None:

        (
            model,
            embedding_dimension

        ) = load_feel_model(

            model_name=
                model_name,

            device=
                device

        )


    else:

        if hasattr(
            model,
            "get_embedding_dimension"
        ):

            embedding_dimension = int(

                model.get_embedding_dimension()

            )


        else:

            embedding_dimension = int(

                model.get_sentence_embedding_dimension()

            )


    # ========================================================
    # ENCODE
    # ========================================================

    anchor_embeddings = model.encode(

        anchor_df[
            "text"
        ].tolist(),

        batch_size=
            batch_size,

        show_progress_bar=
            False,

        convert_to_numpy=
            True,

        normalize_embeddings=
            True

    )


    anchor_embeddings = np.asarray(

        anchor_embeddings,

        dtype=np.float32

    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if anchor_embeddings.ndim != 2:

        raise ValueError(

            "Anchor embeddings must be a 2D matrix."

        )


    if (

        anchor_embeddings.shape[0]

        !=

        len(
            anchor_df
        )

    ):

        raise ValueError(

            "Anchor embedding count does not match "
            "the configured anchors."

        )


    if (

        anchor_embeddings.shape[1]

        !=

        embedding_dimension

    ):

        raise ValueError(

            "Anchor embedding dimensionality does not match "
            "the Sentence-BERT model."

        )


    anchor_embeddings = normalize(

        anchor_embeddings,

        norm="l2",

        axis=1

    ).astype(
        np.float32
    )


    print(
        f"Anchor embedding shape: "
        f"{anchor_embeddings.shape}"
    )


    print()


    print(
        "Configured anchors:"
    )


    for _, row in (
        anchor_df.iterrows()
    ):

        print(

            f"  {row['anchor_index']:>2} | "
            f"{row['anchor_name']:<26} | "
            f"{row['label']}"

        )


    print()


    return (

        anchor_embeddings,

        anchor_df

    )


# ============================================================
# COMPUTE RAW ANCHOR SIMILARITIES
# ============================================================


def compute_anchor_similarities(

    entity_embeddings,

    anchor_embeddings

):
    """
    Compute cosine similarity between every entity and every
    feel anchor.

    Both matrices are L2-normalized first.

    Therefore cosine similarity is simply:

        entity_embeddings @ anchor_embeddings.T

    Returns
    -------
    similarities:
        np.ndarray

        Shape:

            (n_entities, 23)
    """

    entity_embeddings = np.asarray(

        entity_embeddings,

        dtype=np.float32

    )


    anchor_embeddings = np.asarray(

        anchor_embeddings,

        dtype=np.float32

    )


    if entity_embeddings.ndim != 2:

        raise ValueError(

            "entity_embeddings must be a 2D matrix."

        )


    if anchor_embeddings.ndim != 2:

        raise ValueError(

            "anchor_embeddings must be a 2D matrix."

        )


    if (

        entity_embeddings.shape[1]

        !=

        anchor_embeddings.shape[1]

    ):

        raise ValueError(

            "Entity and anchor embedding dimensions differ.\n\n"

            f"Entities: {entity_embeddings.shape[1]}\n"
            f"Anchors:  {anchor_embeddings.shape[1]}"

        )


    # ========================================================
    # NORMALIZE DEFENSIVELY
    # ========================================================

    entity_embeddings = normalize(

        entity_embeddings,

        norm="l2",

        axis=1

    ).astype(
        np.float32
    )


    anchor_embeddings = normalize(

        anchor_embeddings,

        norm="l2",

        axis=1

    ).astype(
        np.float32
    )


    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    similarities = (

        entity_embeddings

        @

        anchor_embeddings.T

    ).astype(
        np.float32
    )


    return similarities


# ============================================================
# BUILD ANCHOR LOOKUP
# ============================================================


def _build_anchor_lookup(
    anchor_df
):
    """
    Map anchor_name -> anchor matrix column index.
    """

    required_columns = {

        "anchor_index",

        "anchor_name",

    }


    missing = (

        required_columns

        -

        set(
            anchor_df.columns
        )

    )


    if missing:

        raise ValueError(

            "Anchor dataframe is missing required columns: "
            f"{sorted(missing)}"

        )


    lookup = {

        str(
            row[
                "anchor_name"
            ]
        ):
            int(
                row[
                    "anchor_index"
                ]
            )

        for _, row
        in anchor_df.iterrows()

    }


    if (

        len(
            lookup
        )

        !=

        len(
            anchor_df
        )

    ):

        raise ValueError(

            "Duplicate anchor names detected."

        )


    return lookup


# ============================================================
# CONVERT 23 ANCHORS -> 13 FEEL DIMENSIONS
# ============================================================


def compute_feel_scores(

    anchor_similarities,

    anchor_df

):
    """
    Convert raw anchor similarities into the final 13
    experiential dimensions.

    Bipolar dimensions
    ------------------

        score = high_similarity - low_similarity


    Unipolar dimensions
    -------------------

        score = anchor_similarity


    Returns
    -------
    feel_scores:
        np.ndarray

        Shape:

            (n_entities, 13)

    dimension_names:
        list[str]
    """

    anchor_similarities = np.asarray(

        anchor_similarities,

        dtype=np.float32

    )


    if anchor_similarities.ndim != 2:

        raise ValueError(

            "anchor_similarities must be a 2D matrix."

        )


    if (

        anchor_similarities.shape[1]

        !=

        len(
            anchor_df
        )

    ):

        raise ValueError(

            "Anchor similarity matrix does not match the "
            "anchor configuration."

        )


    anchor_lookup = (
        _build_anchor_lookup(
            anchor_df
        )
    )


    n_entities = (
        anchor_similarities.shape[0]
    )


    n_dimensions = len(
        FEEL_DIMENSIONS
    )


    feel_scores = np.zeros(

        (
            n_entities,
            n_dimensions
        ),

        dtype=np.float32

    )


    # ========================================================
    # BUILD EACH DIMENSION
    # ========================================================

    for dimension_index, dimension in enumerate(

        FEEL_DIMENSIONS

    ):


        # ====================================================
        # BIPOLAR DIMENSION
        # ====================================================

        if dimension in BIPOLAR_AXES:

            axis = BIPOLAR_AXES[
                dimension
            ]


            high_anchor_name = (

                f"{dimension}__"
                f"{axis['high_label']}"

            )


            low_anchor_name = (

                f"{dimension}__"
                f"{axis['low_label']}"

            )


            if (

                high_anchor_name
                not in
                anchor_lookup

                or

                low_anchor_name
                not in
                anchor_lookup

            ):

                raise ValueError(

                    f"Missing configured anchors for bipolar "
                    f"dimension '{dimension}'."

                )


            high_index = (
                anchor_lookup[
                    high_anchor_name
                ]
            )


            low_index = (
                anchor_lookup[
                    low_anchor_name
                ]
            )


            high_similarity = (

                anchor_similarities[
                    :,
                    high_index
                ]

            )


            low_similarity = (

                anchor_similarities[
                    :,
                    low_index
                ]

            )


            feel_scores[
                :,
                dimension_index
            ] = (

                high_similarity

                -

                low_similarity

            )


        # ====================================================
        # UNIPOLAR DIMENSION
        # ====================================================

        elif dimension in UNIPOLAR_AXES:


            anchor_name = (
                dimension
            )


            if anchor_name not in anchor_lookup:

                raise ValueError(

                    f"Missing configured anchor for "
                    f"unipolar dimension '{dimension}'."

                )


            anchor_index = (
                anchor_lookup[
                    anchor_name
                ]
            )


            feel_scores[
                :,
                dimension_index
            ] = (

                anchor_similarities[
                    :,
                    anchor_index
                ]

            )


        else:

            raise ValueError(

                f"Unknown feel dimension: "
                f"{dimension}"

            )


    return (

        feel_scores,

        list(
            FEEL_DIMENSIONS
        )

    )


# ============================================================
# CREATE FEEL DATAFRAME
# ============================================================


def create_feel_dataframe(

    feel_scores,

    entity_index

):
    """
    Combine entity identifiers / metadata with the 13 feel
    dimensions.

    Parameters
    ----------
    feel_scores:
        Shape:

            (n_entities, 13)

    entity_index:
        DataFrame aligned row-for-row with feel_scores.

        Usually the combined index returned by:

            fuse_multiple_domains()

    Returns
    -------
    pd.DataFrame
    """

    feel_scores = np.asarray(

        feel_scores,

        dtype=np.float32

    )


    if not isinstance(
        entity_index,
        pd.DataFrame
    ):

        raise TypeError(

            "entity_index must be a pandas DataFrame."

        )


    if (

        feel_scores.shape[0]

        !=

        len(
            entity_index
        )

    ):

        raise ValueError(

            "Feel scores and entity index contain different "
            "numbers of rows."

        )


    if (

        feel_scores.shape[1]

        !=

        len(
            FEEL_DIMENSIONS
        )

    ):

        raise ValueError(

            "Feel score dimensionality does not match the "
            "configured Feel Space."

        )


    feel_df = (
        entity_index
        .reset_index(
            drop=True
        )
        .copy()
    )


    for dimension_index, dimension in enumerate(

        FEEL_DIMENSIONS

    ):

        feel_df[
            dimension
        ] = (

            feel_scores[
                :,
                dimension_index
            ]

        )


    return feel_df


# ============================================================
# SAVE FEEL-SPACE OUTPUTS
# ============================================================


def save_feel_projection(

    output_npz,

    output_csv,

    output_metadata_json,

    feel_scores,

    feel_df,

    anchor_embeddings,

    anchor_df,

    source_name,

    model_name=
        FEEL_EMBEDDING_MODEL

):
    """
    Save the final 13D Feel Space and its methodological
    metadata.
    """

    output_npz = Path(
        output_npz
    )


    output_csv = Path(
        output_csv
    )


    output_metadata_json = Path(
        output_metadata_json
    )


    for path in [

        output_npz,

        output_csv,

        output_metadata_json,

    ]:

        path.parent.mkdir(

            parents=True,

            exist_ok=True

        )


    # ========================================================
    # ENTITY IDS
    # ========================================================

    if "id" in feel_df.columns:

        entity_ids = (

            feel_df[
                "id"
            ]
            .astype(str)
            .to_numpy()

        )


    elif "source_id" in feel_df.columns:

        entity_ids = (

            feel_df[
                "source_id"
            ]
            .astype(str)
            .to_numpy()

        )


    else:

        entity_ids = np.asarray(

            [

                str(index)

                for index
                in range(
                    len(
                        feel_df
                    )
                )

            ],

            dtype=str

        )


    # ========================================================
    # NPZ
    # ========================================================

    np.savez_compressed(

        output_npz,

        entity_ids=
            entity_ids,

        feel_scores=
            np.asarray(
                feel_scores,
                dtype=np.float32
            ),

        dimensions=
            np.asarray(
                FEEL_DIMENSIONS,
                dtype=str
            ),

        anchor_names=
            anchor_df[
                "anchor_name"
            ]
            .astype(str)
            .to_numpy(),

        anchor_embeddings=
            np.asarray(
                anchor_embeddings,
                dtype=np.float32
            )

    )


    # ========================================================
    # CSV
    # ========================================================

    feel_df.to_csv(

        output_csv,

        index=False,

        encoding="utf-8"

    )


    # ========================================================
    # DIMENSION METADATA
    # ========================================================

    dimension_metadata = []


    for dimension in FEEL_DIMENSIONS:


        if dimension in BIPOLAR_AXES:

            axis = BIPOLAR_AXES[
                dimension
            ]


            dimension_metadata.append({

                "dimension":
                    dimension,

                "type":
                    "bipolar",

                "positive_direction":
                    axis[
                        "high_label"
                    ],

                "negative_direction":
                    axis[
                        "low_label"
                    ],

                "formula":
                    (
                        "cosine(entity, high_anchor) - "
                        "cosine(entity, low_anchor)"
                    ),

            })


        else:

            axis = UNIPOLAR_AXES[
                dimension
            ]


            dimension_metadata.append({

                "dimension":
                    dimension,

                "type":
                    "unipolar",

                "label":
                    axis[
                        "label"
                    ],

                "formula":
                    (
                        "cosine(entity, anchor)"
                    ),

            })


    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "source_name":
            source_name,

        "method":
            "semantic_anchor_projection",

        "embedding_model":
            model_name,

        "input_embedding_dimensions":
            int(
                anchor_embeddings.shape[1]
            ),

        "anchor_count":
            int(
                len(
                    anchor_df
                )
            ),

        "feel_dimensions":
            list(
                FEEL_DIMENSIONS
            ),

        "feel_dimension_count":
            int(
                len(
                    FEEL_DIMENSIONS
                )
            ),

        "bipolar_dimensions":
            int(
                len(
                    BIPOLAR_AXES
                )
            ),

        "unipolar_dimensions":
            int(
                len(
                    UNIPOLAR_AXES
                )
            ),

        "score_standardization":
            "none",

        "score_normalization":
            "none",

        "dimension_metadata":
            dimension_metadata,

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
# COMPLETE FEEL PROJECTION
# ============================================================


def project_to_feel_space(

    entity_embeddings,

    entity_index,

    model=None,

    model_name=
        FEEL_EMBEDDING_MODEL,

    device=
        "cuda",

    batch_size=
        32,

    output_npz=None,

    output_csv=None,

    output_metadata_json=None,

    source_name=
        "cross_domain"

):
    """
    Complete semantic-anchor projection.

    Parameters
    ----------
    entity_embeddings:
        Shared fused SBERT representations.

        Expected shape:

            (n_entities, 384)

    entity_index:
        Row-aligned entity metadata.

    model:
        Optional already-loaded SentenceTransformer.

    model_name:
        Model used to encode anchor sentences.

    device:
        Usually "cuda".

    batch_size:
        Anchor encoding batch size.

    output_npz / output_csv / output_metadata_json:
        Optional saved outputs.

        If one is supplied, all three must be supplied.

    source_name:
        Human-readable label for metadata.

    Returns
    -------
    feel_scores:
        np.ndarray

        Shape:

            (n_entities, 13)

    feel_df:
        pd.DataFrame

    anchor_embeddings:
        np.ndarray

    anchor_df:
        pd.DataFrame

    projection_info:
        dict
    """

    _section(
        "PROJECTING ENTITIES INTO SHARED FEEL SPACE"
    )


    entity_embeddings = np.asarray(

        entity_embeddings,

        dtype=np.float32

    )


    if entity_embeddings.ndim != 2:

        raise ValueError(

            "entity_embeddings must be a 2D matrix."

        )


    if not isinstance(
        entity_index,
        pd.DataFrame
    ):

        raise TypeError(

            "entity_index must be a pandas DataFrame."

        )


    if (

        len(
            entity_index
        )

        !=

        entity_embeddings.shape[0]

    ):

        raise ValueError(

            "entity_index and entity_embeddings are not "
            "row-aligned."

        )


    print(
        f"Entities:             "
        f"{entity_embeddings.shape[0]:,}"
    )


    print(
        f"Semantic dimensions:  "
        f"{entity_embeddings.shape[1]}"
    )


    # ========================================================
    # SEMANTICALLY DEFINED ENTITIES
    # ========================================================

    entity_norms = np.linalg.norm(

        entity_embeddings,

        axis=1

    )


    defined_mask = (

        entity_norms > 0

    )


    n_defined = int(
        defined_mask.sum()
    )


    n_undefined = (

        len(
            entity_embeddings
        )

        -

        n_defined

    )


    print(
        f"Semantically defined: "
        f"{n_defined:,}"
    )


    print(
        f"Undefined:            "
        f"{n_undefined:,}"
    )


    # ========================================================
    # 1. EMBED ANCHORS
    # ========================================================

    (

        anchor_embeddings,

        anchor_df

    ) = embed_feel_anchors(

        model=
            model,

        model_name=
            model_name,

        device=
            device,

        batch_size=
            batch_size

    )


    # ========================================================
    # DIMENSION CHECK
    # ========================================================

    if (

        entity_embeddings.shape[1]

        !=

        anchor_embeddings.shape[1]

    ):

        raise ValueError(

            "Entity semantic embeddings and feel anchors do "
            "not inhabit the same embedding space.\n\n"

            f"Entities: {entity_embeddings.shape[1]}D\n"
            f"Anchors:  {anchor_embeddings.shape[1]}D"

        )


    # ========================================================
    # 2. RAW ANCHOR SIMILARITIES
    # ========================================================

    _section(
        "COMPUTING ANCHOR SIMILARITIES"
    )


    anchor_similarities = (
        compute_anchor_similarities(

            entity_embeddings,

            anchor_embeddings

        )
    )


    print(
        f"Similarity matrix shape: "
        f"{anchor_similarities.shape}"
    )


    # ========================================================
    # 3. 23 ANCHORS -> 13 DIMENSIONS
    # ========================================================

    _section(
        "CONSTRUCTING 13D EXPERIENTIAL REPRESENTATION"
    )


    (

        feel_scores,

        dimension_names

    ) = compute_feel_scores(

        anchor_similarities,

        anchor_df

    )


    print(
        f"Feel score shape: "
        f"{feel_scores.shape}"
    )


    print()


    print(
        "Dimensions:"
    )


    for dimension in (
        dimension_names
    ):

        print(
            f"  - {dimension}"
        )


    # ========================================================
    # IMPORTANT:
    #
    # Undefined entities currently produce zero anchor
    # similarities because their input semantic vector is
    # zero.
    #
    # For bipolar dimensions this naturally becomes zero.
    #
    # For unipolar dimensions it also becomes zero.
    #
    # They should NOT be interpreted as genuinely neutral
    # entities.
    #
    # Their semantic availability metadata must therefore be
    # preserved, and they may later be excluded before UMAP.
    # ========================================================


    # ========================================================
    # 4. DATAFRAME
    # ========================================================

    feel_df = create_feel_dataframe(

        feel_scores=
            feel_scores,

        entity_index=
            entity_index

    )


    # --------------------------------------------------------
    # Ensure semantic-defined status is available even if the
    # supplied index does not already contain it.
    # --------------------------------------------------------

    if (

        "is_semantically_defined"

        not in

        feel_df.columns

    ):

        feel_df[
            "is_semantically_defined"
        ] = defined_mask


    # ========================================================
    # 5. DIAGNOSTIC SUMMARY
    # ========================================================

    _section(
        "FEEL-SPACE SCORE SUMMARY"
    )


    defined_scores = feel_df.loc[

        feel_df[
            "is_semantically_defined"
        ],

        FEEL_DIMENSIONS

    ]


    if len(
        defined_scores
    ) > 0:

        print(

            defined_scores
            .describe()
            .T[
                [
                    "mean",
                    "std",
                    "min",
                    "50%",
                    "max",
                ]
            ]
            .to_string()

        )


    # ========================================================
    # 6. PROJECTION INFO
    # ========================================================

    projection_info = {

        "entities":
            int(
                len(
                    entity_embeddings
                )
            ),

        "semantically_defined":
            n_defined,

        "semantically_undefined":
            n_undefined,

        "input_dimensions":
            int(
                entity_embeddings.shape[1]
            ),

        "anchor_count":
            int(
                len(
                    anchor_df
                )
            ),

        "feel_dimensions":
            int(
                len(
                    FEEL_DIMENSIONS
                )
            ),

        "dimension_names":
            list(
                FEEL_DIMENSIONS
            ),

        "standardization":
            "none",

    }


    # ========================================================
    # 7. OPTIONAL SAVE
    # ========================================================

    output_flags = [

        output_npz is not None,

        output_csv is not None,

        output_metadata_json is not None,

    ]


    if any(
        output_flags
    ):

        if not all(
            output_flags
        ):

            raise ValueError(

                "If feel-space outputs are requested, all "
                "three paths must be supplied:\n\n"

                "output_npz\n"
                "output_csv\n"
                "output_metadata_json"

            )


        _section(
            "SAVING FEEL-SPACE PROJECTION"
        )


        save_feel_projection(

            output_npz=
                output_npz,

            output_csv=
                output_csv,

            output_metadata_json=
                output_metadata_json,

            feel_scores=
                feel_scores,

            feel_df=
                feel_df,

            anchor_embeddings=
                anchor_embeddings,

            anchor_df=
                anchor_df,

            source_name=
                source_name,

            model_name=
                model_name

        )


        print(
            "Feel-space embeddings:"
        )

        print(
            output_npz
        )


        print()


        print(
            "Feel-score table:"
        )

        print(
            output_csv
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
        "FEEL-SPACE PROJECTION COMPLETE"
    )


    print(
        f"Entities:              "
        f"{len(entity_embeddings):,}"
    )


    print(
        f"Semantic dimensions:   "
        f"{entity_embeddings.shape[1]}"
    )


    print(
        f"Anchor sentences:      "
        f"{len(anchor_df)}"
    )


    print(
        f"Feel dimensions:       "
        f"{len(FEEL_DIMENSIONS)}"
    )


    print(
        f"Defined entities:      "
        f"{n_defined:,}"
    )


    print(
        f"Undefined entities:    "
        f"{n_undefined:,}"
    )


    print()


    return (

        feel_scores,

        feel_df,

        anchor_embeddings,

        anchor_df,

        projection_info

    )