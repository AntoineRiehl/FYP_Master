# src/atlas/cross_domain/feel_space/base_embeddings.py

# ============================================================
# BASE SEMANTIC EMBEDDINGS
# ============================================================
#
# Purpose
# -------
#
# Generate one Sentence-BERT embedding per entity from the
# entity's existing non-review semantic text.
#
#
# Examples
# --------
#
# Movies:
#
#     movieId
#     tags_text
#
#
# Music:
#
#     mbid
#     tags_text
#
#
# Restaurants:
#
#     business_id
#     tags_text
#         =
#     categories + tips
#
#
# All domains use the SAME Sentence-BERT model as the review
# embeddings:
#
#     sentence-transformers/all-MiniLM-L6-v2
#
# Therefore:
#
#     base semantic embeddings
#
# and:
#
#     review semantic embeddings
#
# inhabit the same 384-dimensional semantic space.
#
#
# IMPORTANT
# ---------
#
# This module does NOT:
#
#     - fuse base and review embeddings
#     - compute feel-anchor scores
#     - run UMAP
#     - remove domain-specific vocabulary
#
# It only converts an entity's existing semantic text into a
# shared Sentence-BERT representation.
#
#
# EMPTY TEXT
# ----------
#
# Entities with empty semantic text are PRESERVED.
#
# Their base semantic embedding is stored as a zero vector.
#
# We deliberately do NOT encode the empty string, because an
# SBERT embedding of "" would still create an artificial
# semantic direction.
#
# Later feel-space fusion can therefore distinguish:
#
#     has base semantics
#     has review semantics
#     has both
#     has neither
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


from sentence_transformers import (
    SentenceTransformer
)


from src.atlas.cross_domain.feel_space.feel_anchors import (
    FEEL_EMBEDDING_MODEL
)


# ============================================================
# TEXT CLEANING
# ============================================================


def clean_semantic_text(
    text
) -> str:
    """
    Conservatively clean semantic text before Sentence-BERT.

    Sentence-BERT receives natural language / descriptive text,
    so aggressive NLP preprocessing is deliberately avoided.

    We do NOT:

        - lowercase
        - remove stop words
        - stem
        - lemmatize
        - remove punctuation

    Only whitespace is normalised.
    """

    if pd.isna(
        text
    ):

        return ""


    text = str(
        text
    )


    text = re.sub(

        r"\s+",

        " ",

        text

    )


    return text.strip()


# ============================================================
# ID NORMALISATION
# ============================================================


def _normalise_entity_id(
    value
) -> str:
    """
    Convert entity identifiers to the canonical string form
    used throughout the embedding files.
    """

    if pd.isna(
        value
    ):

        return ""


    return str(
        value
    ).strip()


# ============================================================
# SECTION PRINTING
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
# VALIDATE INPUT DATAFRAME
# ============================================================


def _validate_dataframe(

    df,

    entity_id_column,

    text_column

):
    """
    Validate the entity-level semantic dataframe.
    """

    if not isinstance(
        df,
        pd.DataFrame
    ):

        raise TypeError(

            "df must be a pandas DataFrame."

        )


    if df.empty:

        raise ValueError(

            "Cannot build base semantic embeddings from "
            "an empty dataframe."

        )


    required_columns = {

        entity_id_column,

        text_column,

    }


    missing_columns = (

        required_columns

        -

        set(
            df.columns
        )

    )


    if missing_columns:

        raise ValueError(

            "Input dataframe is missing required columns: "
            f"{sorted(missing_columns)}"

        )


# ============================================================
# PREPARE ENTITY DATA
# ============================================================


def _prepare_entity_data(

    df,

    entity_id_column,

    text_column

):
    """
    Create a clean entity-level dataframe containing:

        entity_id
        semantic_text
        has_semantic_text
        text_char_count
        text_word_count
    """

    data = df[

        [
            entity_id_column,
            text_column
        ]

    ].copy()


    # ========================================================
    # ENTITY IDS
    # ========================================================

    data[
        "entity_id"
    ] = (

        data[
            entity_id_column
        ]
        .apply(
            _normalise_entity_id
        )

    )


    invalid_ids = (

        data[
            "entity_id"
        ]
        ==
        ""

    )


    invalid_id_count = int(
        invalid_ids.sum()
    )


    if invalid_id_count > 0:

        raise ValueError(

            f"{invalid_id_count:,} entities have missing or "
            f"empty IDs in column '{entity_id_column}'."

        )


    # ========================================================
    # DUPLICATE IDS
    # ========================================================

    duplicate_mask = (

        data[
            "entity_id"
        ]
        .duplicated(
            keep=False
        )

    )


    if duplicate_mask.any():

        duplicate_ids = (

            data.loc[

                duplicate_mask,

                "entity_id"

            ]
            .unique()
            .tolist()

        )


        raise ValueError(

            "Duplicate entity IDs detected while building "
            "base semantic embeddings.\n\n"

            f"Examples: {duplicate_ids[:10]}"

        )


    # ========================================================
    # CLEAN SEMANTIC TEXT
    # ========================================================

    data[
        "semantic_text"
    ] = (

        data[
            text_column
        ]
        .apply(
            clean_semantic_text
        )

    )


    data[
        "has_semantic_text"
    ] = (

        data[
            "semantic_text"
        ]
        .str.len()
        >
        0

    )


    data[
        "text_char_count"
    ] = (

        data[
            "semantic_text"
        ]
        .str.len()
        .astype(
            np.int32
        )

    )


    data[
        "text_word_count"
    ] = (

        data[
            "semantic_text"
        ]
        .apply(

            lambda text:
                len(
                    text.split()
                )

        )
        .astype(
            np.int32
        )

    )


    # --------------------------------------------------------
    # Keep only the standardised fields from this point.
    # --------------------------------------------------------

    data = data[

        [
            "entity_id",
            "semantic_text",
            "has_semantic_text",
            "text_char_count",
            "text_word_count",
        ]

    ].reset_index(
        drop=True
    )


    return data


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================


def _load_model(

    model_name,

    device

):
    """
    Load the shared Sentence-BERT model.
    """

    _section(
        "LOADING SENTENCE TRANSFORMER"
    )


    model = SentenceTransformer(

        model_name,

        device=
            device

    )


    # --------------------------------------------------------
    # Newer SentenceTransformers versions expose
    # get_embedding_dimension().
    #
    # Keep a fallback for compatibility with older versions.
    # --------------------------------------------------------

    if hasattr(
        model,
        "get_embedding_dimension"
    ):

        embedding_dimension = (
            model.get_embedding_dimension()
        )


    else:

        embedding_dimension = (
            model.get_sentence_embedding_dimension()
        )


    embedding_dimension = int(
        embedding_dimension
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
# ENCODE TEXT
# ============================================================


def _encode_semantic_texts(

    model,

    entity_data,

    embedding_dimension,

    batch_size,

    encoding_chunk_size

):
    """
    Encode all entities that contain usable semantic text.

    Entities with empty text remain zero vectors.
    """

    _section(
        "ENCODING BASE SEMANTIC TEXT"
    )


    n_entities = len(
        entity_data
    )


    embeddings = np.zeros(

        (
            n_entities,
            embedding_dimension
        ),

        dtype=np.float32

    )


    valid_indices = np.flatnonzero(

        entity_data[
            "has_semantic_text"
        ].to_numpy()

    )


    n_valid = len(
        valid_indices
    )


    print(
        f"Entities total:          "
        f"{n_entities:,}"
    )


    print(
        f"Entities with text:      "
        f"{n_valid:,}"
    )


    print(
        f"Entities without text:   "
        f"{n_entities - n_valid:,}"
    )


    if n_valid == 0:

        print()

        print(
            "No semantic text available. "
            "Returning zero embeddings."
        )

        return embeddings


    # ========================================================
    # CHUNKED ENCODING
    # ========================================================

    encoded_count = 0

    chunk_number = 0


    for start in range(

        0,

        n_valid,

        encoding_chunk_size

    ):


        chunk_number += 1


        end = min(

            start
            +
            encoding_chunk_size,

            n_valid

        )


        chunk_indices = (

            valid_indices[
                start:end
            ]

        )


        chunk_texts = (

            entity_data.loc[

                chunk_indices,

                "semantic_text"

            ]
            .tolist()

        )


        chunk_embeddings = model.encode(

            chunk_texts,

            batch_size=
                batch_size,

            show_progress_bar=
                False,

            convert_to_numpy=
                True,

            normalize_embeddings=
                True

        )


        chunk_embeddings = np.asarray(

            chunk_embeddings,

            dtype=np.float32

        )


        if (

            chunk_embeddings.ndim
            !=
            2

        ):

            raise ValueError(

                "Sentence-BERT returned an unexpected "
                "embedding shape."

            )


        if (

            chunk_embeddings.shape[1]

            !=

            embedding_dimension

        ):

            raise ValueError(

                "Sentence-BERT embedding dimensionality "
                "changed unexpectedly.\n\n"

                f"Expected: {embedding_dimension}\n"
                f"Received: {chunk_embeddings.shape[1]}"

            )


        embeddings[
            chunk_indices
        ] = chunk_embeddings


        encoded_count += len(
            chunk_indices
        )


        print(

            f"\rEncoding chunks: "
            f"{chunk_number:,} | "
            f"Entities encoded: "
            f"{encoded_count:,} / "
            f"{n_valid:,}",

            end=""

        )


    print()


    # ========================================================
    # DEFENSIVE NORMALIZATION
    # ========================================================

    # Zero rows remain zero.
    #
    # Non-zero rows should already be normalized by
    # SentenceTransformer.encode(), but normalize again so
    # downstream code can rely on this property.
    # --------------------------------------------------------

    embeddings = normalize(

        embeddings,

        norm="l2",

        axis=1

    ).astype(
        np.float32
    )


    return embeddings


# ============================================================
# SAVE OUTPUTS
# ============================================================


def _save_outputs(

    entity_data,

    embeddings,

    output_npz,

    output_index_csv,

    output_metadata_json,

    model_name,

    source_name,

    entity_id_column,

    text_column,

    device,

    batch_size,

    encoding_chunk_size

):
    """
    Save embeddings, index information and methodological
    metadata.
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
            entity_data[
                "entity_id"
            ].to_numpy(
                dtype=str
            ),

        embeddings=
            embeddings,

        has_semantic_text=
            entity_data[
                "has_semantic_text"
            ].to_numpy(
                dtype=bool
            ),

        text_char_counts=
            entity_data[
                "text_char_count"
            ].to_numpy(
                dtype=np.int32
            ),

        text_word_counts=
            entity_data[
                "text_word_count"
            ].to_numpy(
                dtype=np.int32
            )

    )


    # ========================================================
    # INDEX CSV
    # ========================================================

    index_df = entity_data[

        [
            "entity_id",
            "has_semantic_text",
            "text_char_count",
            "text_word_count",
        ]

    ].copy()


    index_df[
        "embedding_row"
    ] = np.arange(

        len(
            index_df
        ),

        dtype=np.int32

    )


    index_df = index_df[

        [
            "embedding_row",
            "entity_id",
            "has_semantic_text",
            "text_char_count",
            "text_word_count",
        ]

    ]


    index_df.to_csv(

        output_index_csv,

        index=False,

        encoding="utf-8"

    )


    # ========================================================
    # METADATA JSON
    # ========================================================

    n_entities = len(
        entity_data
    )


    n_with_text = int(

        entity_data[
            "has_semantic_text"
        ].sum()

    )


    metadata = {

        "source_name":
            source_name,

        "method":
            "base_semantic_sentence_embedding",

        "model":
            model_name,

        "embedding_dimensions":
            int(
                embeddings.shape[1]
            ),

        "entity_id_column":
            entity_id_column,

        "text_column":
            text_column,

        "entities":
            int(
                n_entities
            ),

        "entities_with_semantic_text":
            n_with_text,

        "entities_without_semantic_text":
            int(
                n_entities
                -
                n_with_text
            ),

        "semantic_text_coverage":
            (
                float(
                    n_with_text
                    /
                    n_entities
                )

                if n_entities > 0

                else 0.0
            ),

        "empty_text_policy":
            "zero_vector",

        "text_preprocessing":
            "whitespace_normalization_only",

        "embedding_normalization":
            "L2",

        "device_requested":
            device,

        "batch_size":
            int(
                batch_size
            ),

        "encoding_chunk_size":
            int(
                encoding_chunk_size
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
# BUILD BASE SEMANTIC EMBEDDINGS
# ============================================================


def build_base_semantic_embeddings(

    df,

    entity_id_column,

    text_column,

    output_npz,

    output_index_csv,

    output_metadata_json,

    source_name,

    model_name=
        FEEL_EMBEDDING_MODEL,

    batch_size=
        64,

    encoding_chunk_size=
        10_000,

    device=
        "cuda"

):
    """
    Build one shared Sentence-BERT base semantic embedding per
    entity.

    Parameters
    ----------
    df:
        Entity-level dataframe.

        There must be exactly one row per entity.

    entity_id_column:
        Canonical domain-specific entity identifier.

        Examples:

            movieId
            mbid
            business_id

    text_column:
        Existing semantic text representation.

        Examples:

            tags_text

    output_npz:
        Embedding output path.

    output_index_csv:
        Human-readable embedding index.

    output_metadata_json:
        Methodological metadata.

    source_name:
        Human-readable domain/source label.

        Examples:

            movies
            music
            restaurants

    model_name:
        Sentence-BERT model.

        Method B should keep this identical to the review
        embedding model.

    batch_size:
        SentenceTransformer encoding batch size.

    encoding_chunk_size:
        Number of entity texts passed through successive
        encoding chunks.

    device:
        Usually:

            cuda

        for the current project machine.

    Returns
    -------
    embeddings:
        np.ndarray
        Shape:

            (n_entities, 384)

        with the current MiniLM model.

    entity_data:
        pd.DataFrame containing IDs and text-availability
        diagnostics.

    metadata:
        dict
    """

    _section(
        "BUILDING BASE SEMANTIC EMBEDDINGS"
    )


    print(
        f"Source:            "
        f"{source_name}"
    )


    print(
        f"Entity ID:         "
        f"{entity_id_column}"
    )


    print(
        f"Text column:       "
        f"{text_column}"
    )


    print(
        f"Model:             "
        f"{model_name}"
    )


    print(
        f"Entities received: "
        f"{len(df):,}"
    )


    # ========================================================
    # 1. VALIDATE
    # ========================================================

    _validate_dataframe(

        df,

        entity_id_column,

        text_column

    )


    # ========================================================
    # 2. PREPARE ENTITY DATA
    # ========================================================

    _section(
        "PREPARING ENTITY SEMANTIC TEXT"
    )


    entity_data = (
        _prepare_entity_data(

            df,

            entity_id_column,

            text_column

        )
    )


    n_entities = len(
        entity_data
    )


    n_with_text = int(

        entity_data[
            "has_semantic_text"
        ].sum()

    )


    print(
        f"Entities:               "
        f"{n_entities:,}"
    )


    print(
        f"With semantic text:     "
        f"{n_with_text:,}"
    )


    print(
        f"Without semantic text:  "
        f"{n_entities - n_with_text:,}"
    )


    if n_entities > 0:

        print(

            f"Text coverage:          "
            f"{(
                n_with_text
                /
                n_entities
                *
                100
            ):.2f}%"

        )


    print()


    if n_with_text > 0:

        text_lengths = (

            entity_data.loc[

                entity_data[
                    "has_semantic_text"
                ],

                "text_word_count"

            ]

        )


        print(
            "Semantic text word counts:"
        )


        print(

            text_lengths
            .describe(

                percentiles=[

                    0.25,

                    0.50,

                    0.75,

                    0.90,

                    0.95,

                    0.99,

                ]

            )
            .to_string()

        )


    # ========================================================
    # 3. LOAD MODEL
    # ========================================================

    (
        model,
        embedding_dimension

    ) = _load_model(

        model_name,

        device

    )


    # ========================================================
    # 4. ENCODE
    # ========================================================

    embeddings = (
        _encode_semantic_texts(

            model,

            entity_data,

            embedding_dimension,

            batch_size,

            encoding_chunk_size

        )
    )


    # ========================================================
    # 5. VALIDATE EMBEDDINGS
    # ========================================================

    _section(
        "VALIDATING BASE EMBEDDINGS"
    )


    if embeddings.shape[0] != n_entities:

        raise ValueError(

            "Embedding row count does not match entity count."

        )


    if embeddings.shape[1] != embedding_dimension:

        raise ValueError(

            "Embedding dimensionality does not match "
            "Sentence-BERT model dimensionality."

        )


    embedding_norms = np.linalg.norm(

        embeddings,

        axis=1

    )


    has_text = (

        entity_data[
            "has_semantic_text"
        ]
        .to_numpy()

    )


    # --------------------------------------------------------
    # Every entity with text should have a non-zero vector.
    # --------------------------------------------------------

    invalid_nonzero = (

        has_text

        &

        (
            embedding_norms
            <=
            0
        )

    )


    if invalid_nonzero.any():

        raise ValueError(

            f"{int(invalid_nonzero.sum()):,} entities contain "
            "semantic text but received zero embeddings."

        )


    # --------------------------------------------------------
    # Every entity without text should remain exactly zero.
    # --------------------------------------------------------

    invalid_empty = (

        (~has_text)

        &

        (
            embedding_norms
            >
            0
        )

    )


    if invalid_empty.any():

        raise ValueError(

            f"{int(invalid_empty.sum()):,} entities without "
            "semantic text received non-zero embeddings."

        )


    print(
        f"Embedding shape:       "
        f"{embeddings.shape}"
    )


    print(
        f"Non-zero embeddings:   "
        f"{int((embedding_norms > 0).sum()):,}"
    )


    print(
        f"Zero embeddings:       "
        f"{int((embedding_norms == 0).sum()):,}"
    )


    # ========================================================
    # 6. SAVE
    # ========================================================

    _section(
        "SAVING BASE SEMANTIC EMBEDDINGS"
    )


    metadata = _save_outputs(

        entity_data=
            entity_data,

        embeddings=
            embeddings,

        output_npz=
            output_npz,

        output_index_csv=
            output_index_csv,

        output_metadata_json=
            output_metadata_json,

        model_name=
            model_name,

        source_name=
            source_name,

        entity_id_column=
            entity_id_column,

        text_column=
            text_column,

        device=
            device,

        batch_size=
            batch_size,

        encoding_chunk_size=
            encoding_chunk_size

    )


    print(
        "Embedding output:"
    )

    print(
        output_npz
    )


    print()


    print(
        "Index output:"
    )

    print(
        output_index_csv
    )


    print()


    print(
        "Metadata output:"
    )

    print(
        output_metadata_json
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    _section(
        "BASE SEMANTIC EMBEDDINGS COMPLETE"
    )


    print(
        f"Source:                "
        f"{source_name}"
    )


    print(
        f"Entities:              "
        f"{n_entities:,}"
    )


    print(
        f"Entities embedded:     "
        f"{n_with_text:,}"
    )


    print(
        f"Entities with no text: "
        f"{n_entities - n_with_text:,}"
    )


    print(
        f"Embedding shape:       "
        f"{embeddings.shape}"
    )


    print()


    return (
        embeddings,
        entity_data,
        metadata
    )