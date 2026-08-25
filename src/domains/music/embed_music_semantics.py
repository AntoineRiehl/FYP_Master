# src/domains/music/embed_music_semantics.py

# ============================================================
# MUSIC BASE SEMANTIC EMBEDDINGS
# ============================================================
#
# Purpose
# -------
#
# Build one Sentence-BERT base semantic embedding per artist
# using the artist's existing non-review semantic tag
# representation.
#
#
# This is part of Method B:
#
#     Shared Experiential / Feel Space
#
#
# Representation
# --------------
#
# Raw music data
#      ↓
# filter_artists()
#      ↓
# create_tags_text()
#      ↓
# tags_text
#      ↓
# sentence-transformers/all-MiniLM-L6-v2
#      ↓
# 384D base semantic artist embedding
#
#
# These embeddings inhabit the SAME semantic space as:
#
#     - CritiqueBrainz review embeddings
#     - Movie base embeddings
#     - Restaurant base embeddings
#     - Feel-anchor embeddings
#
#
# Later:
#
#     base semantic embedding
#              +
#     review semantic embedding
#              ↓
#     shared artist semantic vector
#              ↓
#        feel-anchor projection
#
#
# IMPORTANT
# ---------
#
# Reviews are NOT used in this script.
#
# This script only builds the non-review semantic component.
#
# ============================================================


from pathlib import Path


# ============================================================
# MUSIC DATA
# ============================================================

from src.domains.music.load_data import (
    load_raw_data
)

from src.domains.music.feature_engineering import (
    filter_artists,
    create_tags_text,
    compute_popularity_score
)


# ============================================================
# GENERIC FEEL-SPACE EMBEDDING ENGINE
# ============================================================

from src.atlas.cross_domain.feel_space.base_embeddings import (
    build_base_semantic_embeddings
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "feel_space"
    / "music"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Base semantic embedding matrix
# ------------------------------------------------------------

OUTPUT_NPZ = (
    OUTPUT_DIR
    / "music_base_semantic_embeddings.npz"
)


# ------------------------------------------------------------
# Human-readable embedding index
# ------------------------------------------------------------

OUTPUT_INDEX = (
    OUTPUT_DIR
    / "music_base_semantic_embedding_index.csv"
)


# ------------------------------------------------------------
# Methodological metadata
# ------------------------------------------------------------

OUTPUT_METADATA = (
    OUTPUT_DIR
    / "music_base_semantic_embeddings_metadata.json"
)


# ============================================================
# CONFIGURATION
# ============================================================


# ------------------------------------------------------------
# Same population threshold used by the existing Music atlas
# and cross-domain pipelines.
# ------------------------------------------------------------

MIN_LISTENERS = 1000


# ------------------------------------------------------------
# Sentence-BERT encoding settings
# ------------------------------------------------------------

BATCH_SIZE = 64


ENCODING_CHUNK_SIZE = 10_000


DEVICE = "cuda"


# ============================================================
# LOAD AND PREPARE MUSIC
# ============================================================


def prepare_music():
    """
    Reconstruct the same Music semantic population used by
    the current Music atlas pipeline.

    Returns
    -------
    pandas.DataFrame

        Contains at minimum:

            mbid
            tags_text
    """

    print()

    print(
        "=" * 65
    )

    print(
        "PREPARING MUSIC BASE SEMANTIC DATA"
    )

    print(
        "=" * 65
    )

    print()


    # ========================================================
    # LOAD RAW MUSIC DATA
    # ========================================================

    music = load_raw_data()


    print(
        f"Raw artists: "
        f"{len(music):,}"
    )


    # ========================================================
    # FILTER ARTISTS
    # ========================================================

    music = filter_artists(

        music,

        min_listeners=
            MIN_LISTENERS

    )


    print(
        f"Artists after listener filter: "
        f"{len(music):,}"
    )


    # ========================================================
    # CREATE EXISTING MUSIC SEMANTIC TEXT
    # ========================================================

    music = create_tags_text(
        music
    )


    music["tags_text"] = (

        music["tags_text"]
        .fillna("")

    )


    # --------------------------------------------------------
    # Popularity itself is not part of the semantic embedding,
    # but keeping this preprocessing step aligned with the
    # existing Music pipeline makes the reconstructed domain
    # population consistent with the current atlas workflow.
    # --------------------------------------------------------

    music = compute_popularity_score(
        music
    )


    # ========================================================
    # STANDARDIZE MBIDs
    # ========================================================

    music["mbid"] = (

        music["mbid"]
        .astype(str)
        .str.strip()

    )


    # ========================================================
    # VALIDATION
    # ========================================================

    invalid_ids = (

        music[
            "mbid"
        ]
        ==
        ""

    )


    if invalid_ids.any():

        raise ValueError(

            f"{int(invalid_ids.sum()):,} artists contain "
            "empty MBIDs."

        )


    if music["mbid"].duplicated().any():

        duplicate_count = int(

            music[
                "mbid"
            ]
            .duplicated()
            .sum()

        )


        raise ValueError(

            f"Prepared Music dataframe contains "
            f"{duplicate_count:,} duplicate MBIDs."

        )


    artists_with_tags = int(

        (
            music[
                "tags_text"
            ]
            .str.strip()
            !=
            ""
        ).sum()

    )


    artists_without_tags = (

        len(
            music
        )

        -

        artists_with_tags

    )


    print()

    print(
        f"Prepared artists:       "
        f"{len(music):,}"
    )


    print(
        f"Artists with tags:      "
        f"{artists_with_tags:,}"
    )


    print(
        f"Artists without tags:   "
        f"{artists_without_tags:,}"
    )


    if len(
        music
    ) > 0:

        coverage = (

            artists_with_tags

            /

            len(
                music
            )

            *

            100

        )


        print(
            f"Tag coverage:           "
            f"{coverage:.2f}%"
        )


    print()


    return music


# ============================================================
# MAIN
# ============================================================


def main():

    print()

    print(
        "=" * 65
    )

    print(
        "MUSIC BASE SEMANTIC EMBEDDINGS"
    )

    print(
        "=" * 65
    )

    print()


    # ========================================================
    # 1. RECONSTRUCT MUSIC SEMANTIC DATA
    # ========================================================

    music = prepare_music()


    # ========================================================
    # 2. BUILD BASE SENTENCE-BERT EMBEDDINGS
    # ========================================================

    (
        embeddings,
        entity_data,
        metadata

    ) = build_base_semantic_embeddings(

        # ----------------------------------------------------
        # Entity dataframe
        # ----------------------------------------------------

        df=
            music,


        # ----------------------------------------------------
        # Canonical MusicBrainz artist identifier
        # ----------------------------------------------------

        entity_id_column=
            "mbid",


        # ----------------------------------------------------
        # Existing non-review semantic representation
        # ----------------------------------------------------

        text_column=
            "tags_text",


        # ----------------------------------------------------
        # Outputs
        # ----------------------------------------------------

        output_npz=
            OUTPUT_NPZ,

        output_index_csv=
            OUTPUT_INDEX,

        output_metadata_json=
            OUTPUT_METADATA,


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        source_name=
            "music",


        # ----------------------------------------------------
        # Encoding
        # ----------------------------------------------------

        batch_size=
            BATCH_SIZE,

        encoding_chunk_size=
            ENCODING_CHUNK_SIZE,

        device=
            DEVICE

    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=" * 65
    )

    print(
        "MUSIC BASE SEMANTIC EMBEDDING COMPLETE"
    )

    print(
        "=" * 65
    )

    print()


    print(
        f"Artists represented:       "
        f"{len(entity_data):,}"
    )


    print(
        f"Artists with base text:    "
        f"{int(entity_data['has_semantic_text'].sum()):,}"
    )


    print(
        f"Artists without base text: "
        f"{int((~entity_data['has_semantic_text']).sum()):,}"
    )


    print(
        f"Embedding shape:           "
        f"{embeddings.shape}"
    )


    print()


    print(
        "Embedding output:"
    )

    print(
        OUTPUT_NPZ
    )


    print()


    print(
        "Index output:"
    )

    print(
        OUTPUT_INDEX
    )


    print()


    print(
        "Metadata output:"
    )

    print(
        OUTPUT_METADATA
    )


    print()


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()