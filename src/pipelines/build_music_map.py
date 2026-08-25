# src/pipelines/build_music_map.py

import pandas as pd

from pathlib import Path


from src.domains.music.load_data import (
    load_raw_data
)

from src.domains.music.feature_engineering import (
    filter_artists,
    create_tags_text,
    compute_popularity_score,
    create_macro_genres
)

from src.atlas.visual.size_normalization import (
    normalize_visual_sizes
)

from src.atlas.embeddings.tfidf_pipeline import (
    get_tfidf_embeddings
)

from src.atlas.embeddings.semantic_fusion import (
    fuse_semantic_embeddings
)

from src.atlas.embeddings.dimensionality_reduction import (
    get_umap_projection
)

from src.atlas.clustering.clustering import (
    compute_clusters
)

from src.atlas.clustering.region_labels import (
    create_region_labels
)

from src.atlas.schema.feature_config import (
    FeatureConfig
)

from src.atlas.builders.build_bundle import (
    build_bundle
)

from src.atlas.export.export_bundle import (
    export_bundle
)


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------
# Analysis dataframe
# ---------------------------------------------------------

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "music_map_v1.csv"
)


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Sentence-BERT review embeddings
# ---------------------------------------------------------

MUSIC_REVIEW_EMBEDDINGS = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "music"
    / "music_review_embeddings.npz"
)


# ---------------------------------------------------------
# Review-level summary metadata
# ---------------------------------------------------------

MUSIC_REVIEW_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "reviews"
    / "music_review_summary.csv"
)


# ---------------------------------------------------------
# Frontend bundle
# ---------------------------------------------------------

JSON_DIR = (
    ROOT
    / "frontend"
    / "public"
    / "data"
    / "music"
)


# =========================================================
# FEATURE CONFIGURATION
# =========================================================

FEATURE_CONFIG = FeatureConfig(

    name="music_tags_reviews",

    use_tags=True,

    use_categories=True,

    use_reviews=True,

    use_metadata=True,

    use_statistics=True

)


# =========================================================
# DATA CONFIGURATION
# =========================================================

MIN_LISTENERS = 1000


# =========================================================
# SEMANTIC CONFIGURATION
# =========================================================

# ---------------------------------------------------------
# TF-IDF representation
#
# Existing artist tags are first represented using TF-IDF.
#
# Before fusion, the TF-IDF representation is reduced with
# TruncatedSVD so that the sparse 5,000-dimensional TF-IDF
# space can be combined with the dense 384-dimensional
# Sentence-BERT review representation.
# ---------------------------------------------------------

TFIDF_COMPONENTS = 256


# ---------------------------------------------------------
# Review contribution
#
# Selected using:
#
#   src/domains/music/evaluate_fusion_weights.py
#
# Diagnostic evaluation:
#
#   Review share 0.00 -> balanced score 0.7844
#   Review share 0.20 -> balanced score 0.7904
#   Review share 0.35 -> balanced score 0.7945
#   Review share 0.50 -> balanced score 0.7970
#   Review share 0.65 -> balanced score 0.7959
#   Review share 0.80 -> balanced score 0.7853
#   Review share 1.00 -> balanced score 0.7235
#
# The highest score was obtained at 0.50.
#
# The surrounding 0.35–0.65 region was relatively flat,
# therefore 0.50 should be interpreted as a simple,
# balanced and empirically supported configuration rather
# than an exact ground-truth optimum.
# ---------------------------------------------------------

REVIEW_SHARE = 0.50


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

RANDOM_STATE = 42


# =========================================================
# VISUAL CONFIGURATION
# =========================================================

VISUAL_SIZE_STRENGTH = 1.8


# =========================================================
# VALIDATE REVIEW FILES
# =========================================================

if not MUSIC_REVIEW_EMBEDDINGS.exists():

    raise FileNotFoundError(

        "Music review embeddings were not found:\n"
        f"{MUSIC_REVIEW_EMBEDDINGS}\n\n"

        "Run:\n"
        "python -m src.domains.music.embed_music_reviews\n"
        "before building the enriched music atlas."

    )


if not MUSIC_REVIEW_SUMMARY.exists():

    raise FileNotFoundError(

        "Music review summary was not found:\n"
        f"{MUSIC_REVIEW_SUMMARY}\n\n"

        "Run:\n"
        "python -m src.domains.music.prepare_music_reviews\n"
        "before building the enriched music atlas."

    )


# =========================================================
# LOAD DATA
# =========================================================

print(
    "\n[1/9] Loading music dataset..."
)

music = load_raw_data()


# =========================================================
# PREPROCESSING
# =========================================================

print(
    "\n[2/9] Filtering artists..."
)

music = filter_artists(

    music,

    min_listeners=
        MIN_LISTENERS

)


print(
    f"Artists remaining: "
    f"{len(music):,}"
)


# =========================================================
# TEXT PROCESSING
# =========================================================

print(
    "\n[3/9] Processing artist tags..."
)

music = create_tags_text(
    music
)


music["tags_text"] = (

    music["tags_text"]
    .fillna("")

)


# =========================================================
# REVIEW METADATA
# =========================================================

print(
    "\nLoading music review metadata..."
)


review_summary = pd.read_csv(

    MUSIC_REVIEW_SUMMARY,

    usecols=[
        "artist_mbid",
        "review_count"
    ]

)


# ---------------------------------------------------------
# Standardize MBIDs before joining.
#
# MusicBrainz MBIDs are canonical string identifiers.
# ---------------------------------------------------------

music["mbid"] = (

    music["mbid"]
    .astype(str)
    .str.strip()

)


review_summary["artist_mbid"] = (

    review_summary[
        "artist_mbid"
    ]
    .astype(str)
    .str.strip()

)


music = music.merge(

    review_summary,

    left_on=
        "mbid",

    right_on=
        "artist_mbid",

    how=
        "left"

)


# ---------------------------------------------------------
# artist_mbid was only required for the merge.
#
# mbid remains the canonical identifier used throughout
# the existing Music atlas pipeline.
# ---------------------------------------------------------

music = music.drop(

    columns=[
        "artist_mbid"
    ],

    errors="ignore"

)


music["review_count"] = (

    music["review_count"]
    .fillna(0)
    .astype(int)

)


print(
    f"Artists with CritiqueBrainz reviews: "
    f"{int((music['review_count'] > 0).sum()):,}"
)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

print(
    "\nCreating atlas features..."
)


music = compute_popularity_score(
    music
)


music = normalize_visual_sizes(

    music,

    strength=
        VISUAL_SIZE_STRENGTH

)


music = create_macro_genres(
    music
)


# =========================================================
# TF-IDF
# =========================================================

print(
    "\n[4/9] Building TF-IDF embeddings..."
)


tfidf_matrix, vectorizer = (
    get_tfidf_embeddings(

        music,

        text_column=
            "tags_text",

        model_name=
            "music_tags"

    )
)


# =========================================================
# SEMANTIC FUSION
# =========================================================

print(
    "\n[5/9] Fusing TF-IDF and review semantics..."
)


(
    semantic_matrix,
    fusion_info,
    review_counts

) = fuse_semantic_embeddings(

    tfidf_matrix=
        tfidf_matrix,

    entity_ids=
        music["mbid"].to_numpy(),

    review_embeddings_path=
        MUSIC_REVIEW_EMBEDDINGS,

    review_share=
        REVIEW_SHARE,

    tfidf_components=
        TFIDF_COMPONENTS,

    random_state=
        RANDOM_STATE,

    return_info=True

)


# ---------------------------------------------------------
# Store review availability in the analysis dataframe.
#
# review_count:
#
#     total number of prepared CritiqueBrainz reviews
#     available for the artist.
#
#
# reviews_used_for_embedding:
#
#     number of reviews actually used by Sentence-BERT.
#
#     This can differ from review_count because the reusable
#     review embedding pipeline caps the number of reviews
#     used per entity at 50.
#
#
# has_review_embedding:
#
#     whether review-derived semantics contributed to the
#     final representation of the artist.
# ---------------------------------------------------------

music["reviews_used_for_embedding"] = (
    review_counts
)


music["has_review_embedding"] = (
    review_counts > 0
)


# ---------------------------------------------------------
# Information
# ---------------------------------------------------------

print()

print(
    "Semantic fusion summary:"
)


print(
    f"  TF-IDF reduced dimensions: "
    f"{fusion_info['tfidf_components']}"
)


print(
    f"  Review dimensions:         "
    f"{fusion_info['review_components']}"
)


print(
    f"  Final fused dimensions:    "
    f"{fusion_info['fused_dimensions']}"
)


print(
    f"  Artists with reviews:      "
    f"{fusion_info['entities_with_reviews']:,}"
)


print(
    f"  Review share:              "
    f"{REVIEW_SHARE:.2f}"
)


print(
    f"  TF-IDF share:              "
    f"{1.0 - REVIEW_SHARE:.2f}"
)


# =========================================================
# UMAP
# =========================================================

print(
    "\n[6/9] Computing UMAP projection "
    "from fused semantic representation..."
)


embedding, umap_model = (
    get_umap_projection(
        semantic_matrix
    )
)


music["umap_x"] = (
    embedding[:, 0]
)


music["umap_y"] = (
    embedding[:, 1]
)


# =========================================================
# CLUSTERING
# =========================================================

print(
    "\n[7/9] Computing clusters..."
)


music = compute_clusters(
    music
)


music = create_region_labels(
    music
)


# =========================================================
# EXPORT ANALYSIS DATASET
# =========================================================

print(
    "\n[8/9] Saving analysis dataframe..."
)


music.to_csv(

    OUTPUT_PATH,

    index=False

)


# =========================================================
# BUILD ATLAS BUNDLE
# =========================================================

print(
    "\n[9/9] Building AtlasBundle..."
)


bundle = build_bundle(

    df=
        music,

    domain=
        "music",

    feature_config=
        FEATURE_CONFIG,

    metadata={

        # -------------------------------------------------
        # Pipeline
        # -------------------------------------------------

        "pipeline":
            "music_map",


        "minimum_listeners":
            MIN_LISTENERS,


        # -------------------------------------------------
        # Semantic representation
        # -------------------------------------------------

        "embedding":
            "TF-IDF + Sentence-BERT reviews",


        "tfidf_model":
            "music_tags",


        "tfidf_original_dimensions":
            int(
                tfidf_matrix.shape[1]
            ),


        "tfidf_svd_components":
            int(
                fusion_info[
                    "tfidf_components"
                ]
            ),


        "tfidf_svd_explained_variance":
            float(
                fusion_info[
                    "svd_explained_variance"
                ]
            ),


        "review_embedding_model":
            "sentence-transformers/all-MiniLM-L6-v2",


        "review_embedding_dimensions":
            int(
                fusion_info[
                    "review_components"
                ]
            ),


        "review_share":
            REVIEW_SHARE,


        "tfidf_share":
            1.0 - REVIEW_SHARE,


        "fused_dimensions":
            int(
                fusion_info[
                    "fused_dimensions"
                ]
            ),


        "items_with_review_embeddings":
            int(
                fusion_info[
                    "entities_with_reviews"
                ]
            ),


        # -------------------------------------------------
        # Review methodology
        # -------------------------------------------------

        "review_source":
            "CritiqueBrainz",


        "max_reviews_per_artist":
            50,


        "review_aggregation":
            "mean pooling",


        "review_embedding_normalization":
            "L2",


        # -------------------------------------------------
        # Projection
        # -------------------------------------------------

        "projection":
            "UMAP",


        "umap_metric":
            "cosine",


        "umap_n_neighbors":
            15,


        "umap_min_dist":
            0.1,


        # -------------------------------------------------
        # Visual configuration
        # -------------------------------------------------

        "visual_size_strength":
            VISUAL_SIZE_STRENGTH,


        # -------------------------------------------------
        # Atlas information
        # -------------------------------------------------

        "items":
            len(music),


        "clusters":
            int(
                music[
                    "cluster"
                ]
                .nunique()
            )

    }

)


# =========================================================
# EXPORT FRONTEND DATA
# =========================================================

print(
    "\nExporting AtlasBundle..."
)


export_bundle(

    bundle,

    JSON_DIR

)


# =========================================================
# COMPLETE
# =========================================================

print()

print(
    "=" * 60
)

print(
    "✅ ENRICHED MUSIC ATLAS COMPLETE"
)

print(
    "=" * 60
)

print()


print(
    f"Artists: "
    f"{len(music):,}"
)


print(
    f"Artists with total review metadata: "
    f"{int((music['review_count'] > 0).sum()):,}"
)


print(
    f"Artists with review embeddings: "
    f"{int((review_counts > 0).sum()):,}"
)


print(
    f"TF-IDF / Review semantic share: "
    f"{1.0 - REVIEW_SHARE:.2f} / "
    f"{REVIEW_SHARE:.2f}"
)


print(
    f"Fused semantic dimensions: "
    f"{semantic_matrix.shape[1]}"
)


print(
    f"Clusters: "
    f"{music['cluster'].nunique():,}"
)


print()

print(
    "Analysis dataframe:"
)

print(
    OUTPUT_PATH
)


print()

print(
    "Frontend data:"
)

print(
    JSON_DIR
)


print()