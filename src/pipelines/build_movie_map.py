# src/pipelines/build_movie_map.py

import pandas as pd

from pathlib import Path

from src.domains.movies.load_data import load_raw_data

from src.domains.movies.merge_data import (
    compute_movie_stats,
    merge_movies
)

from src.domains.movies.feature_engineering import (
    compute_weighted_rating,
    concatenate_tags,
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
    / "movie_map_v1.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Sentence-BERT review embeddings
# ---------------------------------------------------------

MOVIE_REVIEW_EMBEDDINGS = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "movies"
    / "movie_review_embeddings.npz"
)

MOVIE_REVIEW_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "reviews"
    / "movie_review_summary.csv"
)

# ---------------------------------------------------------
# Frontend bundle
# ---------------------------------------------------------

JSON_DIR = (
    ROOT
    / "frontend"
    / "public"
    / "data"
    / "movies"
)


# =========================================================
# FEATURE CONFIGURATION
# =========================================================

FEATURE_CONFIG = FeatureConfig(

    name="movies_tags_reviews",

    use_tags=True,

    use_categories=True,

    use_reviews=True,

    use_metadata=True,

    use_statistics=True

)


# =========================================================
# SEMANTIC CONFIGURATION
# =========================================================

# ---------------------------------------------------------
# TF-IDF representation
#
# Raw TF-IDF:
#     up to 5,000 dimensions
#
# Before fusion it is reduced using TruncatedSVD.
# ---------------------------------------------------------

TFIDF_COMPONENTS = 256


# ---------------------------------------------------------
# Review contribution
#
# Selected using:
#
#   src/domains/movies/evaluate_fusion_weights.py
#
# Evaluation result:
#
#   REVIEW_SHARE = 0.50
#   TFIDF_SHARE  = 0.50
#
# This produced the highest balanced neighbourhood
# coherence among the tested configurations.
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
# VALIDATE REVIEW EMBEDDINGS
# =========================================================

if not MOVIE_REVIEW_EMBEDDINGS.exists():

    raise FileNotFoundError(

        "Movie review embeddings were not found:\n"
        f"{MOVIE_REVIEW_EMBEDDINGS}\n\n"

        "Run:\n"
        "python -m src.domains.movies.embed_movie_reviews\n"
        "before building the enriched movie atlas."

    )


# =========================================================
# LOAD DATA
# =========================================================

print(
    "\n[1/9] Loading raw data..."
)

movies, ratings, tags = load_raw_data()


# =========================================================
# PREPROCESSING
# =========================================================

print(
    "\n[2/9] Computing movie statistics..."
)

movie_stats = compute_movie_stats(
    ratings
)

merged = merge_movies(
    movie_stats,
    movies
)

merged = compute_weighted_rating(
    merged
)

merged["popularity_score"] = (
    merged["rating_count"]
)


# =========================================================
# TAG PROCESSING
# =========================================================

print(
    "\n[3/9] Processing tags..."
)

movie_tags = concatenate_tags(
    tags
)

final = merged.merge(

    movie_tags,

    on="movieId",

    how="left"

)

final["tags_text"] = (

    final["tags_text"]
    .fillna("")

)

# =========================================================
# REVIEW METADATA
# =========================================================

print(
    "\nLoading movie review metadata..."
)

review_summary = pd.read_csv(

    MOVIE_REVIEW_SUMMARY,

    usecols=[
        "movieId",
        "review_count"
    ]

)

final = final.merge(

    review_summary,

    on="movieId",

    how="left"

)

final["review_count"] = (

    final["review_count"]
    .fillna(0)
    .astype(int)

)

# =========================================================
# TF-IDF
# =========================================================

print(
    "\n[4/9] Building TF-IDF embeddings..."
)

tfidf_matrix, vectorizer = (
    get_tfidf_embeddings(

        final,

        text_column="tags_text",

        model_name="movies_tags"

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
        final["movieId"].to_numpy(),

    review_embeddings_path=
        MOVIE_REVIEW_EMBEDDINGS,

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
# review_counts refers to the number of reviews ACTUALLY
# used to construct the Sentence-BERT movie embedding.
#
# Because embeddings were capped at 50 reviews per movie,
# this is intentionally called reviews_used_for_embedding
# rather than total_review_count.
# ---------------------------------------------------------

final["reviews_used_for_embedding"] = (
    review_counts
)

final["has_review_embedding"] = (
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
    f"  Movies with reviews:       "
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

final["umap_x"] = (
    embedding[:, 0]
)

final["umap_y"] = (
    embedding[:, 1]
)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

print(
    "\n[7/9] Creating atlas features..."
)

final = create_macro_genres(
    final
)

final = normalize_visual_sizes(

    final,

    strength=
        VISUAL_SIZE_STRENGTH

)


# =========================================================
# CLUSTERING
# =========================================================

print(
    "\n[8/9] Computing clusters..."
)

final = compute_clusters(
    final
)

final = create_region_labels(
    final
)


# =========================================================
# EXPORT ANALYSIS DATASET
# =========================================================

print(
    "\nSaving analysis dataframe..."
)

final.to_csv(

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

    df=final,

    domain="movies",

    feature_config=
        FEATURE_CONFIG,

    metadata={

        # -------------------------------------------------
        # Pipeline
        # -------------------------------------------------

        "pipeline":
            "movie_map",

        # -------------------------------------------------
        # Semantic representation
        # -------------------------------------------------

        "embedding":
            "TF-IDF + Sentence-BERT reviews",

        "tfidf_model":
            "movies_tags",

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
            len(final),

        "clusters":
            int(
                final["cluster"]
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
    "✅ ENRICHED MOVIE ATLAS COMPLETE"
)
print(
    "=" * 60
)

print()

print(
    f"Movies: "
    f"{len(final):,}"
)

print(
    f"Movies with review embeddings: "
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
    f"{final['cluster'].nunique():,}"
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