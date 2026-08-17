#src/pipelines/build_movie_map.py

from pathlib import Path

from src.domains.movies.load_data import load_raw_data

from src.domains.movies.merge_data import (
    compute_movie_stats,
    merge_movies
)

from src.domains.movies.feature_engineering import (
    compute_weighted_rating,
    concatenate_tags,
    create_macro_genres,
    create_visual_sizes
)

from src.atlas.embeddings.tfidf_pipeline import (
    get_tfidf_embeddings
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

    name="movies_tags",

    use_tags=True,

    use_categories=True,

    use_reviews=False,

    use_metadata=True,

    use_statistics=True

)


# =========================================================
# VISUAL CONFIG
# =========================================================

VISUAL_SIZE_STRENGTH = 1.8


# =========================================================
# LOAD DATA
# =========================================================

print("\n[1/8] Loading raw data...")

movies, ratings, tags = load_raw_data()


# =========================================================
# PREPROCESSING
# =========================================================

print("\n[2/8] Computing movie statistics...")

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


# =========================================================
# TAG PROCESSING
# =========================================================

print("\n[3/8] Processing tags...")

movie_tags = concatenate_tags(tags)

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
# TF-IDF
# =========================================================

print("\n[4/8] Building TF-IDF embeddings...")

tfidf_matrix, vectorizer = (
    get_tfidf_embeddings(
        final,
        text_column="tags_text",
        model_name="movies_tags"
    )
)


# =========================================================
# UMAP
# =========================================================

print("\n[5/8] Computing UMAP projection...")

embedding, umap_model = (
    get_umap_projection(
        tfidf_matrix
    )
)

final["umap_x"] = embedding[:, 0]

final["umap_y"] = embedding[:, 1]


# =========================================================
# FEATURE ENGINEERING
# =========================================================

print("\n[6/8] Creating atlas features...")

final = create_macro_genres(
    final
)

final = create_visual_sizes(

    final,

    strength=VISUAL_SIZE_STRENGTH

)


# =========================================================
# CLUSTERING
# =========================================================

print("\n[7/8] Computing clusters...")

final = compute_clusters(
    final
)

final = create_region_labels(
    final
)


# =========================================================
# EXPORT ANALYSIS DATASET
# =========================================================

print("\nSaving analysis dataframe...")

final.to_csv(

    OUTPUT_PATH,

    index=False

)


# =========================================================
# BUILD ATLAS BUNDLE
# =========================================================

print("\n[8/8] Building AtlasBundle...")

bundle = build_bundle(

    df=final,

    domain="movies",

    feature_config=FEATURE_CONFIG,

    metadata={

        "pipeline": "movie_map",

        "visual_size_strength": VISUAL_SIZE_STRENGTH,

        "embedding": "TF-IDF",

        "projection": "UMAP",

        "items": len(final),

        "clusters": int(final["cluster"].nunique())

    }

)


# =========================================================
# EXPORT FRONTEND DATA
# =========================================================

print("\nExporting AtlasBundle...")

export_bundle(

    bundle,

    JSON_DIR

)


print("\n✅ Movie Atlas complete!")