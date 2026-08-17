#src/pipelines/build_movies_music_atlas.py

from pathlib import Path

from src.domains.movies.load_data import load_raw_data as load_movie_data

from src.domains.movies.merge_data import (
    compute_movie_stats,
    merge_movies
)

from src.domains.movies.feature_engineering import (
    compute_weighted_rating,
    concatenate_tags
)

from src.domains.music.load_data import load_raw_data as load_music_data

from src.domains.music.feature_engineering import (
    filter_artists,
    create_tags_text,
    compute_popularity_score,
    create_visual_sizes
)

from src.atlas.cross_domain.combine_data import (
    combine_domain_data
)

from src.atlas.cross_domain.semantic_text import (
    create_semantic_text
)

from src.atlas.cross_domain.domain_vocabulary import (
    get_domain_vocabulary
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
    / "movies_music_atlas_v1.csv"
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
    / "movies_music"
)


# =========================================================
# FEATURE CONFIGURATION
# =========================================================

FEATURE_CONFIG = FeatureConfig(

    name="movies_music_semantic",

    use_tags=True,

    use_categories=True,

    use_reviews=False,

    use_metadata=True,

    use_statistics=True

)


# =========================================================
# CONFIGURATION
# =========================================================

MIN_LISTENERS = 1000

VISUAL_SIZE_STRENGTH = 1.8

DOMAINS = [
    "movies",
    "music"
]


# =========================================================
# LOAD DATA
# =========================================================

print("\n[1/10] Loading raw datasets...")


movies, ratings, tags = (
    load_movie_data()
)


music = (
    load_music_data()
)


# =========================================================
# PREPROCESS MOVIES
# =========================================================

print("\n[2/10] Preprocessing movies...")


movie_stats = compute_movie_stats(
    ratings
)


movies = merge_movies(
    movie_stats,
    movies
)


movies = compute_weighted_rating(
    movies
)


movie_tags = concatenate_tags(
    tags
)


movies = movies.merge(
    movie_tags,
    on="movieId",
    how="left"
)


movies["tags_text"] = (
    movies["tags_text"]
    .fillna("")
)


# =========================================================
# PREPROCESS MUSIC
# =========================================================

print("\n[3/10] Preprocessing music...")


music = filter_artists(
    music,
    min_listeners=MIN_LISTENERS
)


print(
    f"Artists remaining: {len(music):,}"
)


music = create_tags_text(
    music
)


music = compute_popularity_score(
    music
)


music = create_visual_sizes(
    music,
    strength=VISUAL_SIZE_STRENGTH
)


# =========================================================
# COMBINE DOMAINS
# =========================================================

print("\n[4/10] Combining domain datasets...")


combined = combine_domain_data(

    {
        "movies": movies,
        "music": music
    }

)


print(
    f"Combined items: {len(combined):,}"
)


print(
    "\nItems by domain:"
)


print(
    combined["domain"]
    .value_counts()
)


# =========================================================
# DOMAIN-NEUTRAL SEMANTIC TEXT
# =========================================================

print("\n[5/10] Building domain-neutral semantic text...")


domain_vocabulary = (
    get_domain_vocabulary(
        DOMAINS
    )
)


combined = create_semantic_text(

    combined,

    domain_vocabulary=domain_vocabulary

)


# =========================================================
# TF-IDF
# =========================================================

print("\n[6/10] Building cross-domain TF-IDF embeddings...")


tfidf_matrix, vectorizer = (
    get_tfidf_embeddings(
        combined,
        text_column="semantic_text",
        model_name="movies_music_semantic"
    )
)


# =========================================================
# UMAP
# =========================================================

print("\n[7/10] Computing cross-domain UMAP projection...")


embedding, umap_model = (
    get_umap_projection(
        tfidf_matrix
    )
)


combined["umap_x"] = (
    embedding[:, 0]
)


combined["umap_y"] = (
    embedding[:, 1]
)


# =========================================================
# VISUAL FEATURES
# =========================================================

print("\n[8/10] Creating atlas visual features...")

combined = create_visual_sizes(
    combined,
    strength=VISUAL_SIZE_STRENGTH
)

# =========================================================
# CLUSTERING
# =========================================================

print("\n[9/10] Computing cross-domain clusters...")

combined = compute_clusters(
    combined
)

combined = create_region_labels(
    combined
)


# =========================================================
# EXPORT ANALYSIS DATASET
# =========================================================

print("\nSaving analysis dataframe...")


combined.to_csv(

    OUTPUT_PATH,

    index=False

)


# =========================================================
# BUILD ATLAS BUNDLE
# =========================================================

print("\n[10/10] Building AtlasBundle...")


bundle = build_bundle(

    df=combined,

    domain="movies_music",

    feature_config=FEATURE_CONFIG,

    metadata={

        "pipeline": "movies_music_atlas",

        "domains": DOMAINS,

        "visual_size_strength":
            VISUAL_SIZE_STRENGTH,

        "embedding": "TF-IDF",

        "projection": "UMAP",

        "semantic_text": True,

        "domain_vocabulary_filtering": True,

        "items": len(combined),

        "movies": int(
            (
                combined["domain"]
                == "movies"
            ).sum()
        ),

        "music": int(
            (
                combined["domain"]
                == "music"
            ).sum()
        ),

        "clusters": int(
            combined["cluster"]
            .nunique()
        )

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


print(
    "\n✅ Movies + Music Atlas complete!"
)