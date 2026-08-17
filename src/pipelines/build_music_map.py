# src/pipelines/build_music_map.py

from pathlib import Path

from src.domains.music.load_data import (
    load_raw_data
)

from src.domains.music.feature_engineering import (
    filter_artists,
    create_tags_text,
    compute_popularity_score,
    create_visual_sizes,
    create_macro_genres
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
    / "music_map_v1.csv"
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
    / "music"
)


# =========================================================
# FEATURE CONFIGURATION
# =========================================================

FEATURE_CONFIG = FeatureConfig(

    name="music_tags",

    use_tags=True,

    use_categories=True,

    use_reviews=False,

    use_metadata=True,

    use_statistics=True

)


# =========================================================
# CONFIG
# =========================================================

MIN_LISTENERS = 1000

VISUAL_SIZE_STRENGTH = 1.8


# =========================================================
# LOAD DATA
# =========================================================

print("\n[1/8] Loading music dataset...")

music = load_raw_data()


# =========================================================
# PREPROCESSING
# =========================================================

print("\n[2/8] Filtering artists...")

music = filter_artists(
    music,
    min_listeners=MIN_LISTENERS
)


print(
    f"Artists remaining: {len(music):,}"
)


# =========================================================
# TEXT PROCESSING
# =========================================================

print("\n[3/8] Processing tags...")

music = create_tags_text(
    music
)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

print("\nCreating atlas features...")

music = compute_popularity_score(
    music
)


music = create_visual_sizes(
    music,
    strength=VISUAL_SIZE_STRENGTH
)


music = create_macro_genres(
    music
)


# =========================================================
# EMBEDDINGS
# =========================================================

print("\n[4/8] Building TF-IDF embeddings...")

tfidf_matrix, vectorizer = (
    get_tfidf_embeddings(
        music,
        text_column="tags_text",
        model_name="music_tags"
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


music["umap_x"] = embedding[:, 0]

music["umap_y"] = embedding[:, 1]


# =========================================================
# CLUSTERING
# =========================================================

print("\n[6/8] Computing clusters...")

music = compute_clusters(
    music
)


music = create_region_labels(
    music
)


# =========================================================
# EXPORT ANALYSIS DATASET
# =========================================================

print("\nSaving analysis dataframe...")


music.to_csv(

    OUTPUT_PATH,

    index=False

)


# =========================================================
# BUILD ATLAS BUNDLE
# =========================================================

print("\n[7/8] Building AtlasBundle...")


bundle = build_bundle(

    df=music,

    domain="music",

    feature_config=FEATURE_CONFIG,

    metadata={

        "pipeline": "music_map",

        "visual_size_strength": VISUAL_SIZE_STRENGTH,

        "embedding": "TF-IDF",

        "projection": "UMAP",

        "items": len(music),

        "clusters": int(
            music["cluster"].nunique()
        )

    }

)


# =========================================================
# EXPORT FRONTEND DATA
# =========================================================

print("\n[8/8] Exporting AtlasBundle...")


export_bundle(

    bundle,

    JSON_DIR

)


print("\n✅ Music Atlas complete!")