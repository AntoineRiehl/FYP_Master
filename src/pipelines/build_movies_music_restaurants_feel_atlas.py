# src/pipelines/build_movies_music_restaurants_feel_atlas.py

# ============================================================
# MOVIES + MUSIC + RESTAURANTS SHARED FEEL ATLAS
# ============================================================
#
# METHOD B
# --------
#
# Build the complete three-domain cross-domain atlas from the
# frozen 13-dimensional Shared Experiential / Feel Space.
#
#
# IMPORTANT
# ---------
#
# This script does NOT recompute:
#
#     - base SBERT embeddings
#     - review embeddings
#     - base/review fusion
#     - feel-anchor embeddings
#     - feel scores
#
# It starts from the already-generated:
#
#     feel_space_scores.csv
#
#
# PIPELINE
# --------
#
# Frozen 13D Feel Space
#        ↓
# Keep Movies + Music + Restaurants
#        ↓
# Remove semantically undefined entities
#        ↓
# Attach normal domain metadata
#        ↓
# Normalize visual sizes WITHIN each domain
#        ↓
# Combine all domains
#        ↓
# ONE global StandardScaler
#        ↓
# 13 standardized experiential dimensions
#        ↓
# UMAP with Euclidean distance
#        ↓
# clustering
#        ↓
# region labels
#        ↓
# frontend AtlasBundle
#
#
# EXPECTED FINAL POPULATION
# -------------------------
#
# Movies:        63,324
# Music:        156,738
# Restaurants:   33,941
#
# Total:        254,003
#
# ============================================================


from pathlib import Path
import re

import numpy as np
import pandas as pd
import umap


from sklearn.preprocessing import (
    StandardScaler
)


# ============================================================
# MOVIE DATA
# ============================================================

from src.domains.movies.load_data import (
    load_raw_data as load_movie_data
)

from src.domains.movies.merge_data import (
    compute_movie_stats,
    merge_movies
)

from src.domains.movies.feature_engineering import (
    compute_weighted_rating,
    concatenate_tags,
    create_macro_genres
)


# ============================================================
# MUSIC DATA
# ============================================================

from src.domains.music.load_data import (
    load_raw_data as load_music_data
)

from src.domains.music.feature_engineering import (
    filter_artists,
    create_tags_text as create_music_tags_text,
    compute_popularity_score as compute_music_popularity
)


# ============================================================
# RESTAURANT DATA
# ============================================================

from src.domains.restaurants.load_data import (
    load_raw_data as load_restaurant_data
)

from src.domains.restaurants.feature_engineering import (
    filter_restaurants,
    create_tags_text as create_restaurant_tags_text,
    compute_popularity_score as compute_restaurant_popularity
)


# ============================================================
# ATLAS
# ============================================================

from src.atlas.cross_domain.feel_space.feel_anchors import (
    FEEL_DIMENSIONS
)

from src.atlas.visual.size_normalization import (
    normalize_visual_sizes
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


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------
# Frozen Method B Feel scores
# ------------------------------------------------------------

FEEL_SPACE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "feel_space"
    / "movies_music_restaurants"
    / "feel_space_scores.csv"
)


# ------------------------------------------------------------
# Analysis dataframe
# ------------------------------------------------------------

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "movies_music_restaurants_feel_atlas.csv"
)


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Frontend output
# ------------------------------------------------------------

JSON_DIR = (
    ROOT
    / "frontend"
    / "public"
    / "data"
    / "movies_music_restaurants_feel"
)


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

FEATURE_CONFIG = FeatureConfig(

    name=
        "movies_music_restaurants_shared_feel",

    use_tags=
        True,

    use_categories=
        True,

    use_reviews=
        True,

    use_metadata=
        True,

    use_statistics=
        True

)


# ============================================================
# CONFIGURATION
# ============================================================


# ------------------------------------------------------------
# Domain population filters
# ------------------------------------------------------------

MIN_MUSIC_LISTENERS = 1000

MIN_RESTAURANT_REVIEWS = 20


# ------------------------------------------------------------
# Visual sizing
# ------------------------------------------------------------

VISUAL_SIZE_STRENGTH = 1.8


# ------------------------------------------------------------
# UMAP
#
# The Feel dimensions are explicit coordinates rather than
# raw semantic embedding dimensions.
#
# After global standardization, Euclidean distance gives a
# direct measure of experiential profile difference.
# ------------------------------------------------------------

UMAP_N_NEIGHBORS = 15

UMAP_MIN_DIST = 0.1

UMAP_METRIC = "euclidean"

RANDOM_STATE = 42


# ------------------------------------------------------------
# Frozen Method B semantic choices
# ------------------------------------------------------------

REVIEW_SHARE = 0.50

FEEL_DIMENSION_COUNT = len(
    FEEL_DIMENSIONS
)


# ============================================================
# EXPECTED POPULATION
# ============================================================

EXPECTED_MOVIES = 63_324

EXPECTED_MUSIC = 156_738

EXPECTED_RESTAURANTS = 33_941


EXPECTED_TOTAL = (

    EXPECTED_MOVIES

    +

    EXPECTED_MUSIC

    +

    EXPECTED_RESTAURANTS

)


# ============================================================
# ORIGINAL DOMAIN POPULATION
# ============================================================

ORIGINAL_MOVIES = 84_432

ORIGINAL_MUSIC = 189_948

ORIGINAL_RESTAURANTS = 33_941


# ============================================================
# PRINT SECTION
# ============================================================


def section(
    title
):

    print()

    print(
        "=" * 72
    )

    print(
        title
    )

    print(
        "=" * 72
    )

    print()


# ============================================================
# ID NORMALISATION
# ============================================================


def normalise_source_id(
    value
):
    """
    Convert heterogeneous source IDs into stable strings.

    Numeric identifiers such as:

        123
        123.0
        "123"

    all become:

        "123"

    UUID-like identifiers remain unchanged.
    """

    if pd.isna(
        value
    ):

        return ""


    text = str(
        value
    ).strip()


    if re.fullmatch(

        r"[+-]?\d+\.0+",

        text

    ):

        return text.split(
            "."
        )[0]


    return text


# ============================================================
# BOOLEAN NORMALISATION
# ============================================================


def normalise_boolean_series(
    series
):

    if series.dtype == bool:

        return series


    return (

        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes"
            ]
        )

    )


# ============================================================
# LOAD FROZEN FEEL SPACE
# ============================================================


def load_feel_space():

    section(
        "LOADING FROZEN SHARED FEEL SPACE"
    )


    if not FEEL_SPACE_FILE.exists():

        raise FileNotFoundError(

            "Feel-space score file was not found:\n"
            f"{FEEL_SPACE_FILE}\n\n"

            "Run:\n"
            "python -m src.pipelines.evaluate_feel_space"

        )


    feel_df = pd.read_csv(

        FEEL_SPACE_FILE,

        dtype={

            "domain":
                str,

            "source_id":
                str,

            "id":
                str,

        }

    )


    required_columns = {

        "domain",

        "source_id",

        "is_semantically_defined",

        *FEEL_DIMENSIONS,

    }


    missing = (

        required_columns

        -

        set(
            feel_df.columns
        )

    )


    if missing:

        raise ValueError(

            "Feel-space score file is missing required "
            "columns:\n"
            f"{sorted(missing)}"

        )


    feel_df[
        "is_semantically_defined"
    ] = normalise_boolean_series(

        feel_df[
            "is_semantically_defined"
        ]

    )


    # --------------------------------------------------------
    # Keep all three Method B domains.
    # --------------------------------------------------------

    feel_df = feel_df.loc[

        feel_df[
            "domain"
        ].isin(
            [
                "movies",
                "music",
                "restaurants"
            ]
        )

    ].copy()


    print(
        f"Rows before semantic filter: "
        f"{len(feel_df):,}"
    )


    # --------------------------------------------------------
    # Remove entities with neither base nor review semantics.
    #
    # Their zero Feel vector does NOT mean neutral.
    # It means undefined.
    # --------------------------------------------------------

    feel_df = feel_df.loc[

        feel_df[
            "is_semantically_defined"
        ]

    ].copy()


    feel_df[
        "source_id_key"
    ] = (

        feel_df[
            "source_id"
        ]
        .apply(
            normalise_source_id
        )

    )


    # ========================================================
    # FEEL SCORE VALIDATION
    # ========================================================

    for dimension in FEEL_DIMENSIONS:

        feel_df[
            dimension
        ] = pd.to_numeric(

            feel_df[
                dimension
            ],

            errors="coerce"

        )


    invalid_scores = int(

        feel_df[
            FEEL_DIMENSIONS
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()

    )


    if invalid_scores > 0:

        raise ValueError(

            f"{invalid_scores:,} semantically defined "
            "entities contain missing Feel scores."

        )


    print()

    print(
        "Defined entities:"
    )


    print(

        feel_df[
            "domain"
        ]
        .value_counts()
        .to_string()

    )


    print()


    print(
        f"Total defined entities: "
        f"{len(feel_df):,}"
    )


    return feel_df


# ============================================================
# PREPARE MOVIES
# ============================================================


def prepare_movies():

    section(
        "PREPARING MOVIE METADATA"
    )


    movies, ratings, tags = (
        load_movie_data()
    )


    print(
        f"Raw movies: "
        f"{len(movies):,}"
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    movie_stats = compute_movie_stats(
        ratings
    )


    movie_df = merge_movies(

        movie_stats,

        movies

    )


    movie_df = compute_weighted_rating(
        movie_df
    )


    movie_df[
        "popularity_score"
    ] = movie_df[
        "rating_count"
    ]


    # ========================================================
    # TAG TEXT
    # ========================================================

    movie_tags = concatenate_tags(
        tags
    )


    movie_df = movie_df.merge(

        movie_tags,

        on=
            "movieId",

        how=
            "left"

    )


    movie_df[
        "tags_text"
    ] = (

        movie_df[
            "tags_text"
        ]
        .fillna("")
        .astype(str)

    )


    # ========================================================
    # DISPLAY FEATURES
    # ========================================================

    movie_df = create_macro_genres(
        movie_df
    )


    if "year" not in movie_df.columns:

        movie_df[
            "year"
        ] = (

            movie_df[
                "title"
            ]
            .astype(str)
            .str.extract(

                r"\((\d{4})\)\s*$",

                expand=False

            )

        )


    # ========================================================
    # IDENTIFIERS
    # ========================================================

    movie_df[
        "source_id"
    ] = (

        movie_df[
            "movieId"
        ]
        .apply(
            normalise_source_id
        )

    )


    movie_df[
        "source_id_key"
    ] = movie_df[
        "source_id"
    ]


    movie_df[
        "domain"
    ] = "movies"


    movie_df[
        "id"
    ] = (

        "movies:"

        +

        movie_df[
            "source_id"
        ]

    )


    # ========================================================
    # VISUAL SIZE
    #
    # Must be normalized before cross-domain concatenation.
    # ========================================================

    movie_df = normalize_visual_sizes(

        movie_df,

        strength=
            VISUAL_SIZE_STRENGTH

    )


    print(
        f"Prepared Movie population: "
        f"{len(movie_df):,}"
    )


    return movie_df


# ============================================================
# PREPARE MUSIC
# ============================================================


def prepare_music():

    section(
        "PREPARING MUSIC METADATA"
    )


    music_df = load_music_data()


    print(
        f"Raw artists: "
        f"{len(music_df):,}"
    )


    # ========================================================
    # FILTER
    # ========================================================

    music_df = filter_artists(

        music_df,

        min_listeners=
            MIN_MUSIC_LISTENERS

    )


    print(
        f"Artists after listener filter: "
        f"{len(music_df):,}"
    )


    # ========================================================
    # SEMANTIC TEXT
    # ========================================================

    music_df = create_music_tags_text(
        music_df
    )


    music_df[
        "tags_text"
    ] = (

        music_df[
            "tags_text"
        ]
        .fillna("")
        .astype(str)

    )


    # ========================================================
    # POPULARITY
    # ========================================================

    music_df = compute_music_popularity(
        music_df
    )


    # ========================================================
    # DISPLAY TITLE
    # ========================================================

    if "artist_lastfm" in music_df.columns:

        title = (

            music_df[
                "artist_lastfm"
            ]
            .fillna("")
            .astype(str)
            .str.strip()

        )

    else:

        title = pd.Series(

            "",

            index=
                music_df.index,

            dtype=str

        )


    if "artist_mb" in music_df.columns:

        fallback = (

            music_df[
                "artist_mb"
            ]
            .fillna("")
            .astype(str)
            .str.strip()

        )


        title = title.where(

            title != "",

            fallback

        )


    music_df[
        "title"
    ] = title


    # ========================================================
    # COUNTRY
    # ========================================================

    if "country_lastfm" in music_df.columns:

        music_df[
            "country"
        ] = music_df[
            "country_lastfm"
        ]


    if "country_mb" in music_df.columns:

        if "country" not in music_df.columns:

            music_df[
                "country"
            ] = music_df[
                "country_mb"
            ]


        else:

            music_df[
                "country"
            ] = (

                music_df[
                    "country"
                ]
                .fillna(
                    music_df[
                        "country_mb"
                    ]
                )

            )


    # ========================================================
    # IDENTIFIERS
    # ========================================================

    music_df[
        "source_id"
    ] = (

        music_df[
            "mbid"
        ]
        .apply(
            normalise_source_id
        )

    )


    music_df[
        "source_id_key"
    ] = music_df[
        "source_id"
    ]


    music_df[
        "domain"
    ] = "music"


    music_df[
        "id"
    ] = (

        "music:"

        +

        music_df[
            "source_id"
        ]

    )


    # ========================================================
    # VISUAL SIZE
    # ========================================================

    music_df = normalize_visual_sizes(

        music_df,

        strength=
            VISUAL_SIZE_STRENGTH

    )


    print(
        f"Prepared Music population: "
        f"{len(music_df):,}"
    )


    return music_df


# ============================================================
# PREPARE RESTAURANTS
# ============================================================


def prepare_restaurants():

    section(
        "PREPARING RESTAURANT METADATA"
    )


    businesses, reviews, tips = (
        load_restaurant_data()
    )


    print(
        f"Raw businesses: "
        f"{len(businesses):,}"
    )


    print(
        f"Raw reviews:    "
        f"{len(reviews):,}"
    )


    print(
        f"Raw tips:       "
        f"{len(tips):,}"
    )


    # ========================================================
    # FILTER
    # ========================================================

    restaurant_df = filter_restaurants(

        businesses,

        min_reviews=
            MIN_RESTAURANT_REVIEWS

    )


    print(
        f"Restaurants after filtering: "
        f"{len(restaurant_df):,}"
    )


    # ========================================================
    # SEMANTIC TEXT
    #
    # categories + Yelp tips
    # ========================================================

    restaurant_df = create_restaurant_tags_text(

        restaurant_df,

        tips

    )


    restaurant_df[
        "tags_text"
    ] = (

        restaurant_df[
            "tags_text"
        ]
        .fillna("")
        .astype(str)

    )


    # ========================================================
    # POPULARITY
    # ========================================================

    restaurant_df = compute_restaurant_popularity(
        restaurant_df
    )


    # --------------------------------------------------------
    # Make restaurant name explicit as title.
    # --------------------------------------------------------

    if "name" in restaurant_df.columns:

        restaurant_df[
            "title"
        ] = (

            restaurant_df[
                "name"
            ]
            .fillna("")
            .astype(str)

        )


    # --------------------------------------------------------
    # Categories are useful to preserve explicitly for the
    # frontend.
    # --------------------------------------------------------

    if "categories" not in restaurant_df.columns:

        restaurant_df[
            "categories"
        ] = ""


    # ========================================================
    # IDENTIFIERS
    # ========================================================

    restaurant_df[
        "source_id"
    ] = (

        restaurant_df[
            "business_id"
        ]
        .apply(
            normalise_source_id
        )

    )


    restaurant_df[
        "source_id_key"
    ] = restaurant_df[
        "source_id"
    ]


    restaurant_df[
        "domain"
    ] = "restaurants"


    restaurant_df[
        "id"
    ] = (

        "restaurants:"

        +

        restaurant_df[
            "source_id"
        ]

    )


    # ========================================================
    # VISUAL SIZE
    #
    # Normalize within Restaurants BEFORE cross-domain merge.
    # ========================================================

    restaurant_df = normalize_visual_sizes(

        restaurant_df,

        strength=
            VISUAL_SIZE_STRENGTH

    )


    print(
        f"Prepared Restaurant population: "
        f"{len(restaurant_df):,}"
    )


    return restaurant_df


# ============================================================
# ATTACH FEEL SCORES
# ============================================================


def attach_feel_scores(

    domain_df,

    feel_df,

    domain

):
    """
    Join one domain's normal metadata with the frozen Method B
    Feel representation.
    """

    domain_feel = feel_df.loc[

        feel_df[
            "domain"
        ]
        ==
        domain

    ].copy()


    keep_columns = [

        "source_id_key",

        "semantic_source",

        "has_base_semantics",

        "has_review_semantics",

        "reviews_used_for_embedding",

        "is_semantically_defined",

        *FEEL_DIMENSIONS,

    ]


    keep_columns = [

        column

        for column in keep_columns

        if column in domain_feel.columns

    ]


    domain_feel = domain_feel[
        keep_columns
    ].copy()


    if domain_feel[
        "source_id_key"
    ].duplicated().any():

        raise ValueError(

            f"Duplicate Feel IDs detected for domain "
            f"'{domain}'."

        )


    merged = domain_df.merge(

        domain_feel,

        on=
            "source_id_key",

        how=
            "inner",

        validate=
            "one_to_one"

    )


    # ========================================================
    # FRONTEND REVIEW METADATA
    # ========================================================

    if "has_review_semantics" in merged.columns:

        merged[
            "has_review_embedding"
        ] = (

            merged[
                "has_review_semantics"
            ]
            .astype(bool)

        )


    if "reviews_used_for_embedding" in merged.columns:

        # ----------------------------------------------------
        # Preserve any existing domain review_count.
        #
        # For Restaurants, Yelp already contains the true
        # review count, which should NOT be overwritten.
        # --------------------------------------------------------

        if "review_count" not in merged.columns:

            merged[
                "review_count"
            ] = (

                merged[
                    "reviews_used_for_embedding"
                ]
                .fillna(0)
                .astype(int)

            )


    print(

        f"{domain}: "
        f"{len(merged):,} semantically defined entities "
        f"matched to metadata."

    )


    return merged


# ============================================================
# POPULATION VALIDATION
# ============================================================


def validate_population(

    movies,

    music,

    restaurants

):

    section(
        "VALIDATING METHOD B POPULATION"
    )


    movie_count = len(
        movies
    )


    music_count = len(
        music
    )


    restaurant_count = len(
        restaurants
    )


    total = (

        movie_count

        +

        music_count

        +

        restaurant_count

    )


    print(
        f"Movies:       "
        f"{movie_count:,}"
    )


    print(
        f"Music:        "
        f"{music_count:,}"
    )


    print(
        f"Restaurants:  "
        f"{restaurant_count:,}"
    )


    print(
        f"Total:        "
        f"{total:,}"
    )


    if movie_count != EXPECTED_MOVIES:

        raise ValueError(

            "Movie population does not match the frozen "
            "Method B evaluation.\n\n"

            f"Expected: {EXPECTED_MOVIES:,}\n"
            f"Received: {movie_count:,}"

        )


    if music_count != EXPECTED_MUSIC:

        raise ValueError(

            "Music population does not match the frozen "
            "Method B evaluation.\n\n"

            f"Expected: {EXPECTED_MUSIC:,}\n"
            f"Received: {music_count:,}"

        )


    if restaurant_count != EXPECTED_RESTAURANTS:

        raise ValueError(

            "Restaurant population does not match the frozen "
            "Method B evaluation.\n\n"

            f"Expected: {EXPECTED_RESTAURANTS:,}\n"
            f"Received: {restaurant_count:,}"

        )


    if total != EXPECTED_TOTAL:

        raise ValueError(

            "Combined Method B population does not match the "
            "frozen evaluation.\n\n"

            f"Expected: {EXPECTED_TOTAL:,}\n"
            f"Received: {total:,}"

        )


    print()

    print(
        "Population validation passed."
    )


# ============================================================
# GLOBAL FEEL STANDARDIZATION
# ============================================================


def scale_feel_space(
    final
):
    """
    Apply ONE StandardScaler across all Movies, Music and
    Restaurants.

    This gives each Feel dimension comparable numerical
    influence without forcing domain distributions to match.
    """

    section(
        "GLOBAL STANDARDIZATION OF 13D FEEL SPACE"
    )


    feel_matrix = (

        final[
            FEEL_DIMENSIONS
        ]
        .to_numpy(
            dtype=np.float32
        )

    )


    print(
        f"Raw Feel matrix: "
        f"{feel_matrix.shape}"
    )


    scaler = StandardScaler()


    scaled_matrix = scaler.fit_transform(
        feel_matrix
    )


    scaled_matrix = np.asarray(

        scaled_matrix,

        dtype=np.float32

    )


    print()

    print(
        "Global scaling complete."
    )


    print(
        f"Scaled matrix: "
        f"{scaled_matrix.shape}"
    )


    print()


    print(
        "Scaled dimension means:"
    )


    for index, dimension in enumerate(

        FEEL_DIMENSIONS

    ):

        print(

            f"  {dimension:<12} "
            f"{scaled_matrix[:, index].mean():+.4f}"

        )


    print()


    print(
        "Scaled dimension standard deviations:"
    )


    for index, dimension in enumerate(

        FEEL_DIMENSIONS

    ):

        print(

            f"  {dimension:<12} "
            f"{scaled_matrix[:, index].std():.4f}"

        )


    return (

        scaled_matrix,

        scaler

    )


# ============================================================
# UMAP
# ============================================================


def compute_feel_umap(
    scaled_matrix
):

    section(
        "COMPUTING THREE-DOMAIN FEEL-SPACE UMAP"
    )


    print(
        f"Input shape: "
        f"{scaled_matrix.shape}"
    )


    print(
        f"Metric:      "
        f"{UMAP_METRIC}"
    )


    print(
        f"Neighbours:  "
        f"{UMAP_N_NEIGHBORS}"
    )


    print(
        f"Min dist:    "
        f"{UMAP_MIN_DIST}"
    )


    mapper = umap.UMAP(

        n_components=
            2,

        n_neighbors=
            UMAP_N_NEIGHBORS,

        min_dist=
            UMAP_MIN_DIST,

        metric=
            UMAP_METRIC,

        random_state=
            RANDOM_STATE,

        low_memory=
            True

    )


    embedding = mapper.fit_transform(
        scaled_matrix
    )


    embedding = np.asarray(

        embedding,

        dtype=np.float32

    )


    print(
        f"UMAP output shape: "
        f"{embedding.shape}"
    )


    return (

        embedding,

        mapper

    )


# ============================================================
# MAIN
# ============================================================


def main():

    section(
        "MOVIES + MUSIC + RESTAURANTS SHARED FEEL ATLAS"
    )


    print(
        "Method B representation is frozen."
    )


    print(
        "This pipeline starts from the existing "
        "13-dimensional Feel scores."
    )


    # ========================================================
    # 1. LOAD FROZEN FEEL SPACE
    # ========================================================

    feel_df = load_feel_space()


    # ========================================================
    # 2. PREPARE DOMAIN METADATA
    # ========================================================

    movie_df = prepare_movies()


    music_df = prepare_music()


    restaurant_df = prepare_restaurants()


    # ========================================================
    # 3. ATTACH FEEL SCORES
    # ========================================================

    section(
        "ATTACHING FEEL SCORES TO DOMAIN METADATA"
    )


    movie_df = attach_feel_scores(

        domain_df=
            movie_df,

        feel_df=
            feel_df,

        domain=
            "movies"

    )


    music_df = attach_feel_scores(

        domain_df=
            music_df,

        feel_df=
            feel_df,

        domain=
            "music"

    )


    restaurant_df = attach_feel_scores(

        domain_df=
            restaurant_df,

        feel_df=
            feel_df,

        domain=
            "restaurants"

    )


    # ========================================================
    # 4. VALIDATE POPULATION
    # ========================================================

    validate_population(

        movies=
            movie_df,

        music=
            music_df,

        restaurants=
            restaurant_df

    )


    # ========================================================
    # 5. COMBINE ALL THREE DOMAINS
    # ========================================================

    section(
        "COMBINING MOVIES + MUSIC + RESTAURANTS"
    )


    final = pd.concat(

        [

            movie_df,

            music_df,

            restaurant_df

        ],

        axis=0,

        ignore_index=True,

        sort=False

    )


    # --------------------------------------------------------
    # Global identifiers must remain unique.
    # --------------------------------------------------------

    if final[
        "id"
    ].duplicated().any():

        duplicate_ids = (

            final.loc[

                final[
                    "id"
                ].duplicated(
                    keep=False
                ),

                "id"

            ]
            .head(10)
            .tolist()

        )


        raise ValueError(

            "Duplicate global IDs detected after combining "
            "the three Method B domains.\n\n"

            f"Examples: {duplicate_ids}"

        )


    print(
        f"Combined entities: "
        f"{len(final):,}"
    )


    print()


    print(
        final[
            "domain"
        ]
        .value_counts()
        .to_string()
    )


    # ========================================================
    # 6. GLOBAL STANDARDIZATION
    # ========================================================

    (

        scaled_feel_matrix,

        scaler

    ) = scale_feel_space(
        final
    )


    # --------------------------------------------------------
    # Preserve the standardized axes in the analysis CSV.
    # --------------------------------------------------------

    for index, dimension in enumerate(

        FEEL_DIMENSIONS

    ):

        final[
            f"feel_z_{dimension}"
        ] = (

            scaled_feel_matrix[
                :,
                index
            ]

        )


    # ========================================================
    # 7. UMAP
    # ========================================================

    (

        embedding,

        umap_model

    ) = compute_feel_umap(

        scaled_feel_matrix

    )


    final[
        "umap_x"
    ] = embedding[
        :,
        0
    ]


    final[
        "umap_y"
    ] = embedding[
        :,
        1
    ]


    # ========================================================
    # 8. CLUSTERING
    # ========================================================

    section(
        "COMPUTING THREE-DOMAIN CLUSTERS"
    )


    final = compute_clusters(
        final
    )


    final = create_region_labels(
        final
    )


    print(
        f"Clusters: "
        f"{final['cluster'].nunique():,}"
    )


    # ========================================================
    # 9. SAVE ANALYSIS DATAFRAME
    # ========================================================

    section(
        "SAVING ANALYSIS DATAFRAME"
    )


    final.to_csv(

        OUTPUT_PATH,

        index=False

    )


    print(
        OUTPUT_PATH
    )


    # ========================================================
    # 10. SCALER METADATA
    # ========================================================

    scaler_mean = {

        dimension:
            float(
                scaler.mean_[
                    index
                ]
            )

        for index, dimension
        in enumerate(
            FEEL_DIMENSIONS
        )

    }


    scaler_scale = {

        dimension:
            float(
                scaler.scale_[
                    index
                ]
            )

        for index, dimension
        in enumerate(
            FEEL_DIMENSIONS
        )

    }


    # ========================================================
    # 11. BUILD FRONTEND BUNDLE
    # ========================================================

    section(
        "BUILDING FRONTEND ATLAS BUNDLE"
    )


    bundle = build_bundle(

        df=
            final,

        domain=
            "cross_domain",

        feature_config=
            FEATURE_CONFIG,

        metadata={

            # =================================================
            # PIPELINE
            # =================================================

            "pipeline":
                "movies_music_restaurants_feel_atlas",

            "method":
                "Method B - Shared Experiential Space",

            "domains": [

                "movies",

                "music",

                "restaurants"

            ],


            # =================================================
            # SEMANTIC REPRESENTATION
            # =================================================

            "embedding":
                (
                    "Shared MiniLM base/review semantics "
                    "projected onto experiential anchors"
                ),

            "embedding_model":
                (
                    "sentence-transformers/"
                    "all-MiniLM-L6-v2"
                ),

            "base_review_fusion":
                (
                    "same-space weighted vector fusion"
                ),

            "review_share_when_both_available":
                REVIEW_SHARE,

            "base_share_when_both_available":
                1.0
                -
                REVIEW_SHARE,

            "missing_modality_policy":
                (
                    "use available modality; exclude only "
                    "entities with neither base nor review "
                    "semantics"
                ),


            # =================================================
            # FEEL SPACE
            # =================================================

            "feel_dimensions":
                list(
                    FEEL_DIMENSIONS
                ),

            "feel_dimension_count":
                FEEL_DIMENSION_COUNT,

            "feel_bipolar_dimensions":
                10,

            "feel_unipolar_dimensions":
                3,

            "semantic_anchor_count":
                23,


            # =================================================
            # SCALING
            # =================================================

            "scaling":
                "StandardScaler",

            "scaling_scope":
                (
                    "global_across_movies_music_restaurants"
                ),

            "domain_specific_scaling":
                False,

            "scaler_mean":
                scaler_mean,

            "scaler_scale":
                scaler_scale,


            # =================================================
            # PROJECTION
            # =================================================

            "projection":
                "UMAP",

            "umap_metric":
                UMAP_METRIC,

            "umap_n_neighbors":
                UMAP_N_NEIGHBORS,

            "umap_min_dist":
                UMAP_MIN_DIST,

            "random_state":
                RANDOM_STATE,


            # =================================================
            # VISUALS
            # =================================================

            "visual_size_normalization":
                "within_domain",

            "visual_size_strength":
                VISUAL_SIZE_STRENGTH,


            # =================================================
            # POPULATION
            # =================================================

            "items":
                int(
                    len(
                        final
                    )
                ),

            "movie_items":
                int(

                    (
                        final[
                            "domain"
                        ]
                        ==
                        "movies"
                    ).sum()

                ),

            "music_items":
                int(

                    (
                        final[
                            "domain"
                        ]
                        ==
                        "music"
                    ).sum()

                ),

            "restaurant_items":
                int(

                    (
                        final[
                            "domain"
                        ]
                        ==
                        "restaurants"
                    ).sum()

                ),

            "excluded_semantically_undefined":
                int(

                    (
                        ORIGINAL_MOVIES
                        -
                        EXPECTED_MOVIES
                    )

                    +

                    (
                        ORIGINAL_MUSIC
                        -
                        EXPECTED_MUSIC
                    )

                    +

                    (
                        ORIGINAL_RESTAURANTS
                        -
                        EXPECTED_RESTAURANTS
                    )

                ),

            "clusters":
                int(
                    final[
                        "cluster"
                    ]
                    .nunique()
                )

        }

    )


    # ========================================================
    # 12. EXPORT FRONTEND DATA
    # ========================================================

    section(
        "EXPORTING FRONTEND DATA"
    )


    export_bundle(

        bundle,

        JSON_DIR

    )


    # ========================================================
    # COMPLETE
    # ========================================================

    section(
        "THREE-DOMAIN FEEL ATLAS COMPLETE"
    )


    print(
        f"Movies:       "
        f"{int((final['domain'] == 'movies').sum()):,}"
    )


    print(
        f"Music:        "
        f"{int((final['domain'] == 'music').sum()):,}"
    )


    print(
        f"Restaurants:  "
        f"{int((final['domain'] == 'restaurants').sum()):,}"
    )


    print(
        f"Total:        "
        f"{len(final):,}"
    )


    print(
        f"Feel dimensions: "
        f"{FEEL_DIMENSION_COUNT}"
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


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()