import pandas as pd
import numpy as np


# =========================================================
# WEIGHTED / BAYESIAN RATING
# =========================================================

def compute_weighted_rating(df):

    # Global mean rating
    C = df["avg_rating"].mean()

    # Minimum vote threshold
    m = df["rating_count"].quantile(0.75)

    # Bayesian weighted rating
    df["weighted_rating"] = (
        (df["avg_rating"] * df["rating_count"]) + (C * m)
    ) / (df["rating_count"] + m)

    return df


# =========================================================
# CONCATENATE TAGS
# =========================================================

def concatenate_tags(tags_df):

    # Remove missing tags
    tags_clean = tags_df.dropna(subset=["tag"]).copy()

    # Ensure string
    tags_clean["tag"] = tags_clean["tag"].astype(str)

    # Group tags per movie
    movie_tags = (
        tags_clean
        .groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )

    # Rename column
    movie_tags.rename(
        columns={"tag": "tags_text"},
        inplace=True
    )

    return movie_tags


# =========================================================
# MACRO GENRE MAPPING
# =========================================================

def map_macro_genre(genre_str):

    # Handle missing values
    if pd.isna(genre_str):
        return "Unknown"

    genre_str = str(genre_str).strip()

    genres = genre_str.split("|")

    mapping = {

        "Sci-Fi/Fantasy": [
            "Sci-Fi",
            "Fantasy"
        ],

        "Action/Adventure": [
            "Action",
            "Adventure",
            "War",
            "Western"
        ],

        "Thriller/Horror": [
            "Thriller",
            "Horror",
            "Crime",
            "Mystery",
            "Film-Noir"
        ],

        "Comedy": [
            "Comedy",
            "Musical"
        ],

        "Drama": [
            "Drama"
        ],

        "Family/Animation": [
            "Animation",
            "Children's"
        ],

        "Documentary": [
            "Documentary"
        ]
    }

    # First matching category wins
    for macro, group in mapping.items():

        if any(g in genres for g in group):
            return macro

    return "Other"


# =========================================================
# APPLY MACRO GENRES
# =========================================================

def create_macro_genres(df):

    df["macro_genre"] = (
        df["genres"]
        .apply(map_macro_genre)
        .astype(str)
        .str.strip()
    )

    # Optional: genre count
    df["genre_count"] = (
        df["genres"]
        .fillna("")
        .apply(lambda x: len(str(x).split("|")))
    )

    return df


# =========================================================
# VISUAL SIZE ENGINEERING
# =========================================================

def create_visual_sizes(df):

    # Base importance
    df["visual_size"] = (
        np.log1p(df["rating_count"]) *
        df["weighted_rating"]
    )

    # Normalize
    x = df["visual_size"]

    x = (
        (x - x.min()) /
        (x.max() - x.min())
    )

    # Amplify contrast
    df["visual_size"] = (
        np.power(x, 2.5) * 30
    )

    # Minimum visibility
    df["visual_size"] = np.maximum(
        df["visual_size"],
        2
    )

    return df


# =========================================================
# OPTIONAL FILTERING HELPER
# =========================================================

def filter_movies(
    df,
    min_ratings=50,
    genre=None
):

    filtered = df[
        df["rating_count"] >= min_ratings
    ]

    if genre:
        filtered = filtered[
            filtered["macro_genre"] == genre
        ]

    return filtered