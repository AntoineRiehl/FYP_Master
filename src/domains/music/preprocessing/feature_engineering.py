#src/domains/music/preprocessing/feature_engineering.py

import pandas as pd
import numpy as np


# =========================================================
# FILTER ARTISTS
# =========================================================

def filter_artists(
    df,
    min_listeners=5000
):

    df = df.copy()

    df = df[
        df["listeners_lastfm"] >= min_listeners
    ]

    df = df[
        df["ambiguous_artist"] == False
    ]

    return df


# =========================================================
# CREATE TAG TEXT
# =========================================================

def create_tags_text(df):

    df = df.copy()

    df["tags_text"] = (
        df["tags_lastfm"]
        .fillna("")
        .astype(str)
        .str.replace(";", " ")
    )

    return df


# =========================================================
# POPULARITY SCORE
# =========================================================

def compute_popularity_score(df):

    df = df.copy()

    listeners = np.log1p(
        df["listeners_lastfm"]
    )

    scrobbles = np.log1p(
        df["scrobbles_lastfm"]
    )

    df["popularity_score"] = (
        listeners * 0.6 +
        scrobbles * 0.4
    )

    return df


# =========================================================
# VISUAL SIZE
# =========================================================

def create_visual_sizes(
    df,
    strength=1.8
):

    base = df["popularity_score"]

    min_v = base.min()
    max_v = base.max()

    x = (
        (base - min_v)
        /
        (max_v - min_v)
    )

    x = np.power(
        x,
        strength
    )

    df["visual_size"] = x * 50

    df["visual_size"] = np.maximum(
        df["visual_size"],
        2
    )

    return df


# =========================================================
# MACRO GENRES
# =========================================================

def map_macro_genre(tags):

    if pd.isna(tags):
        return "Other"

    tags = str(tags).lower()

    if any(
        t in tags
        for t in [
            "rock",
            "alternative",
            "metal",
            "punk"
        ]
    ):
        return "Rock"

    if any(
        t in tags
        for t in [
            "pop"
        ]
    ):
        return "Pop"

    if any(
        t in tags
        for t in [
            "hip hop",
            "rap"
        ]
    ):
        return "Hip-Hop"

    if any(
        t in tags
        for t in [
            "electronic",
            "techno",
            "house",
            "trance"
        ]
    ):
        return "Electronic"

    if any(
        t in tags
        for t in [
            "jazz",
            "blues"
        ]
    ):
        return "Jazz"

    if any(
        t in tags
        for t in [
            "classical"
        ]
    ):
        return "Classical"

    return "Other"


# =========================================================
# APPLY MACRO GENRE
# =========================================================

def create_macro_genres(df):

    df["macro_genre"] = (
        df["tags_lastfm"]
        .apply(map_macro_genre)
    )

    return df


# =========================================================
# REGION NODES
# =========================================================

def create_region_nodes(df):

    return (
        df.groupby("cluster")
        .agg(
            umap_x=("umap_x", "mean"),
            umap_y=("umap_y", "mean"),
            visual_size=("visual_size", "mean"),
            cluster_label=("cluster_label", "first"),
            artist_count=("artist_lastfm", "count")
        )
        .reset_index()
    )


# =========================================================
# LANDMARK ARTISTS
# =========================================================

def create_landmark_artists(
    df,
    min_listeners=50000
):

    return df[
        df["listeners_lastfm"]
        >= min_listeners
    ].copy()