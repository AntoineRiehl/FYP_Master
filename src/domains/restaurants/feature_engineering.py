#src/domains/restaurants/preprocessing/feature_engineering.py

import pandas as pd
import numpy as np


# =========================================================
# FILTER RESTAURANTS
# =========================================================

def filter_restaurants(
    businesses,
    min_reviews=20
):

    df = businesses.copy()

    df = df[
        df["categories"]
        .fillna("")
        .str.contains(
            "Restaurants",
            case=False
        )
    ]

    df = df[
        df["review_count"] >= min_reviews
    ]

    return df


# =========================================================
# TAG TEXT
# =========================================================

def create_tags_text(
    restaurants,
    tips
):

    tip_text = (
        tips
        .groupby("business_id")["text"]
        .apply(
            lambda x: " ".join(
                x.astype(str)
            )
        )
        .reset_index()
    )

    restaurants = restaurants.merge(
        tip_text,
        on="business_id",
        how="left"
    )

    restaurants.rename(
        columns={
            "text": "tip_text"
        },
        inplace=True
    )

    restaurants["tip_text"] = (
        restaurants["tip_text"]
        .fillna("")
    )

    restaurants["categories"] = (
        restaurants["categories"]
        .fillna("")
    )

    restaurants["tags_text"] = (
        restaurants["categories"]
        + " "
        + restaurants["tip_text"]
    )

    return restaurants


# =========================================================
# POPULARITY
# =========================================================

def compute_popularity_score(df):

    df = df.copy()

    df["popularity_score"] = (
        df["review_count"]
        .fillna(0)
        .astype(float)
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
# MACRO CUISINES
# =========================================================

def map_macro_genre(categories):

    if pd.isna(categories):
        return "Other"

    c = str(categories).lower()

    if any(x in c for x in [
        "italian",
        "pizza"
    ]):
        return "Italian"

    if any(x in c for x in [
        "mexican",
        "tacos"
    ]):
        return "Mexican"

    if any(x in c for x in [
        "japanese",
        "sushi",
        "ramen"
    ]):
        return "Japanese"

    if any(x in c for x in [
        "chinese"
    ]):
        return "Chinese"

    if any(x in c for x in [
        "thai",
        "vietnamese",
        "korean"
    ]):
        return "Asian"

    if any(x in c for x in [
        "burger",
        "american"
    ]):
        return "American"

    if any(x in c for x in [
        "coffee",
        "cafe"
    ]):
        return "Cafe"

    return "Other"


def create_macro_genres(df):

    df["macro_genre"] = (
        df["categories"]
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
            restaurant_count=("business_id", "count")
        )
        .reset_index()
    )


# =========================================================
# LANDMARK RESTAURANTS
# =========================================================

def create_landmark_restaurants(
    df,
    min_reviews=500
):

    return df[
        df["review_count"] >= min_reviews
    ].copy()