# app/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

ATLAS_PATHS = {

    "Movies":
        ROOT / "data" / "processed" / "movie_map_v1.csv",

    "Music":
        ROOT / "data" / "processed" / "music_map_v1.csv",

    "Restaurants":
        ROOT / "data" / "processed" / "restaurant_map_v1.csv",

    "Cross Domain":
        ROOT / "data" / "processed" / "cross_domain_map_v1.csv"
}

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Semantic Atlas",
    layout="wide"
)

st.title("🌍 Semantic Atlas Explorer")

# =========================================================
# ATLAS SELECTION
# =========================================================

selected_atlas = st.sidebar.selectbox(
    "Atlas",
    list(ATLAS_PATHS.keys())
)

DATA_PATH = ATLAS_PATHS[selected_atlas]

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATA_PATH)

# =========================================================
# COLUMN MAPPING
# =========================================================

# Makes one frontend work for every domain

if "title" in df.columns:
    label_column = "title"

elif "artist_lastfm" in df.columns:
    label_column = "artist_lastfm"

elif "restaurant_name" in df.columns:
    label_column = "restaurant_name"

elif "name" in df.columns:
    label_column = "name"

else:
    label_column = df.columns[0]

# popularity column

if "rating_count" in df.columns:
    popularity_column = "rating_count"

elif "listeners_lastfm" in df.columns:
    popularity_column = "listeners_lastfm"

elif "review_count" in df.columns:
    popularity_column = "review_count"

else:
    popularity_column = None

# category column

if "macro_genre" in df.columns:
    category_column = "macro_genre"

elif "macro_category" in df.columns:
    category_column = "macro_category"

else:
    category_column = None

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Navigation")

search_query = st.sidebar.text_input(
    f"Search {selected_atlas[:-1]}"
)

zoom_mode = st.sidebar.radio(
    "Zoom Level",
    [
        "Overview",
        "Explore",
        "Detail"
    ]
)

show_all_clusters = st.sidebar.checkbox(
    "Expand clusters",
    value=False
)

if popularity_column is not None:

    max_pop = int(df[popularity_column].max())

    min_popularity = st.sidebar.slider(
        "Minimum popularity",
        0,
        max_pop,
        0
    )

else:

    min_popularity = None

if category_column is not None:

    cats = sorted(
        df[category_column]
        .dropna()
        .unique()
    )

    selected_category = st.sidebar.selectbox(
        "Category",
        ["All"] + list(cats)
    )

else:

    selected_category = "All"

# =========================================================
# SEARCH
# =========================================================

focus_item = None

if search_query:

    matches = df[
        df[label_column]
        .astype(str)
        .str.contains(
            search_query,
            case=False,
            na=False
        )
    ]

    if len(matches):

        focus_item = matches.iloc[0]

# =========================================================
# FILTERS
# =========================================================

filtered = df.copy()

if popularity_column is not None:

    filtered = filtered[
        filtered[popularity_column]
        >= min_popularity
    ]

if (
    category_column is not None
    and
    selected_category != "All"
):

    filtered = filtered[
        filtered[category_column]
        == selected_category
    ]

# =========================================================
# LOD
# =========================================================

if zoom_mode == "Overview":

    filtered = filtered[
        filtered["visual_size"]
        >=
        filtered["visual_size"]
        .quantile(0.85)
    ]

elif zoom_mode == "Explore":

    filtered = filtered[
        filtered["visual_size"]
        >=
        filtered["visual_size"]
        .quantile(0.35)
    ]

if not show_all_clusters:

    filtered = filtered[
        filtered["visual_size"]
        >=
        filtered["visual_size"]
        .quantile(0.25)
    ]

# =========================================================
# FOCUS
# =========================================================

if focus_item is not None:

    cx = focus_item["umap_x"]
    cy = focus_item["umap_y"]

    filtered = df[
        (df["umap_x"] > cx - 1.5)
        &
        (df["umap_x"] < cx + 1.5)
        &
        (df["umap_y"] > cy - 1.5)
        &
        (df["umap_y"] < cy + 1.5)
    ]

# =========================================================
# HOVER DATA
# =========================================================

hover = {}

for col in [
    "cluster",
    "cluster_label",
    "weighted_rating",
    "rating_count",
    "listeners_lastfm",
    "scrobbles_lastfm",
    "genres",
    "tags_lastfm"
]:

    if col in filtered.columns:
        hover[col] = True

# =========================================================
# COLOR
# =========================================================

color_column = (
    category_column
    if category_column is not None
    else "cluster"
)

# =========================================================
# PLOT
# =========================================================

fig = px.scatter(

    filtered,

    x="umap_x",
    y="umap_y",

    color=color_column,

    size="visual_size",

    hover_name=label_column,

    hover_data=hover,

    opacity=0.75,

    height=850
)

fig.update_traces(
    marker=dict(
        sizemode="area",
        line=dict(width=0)
    )
)

fig.update_layout(
    template="plotly_dark",
    dragmode="pan",
    hovermode="closest"
)

if focus_item is not None:

    fig.add_scatter(

        x=[focus_item["umap_x"]],

        y=[focus_item["umap_y"]],

        mode="markers+text",

        marker=dict(
            size=18,
            color="white"
        ),

        text=[focus_item[label_column]],

        textposition="top center",

        name="Focus"
    )

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("Atlas Statistics")

st.write(
    "Elements displayed:",
    len(filtered)
)

if "cluster_label" in filtered.columns:

    st.write(
        "Regions:",
        filtered["cluster_label"].nunique()
    )