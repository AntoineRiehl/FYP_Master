# app/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =========================================================
# PATH
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "movie_map_v1.csv"


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Movie Semantic Atlas",
    layout="wide"
)

st.title("🎬 Movie Semantic Atlas v2")


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()


# =========================================================
# SIDEBAR CONTROLS (NEW UX LAYER)
# =========================================================

st.sidebar.header("Navigation")

# ---- Search bar (fly-to simulation)
search_query = st.sidebar.text_input("Search movie")

# ---- Zoom levels (semantic LOD system)
zoom_mode = st.sidebar.radio(
    "Zoom Level",
    ["Overview", "Explore", "Detail"]
)

# ---- Cluster expansion toggle
show_all_clusters = st.sidebar.checkbox(
    "Expand clusters (advanced)",
    value=False
)

# ---- Filters
min_ratings = st.sidebar.slider(
    "Minimum Ratings",
    0, 500, 50
)

genres = sorted(df["macro_genre"].dropna().unique())

selected_genre = st.sidebar.selectbox(
    "Genre",
    ["All"] + genres
)


# =========================================================
# SEARCH → FOCUS LOGIC (SIMULATED CAMERA MOVE)
# =========================================================
focus_movie = None

if search_query:

    match = df[
        df["title"].str.contains(
            search_query,
            case=False,
            na=False
        )
    ]

    if len(match) > 0:
        focus_movie = match.iloc[0]


# =========================================================
# FILTERING BASE
# =========================================================
filtered = df[df["rating_count"] >= min_ratings]

if selected_genre != "All":
    filtered = filtered[filtered["macro_genre"] == selected_genre]


# =========================================================
# ZOOM LOGIC (KEY SEMANTIC FEATURE)
# =========================================================

if zoom_mode == "Overview":

    filtered = filtered[
        filtered["visual_size"] >= filtered["visual_size"].quantile(0.85)
    ]

elif zoom_mode == "Explore":

    filtered = filtered[
        filtered["visual_size"] >= filtered["visual_size"].quantile(0.35)
    ]

# Detail = no filtering


# =========================================================
# CLUSTER EXPANSION (SIMULATED)
# =========================================================
if not show_all_clusters:

    filtered = filtered[
        filtered["visual_size"] >= filtered["visual_size"].quantile(0.25)
    ]


# =========================================================
# FOCUS BEHAVIOR (CENTER ON MOVIE)
# =========================================================
if focus_movie is not None:

    # bias view around selected movie
    cx = focus_movie["umap_x"]
    cy = focus_movie["umap_y"]

    # local window around focus
    filtered = df[
        (df["umap_x"] > cx - 1.5) &
        (df["umap_x"] < cx + 1.5) &
        (df["umap_y"] > cy - 1.5)
    ]


# =========================================================
# PLOT
# =========================================================
fig = px.scatter(

    filtered,

    x="umap_x",
    y="umap_y",

    color="macro_genre",
    size="visual_size",

    hover_name="title",

    hover_data={
        "weighted_rating": True,
        "rating_count": True,
        "genres": True,
        "cluster": True,
        "cluster_label": True
    },

    opacity=0.75,
    height=850
)


# =========================================================
# VISUAL ENHANCEMENTS (FEEL IMPROVEMENT)
# =========================================================
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


# =========================================================
# OPTIONAL: HIGHLIGHT SEARCH RESULT
# =========================================================
if focus_movie is not None:
    fig.add_scatter(
        x=[focus_movie["umap_x"]],
        y=[focus_movie["umap_y"]],
        mode="markers+text",
        marker=dict(size=18, color="white"),
        text=[focus_movie["title"]],
        textposition="top center",
        name="Focus"
    )


# =========================================================
# RENDER
# =========================================================
st.plotly_chart(fig, use_container_width=True)


# =========================================================
# INFO PANEL
# =========================================================
st.subheader("View Stats")

st.write("Movies displayed:", len(filtered))

st.write("Regions:", filtered["cluster_label"].nunique())