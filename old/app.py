import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(page_title="Movie Semantic Atlas", layout="wide")

st.title("🎬 Movie Semantic Atlas v1")

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

df = load_data()

# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("Filters")

min_ratings = st.sidebar.slider(
    "Minimum number of ratings",
    0, 500, 50
)

genres = df["macro_genre"].dropna().unique()
selected_genre = st.sidebar.selectbox(
    "Macro Genre",
    ["All"] + sorted(list(genres))
)

clusters = df["cluster"].dropna().unique()
selected_cluster = st.sidebar.selectbox(
    "Cluster",
    ["All"] + sorted(list(clusters))
)

# =========================================================
# FILTER DATA
# =========================================================

filtered = df[df["rating_count"] >= min_ratings]

if selected_genre != "All":
    filtered = filtered[filtered["macro_genre"] == selected_genre]

if selected_cluster != "All":
    filtered = filtered[filtered["cluster"] == selected_cluster]

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
        "avg_rating": True,
        "weighted_rating": True,
        "rating_count": True,
        "genres": True,
        "cluster": True
    },

    opacity=0.7,
    height=800
)

fig.update_layout(
    template="plotly_dark",
    dragmode="pan",
    legend_title="Genre"
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# BASIC STATS PANEL
# =========================================================

st.subheader("Dataset Overview")

st.write(f"Movies shown: {len(filtered)}")

st.write(filtered["macro_genre"].value_counts())