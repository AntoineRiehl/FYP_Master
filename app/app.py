import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "movie_map_v1.csv"
# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Movie Semantic Atlas",
    layout="wide"
)

st.title("🎬 Movie Semantic Atlas")


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_PATH
    )


df = load_data()


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("Filters")

min_ratings = st.sidebar.slider(
    "Minimum Ratings",
    0,
    500,
    50
)

genres = sorted(
    df["macro_genre"]
    .dropna()
    .unique()
)

selected_genre = st.sidebar.selectbox(
    "Genre",
    ["All"] + list(genres)
)


# =========================================================
# FILTERING
# =========================================================

filtered = df[
    df["rating_count"] >= min_ratings
]

if selected_genre != "All":

    filtered = filtered[
        filtered["macro_genre"] == selected_genre
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
        "genres": True
    },

    opacity=0.7,
    height=850
)

fig.update_layout(
    template="plotly_dark",
    dragmode="pan"
)

st.plotly_chart(
    fig,
    use_container_width=True
)