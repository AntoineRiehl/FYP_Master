#src/atlas/clustering/region_labels.py

import pandas as pd


def create_region_labels(df):

    cluster_labels = {}

    clusters = sorted(df["cluster"].unique())

    for cluster in clusters:

        cluster_movies = df[df["cluster"] == cluster]

        top_genres = (
            cluster_movies["macro_genre"]
            .value_counts()
            .head(2)
            .index
            .tolist()
        )

        label = " / ".join(top_genres)

        cluster_labels[cluster] = label

    df["cluster_label"] = df["cluster"].map(cluster_labels)

    return df