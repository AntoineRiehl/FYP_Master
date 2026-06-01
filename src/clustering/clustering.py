# src/clustering/clustering.py

import hdbscan


def compute_clusters(df):

    coords = df[
        ["umap_x", "umap_y"]
    ].values

    cluster_model = hdbscan.HDBSCAN(
        min_cluster_size=150,
        min_samples=20
    )

    labels = cluster_model.fit_predict(
        coords
    )

    df["cluster"] = labels

    return df