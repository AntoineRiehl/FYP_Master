# src/atlas/embeddings/dimensionality_reduction.py

import umap.umap_ as umap

def get_umap_projection(tfidf_matrix):

    print("Computing UMAP projection...")

    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )

    embedding = umap_model.fit_transform(
        tfidf_matrix
    )

    return embedding, umap_model