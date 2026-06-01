from pathlib import Path
import joblib
import umap.umap_ as umap


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "embeddings"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

UMAP_PATH = MODEL_DIR / "umap_model.pkl"


# =========================================================
# BUILD OR LOAD UMAP
# =========================================================

def get_umap_projection(tfidf_matrix):

    # -----------------------------------------------------
    # LOAD EXISTING MODEL
    # -----------------------------------------------------

    if UMAP_PATH.exists():

        print("Loading existing UMAP model...")

        umap_model = joblib.load(UMAP_PATH)

    else:

        print("Training new UMAP model...")

        umap_model = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42
        )

        umap_model.fit(tfidf_matrix)

        joblib.dump(umap_model, UMAP_PATH)

        print("UMAP model saved.")

    # -----------------------------------------------------
    # PROJECT MOVIES
    # -----------------------------------------------------

    umap_result = umap_model.transform(
        tfidf_matrix
    )

    return umap_result, umap_model