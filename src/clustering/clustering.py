from pathlib import Path
import joblib
import hdbscan


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "clustering"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

HDBSCAN_PATH = MODEL_DIR / "hdbscan_model.pkl"


# =========================================================
# BUILD OR LOAD CLUSTER MODEL
# =========================================================

def compute_clusters(df):

    coords = df[["umap_x", "umap_y"]].values

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    if HDBSCAN_PATH.exists():

        print("Loading existing HDBSCAN model...")

        cluster_model = joblib.load(HDBSCAN_PATH)

    else:

        print("Training HDBSCAN model...")

        cluster_model = hdbscan.HDBSCAN(
            min_cluster_size=150,
            min_samples=20
        )

        cluster_model.fit(coords)

        joblib.dump(cluster_model, HDBSCAN_PATH)

        print("HDBSCAN model saved.")

    # -----------------------------------------------------
    # ASSIGN CLUSTERS
    # -----------------------------------------------------

    df["cluster"] = cluster_model.labels_

    return df