#src/pipelines/check_cross_domain_cluster_sizes.py

from pathlib import Path
import json
from collections import Counter


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

ATLASES = {

    "Movies + Music":
        ROOT
        / "frontend"
        / "public"
        / "data"
        / "movies_music"
        / "atlas.json",

    "Movies + Music + Restaurants":
        ROOT
        / "frontend"
        / "public"
        / "data"
        / "movies_music_restaurants"
        / "atlas.json",
}


# =========================================================
# GET NODES
# =========================================================

def get_nodes(payload):

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):

        for key in (
            "atlas",
            "nodes",
            "items",
        ):

            if isinstance(
                payload.get(key),
                list
            ):

                return payload[key]

    raise ValueError(
        "Could not find node list in atlas JSON."
    )


# =========================================================
# GET CLUSTER
# =========================================================

def get_cluster(node):
    """
    Supports the most likely atlas schemas.
    """

    # Direct field
    if "cluster" in node:
        return node["cluster"]

    # Cluster stored inside visual
    visual = node.get("visual")

    if isinstance(visual, dict):

        if "cluster" in visual:
            return visual["cluster"]

        if "cluster_id" in visual:
            return visual["cluster_id"]

    # Possible region structure
    region = node.get("region")

    if isinstance(region, dict):

        if "cluster" in region:
            return region["cluster"]

        if "id" in region:
            return region["id"]

    return None


# =========================================================
# ANALYSE
# =========================================================

for atlas_name, atlas_path in ATLASES.items():

    print()
    print("=" * 70)
    print(atlas_name)
    print("=" * 70)

    if not atlas_path.exists():

        print(
            f"File not found:\n{atlas_path}"
        )

        continue

    with atlas_path.open(
        "r",
        encoding="utf-8"
    ) as f:

        payload = json.load(f)

    nodes = get_nodes(payload)

    clusters = [
        get_cluster(node)
        for node in nodes
    ]

    missing = sum(
        cluster is None
        for cluster in clusters
    )

    clusters = [
        cluster
        for cluster in clusters
        if cluster is not None
    ]

    if not clusters:

        print(
            "No cluster information found."
        )

        continue

    counts = Counter(clusters)

    largest_cluster, largest_count = (
        counts.most_common(1)[0]
    )

    total_clustered = len(clusters)

    share = (
        largest_count
        /
        total_clustered
        *
        100
    )

    print(
        f"Total nodes:          {len(nodes):,}"
    )

    print(
        f"Clustered nodes:      {total_clustered:,}"
    )

    print(
        f"Number of clusters:   {len(counts):,}"
    )

    print(
        f"Largest cluster ID:   {largest_cluster}"
    )

    print(
        f"Largest cluster size: {largest_count:,}"
    )

    print(
        f"Largest cluster share: {share:.2f}%"
    )

    if missing:

        print(
            f"Nodes without cluster: {missing:,}"
        )