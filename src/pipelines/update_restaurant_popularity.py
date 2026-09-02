from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.atlas.visual.size_normalization import (
    normalize_visual_sizes
)


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    ROOT
    / "frontend"
    / "public"
    / "data"
)

ATLAS_PATHS = [

    DATA_ROOT
    / "restaurants"
    / "atlas.json",

    DATA_ROOT
    / "movies_music_restaurants"
    / "atlas.json",

    DATA_ROOT
    / "movies_music_restaurants_feel"
    / "atlas.json",

]

BACKUP_ROOT = (
    ROOT
    / "backups"
    / "restaurant_popularity_patch"
)

RESTAURANT_DOMAIN = "restaurants"

VISUAL_SIZE_STRENGTH = 1.8


# =========================================================
# FIND NODE LIST
# =========================================================

def get_nodes(payload):
    """
    Return the atlas node list.

    Supports either:

        [{...}, {...}]

    or wrapped structures such as:

        {"atlas": [...]}
        {"nodes": [...]}
        {"items": [...]}
    """

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):

        for key in (
            "atlas",
            "nodes",
            "items",
        ):

            value = payload.get(key)

            if isinstance(value, list):
                return value

    raise ValueError(
        "Could not find atlas node list."
    )


# =========================================================
# NUMERIC VALUE
# =========================================================

def to_float(
    value,
    label,
):

    try:

        return float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            f"Invalid numeric value for "
            f"{label}: {value!r}"
        ) from exc


# =========================================================
# ATOMIC JSON WRITE
# =========================================================

def atomic_write_json(
    path: Path,
    payload,
):

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )

        handle.write("\n")

    os.replace(
        temporary_path,
        path,
    )


# =========================================================
# BACKUP
# =========================================================

def create_backup(
    atlas_path: Path,
    backup_directory: Path,
):

    relative_path = atlas_path.relative_to(
        DATA_ROOT
    )

    destination = (
        backup_directory
        / relative_path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        atlas_path,
        destination,
    )

    return destination


# =========================================================
# PATCH ONE ATLAS
# =========================================================

def patch_atlas(
    atlas_path: Path,
    backup_directory: Path,
):

    print()
    print("=" * 72)
    print(f"Processing: {atlas_path}")
    print("=" * 72)

    # =====================================================
    # LOAD
    # =====================================================

    if not atlas_path.exists():

        raise FileNotFoundError(
            f"Atlas does not exist:\n"
            f"{atlas_path}"
        )

    with atlas_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        payload = json.load(handle)

    nodes = get_nodes(payload)

    if not nodes:

        raise ValueError(
            f"No nodes found in:\n"
            f"{atlas_path}"
        )


    # =====================================================
    # UPDATE RESTAURANT POPULARITY
    # =====================================================

    restaurant_count = 0

    restaurant_popularity = []

    for index, node in enumerate(nodes):

        statistics = node.get(
            "statistics"
        )

        if not isinstance(
            statistics,
            dict,
        ):

            raise ValueError(
                f"Node {index} has no valid "
                "'statistics' dictionary."
            )

        domain = node.get(
            "domain"
        )

        # -------------------------------------------------
        # Yelp review_count is exported in rating_count.
        #
        # Final restaurant popularity definition:
        #
        #     popularity = Yelp review_count
        # -------------------------------------------------

        if domain == RESTAURANT_DOMAIN:

            review_count = to_float(

                statistics.get(
                    "rating_count"
                ),

                (
                    "restaurant rating_count "
                    f"at node {index}"
                ),

            )

            if review_count < 0:

                raise ValueError(
                    f"Negative restaurant review "
                    f"count at node {index}."
                )

            statistics[
                "popularity"
            ] = review_count

            restaurant_popularity.append(
                review_count
            )

            restaurant_count += 1


    if restaurant_count == 0:

        raise ValueError(
            f"No restaurant nodes found in:\n"
            f"{atlas_path}"
        )


    # =====================================================
    # RECOMPUTE VISUAL SIZE WITHIN EACH DOMAIN
    # =====================================================
    #
    # THIS IS THE IMPORTANT PART.
    #
    # The original cross-domain pipelines normalize:
    #
    #     Movies       within Movies
    #     Music        within Music
    #     Restaurants  within Restaurants
    #
    # BEFORE combining the domains.
    #
    # Raw popularity measures therefore never compete
    # directly across domains.
    # =====================================================

    domain_indices = {}

    for index, node in enumerate(nodes):

        domain = node.get(
            "domain"
        )

        if not domain:

            raise ValueError(
                f"Node {index} has no domain."
            )

        domain_indices.setdefault(
            domain,
            []
        ).append(
            index
        )


    # -----------------------------------------------------
    # Normalize each domain independently
    # -----------------------------------------------------

    for domain, indices in domain_indices.items():

        popularity_values = []

        for index in indices:

            statistics = nodes[
                index
            ]["statistics"]

            popularity = to_float(

                statistics.get(
                    "popularity"
                ),

                (
                    f"{domain} popularity "
                    f"at node {index}"
                ),

            )

            popularity_values.append(
                popularity
            )


        domain_df = pd.DataFrame(
            {
                "popularity_score":
                    popularity_values
            }
        )


        normalized = normalize_visual_sizes(

            domain_df,

            popularity_column=
                "popularity_score",

            strength=
                VISUAL_SIZE_STRENGTH,

        )


        visual_sizes = (

            normalized[
                "visual_size"
            ]
            .astype(float)
            .tolist()

        )


        # -------------------------------------------------
        # Apply sizes back to ORIGINAL nodes
        # -------------------------------------------------

        for local_index, node_index in enumerate(
            indices
        ):

            visual = nodes[
                node_index
            ].get(
                "visual"
            )

            if not isinstance(
                visual,
                dict,
            ):

                visual = {}

                nodes[
                    node_index
                ][
                    "visual"
                ] = visual


            visual[
                "size"
            ] = float(
                visual_sizes[
                    local_index
                ]
            )


        # -------------------------------------------------
        # Diagnostics
        # -------------------------------------------------

        size_series = pd.Series(
            visual_sizes,
            dtype=float,
        )

        print()

        print(
            f"{domain.upper()} "
            f"({len(indices):,} nodes)"
        )

        print(
            "  Visual size:"
        )

        print(
            f"    min    = "
            f"{size_series.min():.3f}"
        )

        print(
            f"    median = "
            f"{size_series.median():.3f}"
        )

        print(
            f"    max    = "
            f"{size_series.max():.3f}"
        )


    # =====================================================
    # RESTAURANT DIAGNOSTICS
    # =====================================================

    restaurant_series = pd.Series(
        restaurant_popularity,
        dtype=float,
    )

    print()

    print(
        "Restaurant popularity "
        "(Yelp review count):"
    )

    print(
        f"  min    = "
        f"{restaurant_series.min():,.0f}"
    )

    print(
        f"  median = "
        f"{restaurant_series.median():,.0f}"
    )

    print(
        f"  max    = "
        f"{restaurant_series.max():,.0f}"
    )


    # =====================================================
    # BACKUP CURRENT VERSION
    # =====================================================

    backup_path = create_backup(

        atlas_path,
        backup_directory,

    )

    print()

    print(
        "Backup created:"
    )

    print(
        f"  {backup_path}"
    )


    # =====================================================
    # WRITE
    # =====================================================

    atomic_write_json(
        atlas_path,
        payload,
    )

    print()

    print(
        "Atlas patched successfully."
    )


# =========================================================
# MAIN
# =========================================================

def main():

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_directory = (
        BACKUP_ROOT
        / timestamp
    )


    print()

    print(
        "Restaurant popularity + "
        "within-domain visual-size patch"
    )

    print(
        "=" * 72
    )

    print()

    print(
        "Restaurant popularity:"
    )

    print(
        "  Yelp review_count"
    )

    print()

    print(
        "Visual-size normalization:"
    )

    print(
        "  percentile rank WITHIN EACH DOMAIN"
    )

    print()

    print(
        "Strength:"
    )

    print(
        f"  {VISUAL_SIZE_STRENGTH}"
    )


    for atlas_path in ATLAS_PATHS:

        patch_atlas(

            atlas_path,
            backup_directory,

        )


    print()

    print("=" * 72)

    print(
        "ALL ATLASES PATCHED SUCCESSFULLY"
    )

    print("=" * 72)

    print()

    print(
        "Backups:"
    )

    print(
        backup_directory
    )

    print()


if __name__ == "__main__":

    main()