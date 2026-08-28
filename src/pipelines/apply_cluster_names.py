# src/pipelines/apply_cluster_names.py

from __future__ import annotations

import argparse
import json
import os
import shutil

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =========================================================
# PATHS
# =========================================================

FINAL_NAMES_ROOT = Path(
    "data/processed/cluster_names/final_names"
)

FRONTEND_DATA_ROOT = Path(
    "frontend/public/data"
)

BACKUP_ROOT = Path(
    "data/processed/cluster_names/backups"
)


# =========================================================
# ATLASES
# =========================================================

ATLAS_IDS = [

    "movies",
    "music",
    "restaurants",

    "movies_music",
    "movies_music_feel",

    "movies_music_restaurants",
    "movies_music_restaurants_feel",

]


# =========================================================
# FILE HELPERS
# =========================================================

def load_json(
    path: Path
) -> Any:


    with path.open(
        "r",
        encoding="utf8"
    ) as file:

        return json.load(
            file
        )


def write_json_atomic(
    path: Path,
    payload: Any
) -> None:
    """
    Write JSON to a temporary file first, then replace
    the target atomically.

    This reduces the risk of leaving a partially written
    frontend file if something goes wrong mid-write.
    """


    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )


    with temporary_path.open(
        "w",
        encoding="utf8"
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False
        )


    os.replace(
        temporary_path,
        path
    )


# =========================================================
# CLUSTER HELPERS
# =========================================================

def get_node_cluster_id(
    node: dict[str, Any]
) -> int | None:


    cluster = (

        node
        .get(
            "visual",
            {}
        )
        .get(
            "cluster"
        )

    )


    if cluster is None:

        return None


    try:

        cluster_id = int(
            cluster
        )

    except (
        TypeError,
        ValueError
    ):

        return None


    if cluster_id < 0:

        return None


    return cluster_id


def get_region_cluster_id(
    region: dict[str, Any]
) -> int | None:


    value = region.get(
        "id"
    )


    if value is None:

        return None


    try:

        cluster_id = int(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return None


    if cluster_id < 0:

        return None


    return cluster_id


# =========================================================
# VALIDATE NAME FILE
# =========================================================

def validate_name_file(
    atlas_id: str,
    names: Any
) -> dict[str, dict[str, str]]:


    if not isinstance(
        names,
        dict
    ):

        raise ValueError(

            f"{atlas_id}: naming file must be a JSON "
            "object keyed by cluster ID."

        )


    validated: dict[
        str,
        dict[str, str]
    ] = {}


    for (
        raw_cluster_id,
        record
    ) in names.items():


        try:

            cluster_id = int(
                raw_cluster_id
            )

        except (
            TypeError,
            ValueError
        ) as error:

            raise ValueError(

                f"{atlas_id}: invalid cluster ID "
                f"{raw_cluster_id!r}."

            ) from error


        if cluster_id < 0:

            raise ValueError(

                f"{atlas_id}: cluster ID "
                f"{cluster_id} cannot be negative."

            )


        if not isinstance(
            record,
            dict
        ):

            raise ValueError(

                f"{atlas_id}: cluster {cluster_id} "
                "must contain an object."

            )


        name = str(

            record.get(
                "name",
                ""
            )

        ).strip()


        description = str(

            record.get(
                "description",
                ""
            )

        ).strip()


        if not name:

            raise ValueError(

                f"{atlas_id}: cluster {cluster_id} "
                "has an empty name."

            )


        if not description:

            raise ValueError(

                f"{atlas_id}: cluster {cluster_id} "
                "has an empty description."

            )


        validated[
            str(
                cluster_id
            )
        ] = {

            "name":
                name,

            "description":
                description

        }


    return validated


# =========================================================
# LOAD + VALIDATE ONE ATLAS
# =========================================================

def prepare_atlas(
    atlas_id: str
) -> dict[str, Any]:


    names_path = (

        FINAL_NAMES_ROOT
        /
        f"{atlas_id}.json"

    )


    atlas_path = (

        FRONTEND_DATA_ROOT
        /
        atlas_id
        /
        "atlas.json"

    )


    regions_path = (

        FRONTEND_DATA_ROOT
        /
        atlas_id
        /
        "regions.json"

    )


    # -----------------------------------------------------
    # REQUIRED FILES
    # -----------------------------------------------------

    if not names_path.exists():

        raise FileNotFoundError(

            f"{atlas_id}: naming file not found:\n"
            f"{names_path}"

        )


    if not atlas_path.exists():

        raise FileNotFoundError(

            f"{atlas_id}: frontend atlas not found:\n"
            f"{atlas_path}"

        )


    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    names = validate_name_file(

        atlas_id,

        load_json(
            names_path
        )

    )


    nodes = load_json(
        atlas_path
    )


    if not isinstance(
        nodes,
        list
    ):

        raise ValueError(

            f"{atlas_id}: atlas.json must contain "
            "a JSON array."

        )


    regions = None


    if regions_path.exists():

        regions = load_json(
            regions_path
        )


        if not isinstance(
            regions,
            list
        ):

            raise ValueError(

                f"{atlas_id}: regions.json must contain "
                "a JSON array."

            )


    # -----------------------------------------------------
    # CLUSTER IDS USED BY THE ATLAS
    # -----------------------------------------------------

    atlas_cluster_ids = {

        cluster_id

        for node
        in nodes

        if (
            cluster_id :=
                get_node_cluster_id(
                    node
                )
        )
        is not None

    }


    naming_cluster_ids = {

        int(
            cluster_id
        )

        for cluster_id
        in names.keys()

    }


    # -----------------------------------------------------
    # ENSURE EVERY ATLAS CLUSTER HAS A NAME
    # -----------------------------------------------------

    missing_names = sorted(

        atlas_cluster_ids
        -
        naming_cluster_ids

    )


    if missing_names:

        raise ValueError(

            f"{atlas_id}: naming file is missing "
            f"{len(missing_names)} cluster(s):\n"
            f"{missing_names}"

        )


    # -----------------------------------------------------
    # ENSURE NO UNKNOWN CLUSTERS WERE PROVIDED
    # -----------------------------------------------------

    extra_names = sorted(

        naming_cluster_ids
        -
        atlas_cluster_ids

    )


    if extra_names:

        raise ValueError(

            f"{atlas_id}: naming file contains "
            f"{len(extra_names)} cluster(s) that are "
            "not present in atlas.json:\n"
            f"{extra_names}"

        )


    # -----------------------------------------------------
    # REGION VALIDATION
    # -----------------------------------------------------

    if regions is not None:


        region_cluster_ids = {

            cluster_id

            for region
            in regions

            if (
                cluster_id :=
                    get_region_cluster_id(
                        region
                    )
            )
            is not None

        }


        missing_regions = sorted(

            atlas_cluster_ids
            -
            region_cluster_ids

        )


        if missing_regions:

            print(

                f"WARNING: {atlas_id} regions.json "
                f"does not contain "
                f"{len(missing_regions)} atlas "
                "cluster(s)."

            )


    return {

        "atlas_id":
            atlas_id,

        "names":
            names,

        "nodes":
            nodes,

        "regions":
            regions,

        "names_path":
            names_path,

        "atlas_path":
            atlas_path,

        "regions_path":
            regions_path,

        "cluster_count":
            len(
                atlas_cluster_ids
            )

    }


# =========================================================
# BACKUP
# =========================================================

def create_backup(
    prepared_atlases: list[
        dict[str, Any]
    ]
) -> Path:


    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )


    backup_directory = (

        BACKUP_ROOT
        /
        timestamp

    )


    for prepared in prepared_atlases:


        atlas_id = prepared[
            "atlas_id"
        ]


        atlas_backup_directory = (

            backup_directory
            /
            atlas_id

        )


        atlas_backup_directory.mkdir(
            parents=True,
            exist_ok=True
        )


        shutil.copy2(

            prepared[
                "atlas_path"
            ],

            atlas_backup_directory
            /
            "atlas.json"

        )


        regions_path = prepared[
            "regions_path"
        ]


        if regions_path.exists():

            shutil.copy2(

                regions_path,

                atlas_backup_directory
                /
                "regions.json"

            )


    return backup_directory


# =========================================================
# RESTORE BACKUP
# =========================================================

def restore_backup(
    prepared_atlases: list[
        dict[str, Any]
    ],
    backup_directory: Path
) -> None:


    for prepared in prepared_atlases:


        atlas_id = prepared[
            "atlas_id"
        ]


        atlas_backup_directory = (

            backup_directory
            /
            atlas_id

        )


        backup_atlas = (

            atlas_backup_directory
            /
            "atlas.json"

        )


        if backup_atlas.exists():

            shutil.copy2(

                backup_atlas,

                prepared[
                    "atlas_path"
                ]

            )


        backup_regions = (

            atlas_backup_directory
            /
            "regions.json"

        )


        if backup_regions.exists():

            shutil.copy2(

                backup_regions,

                prepared[
                    "regions_path"
                ]

            )


# =========================================================
# APPLY ONE ATLAS
# =========================================================

def apply_atlas(
    prepared: dict[str, Any]
) -> tuple[int, int]:


    names = prepared[
        "names"
    ]


    nodes = prepared[
        "nodes"
    ]


    regions = prepared[
        "regions"
    ]


    # -----------------------------------------------------
    # UPDATE ATLAS NODES
    # -----------------------------------------------------

    updated_nodes = 0


    for node in nodes:


        cluster_id = get_node_cluster_id(
            node
        )


        if cluster_id is None:

            continue


        record = names[
            str(
                cluster_id
            )
        ]


        node.setdefault(
            "visual",
            {}
        )


        node[
            "visual"
        ][
            "cluster_label"
        ] = record[
            "name"
        ]


        updated_nodes += 1


    # -----------------------------------------------------
    # UPDATE REGIONS
    # -----------------------------------------------------

    updated_regions = 0


    if regions is not None:


        for region in regions:


            cluster_id = get_region_cluster_id(
                region
            )


            if cluster_id is None:

                continue


            record = names.get(
                str(
                    cluster_id
                )
            )


            if record is None:

                continue


            region[
                "label"
            ] = record[
                "name"
            ]


            region[
                "description"
            ] = record[
                "description"
            ]


            updated_regions += 1


    # -----------------------------------------------------
    # WRITE FILES
    # -----------------------------------------------------

    write_json_atomic(

        prepared[
            "atlas_path"
        ],

        nodes

    )


    if regions is not None:

        write_json_atomic(

            prepared[
                "regions_path"
            ],

            regions

        )


    return (
        updated_nodes,
        updated_regions
    )


# =========================================================
# MAIN
# =========================================================

def run(
    atlas_ids: list[str],
    dry_run: bool
) -> None:


    print(
        "\n"
        "=================================================="
    )

    print(
        "APPLY FINAL CLUSTER NAMES"
    )

    print(
        "=================================================="
    )


    # =====================================================
    # LOAD + VALIDATE EVERYTHING BEFORE WRITING ANYTHING
    # =====================================================

    prepared_atlases = []


    for atlas_id in atlas_ids:


        print(
            f"\nValidating {atlas_id}..."
        )


        prepared = prepare_atlas(
            atlas_id
        )


        prepared_atlases.append(
            prepared
        )


        print(

            f"  clusters: "
            f"{prepared['cluster_count']:,}"

        )


        print(

            f"  names:    "
            f"{len(prepared['names']):,}"

        )


    print(
        "\nAll requested atlases validated successfully."
    )


    # =====================================================
    # DRY RUN
    # =====================================================

    if dry_run:


        print(
            "\nDRY RUN COMPLETE."
        )


        print(
            "No frontend files were modified."
        )


        return


    # =====================================================
    # BACKUP
    # =====================================================

    print(
        "\nCreating frontend backup..."
    )


    backup_directory = create_backup(
        prepared_atlases
    )


    print(
        f"Backup saved to:\n"
        f"{backup_directory}"
    )


    # =====================================================
    # APPLY
    # =====================================================

    try:


        print(
            "\nApplying final cluster names..."
        )


        for prepared in prepared_atlases:


            (
                updated_nodes,
                updated_regions
            ) = apply_atlas(
                prepared
            )


            print(

                f"  {prepared['atlas_id']}: "
                f"{updated_nodes:,} nodes, "
                f"{updated_regions:,} regions"

            )


    except Exception:


        print(
            "\nERROR while applying names."
        )


        print(
            "Restoring backup..."
        )


        restore_backup(

            prepared_atlases,

            backup_directory

        )


        print(
            "Frontend files restored."
        )


        raise


    # =====================================================
    # COMPLETE
    # =====================================================

    print(
        "\n"
        "=================================================="
    )

    print(
        "CLUSTER NAMES APPLIED SUCCESSFULLY"
    )

    print(
        "=================================================="
    )


    print(
        f"\nBackup retained at:\n"
        f"{backup_directory}"
    )


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:


    parser = argparse.ArgumentParser(

        description=(

            "Apply final LLM-generated cluster names "
            "to the frontend atlas and region files."

        )

    )


    parser.add_argument(

        "--atlas",

        nargs="+",

        choices=
            ATLAS_IDS,

        default=None,

        help=(

            "Optional subset of atlases. "
            "If omitted, all seven atlases are updated."

        )

    )


    parser.add_argument(

        "--dry-run",

        action="store_true",

        help=(

            "Validate all files without modifying "
            "the frontend."

        )

    )


    return parser.parse_args()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":


    args = parse_args()


    atlas_ids = (

        args.atlas

        if args.atlas

        else ATLAS_IDS

    )


    run(

        atlas_ids=
            atlas_ids,

        dry_run=
            args.dry_run

    )
