# src/atlas/export/export_bundle.py

import json
import math
from pathlib import Path
from dataclasses import asdict

from src.atlas.schema.atlas_bundle import AtlasBundle


# =========================================================
# JSON SANITISATION
# =========================================================

def _sanitize_for_json(value):
    """
    Recursively convert values that are not valid standard
    JSON into JSON-safe representations.

    In particular:

        NaN        -> None
        +Infinity  -> None
        -Infinity  -> None

    This is important because JavaScript's JSON.parse()
    does not accept NaN or Infinity.
    """

    # -----------------------------------------------------
    # FLOAT SPECIAL VALUES
    # -----------------------------------------------------

    if isinstance(value, float):

        if not math.isfinite(value):

            return None

        return value

    # -----------------------------------------------------
    # DICTIONARIES
    # -----------------------------------------------------

    if isinstance(value, dict):

        return {
            key: _sanitize_for_json(item)
            for key, item in value.items()
        }

    # -----------------------------------------------------
    # LISTS / TUPLES
    # -----------------------------------------------------

    if isinstance(value, (list, tuple)):

        return [
            _sanitize_for_json(item)
            for item in value
        ]

    # -----------------------------------------------------
    # EVERYTHING ELSE
    # -----------------------------------------------------

    return value


# =========================================================
# EXPORT ATLAS BUNDLE
# =========================================================

def export_bundle(
    bundle: AtlasBundle,
    output_dir: Path,
):

    # -----------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # -----------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # ATLAS
    # -----------------------------------------------------

    atlas_data = _sanitize_for_json(
        [
            asdict(x)
            for x in bundle.atlas
        ]
    )

    with open(
        output_dir / "atlas.json",
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            atlas_data,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )

    # -----------------------------------------------------
    # LANDMARKS
    # -----------------------------------------------------

    landmarks_data = _sanitize_for_json(
        [
            asdict(x)
            for x in bundle.landmarks
        ]
    )

    with open(
        output_dir / "landmarks.json",
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            landmarks_data,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )

    # -----------------------------------------------------
    # REGIONS
    # -----------------------------------------------------

    regions_data = _sanitize_for_json(
        [
            asdict(x)
            for x in bundle.regions
        ]
    )

    with open(
        output_dir / "regions.json",
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            regions_data,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )

    # -----------------------------------------------------
    # METADATA
    # -----------------------------------------------------

    metadata_data = _sanitize_for_json(
        {
            "domain": bundle.domain,

            "feature_config":
                asdict(
                    bundle.feature_config
                ),

            "metadata":
                bundle.metadata or {}
        }
    )

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            metadata_data,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )

    # -----------------------------------------------------
    # COMPLETE
    # -----------------------------------------------------

    print(
        f"Exported atlas bundle to {output_dir}"
    )