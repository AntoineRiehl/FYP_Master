#src/atlas/export/export_bundle.py

import json
from pathlib import Path
from dataclasses import asdict

from src.atlas.schema.atlas_bundle import AtlasBundle


def export_bundle(
    bundle: AtlasBundle,
    output_dir: Path,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_dir / "atlas.json",
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            [asdict(x) for x in bundle.atlas],
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(
        output_dir / "landmarks.json",
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            [asdict(x) for x in bundle.landmarks],
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(
        output_dir / "regions.json",
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            [asdict(x) for x in bundle.regions],
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            {
                "domain": bundle.domain,
                "feature_config": asdict(bundle.feature_config),
                "metadata": bundle.metadata or {}
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Exported atlas bundle to {output_dir}")