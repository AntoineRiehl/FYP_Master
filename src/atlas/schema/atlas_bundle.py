#src/atlas/schema/atlas_bundle.py

from dataclasses import dataclass, field
from typing import List, Optional

from .atlas_item import AtlasItem
from .region import Region
from .landmark import Landmark
from .feature_config import FeatureConfig


@dataclass
class AtlasBundle:

    domain: str

    feature_config: FeatureConfig

    atlas: List[AtlasItem] = field(default_factory=list)

    landmarks: List[Landmark] = field(default_factory=list)

    regions: List[Region] = field(default_factory=list)

    metadata: Optional[dict] = None