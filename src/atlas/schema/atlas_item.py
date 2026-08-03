#src/atlas/schema/atlas_item.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Review:

    text: str

    source: Optional[str] = None

    rating: Optional[float] = None

    date: Optional[str] = None



@dataclass
class TextFeatures:

    tags: List[str] = field(
        default_factory=list
    )

    categories: List[str] = field(
        default_factory=list
    )

    reviews: List[Review] = field(
        default_factory=list
    )



@dataclass
class MediaMetadata:

    year: Optional[int] = None

    country: Optional[str] = None

    language: Optional[str] = None


    # Movie

    director: Optional[str] = None

    actors: List[str] = field(
        default_factory=list
    )


    # Music

    artist: Optional[str] = None

    album: Optional[str] = None


    # Restaurant

    address: Optional[str] = None

    city: Optional[str] = None

    latitude: Optional[float] = None

    longitude: Optional[float] = None



@dataclass
class Statistics:

    rating: Optional[float] = None

    rating_count: Optional[int] = None

    popularity: Optional[float] = None



@dataclass
class AtlasPosition:

    x: float

    y: float



@dataclass
class AtlasVisual:

    size: float

    cluster: Optional[int] = None

    cluster_label: Optional[str] = None



@dataclass
class AtlasItem:

    # Identity

    id: str

    source_id: Optional[str]

    title: str

    domain: str


    # Information

    metadata: MediaMetadata = field(
        default_factory=MediaMetadata
    )

    text: TextFeatures = field(
        default_factory=TextFeatures
    )

    statistics: Statistics = field(
        default_factory=Statistics
    )


    # Atlas information

    position: Optional[AtlasPosition] = None

    visual: Optional[AtlasVisual] = None


    # Future

    enrichment: Dict[str, Any] = field(
        default_factory=dict
    )