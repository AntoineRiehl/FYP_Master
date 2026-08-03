#src/atlas/schema/region.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class Region:

    id: int

    label: str

    x: float
    y: float

    size: float

    item_count: int

    description: Optional[str] = None

    color: Optional[str] = None