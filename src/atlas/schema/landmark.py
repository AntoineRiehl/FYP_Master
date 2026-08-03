#src/atlas/schema/landmark.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class Landmark:

    id: str

    label: str

    x: float
    y: float

    importance: float

    cluster: Optional[int] = None

    description: Optional[str] = None