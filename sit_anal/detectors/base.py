from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

Detection = Dict[str, object]  # {label, conf, bbox(xywh), centroid(x,y)}

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame) -> List[Detection]:
        ...
