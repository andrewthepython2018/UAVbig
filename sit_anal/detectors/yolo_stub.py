from .base import BaseDetector
from typing import List, Dict

class YOLOStub(BaseDetector):
    """Placeholder class showing how to integrate a real detector later."""
    def __init__(self, model_path:str="yolo.onnx"):
        self.model_path = model_path
    def detect(self, frame) -> List[Dict]:
        # Return empty; replace with your actual inference
        return []
