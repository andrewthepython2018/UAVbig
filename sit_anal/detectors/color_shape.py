import cv2
import numpy as np
from typing import List, Dict
from .base import BaseDetector

class ColorShapeDetector(BaseDetector):
    """Very lightweight, heuristic-based detector:
    - Detects yellow-ish circular regions => 'helmet'
    - Detects gray-ish rectangle => 'book'
    - Detects long thin red-ish rectangle => 'knife'
    """
    def __init__(self):
        pass

    def detect(self, frame) -> List[Dict]:
        dets = []
        h, w = frame.shape[:2]

        # 1) Helmet-like: yellow circle
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([20, 80, 80]); upper = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 500: 
                continue
            (x,y), radius = cv2.minEnclosingCircle(c)
            circ_area = np.pi * radius * radius
            roundness = area / (circ_area + 1e-6)
            if 0.6 < roundness <= 1.2:
                dets.append({
                    "label": "helmet",
                    "conf": float(min(0.99, 0.5 + 0.5*roundness)),
                    "bbox": [int(x-radius), int(y-radius), int(2*radius), int(2*radius)],
                    "centroid": (int(x), int(y))
                })

        # 2) Book-like: gray rectangle
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x,y,w2,h2 = cv2.boundingRect(c)
            if w2*h2 < 2000: 
                continue
            roi = frame[y:y+h2, x:x+w2]
            mean = roi.mean(axis=(0,1))
            if abs(float(mean[0])-float(mean[1])) < 15 and abs(float(mean[1])-float(mean[2])) < 15:  # gray-ish
                aspect = max(w2,h2)/max(1.0,min(w2,h2))
                if 1.0 <= aspect <= 2.5:
                    dets.append({
                        "label": "book",
                        "conf": 0.7,
                        "bbox": [x,y,w2,h2],
                        "centroid": (int(x+w2/2), int(y+h2/2))
                    })

        # 3) Knife-like: thin red rectangle
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, np.array([0,70,50]), np.array([10,255,255]))
        red2 = cv2.inRange(hsv, np.array([170,70,50]), np.array([180,255,255]))
        red = cv2.bitwise_or(red1, red2)
        cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x,y,w2,h2 = cv2.boundingRect(c)
            if w2*h2 < 1000:
                continue
            aspect = max(w2,h2)/max(1.0,min(w2,h2))
            if aspect > 5.0:  # long and thin
                dets.append({
                    "label": "knife",
                    "conf": 0.85,
                    "bbox": [x,y,w2,h2],
                    "centroid": (int(x+w2/2), int(y+h2/2))
                })
        return dets
