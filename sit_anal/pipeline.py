import cv2, csv, time
from typing import Optional
from .detectors.color_shape import ColorShapeDetector
from .tracker import CentroidTracker
from .rules import load_config, score_risk, EMA

class Pipeline:
    def __init__(self, config_path="config.json"):
        self.cfg = load_config(config_path)
        self.detector = ColorShapeDetector()
        self.tracker = CentroidTracker(max_distance=60)
        self.ema = EMA(alpha=self.cfg['ema_alpha'])
        self.last_ts = None
        self.logfile = open("events.csv", "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.logfile)
        self.writer.writerow(["timestamp","label","conf","track_id","x","y","w","h","risk_raw","risk_smooth"])

    def close(self):
        self.logfile.close()

    def annotate(self, frame, dets, risk, risk_smooth):
        for d in dets:
            x,y,w,h = d['bbox']
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            tid = d.get('track_id', 0)
            cv2.putText(frame, f"{d['label']} {d['conf']:.2f} id={tid}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # risk bar
        H,W = frame.shape[:2]
        bar_w = int(W * 0.02)
        val = int((1.0 - risk_smooth) * (H-20))
        cv2.rectangle(frame, (W - 20, 10), (W - 20 + bar_w, H - 10), (50,50,50), 1)
        cv2.rectangle(frame, (W - 20, 10 + val), (W - 20 + bar_w, H - 10), (0,0,255), -1)
        cv2.putText(frame, f"Risk: {risk_smooth:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        return frame

    def step(self, frame):
        dets = self.detector.detect(frame)
        dets = self.tracker.update(dets)

        risk_raw = score_risk(dets, self.cfg)
        risk_smooth = self.ema.update(risk_raw)

        ts = time.time()
        for d in dets:
            x,y,w,h = d['bbox']
            self.writer.writerow([ts, d['label'], f"{d['conf']:.3f}", d.get('track_id',0), x,y,w,h, f"{risk_raw:.3f}", f"{risk_smooth:.3f}"])
        return dets, risk_raw, risk_smooth
