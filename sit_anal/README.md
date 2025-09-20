# Situational Analysis (Computer Vision)

A small, **extensible** pipeline to perform situational analysis on a video stream:
- pluggable detectors (color/shape demo + YOLO stub),
- simple multi-object **centroid tracker**,
- **rules engine** + optional logistic scoring to compute a risk score,
- overlay rendering with a **dynamic risk bar** and per-event timeline,
- CSV logging.

> Works offline out-of-the-box (no heavy models required). You can later plug your own detector.

## Quickstart
```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run on synthetic demo (generates frames)
python src/app.py --source synthetic

# Or run on an image/video
python src/app.py --image sample.png
# python src/app.py --video input.mp4
```
