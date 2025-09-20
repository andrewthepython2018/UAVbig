# UAV Ground Station (Streamlit)

A compact ground station you can run in a browser. It supports:
- **Mission editor** (add/move/remove waypoints with lat/lon/alt),
- **Simulated telemetry** following the mission,
- **Map view** (pydeck) with path and current vehicle position,
- **Mission export/import** (CSV and simple QGC WPL v110 format),
- **Data logging** (CSV).

> No hardware required. You can later swap the simulator with a MAVLink backend.

## Quickstart
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
