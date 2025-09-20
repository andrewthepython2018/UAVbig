import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="UAV Ground Station", layout="wide")
st.title("UAV Ground Station — Demo")


def new_mission():
    return pd.DataFrame([
        {"seq":0,"frame":3,"command":"WAYPOINT","lat":41.3111,"lon":69.2797,"alt":50.0},
        {"seq":1,"frame":3,"command":"WAYPOINT","lat":41.3200,"lon":69.2700,"alt":50.0},
        {"seq":2,"frame":3,"command":"WAYPOINT","lat":41.3300,"lon":69.2850,"alt":55.0},
    ])

if "mission" not in st.session_state:
    st.session_state.mission = new_mission()
if "sim" not in st.session_state:
    st.session_state.sim = {"active": False, "idx": 0, "pos": [st.session_state.mission.iloc[0].lat, st.session_state.mission.iloc[0].lon, st.session_state.mission.iloc[0].alt], "speed": 20.0}

def simulate_step():
    m = st.session_state.mission.reset_index(drop=True)
    s = st.session_state.sim
    if not s["active"] or len(m)==0: return
    i = s["idx"]; j = (i+1) % len(m)
    lat, lon, alt = s["pos"]
    target = np.array([m.loc[j,"lat"], m.loc[j,"lon"], m.loc[j,"alt"]], dtype=float)
    cur = np.array([lat, lon, alt], dtype=float)
    step = target - cur
    # simple proportional step
    cur = cur + 0.10 * step
    # close enough -> next wp
    if np.linalg.norm(step[:2]) < 1e-4 and abs(step[2])<0.5:
        s["idx"] = j
    s["pos"] = cur.tolist()

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Mission Editor")
    edited = st.data_editor(
        st.session_state.mission,
        num_rows="dynamic",
        use_container_width=True,
        column_config={"command": st.column_config.SelectboxColumn(options=["WAYPOINT"])}
    )
    st.session_state.mission = edited

    c1,c2,c3,c4 = st.columns(4)
    if c1.button("New Mission"):
        st.session_state.mission = new_mission()
    uploaded = c2.file_uploader("Import CSV/QGC WPL", type=["csv","wpl"])
    if uploaded:
        name = uploaded.name.lower()
        if name.endswith(".csv"):
            st.session_state.mission = pd.read_csv(uploaded)
        else:
            # very small WPL v110 parser
            lines = uploaded.read().decode("utf-8").strip().splitlines()
            rows = []
            for ln in lines:
                if ln.startswith("QGC"): continue
                parts = ln.split("\t")
                if len(parts) < 12: continue
                seq = int(parts[0]); frame = int(parts[2]); cmd = int(parts[3])
                lat = float(parts[8]); lon = float(parts[9]); alt = float(parts[10])
                rows.append({"seq":seq, "frame":frame, "command":"WAYPOINT", "lat":lat, "lon":lon, "alt":alt})
            st.session_state.mission = pd.DataFrame(rows)
    if c3.button("Export CSV"):
        st.download_button("Download mission.csv", data=st.session_state.mission.to_csv(index=False).encode("utf-8"), file_name="mission.csv", mime="text/csv", key="dlcsv")
    if c4.button("Export WPL"):
        # QGC WPL v110
        m = st.session_state.mission.reset_index(drop=True)
        lines = ["QGC WPL 110"]
        for i,row in m.iterrows():
            lines.append("\t".join(map(str,[i, 0, row.frame, 16, 0,0,0,0, row.lat, row.lon, row.alt, 1])))
        st.download_button("Download mission.wpl", data=("\n".join(lines)).encode("utf-8"), file_name="mission.wpl", mime="text/plain", key="dlwpl")

with col2:
    st.subheader("Telemetry & Map")
    c1,c2,c3 = st.columns(3)
    s = st.session_state.sim
    s["active"] = c1.toggle("Simulation", value=s["active"]) 
    s["speed"] = c2.slider("Speed (m/s) [sim only]", 1.0, 50.0, s["speed"]) 
    if c3.button("Step once"):
        simulate_step()
    if s["active"]:
        simulate_step()

    lat, lon, alt = st.session_state.sim["pos"]
    st.metric("Lat", f"{lat:.5f}"); st.metric("Lon", f"{lon:.5f}"); st.metric("Alt", f"{alt:.1f} m")

    m = st.session_state.mission.reset_index(drop=True)
    path = [[row["lon"], row["lat"]] for _,row in m.iterrows()]
    current = [[lon, lat]]

    deck = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=12, pitch=0),
        layers=[
            pdk.Layer("PathLayer", data=[{"path": path, "name":"mission"}], get_path="path", width_scale=20, width_min_pixels=2, get_color=[0,0,255]),
            pdk.Layer("ScatterplotLayer", data=[{"position": current[0]}], get_position="position", get_radius=40, radius_min_pixels=6),
        ]
    )
    st.pydeck_chart(deck)

    # Logging
    if "log" not in st.session_state:
        st.session_state.log = []
    if st.button("Log snapshot"):
        st.session_state.log.append({"ts": datetime.utcnow().isoformat()+"Z", "lat": lat, "lon": lon, "alt": alt})
    if len(st.session_state.log):
        df = pd.DataFrame(st.session_state.log)
        st.download_button("Download log.csv", data=df.to_csv(index=False).encode("utf-8"), file_name="telemetry_log.csv", mime="text/csv")
