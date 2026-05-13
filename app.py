import os
TOMTOM_KEY = os.getenv("TOMTOM_KEY")
# ==========================================================
# OPTIMIZED STREAMLIT EV ROUTE DASHBOARD
# Performance-focused refactor
# ==========================================================

import os
import streamlit as st
import folium
import pandas as pd
import numpy as np
import requests
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor

# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(layout="wide")
TOMTOM_KEY = os.getenv("TOMTOM_KEY")

# ==========================================================
# STATIC DATA (NO RECOMPUTE)
# ==========================================================

routes = {
    "Toril → Roxas": {
        "start": (7.0160,125.5000),
        "end": (7.0725,125.6132),
        "gradient": 4,
        "type": "Highway"
    },
    "Panabo → Roxas": {
        "start": (7.3080,125.6840),
        "end": (7.0725,125.6132),
        "gradient": 5,
        "type": "Highway"
    },
    "Short City Trip": {
        "start": (7.0700,125.6000),
        "end": (7.0720,125.6040),
        "gradient": 2,
        "type": "Urban"
    }
}

vehicle_specs = {
    "Sedan": {"max_torque":180,"battery":45,"gradient_limit":8},
    "Mini Bus": {"max_torque":350,"battery":250,"gradient_limit":15},
    "E-Trike": {"max_torque":95,"battery":12,"gradient_limit":6}
}

# ==========================================================
# CACHED ML MODEL (IMPORTANT FIX)
# ==========================================================

@st.cache_resource
def load_models():
    np.random.seed(42)
    rows = []

    for _ in range(500):
        distance = np.random.uniform(1,35)
        speed = np.random.uniform(15,65)
        gradient = np.random.uniform(1,12)
        traffic = np.random.randint(1,4)

        torque = 40 + speed*0.8 + gradient*8 + traffic*18 + np.random.normal(0,6)
        power = 2 + distance*0.55 + speed*0.18 + gradient*1.4 + traffic*2.5 + np.random.normal(0,2)

        rows.append([distance,speed,gradient,traffic,torque,power])

    df = pd.DataFrame(rows, columns=["Distance","Speed","Gradient","Traffic","Torque","Power"])
    X = df[["Distance","Speed","Gradient","Traffic"]]

    torque_model = RandomForestRegressor(n_estimators=120, random_state=42)
    power_model = RandomForestRegressor(n_estimators=120, random_state=42)

    torque_model.fit(X, df["Torque"])
    power_model.fit(X, df["Power"])

    return torque_model, power_model


torque_model, power_model = load_models()

# ==========================================================
# CACHED ROUTE (CRITICAL API REDUCTION)
# ==========================================================

@st.cache_data
def get_route(start, end):
    start_str = f"{start[0]},{start[1]}"
    end_str = f"{end[0]},{end[1]}"

    url = f"https://api.tomtom.com/routing/1/calculateRoute/{start_str}:{end_str}/json?key={TOMTOM_KEY}"
    return requests.get(url, timeout=10).json()

# ==========================================================
# CACHED TRAFFIC (REDUCED CALLS)
# ==========================================================

@st.cache_data
def get_traffic(lat, lon):
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key={TOMTOM_KEY}"
    try:
        return requests.get(url, timeout=5).json()
    except:
        return {}

# ==========================================================
# UI
# ==========================================================

st.sidebar.header("Settings")
route_name = st.sidebar.selectbox("Choose Route", list(routes.keys()))
vehicle = st.sidebar.selectbox("Vehicle Type", list(vehicle_specs.keys()))
map_type = st.sidebar.radio("Map", ["Route Map","Torque Zoning","Energy Consumption"])

route = routes[route_name]

# ==========================================================
# ROUTE BUILDING (CACHED LOGIC)
# ==========================================================

if "points" in route:
    route_points = route["points"]
    distance = 11
    duration = 35
else:
    data = get_route(route["start"], route["end"])

    if "routes" not in data:
        st.error("Routing API error")
        st.stop()

    summary = data["routes"][0]["summary"]
    distance = summary["lengthInMeters"] / 1000
    duration = summary["travelTimeInSeconds"] / 60

    route_points = [
        (p["latitude"], p["longitude"])
        for leg in data["routes"][0]["legs"]
        for p in leg["points"]
    ]

gradient = route["gradient"]

# ==========================================================
# TRAFFIC (LIMITED SAMPLE POINTS ONLY)
# ==========================================================

sample_points = route_points[::max(1, len(route_points)//5)][:5]
ratios = []
speeds = []

for lat, lon in sample_points:
    t = get_traffic(lat, lon)
    if "flowSegmentData" in t:
        cur = t["flowSegmentData"]["currentSpeed"]
        free = t["flowSegmentData"]["freeFlowSpeed"]
        ratios.append(cur/free)
        speeds.append(cur)

if ratios:
    traffic_ratio = np.mean(ratios)
    speed = np.mean(speeds)
else:
    traffic_ratio = 0.8
    speed = 35

traffic = 3 if traffic_ratio < 0.45 else 2 if traffic_ratio < 0.75 else 1
traffic_label = {1:"Low",2:"Moderate",3:"Heavy"}[traffic]

# ==========================================================
# PREDICTION (FAST)
# ==========================================================

X = pd.DataFrame({
    "Distance":[distance],
    "Speed":[speed],
    "Gradient":[gradient],
    "Traffic":[traffic]
})

torque_pred = torque_model.predict(X)[0]
power_pred = power_model.predict(X)[0]

if vehicle == "E-Trike":
    power_pred = distance * 0.07
    torque_pred = min(torque_pred, 80)

spec = vehicle_specs[vehicle]

score = (
    0.4*(spec["max_torque"]/torque_pred)*100 +
    0.4*max(0,100-(power_pred/spec["battery"])*100) +
    0.2*min(100,(spec["gradient_limit"]/gradient)*100)
)

# ==========================================================
# MAP (SIMPLIFIED RENDER)
# ==========================================================

st.subheader(map_type)

m = folium.Map(location=route_points[0], zoom_start=11, tiles="CartoDB positron")

if map_type == "Route Map":
    folium.PolyLine(route_points, color="blue", weight=6).add_to(m)

    folium.TileLayer(
        tiles=f"https://api.tomtom.com/traffic/map/4/tile/flow/relative/{{z}}/{{x}}/{{y}}.png?key={TOMTOM_KEY}",
        overlay=True,
        opacity=0.7
    ).add_to(m)

else:
    for i in range(len(route_points)-1):
        p1, p2 = route_points[i], route_points[i+1]

        progress = i/max(1,len(route_points)-1)

        seg_torque = torque_pred*(0.8 + 0.4*np.sin(progress*3.14))
        seg_power = power_pred*(0.8 + 0.4*np.cos(progress*3.14))

        if map_type == "Torque Zoning":
            color = "#FD8D3C" if seg_torque<300 else "#D94801"
        else:
            color = "#41AB5D" if seg_power<20 else "#005A32"

        folium.PolyLine([p1,p2], color=color, weight=5).add_to(m)

folium.Marker(route_points[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(route_points[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(m)

folium.LayerControl().add_to(m)

st_folium(m, width=1100, height=600)

# ==========================================================
# SUMMARY UI
# ==========================================================

st.sidebar.markdown("---")
st.sidebar.metric("Vehicle", vehicle)
st.sidebar.metric("Traffic", traffic_label)
st.sidebar.metric("Distance", f"{distance:.1f} km")
st.sidebar.metric("Speed", f"{speed:.0f} km/h")
st.sidebar.metric("Score", f"{score:.1f}")
