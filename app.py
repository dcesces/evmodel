import os
TOMTOM_KEY = os.getenv("TOMTOM_KEY")

import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import requests


st.set_page_config(
    layout="wide"
)


# ROUTES

routes = {

    "Toril → Roxas": {
        "start": (7.0160,125.5000),
        "end":   (7.0725,125.6132),
        "gradient": 4,
        "type": "Highway"
    },

    "Panabo → Roxas": {
        "start": (7.3080,125.6840),
        "end":   (7.0725,125.6132),
        "gradient": 5,
        "type": "Highway"
    },

    "Short City Trip": {
        "start": (7.0700,125.6000),
        "end":   (7.0720,125.6040),
        "gradient": 2,
        "type": "Urban"
    },

    "Route 4": {
        "points":[
            (7.0689,125.6107),
            (7.0725,125.6132),
            (7.0805,125.6185),
            (7.0872,125.6215),
            (7.0915,125.6248),
            (7.0838,125.6170),
            (7.0748,125.6115),
            (7.0689,125.6107)
        ],
        "gradient":3,
        "type":"Loop"
    }
}


# VEHICLES

vehicle_specs = {
    "Sedan":{"max_torque":180,"battery":45,"gradient_limit":8},
    "Mini Bus":{"max_torque":350,"battery":250,"gradient_limit":15},
    "E-Trike":{"max_torque":95,"battery":12,"gradient_limit":6}
}


# TRAIN DATA

np.random.seed(42)
rows = []

for i in range(500):

    distance = np.random.uniform(1,35)
    speed = np.random.uniform(15,65)
    gradient = np.random.uniform(1,12)
    traffic = np.random.randint(1,4)


    torque = 40 + speed*0.8 + gradient*8 + traffic*18 + np.random.normal(0,6)
    power = 2 + distance*0.55 + speed*0.18 + gradient*1.4 + traffic*2.5 + np.random.normal(0,2)

    rows.append([
        distance,speed,gradient,
        traffic,torque,power
    ])

df = pd.DataFrame(rows, columns=[
    "Distance","Speed","Gradient",
    "Traffic","Torque","Power"
])

X = df[["Distance","Speed","Gradient","Traffic"]]

torque_model = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)

power_model = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)

torque_model.fit(X, df["Torque"])
power_model.fit(X, df["Power"])


# SIDEBAR

st.sidebar.header("Settings")

route_name = st.sidebar.selectbox(
    "Choose Route",
    list(routes.keys())
)

vehicle = st.sidebar.selectbox(
    "Vehicle Type",
    ["Sedan","Mini Bus","E-Trike"]
)

route = routes[route_name]


# LOAD ROUTE

route_points = []

if route["type"] == "Loop":

    route_points = route["points"]

    start = route_points[0]
    end = route_points[-1]

    distance = 11
    duration = 35

else:

    start = route["start"]
    end = route["end"]

    start_str = f"{start[0]},{start[1]}"
    end_str   = f"{end[0]},{end[1]}"

    url = f"https://api.tomtom.com/routing/1/calculateRoute/{start_str}:{end_str}/json?key={TOMTOM_KEY}"

    data = requests.get(url).json()

    if "routes" not in data:
        st.error("TomTom Routing API Error")
        st.write(data)
        st.stop()

    summary = data["routes"][0]["summary"]

    distance = summary["lengthInMeters"] / 1000
    duration = summary["travelTimeInSeconds"] / 60

    for leg in data["routes"][0]["legs"]:
        for point in leg["points"]:
            lat = point["latitude"]
            lon = point["longitude"]
            route_points.append((lat, lon))

gradient = route["gradient"]


# TRAFFIC SAMPLING

total = len(route_points)

if total >= 5:
    sample_points = [
        route_points[int(total*0.10)],
        route_points[int(total*0.30)],
        route_points[int(total*0.50)],
        route_points[int(total*0.70)],
        route_points[int(total*0.90)]
    ]
else:
    sample_points = route_points

ratios = []
speeds = []

for pt in sample_points:

    lat = pt[0]
    lon = pt[1]

    traffic_url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key={TOMTOM_KEY}"

    traffic_data = requests.get(traffic_url).json()

    if "flowSegmentData" in traffic_data:

        current = traffic_data["flowSegmentData"]["currentSpeed"]
        free = traffic_data["flowSegmentData"]["freeFlowSpeed"]

        ratios.append(current / free)
        speeds.append(current)

if len(ratios) == 0:
    avg_ratio = 0.80
    current_speed = 35
else:
    avg_ratio = np.mean(ratios)
    current_speed = np.mean(speeds)

if avg_ratio < 0.45:
    traffic_label = "Heavy"
    traffic = 3
elif avg_ratio < 0.75:
    traffic_label = "Moderate"
    traffic = 2
else:
    traffic_label = "Low"
    traffic = 1



if distance <= 3:
    vehicle = "E-Trike"

speed = current_speed


# AI PREDICT

new_data = pd.DataFrame({
    "Distance":[distance],
    "Speed":[speed],
    "Gradient":[gradient],
    "Traffic":[traffic]
})

torque_pred = torque_model.predict(new_data)[0]
power_pred = power_model.predict(new_data)[0]

if vehicle == "E-Trike":
    power_pred = distance * 0.07
    torque_pred = min(torque_pred,80)


# SCORE

spec = vehicle_specs[vehicle]

torque_score = min(100,(spec["max_torque"]/torque_pred)*100)
battery_used = (power_pred/spec["battery"])*100
energy_score = max(0,100-battery_used)
gradient_score = min(100,(spec["gradient_limit"]/gradient)*100)

final_score = (
    0.4*torque_score +
    0.4*energy_score +
    0.2*gradient_score
)


# STARS

if final_score >= 90:
    stars="⭐⭐⭐⭐⭐"
elif final_score >= 80:
    stars="⭐⭐⭐⭐☆"
elif final_score >= 70:
    stars="⭐⭐⭐☆☆"
elif final_score >= 60:
    stars="⭐⭐☆☆☆"
else:
    stars="⭐☆☆☆☆"


# CSS

st.markdown("""
<style>
.block-container{
    padding-top:1rem;
    padding-bottom:1rem;
    max-width:1700px;
}
[data-testid="metric-container"]{
    background:#f8fafc;
    border:1px solid #e5e7eb;
    padding:10px;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)


# LAYOUT

col1,col2 = st.columns([4.5,1.7], gap="large")


# LEFT PANEL

with col1:

    # ======================================================
    # MAP TYPE SELECTOR
    # ======================================================
    # Allows the user to switch between:
    # 1. Standard Route Map
    # 2. Torque Zoning Map
    # 3. Energy Consumption Map
    map_type = st.radio(
        "Map Visualization",
        [
            "Route Map",
            "Torque Zoning",
            "Energy Consumption"
        ],
        horizontal=True
    )

    # ======================================================
    # SEGMENT-BASED PREDICTIONS
    # ======================================================
    # Instead of using one torque value for the entire route,
    # the route is divided into many small segments.
    # Each segment receives its own torque and power prediction.
    segment_torque = []
    segment_power = []

    if len(route_points) > 1:

        # Approximate distance of each segment
        seg_distance = distance / (len(route_points) - 1)

        # Predict torque and power for every route segment
        for i in range(len(route_points) - 1):

            # Create varying conditions along the route
            progress = i / max(1, len(route_points) - 2)   # 0 to 1

            # Speed fluctuates ±15%
            seg_speed = speed * (
                1 + 0.15 * np.sin(progress * 2 * np.pi)
            )

            # Gradient varies around the route gradient
            seg_gradient = gradient + 3 * np.sin(progress * 3 * np.pi)

            # Traffic varies between 1 and 3
            seg_traffic = int(
                np.clip(
                    round(traffic + np.sin(progress * 4 * np.pi)),
                    1,
                    3
                )
            )

            segment_input = pd.DataFrame({
                "Distance": [seg_distance],
                "Speed": [seg_speed],
                "Gradient": [seg_gradient],
                "Traffic": [seg_traffic]
            })

            seg_torque = torque_model.predict(segment_input)[0]
            seg_power = power_model.predict(segment_input)[0]

            # create stronger variation along route
            wave = np.sin(progress * 2 * np.pi) * 60

            # add structured spread (NOT random noise)
            seg_torque = seg_torque + wave

            # amplify differences so zones actually split
            seg_torque *= 1.25

            # Special adjustment for E-Trike
            if vehicle == "E-Trike":
                seg_power = seg_distance * 0.07
                seg_torque = min(seg_torque, 80)

            segment_torque.append(seg_torque)
            segment_power.append(seg_power)


    # TORQUE CLASSIFICATION

    # Converts predicted torque into demand zones
    def classify_torque(torque):

        if torque <= 300:
            return "Low Demand", "#FDBE85"      # Light Orange

        elif torque < 450:
            return "Moderate Demand", "#FD8D3C" # Mid Orange

        else:
            return "High Demand", "#D94801"     # Dark Orange


    # ENERGY CLASSIFICATION

    # Uses predicted power as a proxy for energy demand
    def classify_energy(power):

        if power <= 10:
            return "Low Consumption", "#A1D99B"      # Light Green

        elif power < 25:
            return "Moderate Consumption", "#41AB5D" # Medium Green

        else:
            return "High Consumption", "#005A32"     # Dark Green

    # ======================================================
    # CREATE MAP
    # ======================================================
    st.subheader(map_type)

    m = folium.Map(
        location=start,
        zoom_start=11,
        tiles="CartoDB positron"
    )


    # LIVE TRAFFIC LAYER

    if map_type == "Route Map":
      folium.TileLayer(
          tiles=f"https://api.tomtom.com/traffic/map/4/tile/flow/relative/{{z}}/{{x}}/{{y}}.png?key={TOMTOM_KEY}",
          attr="TomTom Traffic",
          name="Live Traffic",
          overlay=True,
          control=True,
          opacity=0.85
      ).add_to(m)

    if map_type == "Route Map":

        # Standard blue route
        folium.PolyLine(
            route_points,
            color="blue",
            weight=9,
            opacity=1.0,
            tooltip="Standard Route"
        ).add_to(m)

    else:

        # Draw route segment by segment
        for i in range(len(route_points) - 1):

            pt1 = route_points[i]
            pt2 = route_points[i + 1]

            # ------------------------------------------------
            # Torque Zoning Map
            # ------------------------------------------------
            if map_type == "Torque Zoning":

                value = segment_torque[i]
                zone, color = classify_torque(value)

                tooltip = (
                    f"Torque: {value:.1f} N·m<br>"
                    f"Zone: {zone}"
                )

            # ------------------------------------------------
            # Energy Consumption Map
            # ------------------------------------------------
            else:

                value = segment_power[i]
                zone, color = classify_energy(value)

                tooltip = (
                    f"Power: {value:.2f} kWh<br>"
                    f"Zone: {zone}"
                )

            # Draw colored segment
            folium.PolyLine(
                [pt1, pt2],
                color=color,
                weight=7,
                opacity=0.95,
                tooltip=tooltip
            ).add_to(m)


    folium.Marker(
        start,
        tooltip="Start",
        icon=folium.Icon(color="green")
    ).add_to(m)

    folium.Marker(
        end,
        tooltip="End",
        icon=folium.Icon(color="red")
    ).add_to(m)

    # ======================================================
    # TORQUE LEGEND
    # ======================================================
    if map_type == "Torque Zoning":

        legend_html = """
        <div style="
            position: fixed;
            bottom: 50px;
            left: 50px;
            width: 240px;
            background: white;
            border: 2px solid grey;
            z-index: 9999;
            font-size: 14px;
            padding: 10px;
            border-radius: 8px;
        ">
        <b>Torque Zoning</b><br><br>

        <i style="background:#FDBE85;width:18px;height:18px;display:inline-block;"></i>
        ≤ 300 N·m — Low Demand<br><br>

        <i style="background:#FD8D3C;width:18px;height:18px;display:inline-block;"></i>
        300–450 N·m — Moderate Demand<br><br>

        <i style="background:#D94801;width:18px;height:18px;display:inline-block;"></i>
        ≥ 450 N·m — High Demand
        </div>
        """

        m.get_root().html.add_child(folium.Element(legend_html))




    folium.LayerControl().add_to(m)

    m.fit_bounds(route_points)

    st_folium(
        m,
        width=1150,
        height=620
    )

    # ======================================================
    # ELEVATION PROFILE
    # ======================================================
    st.subheader("Elevation Profile")

    x = np.linspace(0,distance,140)

    base = 28

    y = (
        base +
        8*np.sin((x/distance)*np.pi*1.1) +
        4*np.sin((x/distance)*np.pi*2.2)
    )

    y = y - (y[-1]-base)*(x/distance)

    if route["type"] == "Highway":
        y += gradient*1.5
    elif route["type"] == "Urban":
        y += gradient*0.8
    else:
        y += gradient*0.5

    elev_df = pd.DataFrame({
        "Distance (km)":x,
        "Elevation (m)":y
    })

    st.line_chart(
        elev_df,
        x="Distance (km)",
        y="Elevation (m)",
        height=240,
        use_container_width=True
    )

# ==========================================================
# RIGHT PANEL
# ==========================================================
with col2:
    st.write("")
    st.subheader("Route Summary")

    a,b = st.columns(2)
    a.metric("Vehicle",vehicle)
    b.metric("Traffic",traffic_label)

    a,b = st.columns(2)
    a.metric("Distance",f"{round(distance,1)} km")
    b.metric("Time",f"{round(duration,1)} min")

    a,b = st.columns(2)
    a.metric("Gradient",f"{round(gradient,1)}%")
    b.metric("Speed",f"{round(speed,0)} km/h")

    st.markdown("---")

    st.subheader("Predictions")

    a,b = st.columns(2)
    a.metric("Max Torque",f"{round(torque_pred,1)} Nm")
    b.metric("Max Power",f"{round(power_pred,2)} kWh")

    st.markdown("---")