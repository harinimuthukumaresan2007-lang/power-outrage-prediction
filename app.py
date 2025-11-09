# =====================================================
# ⚡ Tamil Nadu Power Outage Prediction & Substation Dashboard
# Author: ChatGPT (GPT-5)
# =====================================================

import pandas as pd
import numpy as np
import joblib
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import concurrent.futures

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Power Outage Prediction Dashboard ⚡", layout="wide")

MODEL_PATH = "outage_model.pkl"
API_KEY = "YOUR_OPENWEATHER_API_KEY"  # ← replace with your key

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except:
        return None

def fetch_real_weather(lat, lon, api_key):
    """Fetch current weather data from OpenWeather API"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        res = requests.get(url, timeout=10)
        data = res.json()
        if res.status_code != 200:
            return None
        return {
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
            "weather": data["weather"][0]["main"]
        }
    except:
        return None

def predict_outage_from_weather(model, weather):
    """Predict outage risk using model"""
    if model is None or weather is None:
        return {"label": "Unknown", "probability": 0}
    wmap = {"Clear": 0, "Clouds": 1, "Rain": 2, "Thunderstorm": 3, "Drizzle": 4, "Snow": 5, "Mist": 6}
    weather_code = wmap.get(weather["weather"], 1)
    X = np.array([[weather["temp"], weather["humidity"], weather["pressure"], weather["wind_speed"], weather_code]])
    prob = model.predict_proba(X)[0][1]
    label = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
    return {"label": label, "probability": round(float(prob), 2)}

def fetch_nearby_substations(lat, lon, radius_km=50):
    """Mock list of nearby substations"""
    np.random.seed(42)
    substations = []
    for i in range(10):
        dlat = np.random.uniform(-0.3, 0.3)
        dlon = np.random.uniform(-0.3, 0.3)
        substations.append({
            "name": f"Substation-{i+1}",
            "lat": lat + dlat,
            "lon": lon + dlon
        })
    return substations

# =====================================================
# STREAMLIT UI
# =====================================================
st.title("⚡ Tamil Nadu Power Outage Prediction & Substation Dashboard")
st.markdown("This app predicts *power outage risk* based on live weather and shows connected substations with their area names.")

col1, col2 = st.columns(2)
with col1:
    city_name = st.text_input("Enter your City / Area:", "Chennai")
with col2:
    radius_km = st.slider("Nearby Substation Radius (km)", 10, 100, 50)

model = load_model()

# =====================================================
# GEOCODING USER INPUT
# =====================================================
if st.button("🔍 Predict Outage Risk & Show Substations"):
    geolocator = Nominatim(user_agent="outage_locator")
    location = geolocator.geocode(city_name)
    if not location:
        st.error("❌ Location not found. Try another city name.")
    else:
        lat, lon = location.latitude, location.longitude
        st.success(f"📍 Location found: {city_name} ({lat:.2f}, {lon:.2f})")

        # Fetch weather for main location
        weather = fetch_real_weather(lat, lon, API_KEY)
        if not weather:
            st.error("Could not fetch weather data. Check API key or connection.")
        else:
            pred = predict_outage_from_weather(model, weather)
            st.metric(label="Outage Risk Level", value=pred["label"], delta=f"{pred['probability']*100:.0f}%")

            # =====================================================
            # SUBSTATION ANALYSIS (OPTIMIZED + AREA NAMES)
            # =====================================================
            from functools import lru_cache
            geolocator = Nominatim(user_agent="outage_locator")

            @st.cache_data(show_spinner=False)
            def cached_weather(lat, lon, api_key):
                return fetch_real_weather(lat, lon, api_key)

            @st.cache_data(show_spinner=False)
            def cached_location_name(lat, lon):
                try:
                    loc = geolocator.reverse((lat, lon), exactly_one=True, language="en")
                    if loc:
                        return loc.address.split(",")[0]
                except:
                    return "Unknown Area"
                return "Unknown Area"

            subs = fetch_nearby_substations(lat, lon, radius_km)
            if not subs:
                st.info("No substations found nearby.")
            else:
                subs = subs[:6]
                st.info(f"Analyzing {len(subs)} nearest substations...")

                def process_sub(s):
                    sw = cached_weather(s["lat"], s["lon"], API_KEY)
                    if not sw:
                        return None
                    sres = predict_outage_from_weather(model, sw)
                    area_name = cached_location_name(s["lat"], s["lon"])
                    return {
                        "Substation": s["name"],
                        "Area": area_name,
                        "Latitude": s["lat"],
                        "Longitude": s["lon"],
                        "Risk": sres["label"],
                        "Prob": round(sres["probability"], 2)
                    }

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = list(executor.map(process_sub, subs))

                sub_data = [r for r in results if r]

                if not sub_data:
                    st.warning("Could not predict for substations.")
                else:
                    df = pd.DataFrame(sub_data)
                    st.dataframe(df)

                    m2 = folium.Map(location=[lat, lon], zoom_start=9, tiles="CartoDB Positron")

                    folium.Marker(
                        [lat, lon],
                        popup=f"Selected Location: {city_name}",
                        icon=folium.Icon(color="blue", icon="home")
                    ).add_to(m2)

                    for s in sub_data:
                        c = {"Low": "green", "Medium": "orange", "High": "red"}[s["Risk"]]

                        folium.Marker(
                            [s["Latitude"], s["Longitude"]],
                            popup=f"<b>{s['Substation']}</b><br>{s['Area']}<br>Risk: {s['Risk']} ({s['Prob']})",
                            icon=folium.Icon(color=c)
                        ).add_to(m2)

                        folium.PolyLine(
                            locations=[[lat, lon], [s["Latitude"], s["Longitude"]]],
                            color=c, weight=2.5, opacity=0.8,
                            tooltip=f"Connection to {s['Area']}"
                        ).add_to(m2)

                    st.markdown("### 📍 Substation Network Map (With Connected Area Names)")
                    st_folium(m2, width=900, height=600)
