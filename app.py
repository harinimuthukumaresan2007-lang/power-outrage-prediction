import pandas as pd
import numpy as np
import joblib
import requests
import streamlit as st
import concurrent.futures
from functools import lru_cache
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

MODEL_PATH = "outage_model.joblib"
HISTORICAL_CSV = "historical_outages.csv"

# ---------- Weather API ----------
def fetch_real_weather(lat, lon, api_key):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        temp = data["main"]["temp"]
        humidity = data["main"].get("humidity", 0)
        wind_speed = data.get("wind", {}).get("speed", 0)
        prec = 0.0
        prec += data.get("rain", {}).get("1h", 0)
        prec += data.get("snow", {}).get("1h", 0)
        return {
            "main": {"temp": temp, "humidity": humidity},
            "wind": {"speed": wind_speed},
            "rain": {"1h": data.get("rain", {}).get("1h", 0)},
            "snow": {"1h": data.get("snow", {}).get("1h", 0)}
        }
    except Exception as e:
        st.error(f"Error fetching weather: {e}")
        return None

# ---------- Model ----------
def build_or_load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        st.info("Training demo model (no pre-trained model found).")
    try:
        df = pd.read_csv(HISTORICAL_CSV)
        X = df[["temperature","humidity","wind_speed","precipitation"]].fillna(0)
        y = df["outage"]
    except Exception:
        rng = np.random.RandomState(42)
        n = 2000
        temperature = rng.normal(25, 8, size=n)
        humidity = rng.uniform(20, 100, size=n)
        wind_speed = rng.exponential(2, size=n)
        precipitation = rng.exponential(0.5, size=n)
        outage_prob = (wind_speed/(1+wind_speed))*0.6 + (precipitation/(1+precipitation))*0.5 + (humidity/200)
        outage = (rng.rand(n) < outage_prob).astype(int)
        X = pd.DataFrame({
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "precipitation": precipitation
        })
        y = outage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, MODEL_PATH)
    st.success(f"Trained demo model — test accuracy: {acc:.2f}")
    return model

# ---------- Prediction ----------
def predict_outage_from_weather(model, weather_json):
    try:
        temp = weather_json["main"]["temp"]
        humidity = weather_json["main"].get("humidity", 0)
        wind_speed = weather_json.get("wind", {}).get("speed", 0)
        prec = 0.0
        prec += weather_json.get("rain", {}).get("1h", 0)
        prec += weather_json.get("snow", {}).get("1h", 0)
        features = np.array([[temp, humidity, wind_speed, prec]])
        prob = model.predict_proba(features)[0][1]
        label = "Low" if prob < 0.25 else "Medium" if prob < 0.6 else "High"
        return {"temperature": temp, "humidity": humidity,
                "wind_speed": wind_speed, "precipitation": prec,
                "probability": prob, "label": label}
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ---------- Substation Data ----------
def fetch_nearby_substations(lat, lon, radius_km=25):
    try:
        query = f"""
        [out:json];
        node["power"="substation"](around:{int(radius_km * 1000)},{lat},{lon});
        out center;
        """
        url = "https://overpass-api.de/api/interpreter"
        r = requests.get(url, params={"data": query}, timeout=25)
        r.raise_for_status()
        data = r.json()
        subs = []
        for e in data.get("elements", []):
            name = e.get("tags", {}).get("name", "Unnamed Substation")
            subs.append({"name": name, "lat": e["lat"], "lon": e["lon"]})
        return subs
    except Exception as e:
        st.warning(f"Failed to fetch substations: {e}")
        return []

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Power Outage Prediction", layout="wide")
st.title("⚡ Power Outage Prediction & Substation Risk Monitor")

api_key = st.sidebar.text_input("🔑 OpenWeatherMap API Key", type="password")
address_input = st.sidebar.text_input("📍 Enter Address / Landmark (e.g., Dindigul PSNACET)")
city_input = st.sidebar.text_input("🏙 City name (optional)")
lat_input = st.sidebar.number_input("Latitude", value=12.97, format="%.6f")
lon_input = st.sidebar.number_input("Longitude", value=77.59, format="%.6f")
radius_km = st.sidebar.slider("📡 Radius for Nearby Substations (km)", min_value=5, max_value=50, value=25, step=1)
predict_btn = st.sidebar.button("🔍 Predict")

if "lat" not in st.session_state: st.session_state.lat = lat_input
if "lon" not in st.session_state: st.session_state.lon = lon_input
if "auto_predict" not in st.session_state: st.session_state.auto_predict = False

if address_input:
    try:
        loc = Nominatim(user_agent="power_outage_app").geocode(address_input)
        if loc:
            st.session_state.lat, st.session_state.lon = loc.latitude, loc.longitude
    except Exception:
        st.warning("❌ Could not find address location.")
elif city_input:
    try:
        loc = Nominatim(user_agent="power_outage_app").geocode(city_input)
        if loc:
            st.session_state.lat, st.session_state.lon = loc.latitude, loc.longitude
    except Exception:
        st.warning("❌ Could not find city location.")

st.markdown("### 🗺 Click anywhere on the map to auto-predict outage risk")

def build_map(lat_c, lon_c, zoom=8):
    m = folium.Map(location=[lat_c, lon_c], zoom_start=zoom, tiles="CartoDB Positron")
    folium.Marker([lat_c, lon_c], popup=f"Selected: {lat_c:.4f}, {lon_c:.4f}",
                  icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)
    return m

m = build_map(st.session_state.lat, st.session_state.lon)
map_response = st_folium(m, width=900, height=500, returned_objects=["last_clicked", "last_object_clicked"])
clicked = None
if map_response:
    if map_response.get("last_clicked"):
        clicked = map_response["last_clicked"]
    elif map_response.get("last_object_clicked"):
        clicked = map_response["last_object_clicked"]

if clicked:
    st.session_state.lat = clicked.get("lat", st.session_state.lat)
    st.session_state.lon = clicked.get("lng", st.session_state.lon)
    st.session_state.auto_predict = True

lat, lon = st.session_state.lat, st.session_state.lon
st.sidebar.markdown(f"*Selected Latitude:* {lat:.6f}")
st.sidebar.markdown(f"*Selected Longitude:* {lon:.6f}")
st.sidebar.markdown(f"*Substation radius:* {radius_km} km")

# ---------- Prediction + Optimized Substation Section ----------
if predict_btn or st.session_state.auto_predict:
    st.session_state.auto_predict = False
    if not api_key:
        st.error("⚠ Please enter a valid OpenWeatherMap API key.")
    else:
        model = build_or_load_model()
        weather = fetch_real_weather(lat, lon, api_key)
        if weather:
            res = predict_outage_from_weather(model, weather)
            if res is None:
                st.error("Prediction failed.")
            else:
                color = {"Low": "green", "Medium": "orange", "High": "red"}[res["label"]]
                st.markdown(f"### ⚠ Outage Risk: <span style='color:{color}'>{res['label']}</span> "
                            f"(prob: {res['probability']:.2f})", unsafe_allow_html=True)
                st.write(f"🌡 Temp: {res['temperature']}°C | 💧 Humidity: {res['humidity']}% | "
                         f"🌬 Wind: {res['wind_speed']} m/s | 🌧 Prec: {res['precipitation']} mm")

                st.markdown("### 🏭 Nearby Substations")

                # Cache weather results
                @st.cache_data(show_spinner=False)
                def cached_weather(lat, lon, api_key):
                    return fetch_real_weather(lat, lon, api_key)

                subs = fetch_nearby_substations(lat, lon, radius_km)

                if not subs:
                    st.info("No substations found nearby.")
                else:
                    subs = subs[:6]
                    st.info(f"Analyzing {len(subs)} nearest substations...")

                    def process_sub(s):
                        sw = cached_weather(s["lat"], s["lon"], api_key)
                        if not sw:
                            return None
                        sres = predict_outage_from_weather(model, sw)
                        return {
                            "Substation": s["name"],
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
                        folium.Marker([lat, lon],
                                      popup="Selected Location",
                                      icon=folium.Icon(color="blue", icon="home")).add_to(m2)

                        for s in sub_data:
                            c = {"Low": "green", "Medium": "orange", "High": "red"}[s["Risk"]]
                            folium.Marker([s["Latitude"], s["Longitude"]],
                                          popup=f"{s['Substation']}<br>Risk: {s['Risk']} ({s['Prob']})",
                                          icon=folium.Icon(color=c)).add_to(m2)
                            folium.PolyLine([[lat, lon], [s["Latitude"], s["Longitude"]]],
                                            color=c, weight=2.5, opacity=0.8).add_to(m2)

                        st.markdown("### 📍 Substation Network Map (Optimized)")
                        st_folium(m2, width=900, height=600)
