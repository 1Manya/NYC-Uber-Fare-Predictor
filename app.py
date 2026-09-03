import json
import os
import sys
from datetime import datetime, time as dtime

import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from features import build_features

try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
    SHAP_IMPORT_ERROR = None
except Exception as e:
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = f"{type(e).__name__}: {e}"

st.set_page_config(page_title="NYC Uber Fare Predictor", page_icon="🚕", layout="wide")

MODEL_PATH = "models/best_model.joblib"
METRICS_PATH = "models/metrics.json"
DATA_PATH = "data/uber.csv"

NYC_LAT_RANGE = (40.5, 41.5)
NYC_LON_RANGE = (-74.5, -72.5)

NYC_LANDMARKS = {
    "JFK Airport": (40.6413, -73.7781),
    "LaGuardia Airport": (40.7769, -73.8740),
    "Newark Airport": (40.6895, -74.1745),
    "Times Square": (40.7580, -73.9855),
    "Central Park": (40.7829, -73.9654),
    "Wall Street": (40.7074, -74.0113),
    "Brooklyn Bridge": (40.7061, -73.9969),
    "Grand Central Terminal": (40.7527, -73.9772),
    "Empire State Building": (40.7484, -73.9857),
    "Custom (enter coordinates)": None,
}

RUSH_HOURS = {7, 8, 9, 17, 18, 19}
NIGHT_HOURS = {23, 0, 1, 2, 3, 4}


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    metrics = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return model, metrics


@st.cache_data
def get_shap_background(n=100):
    """A small sample of real feature rows for SHAP to use as its baseline
    ('what does an average trip look like'). Returns None if the raw CSV
    isn't available at runtime (e.g. deployed without the data folder)."""
    if not os.path.exists(DATA_PATH):
        return None
    raw = pd.read_csv(DATA_PATH, nrows=5000)
    raw = raw.dropna()
    raw = raw[(raw["fare_amount"] > 0) & (raw["fare_amount"] < 300)]
    raw = raw[raw["passenger_count"].between(1, 6)]
    raw = raw[raw["pickup_latitude"].between(*NYC_LAT_RANGE)]
    raw = raw[raw["dropoff_latitude"].between(*NYC_LAT_RANGE)]
    raw = raw[raw["pickup_longitude"].between(*NYC_LON_RANGE)]
    raw = raw[raw["dropoff_longitude"].between(*NYC_LON_RANGE)]
    if len(raw) == 0:
        return None
    sample = raw.sample(min(n, len(raw)), random_state=42)
    return build_features(sample)


@st.cache_resource
def get_shap_explainer(_model, _background):
    # Model-agnostic: explains model.predict() directly (in dollars), so it
    # works whether the saved model is Ridge, a single tree model, or the
    # weighted ensemble — no need to unwrap LogTargetRegressor internals.
    return shap.Explainer(_model.predict, _background)


def location_picker(label, key_prefix, default_choice_index, default_lat, default_lon):
    choice = st.selectbox(label, list(NYC_LANDMARKS.keys()), index=default_choice_index, key=f"{key_prefix}_choice")
    if NYC_LANDMARKS[choice] is None:
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Latitude", value=default_lat, format="%.5f", key=f"{key_prefix}_lat")
        with c2:
            lon = st.number_input("Longitude", value=default_lon, format="%.5f", key=f"{key_prefix}_lon")
        return lat, lon
    return NYC_LANDMARKS[choice]


def traffic_level(hour):
    """Rough, non-live proxy for traffic conditions based on time of day only
    — NOT real traffic data. Used purely to color the route for intuition."""
    if hour in RUSH_HOURS:
        return "Likely heavy (rush hour)", [230, 57, 70]
    if hour in NIGHT_HOURS:
        return "Likely light (night)", [42, 157, 143]
    return "Likely moderate", [244, 162, 97]


def render_trip_map(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, hour):
    label, color = traffic_level(hour)

    points = pd.DataFrame([
        {"lat": pickup_lat, "lon": pickup_lon, "label": "Pickup", "color": [0, 158, 255]},
        {"lat": dropoff_lat, "lon": dropoff_lon, "label": "Dropoff", "color": [255, 64, 64]},
    ])
    line_df = pd.DataFrame([{
        "start": [pickup_lon, pickup_lat],
        "end": [dropoff_lon, dropoff_lat],
    }])

    view_state = pdk.ViewState(
        latitude=(pickup_lat + dropoff_lat) / 2,
        longitude=(pickup_lon + dropoff_lon) / 2,
        zoom=11.5,
        pitch=25,
    )
    line_layer = pdk.Layer(
        "LineLayer",
        data=line_df,
        get_source_position="start",
        get_target_position="end",
        get_width=5,
        get_color=color,
    )
    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=140,
        pickable=True,
    )
    deck = pdk.Deck(
        map_style=None,
        layers=[line_layer, point_layer],
        initial_view_state=view_state,
        tooltip={"text": "{label}"},
    )
    st.pydeck_chart(deck, use_container_width=True)
    st.caption(f"Route color = **{label}**, estimated from time of day only "
               f"(not live traffic data).")


# ---- Prime widget defaults from a shared URL, if present ----
qp = st.query_params


def _qp_float(key, default):
    try:
        return float(qp[key])
    except (KeyError, TypeError, ValueError):
        return default


if "primed_from_url" not in st.session_state:
    if all(k in qp for k in ("plat", "plon", "dlat", "dlon")):
        st.session_state["pickup_choice"] = "Custom (enter coordinates)"
        st.session_state["pickup_lat"] = _qp_float("plat", 40.758)
        st.session_state["pickup_lon"] = _qp_float("plon", -73.985)
        st.session_state["dropoff_choice"] = "Custom (enter coordinates)"
        st.session_state["dropoff_lat"] = _qp_float("dlat", 40.641)
        st.session_state["dropoff_lon"] = _qp_float("dlon", -73.778)
    try:
        if "date" in qp:
            st.session_state["ride_date_shared"] = datetime.strptime(qp["date"], "%Y-%m-%d").date()
    except ValueError:
        pass
    try:
        if "time" in qp:
            st.session_state["ride_time_shared"] = datetime.strptime(qp["time"], "%H:%M").time()
    except ValueError:
        pass
    try:
        if "pax" in qp:
            st.session_state["passengers_shared"] = int(qp["pax"])
    except ValueError:
        pass
    st.session_state["primed_from_url"] = True


model, metrics = load_model()

st.title("🚕 NYC Uber Fare Predictor")
st.caption("Predicts fare using pickup/dropoff location, time, and passenger count.")

if model is None:
    st.error("No trained model found. Run `python src/train.py` first, then redeploy.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

tab_predict, tab_performance, tab_about = st.tabs(["🔮 Predict", "📊 Model performance", "ℹ️ About"])

with tab_predict:
    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("Trip details")
        pickup_lat, pickup_lon = location_picker("Pickup location", "pickup", 3, 40.758, -73.985)
        dropoff_lat, dropoff_lon = location_picker("Dropoff location", "dropoff", 0, 40.641, -73.778)

        c1, c2 = st.columns(2)
        with c1:
            ride_date = st.date_input("Date", value=st.session_state.get("ride_date_shared", datetime(2026, 6, 15).date()))
        with c2:
            ride_time = st.time_input("Time", value=st.session_state.get("ride_time_shared", dtime(18, 30)))
        passenger_count = st.slider("Passengers", 1, 6, st.session_state.get("passengers_shared", 1))

        out_of_bounds = not (
            NYC_LAT_RANGE[0] <= pickup_lat <= NYC_LAT_RANGE[1]
            and NYC_LAT_RANGE[0] <= dropoff_lat <= NYC_LAT_RANGE[1]
            and NYC_LON_RANGE[0] <= pickup_lon <= NYC_LON_RANGE[1]
            and NYC_LON_RANGE[0] <= dropoff_lon <= NYC_LON_RANGE[1]
        )
        if out_of_bounds:
            st.warning("⚠️ These coordinates fall outside the NYC region the model was trained on "
                       "— the prediction below may be unreliable.")

        bcol1, bcol2 = st.columns([1, 1])
        with bcol1:
            predict_clicked = st.button("Predict fare", type="primary", use_container_width=True)
        with bcol2:
            share_clicked = st.button("🔗 Share this trip", use_container_width=True)

        if share_clicked:
            st.query_params["plat"] = f"{pickup_lat:.5f}"
            st.query_params["plon"] = f"{pickup_lon:.5f}"
            st.query_params["dlat"] = f"{dropoff_lat:.5f}"
            st.query_params["dlon"] = f"{dropoff_lon:.5f}"
            st.query_params["date"] = ride_date.strftime("%Y-%m-%d")
            st.query_params["time"] = ride_time.strftime("%H:%M")
            st.query_params["pax"] = str(passenger_count)
            st.success("Your browser's address bar now encodes this trip — copy the URL to share it. "
                       "Opening that link will pre-fill these exact inputs.")

    with right:
        st.subheader("Trip map")
        render_trip_map(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, ride_time.hour)

    if predict_clicked:
        pickup_dt = datetime.combine(ride_date, ride_time)
        row = pd.DataFrame([{
            "pickup_datetime": pickup_dt,
            "pickup_latitude": pickup_lat,
            "pickup_longitude": pickup_lon,
            "dropoff_latitude": dropoff_lat,
            "dropoff_longitude": dropoff_lon,
            "passenger_count": passenger_count,
        }])
        X = build_features(row)
        pred = float(model.predict(X)[0])
        distance_km = float(X["distance_km"].iloc[0])
        fare_per_km = pred / distance_km if distance_km > 0 else float("nan")

        mae = None
        if metrics:
            mae = metrics["results"][metrics["best_model"]]["mae"]

        st.divider()
        st.subheader("Prediction")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted fare", f"${pred:.2f}")
        if mae is not None:
            m2.metric("Give or take", f"± ${mae:.2f}")
        m3.metric("Distance", f"{distance_km:.2f} km")
        m4.metric("Fare / km", f"${fare_per_km:.2f}")

        tags = []
        tags.append("✈️ Airport trip" if X["is_airport_trip"].iloc[0] else "🏙️ Non-airport trip")
        tags.append("🚦 Rush hour" if X["is_rush_hour"].iloc[0] else "🟢 Off-peak")
        tags.append("🌙 Night" if X["is_night"].iloc[0] else "☀️ Daytime")
        tags.append("📅 Weekend" if X["is_weekend"].iloc[0] else "📅 Weekday")
        st.caption("  ·  ".join(tags))

        st.session_state.history.insert(0, {
            "Time": pickup_dt.strftime("%Y-%m-%d %H:%M"),
            "Pickup": f"{pickup_lat:.3f}, {pickup_lon:.3f}",
            "Dropoff": f"{dropoff_lat:.3f}, {dropoff_lon:.3f}",
            "Distance (km)": round(distance_km, 2),
            "Passengers": passenger_count,
            "Predicted fare": f"${pred:.2f}",
        })

        with st.expander("🔍 Explain this prediction (SHAP)"):
            if not SHAP_AVAILABLE:
                st.info("SHAP isn't installed. Run `pip install shap` and restart the app to enable this.")
                if SHAP_IMPORT_ERROR:
                    st.code(f"Import error: {SHAP_IMPORT_ERROR}")
            else:
                background = get_shap_background()
                if background is None:
                    st.warning(f"Couldn't find `{DATA_PATH}` to build a SHAP background sample.")
                else:
                    with st.spinner("Computing SHAP values..."):
                        try:
                            explainer = get_shap_explainer(model, background)
                            shap_values = explainer(X, max_evals=500)
                            fig = plt.figure()
                            shap.plots.waterfall(shap_values[0], show=False)
                            st.pyplot(fig, use_container_width=True)
                            st.caption("How much each feature pushed this prediction up (red) or down (blue) "
                                       "from the average predicted fare, in dollars.")
                        except Exception as e:
                            st.warning(f"Could not compute a SHAP explanation: {e}")

    if st.session_state.history:
        st.divider()
        st.subheader("This session's predictions")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

with tab_performance:
    st.subheader("Model comparison")
    if metrics:
        results_df = pd.DataFrame(metrics["results"]).T
        results_df.index.name = "Model"
        best = metrics["best_model"]

        st.info(f"Model currently in use: **{best}**  ·  Test R² = **{metrics['results'][best]['r2']}**  "
                f"·  Test MAE = **${metrics['results'][best]['mae']}**")

        bc1, bc2 = st.columns(2)
        with bc1:
            st.caption("R² (higher is better)")
            st.bar_chart(results_df["r2"])
        with bc2:
            st.caption("MAE in dollars (lower is better)")
            st.bar_chart(results_df["mae"])

        st.caption("Full metrics")
        st.dataframe(results_df, use_container_width=True)
    else:
        st.warning("No metrics.json found — run `python src/train.py` to generate it.")

with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
        This app predicts NYC Uber/taxi fares from pickup/dropoff coordinates,
        trip time, and passenger count.

        **Pipeline:**
        - Haversine distance + airport-proximity flags computed from raw coordinates
        - Cyclical (sin/cos) time-of-day, weekday, and month encodings
        - Models trained on `log1p(fare)` and evaluated after inverting back to dollars,
          with a safety clip on the log-space prediction to prevent outlier blowups
        - Candidates compared: Ridge-regularized Linear Regression, XGBoost, LightGBM,
          and a weighted XGBoost + LightGBM ensemble — the best on test R² is deployed

        **App features:**
        - Interactive map with pickup/dropoff pins and a route colored by an estimated
          traffic proxy (time-of-day based — not live traffic data)
        - SHAP explanation of each individual prediction
        - Shareable trip links via URL query parameters
        - Out-of-training-region warning for coordinates outside NYC

        Built with scikit-learn / XGBoost / LightGBM / Streamlit / SHAP.
        [View source on GitHub](https://github.com/YOUR_USERNAME/uber-fare-prediction)
        """
    )
