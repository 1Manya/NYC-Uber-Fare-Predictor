"""Feature engineering shared between training and the Streamlit app."""
import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# Known NYC airport coordinates — used to flag flat-fare airport trips,
# where fare doesn't scale with distance the way normal trips do.
AIRPORTS = {
    "jfk": (40.6413, -73.7781),
    "lga": (40.7769, -73.8740),
    "ewr": (40.6895, -74.1745),
}
AIRPORT_RADIUS_KM = 2.0

# Distances beyond this are almost certainly bad GPS data for an NYC ride,
# not a real trip.
MAX_PLAUSIBLE_DISTANCE_KM = 100

# Fixed reference year (this dataset spans roughly 2009-2015). Using a
# fixed constant instead of `year.min()` computed on the fly means a
# single-row prediction at inference time (e.g. from app.py) gets the
# exact same encoding as training did.
REFERENCE_YEAR = 2009


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Takes raw columns (pickup/dropoff lat-lon, passenger_count, pickup_datetime)
    and returns a feature matrix ready for model input."""
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    df["distance_km"] = haversine_km(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    df["distance_km"] = df["distance_km"].clip(lower=0.01, upper=MAX_PLAUSIBLE_DISTANCE_KM)

    # Fare tends to scale roughly like log(distance) at short range and
    # more linearly at long range. Giving the linear model this feature
    # directly (instead of making it approximate the curve with one raw
    # distance term) is what actually improves its fit quality, not just
    # keeps it from blowing up.
    df["log_distance_km"] = np.log1p(df["distance_km"])

    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday  # 0=Mon
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year
    df["years_since_start"] = (df["year"] - REFERENCE_YEAR).clip(lower=0)

    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_night"] = df["hour"].isin([23, 0, 1, 2, 3, 4]).astype(int)

    # Cyclical encodings replace the raw hour/weekday/month integers for
    # modeling purposes. Two reasons:
    #   1. Raw hour treats 23 and 0 as far apart, when they're one hour
    #      apart in reality. Sin/cos wraps this correctly.
    #   2. Raw hour/weekday were highly redundant with is_rush_hour/
    #      is_night/is_weekend (each just a deterministic function of the
    #      other). That near-collinearity was destabilizing Linear
    #      Regression's coefficients — large, poorly-constrained values
    #      that mostly canceled out on training data but not on unseen
    #      rows, which is what caused wildly-off predictions.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Airport flat-fare flag: 1 if pickup OR dropoff is within AIRPORT_RADIUS_KM
    # of JFK/LGA/EWR. These trips often charge a flat fare regardless of
    # distance, which otherwise looks like noise to the model.
    is_airport = np.zeros(len(df), dtype=int)
    for lat, lon in AIRPORTS.values():
        near_pickup = haversine_km(df["pickup_latitude"], df["pickup_longitude"], lat, lon) <= AIRPORT_RADIUS_KM
        near_dropoff = haversine_km(df["dropoff_latitude"], df["dropoff_longitude"], lat, lon) <= AIRPORT_RADIUS_KM
        is_airport = is_airport | near_pickup.to_numpy() | near_dropoff.to_numpy()
    df["is_airport_trip"] = is_airport.astype(int)

    return df[FEATURE_COLUMNS]


FEATURE_COLUMNS = [
    "distance_km", "log_distance_km", "passenger_count",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "month_sin", "month_cos", "years_since_start",
    "is_weekend", "is_rush_hour", "is_night", "is_airport_trip",
]
