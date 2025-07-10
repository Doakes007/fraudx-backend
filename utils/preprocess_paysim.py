import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import joblib


def preprocess_paysim(df, rng_seed=42, train_mode=True,
                      home_lookup=None, dist_cutoff_km=None, real_mode=False):
    df = df.copy()

    # 🔹 1. Remove negative values
    CRITICAL = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    df = df.loc[(df[CRITICAL] >= 0).all(axis=1)].reset_index(drop=True)

    # 🔹 2. Encode transaction type & risk
    df = pd.get_dummies(df, columns=["type"], drop_first=False)
    expected_types = ["type_CASH_OUT", "type_TRANSFER", "type_PAYMENT", "type_DEBIT", "type_CASH_IN"]
    for col in expected_types:
        df[col] = df.get(col, 0)
    df["high_risk_type"] = df[["type_TRANSFER", "type_CASH_OUT"]].sum(axis=1).astype("int8")

    # 🔹 3. Assign or map device_id
    if not real_mode:
        rng = np.random.default_rng(rng_seed)
        device_map = {user: rng.choice([f"{user}_dev{i}" for i in range(rng.integers(1, 4))])
                      for user in df["nameOrig"].unique()}
        df["device_id"] = df["nameOrig"].map(device_map)
    elif "device_id" not in df.columns:
        df["device_id"] = df["nameOrig"].apply(lambda x: f"{x}_device")

    # 🔹 4. Home location generation
    sender_ids, senders = pd.factorize(df["nameOrig"], sort=False)
    if real_mode:
        df["latitude"] = df["latitude"].astype("float32")
        df["longitude"] = df["longitude"].astype("float32")
        home_lookup = {}
        for user, group in df.groupby("nameOrig"):
            coords = group[["latitude", "longitude"]].dropna().to_numpy()
            if coords.size == 0: continue
            try:
                db = DBSCAN(eps=0.3, min_samples=3).fit(coords)
                labels = db.labels_
                if (labels == -1).all(): continue
                mode_label = pd.Series(labels).value_counts().idxmax()
                home_coords = coords[labels == mode_label].mean(axis=0)
                home_lookup[user] = tuple(home_coords)
            except Exception:
                continue
        base_lat = df["nameOrig"].map(lambda u: home_lookup.get(u, (0, 0))[0]).astype("float32")
        base_lon = df["nameOrig"].map(lambda u: home_lookup.get(u, (0, 0))[1]).astype("float32")
    else:
        rng = np.random.default_rng(rng_seed)
        if train_mode:
            home_lat = rng.uniform(8, 37, senders.size).astype("float32")
            home_lon = rng.uniform(68, 97, senders.size).astype("float32")
            home_lookup = dict(zip(senders, zip(home_lat, home_lon)))
        else:
            home_lat, home_lon = [], []
            for s in df["nameOrig"]:
                lat, lon = home_lookup.get(s, (rng.uniform(8, 37), rng.uniform(68, 97)))
                home_lookup[s] = (lat, lon)
                home_lat.append(lat)
                home_lon.append(lon)
            home_lat = np.array(home_lat, dtype="float32")
            home_lon = np.array(home_lon, dtype="float32")
        base_lat = home_lat[sender_ids]
        base_lon = home_lon[sender_ids]
        lat = rng.normal(base_lat, 0.05)
        lon = rng.normal(base_lon, 0.05)
        if "isFraud" in df.columns:
            fraud_mask = df["isFraud"].astype(bool).to_numpy()
            teleport_mask = fraud_mask & (rng.random(df.shape[0]) < 0.30)
            jump = rng.uniform(6, 9, teleport_mask.sum())
            sign = rng.choice([-1, 1], jump.size)
            lat[teleport_mask] += sign * jump
            lon[teleport_mask] += sign * jump
        df["latitude"] = lat.astype("float32")
        df["longitude"] = lon.astype("float32")

    # 🔹 5. Geo distance
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    df["dist_from_home_km"] = haversine(df["latitude"], df["longitude"], base_lat, base_lon).astype("float32")
    if train_mode and "isFraud" in df.columns:
        dist_cutoff_km = df.loc[df["isFraud"] == 0, "dist_from_home_km"].quantile(0.995)
    df["far_from_home_flag"] = (df["dist_from_home_km"] >= dist_cutoff_km).astype("int8")

    # 🔹 6. Time-based features (safe fallback if step is missing)
    if "step" in df.columns:
        df["hour"] = (df["step"] % 24).astype(int)
        df["is_night"] = df["hour"].apply(lambda x: int(x < 6 or x > 22))
    else:
        df["hour"] = 12
        df["is_night"] = 0

    # 🔹 7. Balance ratio
    df["amt_to_bal_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["amt_to_bal_ratio"] = df["amt_to_bal_ratio"].clip(0, 1)

    # 🔹 8. Sender behavior
    amt = df["amount"].to_numpy("float32")
    cum_amt = np.bincount(sender_ids, weights=amt, minlength=sender_ids.max()+1).cumsum()[sender_ids]
    cum_cnt = np.bincount(sender_ids, minlength=sender_ids.max()+1).cumsum()[sender_ids]
    user_avg = (cum_amt - amt) / np.maximum(cum_cnt - 1, 1)
    user_avg[cum_cnt == 1] = np.nan
    df["user_avg_amount"] = user_avg.astype("float32")
    df["amt_to_user_avg_ratio"] = (amt / (user_avg + 1e-9)).astype("float32")

    # 🔹 9. Destination behavior
    dest_ids, _ = pd.factorize(df["nameDest"], sort=False)
    cum_amt_d = np.bincount(dest_ids, weights=amt, minlength=dest_ids.max()+1).cumsum()[dest_ids]
    cum_cnt_d = np.bincount(dest_ids, minlength=dest_ids.max()+1).cumsum()[dest_ids]
    dest_avg = (cum_amt_d - amt) / np.maximum(cum_cnt_d - 1, 1)
    dest_avg[cum_cnt_d == 1] = np.nan
    df["dest_avg_amount"] = dest_avg.astype("float32")
    df["amount_to_dest_avg_ratio"] = (amt / (dest_avg + 1e-9)).astype("float32")

    # 🔹 10. Device-based metrics
    seen_devices = set()
    df["is_new_device"] = [int(dev not in seen_devices and not seen_devices.add(dev)) for dev in df["device_id"]]
    df["device_txn_count"] = df.groupby("device_id").cumcount()
    df["device_user_count"] = df["device_id"].map(df.groupby("device_id")["nameOrig"].nunique())

    # 🔹 11. Cleanup — drop only if columns exist
    drop_cols = [col for col in ["step", "hour", "nameOrig", "nameDest", "device_id", "orig_balance_leak_flag", "isFlaggedFraud"]
                 if col in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # 🔹 12. Cast binary
    for col in ["high_risk_type", "far_from_home_flag", "is_night", "isFraud"]:
        if col in df.columns:
            df[col] = df[col].astype("int8")

    # 🔹 13. Align with model columns
    if not train_mode:
        try:
            REQUIRED_COLUMNS = joblib.load("model/model_columns.pkl")
            for col in REQUIRED_COLUMNS:
                if col not in df.columns:
                    df[col] = 0
            df = df[REQUIRED_COLUMNS]
        except Exception as e:
            print(f"⚠️ Warning: Could not enforce model columns: {e}")

    return df, home_lookup, dist_cutoff_km

