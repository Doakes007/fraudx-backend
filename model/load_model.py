import joblib
import os

# Get the path to the current directory (backend/model/)
MODEL_DIR = os.path.dirname(__file__)

# ✅ Load the trained RandomForest model
model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))

# ✅ Load expected feature columns (for aligning during inference)
try:
    expected_features = joblib.load(os.path.join(MODEL_DIR, "model_columns.pkl"))
except Exception as e:
    print(f"Warning: Could not load model_columns.pkl: {e}")
    expected_features = None

# ✅ Load distance cutoff for far_from_home_flag
try:
    dist_cutoff = joblib.load(os.path.join(MODEL_DIR, "dist_cutoff.pkl"))
except Exception as e:
    print(f"Warning: Could not load dist_cutoff.pkl: {e}")
    dist_cutoff = None

# ✅ Load home_lookup dictionary (optional)
try:
    home_lookup = joblib.load(os.path.join(MODEL_DIR, "home_lookup.pkl"))
except Exception as e:
    print(f"Warning: Could not load home_lookup.pkl: {e}")
    home_lookup = None

