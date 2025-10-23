import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# Configuration for debugging
# -------------------------
CHUNK_FOLDER = "chunks"
PROCESSED_FOLDER = "processed_chunks"
TARGET_COLUMNS = ["attack_cat", "label"]
TOP_FEATURES = 30  # number of top features to select per chunk

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

label_encoders = {}
feature_cols = None
max_vals = None

# -------------------------
# Pick the first CSV chunk automatically
# -------------------------
chunk_files = sorted([f for f in os.listdir(CHUNK_FOLDER) if f.endswith(".csv")])
if not chunk_files:
    raise FileNotFoundError(f"No CSV files found in folder '{CHUNK_FOLDER}'")
DEBUG_CHUNK_FILE = os.path.join(CHUNK_FOLDER, chunk_files[0])
print(f"🚀 Debug: loading chunk {DEBUG_CHUNK_FILE}")

# -------------------------
# Helper function
# -------------------------
def safe_normalize(X):
    global max_vals
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if max_vals is None:
        max_vals = np.max(np.abs(X), axis=0)
    X = np.divide(X, max_vals, out=np.zeros_like(X), where=max_vals != 0)
    return X

# -------------------------
# Load and process the debug chunk
# -------------------------
df = pd.read_csv(DEBUG_CHUNK_FILE)
df.columns = df.columns.str.strip().str.lower()
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# Detect target column
target_col = next((col for col in TARGET_COLUMNS if col in df.columns), None)
if target_col is None:
    raise ValueError("No target column found in debug chunk!")

# Drop rows with NaN in target
df = df.dropna(subset=[target_col])
y = df[target_col]
X = df.drop(columns=[target_col])

print("X type:", type(X), "y type:", type(y))
print("X shape:", X.shape, "y shape:", y.shape)

# Encode categorical columns
for col in X.select_dtypes(include='object').columns:
    if col not in label_encoders:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col].astype(str))
    else:
        X[col] = X[col].map(lambda s: label_encoders[col].transform([s])[0] if s in label_encoders[col].classes_ else -1)

feature_cols = list(X.columns)
print("Feature columns:", feature_cols)

# Replace NaN and normalize
X = X.fillna(0)
X = safe_normalize(X)

# Convert X back to DataFrame before saving
X_df = pd.DataFrame(X, columns=feature_cols)
print("X_df head:\n", X_df.head())

# -------------------------
# Compute feature importance for debug
# -------------------------
try:
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_df, y)
    importances = rf.feature_importances_
    top_features_idx = np.argsort(importances)[::-1][:min(TOP_FEATURES, X_df.shape[1])]
    top_features_names = [feature_cols[j] for j in top_features_idx]
    print(f"Top {len(top_features_names)} features:", top_features_names)
except Exception as e:
    print(f"⚠ Feature importance computation failed: {e}")
    top_features_idx = list(range(X_df.shape[1]))
    top_features_names = feature_cols

# -------------------------
# Save processed debug chunk and top features
# -------------------------
processed_file = os.path.join(PROCESSED_FOLDER, "debug_chunk_processed.csv")
pd.concat([X_df, y.reset_index(drop=True)], axis=1).to_csv(processed_file, index=False)
importance_file = os.path.join(PROCESSED_FOLDER, "debug_chunk_features.pkl")
pickle.dump({"top_features_idx": top_features_idx, "top_features_names": top_features_names}, open(importance_file, "wb"))

print(f"✅ Debug chunk saved: {processed_file}")
print(f"✅ Top features saved: {importance_file}")
