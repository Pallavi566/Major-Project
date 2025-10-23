import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# Configuration
# -------------------------
CHUNK_FOLDER = "chunks"
PROCESSED_FOLDER = "processed_chunks"
TARGET_COLUMNS = ["attack_cat", "label"]
TOP_FEATURES = 30  # Number of top features to select per chunk

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

label_encoders = {}
feature_cols = None
max_vals = None
skipped_chunks = []

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
# Load and process chunks
# -------------------------
chunk_files = sorted([os.path.join(CHUNK_FOLDER, f) for f in os.listdir(CHUNK_FOLDER) if f.endswith(".csv")])
print("🚀 Starting feature extraction...")

for i, file in enumerate(chunk_files):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    # Detect target column
    target_col = next((col for col in TARGET_COLUMNS if col in df.columns), None)
    if target_col is None:
        print(f"⚠ Skipping chunk {i+1}: no target column found")
        skipped_chunks.append(file)
        continue

    # Drop rows with NaN in target
    df = df.dropna(subset=[target_col])
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode categorical columns
    for col in X.select_dtypes(include='object').columns:
        if col not in label_encoders:
            label_encoders[col] = LabelEncoder()
            X[col] = label_encoders[col].fit_transform(X[col].astype(str))
        else:
            X[col] = X[col].map(lambda s: label_encoders[col].transform([s])[0] if s in label_encoders[col].classes_ else -1)

    feature_cols = list(X.columns)

    # Replace NaN and normalize
    X = X.fillna(0)
    X = safe_normalize(X)

    # -------------------------
    # Compute feature importance using RandomForest
    # -------------------------
    try:
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        top_features_idx = np.argsort(importances)[::-1][:min(TOP_FEATURES, X.shape[1])]
        top_features_names = [feature_cols[j] for j in top_features_idx]
    except Exception as e:
        print(f"⚠ Feature importance failed for chunk {i+1}: {e}")
        top_features_idx = list(range(X.shape[1]))
        top_features_names = feature_cols

    # -------------------------
    # Save processed chunk with top features info
    # -------------------------
    X_df = pd.DataFrame(X, columns=feature_cols)
    processed_file = os.path.join(PROCESSED_FOLDER, f"chunk_{i+1:03d}.csv")
    pd.concat([X_df, y.reset_index(drop=True)], axis=1).to_csv(processed_file, index=False)

    # Save feature importance for this chunk
    importance_file = os.path.join(PROCESSED_FOLDER, f"chunk_{i+1:03d}_features.pkl")
    pickle.dump({"top_features_idx": top_features_idx, "top_features_names": top_features_names}, open(importance_file, "wb"))

    print(f"✅ Saved processed chunk {i+1} — {processed_file}")
    print(f"   Top {len(top_features_names)} features saved → {importance_file}")

# -------------------------
# Save encoders, features, normalization
# -------------------------
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))
pickle.dump(feature_cols, open("feature_cols.pkl", "wb"))
np.save("max_vals.npy", max_vals)

print(f"\n🎯 Feature extraction complete! Skipped {len(skipped_chunks)} chunks")
