import pandas as pd
import numpy as np
import pickle
import os

# -----------------------------
# CONFIG
# -----------------------------
chunks_folder = "chunks/"            # folder with your dataset chunks
processed_folder = "processed_chunks/"  # folder to save processed chunks
target_col = "attack_cat"            # target label

feature_cols_file = "feature_cols.pkl"
max_vals_file = "max_vals.npy"
label_encoders_file = "label_encoders.pkl"

os.makedirs(processed_folder, exist_ok=True)

# -----------------------------
# Load feature extraction artifacts
# -----------------------------
with open(feature_cols_file, "rb") as f:
    selected_features = pickle.load(f)

max_vals = np.load(max_vals_file)

with open(label_encoders_file, "rb") as f:
    label_encoders = pickle.load(f)

categorical_cols = list(label_encoders.keys())

# -----------------------------
# Process each chunk
# -----------------------------
chunk_files = sorted([f for f in os.listdir(chunks_folder) if f.endswith(".csv")])

print(f"Found {len(chunk_files)} chunks to process.")

for file in chunk_files:
    print(f"\nProcessing chunk: {file}")
    df = pd.read_csv(os.path.join(chunks_folder, file))

    # Drop label column if exists
    if "label" in df.columns:
        df.drop("label", axis=1, inplace=True)

    # Encode categorical columns using saved label encoders
    for col in categorical_cols:
        if col in df.columns:
            le = label_encoders[col]
            df[col] = df[col].map(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)

    # Keep only selected features + target
    df_selected = df[selected_features + [target_col]]

    # Normalize numeric features using max_vals
    for i, col in enumerate(selected_features):
        if col in df_selected.columns:
            if max_vals[i] > 0:
                df_selected[col] = np.round(df_selected[col] / max_vals[i], 3)
            else:
                df_selected[col] = 0.0

    # Save processed chunk
    processed_file = os.path.join(processed_folder, file)
    df_selected.to_csv(processed_file, index=False)
    print(f"✅ Saved processed chunk to {processed_file}")

print("\nAll chunks processed successfully!")
