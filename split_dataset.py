import pandas as pd
import os

DATA_FILE = "UNSW_NB15_training-set.csv"
CHUNK_FOLDER = "chunks"
CHUNK_SIZE = 50000
TARGET_COLUMN = "attack_cat"

os.makedirs(CHUNK_FOLDER, exist_ok=True)

print("🔄 Splitting dataset into chunks...")

missing_target_chunks = []

for i, chunk in enumerate(pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE)):
    # Normalize column names
    chunk.columns = chunk.columns.str.strip()

    # Check for target column
    if TARGET_COLUMN not in chunk.columns:
        missing_target_chunks.append(f"chunk_{i+1}")
        print(f"⚠️ Target column '{TARGET_COLUMN}' missing in chunk {i+1}")

    chunk_file = f"{CHUNK_FOLDER}/chunk_{i+1:03d}.csv"
    chunk.to_csv(chunk_file, index=False)
    print(f"✅ Saved {chunk_file} — Shape: {chunk.shape}")

print(f"\n🎯 Total chunks created: {i+1}")
if missing_target_chunks:
    print(f"⚠️ Chunks missing target column: {missing_target_chunks}")
