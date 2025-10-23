import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# Configurations
# -------------------------
PROCESSED_CHUNK_FOLDER = "processed_chunks"
CLUSTER_MODELS_DIR = "cluster_models"
os.makedirs(CLUSTER_MODELS_DIR, exist_ok=True)
SKIPPED_LOG_FILE = "skipped_chunks.log"
SVC_MAX_ROWS = 3000  # threshold to skip SVC for large clusters

# Base models with faster settings
BASE_MODELS = [
    RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42),
    GradientBoostingClassifier(n_estimators=50, random_state=42),
    ExtraTreesClassifier(n_estimators=50, n_jobs=-1, random_state=42),
    KNeighborsClassifier(n_neighbors=5),
    GaussianNB(),
    SVC(probability=False, random_state=42),  # optimized for speed
    SGDClassifier(max_iter=500, random_state=42),
    MLPClassifier(max_iter=200, random_state=42)
]

# -------------------------
# Cloud-aware selection
# -------------------------
def select_best_model_for_cloud(cluster_models, cloud_env):
    models_sorted = sorted(cluster_models, key=lambda x: x["f1_score"], reverse=True)
    for m in models_sorted:
        name = m["model_name"]
        if cloud_env.get("GPU_available") and name in ["MLPClassifier", "GradientBoostingClassifier"]:
            return m
        if cloud_env.get("RAM_GB", 16) < 8 and name in ["RandomForestClassifier", "ExtraTreesClassifier"]:
            return m
        if not cloud_env.get("GPU_available") and name in ["KNeighborsClassifier", "SGDClassifier", "SVC"]:
            return m
    return models_sorted[0]

# -------------------------
# Train safely
# -------------------------
def train_model_safe(clf, X, y, model_name, cluster_id):
    if len(set(y)) < 2:
        print(f"⚠ Skipping {model_name} for Cluster {cluster_id}: only one class present.")
        return None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        return {"model_name": model_name, "model": clf, "accuracy": acc, "f1_score": f1}
    except Exception as e:
        print(f"⚠ Training failed for {model_name} Cluster {cluster_id}: {e}")
        return None

# -------------------------
# Load chunks & features
# -------------------------
chunk_files = sorted([f for f in os.listdir(PROCESSED_CHUNK_FOLDER) 
                      if f.startswith("chunk_") and f.endswith(".csv")])

if not chunk_files:
    raise ValueError("❌ No chunk CSV files found in processed_chunks folder.")

print(f"🚀 Found {len(chunk_files)} chunk files.")

X_all, y_all = [], []
skipped_chunks = []

for chunk_file in chunk_files:
    chunk_path = os.path.join(PROCESSED_CHUNK_FOLDER, chunk_file)
    df = pd.read_csv(chunk_path)
    df.columns = df.columns.str.strip().str.lower()
    
    target_col = "attack_cat" if "attack_cat" in df.columns else df.columns[-1]
    
    # Load top features
    feature_file = chunk_file.replace(".csv", "_features.pkl")
    feature_path = os.path.join(PROCESSED_CHUNK_FOLDER, feature_file)
    
    if os.path.exists(feature_path):
        try:
            feature_data = pickle.load(open(feature_path, "rb"))
            top_features = [f.strip().lower() for f in feature_data.get("top_features_names", [])]
            available_features = [f for f in top_features if f in df.columns]
            if not available_features:
                print(f"⚠ Chunk {chunk_file}: top features not in CSV, using all except target.")
                available_features = [c for c in df.columns if c != target_col]
        except Exception as e:
            print(f"⚠ Failed to load features for {chunk_file}: {e}, using all except target.")
            available_features = [c for c in df.columns if c != target_col]
    else:
        print(f"⚠ No feature file for {chunk_file}, using all except target.")
        available_features = [c for c in df.columns if c != target_col]
    
    if not available_features:
        print(f"⚠ Chunk {chunk_file} skipped: no features available.")
        skipped_chunks.append(chunk_file)
        continue
    
    X_all.append(df[available_features].values)
    y_all.append(df[target_col].values)

# Log skipped chunks
if skipped_chunks:
    with open(SKIPPED_LOG_FILE, "w") as f:
        for c in skipped_chunks:
            f.write(f"{c}\n")
    print(f"⚠ Skipped {len(skipped_chunks)} chunks. Details in {SKIPPED_LOG_FILE}")

if not X_all:
    raise ValueError("❌ No valid data to train models!")

X_all = np.vstack(X_all)
y_all = np.hstack(y_all)
print(f"✅ Combined data: {X_all.shape[0]} rows, {X_all.shape[1]} features")

# -------------------------
# KMeans clustering
# -------------------------
kmeans_file = "kmeans_model.pkl"
if os.path.exists(kmeans_file):
    kmeans = pickle.load(open(kmeans_file, "rb"))
    print("✅ Loaded existing KMeans model.")
else:
    print("⚙️ Training new KMeans cluster model...")
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    kmeans.fit(X_all)
    pickle.dump(kmeans, open(kmeans_file, "wb"))
    print(f"✅ Saved KMeans model to {kmeans_file}")

cluster_labels = kmeans.predict(X_all)

# -------------------------
# Train cluster-specific models
# -------------------------
cluster_models = defaultdict(list)
cloud_env_example = {"GPU_available": True, "RAM_GB": 16, "CPU_cores": 8}

for cluster_id in np.unique(cluster_labels):
    indices = np.where(cluster_labels == cluster_id)[0]
    X_cluster = X_all[indices]
    y_cluster = y_all[indices]

    print(f"\n📌 Training models for Cluster {cluster_id} ({len(y_cluster)} records)")

    for clf_template in BASE_MODELS:
        # Skip SVC if cluster too large
        if isinstance(clf_template, SVC) and len(y_cluster) > SVC_MAX_ROWS:
            print(f"⚠ Skipping SVC for Cluster {cluster_id} (too many rows: {len(y_cluster)})")
            continue

        trained = train_model_safe(clf_template, X_cluster, y_cluster, type(clf_template).__name__, cluster_id)
        if trained:
            cluster_models[cluster_id].append(trained)
            print(f"✅ Model: {trained['model_name']} | Accuracy: {trained['accuracy']:.3f} | F1: {trained['f1_score']:.3f}")

    if cluster_models[cluster_id]:
        best_model = select_best_model_for_cloud(cluster_models[cluster_id], cloud_env_example)
        print(f"🎯 Best model for Cluster {cluster_id} (Cloud-aware): {best_model['model_name']} | Accuracy: {best_model['accuracy']:.3f} | F1: {best_model['f1_score']:.3f}")

    # Save all models
    cluster_file = os.path.join(CLUSTER_MODELS_DIR, f"cluster_{cluster_id}_models.pkl")
    pickle.dump(cluster_models[cluster_id], open(cluster_file, "wb"))
    print(f"💾 Saved all models for Cluster {cluster_id} to {cluster_file}")

print("\n🎯 All clusters trained with cloud-aware model selection successfully! ✅")
