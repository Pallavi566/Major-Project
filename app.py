import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import time
import copy
import plotly.express as px
import plotly.graph_objects as go
import shap
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

# -------------------------------
# Page Config (must be first)
# -------------------------------
st.set_page_config(
    page_title="Dynamic Auto-Selection ML Dashboard",
    layout="wide",
    page_icon="🚀"
)

# -------------------------------
# File Paths
# -------------------------------
CHUNK_FOLDER = "processed_chunks"
CLUSTER_MODELS_FILE = "cluster_models.pkl"
FEATURE_COLS_FILE = "feature_cols.pkl"
LABEL_ENCODERS_FILE = "label_encoders.pkl"
MAX_VALS_FILE = "max_vals.npy"

# -------------------------------
# Load Models & Preprocessing
# -------------------------------
try:
    cluster_models = pickle.load(open(CLUSTER_MODELS_FILE, "rb"))
    feature_cols = pickle.load(open(FEATURE_COLS_FILE, "rb"))
    label_encoders = pickle.load(open(LABEL_ENCODERS_FILE, "rb"))
    max_vals = np.load(MAX_VALS_FILE)
    CATEGORICAL_COLS = list(label_encoders.keys())
except Exception as e:
    st.error(f"Failed to load model files: {e}")
    st.stop()

# -------------------------------
# Load Chunk Files
# -------------------------------
chunk_files = sorted([os.path.join(CHUNK_FOLDER, f) for f in os.listdir(CHUNK_FOLDER) if f.endswith(".csv")])
if not chunk_files:
    st.error(f"No processed chunks found in {CHUNK_FOLDER}")
    st.stop()

# -------------------------------
# Session State
# -------------------------------
if 'chunk_idx' not in st.session_state:
    st.session_state.chunk_idx = 0
if 'all_predictions' not in st.session_state:
    st.session_state.all_predictions = []
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {type(clf_dict["model"]).__name__: [] 
                                      for cluster in cluster_models.values() 
                                      for clf_dict in cluster}

# -------------------------------
# Helper Functions
# -------------------------------
def preprocess_record(record):
    rec = record.copy()
    for i, col in enumerate(feature_cols):
        if col in CATEGORICAL_COLS:
            try:
                rec[i] = label_encoders[col].transform([str(rec[i])])[0]
            except:
                rec[i] = -1
        else:
            try:
                rec[i] = float(rec[i])
            except:
                rec[i] = 0.0
    rec = np.array(rec, dtype=np.float64)
    rec = np.divide(rec, max_vals, out=np.zeros_like(rec), where=max_vals != 0)
    return rec.reshape(1, -1)

def predict_record(record):
    rec_norm = preprocess_record(record)
    preds = []
    for cluster_id, models in cluster_models.items():
        for clf_dict in models:
            clf = clf_dict["model"]
            features = clf_dict["features"]
            try:
                pred = clf.predict(rec_norm[:, [feature_cols.index(f) for f in features]])[0]
                preds.append(pred)
            except:
                continue
    if preds:
        return max(set(preds), key=preds.count)
    return "No prediction"

# -------------------------------
# Tabs Layout
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📡 Streaming Predictions", "📈 Model Accuracy & Confusion", "🧩 SHAP Explainability"])

# -------------------------------
# Tab 1: Streaming Predictions
# -------------------------------
with tab1:
    st.subheader("🚀 Streaming Predictions Per Chunk")
    status_text = st.empty()
    prediction_table = st.empty()
    per_model_table = st.empty()

    DISPLAY_RECORDS = 20

    for idx in range(st.session_state.chunk_idx, len(chunk_files)):
        file = chunk_files[idx]
        df_chunk = pd.read_csv(file)
        status_text.text(f"Processing Chunk {idx+1}/{len(chunk_files)}: {os.path.basename(file)}")

        # Preprocess categorical and numeric columns
        for col in df_chunk.columns:
            if col in CATEGORICAL_COLS:
                df_chunk[col] = df_chunk[col].fillna("missing").astype(str).apply(
                    lambda x: label_encoders[col].transform([x])[0]
                    if str(x) in label_encoders[col].classes_ else -1
                )
            else:
                df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce').fillna(0)

        per_model_predictions = []

        for record_idx, row in df_chunk.iterrows():
            rec_list = row[feature_cols].values.tolist()
            pred = predict_record(rec_list)
            st.session_state.all_predictions.append({
                "Chunk": idx + 1,
                "Record": record_idx,
                "Predicted": pred,
                "Actual": row.get("attack_cat", "N/A")
            })

            # Per-model predictions
            record_preds = {"Record": record_idx}
            for cluster_id, models in cluster_models.items():
                for clf_dict in models:
                    clf = clf_dict["model"]
                    features = clf_dict["features"]
                    model_name = type(clf).__name__
                    try:
                        record_pred = clf.predict(np.array([rec_list])[:, [feature_cols.index(f) for f in features]])[0]
                        record_preds[model_name] = record_pred
                    except:
                        record_preds[model_name] = "N/A"
            per_model_predictions.append(record_preds)

            # Update tables dynamically
            pred_df = pd.DataFrame(st.session_state.all_predictions[-DISPLAY_RECORDS:])
            prediction_table.dataframe(pred_df)

            per_model_df = pd.DataFrame(per_model_predictions[-DISPLAY_RECORDS:])
            per_model_table.dataframe(per_model_df)

            time.sleep(0.05)  # simulate streaming

        # Update model metrics
        for cluster_id, models in cluster_models.items():
            for clf_dict in models:
                clf = clf_dict["model"]
                features = clf_dict["features"]
                try:
                    X_eval = df_chunk[features].values.astype(np.float64)
                    Y_true = df_chunk.get("attack_cat", pd.Series([0]*len(df_chunk)))
                    acc = clf.score(X_eval, Y_true)
                    st.session_state.model_metrics[type(clf).__name__].append(acc)
                except:
                    continue

        st.session_state.chunk_idx += 1

# -------------------------------
# Tab 2: Model Accuracy & Confusion Matrices
# -------------------------------
with tab2:
    st.subheader("📊 Dynamic Model Accuracy & Confusion Matrices")

    # Accuracy chart
    metric_df = pd.DataFrame({k: pd.Series(v) for k, v in st.session_state.model_metrics.items()})
    metric_df["Chunk"] = range(1, len(metric_df)+1)
    metric_df_melt = metric_df.melt(id_vars="Chunk", var_name="Model", value_name="Accuracy")

    fig = px.line(metric_df_melt, x="Chunk", y="Accuracy", color="Model", markers=True,
                  title="Model Accuracy Over Chunks", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices for last chunk
    if st.session_state.all_predictions:
        last_chunk_df = pd.DataFrame(st.session_state.all_predictions)
        last_chunk = last_chunk_df["Chunk"].max()
        df_chunk_last = last_chunk_df[last_chunk_df["Chunk"] == last_chunk]
        st.write(f"Confusion Matrices for Chunk {last_chunk}")

        for model_name in st.session_state.model_metrics.keys():
            y_true = df_chunk_last["Actual"]
            y_pred = df_chunk_last.get(model_name, pd.Series(["N/A"]*len(df_chunk_last)))
            if not y_true.empty and set(y_true) != {"N/A"}:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
                st.write(f"**{model_name}**")
                st.dataframe(pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true)))

# -------------------------------
# Tab 3: SHAP Explainability
# -------------------------------
with tab3:
    st.subheader("🧩 SHAP Feature Explainability")
    shap_chunk_select = st.selectbox("Select Chunk for SHAP", range(1, len(chunk_files)+1))
    shap_record_select = st.number_input("Select Record Index", min_value=0, value=0)

    df_shap = pd.read_csv(chunk_files[shap_chunk_select-1])

    try:
        # Preprocess chunk for SHAP
        sample_features = cluster_models[0][0]["features"]
        df_sample_pre = df_shap[sample_features].copy()
        for col in df_sample_pre.columns:
            if col in CATEGORICAL_COLS:
                df_sample_pre[col] = df_sample_pre[col].fillna("missing").apply(
                    lambda x: label_encoders[col].transform([x])[0]
                    if str(x) in label_encoders[col].classes_ else -1
                )
            else:
                df_sample_pre[col] = pd.to_numeric(df_sample_pre[col], errors='coerce').fillna(0)

        X_shap = np.divide(
            df_sample_pre.values,
            max_vals[[feature_cols.index(f) for f in sample_features]],
            out=np.zeros_like(df_sample_pre.values),
            where=max_vals[[feature_cols.index(f) for f in sample_features]] != 0
        )

        # SHAP Explainer
        explainer = shap.Explainer(cluster_models[0][0]["model"], X_shap)
        shap_values = explainer(X_shap)

        st.write("SHAP Summary Plot (Chunk-Level)")
        st.pyplot(shap.plots.bar(shap_values))

        if shap_record_select < len(X_shap):
            st.write(f"SHAP Force Plot for Record {shap_record_select}")
            st_shap = shap.plots.force(shap_values[shap_record_select], matplotlib=True)
            st.pyplot(st_shap)
    except Exception as e:
        st.warning(f"SHAP explainability not available: {e}")

# -------------------------------
# Download Trained Models
# -------------------------------
st.subheader("💾 Download Trained Models")
if st.button("Download Cluster Models"):
    import shutil
    shutil.make_archive("trained_models", 'zip', CHUNK_FOLDER)
    st.success("✅ Trained models zipped and ready to download.")
