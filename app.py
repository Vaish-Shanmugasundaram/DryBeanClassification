from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

with open(BASE_DIR / "metadata.json", "r", encoding="utf-8") as file:
    metadata = json.load(file)

FEATURE_NAMES = metadata["feature_names"]
TARGET_COLUMN = metadata["target_column"]
CLASS_NAMES = metadata["class_names"]
MODEL_NAMES = metadata["model_names"]

@st.cache_resource
def load_models():
    loaded = {}
    for name in MODEL_NAMES:
        filename = name.lower().replace(" ", "_") + ".joblib"
        loaded[name] = joblib.load(MODEL_DIR / filename)
    return loaded

st.set_page_config(page_title="Dry Bean Classifier", layout="wide")
st.title("Dry Bean Classification")
st.write("Upload a labelled CSV with the required features and target column to evaluate a saved model.")

uploaded_file = st.file_uploader("Upload labelled test CSV", type=["csv"])
selected_model_name = st.selectbox("Choose a trained model", MODEL_NAMES)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    required_columns = FEATURE_NAMES + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        st.error("Missing required columns: " + ", ".join(missing_columns))
    elif data.empty:
        st.error("The uploaded CSV is empty.")
    else:
        X_upload = data[FEATURE_NAMES].copy()
        y_text = data[TARGET_COLUMN].astype(str)

        class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
        unknown_classes = sorted(set(y_text) - set(class_to_index))

        if unknown_classes:
            st.error("Unknown target class values: " + ", ".join(unknown_classes))
        else:
            y_true = y_text.map(class_to_index).to_numpy()
            model = load_models()[selected_model_name]
            y_pred = model.predict(X_upload)
            y_probability = model.predict_proba(X_upload)

            score_values = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision (weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "Recall (weighted)": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "F1 Score (weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "MCC": matthews_corrcoef(y_true, y_pred)
            }

            try:
                score_values["AUC (weighted OVR)"] = roc_auc_score(
                    y_true, y_probability,
                    labels=model.classes_,
                    multi_class="ovr",
                    average="weighted"
                )
            except ValueError:
                score_values["AUC (weighted OVR)"] = float("nan")
                st.warning("AUC could not be calculated. The uploaded data may not contain every class.")

            st.subheader("Evaluation metrics")
            st.dataframe(pd.DataFrame([score_values]).round(4), use_container_width=True)

            st.subheader("Confusion matrix")
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
            cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
            st.dataframe(cm_df, use_container_width=True)

            st.subheader("Classification report")
            report = classification_report(
                y_true, y_pred,
                labels=np.arange(len(CLASS_NAMES)),
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0
            )
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
else:
    st.info("Upload a labelled CSV to view evaluation results.")
