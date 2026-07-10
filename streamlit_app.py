import streamlit as st
import pandas as pd
import torch
import numpy as np
import joblib

from model import LayoffRNN

# Load model, encoder & scaler
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("../models/scaler.pkl")
    encoder = joblib.load("../models/encoder.pkl")

    # Determine input size (numeric features + department one-hot)
    numeric_feature_count = len(scaler.mean_)
    dept_feature_count = len(encoder.categories_[0])
    input_size = numeric_feature_count + dept_feature_count

    # Load model
    model = LayoffRNN(input_size=input_size)
    model.load_state_dict(torch.load("../models/best_model.pt", map_location="cpu"))
    model.eval()

    return model, scaler, encoder


# Prepare sequences for each employee
def prepare_features(df, scaler, encoder, seq_len=6):
    numeric = [
        "TenureYears", "Performance", "TrainingHours",
        "Projects", "OvertimeHours", "LastPromotionYears", "Salary"
    ]

    dept_cols = [f"DEPT_{c}" for c in encoder.categories_[0]]
    sequences = {}

    for emp_id, g in df.groupby("EmployeeID"):
        g = g.sort_values("RecordMonth")

        g = g.tail(seq_len).copy()

        if len(g) < seq_len:
            pad_rows = seq_len - len(g)
            first_row = g.iloc[[0]]
            g = pd.concat([first_row] * pad_rows + [g], ignore_index=True)

        # Encode department
        dept_encoded = encoder.transform(g[["Department"]])
        dept_df = pd.DataFrame(dept_encoded, columns=dept_cols)

        # Merge numeric + one-hot
        g = pd.concat([g[numeric].reset_index(drop=True),
                       dept_df.reset_index(drop=True)], axis=1)

        # Scale numeric
        nums = g[numeric].values
        nums_scaled = scaler.transform(nums)

        final = np.hstack([nums_scaled, g[dept_cols].values])
        sequences[emp_id] = torch.tensor(final, dtype=torch.float32).unsqueeze(0)

    return sequences

# Classify risk level
def classify(prob):
    if prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Moderate"
    else:
        return "Low"

# Row color formatting
def color_row(row):
    if row["Risk Level"] == "High":
        return ['background-color: #ff4d4d; color:white; font-weight:bold'] * len(row)
    elif row["Risk Level"] == "Moderate":
        return ['background-color: #ffa64d; color:black; font-weight:bold'] * len(row)
    else:
        return ['background-color: #5cd65c; color:black; font-weight:bold'] * len(row)

# STREAMLIT UI
st.title("Layoff Prediction Dashboard")

file = st.file_uploader("Upload Employee CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    model, scaler, encoder = load_artifacts()
    sequences = prepare_features(df, scaler, encoder)

    # Run predictions
    results = []
    for emp_id, x in sequences.items():
        with torch.no_grad():
            logit = model(x).item()
            prob = float(1 / (1 + np.exp(-logit)))

        results.append([emp_id, round(prob, 4), classify(prob)])

    results_df = pd.DataFrame(results, columns=["EmployeeID", "Probability", "Risk Level"])
    results_df = results_df.sort_values("Probability", ascending=False)

    st.subheader("Predictions")

    # Apply row coloring
    styled_table = results_df.style.apply(color_row, axis=1)

    st.dataframe(styled_table, use_container_width=True)
