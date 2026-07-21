import streamlit as st
import pandas as pd
import numpy as np 
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Anime Score Prediction",
    layout="centered"
)

st.title("🎌 Anime Score Prediction")
st.write("Enter Anime Details")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")

# -----------------------------
# User Input
# -----------------------------
anime_id = st.number_input("Anime ID", value=1)

rank = st.number_input("Rank", value=1)

popularity = st.number_input("Popularity", value=1)

members = st.number_input("Members", value=1000)

episodes = st.number_input("Episodes", value=12)

type_option = st.selectbox(
    "Type",
    [
        "Movie",
        "ONA",
        "OVA",
        "Special",
        "TV"
    ]
)

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_df = pd.DataFrame([[anime_id,
                          rank,
                          popularity,
                          members,
                          episodes]],
                        columns=[
                            "anime_id",
                            "rank",
                            "popularity",
                            "members",
                            "episodes"
                        ])

# One Hot Encoding

for col in columns:
    if col.startswith("type_"):
        input_df[col] = 0

selected = "type_" + type_option

if selected in input_df.columns:
    input_df[selected] = 1

# Reindex

input_df = input_df.reindex(columns=columns, fill_value=0)

# Scaling

input_scaled = scaler.transform(input_df)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Score"):

    prediction = model.predict(input_scaled)

    st.success(f"Predicted Anime Score : {prediction[0]:.2f}")