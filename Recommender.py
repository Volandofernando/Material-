pip install sklearn.ensemble
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import shap

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Dataset.xlsx")
    return df

# ---------------------------
# Train Model
# ---------------------------
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["FabricRecommendation"])  # change to your target column name
    y = df["FabricRecommendation"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

# ---------------------------
# App UI
# ---------------------------
st.title("🧵 AI-Powered Fabric Recommender")

st.write("Upload your dataset, train a model, and get recommendations for sweat-free clothing.")

# Load data
df = load_data()
st.write("### Dataset Preview", df.head())

# Train model
model, scaler = train_model(df)

# Input section
st.write("### Enter Your Conditions")
temp = st.number_input("Temperature (°C)", 20, 40, 25)
humidity = st.number_input("Humidity (%)", 30, 90, 60)
activity = st.number_input("Activity Level (1–10)", 1, 10, 5)

# Predict
if st.button("Recommend Fabric"):
    user_input = np.array([[temp, humidity, activity]])
    user_scaled = scaler.transform(user_input)
    prediction = model.predict(user_scaled)[0]
    st.success(f"✅ Recommended Fabric: **{prediction}**")

    # SHAP explanation
    explainer = shap.Explainer(model, scaler.transform(df.drop(columns=["FabricRecommendation"])))
    shap_values = explainer(user_scaled)
    st.write("### 🔍 Why this fabric?")
    st.pyplot(shap.plots.waterfall(shap_values[0], show=False))

