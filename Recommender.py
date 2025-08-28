# ================================
# 📌 Professional Fabric Recommender Web App (Enhanced UI)
# ================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import altair as alt


# -------------------------------
# STEP 1: Load & Clean Datasets
# -------------------------------
@st.cache_data
def load_data():
    url_lit = "https://raw.githubusercontent.com/Volandofernando/Material-Literature-data-/main/Dataset.xlsx"
    url_survey = "https://raw.githubusercontent.com/Volandofernando/REAL-TIME-Dataset/main/IT%20Innovation%20in%20Fabric%20Industry%20%20(Responses).xlsx"
    
    df_lit = pd.read_excel(url_lit)
    df_survey = pd.read_excel(url_survey)

    def clean_columns(df):
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(r"[^\w]", "_", regex=True)
        return df

    df_lit = clean_columns(df_lit)
    df_survey = clean_columns(df_survey)
    df = pd.concat([df_lit, df_survey], ignore_index=True, sort=False)
    
    return df

df = load_data()

# -------------------------------
# STEP 2: Detect Features & Target Dynamically
# -------------------------------
feature_keywords = {
    "moisture_regain": ["moisture", "regain"],
    "water_absorption": ["water", "absorption"],
    "drying_time": ["drying", "time"],
    "thermal_conductivity": ["thermal", "conductivity"]
}
target_keywords = ["comfort", "score"]

def find_column(df_cols, keywords):
    for col in df_cols:
        if all(k in col for k in keywords):
            return col
    return None

feature_cols = [find_column(df.columns, kw) for kw in feature_keywords.values()]
target_col = find_column(df.columns, target_keywords)
feature_cols = [c for c in feature_cols if c is not None]

if target_col is None or len(feature_cols) < 4:
    st.error("❌ Could not detect all required features or target column. Check dataset columns!")
    st.stop()

# -------------------------------
# STEP 3: Prepare Data & Train Model
# -------------------------------
df_clean = df.dropna(subset=feature_cols + [target_col])

X = df_clean[feature_cols]
y = df_clean[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# STEP 4: Sidebar Input
# -------------------------------
st.sidebar.header("📌 Input Your Conditions")
temperature = st.sidebar.slider("🌡️ Temperature (°C)", 15, 40, 30)
humidity = st.sidebar.slider("💧 Humidity (%)", 10, 100, 70)
sweat_sensitivity = st.sidebar.selectbox("🧍 Sweat Sensitivity", ["Low", "Medium", "High"])
activity_intensity = st.sidebar.selectbox("🏃 Activity Intensity", ["Low", "Moderate", "High"])

sweat_map = {"Low": 1, "Medium": 2, "High": 3}
activity_map = {"Low": 1, "Moderate": 2, "High": 3}

sweat_num = sweat_map[sweat_sensitivity]
activity_num = activity_map[activity_intensity]

# -------------------------------
# STEP 5: Prepare Features for Prediction
# -------------------------------
moisture_regain = sweat_num * 5
water_absorption = 800 + humidity * 5
drying_time = 60 + activity_num * 10
thermal_conductivity = 0.04 + (temperature - 25) * 0.001

user_input = np.array([[moisture_regain, water_absorption, drying_time, thermal_conductivity]])
user_input_scaled = scaler.transform(user_input)

# -------------------------------
# STEP 6: Predict & Recommend Top 3
# -------------------------------
predicted_score = model.predict(user_input_scaled)[0]

df_clean["predicted_diff"] = abs(df_clean[target_col] - predicted_score)
top_matches = df_clean.sort_values(by="predicted_diff").head(3)

# -------------------------------
# STEP 7: Display Recommendation
# -------------------------------
st.markdown("## 🔹 Top Fabric Recommendations")
cols = st.columns(3)
for i, (_, row) in enumerate(top_matches.iterrows()):
    with cols[i]:
        st.markdown(f"### {row.get('fabric_type','Unknown')}")
        st.metric("Comfort Score", round(row[target_col],2))
        st.markdown(f"**Moisture Regain:** {row[feature_cols[0]]} %")
        st.markdown(f"**Water Absorption:** {row[feature_cols[1]]} g/m²")
        st.markdown(f"**Drying Time:** {row[feature_cols[2]]} min")
        st.markdown(f"**Thermal Conductivity:** {row[feature_cols[3]]} W/m·K")
        st.markdown(f"📖 {row.get('source__literature_reference','Survey Data')}")

# -------------------------------
# Optional: Bar Chart Comparison
# -------------------------------
chart_data = top_matches[[target_col, "fabric_type"]].rename(columns={target_col:"Comfort Score"})
chart = alt.Chart(chart_data).mark_bar(color="#4CAF50").encode(
    x="fabric_type",
    y="Comfort Score",
    tooltip=["fabric_type","Comfort Score"]
).properties(height=300)
st.altair_chart(chart, use_container_width=True)

# -------------------------------
# Custom Styling (Optional)
# -------------------------------
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
}
h2, h3 {
    color: #1F77B4;
}
</style>
""", unsafe_allow_html=True)



