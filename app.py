import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="World Happiness Predictor",
    page_icon="🌍",
    layout="centered"
)

st.title("🌟 World Happiness Level Predictor")
st.write("Adjust the sliders to predict a country's happiness level.")

# Load model and label encoder
model = joblib.load("model_4_features.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Features (MATCH TRAINING)
features = {
    "Economy (GDP per Capita)": (0.0, 2.0, 1.0),
    "Health (Life Expectancy)": (0.0, 1.5, 1.0),
    "Freedom": (0.0, 1.0, 0.6)
}

st.subheader("Set feature values:")

# Sliders
user_input = {}
for feature, (min_val, max_val, default_val) in features.items():
    user_input[feature] = st.slider(
        feature,
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=0.01
    )

# DataFrame with correct order
input_df = pd.DataFrame([user_input])[
    [
        "Economy (GDP per Capita)",
        "Health (Life Expectancy)",
        "Freedom"
    ]
]

# Predict
if st.button("Predict Happiness Level"):
    prediction = model.predict(input_df)[0]
    label = label_encoder.inverse_transform([prediction])[0]

    if label == "High":
        st.success("😄 Predicted Happiness Level: **HIGH**")
    elif label == "Medium":
        st.warning("😐 Predicted Happiness Level: **MEDIUM**")
    else:
        st.error("😢 Predicted Happiness Level: **LOW**")
