# NovaDhruv Financial — ChurnShield AI
# Professional Multi-Region Banking Version with Gemini AI Insights

import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os
from dotenv import load_dotenv
from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras.optimizers import Adam
import google.generativeai as genai  # Gemini SDK
import shap

# ── Load API Key ─────────────────────────────────────────────────────────────
load_dotenv()
GEMINI_KEY = os.getenv("Gemini_API_Key")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
else:
    st.warning("⚠️ Gemini API key not found. Please add it to .env")

# ── Load Model & Encoders ─────────────────────────────────────────────────────
model = tf.keras.models.load_model(
    "churn_model.h5",
    custom_objects={"LeakyReLU": LeakyReLU, "ReLU": ReLU, "Adam": Adam},
    compile=False
)

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehotencoder_geography.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("churn_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ── App Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NovaDhruv Financial — ChurnShield AI", layout="wide")

# ── Branding (Logo + Header) ─────────────────────────────────────────────────
st.image("pic.png", width=200)   # Main logo at top
st.title("🏦 NovaDhruv Financial — ChurnShield AI")
st.markdown("### Guiding global banks to retain customers with AI-powered insights.")

# Sidebar branding
st.sidebar.image("pic.png", width=120)
st.sidebar.header("📂 Data Options")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Overview",
    "📋 RM Dashboard",
    "📢 Campaign Generator",
    "➕ Add Single Customer"
])

# ── Session Dataset ──────────────────────────────────────────────────────────
if "customer_data" not in st.session_state:
    st.session_state["customer_data"] = pd.DataFrame()

# ── File Upload ──────────────────────────────────────────────────────────────
uploaded_file = st.sidebar.file_uploader("📂 Upload a CSV for batch predictions:", type="csv")

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    # Drop unused columns if present
    drop_cols = ["RowNumber", "CustomerId", "Surname", "Exited"]
    raw_df = raw_df.drop(columns=[c for c in drop_cols if c in raw_df.columns])

    # Encode Gender
    if "Gender" in raw_df.columns:
        raw_df["Gender"] = label_encoder_gender.transform(raw_df["Gender"])

    # Encode Geography
    if "Geography" in raw_df.columns:
        geo_encoded = onehot_encoder_geo.transform(raw_df[["Geography"]])
        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
        )
        raw_df = pd.concat([raw_df.drop("Geography", axis=1).reset_index(drop=True),
                            geo_encoded_df.reset_index(drop=True)], axis=1)

    # Scale + Predictions
    batch_scaled = scaler.transform(raw_df)
    batch_preds = model.predict(batch_scaled).flatten()
    raw_df["Churn Probability"] = batch_preds
    raw_df["Risk Level"] = pd.cut(
        raw_df["Churn Probability"],
        bins=[0, 0.3, 0.7, 1],
        labels=["Low", "Medium", "High"]
    )

    # Save in session state
    st.session_state["customer_data"] = raw_df.copy()

# ── Executive Overview Tab ───────────────────────────────────────────────────
with tab1:
    st.header("📊 Executive Overview")

    if len(st.session_state["customer_data"]) > 0:
        df = st.session_state["customer_data"]
        summary = df["Risk Level"].value_counts(normalize=True).mul(100).round(2).to_dict()

        st.metric("Total Customers", len(df))
        st.metric("High Risk %", f"{summary.get('High',0):.2f}%")

        st.subheader("Portfolio Risk Distribution")
        st.bar_chart(df["Risk Level"].value_counts())

        # SHAP on a sample
        explainer = shap.Explainer(model, scaler.transform(df.drop(columns=["Churn Probability","Risk Level"]).head(50)))
        shap_values = explainer(scaler.transform(df.drop(columns=["Churn Probability","Risk Level"]).head(50)))
        shap_importance = pd.DataFrame({
            "Feature": df.drop(columns=["Churn Probability","Risk Level"]).columns,
            "Mean |SHAP|": np.abs(shap_values.values).mean(axis=0)
        }).sort_values(by="Mean |SHAP|", ascending=False).head(5)

        st.subheader("Top Global Churn Drivers")
        st.table(shap_importance)

        if GEMINI_KEY:
            with st.spinner("🤖 Executive Report from Gemini..."):
                model_g = genai.GenerativeModel("gemini-2.0-flash")
                prompt = f"""
                Portfolio churn risk summary: {summary}
                Top churn drivers: {shap_importance.to_dict(orient='records')}
                👉 Write an executive briefing:
                - Portfolio-level churn trends
                - Most vulnerable segments
                - 3 global retention actions
                """
                response = model_g.generate_content(prompt)
                st.subheader("🤖 Gemini Executive Report")
                st.write(response.text)
    else:
        st.info("📂 Please upload a dataset or add customers to view executive insights.")

# ── RM Dashboard Tab ────────────────────────────────────────────────────────
with tab2:
    st.header("📋 Relationship Manager Dashboard")
    if len(st.session_state["customer_data"]) > 0:
        df = st.session_state["customer_data"]
        top_customers = df.sort_values(by="Churn Probability", ascending=False).head(50)
        st.subheader("Top 50 At-Risk Customers")
        st.dataframe(top_customers)

        if GEMINI_KEY:
            with st.spinner("🤖 RM Playbook from Gemini..."):
                model_g = genai.GenerativeModel("gemini-2.0-flash")
                prompt = f"""
                Top at-risk customers: {top_customers[['Churn Probability','Risk Level']].to_dict(orient='records')}
                👉 Generate a playbook:
                - How should RMs prioritize?
                - Suggested actions/scripts
                """
                response = model_g.generate_content(prompt)
                st.subheader("🤖 RM Playbook")
                st.write(response.text)
    else:
        st.info("📂 Please upload a dataset or add customers to view RM dashboard.")

# ── Campaign Generator Tab ──────────────────────────────────────────────────
with tab3:
    st.header("📢 Campaign Generator")
    if len(st.session_state["customer_data"]) > 0:
        df = st.session_state["customer_data"]
        summary = df["Risk Level"].value_counts(normalize=True).mul(100).round(2).to_dict()

        if GEMINI_KEY:
            with st.spinner("🤖 Generating Campaigns..."):
                model_g = genai.GenerativeModel("gemini-2.0-flash")
                prompt = f"""
                Customer churn segmentation: {summary}
                👉 Create campaign drafts:
                - High risk: win-back offers
                - Medium risk: engagement offers
                - Low risk: cross-sell/upsell offers
                Provide sample email + SMS copy.
                """
                response = model_g.generate_content(prompt)
                st.subheader("🤖 Campaign Drafts")
                st.write(response.text)
    else:
        st.info("📂 Please upload a dataset or add customers to view campaigns.")

# ── Add Single Customer Tab ─────────────────────────────────────────────────
with tab4:
    st.header("➕ Add Single Customer")

    with st.form("add_customer_form"):
        geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
        gender = st.selectbox("👤 Gender", label_encoder_gender.classes_)
        age = st.slider("🎂 Age", 18, 80, 35)
        credit_score = st.slider("💳 Credit Score", 300, 850, 650)
        balance = st.number_input("💰 Balance ($)", min_value=0.0, max_value=200000.0, value=5000.0, step=500.0)
        estimated_salary = st.number_input("💵 Estimated Salary ($)", min_value=10000.0, max_value=300000.0, value=60000.0, step=1000.0)
        tenure = st.slider("📆 Tenure (years)", 0, 10, 3)
        num_products = st.slider("🛍️ Number of Products", 1, 4, 1)
        has_card = st.radio("💳 Has Credit Card?", [0,1])
        active = st.radio("✅ Active Member?", [0,1])
        submit_btn = st.form_submit_button("Predict & Save")

    if submit_btn:
        new_data = pd.DataFrame({
            "CreditScore":[credit_score],
            "Gender":[label_encoder_gender.transform([gender])[0]],
            "Age":[age],
            "Tenure":[tenure],
            "Balance":[balance],
            "NumOfProducts":[num_products],
            "HasCrCard":[has_card],
            "IsActiveMember":[active],
            "EstimatedSalary":[estimated_salary]
        })

        # Encode geography
        geo_encoded = onehot_encoder_geo.transform([[geography]])
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))
        new_data = pd.concat([new_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Scale + Predict
        new_scaled = scaler.transform(new_data)
        churn_proba = model.predict(new_scaled)[0][0]
        risk_level = "Low" if churn_proba < 0.3 else "Medium" if churn_proba < 0.7 else "High"

        new_data["Churn Probability"] = churn_proba
        new_data["Risk Level"] = risk_level

        # Append to session dataset
        st.session_state["customer_data"] = pd.concat([st.session_state["customer_data"], new_data], ignore_index=True)

        # Persist to CSV
        csv_path = "Churn_Modelling.csv"
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, new_data], ignore_index=True)
        else:
            combined_df = new_data
        combined_df.to_csv(csv_path, index=False)

        st.success(f"Prediction complete! Churn Probability: {churn_proba:.2%} ({risk_level} Risk)")
        st.dataframe(new_data)

# ── Footer ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
if len(st.session_state["customer_data"]) > 0:
    st.sidebar.download_button(
        "⬇️ Download Results CSV",
        st.session_state["customer_data"].to_csv(index=False),
        "batch_predictions.csv",
        "text/csv"
    )

st.markdown("---")
st.caption("© 2025 NovaDhruv Financial — AI-Powered ChurnShield")