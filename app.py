import streamlit as st
import pickle
import pandas as pd
import os
import google.generativeai as genai


# ============================================
# 🔧 Load ML Model (RandomForest)
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ads_predictor.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error("❌ Failed to load ML model. Make sure 'ads_predictor.pkl' is inside the repo.")
    st.stop()


# ============================================
# 🔐 Load Gemini API Key (from Streamlit Secrets)
# ============================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY missing. Add it in Streamlit → Settings → Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


# ============================================
# 🎨 Streamlit UI Setup
# ============================================
st.set_page_config(page_title="Ad Engagement Predictor + Gemini Chat", page_icon="🤖")
st.title("🤖 Ad Engagement Predictor + Gemini AI Assistant")


# ============================================
# 🧮 User Inputs for ML Model
# ============================================
st.subheader("📊 Enter Values for Engagement Prediction")

# Example inputs (you must change according to your dataset!)
col1, col2 = st.columns(2)

impressions = col1.number_input("Impressions", min_value=0, value=1000)
clicks = col2.number_input("Clicks", min_value=0, value=50)

spend = col1.number_input("Ad Spend ($)", min_value=0.0, value=10.0)
reach = col2.number_input("Reach", min_value=0, value=500)


# Predict Button
if st.button("🔮 Predict Engagement"):
    try:
        input_df = pd.DataFrame({
            "impressions": [impressions],
            "clicks": [clicks],
            "spend": [spend],
            "reach": [reach]
        })

        prediction = model.predict(input_df)[0]
        st.success(f"📈 Predicted Engagement Score: **{prediction}**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")


# ============================================
# 💬 Gemini Chat Section
# ============================================
st.subheader("🤖 Ask Gemini Anything")

user_prompt = st.text_input("Type your question for Gemini:")

if user_prompt:
    try:
        response = gemini_model.generate_content(user_prompt)
        st.write("### Gemini’s Answer:")
        st.write(response.text)

    except Exception as e:
        st.error(f"❌ Gemini generation failed: {e}")