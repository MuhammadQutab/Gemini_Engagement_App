import streamlit as st
import pickle
import pandas as pd
import os
import google.generativeai as genai


# ================================
# 🔧 Load RandomForest ML Model
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ads_predictor.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ================================
# 🔐 Setup Gemini API
# ================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is missing. Add it in Streamlit → Settings → Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
client = genai.GenerativeModel("gemini-pro-latest")


# ================================
# 🎨 Streamlit UI
# ================================
st.set_page_config(page_title="Social Media Engagement Predictor", layout="wide")

st.title("📊 Social Media Engagement Predictor")


# ================================
# 🧩 Feature Inputs (Original Model)
# ================================
st.subheader("📝 Enter Post Details")

col1, col2 = st.columns(2)

with col1:
    account_name = st.text_input("Account Name:")
    caption = st.text_area("Caption Text:")

    platform = st.selectbox(
        "Platform:",
        ["Instagram", "Facebook", "YouTube", "TikTok", "Twitter (X)"]
    )

with col2:
    caption_length = st.number_input("Caption Length (characters):", min_value=0)
    word_count = st.number_input("Word Count:", min_value=0)
    sentiment_score = st.number_input("Sentiment Score (-1 to 1):", min_value=-1.0, max_value=1.0, step=0.01)
    like_count = st.number_input("Like Count:", min_value=0)
    comment_count = st.number_input("Comment Count:", min_value=0)


# ================================
# 📌 Prepare DataFrame for Model
# ================================
def prepare_input():
    return pd.DataFrame([{
        "account_name": account_name,
        "caption": caption,
        "caption_length": caption_length,
        "word_count": word_count,
        "sentiment_score": sentiment_score,
        "like_count": like_count,
        "comment_count": comment_count,
        "platform": platform
    }])


# ================================
# 🔮 Prediction
# ================================
if st.button("Predict Engagement", type="primary"):
    df = prepare_input()

    try:
        prediction = model.predict(df)[0]
        st.success(f"✅ **Predicted Engagement Score:** {prediction}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")


# ================================
# 🤖 Gemini AI Section
# ================================
st.markdown("---")
st.header("💬 Ask Gemini Anything")

user_prompt = st.text_area("Type your question for Gemini:")

if st.button("Ask Gemini"):
    if user_prompt.strip():
        with st.spinner("Thinking..."):
            response = client.generate_content(user_prompt)
            st.write(response.text)
    else:
        st.warning("Please enter a question before submitting.")