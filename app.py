import streamlit as st
import pickle
import pandas as pd
import os
from google import genai

# ================================
# 🔧 Load RandomForest ML Model
# ================================

# Automatically detect correct path on Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ads_predictor.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Could not load ML model: {e}")
    st.stop()


# ================================
# 🔧 Setup Gemini API
# ================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # From Streamlit Secrets

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is missing. Add it in Streamlit → Settings → Secrets.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)
gemini_model = client

# Gemini helper function
def get_gemini_suggestions(caption, engagement_score):
    prompt = f"""
    I have a social media post:

    Caption: {caption}
    Predicted Engagement Score: {engagement_score}

    Provide:
    1. Explanation of the score
    2. Suggestions to improve engagement
    3. An improved caption
    4. 5 relevant hashtags
    """

    try:
        response = gemini_model.models.generate_content(
            model="gemini-pro-latest",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"❌ Gemini API Error: {e}"


# ================================
# 🔥 Streamlit UI
# ================================

st.title("📈 Social Media Engagement Predictor + Gemini AI Assistant")
st.write("Predict engagement and improve your captions using ML + Gemini AI.")

caption = st.text_area("✍️ Enter your caption:")
account_name = st.text_input("Account Name:", "Google")
platform = st.selectbox("Platform:", ["Instagram", "Facebook", "Twitter"])
comment_count = st.number_input("Comment Count:", min_value=0, value=10)
like_count = st.number_input("Like Count:", min_value=0, value=100)
sentiment_score = st.slider("Sentiment Score:", 0.0, 1.0, 0.5)

if st.button("🚀 Predict Engagement + Get AI Recommendations"):

    if not caption.strip():
        st.warning("Please enter a caption first.")
        st.stop()

    # Prepare DataFrame for ML model
    new_data = pd.DataFrame([{
        "caption": caption,
        "account_name": account_name,
        "platform": platform,
        "comment_count": comment_count,
        "like_count": like_count,
        "caption_length": len(caption),
        "word_count": len(caption.split()),
        "sentiment_score": sentiment_score
    }])

    # ML prediction
    try:
        engagement_score = model.predict(new_data)[0]
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
        st.stop()

    st.subheader("📊 Predicted Engagement Score")
    st.success(f"**{engagement_score:.2f}**")

    # Gemini AI analysis
    st.subheader("🤖 Gemini AI Suggestions")

    analysis = get_gemini_suggestions(caption, engagement_score)

    st.write(analysis)