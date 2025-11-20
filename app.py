import streamlit as st
import pandas as pd
import pickle
import google.generativeai as genai

# -------------------------------
# Load ML Model
# -------------------------------
model = pickle.load(open("ads_predictor.pkl", "rb"))

# -------------------------------
# Configure Gemini API
# -------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Gemini function
def get_gemini_suggestions(caption, engagement_score):
    prompt = f"""
I have a social media post with the following:
Caption: {caption}
Predicted Engagement Score: {engagement_score}

Provide:
1. Explanation
2. Suggestions
3. Improved caption
4. Hashtags
"""
    gemini_model = genai.GenerativeModel("gemini-pro-latest")
    response = gemini_model.generate_content(prompt)
    return response.text

# Predict + Gemini combined
def predict_with_gemini(caption, account_name, platform,
                        comment_count, like_count, caption_length,
                        word_count, sentiment_score):

    new_data = pd.DataFrame([{
        "caption": caption,
        "account_name": account_name,
        "platform": platform,
        "comment_count": comment_count,
        "like_count": like_count,
        "caption_length": caption_length,
        "word_count": word_count,
        "sentiment_score": sentiment_score
    }])

    predicted_score = model.predict(new_data)[0]
    ai_output = get_gemini_suggestions(caption, predicted_score)

    return predicted_score, ai_output


# -------------------------------
# Streamlit UI
# -------------------------------

st.title("📈 AI Engagement Predictor + Gemini Assistant")

caption = st.text_area("Post Caption")
account_name = st.text_input("Account Name")
platform = st.selectbox("Platform", ["Instagram", "Facebook", "Twitter", "LinkedIn"])
comment_count = st.number_input("Comment Count", min_value=0)
like_count = st.number_input("Like Count", min_value=0)
sentiment_score = st.slider("Sentiment Score", 0.0, 1.0, 0.5)

if st.button("Predict & Generate Suggestions"):
    if caption.strip() == "":
        st.warning("Caption cannot be empty!")
    else:
        caption_length = len(caption)
        word_count = len(caption.split())

        score, analysis = predict_with_gemini(
            caption, account_name, platform,
            comment_count, like_count,
            caption_length, word_count,
            sentiment_score
        )

        st.subheader("📊 Predicted Engagement Score")
        st.write(score)

        st.subheader("🤖 Gemini AI Suggestions")
        st.markdown(analysis)