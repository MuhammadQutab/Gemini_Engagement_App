🚀 Gemini-Powered Social Media Engagement Predictor 
Machine Learning + Google Gemini API + Streamlit Deployment

This project predicts engagement performance for social media posts and generates AI-powered recommendations (improved captions, hashtags, suggestions) using Google Gemini Pro.

Built as part of a remote internship project.


🔥 Features
1. ML-Based Engagement Prediction

- Trained RandomForest model

- Uses post metadata (likes, comments, sentiment, caption length)

- Returns an engagement score

2. Google Gemini-Powered Enhancements

- Explains predicted performance

- Suggests improvements

- Generates a refined caption

- Produces optimized hashtags

3. Streamlit Web App

Users can:

- Enter a caption

- Provide post metadata

- View predicted engagement

- Get AI-generated recommendations instantly

📁 Project Structure 

Gemini_Engagement_App/
│

├── app.py                 # Main Streamlit application

├── ads_predictor.pkl      # Trained ML model 

├── requirements.txt       # Dependencies for deployment 
 
├── .gitignore             # Ignore sensitive files 

└── .streamlit/

      └── secrets.toml     # API key (NOT uploaded to GitHub) 