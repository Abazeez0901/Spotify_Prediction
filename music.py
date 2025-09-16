import pandas as pd
import streamlit as st
import pickle as pkl
import streamlit as st
import numpy as np


model = pkl.load(open("spotify_model.pkl", "rb"))
encoder = pkl.load(open("encoder.pkl", "rb"))

st.title("🎵 Spotify Popularity Prediction")


year = st.number_input("Enter Song Year", min_value=1900, max_value=2100, step=1)
language = st.text_input("Enter Language (e.g., English, Hindi, Tamil)")

if st.button("Predict Popularity"):
    try:
        
        lang_encoded = encoder.transform([language])[0]
        
        
        input_data = np.array([[year, lang_encoded]])
        
       
        prediction = model.predict(input_data)
        st.success(f"Predicted Popularity: {prediction[0]:.2f}")
        st.balloons()   
    except Exception as e:
        st.error(f"Error: {e}")
        st.balloons

