import streamlit as st
import numpy as np
import sys
import os
import tempfile

# ✅ Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import extract_features
from src.model_predict import load_model, predict_genre

# Genre labels (edit this according to your model)
genre_list = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']


# Streamlit UI
st.title("🎵 Music Genre Classifier")
st.write("Upload a music file (mp3 or wav) and get its genre prediction.")

uploaded_file = st.file_uploader("Choose a file", type=["wav", "mp3"])

if uploaded_file is not None:
    with st.spinner("Processing..."):
        try:
            # ✅ Save uploaded file to a temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                temp_audio_file.write(uploaded_file.read())
                temp_audio_path = temp_audio_file.name
            
            # ✅ Preprocess the saved temp file
            features = extract_features(temp_audio_path)
            # features = np.expand_dims(features, axis=0)  # Add batch dimension
            st.write("Shape of features:", features.shape)

            # Predict
            model = load_model("best_model.h5")
            if model is None:
                st.error("Model not loaded properly.")
                
            print(model.input_shape)
            # print(model.summary())
            predicted_genre = predict_genre(model, features, genre_list)
            
            st.success(f"🎧 Predicted Genre: **{predicted_genre}**")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
