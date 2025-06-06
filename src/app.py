import streamlit as st
import joblib
import pandas as pd
import os
import torch
from pathlib import Path
import sys

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.model import EmbeddingClassifier

os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"

# Page configuration
st.set_page_config(page_title="Movie Genre Prediction", layout="wide")
st.title("Movie Genre Prediction")

# Load models
@st.cache_resource
def load_models():
    # Define model paths
    model_dir = Path("outputs/models")
    tfidf_model_path = model_dir / "movie_tfidf_model.joblib"
    embedding_model_path = model_dir / "embedding_model.pth"
    
    # Load TF-IDF and label_map
    tfidf_data = joblib.load(tfidf_model_path)
    tfidf_model = tfidf_data['tfidf']
    label_map = tfidf_data['label_map']
    
    # Load embedding model
    embedding_model = EmbeddingClassifier(384, len(label_map))
    embedding_model.load_state_dict(torch.load(embedding_model_path, map_location=embedding_model.device))
    embedding_model.eval()
    
    return tfidf_model, embedding_model, label_map

# User interface
st.write("""
Enter a movie synopsis to predict its genre using two different models:
1. TF-IDF Model (based on words)
2. Embedding Model (based on meaning)
""")

# Load models
try:
    tfidf_model, embedding_model, label_map = load_models()
    
    # Text input for synopsis
    synopsis = st.text_area("Movie Synopsis", height=200)
    
    if synopsis:
        # Predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("TF-IDF Prediction")
            tfidf_pred = tfidf_model.predict(synopsis)
            tfidf_probs = tfidf_model.predict_proba(synopsis)
            
            # Display prediction
            predicted_genre = list(label_map.keys())[tfidf_pred]
            st.write(f"Predicted genre: **{predicted_genre}**")
            
            # Display probabilities
            st.write("Probabilities:")
            for genre, prob in zip(label_map.keys(), tfidf_probs):
                st.write(f"- {genre}: {prob:.2%}")
        
        with col2:
            st.subheader("Embedding Prediction")
            embedding_pred = embedding_model.predict(synopsis)
            embedding_probs = embedding_model.predict_proba(synopsis)
            
            # Display prediction
            predicted_genre = list(label_map.keys())[embedding_pred]
            st.write(f"Predicted genre: **{predicted_genre}**")
            
            # Display probabilities
            st.write("Probabilities:")
            for genre, prob in zip(label_map.keys(), embedding_probs):
                st.write(f"- {genre}: {prob:.2%}")
    
except Exception as e:
    st.error(f"Error while loading models: {str(e)}")
    st.write("Make sure you have run train.py to train and save the models.")
