import streamlit as st
import pandas as pd
from keyword_based_classifier import KeywordBasedClassifier, ThematicVectorClassifier, GENRE_KEYWORDS
import matplotlib.pyplot as plt
import seaborn as sns
import io
import torch
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from embedding_model import MovieClassifier
from movie_predictor import MovieGenrePredictor

# Set page config
st.set_page_config(
    page_title="Movie Genre Classifier Demo",
    page_icon="🎬",
    layout="wide"
)

# Initialize classifiers
@st.cache_resource
def load_predictor():
    predictor = MovieGenrePredictor.load_models("movie_predictor.joblib")
    return predictor

predictor = load_predictor()

# Title and description
st.title("🎬 Movie Genre Classifier Demo")
st.markdown("""
This demo application uses four different methods to classify movie synopses into genres:
1. **TF-IDF + Logistic Regression**: Uses word frequencies and a linear classifier
2. **Neural Network with Embeddings**: Uses sentence embeddings and a neural network
3. **Keyword-based Classification**: Uses genre-specific keywords to calculate a density score
4. **Thematic Vector Classification**: Uses sentence embeddings to compare with genre-specific thematic vectors

Try it out by either:
- Pasting a movie synopsis in the text area below
- Uploading a text file containing a synopsis
""")

# Create two columns for input methods
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Text Input")
    text_input = st.text_area("Enter a movie synopsis:", height=200)

# Process input
text_to_analyze = None
if text_input:
    text_to_analyze = text_input

if text_to_analyze:
    st.subheader("Prédictions des modèles (TF‑IDF, embedding, thematic) sur le synopsis :")
    predictions = predictor.predict(text_to_analyze)
    col1, col2, col3 = st.columns(3)
    with col1:
         st.subheader("TF‑IDF")
         st.write("Prédiction : " + predictions["tfidf"]["prediction"])
         st.write("Scores : " + str(predictions["tfidf"]["scores"]))
    with col2:
         st.subheader("Embedding (MovieClassifier)")
         st.write("Prédiction : " + predictions["embedding"]["prediction"])
         st.write("Scores : " + str(predictions["embedding"]["scores"]))
    with col3:
         st.subheader("Thematic Vector")
         st.write("Prédiction : " + predictions["thematic"]["prediction"])

# Add information about the classifiers
with st.expander("ℹ️ About the Classifiers"):
    st.markdown("""
    ### How the Classifiers Work
    
    1. **TF-IDF + Logistic Regression:**
       - Converts text into TF-IDF features (word frequencies)
       - Uses a logistic regression model to predict the genre
       - Good at capturing word importance and frequency
    
    2. **Neural Network with Embeddings:**
       - Uses sentence embeddings to capture semantic meaning
       - Employs a neural network for classification
       - Good at understanding context and relationships
    
    3. **Keyword-based Classifier:**
       - Counts how many genre-specific keywords appear in the text
       - Calculates a density score (keywords found / total words)
       - Predicts the genre with the highest density score
    
    4. **Thematic Vector Classifier:**
       - Uses sentence embeddings to create thematic vectors
       - Compares text similarity with genre-specific themes
       - Good at capturing overall thematic elements
    
    ### Keyword Lists Used
    
    **Dramatic Keywords:**
    - tragedy, betrayal, family, love, loss, conflict, emotion
    - drama, relationship, struggle, death, past, affair, sacrifice
    
    **Paranormal Keywords:**
    - ghost, spirit, haunted, demon, possessed, supernatural
    - medium, exorcism, curse, psychic, apparition, entity, ouija
    
    **Cult Keywords:**
    - bizarre, underground, iconic, midnight, weird, classic
    - fanbase, counterculture, retro, quirky, experimental, campy
    - nonconformist, satire, stylized
    """) 