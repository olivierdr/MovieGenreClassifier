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

# Set page config
st.set_page_config(
    page_title="Movie Genre Classifier Demo",
    page_icon="🎬",
    layout="wide"
)

# Initialize classifiers
@st.cache_resource
def load_classifiers():
    # Load TF-IDF model and vectorizer
    tfidf_clf = joblib.load('tfidf_logistic_model.joblib')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    
    # Load embedding model
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    try:
        embedding_clf = MovieClassifier(embedding_model.get_sentence_embedding_dimension(), 3)
        embedding_clf.load_state_dict(torch.load('embedding_classifier.pt'))
        embedding_clf.eval()
    except Exception as e:
        st.error("Erreur lors du chargement du modèle embedding (embedding_classifier.pt) : " + str(e))
        raise
    
    # Initialize keyword-based classifiers
    keyword_classifier = KeywordBasedClassifier(GENRE_KEYWORDS)
    thematic_classifier = ThematicVectorClassifier(GENRE_KEYWORDS)
    
    return {
        'tfidf': (tfidf_clf, tfidf_vectorizer),
        'embedding': (embedding_model, embedding_clf),
        'keyword': keyword_classifier,
        'thematic': thematic_classifier
    }

# Load all classifiers
classifiers = load_classifiers()
tfidf_clf, tfidf_vectorizer = classifiers['tfidf']
embedding_model, embedding_clf = classifiers['embedding']
keyword_classifier = classifiers['keyword']
thematic_classifier = classifiers['thematic']

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
    # Create four columns for results
    results_col1, results_col2, results_col3, results_col4 = st.columns(4)
    
    with results_col1:
        st.subheader("📊 TF-IDF Analysis")
        # Transform text and get prediction
        text_tfidf = tfidf_vectorizer.transform([text_to_analyze])
        tfidf_probs = tfidf_clf.predict_proba(text_tfidf)[0]
        tfidf_pred = tfidf_clf.classes_[np.argmax(tfidf_probs)]
        
        # Display probabilities as a bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        genres = list(GENRE_KEYWORDS.keys())
        ax.bar(genres, tfidf_probs)
        ax.set_title("TF-IDF Prediction Probabilities")
        ax.set_ylim(0, 1)
        for i, v in enumerate(tfidf_probs):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        st.pyplot(fig)
        plt.close()
        
        #st.markdown(f"**Predicted Genre:** {tfidf_pred.capitalize()}")
    
    with results_col2:
        st.subheader("🧠 Neural Network Analysis")
        # Get embedding and prediction
        with torch.no_grad():
            text_embedding = embedding_model.encode(text_to_analyze, convert_to_tensor=True)
            logits = embedding_clf(text_embedding.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0]
            pred = torch.argmax(probs).item()
        
        # Display probabilities as a bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        genres = list(GENRE_KEYWORDS.keys())
        ax.bar(genres, probs.cpu().numpy())
        ax.set_title("Neural Network Prediction Probabilities")
        ax.set_ylim(0, 1)
        for i, v in enumerate(probs):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        st.pyplot(fig)
        plt.close()
        
        #st.markdown(f"**Predicted Genre:** {genres[pred].capitalize()}")
    
    with results_col3:
        st.subheader("🔍 Keyword-based Analysis")
        # Calculate keyword scores
        scores = keyword_classifier.calculate_keyword_density(text_to_analyze)
        
        # Display scores as a bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        genres = list(scores.keys())
        values = list(scores.values())
        ax.bar(genres, values)
        ax.set_title("Keyword Density Scores")
        ax.set_ylim(0, max(values) * 1.1)
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        st.pyplot(fig)
        plt.close()
        
        # Display prediction
        keyword_pred = keyword_classifier.predict(text_to_analyze)
        #st.markdown(f"**Predicted Genre:** {keyword_pred.capitalize()}")
        
        # Display found keywords
        st.markdown("**Keywords found in text:**")
        words = keyword_classifier.preprocess_text(text_to_analyze)
        for genre, keywords in GENRE_KEYWORDS.items():
            found = [k for k in keywords if k in words]
            if found:
                #st.markdown(f"- **{genre.capitalize()}:** {', '.join(found)}")
                pass
    
    with results_col4:
        st.subheader("🎯 Thematic Vector Analysis")
        # Calculate similarities
        with st.spinner("Computing thematic similarities..."):
            with torch.no_grad():
                text_vector = thematic_classifier.model.encode(text_to_analyze, convert_to_tensor=True)
                similarities = {
                    genre: thematic_classifier.cosine_similarity(text_vector, vec).item()
                    for genre, vec in thematic_classifier.thematic_vectors.items()
                }
        
        # Display similarities as a bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        genres = list(similarities.keys())
        values = list(similarities.values())
        ax.bar(genres, values)
        ax.set_title("Thematic Similarity Scores")
        ax.set_ylim(0, 1)
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        st.pyplot(fig)
        plt.close()
        
        # Display prediction
        thematic_pred = thematic_classifier.predict(text_to_analyze)
        #st.markdown(f"**Predicted Genre:** {thematic_pred.capitalize()}")
    
    # Display agreement
    st.markdown("---")
    predictions = {
        'TF-IDF': tfidf_pred,
        'Neural Network': genres[pred],
        'Keyword-based': keyword_pred,
        'Thematic Vector': thematic_pred
    }
    
    # Count occurrences of each prediction

    # pred_counts = {}
    # for pred in predictions.values():
    #     pred_counts[pred] = pred_counts.get(pred, 0) + 1
    
    # # Find the most common prediction
    # most_common = max(pred_counts.items(), key=lambda x: x[1])
    
    # if most_common[1] == 4:
    #     st.success(f"✅ All classifiers agree on the genre: {most_common[0].capitalize()}")
    # elif most_common[1] == 3:
    #     st.warning(f"⚠️ Three classifiers predict {most_common[0].capitalize()}, while one disagrees")
    # elif most_common[1] == 2:
    #     st.warning(f"⚠️ Two classifiers predict {most_common[0].capitalize()}, while two disagree")
    # else:
    #     st.error("❌ All classifiers disagree on the genre")
    
    # # Show detailed predictions
    # st.markdown("**Detailed predictions:**")
    # for model, pred in predictions.items():
    #     st.markdown(f"- {model}: {pred.capitalize()}")

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