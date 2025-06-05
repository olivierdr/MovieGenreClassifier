import streamlit as st
import joblib
import pandas as pd

# Configuration de la page
st.set_page_config(page_title="Prédiction de Genre de Film", layout="wide")
st.title("Prédiction de Genre de Film")

# Charger les modèles
@st.cache_resource
def load_models():
    models = joblib.load('movie_models.joblib')
    return models['tfidf'], models['embedding'], models['label_map']

# Interface utilisateur
st.write("""
Entrez le synopsis d'un film pour prédire son genre en utilisant deux modèles différents :
1. Modèle TF-IDF (basé sur les mots)
2. Modèle Embedding (basé sur le sens)
""")

# Charger les modèles
try:
    tfidf_model, embedding_model, label_map = load_models()
    
    # Zone de texte pour le synopsis
    synopsis = st.text_area("Synopsis du film", height=200)
    
    if synopsis:
        # Prédictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prédiction TF-IDF")
            tfidf_pred = tfidf_model.predict(synopsis)
            tfidf_probs = tfidf_model.predict_proba(synopsis)
            
            # Afficher la prédiction
            predicted_genre = list(label_map.keys())[tfidf_pred]
            st.write(f"Genre prédit : **{predicted_genre}**")
            
            # Afficher les probabilités
            st.write("Probabilités :")
            for genre, prob in zip(label_map.keys(), tfidf_probs):
                st.write(f"- {genre}: {prob:.2%}")
        
        with col2:
            st.subheader("Prédiction Embedding")
            embedding_pred = embedding_model.predict(synopsis)
            embedding_probs = embedding_model.predict_proba(synopsis)
            
            # Afficher la prédiction
            predicted_genre = list(label_map.keys())[embedding_pred]
            st.write(f"Genre prédit : **{predicted_genre}**")
            
            # Afficher les probabilités
            st.write("Probabilités :")
            for genre, prob in zip(label_map.keys(), embedding_probs):
                st.write(f"- {genre}: {prob:.2%}")
    
except Exception as e:
    st.error(f"Erreur lors du chargement des modèles : {str(e)}")
    st.write("Assurez-vous d'avoir exécuté train.py pour entraîner et sauvegarder les modèles.") 