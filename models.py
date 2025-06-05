from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import numpy as np

class TFIDFClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
        
    def fit(self, texts, labels):
        # Transformer les textes en vecteurs TF-IDF
        X = self.vectorizer.fit_transform(texts)
        # Entraîner le classifieur
        self.classifier.fit(X, labels)
        
    def predict(self, text):
        # Transformer le texte en vecteur TF-IDF
        X = self.vectorizer.transform([text])
        # Prédire la classe
        return self.classifier.predict(X)[0]
    
    def predict_proba(self, text):
        # Transformer le texte en vecteur TF-IDF
        X = self.vectorizer.transform([text])
        # Retourner les probabilités
        return self.classifier.predict_proba(X)[0]

class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim, num_labels):
        super().__init__()
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim // 2, num_labels)
        ).to(self.device)
        
    def forward(self, text):
        # Générer les embeddings
        with torch.no_grad():
            embeddings = self.embedding_model.encode(text, convert_to_tensor=True)
            embeddings = embeddings.to(self.device)
        # Passer à travers le classifieur
        return self.classifier(embeddings)
    
    def predict(self, text):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(text)
            return torch.argmax(outputs).item()
    
    def predict_proba(self, text):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(text)
            probs = torch.softmax(outputs, dim=0)
            return probs.cpu().numpy() 