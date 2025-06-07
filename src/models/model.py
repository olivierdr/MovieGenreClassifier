from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import numpy as np

class TFIDFClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
    
    def fit(self, texts, labels):
        """Fit the model on training data."""
        # Convert texts to list if it's a Series
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        
        # Fit and transform the texts
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        return self
    
    def predict(self, texts):
        """Predict labels for texts."""
        # Convert texts to list if it's a Series
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Transform and predict
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)
    
    def predict_proba(self, texts):
        """Predict probabilities for texts."""
        # Convert texts to list if it's a Series
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Transform and predict probabilities
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, class_weights=None):
        super().__init__()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.class_weights = torch.tensor(class_weights, device=self.device) if class_weights is not None else None
        self.to(self.device)
    
    def forward(self, texts):
        """Forward pass through the model."""
        # Get embeddings
        with torch.no_grad():
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
            embeddings = embeddings.to(self.device)
        
        # Classify
        return self.classifier(embeddings)
    
    def predict(self, texts):
        """Predict labels for texts."""
        self.eval()
        with torch.no_grad():
            outputs = self(texts)
            return torch.argmax(outputs, dim=1).cpu().numpy()
    
    def predict_proba(self, texts):
        """Predict probabilities for texts."""
        self.eval()
        with torch.no_grad():
            outputs = self(texts)
            return torch.softmax(outputs, dim=1).cpu().numpy() 