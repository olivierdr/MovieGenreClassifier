import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Define genre-specific keywords
GENRE_KEYWORDS = {
    'dramatic': {
        "tragedy", "betrayal", "family", "love", "loss", "conflict", "emotion",
        "drama", "relationship", "struggle", "death", "past", "affair", "sacrifice"
    },
    'paranormal': {
        "ghost", "spirit", "haunted", "demon", "possessed", "supernatural",
        "medium", "exorcism", "curse", "psychic", "apparition", "entity", "ouija"
    },
    'cult': {
        "bizarre", "underground", "iconic", "midnight", "weird", "classic",
        "fanbase", "counterculture", "retro", "quirky", "experimental", "campy",
        "nonconformist", "satire", "stylized"
    }
}

class KeywordBasedClassifier:
    """Classifier using keyword density scores"""
    def __init__(self, genre_keywords):
        self.genre_keywords = genre_keywords
        
    def preprocess_text(self, text):
        """Convert text to lowercase and split into words"""
        return set(re.findall(r'\b\w+\b', text.lower()))
    
    def calculate_keyword_density(self, text):
        """Calculate keyword density scores for each genre"""
        words = self.preprocess_text(text)
        total_words = len(words)
        if total_words == 0:
            return {genre: 0.0 for genre in self.genre_keywords}
            
        scores = {}
        for genre, keywords in self.genre_keywords.items():
            # Count how many keywords appear in the text
            keyword_count = sum(1 for keyword in keywords if keyword in words)
            # Calculate density score
            scores[genre] = keyword_count / total_words
            
        return scores
    
    def predict(self, text):
        """Predict genre based on highest keyword density"""
        scores = self.calculate_keyword_density(text)
        return max(scores.items(), key=lambda x: x[1])[0]

class ThematicVectorClassifier:
    """Classifier using thematic vector similarity"""
    def __init__(self, genre_keywords, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.genre_keywords = genre_keywords
        self.thematic_vectors = self._compute_thematic_vectors()
        
    def _compute_thematic_vectors(self):
        """Compute average embedding vectors for each genre's keywords"""
        thematic_vectors = {}
        for genre, keywords in self.genre_keywords.items():
            # Convert keywords to a single string
            keyword_text = ' '.join(keywords)
            # Get embedding for the keywords
            with torch.no_grad():
                vector = self.model.encode(keyword_text, convert_to_tensor=True)
            thematic_vectors[genre] = vector
        return thematic_vectors
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        return torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
    
    def predict(self, text):
        """Predict genre based on highest cosine similarity with thematic vectors"""
        # Get embedding for the input text
        with torch.no_grad():
            text_vector = self.model.encode(text, convert_to_tensor=True)
        
        # Calculate similarities with each genre's thematic vector
        similarities = {
            genre: self.cosine_similarity(text_vector, vec).item()
            for genre, vec in self.thematic_vectors.items()
        }
        
        return max(similarities.items(), key=lambda x: x[1])[0]

def evaluate_classifiers(df, keyword_classifier, thematic_classifier):
    """Evaluate both classifiers and compare their performance"""
    print("\n=== Evaluating Keyword-based Classifier ===")
    keyword_preds = [keyword_classifier.predict(text) for text in tqdm(df['cleaned_text'])]
    print("\nClassification Report:")
    print(classification_report(df['Tag'], keyword_preds))
    
    # Plot confusion matrix for keyword classifier
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(df['Tag'], keyword_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=GENRE_KEYWORDS.keys(),
                yticklabels=GENRE_KEYWORDS.keys())
    plt.title('Confusion Matrix - Keyword-based Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_keywords.png')
    plt.close()
    
    print("\n=== Evaluating Thematic Vector Classifier ===")
    thematic_preds = [thematic_classifier.predict(text) for text in tqdm(df['cleaned_text'])]
    print("\nClassification Report:")
    print(classification_report(df['Tag'], thematic_preds))
    
    # Plot confusion matrix for thematic classifier
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(df['Tag'], thematic_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=GENRE_KEYWORDS.keys(),
                yticklabels=GENRE_KEYWORDS.keys())
    plt.title('Confusion Matrix - Thematic Vector Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_thematic.png')
    plt.close()
    
    # Compare predictions
    agreement = sum(1 for k, t in zip(keyword_preds, thematic_preds) if k == t)
    print(f"\nAgreement between classifiers: {agreement/len(df):.2%}")

def analyze_keyword_distribution(df, keyword_classifier):
    """Analyze the distribution of keyword scores for each genre"""
    print("\n=== Analyzing Keyword Distribution ===")
    
    # Calculate scores for all texts
    all_scores = []
    for text in tqdm(df['cleaned_text']):
        scores = keyword_classifier.calculate_keyword_density(text)
        all_scores.append(scores)
    
    # Convert to DataFrame
    scores_df = pd.DataFrame(all_scores)
    scores_df['true_label'] = df['Tag']
    
    # Plot score distributions
    plt.figure(figsize=(15, 5))
    for i, genre in enumerate(GENRE_KEYWORDS.keys(), 1):
        plt.subplot(1, 3, i)
        for label in GENRE_KEYWORDS.keys():
            sns.kdeplot(data=scores_df[scores_df['true_label'] == label], 
                       x=genre, label=label)
        plt.title(f'{genre.capitalize()} Keyword Scores')
        plt.xlabel('Score')
        plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig('keyword_score_distribution.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('cleaned_task.csv')
    
    # Print class distribution
    print("\nClass distribution:")
    print(df['Tag'].value_counts())
    
    # Initialize classifiers
    keyword_classifier = KeywordBasedClassifier(GENRE_KEYWORDS)
    thematic_classifier = ThematicVectorClassifier(GENRE_KEYWORDS)
    
    # Evaluate classifiers
    evaluate_classifiers(df, keyword_classifier, thematic_classifier)
    
    # Analyze keyword distribution
    analyze_keyword_distribution(df, keyword_classifier)

if __name__ == "__main__":
    main() 