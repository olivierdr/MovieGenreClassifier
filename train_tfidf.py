import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path='cleaned_task.csv'):
    """Load and prepare the dataset"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Convert labels to numeric values
    label_map = {'cult': 0, 'paranormal': 1, 'dramatic': 2}
    df['label'] = df['Tag'].map(label_map)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], 
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    return X_train, X_test, y_train, y_test, label_map

def train_and_evaluate(X_train, X_test, y_train, y_test, label_map):
    """Train and evaluate TF-IDF + Logistic Regression model"""
    print("\n=== Training TF-IDF + Logistic Regression Model ===")
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_tfidf)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_map.keys()))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_map.keys(),
                yticklabels=label_map.keys())
    plt.title('Confusion Matrix - TF-IDF + Logistic Regression')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_tfidf.png')
    plt.close()
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return clf, vectorizer

def save_models(clf, vectorizer):
    """Save the trained models"""
    print("\nSaving models...")
    joblib.dump(clf, 'tfidf_logistic_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    print("Models saved successfully!")

def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test, label_map = load_and_prepare_data()
    
    # Train and evaluate model
    clf, vectorizer = train_and_evaluate(X_train, X_test, y_train, y_test, label_map)
    
    # Save models
    save_models(clf, vectorizer)

if __name__ == "__main__":
    main() 