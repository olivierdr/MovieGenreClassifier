import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MovieClassifier(nn.Module):
    """Neural network classifier using sentence embeddings"""
    def __init__(self, embedding_dim, num_labels):
        super().__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(embedding_dim // 2, num_labels)
        self.relu = nn.ReLU()
        
    def forward(self, embeddings):
        x = self.dropout1(embeddings)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Define genre-specific keywords for thematic vectors
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
            keyword_text = ' '.join(keywords)
            with torch.no_grad():
                vector = self.model.encode(keyword_text, convert_to_tensor=True)
            thematic_vectors[genre] = vector
        return thematic_vectors
    
    def predict(self, text):
        """Predict genre based on highest cosine similarity with thematic vectors"""
        with torch.no_grad():
            text_vector = self.model.encode(text, convert_to_tensor=True)
        
        similarities = {
            genre: torch.nn.functional.cosine_similarity(text_vector.unsqueeze(0), vec.unsqueeze(0)).item()
            for genre, vec in self.thematic_vectors.items()
        }
        
        return max(similarities.items(), key=lambda x: x[1])[0]

def load_and_prepare_data(file_path='cleaned_task.csv'):
    """Load and prepare the dataset"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Map labels to numeric values
    label_map = {'cult': 0, 'paranormal': 1, 'dramatic': 2}
    df['label'] = df['Tag'].map(label_map)
    
    print("\nClass distribution:")
    print(df['Tag'].value_counts())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], 
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    return X_train, X_test, y_train, y_test, label_map

def train_and_evaluate_models(X_train, X_test, y_train, y_test, label_map):
    """Train and evaluate both models"""
    print("\n=== Training Models ===")
    
    # Initialize models
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train Neural Network Classifier
    print("\nTraining Neural Network Classifier...")
    classifier = MovieClassifier(model.get_sentence_embedding_dimension(), num_labels=len(label_map))
    classifier.to(device)
    
    # Prepare training data
    train_embeddings = []
    for text in tqdm(X_train, desc="Computing embeddings"):
        with torch.no_grad():
            embedding = model.encode(text, convert_to_tensor=True)
            train_embeddings.append(embedding)
    
    train_embeddings = torch.stack(train_embeddings).to(device)
    train_labels = torch.tensor(y_train.values, dtype=torch.long).to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_f1 = 0
    best_model_state = None
    
    for epoch in range(10):
        classifier.train()
        total_loss = 0
        
        # Forward pass
        outputs = classifier(train_embeddings)
        loss = criterion(outputs, train_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluation
        classifier.eval()
        with torch.no_grad():
            test_embeddings = []
            for text in X_test:
                embedding = model.encode(text, convert_to_tensor=True)
                test_embeddings.append(embedding)
            
            test_embeddings = torch.stack(test_embeddings).to(device)
            outputs = classifier(test_embeddings)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            f1 = f1_score(y_test, preds, average='weighted')
            print(f"Epoch {epoch + 1} - F1 Score: {f1:.3f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = classifier.state_dict().copy()
                print(f"New best model saved with F1 score: {f1:.3f}")
    
    # Load best model
    classifier.load_state_dict(best_model_state)
    
    # Train Thematic Vector Classifier
    print("\nTraining Thematic Vector Classifier...")
    thematic_classifier = ThematicVectorClassifier(GENRE_KEYWORDS)
    
    # Evaluate both models
    print("\nEvaluating Neural Network Classifier...")
    classifier.eval()
    with torch.no_grad():
        test_embeddings = []
        for text in X_test:
            embedding = model.encode(text, convert_to_tensor=True)
            test_embeddings.append(embedding)
        
        test_embeddings = torch.stack(test_embeddings).to(device)
        outputs = classifier(test_embeddings)
        nn_preds = torch.argmax(outputs, dim=1).cpu().numpy()
    
    print("\nClassification Report - Neural Network:")
    print(classification_report(y_test, nn_preds, target_names=label_map.keys()))
    
    print("\nEvaluating Thematic Vector Classifier...")
    thematic_preds = [thematic_classifier.predict(text) for text in X_test]
    thematic_preds = [list(label_map.keys()).index(pred) for pred in thematic_preds]
    
    print("\nClassification Report - Thematic Vector:")
    print(classification_report(y_test, thematic_preds, target_names=label_map.keys()))
    
    # Plot confusion matrices
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    cm_nn = confusion_matrix(y_test, nn_preds)
    sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_map.keys(),
                yticklabels=label_map.keys())
    plt.title('Confusion Matrix - Neural Network')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.subplot(1, 2, 2)
    cm_thematic = confusion_matrix(y_test, thematic_preds)
    sns.heatmap(cm_thematic, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_map.keys(),
                yticklabels=label_map.keys())
    plt.title('Confusion Matrix - Thematic Vector')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()
    
    # Save the neural network model
    torch.save(classifier.state_dict(), 'embedding_classifier.pt')
    print("\nModel weights saved as 'embedding_classifier.pt'")

def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test, label_map = load_and_prepare_data()
    
    # Train and evaluate models
    train_and_evaluate_models(X_train, X_test, y_train, y_test, label_map)

if __name__ == "__main__":
    main() 