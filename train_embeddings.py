import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings('ignore')

class SmartTruncator:
    """Class to handle smart text truncation at sentence boundaries"""
    def __init__(self, model, max_length=256):
        self.model = model
        self.max_length = max_length
        
    def truncate_at_sentence(self, text):
        """Truncate text at the last complete sentence before max_length"""
        tokens = self.model.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.max_length:
            return text
            
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_text = ""
        current_tokens = []
        
        for sentence in sentences:
            if current_text:
                sentence = " " + sentence
            new_tokens = self.model.tokenizer.encode(sentence, add_special_tokens=False)
            if len(current_tokens) + len(new_tokens) <= self.max_length:
                current_text += sentence
                current_tokens.extend(new_tokens)
            else:
                break
                
        return current_text.strip()

class MovieDataset(Dataset):
    """Custom Dataset for loading movie synopses and labels"""
    def __init__(self, texts, labels, model, max_length=256):
        self.texts = texts
        self.labels = labels
        self.model = model
        self.truncator = SmartTruncator(model, max_length)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        text = self.truncator.truncate_at_sentence(text)
        
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_tensor=True)
        
        return {
            'embedding': embedding,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class MovieClassifier(nn.Module):
    """Neural network classifier using sentence embeddings"""
    def __init__(self, embedding_dim, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embedding_dim, num_labels)
        
    def forward(self, embeddings):
        embeddings = self.dropout(embeddings)
        logits = self.classifier(embeddings)
        return logits

def load_and_prepare_data(file_path='cleaned_task.csv'):
    """Load and prepare the dataset"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Map labels to numeric values
    label_map = {'cult': 0, 'paranormal': 1, 'dramatic': 2}
    df['label'] = df['Tag'].map(label_map)
    
    # Print class distribution
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

def train_and_evaluate(X_train, X_test, y_train, y_test, label_map, 
                      model_name='sentence-transformers/all-MiniLM-L6-v2', 
                      batch_size=32,
                      epochs=5):
    """Train and evaluate sentence transformer-based classifier"""
    print("\n=== Training Classifier ===")
    
    # Initialize model
    model = SentenceTransformer(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = MovieDataset(X_train.tolist(), y_train.tolist(), model)
    test_dataset = MovieDataset(X_test.tolist(), y_test.tolist(), model)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize classifier
    classifier = MovieClassifier(model.get_sentence_embedding_dimension(), num_labels=len(label_map))
    classifier.to(device)
    
    # Setup training
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4)
    
    # Setup loss function with class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    print("\nTraining progress:")
    classifier.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        
        for batch in progress_bar:
            embeddings = batch['embedding'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
    
    # Evaluation
    print("\nEvaluating model...")
    classifier.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embedding'].to(device)
            labels = batch['labels']
            
            outputs = classifier(embeddings)
            preds = torch.argmax(outputs, dim=1).cpu()
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    # Print results
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_map.keys()))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_map.keys(),
                yticklabels=label_map.keys())
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test, label_map = load_and_prepare_data()
    
    # Train and evaluate model
    train_and_evaluate(X_train, X_test, y_train, y_test, label_map)

if __name__ == "__main__":
    main() 