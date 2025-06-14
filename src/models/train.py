import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.model import TFIDFClassifier, EmbeddingClassifier
from src.models.evaluation import evaluate_model, compare_models

def load_data(file_path='data/processed/movies_cleaned.csv'):
    """Load and prepare the data"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Label mapping
    label_map = {'cult': 0, 'paranormal': 1, 'dramatic': 2}
    df['label'] = df['Tag'].map(label_map)
    
    print("\nClass distribution:")
    print(df['Tag'].value_counts())
    
    return df['Synopsis'], df['label'], label_map

def train_models():
    """Train and save the models"""
    # Load data
    texts, labels, label_map = load_data()
    label_names = list(label_map.keys())
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Calculate class weights
    class_counts = pd.Series(y_train).value_counts()
    total_samples = len(y_train)
    class_weights = {i: total_samples / (len(class_counts) * count) 
                    for i, count in class_counts.items()}
    class_weights = [class_weights[i] for i in range(len(class_counts))]
    
    # Create output directory for models
    model_dir = Path("outputs/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store model metrics
    all_metrics = {}
    
    # Train TF-IDF model
    print("\nTraining TF-IDF model...")
    tfidf_model = TFIDFClassifier()
    tfidf_model.fit(X_train, y_train)
    
    # Get TF-IDF predictions
    tfidf_preds = tfidf_model.predict(X_test)
    
    # Evaluate TF-IDF model
    tfidf_metrics = evaluate_model(
        y_test, tfidf_preds, label_names,
        model_name="TF-IDF"
    )
    all_metrics['TF-IDF'] = tfidf_metrics
    
    # Train Embedding model
    print("\nTraining Embedding model...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    embedding_model = EmbeddingClassifier(384, len(label_map), class_weights=class_weights)
    optimizer = torch.optim.AdamW(embedding_model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    
    # Convert labels to tensor
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
    
    # Training loop
    best_f1 = 0
    for epoch in range(10):  # Increased epochs
        embedding_model.train()
        total_loss = 0
        
        # Forward pass
        outputs = embedding_model(X_train.tolist())
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(embedding_model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        # Evaluation
        embedding_model.eval()
        with torch.no_grad():
            test_outputs = embedding_model(X_test.tolist())
            test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
            
            # Evaluate each epoch
            epoch_metrics = evaluate_model(
                y_test, test_preds, label_names,
                model_name=f"Embedding_Epoch_{epoch+1}"
            )
            all_metrics[f'Embedding_Epoch_{epoch+1}'] = epoch_metrics
            
            # DEBUG: Print keys to find correct F1 key
            if epoch == 0:
                print('DEBUG epoch_metrics keys:', epoch_metrics.keys())
            
            # Update learning rate based on F1 score
            current_f1 = epoch_metrics['classification_report']['weighted avg']['f1-score']
            scheduler.step(current_f1)
            
            # Save best model
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_model_state = embedding_model.state_dict()
    
    # Load best model for final evaluation
    embedding_model.load_state_dict(best_model_state)
    embedding_model.eval()
    with torch.no_grad():
        test_outputs = embedding_model(X_test.tolist())
        test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
        
        embedding_metrics = evaluate_model(
            y_test, test_preds, label_names,
            model_name="Embedding"
        )
        all_metrics['Embedding'] = embedding_metrics
    
    # Compare models
    print("\nComparing models...")
    comparison_df = compare_models(all_metrics, "outputs/evaluation")
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Save models
    print("\nSaving models...")
    
    # Save TF-IDF model and label map
    tfidf_model_path = model_dir / "movie_tfidf_model.joblib"
    joblib.dump({
        'tfidf': tfidf_model,
        'label_map': label_map,
        'metrics': tfidf_metrics
    }, tfidf_model_path)
    
    # Save PyTorch model weights and metrics
    embedding_model_path = model_dir / "embedding_model.pth"
    torch.save({
        'model_state_dict': embedding_model.state_dict(),
        'metrics': embedding_metrics
    }, embedding_model_path)
    
    print(f"Models and metrics saved to {model_dir}")

if __name__ == "__main__":
    train_models() 