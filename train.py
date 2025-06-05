import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from models import TFIDFClassifier, EmbeddingClassifier

def load_data(file_path='cleaned_task.csv'):
    """Charge et prépare les données"""
    print("Chargement des données...")
    df = pd.read_csv(file_path)
    
    # Mapping des labels
    label_map = {'cult': 0, 'paranormal': 1, 'dramatic': 2}
    df['label'] = df['Tag'].map(label_map)
    
    print("\nDistribution des classes:")
    print(df['Tag'].value_counts())
    
    return df['cleaned_text'], df['label'], label_map

def train_models():
    """Entraîne et sauvegarde les modèles"""
    # Charger les données
    texts, labels, label_map = load_data()
    
    # Séparer en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Entraîner le modèle TF-IDF
    print("\nEntraînement du modèle TF-IDF...")
    tfidf_model = TFIDFClassifier()
    tfidf_model.fit(X_train, y_train)
    
    # Évaluer le modèle TF-IDF
    tfidf_preds = [tfidf_model.predict(text) for text in X_test]
    print("\nRapport de classification - TF-IDF:")
    print(classification_report(y_test, tfidf_preds, target_names=label_map.keys()))
    
    # Entraîner le modèle Embedding
    print("\nEntraînement du modèle Embedding...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")
    
    embedding_model = EmbeddingClassifier(384, len(label_map))  # 384 est la dimension des embeddings de MiniLM
    optimizer = torch.optim.AdamW(embedding_model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Convertir les labels en tensor
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
    
    # Boucle d'entraînement
    for epoch in range(5):
        embedding_model.train()
        total_loss = 0
        
        # Forward pass
        outputs = embedding_model(X_train.tolist())
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Évaluation
        embedding_model.eval()
        with torch.no_grad():
            test_outputs = embedding_model(X_test.tolist())
            test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
            print(f"\nEpoch {epoch + 1} - Rapport de classification:")
            print(classification_report(y_test, test_preds, target_names=label_map.keys()))
    
    # Sauvegarder les modèles
    print("\nSauvegarde des modèles...")
    models = {
        'tfidf': tfidf_model,
        'embedding': embedding_model,
        'label_map': label_map
    }
    joblib.dump(models, 'movie_models.joblib')
    print("Modèles sauvegardés dans 'movie_models.joblib'")

if __name__ == "__main__":
    train_models() 