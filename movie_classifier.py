import torch
import torch.nn as nn

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