import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
from pathlib import Path
import logging
from datetime import datetime

def setup_evaluation_logging():
    """Set up logging for model evaluation."""
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"model_evaluation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def plot_confusion_matrix(y_true, y_pred, labels, output_dir, model_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / f'confusion_matrix_{model_name.lower()}.png')
    plt.close()
    
    return cm

def compare_models(metrics_dict, output_dir):
    """Compare models and save comparison metrics."""
    comparison = {}
    
    # Compare accuracy and F1 scores
    for model_name, metrics in metrics_dict.items():
        report = metrics['classification_report']
        comparison[model_name] = {
            'accuracy': report['accuracy'],
            'macro_avg_f1': report['macro avg']['f1-score'],
            'weighted_avg_f1': report['weighted avg']['f1-score']
        }
        
        # Add per-class F1 scores
        for label in report.keys():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                comparison[model_name][f'{label}_f1'] = report[label]['f1-score']
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison).T
    
    # Save comparison
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    comparison_df.to_csv(metrics_dir / 'model_comparison.csv')
    
    # Save as JSON
    with open(metrics_dir / 'model_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=4)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    comparison_df[['accuracy', 'macro_avg_f1', 'weighted_avg_f1']].plot(kind='bar')
    plt.title('Model Comparison - Overall Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / 'model_comparison.png')
    plt.close()
    
    return comparison_df

def save_metrics(metrics, output_dir, model_name):
    """Save evaluation metrics to JSON file."""
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = metrics_dir / f'metrics_{model_name.lower()}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics_file

def evaluate_model(y_true, y_pred, labels, output_dir="outputs/evaluation", model_name="Model"):
    """
    Model evaluation focusing on confusion matrix and classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        output_dir: Directory to save outputs
        model_name: Name of the model for plots and files
    """
    # Set up logging
    log_file = setup_evaluation_logging()
    logging.info(f"Starting evaluation for {model_name}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get classification report
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    logging.info(f"\nClassification Report - {model_name}:")
    logging.info(classification_report(y_true, y_pred, target_names=labels))
    
    # Plot confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred, labels, output_path, model_name)
    logging.info(f"\nConfusion Matrix - {model_name}:")
    logging.info(cm)
    
    # Compile metrics
    metrics = {
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    # Save metrics
    metrics_file = save_metrics(metrics, output_path, model_name)
    logging.info(f"\nMetrics saved to {metrics_file}")
    logging.info(f"Evaluation log saved to {log_file}")
    
    return metrics 