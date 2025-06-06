import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os
from pathlib import Path
from datetime import datetime

# Set up logging
def setup_logging():
    """Set up logging to both file and console."""
    # Create logs directory if it doesn't exist
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"data_cleaning_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    return log_file

def calculate_similarity(texts):
    """Calculate cosine similarity between texts using TF-IDF."""
    if len(texts) <= 1:
        return np.array([[1.0]])
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf_matrix)

def clean_dataset(input_file, output_dir, similarity_threshold=0.7):
    """
    Clean the dataset according to the specified rules.
    
    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Directory to save processed datasets
        similarity_threshold (float): Threshold for considering synopses similar (0-1)
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading dataset from {input_file}")
    df = pd.read_csv(input_file)
    
    # Add ambiguous column
    df['ambiguous'] = False
    
    # Process duplicate titles
    logging.info("Processing duplicate titles...")
    title_groups = df.groupby('Title')
    
    for title, group in title_groups:
        if len(group) > 1:
            synopses = group['Synopsis'].tolist()
            similarities = calculate_similarity(synopses)
            
            # Check if all synopses are similar
            all_similar = np.all(similarities >= similarity_threshold)
            
            if all_similar:
                # Keep the longest synopsis
                longest_idx = group['Synopsis'].str.len().idxmax()
                df.loc[group.index, 'ambiguous'] = True
                df.loc[longest_idx, 'ambiguous'] = False
            else:
                # Mark all entries as ambiguous
                df.loc[group.index, 'ambiguous'] = True
    
    # Process duplicate synopses
    logging.info("Processing duplicate synopses...")
    synopsis_groups = df.groupby('Synopsis')
    
    for synopsis, group in synopsis_groups:
        if len(group) > 1:
            # Check if all tags are the same
            if group['Tag'].nunique() == 1:
                # Keep the first row
                df.loc[group.index[1:], 'ambiguous'] = True
            else:
                # Mark all entries as ambiguous
                df.loc[group.index, 'ambiguous'] = True
    
    # Save the full dataset with ambiguous column
    full_output_path = output_path / "movies_checked_ambiguous.csv"
    logging.info(f"Saving full dataset with ambiguous column to {full_output_path}")
    df.to_csv(full_output_path, index=False)
    
    # Save the cleaned dataset (keeping ambiguous column)
    cleaned_df = df[~df['ambiguous']]
    cleaned_output_path = output_path / "movies_cleaned.csv"
    logging.info(f"Saving cleaned dataset to {cleaned_output_path}")
    cleaned_df.to_csv(cleaned_output_path, index=False)
    
    # Save only the ambiguous rows
    ambiguous_df = df[df['ambiguous']]
    ambiguous_output_path = output_path / "movies_ambiguous.csv"
    logging.info(f"Saving ambiguous rows to {ambiguous_output_path}")
    ambiguous_df.to_csv(ambiguous_output_path, index=False)
    
    # Log statistics
    total_rows = len(df)
    cleaned_rows = len(cleaned_df)
    removed_rows = total_rows - cleaned_rows
    ambiguous_rows = df['ambiguous'].sum()
    
    stats = {
        "total_rows": total_rows,
        "cleaned_rows": cleaned_rows,
        "ambiguous_rows": ambiguous_rows,
        "removed_rows": removed_rows,
        "removed_percentage": (removed_rows/total_rows)*100,
        "similarity_threshold": similarity_threshold
    }
    
    logging.info("Dataset cleaning complete:")
    logging.info(f"Total rows: {stats['total_rows']}")
    logging.info(f"Cleaned rows: {stats['cleaned_rows']}")
    logging.info(f"Ambiguous rows: {stats['ambiguous_rows']}")
    logging.info(f"Removed rows: {stats['removed_rows']} ({stats['removed_percentage']:.2f}%)")
    
    return stats

if __name__ == "__main__":
    # Set up logging
    log_file = setup_logging()
    logging.info("Starting data cleaning process")
    
    # Define paths
    input_file = "data/raw/task.csv"
    output_dir = "data/processed"
    
    # Run cleaning
    stats = clean_dataset(input_file, output_dir)
    
    logging.info(f"Log file saved to: {log_file}")
    logging.info("Data cleaning process completed successfully") 