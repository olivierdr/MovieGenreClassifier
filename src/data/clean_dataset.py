import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging():
    """Set up logging for data cleaning process."""
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"data_cleaning_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def load_and_explore_data(file_path='data/raw/task.csv'):
    df = pd.read_csv(file_path)
    logging.info("STARTING CLEANING DATA PROCESS")
    print("=" * 59)
    logging.info(f"Number of rows: {len(df)}")
    logging.info(f"Number of columns: {len(df.columns)}")
    logging.info(f"Columns: {list(df.columns)}")
    return df

def check_missing_values(df):
    if df.isnull().sum().sum() == 0:
        logging.info("No missing values found in the dataset.")
    else:
        missing = df.isnull().sum()
        logging.info("Missing values found:")
        logging.info(missing[missing > 0].to_string())

def check_duplicates(df):
    title_dupes = df[df.duplicated(subset=['Title'], keep=False)]
    synopsis_dupes = df[df.duplicated(subset=['Synopsis'], keep=False)]
    logging.info(f"Found {len(title_dupes)} rows with duplicate titles")
    logging.info(f"Found {len(synopsis_dupes)} rows with duplicate synopses")
    return title_dupes, synopsis_dupes

def analyze_text_length(df):
    df['synopsis_length'] = df['Synopsis'].str.len()
    mean_length = df['synopsis_length'].mean()
    logging.info(f"Mean synopsis length (characters): {int(mean_length)}")
    
    # Create basic plot
    plot_dir = Path("outputs/analysis")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='synopsis_length', bins=50)
    plt.title('Synopsis Length Distribution')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.savefig(plot_dir / 'text_length_distribution.png')
    plt.close()

def analyze_labels(df):
    label_dist = df['Tag'].value_counts()
    label_percentages = (label_dist / len(df) * 100).round(2)
    logging.info("Label Distribution: " + ", ".join([f"{k}: {v} ({label_percentages[k]}%)" for k, v in label_dist.items()]))
    
    # Create basic plot
    plot_dir = Path("outputs/analysis")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    label_dist.plot(kind='bar')
    plt.title('Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_dir / 'label_distribution.png')
    plt.close()
    
    return label_dist

def calculate_similarity(df, threshold=0.8):
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    tfidf_matrix = vectorizer.fit_transform(df['Synopsis'])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    similar_pairs = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if similarity_matrix[i, j] > threshold:
                similar_pairs.append((i, j))

    logging.info(f"Similarity check - Found {len(similar_pairs)} pairs of similar synopses (threshold: {threshold})")
    return similar_pairs

def handle_title_duplicates(df, similarity_matrix, threshold=0.8):
    """
    Handle cases where the same title has multiple synopses.
    If synopses are similar, keep the longest one.
    If not similar, mark as ambiguous.
    """
    df_processed = df.copy()
    df_processed['ambiguous'] = False
    
    # Get all duplicate titles
    title_groups = df[df.duplicated(subset=['Title'], keep=False)].groupby('Title')
    
    for title, group in title_groups:
        if len(group) == 1:
            continue
            
        # Get indices of rows with this title
        indices = group.index.tolist()
        
        # Check if any pair of synopses is similar
        is_similar = False
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if similarity_matrix[indices[i], indices[j]] > threshold:
                    is_similar = True
                    break
            if is_similar:
                break
        
        if is_similar:
            # Keep the longest synopsis
            longest_idx = group['Synopsis'].str.len().idxmax()
            df_processed.loc[indices, 'ambiguous'] = True
            df_processed.loc[longest_idx, 'ambiguous'] = False
        else:
            # Mark all as ambiguous if synopses are not similar
            df_processed.loc[indices, 'ambiguous'] = True
    
    return df_processed

def handle_synopsis_duplicates(df):
    """
    Handle cases where the same synopsis has multiple titles.
    If tags are the same, keep the first row.
    If tags are different, mark as ambiguous.
    """
    df_processed = df.copy()
    
    # Get all duplicate synopses
    synopsis_groups = df[df.duplicated(subset=['Synopsis'], keep=False)].groupby('Synopsis')
    
    for synopsis, group in synopsis_groups:
        if len(group) == 1:
            continue
            
        # Get indices of rows with this synopsis
        indices = group.index.tolist()
        
        # Check if all tags are the same
        if group['Tag'].nunique() == 1:
            # Keep the first row, mark others as ambiguous
            df_processed.loc[indices[1:], 'ambiguous'] = True
        else:
            # Mark all as ambiguous if tags are different
            df_processed.loc[indices, 'ambiguous'] = True
    
    return df_processed

def clean_dataset(df, similar_pairs):
    """
    Clean the dataset according to the specified strategy:
    1. For same titles with multiple synopses:
       - If similar, keep longest synopsis
       - If not similar, mark as ambiguous
    2. For same synopses with multiple titles:
       - If same tags, keep first row
       - If different tags, mark as ambiguous
    """
    # Calculate similarity matrix for title duplicate handling
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    tfidf_matrix = vectorizer.fit_transform(df['Synopsis'])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Initialize with all rows not ambiguous
    df_cleaned = df.copy()
    df_cleaned['ambiguous'] = False
    
    # Handle title duplicates
    df_cleaned = handle_title_duplicates(df_cleaned, similarity_matrix)
    
    # Handle synopsis duplicates
    df_cleaned = handle_synopsis_duplicates(df_cleaned)
    
    # Split into clean and ambiguous datasets
    df_clean = df_cleaned[~df_cleaned['ambiguous']].drop('ambiguous', axis=1)
    df_ambiguous = df_cleaned[df_cleaned['ambiguous']].drop('ambiguous', axis=1)
    
    logging.info("CLEANING COMPLETE")
    print("=" * 59)
    logging.info(f"Original dataset size: {len(df)} rows")
    logging.info(f"Clean dataset size: {len(df_clean)} rows")
    logging.info(f"Ambiguous dataset size: {len(df_ambiguous)} rows")
    print("=" * 59)
    
    return df_clean, df_ambiguous

def main():
    log_file = setup_logging()

    try:
        df = load_and_explore_data()
        check_missing_values(df)
        check_duplicates(df)
        analyze_text_length(df)
        analyze_labels(df)
        similar_pairs = calculate_similarity(df)
        df_clean, df_ambiguous = clean_dataset(df, similar_pairs)

        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        clean_path = output_dir / "movies_cleaned.csv"
        ambiguous_path = output_dir / "movies_ambiguous.csv"
        df_clean.to_csv(clean_path, index=False)
        df_ambiguous.to_csv(ambiguous_path, index=False)

        logging.info(f"- Cleaned dataset saved to {clean_path}")
        logging.info(f"- Ambiguous dataset saved to {ambiguous_path}")

    except Exception as e:
        logging.error(f"An error occurred during data cleaning: {str(e)}")
        raise

if __name__ == "__main__":
    main()
