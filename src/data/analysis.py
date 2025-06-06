import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
def setup_logging():
    """Set up logging to both file and console."""
    # Create logs directory if it doesn't exist
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"data_analysis_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

# Display configuration
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8')

def load_and_explore_data(file_path):
    """Load and explore initial data"""
    logging.info("Loading data...")
    df = pd.read_csv(file_path)
    
    logging.info("\n=== Basic Information ===")
    logging.info(f"Number of rows: {len(df)}")
    logging.info(f"Number of columns: {len(df.columns)}")
    logging.info("\nColumn names:")
    logging.info(df.columns.tolist())
    
    logging.info("\n=== Data Preview ===")
    logging.info(df.head())
    
    return df

def check_missing_values(df):
    """Analyze missing values"""
    logging.info("\n=== Missing Values Analysis ===")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    logging.info(missing_info[missing_info['Missing Values'] > 0])
    
    return missing_info

def check_duplicates(df):
    """Detect and analyze duplicates"""
    logging.info("\n=== Duplicate Analysis ===")
    duplicates = df.duplicated()
    logging.info(f"Number of duplicates: {duplicates.sum()}")
    logging.info(f"Duplicate percentage: {(duplicates.sum() / len(df)) * 100:.2f}%")
    
    if duplicates.sum() > 0:
        logging.info("\nExample of duplicates:")
        logging.info(df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist()).head())
    
    return duplicates

def analyze_text_length(df, text_column, output_dir):
    """Analyze text lengths"""
    logging.info(f"\n=== Text Length Analysis ({text_column}) ===")
    df['text_length'] = df[text_column].str.len()
    
    logging.info("\nText length statistics:")
    logging.info(df['text_length'].describe())
    
    # Create output directory for plots
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize length distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='text_length', bins=50)
    plt.title('Text Length Distribution')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.savefig(plots_dir / 'text_length_distribution.png')
    plt.close()

def analyze_labels(df, label_column, output_dir):
    """Analyze label distribution"""
    logging.info(f"\n=== Label Distribution Analysis ({label_column}) ===")
    label_counts = df[label_column].value_counts()
    logging.info("\nLabel distribution:")
    logging.info(label_counts)
    logging.info("\nPercentages:")
    logging.info((label_counts / len(df) * 100).round(2))
    
    # Create output directory for plots
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize label distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=label_column)
    plt.title('Label Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / 'label_distribution.png')
    plt.close()

def clean_text(text):
    """Clean text data"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return text

def handle_duplicate_titles(df, title_column='Title', synopsis_column='Synopsis'):
    """Handle duplicate movie titles by keeping the most informative synopsis"""
    print("\n=== Handling Duplicate Movie Titles ===")
    
    # Find duplicate titles
    duplicate_titles = df[df.duplicated(subset=[title_column], keep=False)]
    print(f"Found {len(duplicate_titles)} rows with duplicate titles")
    
    if len(duplicate_titles) > 0:
        print("\nExample of duplicate titles:")
        print(duplicate_titles[[title_column, synopsis_column]].head(10))
        
        # Group by title and keep the row with the longest synopsis
        df['synopsis_length'] = df[synopsis_column].str.len()
        df = df.sort_values('synopsis_length', ascending=False)
        df = df.drop_duplicates(subset=[title_column], keep='first')
        df = df.drop('synopsis_length', axis=1)
        
        print(f"\nAfter handling duplicates: {len(df)} rows remaining")
    
    return df

def main():
    # Set up logging
    log_file = setup_logging()
    logging.info("Starting data analysis")
    
    # Define paths
    input_file = "data/raw/task.csv"
    output_dir = "outputs/analysis"
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_and_explore_data(input_file)
    
    # Check missing values
    missing_info = check_missing_values(df)
    
    # Check duplicates
    duplicates = check_duplicates(df)
    
    # Remove exact duplicates if present
    if duplicates.sum() > 0:
        logging.info("\nRemoving exact duplicates...")
        df = df.drop_duplicates()
        logging.info(f"New dataset size: {len(df)}")
    
    # Handle duplicate movie titles
    df = handle_duplicate_titles(df)
    
    # Analyze text length
    text_column = 'Synopsis'
    analyze_text_length(df, text_column, output_dir)
    
    # Analyze label distribution
    label_column = 'Tag'
    analyze_labels(df, label_column, output_dir)
    
    # Clean texts
    logging.info("\n=== Text Cleaning ===")
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # Save cleaned dataset
    output_file = output_path / "cleaned_data.csv"
    df.to_csv(output_file, index=False)
    logging.info(f"\nCleaned dataset saved to {output_file}")
    logging.info(f"Analysis log saved to {log_file}")

if __name__ == "__main__":
    main()