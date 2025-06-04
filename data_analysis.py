import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Display configuration
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8')

def load_and_explore_data(file_path):
    """Load and explore initial data"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    print("\n=== Basic Information ===")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumn names:")
    print(df.columns.tolist())
    
    print("\n=== Data Preview ===")
    print(df.head())
    
    return df

def check_missing_values(df):
    """Analyze missing values"""
    print("\n=== Missing Values Analysis ===")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    print(missing_info[missing_info['Missing Values'] > 0])
    
    return missing_info

def check_duplicates(df):
    """Detect and analyze duplicates"""
    print("\n=== Duplicate Analysis ===")
    duplicates = df.duplicated()
    print(f"Number of duplicates: {duplicates.sum()}")
    print(f"Duplicate percentage: {(duplicates.sum() / len(df)) * 100:.2f}%")
    
    if duplicates.sum() > 0:
        print("\nExample of duplicates:")
        print(df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist()).head())
    
    return duplicates

def analyze_text_length(df, text_column):
    """Analyze text lengths"""
    print(f"\n=== Text Length Analysis ({text_column}) ===")
    df['text_length'] = df[text_column].str.len()
    
    print("\nText length statistics:")
    print(df['text_length'].describe())
    
    # Visualize length distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='text_length', bins=50)
    plt.title('Text Length Distribution')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.savefig('text_length_distribution.png')
    plt.close()

def analyze_labels(df, label_column):
    """Analyze label distribution"""
    print(f"\n=== Label Distribution Analysis ({label_column}) ===")
    label_counts = df[label_column].value_counts()
    print("\nLabel distribution:")
    print(label_counts)
    print("\nPercentages:")
    print((label_counts / len(df) * 100).round(2))
    
    # Visualize label distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=label_column)
    plt.title('Label Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('label_distribution.png')
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
    # Load data
    df = load_and_explore_data('task.csv')
    
    # Check missing values
    missing_info = check_missing_values(df)
    
    # Check duplicates
    duplicates = check_duplicates(df)
    
    # Remove exact duplicates if present
    if duplicates.sum() > 0:
        print("\nRemoving exact duplicates...")
        df = df.drop_duplicates()
        print(f"New dataset size: {len(df)}")
    
    # Handle duplicate movie titles
    df = handle_duplicate_titles(df)
    
    # Analyze text length
    text_column = 'Synopsis'
    analyze_text_length(df, text_column)
    
    # Analyze label distribution
    label_column = 'Tag'
    analyze_labels(df, label_column)
    
    # Clean texts
    print("\n=== Text Cleaning ===")
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # Save cleaned dataset
    df.to_csv('cleaned_task.csv', index=False)
    print("\nCleaned dataset saved to 'cleaned_task.csv'")

if __name__ == "__main__":
    main()