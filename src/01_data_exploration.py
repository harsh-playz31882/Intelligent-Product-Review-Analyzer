
# Data Exploration : Understanding the structure



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

def load_and_explore_data():
    print("=== Loading Dataset ===")
    df = pd.read_csv('Reviews.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df

def analyze_dataset_info(df):
    """Check basic information about the dataset"""
    print("\n=== Dataset Information ===")
    print("Dataset Info:")
    df.info()
    
    print("\nMissing values:")
    print(df.isnull().sum())

def analyze_score_distribution(df):
    """Analyze and visualize the Score distribution"""
    print("\n=== Score Distribution Analysis ===")
    print("Score Distribution:")
    print(df['Score'].value_counts().sort_index())
    
    # Visualize the distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Score')
    plt.title('Distribution of Review Scores')
    plt.xlabel('Score (1-5)')
    plt.ylabel('Count')
    plt.savefig('results/score_distribution.png')
    plt.show()

def convert_to_sentiment(df):
    """Convert scores to sentiment classes"""
    print("\n=== Converting Scores to Sentiment ===")
    
    # 1-2: Negative, 3: Neutral, 4-5: Positive
    def score_to_sentiment(score):
        if score <= 2:
            return 'Negative'
        elif score == 3:
            return 'Neutral'
        else:
            return 'Positive'
    
    df['Sentiment'] = df['Score'].apply(score_to_sentiment)
    
    print("Sentiment Distribution:")
    print(df['Sentiment'].value_counts())
    
    # Visualize sentiment distribution
    plt.figure(figsize=(8, 6))
    df['Sentiment'].value_counts().plot(kind='bar')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig('results/sentiment_distribution.png')
    plt.show()
    
    return df

def analyze_text_length(df):
    """Analyze review text and summary length"""
    print("\n=== Text Length Analysis ===")
    
    df['Text_Length'] = df['Text'].str.len()
    df['Summary_Length'] = df['Summary'].str.len()
    
    print("Review text statistics:")
    print(df['Text_Length'].describe())
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['Text_Length'], bins=50, alpha=0.7)
    plt.title('Distribution of Review Text Length')
    plt.xlabel('Character Count')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(df['Summary_Length'], bins=30, alpha=0.7, color='orange')
    plt.title('Distribution of Summary Length')
    plt.xlabel('Character Count')
    
    plt.tight_layout()
    plt.savefig('results/text_length_distribution.png')
    plt.show()

def sample_reviews_by_sentiment(df):
    """Display sample reviews for each sentiment"""
    print("\n=== Sample Reviews by Sentiment ===")
    
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        print(f"\n=== {sentiment} Reviews ===")
        sample_reviews = df[df['Sentiment'] == sentiment]['Text'].head(3)
        for i, review in enumerate(sample_reviews, 1):
            print(f"\nReview {i}:")
            print(review[:200] + "..." if len(review) > 200 else review)

def remove_duplicates(df):
    """Remove duplicate reviews"""
    print("\n=== Removing Duplicates ===")
    
    duplicates = df.duplicated(subset=['Text', 'Score']).sum()
    print(f"Number of duplicate reviews: {duplicates}")
    
    # Remove duplicates for cleaner dataset
    df_clean = df.drop_duplicates(subset=['Text', 'Score'])
    print(f"Dataset size after removing duplicates: {len(df_clean)}")
    
    return df_clean

def create_training_samples(df_clean):
    """Create balanced samples for training cycles"""
    print("\n=== Creating Training Samples ===")
    
    # Save a sample of the data for initial training (10k entries)
    # We'll create a balanced sample
    sample_size_per_class = 3334  # Approximately 10k total
    
    sampled_data = []
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        sentiment_data = df_clean[df_clean['Sentiment'] == sentiment]
        if len(sentiment_data) >= sample_size_per_class:
            sampled_data.append(sentiment_data.sample(n=sample_size_per_class, random_state=42))
        else:
            sampled_data.append(sentiment_data)
    
    sample_10k = pd.concat(sampled_data)
    print(f"Sample dataset size: {len(sample_10k)}")
    print(f"Sample sentiment distribution:")
    print(sample_10k['Sentiment'].value_counts())
    
    # Save the sample
    sample_10k.to_csv('data/sample_10k.csv', index=False)
    print("\nSample dataset saved as 'sample_10k.csv'")
    
    return sample_10k

def main():
    print("Starting Data Exploration...")
    

    df = load_and_explore_data()
    
  
    analyze_dataset_info(df)
    
   
    analyze_score_distribution(df)
    
    df = convert_to_sentiment(df)
    
    analyze_text_length(df)

    sample_reviews_by_sentiment(df)
    
    df_clean = remove_duplicates(df)
    
    sample_10k = create_training_samples(df_clean)
    
    print("\n=== Data Exploration Complete ===")
    print("Next step: Data preprocessing and text cleaning")
    
    return df_clean, sample_10k

if __name__ == "__main__":
    df_clean, sample_10k = main()
