
# Step 2: Data Preprocessing and Text Cleaning


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import string

# Download necessary NLTK data (run once)
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

class TextPreprocessor:
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers (keep words with numbers like '3d' might be relevant)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_remove_stopwords(self, text):
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words (length < 2)
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def lemmatize_tokens(self, tokens):
        """Lemmatize tokens to their base form"""
        if not tokens:
            return []
        
        # Simple lemmatization (could be enhanced with POS tagging)
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized
    
    def preprocess_text(self, text, join_tokens=True):
        
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize and remove stopwords
        tokens = self.tokenize_and_remove_stopwords(cleaned)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        if join_tokens:
            return ' '.join(tokens)
        else:
            return tokens

def preprocess_dataset(df, text_column='Text', summary_column='Summary'):
   
    print("Starting Data Preprocessing ")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Create copies of original columns
    df['Original_Text'] = df[text_column].copy()
    df['Original_Summary'] = df[summary_column].copy()
    
    print("Preprocessing review text...")
    # Preprocess main text
    df['Cleaned_Text'] = df[text_column].apply(
        lambda x: preprocessor.preprocess_text(x, join_tokens=True)
    )
    
    print("Preprocessing review summary...")
    # Preprocess summary
    df['Cleaned_Summary'] = df[summary_column].apply(
        lambda x: preprocessor.preprocess_text(x, join_tokens=True)
    )
    
    # Combine text and summary for better features
    print("Combining text and summary...")
    df['Combined_Text'] = df['Cleaned_Text'] + ' ' + df['Cleaned_Summary']
    df['Combined_Text'] = df['Combined_Text'].str.strip()
    
    # Remove rows with empty text after preprocessing
    initial_count = len(df)
    df = df[df['Combined_Text'].str.len() > 0]
    final_count = len(df)
    
    print(f"Removed {initial_count - final_count} rows with empty text after preprocessing")
    print(f"Final dataset size: {final_count}")
    
    # Display before and after examples
    print("\n=== Preprocessing Examples ===")
    for i in range(min(3, len(df))):
        print(f"\nExample {i+1}:")
        print(f"Original: {df.iloc[i]['Original_Text'][:100]}...")
        print(f"Cleaned:  {df.iloc[i]['Cleaned_Text'][:100]}...")
    
    return df

def analyze_preprocessed_data(df):
    """Analyze the preprocessed data"""
    print("\n=== Preprocessed Data Analysis ===")
    
    # Text length statistics after preprocessing
    df['Processed_Text_Length'] = df['Combined_Text'].str.len()
    
    print("Processed text statistics:")
    print(df['Processed_Text_Length'].describe())
    
    # Check for empty or very short texts
    very_short = (df['Processed_Text_Length'] < 10).sum()
    print(f"\nNumber of very short texts (<10 chars): {very_short}")
    
    # Word count analysis
    df['Word_Count'] = df['Combined_Text'].str.split().str.len()
    print("\nWord count statistics:")
    print(df['Word_Count'].describe())
    
    return df

def split_data(df, test_size=0.2, random_state=42):
   
    # Prepare features and target
    X = df['Combined_Text']
    y = df['Sentiment']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Display class distribution in train and test sets
    print("\nTraining set sentiment distribution:")
    print(y_train.value_counts())
    
    print("\nTest set sentiment distribution:")
    print(y_test.value_counts())
    
    return X_train, X_test, y_train, y_test

def save_preprocessed_data(df, X_train, X_test, y_train, y_test, sample_size='10k'):
    print(f"\n=== Saving Preprocessed Data ({sample_size}) ===")
    
    # Save main preprocessed dataframe
    df.to_csv(f'.git/data/preprocessed_{sample_size}.csv', index=False)
    print(f"Saved preprocessed data: preprocessed_{sample_size}.csv")
    
    # Save train/test splits
    train_data = pd.DataFrame({
        'Text': X_train,
        'Sentiment': y_train
    })
    test_data = pd.DataFrame({
        'Text': X_test,
        'Sentiment': y_test
    })
    
    train_data.to_csv(f'.git/data/train_{sample_size}.csv', index=False)
    test_data.to_csv(f'.git/data/test_{sample_size}.csv', index=False)
    
    print(f"Saved training data: train_{sample_size}.csv")
    print(f"Saved testing data: test_{sample_size}.csv")

def main():
    print("Starting Data Preprocessing...")
    
    # Download NLTK data
    print("Downloading NLTK data...")
    download_nltk_data()
    
    # Load the sample data
    print("Loading sample data...")
    df = pd.read_csv('.git/data/sample_10k.csv')
    print(f"Loaded dataset with {len(df)} rows")
    
    # Preprocess the data
    df_processed = preprocess_dataset(df)
    
    # Analyze preprocessed data
    df_processed = analyze_preprocessed_data(df_processed)
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = split_data(df_processed)
    
    # Save preprocessed data
    save_preprocessed_data(df_processed, X_train, X_test, y_train, y_test, '10k')
    
    print("\n=== Data Preprocessing Complete ===")
    print("Next step: Feature extraction and model training")
    
    return df_processed, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df_processed, X_train, X_test, y_train, y_test = main()
