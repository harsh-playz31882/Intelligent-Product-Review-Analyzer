
#Step 3: Feature Extraction


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import pickle
import os
from collections import Counter
import re

class FeatureExtractor:
    """Class for extracting various types of features from text"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.label_encoder = None
        
    def extract_tfidf_features(self, texts, max_features=5000, ngram_range=(1, 2)):
       
        print("Extracting TF-IDF Features ")
        print(f"Max features: {max_features}, N-gram range: {ngram_range}")
        
        # Initialize TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        
        # Fit and transform the texts
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        
        print(f"TF-IDF feature matrix shape: {tfidf_features.shape}")
        print(f"Number of features extracted: {len(self.tfidf_vectorizer.get_feature_names_out())}")
        
        # Display some feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        print(f"Sample features: {list(feature_names[:20])}")
        
        return tfidf_features
    
    def train_word2vec(self, texts, vector_size=100, window=5, min_count=2, workers=4):
        
        print("Training Word2Vec Model ")
        print(f"Vector size: {vector_size}, Window: {window}, Min count: {min_count}")
        
        # Tokenize texts for Word2Vec
        tokenized_texts = [text.split() for text in texts]
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=1  # Skip-gram model (better for semantic understanding)
        )
        
        print(f"Word2Vec model trained with {len(self.word2vec_model.wv)} words")
        print(f"Vocabulary size: {len(self.word2vec_model.wv.key_to_index)}")
        
        # Display similar words for some examples
        sample_words = ['good', 'bad', 'product', 'price', 'quality']
        for word in sample_words:
            if word in self.word2vec_model.wv:
                similar_words = self.word2vec_model.wv.most_similar(word, topn=3)
                print(f"Words similar to '{word}': {similar_words}")
        
        return self.word2vec_model
    
    def get_document_vector(self, text, use_tfidf_weighting=False):
        if not self.word2vec_model:
            raise ValueError("Word2Vec model not trained. Call train_word2vec first.")
        
        words = text.split()
        word_vectors = []
        
        for word in words:
            if word in self.word2vec_model.wv:
                word_vectors.append(self.word2vec_model.wv[word])
        
        if not word_vectors:
            # Return zero vector if no words found in vocabulary
            return np.zeros(self.word2vec_model.vector_size)
        
        # Average the word vectors
        doc_vector = np.mean(word_vectors, axis=0)
        
        return doc_vector
    
    def extract_word2vec_features(self, texts):
        print("=== Extracting Word2Vec Features ===")
        
        if not self.word2vec_model:
            raise ValueError("Word2Vec model not trained. Call train_word2vec first.")
        
        doc_vectors = []
        
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"Processing document {i}/{len(texts)}")
            
            doc_vector = self.get_document_vector(text)
            doc_vectors.append(doc_vector)
        
        word2vec_features = np.array(doc_vectors)
        print(f"Word2Vec feature matrix shape: {word2vec_features.shape}")
        
        return word2vec_features
    
    def extract_text_statistics(self, texts):
        print("=== Extracting Text Statistics ===")
        
        features = []
        
        for text in texts:
            # Basic counts
            char_count = len(text)
            word_count = len(text.split())
            sentence_count = len(re.split(r'[.!?]+', text))
            
            # Average word length
            words = text.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Punctuation counts
            exclamation_count = text.count('!')
            question_count = text.count('?')
            period_count = text.count('.')
            
            # Capitalization features
            uppercase_count = sum(1 for c in text if c.isupper())
            uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0
            
            # Digit count
            digit_count = sum(c.isdigit() for c in text)
            
            features.append({
                'char_count': char_count,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_word_length': avg_word_length,
                'exclamation_count': exclamation_count,
                'question_count': question_count,
                'period_count': period_count,
                'uppercase_count': uppercase_count,
                'uppercase_ratio': uppercase_ratio,
                'digit_count': digit_count
            })
        
        stats_df = pd.DataFrame(features)
        print(f"Text statistics shape: {stats_df.shape}")
        print("Sample statistics:")
        print(stats_df.head())
        
        return stats_df
    
    def encode_labels(self, labels):
        print("=== Encoding Labels ===")
        
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        print(f"Original labels: {list(self.label_encoder.classes_)}")
        print(f"Encoded labels: {np.unique(encoded_labels)}")
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, encoded_labels))}")
        
        return encoded_labels
    
    def save_features(self, features, filename, path='../models/'):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(features, f)
        
        print(f"Features saved to: {filepath}")
    
    def save_models(self, path='../models/'):
        os.makedirs(path, exist_ok=True)
        
        if self.tfidf_vectorizer:
            with open(os.path.join(path, 'tfidf_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            print("TF-IDF vectorizer saved")
        
        if self.word2vec_model:
            self.word2vec_model.save(os.path.join(path, 'word2vec_model.bin'))
            print("Word2Vec model saved")
        
        if self.label_encoder:
            with open(os.path.join(path, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print("Label encoder saved")

def extract_all_features(X_train, X_test, y_train, y_test, sample_size='10k'):
    print(f"Extracting All Features for {sample_size} Dataset ")
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Encode labels
    y_train_encoded = extractor.encode_labels(y_train)
    y_test_encoded = extractor.encode_labels(y_test)
    
    # Extract TF-IDF features
    print("\n" + "="*50)
    X_train_tfidf = extractor.extract_tfidf_features(X_train)
    X_test_tfidf = extractor.tfidf_vectorizer.transform(X_test)
    
    # Train Word2Vec and extract features
    print("\n" + "="*50)
    extractor.train_word2vec(X_train)
    X_train_w2v = extractor.extract_word2vec_features(X_train)
    X_test_w2v = extractor.extract_word2vec_features(X_test)
    
    # Extract text statistics
    print("\n" + "="*50)
    X_train_stats = extractor.extract_text_statistics(X_train)
    X_test_stats = extractor.extract_text_statistics(X_test)
    
    # Save all features and models
    print("\n" + "="*50)
    print("Saving Features and Models")
    
    # Save features
    X_train_stats.to_csv(f'.git/X_train_stats_{sample_size}.csv', index=False)
    X_test_stats.to_csv(f'.git/X_test_stats_{sample_size}.csv', index=False)
    extractor.save_features(X_train_tfidf, f'X_train_tfidf_{sample_size}.pkl')
    extractor.save_features(X_test_tfidf, f'X_test_tfidf_{sample_size}.pkl')
    extractor.save_features(X_train_w2v, f'X_train_w2v_{sample_size}.pkl')
    extractor.save_features(X_test_w2v, f'X_test_w2v_{sample_size}.pkl')
    return {
        'tfidf': (X_train_tfidf, X_test_tfidf),
        'word2vec': (X_train_w2v, X_test_w2v),
        'statistics': (X_train_stats, X_test_stats),
        'labels': (y_train_encoded, y_test_encoded),
        'extractor': extractor
    }

def main():
    print("Starting Feature Extraction...")
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    train_data = pd.read_csv('.git/train_10k.csv')
    test_data = pd.read_csv('.git/test_10k.csv')
    
    X_train = train_data['Text']
    X_test = test_data['Text']
    y_train = train_data['Sentiment']
    y_test = test_data['Sentiment']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Extract all features
    features = extract_all_features(X_train, X_test, y_train, y_test, '10k')
    
    print("\n=== Ready for Model Training ===")
    print("Available feature types:")
    print("- TF-IDF features (sparse matrix)")
    print("- Word2Vec features (dense matrix)")
    print("- Text statistics features (DataFrame)")
    
    return features

if __name__ == "__main__":
    features = main()
