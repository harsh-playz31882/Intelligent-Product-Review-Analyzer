"""
Step 5: Scaling Experiments

This script handles training models with progressively larger datasets:
- 10k entries (already done)
- 20k entries
- 40k entries
- 80k entries
- Compare performance across different dataset sizes
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Import our previous modules
from src_02_data_preprocessing import preprocess_dataset, split_data
from src_03_feature_extraction import FeatureExtractor
from src_04_model_training import ModelTrainer

def create_larger_samples(df_clean, sizes=[20000, 40000, 80000]):
    """
    Create larger balanced samples from the cleaned dataset
    Returns dictionary of DataFrames for each size
    """
    print("=== Creating Larger Dataset Samples ===")
    
    samples = {}
    
    for size in sizes:
        print(f"\nCreating sample of {size} entries...")
        
        # Calculate samples per class for balanced dataset
        samples_per_class = size // 3
        
        sampled_data = []
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            sentiment_data = df_clean[df_clean['Sentiment'] == sentiment]
            if len(sentiment_data) >= samples_per_class:
                sampled_data.append(sentiment_data.sample(n=samples_per_class, random_state=42))
            else:
                sampled_data.append(sentiment_data)
        
        sample_df = pd.concat(sampled_data)
        samples[f'{size//1000}k'] = sample_df
        
        print(f"Sample {size//1000}k created with {len(sample_df)} rows")
        print(f"Sentiment distribution:")
        print(sample_df['Sentiment'].value_counts())
        
        # Save the sample
        sample_df.to_csv(f'../data/sample_{size//1000}k.csv', index=False)
        print(f"Saved as sample_{size//1000}k.csv")
    
    return samples

def run_complete_pipeline(sample_size='20k'):
    """
    Run the complete pipeline for a given sample size
    Returns training results and performance metrics
    """
    print(f"\n{'='*60}")
    print(f"RUNNING COMPLETE PIPELINE FOR {sample_size.upper()} DATASET")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Step 1: Load and preprocess data
    print("\n=== Step 1: Data Preprocessing ===")
    df = pd.read_csv(f'../data/sample_{sample_size}.csv')
    print(f"Loaded {len(df)} rows")
    
    df_processed = preprocess_dataset(df)
    X_train, X_test, y_train, y_test = split_data(df_processed)
    
    # Step 2: Feature extraction
    print("\n=== Step 2: Feature Extraction ===")
    extractor = FeatureExtractor()
    
    # Encode labels
    y_train_encoded = extractor.encode_labels(y_train)
    y_test_encoded = extractor.encode_labels(y_test)
    
    # Extract TF-IDF features (faster than Word2Vec for large datasets)
    print("Extracting TF-IDF features...")
    X_train_tfidf = extractor.extract_tfidf_features(X_train, max_features=3000)  # Reduced features for speed
    X_test_tfidf = extractor.tfidf_vectorizer.transform(X_test)
    
    # Step 3: Model training (focus on best performing models)
    print("\n=== Step 3: Model Training ===")
    trainer = ModelTrainer()
    
    # Train selected models based on previous results
    models_to_train = [
        ('logistic_regression', trainer.train_logistic_regression),
        ('naive_bayes', trainer.train_naive_bayes),
        ('random_forest', trainer.train_random_forest),
    ]
    
    results = {}
    
    for model_name, train_func in models_to_train:
        print(f"\nTraining {model_name}...")
        try:
            model, training_time = train_func(X_train_tfidf, y_train_encoded)
            
            # Evaluate
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test_encoded, y_pred)
            report = classification_report(y_test_encoded, y_pred, output_dict=True)
            
            results[model_name] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'report': report
            }
            
            print(f"{model_name}: {accuracy:.4f} accuracy, {training_time:.2f}s training time")
            
            # Save model
            trainer.save_model(model, f"{model_name}_{sample_size}")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Save features and models
    extractor.save_features(X_train_tfidf, f'X_train_tfidf_{sample_size}.pkl')
    extractor.save_features(X_test_tfidf, f'X_test_tfidf_{sample_size}.pkl')
    extractor.save_features(y_train_encoded, f'y_train_{sample_size}.pkl')
    extractor.save_features(y_test_encoded, f'y_test_{sample_size}.pkl')
    extractor.save_models()
    
    total_time = time.time() - start_time
    print(f"\nTotal pipeline time for {sample_size}: {total_time:.2f} seconds")
    
    return results, total_time

def compare_scaling_results(all_results):
    """
    Compare results across different dataset sizes
    Creates visualizations and analysis
    """
    print("\n=== SCALING ANALYSIS ===")
    
    # Prepare data for visualization
    comparison_data = []
    
    for size, results in all_results.items():
        for model_name, metrics in results.items():
            comparison_data.append({
                'Dataset Size': size,
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Training Time': metrics['training_time']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\nPerformance Comparison:")
    print(comparison_df.pivot(index='Dataset Size', columns='Model', values='Accuracy').round(4))
    
    print("\nTraining Time Comparison (seconds):")
    print(comparison_df.pivot(index='Dataset Size', columns='Model', values='Training Time').round(2))
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy plot
    sns.lineplot(data=comparison_df, x='Dataset Size', y='Accuracy', 
                hue='Model', marker='o', ax=ax1)
    ax1.set_title('Model Accuracy vs Dataset Size')
    ax1.set_ylabel('Accuracy')
    
    # Training time plot
    sns.lineplot(data=comparison_df, x='Dataset Size', y='Training Time', 
                hue='Model', marker='s', ax=ax2)
    ax2.set_title('Training Time vs Dataset Size')
    ax2.set_ylabel('Training Time (seconds)')
    
    plt.tight_layout()
    plt.savefig('../results/scaling_analysis.png')
    plt.show()
    
    # Find best model for each size
    print("\nBest performing model by dataset size:")
    for size in comparison_df['Dataset Size'].unique():
        size_data = comparison_df[comparison_df['Dataset Size'] == size]
        best_model = size_data.loc[size_data['Accuracy'].idxmax()]
        print(f"{size}: {best_model['Model']} ({best_model['Accuracy']:.4f} accuracy)")
    
    return comparison_df

def main():
    """Main scaling experiment pipeline"""
    print("Starting Scaling Experiments...")
    
    # Load the cleaned dataset
    print("Loading cleaned dataset...")
    df_clean = pd.read_csv('../data/preprocessed_10k.csv')
    
    # Create larger samples
    samples = create_larger_samples(df_clean, [20000, 40000, 80000])
    
    # Run experiments for each size
    sizes_to_test = ['20k', '40k', '80k']
    all_results = {}
    
    for size in sizes_to_test:
        try:
            results, total_time = run_complete_pipeline(size)
            all_results[size] = results
            print(f"Completed {size} experiment in {total_time:.2f} seconds")
        except Exception as e:
            print(f"Error in {size} experiment: {str(e)}")
            continue
    
    # Compare results
    if all_results:
        comparison_df = compare_scaling_results(all_results)
        
        # Save comparison results
        comparison_df.to_csv('../results/scaling_comparison.csv', index=False)
        with open('../results/scaling_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        
        print("\n=== Scaling Experiments Complete ===")
        print("Results saved to ../results/")
    else:
        print("No experiments completed successfully")
    
    return all_results

if __name__ == "__main__":
    all_results = main()
