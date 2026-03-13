"""
Step 4: Model Training

This script handles training various machine learning models for sentiment analysis:
- Naive Bayes
- Logistic Regression
- Random Forest
- SVM
- Gradient Boosting
- Neural Network (MLP)
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import time

class ModelTrainer:
    """Class for training and evaluating ML models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_naive_bayes(self, X_train, y_train, alpha=1.0):
        """Train Multinomial Naive Bayes"""
        print("=== Training Naive Bayes ===")
        
        model = MultinomialNB(alpha=alpha)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.models['naive_bayes'] = model
        print(f"Naive Bayes trained in {training_time:.2f} seconds")
        
        return model, training_time
    
    def train_logistic_regression(self, X_train, y_train, C=1.0, max_iter=1000):
        """Train Logistic Regression"""
        print("=== Training Logistic Regression ===")
        
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.models['logistic_regression'] = model
        print(f"Logistic Regression trained in {training_time:.2f} seconds")
        
        return model, training_time
    
    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=None):
        """Train Random Forest"""
        print("=== Training Random Forest ===")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.models['random_forest'] = model
        print(f"Random Forest trained in {training_time:.2f} seconds")
        
        return model, training_time
    
    def train_svm(self, X_train, y_train, C=1.0, kernel='linear'):
        """Train Support Vector Machine"""
        print("=== Training SVM ===")
        
        model = SVC(C=C, kernel=kernel, random_state=42)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.models['svm'] = model
        print(f"SVM trained in {training_time:.2f} seconds")
        
        return model, training_time
    
    def train_gradient_boosting(self, X_train, y_train, n_estimators=100, learning_rate=0.1):
        """Train Gradient Boosting"""
        print("=== Training Gradient Boosting ===")
        
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.models['gradient_boosting'] = model
        print(f"Gradient Boosting trained in {training_time:.2f} seconds")
        
        return model, training_time
    
    def train_mlp(self, X_train, y_train, hidden_layer_sizes=(100,), max_iter=500):
        """Train Multi-layer Perceptron"""
        print("=== Training MLP Neural Network ===")
        
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.models['mlp'] = model
        print(f"MLP trained in {training_time:.2f} seconds")
        
        return model, training_time
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a trained model"""
        print(f"\n=== Evaluating {model_name} ===")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm
        }
        
        return accuracy, report, cm
    
    def cross_validate_model(self, model, X_train, y_train, cv=5):
        """Perform cross-validation on a model"""
        print(f"\n=== Cross-Validation (cv={cv}) ===")
        
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation scores: {scores}")
        print(f"Mean CV accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def plot_confusion_matrix(self, cm, model_name, labels=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'../results/confusion_matrix_{model_name}.png')
        plt.show()
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n=== Model Comparison ===")
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision (macro)': results['report']['macro avg']['precision'],
                'Recall (macro)': results['report']['macro avg']['recall'],
                'F1-score (macro)': results['report']['macro avg']['f1-score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print(comparison_df.round(4))
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        comparison_df.set_index('Model')[['Accuracy', 'Precision (macro)', 
                                         'Recall (macro)', 'F1-score (macro)']].plot(kind='bar')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('../results/model_comparison.png')
        plt.show()
        
        return comparison_df
    
    def save_model(self, model, model_name, path='../models/'):
        """Save trained model to disk"""
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f'{model_name}_model.pkl')
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model saved to: {filepath}")
    
    def save_results(self, path='../results/'):
        """Save evaluation results to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save results as pickle
        with open(os.path.join(path, 'training_results.pkl'), 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save comparison as CSV
        comparison_df = self.compare_models()
        comparison_df.to_csv(os.path.join(path, 'model_comparison.csv'), index=False)
        
        print("Results saved to disk")

def train_all_models(X_train, X_test, y_train, y_test, feature_type='tfidf'):
    """
    Train all models with specified feature type
    Returns trained models and evaluation results
    """
    print(f"=== Training All Models with {feature_type.upper()} Features ===")
    
    trainer = ModelTrainer()
    
    # Train models
    models_to_train = [
        ('naive_bayes', trainer.train_naive_bayes),
        ('logistic_regression', trainer.train_logistic_regression),
        ('random_forest', trainer.train_random_forest),
        ('gradient_boosting', trainer.train_gradient_boosting),
        ('mlp', trainer.train_mlp)
    ]
    
    # Skip SVM for large datasets due to computational cost
    if X_train.shape[1] < 10000:  # Only train SVM if feature space is manageable
        models_to_train.append(('svm', trainer.train_svm))
    
    training_results = {}
    
    for model_name, train_func in models_to_train:
        print(f"\n{'='*50}")
        try:
            model, training_time = train_func(X_train, y_train)
            training_results[model_name] = training_time
            
            # Evaluate model
            accuracy, report, cm = trainer.evaluate_model(model, X_test, y_test, model_name)
            
            # Plot confusion matrix
            trainer.plot_confusion_matrix(cm, model_name)
            
            # Save model
            trainer.save_model(model, f"{model_name}_{feature_type}")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Compare all models
    comparison_df = trainer.compare_models()
    
    # Save results
    trainer.save_results()
    
    return trainer, comparison_df

def main():
    """Main training pipeline"""
    print("Starting Model Training...")
    
    # Load features
    print("Loading extracted features...")
    with open('../models/X_train_tfidf_10k.pkl', 'rb') as f:
        X_train_tfidf = pickle.load(f)
    with open('../models/X_test_tfidf_10k.pkl', 'rb') as f:
        X_test_tfidf = pickle.load(f)
    with open('../models/y_train_10k.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('../models/y_test_10k.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    print(f"Training features shape: {X_train_tfidf.shape}")
    print(f"Test features shape: {X_test_tfidf.shape}")
    
    # Train models with TF-IDF features
    trainer, comparison_df = train_all_models(
        X_train_tfidf, X_test_tfidf, y_train, y_test, 'tfidf'
    )
    
    print("\n=== Training Complete ===")
    print("Best performing model:")
    best_model = comparison_df.iloc[0]
    print(f"{best_model['Model']}: {best_model['Accuracy']:.4f} accuracy")
    
    return trainer, comparison_df

if __name__ == "__main__":
    trainer, comparison_df = main()
