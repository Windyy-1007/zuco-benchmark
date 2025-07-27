"""
Simple Neural Network Classifier for Dyslexia Prediction (without TensorFlow)

This module implements a simple multi-layer perceptron using scikit-learn
as an alternative to CNN when TensorFlow is not available.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import config

def build_data_mlp(data, labels):
    """
    Build data for MLP classification
    """
    X, y = [], []
    for subj in data:
        for features in data[subj]:
            X.append(features)
        for label in labels[subj]:
            if config.task_type == "dyslexia_prediction":
                y.append(int(label))
            else:
                y.append(1 if label == "NR" else 0)
    return np.array(X), np.array(y)

def predict_subject_mlp(mlp_classifier, test_X, test_y, index, subj):
    """
    Predict on a single subject using MLP
    """
    print(f"\nPredicting on subject {subj}")
    prediction = mlp_classifier.predict(test_X[index])
    
    if len(test_y[index]) > 0:  # If we have true labels
        accuracy = accuracy_score(test_y[index], prediction)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_y[index], prediction, average='binary', zero_division=0)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 score: {f1:.4f}")
        
        return [prediction], accuracy, f1, precision, recall
    else:
        return [prediction], 0, 0, 0, 0

def plot_learning_curve(mlp_classifier, save_path=None):
    """
    Plot learning curve for MLP
    """
    if hasattr(mlp_classifier, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(mlp_classifier.loss_curve_)
        plt.title('MLP Training Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curve saved to: {save_path}")
        
        plt.show()
    else:
        print("No learning curve available for this classifier.")

def benchmark_mlp(X, y, test_X, test_y):
    """
    Multi-Layer Perceptron benchmark for dyslexia prediction
    """
    np.random.seed(config.seed)
    
    print("\n" + "="*50)
    print("MULTI-LAYER PERCEPTRON (MLP)")
    print("="*50)
    
    print("\nPre-processing data...")
    train_X, train_y = build_data_mlp(X, y)
    
    # Build test data
    builded = zip(*[build_data_mlp({subj: test_X[subj]}, {subj: test_y[subj]})
                   for subj in test_X])
    builded = list(map(list, builded))
    test_X_array, test_y_array = builded
    
    train_X, train_y = shuffle(train_X, train_y)
    
    print(f"Training samples: {len(train_X)}")
    print(f"Training features: {train_X.shape[1]}")
    print(f"Training class distribution: {np.bincount(train_y)}")
    
    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X = scaler.fit_transform(train_X)
    test_X_scaled = [scaler.transform(subj) for subj in test_X_array]
    
    print("\nTraining MLP classifier...")
    
    # Configure MLP based on feature size
    n_features = train_X.shape[1]
    if n_features > 100:
        # For large feature sets (like electrode data)
        hidden_layer_sizes = (128, 64, 32)
    else:
        # For smaller feature sets
        hidden_layer_sizes = (64, 32)
    
    mlp_classifier = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=500,
        shuffle=True,
        random_state=config.seed,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=True,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    
    mlp_classifier.fit(train_X, train_y)
    
    train_acc = accuracy_score(train_y, mlp_classifier.predict(train_X))
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Number of layers: {mlp_classifier.n_layers_}")
    print(f"Number of iterations: {mlp_classifier.n_iter_}")
    
    # Plot learning curve
    plot_learning_curve(mlp_classifier, save_path="../results/mlp_learning_curve.png")
    
    # Predict on all test subjects
    results = zip(*[predict_subject_mlp(mlp_classifier, test_X_scaled, test_y_array, 
                                       index, subj)
                   for index, subj in enumerate(config.heldout_subjects)])
    results = list(map(list, results))
    
    # No bootstrap for MLP (takes too long)
    bootstrap_results = [[]]
    
    return [results] + [bootstrap_results]

def benchmark_mlp_baseline(X, y, test_X, test_y):
    """
    MLP baseline (no evaluation)
    """
    np.random.seed(config.seed)
    
    print("\n" + "="*50)
    print("MLP BASELINE")
    print("="*50)
    
    print("\nPre-processing data...")
    train_X, train_y = build_data_mlp(X, y)
    
    # Build test data
    builded = zip(*[build_data_mlp({subj: test_X[subj]}, {subj: test_y[subj]})
                   for subj in test_X])
    builded = list(map(list, builded))
    test_X_array, test_y_array = builded
    
    train_X, train_y = shuffle(train_X, train_y)
    
    print(f"Training samples: {len(train_X)}")
    print(f"Training features: {train_X.shape[1]}")
    
    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X = scaler.fit_transform(train_X)
    test_X_scaled = [scaler.transform(subj) for subj in test_X_array]
    
    print("\nTraining MLP classifier...")
    
    # Configure MLP
    n_features = train_X.shape[1]
    if n_features > 100:
        hidden_layer_sizes = (128, 64, 32)
    else:
        hidden_layer_sizes = (64, 32)
    
    mlp_classifier = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=500,
        random_state=config.seed,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    mlp_classifier.fit(train_X, train_y)
    
    train_acc = accuracy_score(train_y, mlp_classifier.predict(train_X))
    print(f"Train accuracy: {train_acc:.4f}")
    
    # Plot learning curve
    plot_learning_curve(mlp_classifier, save_path="../results/mlp_learning_curve.png")
    
    # Generate predictions
    results = []
    for index, subj in enumerate(config.heldout_subjects):
        print(f"\nPredicting on subject {subj}")
        prediction = mlp_classifier.predict(test_X_scaled[index])
        results.append(prediction)
        print(f"Generated {len(prediction)} predictions")
    
    return results

if __name__ == "__main__":
    print("Multi-Layer Perceptron (MLP) Classifier for Dyslexia Prediction")
    print("Use this module through the main benchmark scripts.")
