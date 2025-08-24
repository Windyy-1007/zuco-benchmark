"""
Random Forest Classifier for Dyslexia Prediction

This module implements a Random Forest classifier for dyslexia prediction
using EEG and eye-tracking features.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.utils import shuffle, resample
import matplotlib.pyplot as plt
import seaborn as sns
import config

def build_data_rf(data, labels):
    """
    Build data for Random Forest classification
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

def bootstrap_confidence_rf(clf, test_X, test_y, index, subj):
    """
    Bootstrap confidence intervals for Random Forest
    """
    resampled_Xs, resampled_ys = zip(*[resample(test_X[index], test_y[index], replace=True,
                                               n_samples=len(test_X[index]),
                                               random_state=config.seed+i)
                                      for i in range(config.n_bootstraps)])
    test = list(map(list, zip(*[predict_subject_rf(clf, resampled_Xs, resampled_ys, 
                                                  data_index, subj)
                               for data_index in range(config.n_bootstraps)])))
    return test

def predict_subject_rf(clf, test_X, test_y, index, subj):
    """
    Predict on a single subject using Random Forest
    """
    print(f"\nPredicting on subject {subj}")
    prediction = clf.predict(test_X[index])
    
    if len(test_y[index]) > 0:  # If we have true labels
        accuracy = accuracy_score(test_y[index], prediction)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_y[index], prediction, average='binary', zero_division=0)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 score: {f1:.4f}")
        
        return [prediction], accuracy, f1, precision, recall
    else:
        return [prediction], 0, 0, 0, 0

def plot_feature_importance(clf, feature_names=None, top_n=20, save_path=None):
    """
    Plot feature importance from Random Forest
    """
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importances (Random Forest)')
    
    if feature_names is not None:
        labels = [feature_names[i] for i in indices]
    else:
        labels = [f'Feature {i}' for i in indices]
    
    plt.bar(range(top_n), importance[indices])
    plt.xticks(range(top_n), labels, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
    
    plt.show()
    
    return indices, importance[indices]

def benchmark_random_forest(X, y, test_X, test_y):
    """
    Random Forest benchmark for dyslexia prediction
    """
    np.random.seed(config.seed)
    
    print("\n" + "="*50)
    print("RANDOM FOREST CLASSIFIER")
    print("="*50)
    
    print("\nPre-processing data...")
    train_X, train_y = build_data_rf(X, y)
    
    # Build test data
    builded = zip(*[build_data_rf({subj: test_X[subj]}, {subj: test_y[subj]})
                   for subj in test_X])
    builded = list(map(list, builded))
    test_X_array, test_y_array = builded
    
    train_X, train_y = shuffle(train_X, train_y)
    
    print(f"Training samples: {len(train_X)}")
    print(f"Training features: {train_X.shape[1]}")
    print(f"Class distribution: {np.bincount(train_y)}")
    
    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X = scaler.fit_transform(train_X)
    test_X_scaled = [scaler.transform(subj) for subj in test_X_array]
    
    print("\nTraining Random Forest classifier...")
    # Initialize Random Forest with config parameters
    clf = RandomForestClassifier(
        n_estimators=config.rf_config["n_estimators"],
        max_depth=config.rf_config["max_depth"],
        min_samples_split=config.rf_config["min_samples_split"],
        min_samples_leaf=config.rf_config["min_samples_leaf"],
        random_state=config.rf_config["random_state"],
        n_jobs=-1  # Use all available cores
    )
    
    clf.fit(train_X, train_y)
    
    train_acc = accuracy_score(train_y, clf.predict(train_X))
    print(f"Train accuracy: {train_acc:.4f}")
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    top_features, top_importance = plot_feature_importance(
        clf, top_n=20, save_path="../results/rf_feature_importance.png")
    
    print(f"Top 5 most important features: {top_features[:5]}")
    print(f"Top 5 importance values: {top_importance[:5]}")
    
    # Predict on all test subjects
    results = zip(*[predict_subject_rf(clf, test_X_scaled, test_y_array, index, subj)
                   for index, subj in enumerate(config.heldout_subjects)])
    results = list(map(list, results))
    
    # Bootstrap if enabled
    bootstrap_results = [[]]
    if config.bootstrap:
        print("\nRunning bootstrap analysis...")
        bootstrap_results = zip(*[bootstrap_confidence_rf(clf, test_X_scaled, test_y_array,
                                                         index, subj)
                                 for index, subj in enumerate(config.heldout_subjects)])
        bootstrap_results = list(map(list, bootstrap_results))
    
    return [results] + [bootstrap_results]

def benchmark_random_forest_baseline(X, y, test_X, test_y):
    """
    Random Forest baseline (no evaluation)
    """
    np.random.seed(config.seed)
    
    print("\n" + "="*50)
    print("RANDOM FOREST BASELINE")
    print("="*50)
    
    print("\nPre-processing data...")
    train_X, train_y = build_data_rf(X, y)
    
    # Build test data
    builded = zip(*[build_data_rf({subj: test_X[subj]}, {subj: test_y[subj]})
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
    
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=config.rf_config["n_estimators"],
        max_depth=config.rf_config["max_depth"],
        min_samples_split=config.rf_config["min_samples_split"],
        min_samples_leaf=config.rf_config["min_samples_leaf"],
        random_state=config.rf_config["random_state"],
        n_jobs=-1
    )
    
    clf.fit(train_X, train_y)
    
    train_acc = accuracy_score(train_y, clf.predict(train_X))
    print(f"Train accuracy: {train_acc:.4f}")
    
    # Feature importance
    plot_feature_importance(clf, top_n=20, save_path="../results/rf_feature_importance.png")
    
    # Generate predictions
    results = []
    for index, subj in enumerate(config.heldout_subjects):
        print(f"\nPredicting on subject {subj}")
        prediction = clf.predict(test_X_scaled[index])
        results.append(prediction)
        print(f"Generated {len(prediction)} predictions")
    
    return results

if __name__ == "__main__":
    print("Random Forest Classifier for Dyslexia Prediction")
    print("Use this module through the main benchmark scripts.")
