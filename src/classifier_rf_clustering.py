"""
Random Forest + Hierarchical Clustering Classifier for Dyslexia Prediction

This module implements a Random Forest classifier enhanced with hierarchical clustering
for feature selection and subject grouping to improve dyslexia prediction performance.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.utils import shuffle, resample
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import seaborn as sns
import config

class RandomForestClusteringClassifier:
    """
    Random Forest classifier with hierarchical clustering for feature selection
    and subject grouping.
    """
    
    def __init__(self, rf_config=None, clustering_config=None):
        self.rf_config = rf_config or config.rf_config
        self.clustering_config = clustering_config or config.rf_clustering_config
        
        self.rf_classifier = RandomForestClassifier(
            n_estimators=self.rf_config["n_estimators"],
            max_depth=self.rf_config["max_depth"],
            min_samples_split=self.rf_config["min_samples_split"],
            min_samples_leaf=self.rf_config["min_samples_leaf"],
            random_state=self.rf_config["random_state"]
        )
        
        self.feature_clusters = None
        self.subject_clusters = None
        self.selected_features = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def _cluster_features(self, X, y):
        """
        Perform hierarchical clustering on features to identify important feature groups
        """
        print(f"Performing feature clustering with {self.clustering_config['n_clusters']} clusters...")
        
        # Calculate feature importance using a preliminary RF
        temp_rf = RandomForestClassifier(n_estimators=50, random_state=42)
        temp_rf.fit(X, y)
        feature_importance = temp_rf.feature_importances_
        
        # Cluster features based on correlation and importance
        if X.shape[1] > 1:
            # Calculate feature correlation matrix
            feature_corr = np.corrcoef(X.T)
            
            # Replace NaN with 0 (happens when features have zero variance)
            feature_corr = np.nan_to_num(feature_corr)
            
            # Perform hierarchical clustering on features
            try:
                distance_matrix = 1 - np.abs(feature_corr)
                clustering = AgglomerativeClustering(
                    n_clusters=min(self.clustering_config['n_clusters'], X.shape[1]),
                    linkage=self.clustering_config['linkage']
                )
                feature_clusters = clustering.fit_predict(distance_matrix)
                
                # Select top features from each cluster
                self.selected_features = []
                for cluster_id in range(max(feature_clusters) + 1):
                    cluster_features = np.where(feature_clusters == cluster_id)[0]
                    # Select feature with highest importance in each cluster
                    cluster_importance = feature_importance[cluster_features]
                    best_feature = cluster_features[np.argmax(cluster_importance)]
                    self.selected_features.append(best_feature)
                
                self.feature_clusters = feature_clusters
                print(f"Selected {len(self.selected_features)} features from {max(feature_clusters) + 1} clusters")
                
            except Exception as e:
                print(f"Feature clustering failed: {e}. Using all features.")
                self.selected_features = list(range(X.shape[1]))
        else:
            self.selected_features = list(range(X.shape[1]))
            
        return self.selected_features
    
    def _cluster_subjects(self, subject_features, subject_labels):
        """
        Perform hierarchical clustering on subjects to identify similar patterns
        """
        print(f"Performing subject clustering...")
        
        # Create subject-level feature representations (mean across samples)
        subject_means = {}
        for subject, features in subject_features.items():
            subject_means[subject] = np.mean(features, axis=0)
        
        subjects = list(subject_means.keys())
        subject_matrix = np.array([subject_means[subj] for subj in subjects])
        
        if len(subjects) > 1:
            try:
                # Perform hierarchical clustering on subjects
                clustering = AgglomerativeClustering(
                    n_clusters=min(3, len(subjects)),  # Max 3 subject clusters
                    linkage=self.clustering_config['linkage']
                )
                subject_clusters = clustering.fit_predict(subject_matrix)
                
                # Store clustering results
                self.subject_clusters = dict(zip(subjects, subject_clusters))
                
                print(f"Grouped {len(subjects)} subjects into {max(subject_clusters) + 1} clusters")
                
                # Print cluster composition
                for cluster_id in range(max(subject_clusters) + 1):
                    cluster_subjects = [subj for subj, cluster in self.subject_clusters.items() 
                                     if cluster == cluster_id]
                    cluster_labels = [subject_labels[subj][0] if subj in subject_labels 
                                    and len(subject_labels[subj]) > 0 else 0 
                                    for subj in cluster_subjects]
                    dyslexic_count = sum(cluster_labels)
                    print(f"  Cluster {cluster_id}: {cluster_subjects} "
                          f"({dyslexic_count}/{len(cluster_subjects)} dyslexic)")
                          
            except Exception as e:
                print(f"Subject clustering failed: {e}")
                self.subject_clusters = {subj: 0 for subj in subjects}
        else:
            self.subject_clusters = {subjects[0]: 0} if subjects else {}

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

def benchmark_rf_clustering(X, y, test_X, test_y):
    """
    Random Forest + Hierarchical Clustering benchmark
    """
    np.random.seed(config.seed)
    
    print("\n" + "="*50)
    print("RANDOM FOREST + HIERARCHICAL CLUSTERING")
    print("="*50)
    
    # Initialize classifier
    rf_clustering = RandomForestClusteringClassifier()
    
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
    train_X = rf_clustering.scaler.fit_transform(train_X)
    test_X_scaled = [rf_clustering.scaler.transform(subj) for subj in test_X_array]
    
    # Feature clustering if enabled
    if rf_clustering.clustering_config["cluster_features"]:
        selected_features = rf_clustering._cluster_features(train_X, train_y)
        train_X = train_X[:, selected_features]
        test_X_scaled = [subj[:, selected_features] for subj in test_X_scaled]
        print(f"Using {len(selected_features)} selected features")
    
    # Subject clustering if enabled
    if rf_clustering.clustering_config["cluster_subjects"]:
        rf_clustering._cluster_subjects(X, y)
    
    print("\nTraining Random Forest classifier...")
    rf_clustering.rf_classifier.fit(train_X, train_y)
    
    train_acc = accuracy_score(train_y, rf_clustering.rf_classifier.predict(train_X))
    print(f"Train accuracy: {train_acc:.4f}")
    
    # Feature importance analysis
    feature_importance = rf_clustering.rf_classifier.feature_importances_
    print(f"Top 5 most important features: {np.argsort(feature_importance)[-5:][::-1]}")
    
    # Predict on all test subjects
    results = zip(*[predict_subject_rf(rf_clustering.rf_classifier, test_X_scaled, 
                                      test_y_array, index, subj)
                   for index, subj in enumerate(config.heldout_subjects)])
    results = list(map(list, results))
    
    # Bootstrap if enabled
    bootstrap_results = [[]]
    if config.bootstrap:
        bootstrap_results = zip(*[bootstrap_confidence_rf(rf_clustering.rf_classifier,
                                                         test_X_scaled, test_y_array,
                                                         index, subj)
                                 for index, subj in enumerate(config.heldout_subjects)])
        bootstrap_results = list(map(list, bootstrap_results))
    
    return [results] + [bootstrap_results]

def benchmark_rf_clustering_baseline(X, y, test_X, test_y):
    """
    Random Forest + Clustering baseline (no evaluation)
    """
    np.random.seed(config.seed)
    
    print("\n" + "="*50)
    print("RANDOM FOREST + CLUSTERING BASELINE")
    print("="*50)
    
    # Initialize classifier
    rf_clustering = RandomForestClusteringClassifier()
    
    print("\nPre-processing data...")
    train_X, train_y = build_data_rf(X, y)
    
    # Build test data
    builded = zip(*[build_data_rf({subj: test_X[subj]}, {subj: test_y[subj]})
                   for subj in test_X])
    builded = list(map(list, builded))
    test_X_array, test_y_array = builded
    
    train_X, train_y = shuffle(train_X, train_y)
    
    # Scale features
    train_X = rf_clustering.scaler.fit_transform(train_X)
    test_X_scaled = [rf_clustering.scaler.transform(subj) for subj in test_X_array]
    
    # Feature clustering
    if rf_clustering.clustering_config["cluster_features"]:
        selected_features = rf_clustering._cluster_features(train_X, train_y)
        train_X = train_X[:, selected_features]
        test_X_scaled = [subj[:, selected_features] for subj in test_X_scaled]
    
    # Subject clustering
    if rf_clustering.clustering_config["cluster_subjects"]:
        rf_clustering._cluster_subjects(X, y)
    
    print("\nTraining Random Forest classifier...")
    rf_clustering.rf_classifier.fit(train_X, train_y)
    
    train_acc = accuracy_score(train_y, rf_clustering.rf_classifier.predict(train_X))
    print(f"Train accuracy: {train_acc:.4f}")
    
    # Generate predictions
    results = []
    for index, subj in enumerate(config.heldout_subjects):
        print(f"\nPredicting on subject {subj}")
        prediction = rf_clustering.rf_classifier.predict(test_X_scaled[index])
        results.append(prediction)
    
    return results

if __name__ == "__main__":
    print("Random Forest + Hierarchical Clustering Classifier for Dyslexia Prediction")
    print("Use this module through the main benchmark scripts.")
