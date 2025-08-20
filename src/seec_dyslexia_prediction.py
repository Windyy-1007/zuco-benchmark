"""
Machine Learning Pipeline for Dyslexia Prediction - SEEC Data Only
Focuses on sentence-level eye-tracking, gaze, and saccade features
Applies MLP, Random Forest, SVM, and clustering
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    silhouette_score, adjusted_rand_score
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class SEECDyslexiaPrediction:
    def __init__(self, feat_csv_dir="c:\\Projects\\zuco-benchmark\\feat_csv"):
        self.feat_csv_dir = feat_csv_dir
        self.scalers = {}
        self.models = {}
        self.feature_importances = {}
        self.results = {}
        
    def load_seec_data(self):
        """Load only SEEC (sentence-level eye-tracking, EEG, cognitive) data"""
        print("Loading SEEC (eye-tracking and cognitive) data only...")
        
        # Get CSV files - exclude electrode features
        csv_files = glob.glob(os.path.join(self.feat_csv_dir, "*.csv"))
        
        # Group files by SEEC feature type (no electrode features)
        seec_feature_types = {
            'gaze_saccade': [],
            'gaze_saccade_eeg_means': []
        }
        
        for file in csv_files:
            filename = os.path.basename(file)
            if 'sent_gaze_sacc_eeg_means_detailed' in filename:
                seec_feature_types['gaze_saccade_eeg_means'].append(file)
            elif 'sent_gaze_sacc_detailed' in filename and 'eeg_means' not in filename:
                seec_feature_types['gaze_saccade'].append(file)
        
        # Load each SEEC feature type
        self.datasets = {}
        
        for feat_type, files in seec_feature_types.items():
            if files:  # Only process if files exist
                print(f"Loading {feat_type} features from {len(files)} files...")
                dfs = []
                
                for file in files:
                    try:
                        df = pd.read_csv(file)
                        dfs.append(df)
                    except Exception as e:
                        print(f"  Warning: Could not load {file}: {e}")
                
                if dfs:
                    combined_df = pd.concat(dfs, ignore_index=True)
                    self.datasets[feat_type] = combined_df
                    print(f"  - {feat_type}: {combined_df.shape}")
                    print(f"  - Classes: {combined_df['Dyslexia_Class'].value_counts()}")
        
        return self.datasets
    
    def preprocess_data(self, dataset_name):
        """Preprocess a specific SEEC dataset"""
        print(f"\nPreprocessing {dataset_name}...")
        df = self.datasets[dataset_name].copy()
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col.startswith('Feature_')]
        X = df[feature_cols]
        y = df['Dyslexia_Label']
        subject_ids = df['Subject_ID']
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        # Remove any constant features
        constant_features = X.columns[X.std() == 0]
        if len(constant_features) > 0:
            print(f"  - Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
            feature_cols = X.columns.tolist()
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        self.scalers[dataset_name] = scaler
        
        print(f"  - Features shape: {X_scaled.shape}")
        print(f"  - Class distribution: Normal: {sum(y==0)}, Dyslexic: {sum(y==1)}")
        
        return X_scaled, y, subject_ids
    
    def apply_machine_learning_models(self, dataset_name):
        """Apply MLP, Random Forest, and SVM to SEEC data"""
        print(f"\n{'='*50}")
        print(f"SEEC MACHINE LEARNING ANALYSIS: {dataset_name.upper()}")
        print(f"{'='*50}")
        
        X, y, subject_ids = self.preprocess_data(dataset_name)
        
        # Check if we have enough samples for each class
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        
        if len(unique_classes) < 2:
            print("Warning: Only one class present. Cannot train classifiers.")
            return {}
        
        if min(class_counts) < 2:
            print("Warning: Insufficient samples for cross-validation.")
            # Use simple train-test split without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        
        # Define models optimized for small datasets
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=50,  # Reduced for small dataset
                max_depth=10,     # Prevent overfitting
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42, 
                class_weight='balanced'
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(20, 10),  # Smaller network for small data
                max_iter=2000,
                alpha=0.01,  # Regularization
                random_state=42
            ),
            'SVM': SVC(
                C=1.0,
                gamma='scale',
                kernel='rbf',
                probability=True, 
                random_state=42, 
                class_weight='balanced'
            )
        }
        
        # Train and evaluate models
        results = {}
        
        for model_name, model in models.items():
            print(f"\n--- {model_name} ---")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                print(f"Accuracy: {accuracy:.3f}")
                print(f"Precision: {precision:.3f}")
                print(f"Recall: {recall:.3f}")
                print(f"F1-Score: {f1:.3f}")
                
                # AUC only if we have both classes in test set
                if y_pred_proba is not None and len(np.unique(y_test)) > 1:
                    try:
                        auc = roc_auc_score(y_test, y_pred_proba)
                        print(f"AUC-ROC: {auc:.3f}")
                    except ValueError:
                        auc = None
                        print("AUC-ROC: Not available (single class)")
                else:
                    auc = None
                
                # Cross-validation (if we have enough samples)
                if min(class_counts) >= 3:
                    try:
                        cv_scores = cross_val_score(model, X, y, cv=min(3, min(class_counts)), 
                                                  scoring='accuracy')
                        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
                    except Exception as e:
                        print(f"CV not available: {e}")
                        cv_mean, cv_std = None, None
                else:
                    cv_mean, cv_std = None, None
                
                # Store results
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred, zero_division=0)
                }
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance = list(zip(X.columns, importances))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    results[model_name]['feature_importance'] = feature_importance[:10]
                    
                    print("Top 10 Important SEEC Features:")
                    for feat, imp in feature_importance[:10]:
                        print(f"  {feat}: {imp:.4f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        self.results[dataset_name] = results
        return results
    
    def apply_clustering(self, dataset_name):
        """Apply clustering analysis to SEEC data"""
        print(f"\n--- SEEC CLUSTERING ANALYSIS: {dataset_name.upper()} ---")
        
        X, y, subject_ids = self.preprocess_data(dataset_name)
        
        # Apply PCA for dimensionality reduction and visualization
        n_components = min(2, X.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        
        # K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Evaluate clustering
        silhouette = silhouette_score(X, cluster_labels)
        ari = adjusted_rand_score(y, cluster_labels)
        
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Adjusted Rand Index: {ari:.3f}")
        
        # Cluster composition
        print("Cluster composition:")
        for cluster in [0, 1]:
            mask = cluster_labels == cluster
            cluster_subjects = subject_ids[mask]
            cluster_labels_true = y[mask]
            dyslexic_count = sum(cluster_labels_true)
            normal_count = len(cluster_labels_true) - dyslexic_count
            print(f"  Cluster {cluster}: {len(cluster_subjects)} subjects")
            print(f"    - Dyslexic: {dyslexic_count}")
            print(f"    - Normal: {normal_count}")
        
        # Store clustering results
        if dataset_name not in self.results:
            self.results[dataset_name] = {}
        
        self.results[dataset_name]['clustering'] = {
            'silhouette_score': silhouette,
            'adjusted_rand_index': ari,
            'cluster_labels': cluster_labels,
            'pca_components': X_pca,
            'true_labels': y.values,
            'explained_variance_ratio': pca.explained_variance_ratio_
        }
        
        return cluster_labels, X_pca
    
    def create_visualizations(self, dataset_name):
        """Create visualizations for SEEC results"""
        print(f"\nCreating SEEC visualizations for {dataset_name}...")
        
        if dataset_name not in self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'SEEC Dyslexia Prediction Results - {dataset_name.upper()}', fontsize=16)
        
        results = self.results[dataset_name]
        
        # 1. Model Comparison
        model_names = [name for name in ['Random Forest', 'MLP', 'SVM'] if name in results]
        if model_names:
            accuracies = [results[name]['accuracy'] for name in model_names]
            f1_scores = [results[name]['f1_score'] for name in model_names]
            
            x = np.arange(len(model_names))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
            axes[0, 0].bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
            axes[0, 0].set_xlabel('Models')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_title('SEEC Model Performance Comparison')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(model_names, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Feature Importance (Random Forest)
        if 'Random Forest' in results and 'feature_importance' in results['Random Forest']:
            features, importances = zip(*results['Random Forest']['feature_importance'])
            axes[0, 1].barh(range(len(features)), importances)
            axes[0, 1].set_yticks(range(len(features)))
            axes[0, 1].set_yticklabels([f'Feature_{i+1}' for i in range(len(features))])
            axes[0, 1].set_xlabel('Importance')
            axes[0, 1].set_title('Top 10 SEEC Feature Importances')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confusion Matrix (best performing model)
        if model_names:
            best_model = max(model_names, key=lambda x: results[x]['accuracy'])
            cm = results[best_model]['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('Actual')
            axes[1, 0].set_title(f'Confusion Matrix - {best_model}')
        
        # 4. Clustering Visualization
        if 'clustering' in results:
            cluster_data = results['clustering']
            X_pca = cluster_data['pca_components']
            cluster_labels = cluster_data['cluster_labels']
            true_labels = cluster_data['true_labels']
            
            if X_pca.shape[1] >= 2:
                # Plot clusters
                scatter = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                                           cmap='viridis', alpha=0.6, s=50)
                axes[1, 1].set_xlabel(f'PC1 ({cluster_data["explained_variance_ratio"][0]:.3f})')
                axes[1, 1].set_ylabel(f'PC2 ({cluster_data["explained_variance_ratio"][1]:.3f})')
                axes[1, 1].set_title('SEEC K-means Clustering (PCA projection)')
                
                # Add legend
                axes[1, 1].legend(*scatter.legend_elements(), title="Clusters")
            else:
                axes[1, 1].text(0.5, 0.5, 'PCA visualization\nnot available\n(insufficient dimensions)', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('SEEC Clustering')
        
        plt.tight_layout()
        
        # Save plot
        output_dir = os.path.join(os.path.dirname(self.feat_csv_dir), 'results', 'seec_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        plot_path = os.path.join(output_dir, f'seec_{dataset_name}_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {plot_path}")
        
        plt.show()
        plt.close()
    
    def generate_summary_report(self):
        """Generate a comprehensive SEEC summary report"""
        print(f"\n{'='*60}")
        print("SEEC DYSLEXIA PREDICTION SUMMARY REPORT")
        print(f"{'='*60}")
        
        report_lines = []
        report_lines.append("SEEC (SENTENCE-LEVEL EYE-TRACKING & COGNITIVE) DYSLEXIA ANALYSIS")
        report_lines.append("="*65)
        report_lines.append("")
        report_lines.append("Focus: Eye-tracking, gaze patterns, saccades, and cognitive measures")
        report_lines.append("Excluded: Raw electrode EEG signals")
        report_lines.append("")
        
        for dataset_name, results in self.results.items():
            report_lines.append(f"Dataset: {dataset_name.upper()}")
            report_lines.append("-" * 30)
            
            # Machine Learning Results
            model_names = [name for name in ['Random Forest', 'MLP', 'SVM'] if name in results]
            if model_names:
                report_lines.append("\nMACHINE LEARNING RESULTS:")
                
                best_model = None
                best_accuracy = 0
                
                for model_name in model_names:
                    res = results[model_name]
                    report_lines.append(f"\n{model_name}:")
                    report_lines.append(f"  Accuracy: {res['accuracy']:.3f}")
                    report_lines.append(f"  Precision: {res['precision']:.3f}")
                    report_lines.append(f"  Recall: {res['recall']:.3f}")
                    report_lines.append(f"  F1-Score: {res['f1_score']:.3f}")
                    if res['auc']:
                        report_lines.append(f"  AUC-ROC: {res['auc']:.3f}")
                    if res['cv_mean'] is not None:
                        report_lines.append(f"  CV Accuracy: {res['cv_mean']:.3f} ± {res['cv_std']:.3f}")
                    
                    if res['accuracy'] > best_accuracy:
                        best_accuracy = res['accuracy']
                        best_model = model_name
                
                if best_model:
                    report_lines.append(f"\nBest SEEC model: {best_model} (Accuracy: {best_accuracy:.3f})")
            
            # Clustering Results
            if 'clustering' in results:
                cluster_res = results['clustering']
                report_lines.append("\nSEEC CLUSTERING ANALYSIS:")
                report_lines.append(f"  Silhouette Score: {cluster_res['silhouette_score']:.3f}")
                report_lines.append(f"  Adjusted Rand Index: {cluster_res['adjusted_rand_index']:.3f}")
                if 'explained_variance_ratio' in cluster_res:
                    evr = cluster_res['explained_variance_ratio']
                    report_lines.append(f"  PCA Variance Explained: {evr.sum():.3f}")
            
            report_lines.append("\n" + "="*50)
        
        # Add interpretation
        report_lines.append("\nINTERPRETATION:")
        report_lines.append("- SEEC features focus on reading behavior patterns")
        report_lines.append("- Eye-tracking metrics capture visual attention and processing")
        report_lines.append("- Gaze/saccade patterns may reveal dyslexic reading difficulties")
        report_lines.append("- Higher-level cognitive features complement low-level eye movements")
        
        # Print and save report
        report_text = "\n".join(report_lines)
        print(report_text)
        
        # Save report to file
        output_dir = os.path.join(os.path.dirname(self.feat_csv_dir), 'results', 'seec_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'seec_dyslexia_prediction_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"\nSEEC report saved to: {report_path}")
    
    def run_seec_analysis(self):
        """Run the complete SEEC analysis pipeline"""
        print("Starting SEEC Dyslexia Prediction Analysis")
        print("Focus: Sentence-level Eye-tracking, EEG, and Cognitive features")
        print("="*60)
        
        # Load SEEC data only
        datasets = self.load_seec_data()
        
        if not datasets:
            print("No SEEC datasets loaded. Please check the CSV files.")
            return
        
        # Analyze each SEEC dataset
        for dataset_name in datasets.keys():
            print(f"\nProcessing SEEC {dataset_name}...")
            
            # Apply machine learning models
            self.apply_machine_learning_models(dataset_name)
            
            # Apply clustering
            self.apply_clustering(dataset_name)
            
            # Create visualizations
            self.create_visualizations(dataset_name)
        
        # Generate summary report
        self.generate_summary_report()
        
        print("\n✅ SEEC analysis finished!")
        print("Check the results/seec_analysis directory for detailed outputs.")

def main():
    """Main function to run SEEC analysis"""
    # Create SEEC pipeline
    pipeline = SEECDyslexiaPrediction()
    
    # Run complete SEEC analysis
    pipeline.run_seec_analysis()

if __name__ == "__main__":
    main()
