"""
Machine Learning Pipeline for Dyslexia Prediction
Applies MLP, Random Forest, SVM, and clustering to CSV feature data
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

class DyslexiaPredictionPipeline:
    def __init__(self, feat_csv_dir="c:\\Projects\\zuco-benchmark\\feat_csv"):
        self.feat_csv_dir = feat_csv_dir
        self.scalers = {}
        self.models = {}
        self.feature_importances = {}
        self.results = {}
        
    def load_data(self):
        """Load and combine all CSV files by feature type"""
        print("Loading CSV data...")
        
        # Get all CSV files
        csv_files = glob.glob(os.path.join(self.feat_csv_dir, "*.csv"))
        
        # Group files by feature type
        feature_types = {
            'electrode_features': [],
            'sent_gaze_sacc': [],
            'sent_gaze_sacc_eeg_means': []
        }
        
        for file in csv_files:
            filename = os.path.basename(file)
            if 'electrode_features_all_detailed' in filename:
                feature_types['electrode_features'].append(file)
            elif 'sent_gaze_sacc_eeg_means_detailed' in filename:
                feature_types['sent_gaze_sacc_eeg_means'].append(file)
            elif 'sent_gaze_sacc_detailed' in filename:
                feature_types['sent_gaze_sacc'].append(file)
        
        # Load each feature type
        self.datasets = {}
        
        for feat_type, files in feature_types.items():
            print(f"Loading {feat_type} features from {len(files)} files...")
            dfs = []
            
            for file in files:
                df = pd.read_csv(file)
                dfs.append(df)
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                self.datasets[feat_type] = combined_df
                print(f"  - {feat_type}: {combined_df.shape}")
                print(f"  - Classes: {combined_df['Dyslexia_Class'].value_counts()}")
        
        return self.datasets
    
    def preprocess_data(self, dataset_name):
        """Preprocess a specific dataset"""
        print(f"\nPreprocessing {dataset_name}...")
        df = self.datasets[dataset_name].copy()
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col.startswith('Feature_')]
        X = df[feature_cols]
        y = df['Dyslexia_Label']
        subject_ids = df['Subject_ID']
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        self.scalers[dataset_name] = scaler
        
        print(f"  - Features shape: {X_scaled.shape}")
        print(f"  - Class distribution: {np.bincount(y)}")
        
        return X_scaled, y, subject_ids
    
    def feature_selection(self, X, y, k=50):
        """Select top k features using univariate statistical tests"""
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()]
        print(f"  - Selected {len(selected_features)} features")
        
        return X_selected, selected_features, selector
    
    def apply_machine_learning_models(self, dataset_name):
        """Apply MLP, Random Forest, and SVM to a dataset"""
        print(f"\n{'='*50}")
        print(f"MACHINE LEARNING ANALYSIS: {dataset_name.upper()}")
        print(f"{'='*50}")
        
        X, y, subject_ids = self.preprocess_data(dataset_name)
        
        # Feature selection for high-dimensional data
        if X.shape[1] > 100:
            X_selected, selected_features, selector = self.feature_selection(X, y, k=50)
        else:
            X_selected = X.values
            selected_features = X.columns
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                class_weight='balanced'
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                max_iter=1000, 
                random_state=42
            ),
            'SVM': SVC(
                probability=True, 
                random_state=42, 
                class_weight='balanced'
            )
        }
        
        # Train and evaluate models
        results = {}
        
        for model_name, model in models.items():
            print(f"\n--- {model_name} ---")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1-Score: {f1:.3f}")
            
            if y_pred_proba is not None and len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, y_pred_proba)
                print(f"AUC-ROC: {auc:.3f}")
            else:
                auc = None
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
            print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                if len(selected_features) == len(importances):
                    feature_importance = list(zip(selected_features, importances))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    results[model_name]['feature_importance'] = feature_importance[:10]
                    
                    print("Top 10 Important Features:")
                    for feat, imp in feature_importance[:10]:
                        print(f"  {feat}: {imp:.4f}")
        
        self.results[dataset_name] = results
        return results
    
    def apply_clustering(self, dataset_name):
        """Apply clustering analysis"""
        print(f"\n--- CLUSTERING ANALYSIS: {dataset_name.upper()} ---")
        
        X, y, subject_ids = self.preprocess_data(dataset_name)
        
        # Feature selection for high-dimensional data
        if X.shape[1] > 50:
            X_selected, selected_features, selector = self.feature_selection(X, y, k=20)
        else:
            X_selected = X.values
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_selected)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        cluster_labels = kmeans.fit_predict(X_selected)
        
        # Evaluate clustering
        silhouette = silhouette_score(X_selected, cluster_labels)
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
            'true_labels': y.values
        }
        
        return cluster_labels, X_pca
    
    def create_visualizations(self, dataset_name):
        """Create visualizations for results"""
        print(f"\nCreating visualizations for {dataset_name}...")
        
        if dataset_name not in self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Dyslexia Prediction Results - {dataset_name.upper()}', fontsize=16)
        
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
            axes[0, 0].set_title('Model Performance Comparison')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(model_names)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Feature Importance (Random Forest)
        if 'Random Forest' in results and 'feature_importance' in results['Random Forest']:
            features, importances = zip(*results['Random Forest']['feature_importance'])
            axes[0, 1].barh(range(len(features)), importances)
            axes[0, 1].set_yticks(range(len(features)))
            axes[0, 1].set_yticklabels([f[:15] for f in features])  # Truncate long feature names
            axes[0, 1].set_xlabel('Importance')
            axes[0, 1].set_title('Top 10 Feature Importances (Random Forest)')
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
            
            # Plot clusters
            scatter = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                                       cmap='viridis', alpha=0.6, s=50)
            axes[1, 1].set_xlabel('First Principal Component')
            axes[1, 1].set_ylabel('Second Principal Component')
            axes[1, 1].set_title('K-means Clustering (PCA projection)')
            
            # Add legend
            axes[1, 1].legend(*scatter.legend_elements(), title="Clusters")
        
        plt.tight_layout()
        
        # Save plot
        output_dir = os.path.join(os.path.dirname(self.feat_csv_dir), 'results', 'ml_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        plot_path = os.path.join(output_dir, f'{dataset_name}_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {plot_path}")
        
        plt.show()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE DYSLEXIA PREDICTION SUMMARY REPORT")
        print(f"{'='*60}")
        
        report_lines = []
        report_lines.append("DYSLEXIA PREDICTION ANALYSIS SUMMARY")
        report_lines.append("="*50)
        report_lines.append("")
        
        for dataset_name, results in self.results.items():
            report_lines.append(f"Dataset: {dataset_name.upper()}")
            report_lines.append("-" * 30)
            
            # Machine Learning Results
            if any(key in results for key in ['Random Forest', 'MLP', 'SVM']):
                report_lines.append("\nMACHINE LEARNING RESULTS:")
                
                best_model = None
                best_accuracy = 0
                
                for model_name in ['Random Forest', 'MLP', 'SVM']:
                    if model_name in results:
                        res = results[model_name]
                        report_lines.append(f"\n{model_name}:")
                        report_lines.append(f"  Accuracy: {res['accuracy']:.3f}")
                        report_lines.append(f"  Precision: {res['precision']:.3f}")
                        report_lines.append(f"  Recall: {res['recall']:.3f}")
                        report_lines.append(f"  F1-Score: {res['f1_score']:.3f}")
                        if res['auc']:
                            report_lines.append(f"  AUC-ROC: {res['auc']:.3f}")
                        report_lines.append(f"  CV Accuracy: {res['cv_mean']:.3f} ± {res['cv_std']:.3f}")
                        
                        if res['accuracy'] > best_accuracy:
                            best_accuracy = res['accuracy']
                            best_model = model_name
                
                if best_model:
                    report_lines.append(f"\nBest performing model: {best_model} (Accuracy: {best_accuracy:.3f})")
            
            # Clustering Results
            if 'clustering' in results:
                cluster_res = results['clustering']
                report_lines.append("\nCLUSTERING ANALYSIS:")
                report_lines.append(f"  Silhouette Score: {cluster_res['silhouette_score']:.3f}")
                report_lines.append(f"  Adjusted Rand Index: {cluster_res['adjusted_rand_index']:.3f}")
            
            report_lines.append("\n" + "="*50)
        
        # Print and save report
        report_text = "\n".join(report_lines)
        print(report_text)
        
        # Save report to file
        output_dir = os.path.join(os.path.dirname(self.feat_csv_dir), 'results', 'ml_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'dyslexia_prediction_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"\nFull report saved to: {report_path}")
    
    def run_complete_analysis(self):
        """Run the complete machine learning analysis pipeline"""
        print("Starting Comprehensive Dyslexia Prediction Analysis")
        print("="*60)
        
        # Load data
        datasets = self.load_data()
        
        if not datasets:
            print("No datasets loaded. Please check the CSV files.")
            return
        
        # Analyze each dataset
        for dataset_name in datasets.keys():
            print(f"\nProcessing {dataset_name}...")
            
            # Apply machine learning models
            self.apply_machine_learning_models(dataset_name)
            
            # Apply clustering
            self.apply_clustering(dataset_name)
            
            # Create visualizations
            self.create_visualizations(dataset_name)
        
        # Generate summary report
        self.generate_summary_report()
        
        print("\n✅ Complete analysis finished!")
        print("Check the results directory for detailed outputs.")

def main():
    """Main function to run the analysis"""
    # Create pipeline
    pipeline = DyslexiaPredictionPipeline()
    
    # Run complete analysis
    pipeline.run_complete_analysis()

if __name__ == "__main__":
    main()
