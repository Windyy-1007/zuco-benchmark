"""
---------------------------------------------------------------
Dyslexia Prediction Benchmark
---------------------------------------------------------------

Description:
    This script performs dyslexia prediction using EEG and eye-tracking
    features from the ZuCo dataset. It classifies subjects as dyslexic
    or normal based on their neural responses during reading.

Usage:
    python dyslexia_benchmark.py

Note:
    Please ensure dyslexia labels have been generated using 
    analyze_eeg_performance.py before running this script.
"""

import time
from datetime import timedelta
import numpy as np
import pandas as pd
import config
import data_helpers as dh
import classifier as clf
import dyslexia_labels as dl
from sklearn.decomposition import PCA

def main():
    start = time.time()
    
    print("="*60)
    print("DYSLEXIA PREDICTION BENCHMARK")
    print("="*60)
    
    # Ensure directories exist
    dh.ensure_dir_exists("results")
    
    # Load and verify dyslexia labels
    print("\n1. Loading dyslexia labels...")
    dyslexia_labels = dl.load_dyslexia_labels()
    if dyslexia_labels is None:
        print("Warning: Dyslexia labels not found. Please run analyze_eeg_performance.py first.")
        print("Using default dyslexia labels for demonstration.")
        dyslexia_labels = dl.get_default_dyslexia_labels()
    
    dl.print_dyslexia_summary()
    
    # Load training data
    print("\n2. Loading training data...")
    train_data = dh.get_or_extract_features(config.subjects, config.rootdir, train_feats=True)
    
    # Load test data 
    print("\n3. Loading test data...")
    test_data = dh.get_or_extract_features(config.heldout_subjects, config.heldout_dir, train_feats=False)
    
    # Prepare output file
    print("\n4. Preparing output files...")
    result_file = dh.prepare_output_file()
    
    print(f"\n5. Running classification with {len(config.feature_sets)} feature sets...")
    all_results = {}
    
    for feature_set in config.feature_sets:
        print(f"\n" + "="*50)
        print(f"Processing feature set: {feature_set}")
        print("="*50)
        
        # Get training features and labels
        train_X = train_data['features'][feature_set]
        train_y = train_data['labels'][feature_set] 
        
        # Get test features and labels
        test_X = test_data['features'][feature_set]
        test_y = test_data['labels'][feature_set]
        
        print(f"Training subjects: {len(train_X)}")
        print(f"Test subjects: {len(test_X)}")
        
        # Check data balance
        train_labels_flat = [label for subj_labels in train_y.values() for label in subj_labels]
        dyslexic_count = sum(1 for label in train_labels_flat if label == 1)
        normal_count = len(train_labels_flat) - dyslexic_count
        print(f"Training data balance: {dyslexic_count} dyslexic, {normal_count} normal samples")
        
        # Apply PCA if configured
        if config.pca_preprocessing and 'electrode' in feature_set:
            print(f"Applying PCA (explained variance: {config.explained_variance})...")
            # Prepare data for PCA
            all_train_X = []
            for subj_data in train_X.values():
                all_train_X.extend(subj_data)
            
            pca = PCA()
            pca.fit(all_train_X)
            
            # Find number of components for desired explained variance
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= config.explained_variance) + 1
            
            print(f"Using {n_components} PCA components (explains {cumsum[n_components-1]:.3f} variance)")
            
            # Apply PCA transformation
            pca_final = PCA(n_components=n_components)
            pca_final.fit(all_train_X)
            
            # Transform training data
            for subj in train_X:
                train_X[subj] = pca_final.transform(train_X[subj]).tolist()
            
            # Transform test data
            for subj in test_X:
                test_X[subj] = pca_final.transform(test_X[subj]).tolist()
        
        # Run classification
        print(f"Running SVM classification...")
        if config.bootstrap and any(len(labels) > 0 for labels in test_y.values()):
            # If we have true labels, use full benchmark with evaluation
            results = clf.benchmark(train_X, train_y, test_X, test_y)
            predictions, bootstrap_results = results[0], results[1] if len(results) > 1 else [[]]
            
            # Extract metrics
            pred_lists, accuracies, f1_scores, precisions, recalls = predictions
            
            # Print and save results
            for i, subj in enumerate(config.heldout_subjects):
                accuracy, f1, precision, recall = accuracies[i], f1_scores[i], precisions[i], recalls[i]
                
                print(f"Subject {subj}: Acc={accuracy:.3f}, F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}")
                
                # Save to file
                result_file.write(f"{subj} {feature_set} accuracy {accuracy} 0.0\n")
                result_file.write(f"{subj} {feature_set} f1 {f1} 0.0\n")
                result_file.write(f"{subj} {feature_set} precision {precision} 0.0\n")
                result_file.write(f"{subj} {feature_set} recall {recall} 0.0\n")
                
                # Store predictions for submission
                if config.create_submission:
                    all_results[f"{subj}_{feature_set}"] = pred_lists[i][0]
        else:
            # Baseline prediction without evaluation
            print("Running baseline prediction (no true labels available)...")
            results = clf.benchmark_baseline(train_X, train_y, test_X, test_y)
            
            # Store predictions for submission
            if config.create_submission:
                for i, subj in enumerate(config.heldout_subjects):
                    all_results[f"{subj}_{feature_set}"] = results[i]
                    print(f"Subject {subj}: {len(results[i])} predictions generated")
    
    # Create submission files if requested
    if config.create_submission:
        print(f"\n6. Creating submission files...")
        
        # Save predictions as numpy file
        if config.save_prediction_npy:
            submission_path = f"submissions/dyslexia_predictions_{config.dataset}.npy"
            dh.ensure_dir_exists("submissions")
            np.save(submission_path, all_results)
            print(f"Saved predictions to: {submission_path}")
        
        # Save predictions as CSV for human inspection
        submission_df = []
        for key, predictions in all_results.items():
            subject, feature_set = key.rsplit('_', 1)
            for i, pred in enumerate(predictions):
                submission_df.append({
                    'Subject': subject,
                    'Feature_Set': feature_set,
                    'Sample_Index': i,
                    'Prediction': int(pred),
                    'Prediction_Label': 'Dyslexic' if pred == 1 else 'Normal'
                })
        
        submission_df = pd.DataFrame(submission_df)
        csv_path = f"submissions/dyslexia_predictions_{config.dataset}.csv"
        submission_df.to_csv(csv_path, index=False)
        print(f"Saved predictions to: {csv_path}")
        
        # Summary statistics
        print(f"\nPrediction Summary:")
        summary = submission_df.groupby(['Subject', 'Feature_Set'])['Prediction'].agg(['count', 'sum', 'mean']).round(3)
        summary.columns = ['Total_Samples', 'Dyslexic_Predictions', 'Dyslexic_Ratio']
        print(summary)
    
    result_file.close()
    
    elapsed = time.time() - start
    print(f"\n" + "="*60)
    print(f"DYSLEXIA PREDICTION BENCHMARK COMPLETED")
    print(f"Total runtime: {str(timedelta(seconds=elapsed))}")
    print("="*60)

if __name__ == '__main__':
    main()
