"""
---------------------------------------------------------------
Multi-Algorithm Dyslexia Prediction Benchmark
---------------------------------------------------------------

Description:
    This script performs dyslexia prediction using multiple algorithms:
    1. SVM (Support Vector Machine)
    2. Random Forest
    3. Random Forest + Hierarchical Clustering
    4. CNN (Convolutional Neural Network)

Usage:
    python multi_algorithm_benchmark.py

Configuration:
    Set algorithms_to_run in config.py to control which algorithms to run:
    - "all": Run all available algorithms
    - ["svm"]: Run only SVM
    - ["svm", "random_forest"]: Run SVM and Random Forest
    - etc.
"""

import time
from datetime import timedelta
import numpy as np
import pandas as pd
import config
import dyslexia_labels as dl
from sklearn.decomposition import PCA

# Import all classifier modules
import classifier as clf_svm
import classifier_random_forest as clf_rf
import classifier_rf_clustering as clf_rf_cluster
import classifier_mlp as clf_mlp
# import classifier_cnn as clf_cnn  # Commented out due to TensorFlow issues

def load_features_with_dyslexia_labels(subjects, feature_sets):
    """
    Load pre-extracted features and apply dyslexia labels
    """
    print("Loading dyslexia labels...")
    dyslexia_labels = dl.load_dyslexia_labels()
    if dyslexia_labels is None:
        print("Warning: Using default dyslexia labels")
        dyslexia_labels = dl.get_default_dyslexia_labels()
    
    features_data = {}
    labels_data = {}
    
    for feature_set in feature_sets:
        features_data[feature_set] = {}
        labels_data[feature_set] = {}
        
        for subject in subjects:
            try:
                # Load pre-extracted features
                feature_path = f"../features/{subject}_{feature_set}.npy"
                subject_features = np.load(feature_path, allow_pickle=True).item()
                
                # Apply dyslexia labels to each sample
                subject_dyslexia_label = dyslexia_labels.get(subject, 0)
                
                features_list = []
                labels_list = []
                
                for sample_id, sample_data in subject_features.items():
                    # Extract features (all except last element which is original label)
                    features = sample_data[:-1]
                    features_list.append(features)
                    labels_list.append(subject_dyslexia_label)  # Use dyslexia label instead
                
                features_data[feature_set][subject] = features_list
                labels_data[feature_set][subject] = labels_list
                
                print(f"Loaded {len(features_list)} samples for {subject} ({feature_set}), "
                      f"label: {'Dyslexic' if subject_dyslexia_label == 1 else 'Normal'}")
                
            except FileNotFoundError:
                print(f"Warning: Feature file not found for {subject} - {feature_set}")
            except Exception as e:
                print(f"Error loading {subject} - {feature_set}: {e}")
    
    return features_data, labels_data

def get_algorithm_functions():
    """
    Return dictionary of algorithm functions
    """
    algorithms = {
        "svm": {
            "benchmark": clf_svm.benchmark,
            "baseline": clf_svm.benchmark_baseline,
            "name": "Support Vector Machine"
        },
        "random_forest": {
            "benchmark": clf_rf.benchmark_random_forest,
            "baseline": clf_rf.benchmark_random_forest_baseline,
            "name": "Random Forest"
        },
        "rf_clustering": {
            "benchmark": clf_rf_cluster.benchmark_rf_clustering,
            "baseline": clf_rf_cluster.benchmark_rf_clustering_baseline,
            "name": "Random Forest + Hierarchical Clustering"
        },
        "mlp": {
            "benchmark": clf_mlp.benchmark_mlp,
            "baseline": clf_mlp.benchmark_mlp_baseline,
            "name": "Multi-Layer Perceptron"
        }
        # "cnn": {
        #     "benchmark": clf_cnn.benchmark_cnn,
        #     "baseline": clf_cnn.benchmark_cnn_baseline,
        #     "name": "Convolutional Neural Network"
        # }
    }
    return algorithms

def prepare_output_file(algorithm_name):
    """Create output file for results"""
    from datetime import datetime
    import os
    
    if not os.path.exists("../results"):
        os.makedirs("../results")
    
    filename = ("../results/" + 
                datetime.now().strftime("%Y%m%d-%H%M%S") + 
                f"_{algorithm_name}_results_" + config.class_task + 
                "_" + config.dataset + 
                "_random" + str(config.randomized) + ".csv")
    
    return open(filename, 'w')

def run_algorithm(algorithm_name, algorithm_funcs, train_X, train_y, test_X, test_y, feature_set):
    """
    Run a specific algorithm and return results
    """
    print(f"\n{'='*60}")
    print(f"RUNNING {algorithm_funcs['name'].upper()}")
    print(f"Feature set: {feature_set}")
    print(f"{'='*60}")
    
    try:
        # Check if we have true test labels for evaluation
        test_labels_exist = any(len(labels) > 0 for labels in test_y.values())
        
        if test_labels_exist:
            # Use full benchmark with evaluation
            results = algorithm_funcs["benchmark"](train_X, train_y, test_X, test_y)
            if len(results) > 0 and len(results[0]) > 0:
                predictions, bootstrap_results = results[0], results[1] if len(results) > 1 else [[]]
                
                # Extract metrics if available
                if len(predictions) >= 5:  # pred_lists, accuracies, f1_scores, precisions, recalls
                    pred_lists, accuracies, f1_scores, precisions, recalls = predictions
                    return {
                        "success": True,
                        "predictions": pred_lists,
                        "accuracies": accuracies,
                        "f1_scores": f1_scores,
                        "precisions": precisions,
                        "recalls": recalls,
                        "bootstrap": bootstrap_results
                    }
                else:
                    return {"success": False, "error": "Unexpected result format"}
            else:
                return {"success": False, "error": "Empty results"}
        else:
            # Use baseline prediction without true labels
            results = algorithm_funcs["baseline"](train_X, train_y, test_X, test_y)
            return {
                "success": True,
                "predictions": results,
                "baseline": True
            }
            
    except Exception as e:
        print(f"Error running {algorithm_name}: {e}")
        return {"success": False, "error": str(e)}

def main():
    start = time.time()
    
    print("="*60)
    print("MULTI-ALGORITHM DYSLEXIA PREDICTION BENCHMARK")
    print("="*60)
    
    # Print dyslexia label summary
    dl.print_dyslexia_summary()
    
    # Determine which algorithms to run
    if config.algorithms_to_run == "all":
        algorithms_to_run = config.available_algorithms
    else:
        algorithms_to_run = config.algorithms_to_run
    
    print(f"\nAlgorithms to run: {algorithms_to_run}")
    
    # Get algorithm functions
    algorithm_funcs = get_algorithm_functions()
    
    # Load training data
    print(f"\n1. Loading training data from {len(config.subjects)} subjects...")
    train_features, train_labels = load_features_with_dyslexia_labels(
        config.subjects, config.feature_sets)
    
    # Load test data
    print(f"\n2. Loading test data from {len(config.heldout_subjects)} subjects...")
    test_features, test_labels = load_features_with_dyslexia_labels(
        config.heldout_subjects, config.feature_sets)
    
    # Store all results
    all_results = {}
    algorithm_summaries = {}
    
    print(f"\n3. Running {len(algorithms_to_run)} algorithms on {len(config.feature_sets)} feature sets...")
    
    for algorithm_name in algorithms_to_run:
        if algorithm_name not in algorithm_funcs:
            print(f"Warning: Unknown algorithm {algorithm_name}, skipping...")
            continue
        
        algorithm_summaries[algorithm_name] = {}
        
        # Prepare output file for this algorithm
        result_file = prepare_output_file(algorithm_name)
        
        for feature_set in config.feature_sets:
            print(f"\n{'-'*50}")
            print(f"Processing: {algorithm_name} with {feature_set}")
            print(f"{'-'*50}")
            
            # Get training features and labels
            train_X = train_features[feature_set]
            train_y = train_labels[feature_set]
            
            # Get test features and labels
            test_X = test_features[feature_set]
            test_y = test_labels[feature_set]
            
            if not train_X or not test_X:
                print(f"No data available for {feature_set}, skipping...")
                continue
            
            # Check data balance in training set
            train_labels_flat = [label for subj_labels in train_y.values() for label in subj_labels]
            dyslexic_count = sum(1 for label in train_labels_flat if label == 1)
            normal_count = len(train_labels_flat) - dyslexic_count
            print(f"Training data balance: {dyslexic_count} dyslexic, {normal_count} normal samples")
            
            # Apply PCA if configured and dealing with electrode features
            current_train_X, current_test_X = train_X, test_X
            if config.pca_preprocessing and 'electrode' in feature_set:
                print(f"Applying PCA (explained variance: {config.explained_variance})...")
                
                # Prepare data for PCA
                all_train_X = []
                for subj_data in train_X.values():
                    all_train_X.extend(subj_data)
                
                if len(all_train_X) > 0:
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
                    current_train_X = {}
                    for subj in train_X:
                        current_train_X[subj] = pca_final.transform(train_X[subj]).tolist()
                    
                    # Transform test data
                    current_test_X = {}
                    for subj in test_X:
                        current_test_X[subj] = pca_final.transform(test_X[subj]).tolist()
            
            # Run the algorithm
            results = run_algorithm(algorithm_name, algorithm_funcs[algorithm_name],
                                  current_train_X, train_y, current_test_X, test_y, feature_set)
            
            if results["success"]:
                # Store results
                key = f"{algorithm_name}_{feature_set}"
                all_results[key] = results
                
                # Process and save results
                if "accuracies" in results:  # Full evaluation
                    accuracies = results["accuracies"]
                    f1_scores = results["f1_scores"]
                    precisions = results["precisions"]
                    recalls = results["recalls"]
                    
                    algorithm_summaries[algorithm_name][feature_set] = {
                        "mean_accuracy": np.mean(accuracies),
                        "mean_f1": np.mean(f1_scores),
                        "mean_precision": np.mean(precisions),
                        "mean_recall": np.mean(recalls)
                    }
                    
                    # Save detailed results
                    for i, subj in enumerate(config.heldout_subjects):
                        if i < len(accuracies):
                            result_file.write(f"{subj} {feature_set} accuracy {accuracies[i]} 0.0\n")
                            result_file.write(f"{subj} {feature_set} f1 {f1_scores[i]} 0.0\n")
                            result_file.write(f"{subj} {feature_set} precision {precisions[i]} 0.0\n")
                            result_file.write(f"{subj} {feature_set} recall {recalls[i]} 0.0\n")
                    
                    print(f"\nResults for {algorithm_name} - {feature_set}:")
                    print(f"  Mean Accuracy: {np.mean(accuracies):.3f}")
                    print(f"  Mean F1: {np.mean(f1_scores):.3f}")
                    
                elif "predictions" in results:  # Baseline
                    algorithm_summaries[algorithm_name][feature_set] = {"baseline": True}
                    print(f"Baseline predictions generated for {algorithm_name} - {feature_set}")
            
            else:
                print(f"Failed to run {algorithm_name} on {feature_set}: {results.get('error', 'Unknown error')}")
        
        result_file.close()
    
    # Print algorithm comparison summary
    print(f"\n{'='*60}")
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*60)
    
    comparison_df = []
    for algorithm in algorithm_summaries:
        for feature_set in algorithm_summaries[algorithm]:
            if "mean_accuracy" in algorithm_summaries[algorithm][feature_set]:
                metrics = algorithm_summaries[algorithm][feature_set]
                comparison_df.append({
                    "Algorithm": algorithm,
                    "Feature_Set": feature_set,
                    "Accuracy": metrics["mean_accuracy"],
                    "F1_Score": metrics["mean_f1"],
                    "Precision": metrics["mean_precision"],
                    "Recall": metrics["mean_recall"]
                })
    
    if comparison_df:
        df = pd.DataFrame(comparison_df)
        print(df.round(3).to_string(index=False))
        
        # Save comparison to CSV
        df.to_csv("../results/algorithm_comparison.csv", index=False)
        print(f"\nDetailed comparison saved to: ../results/algorithm_comparison.csv")
        
        # Find best performing algorithm
        best_accuracy = df.loc[df['Accuracy'].idxmax()]
        best_f1 = df.loc[df['F1_Score'].idxmax()]
        
        print(f"\nBest Accuracy: {best_accuracy['Algorithm']} with {best_accuracy['Feature_Set']} ({best_accuracy['Accuracy']:.3f})")
        print(f"Best F1 Score: {best_f1['Algorithm']} with {best_f1['Feature_Set']} ({best_f1['F1_Score']:.3f})")
    
    # Create submission files if requested
    if config.create_submission and all_results:
        print(f"\n4. Creating submission files...")
        
        # Ensure submission directory exists
        import os
        if not os.path.exists("../submissions"):
            os.makedirs("../submissions")
        
        # Combine all predictions
        all_predictions = {}
        for key, results in all_results.items():
            if "predictions" in results:
                if "baseline" in results:  # Baseline results
                    predictions = results["predictions"]
                    for i, subj in enumerate(config.heldout_subjects):
                        if i < len(predictions):
                            all_predictions[f"{subj}_{key}"] = predictions[i]
                else:  # Full evaluation results
                    pred_lists = results["predictions"]
                    for i, subj in enumerate(config.heldout_subjects):
                        if i < len(pred_lists):
                            all_predictions[f"{subj}_{key}"] = pred_lists[i][0]
        
        # Save predictions as CSV
        if all_predictions:
            submission_df = []
            for key, predictions in all_predictions.items():
                parts = key.split('_')
                subject = parts[0]
                algorithm_feature = '_'.join(parts[1:])
                
                for i, pred in enumerate(predictions):
                    submission_df.append({
                        'Subject': subject,
                        'Algorithm_Feature': algorithm_feature,
                        'Sample_Index': i,
                        'Prediction': int(pred),
                        'Prediction_Label': 'Dyslexic' if pred == 1 else 'Normal'
                    })
            
            submission_df = pd.DataFrame(submission_df)
            csv_path = f"../submissions/multi_algorithm_predictions_{config.dataset}.csv"
            submission_df.to_csv(csv_path, index=False)
            print(f"  Saved all predictions to: {csv_path}")
            
            # Summary by algorithm
            summary = submission_df.groupby(['Algorithm_Feature'])['Prediction'].agg(['count', 'sum', 'mean']).round(3)
            summary.columns = ['Total_Samples', 'Dyslexic_Predictions', 'Dyslexic_Ratio']
            print(f"\n  Prediction Summary by Algorithm:")
            print(summary.head(10))
    
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"MULTI-ALGORITHM BENCHMARK COMPLETED")
    print(f"Algorithms run: {algorithms_to_run}")
    print(f"Total runtime: {str(timedelta(seconds=elapsed))}")
    print("="*60)

if __name__ == '__main__':
    main()
