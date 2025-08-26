"""
---------------------------------------------------------------
Perform Leave-One-Out Validation On the Training Subjects
---------------------------------------------------------------

Description:
    This script is can be used to test various classification methods
    using the Leave-One-Out validation approach on the training data. 
    It trains on all except for one subject's data and tests on the left out subject's data.

Usage:
    This script can act as a validation step in benchmark task. 
    It is intended for comparing different models and feature sets.

Note:
    Ensure that you have configured the correct data paths and parameters
    in `config.py` before running the script.
    
"""

import numpy as np
import extract_features as fe  # Custom module for feature extraction
import config  # Custom module for global configurations
import os
import time
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support
import data_helpers as dh  # Custom module for data-related helper functions
from datetime import timedelta
from basic_models.svm_class import classify_svm, classify_svm_pso, classify_svm_rfe
from basic_models.nn_class import classify_knn
from basic_models.other_basic_models import classify_random_forest, classify_logistic_regression, classify_gradient_boosting
from basic_models.voting_class import classify_voting, classify_voting_soft, classify_voting_hard
from tqdm import tqdm


def extract_features():
    """
    Extract or load features from MAT files for all subjects and feature sets defined in config.
    
    Returns:
    dict: A dictionary containing the features extracted or loaded.
    """
    features = {}
    for subject in config.subjects:
        print(f"Extracting features for subject {subject}")
        for feature_set in config.feature_sets:
            features.setdefault(feature_set, {})
            dh.ensure_dir_exists("features")
            feature_path = os.path.join("features", f"{subject}_{feature_set}.npy")
            
            # Check if pre-computed features exist
            if os.path.isfile(feature_path):
                print(f"Loading saved features: {feature_set}")
                features[feature_set].update(np.load(feature_path, allow_pickle='TRUE').item())
            else:
                print(f"Extracting features: {feature_set}")
                f_nr = dh.read_mat_file(os.path.join(config.rootdir, f"results{subject}_NR.mat"))
                f_tsr = dh.read_mat_file(os.path.join(config.rootdir, f"results{subject}_TSR.mat"))
                fe.extract_sentence_features(subject, f_nr, feature_set, features, "NR")
                fe.extract_sentence_features(subject, f_tsr, feature_set, features, "TSR")

                # Save the extracted features
                only_subjects = {subj_feat: features[feature_set][subj_feat] for subj_feat in features[feature_set] if subj_feat.startswith(subject)}
                np.save(feature_path, only_subjects)

    print(f"{len(features[feature_set])} samples collected for {feature_set}")
    return features
    

def prepare_data_splits(samples, test_subject):
    """
    Prepare the training and testing data splits.
    
    Args:
    samples (dict): Dictionary containing feature samples.
    test_subject (str): The subject to be used for testing.
    
    Returns:
    tuple: Training and testing data and labels.
    """
    train_X, train_y, test_X, test_y = [], [], [], []
    
    # Loop through each sample to prepare training and test splits
    for sample_id, features in samples.items():
        subject, label = sample_id.split("_")[0], sample_id.split("_")[1]
        if subject != test_subject:
            train_X.append(features[:-1])
            train_y.append(1 if label == "NR" else 0)
        else:
            test_X.append(features[:-1])
            test_y.append(1 if label == "NR" else 0)

    train_X, train_y = shuffle(train_X, train_y)
    return train_X, train_y, test_X, test_y


def main():
    np.random.seed(config.seed)
    start = time.time()
    features = extract_features()
    
    # Store results for overall statistics
    all_acc, all_f1, all_p, all_r = [], [], [], []
    all_acc_r, all_f1_r, all_p_r, all_r_r = [], [], [], []

    # Loop through each subject and feature set
    with tqdm(total=len(config.subjects) * len(features), desc="Processing subjects and features") as pbar:
        for subject in config.subjects:
            for feature_set, feats in features.items():
                # print(f"\nTraining on all subjects, testing on {subject}")
                train_X, train_y, test_X, test_y = prepare_data_splits(feats, subject)
                # acc, p, r, f1 = classify_svm(train_X, train_y, test_X, test_y)
                # acc, p, r, f1 = classify_knn(train_X, train_y, test_X, test_y)
                acc, p, r, f1 = classify_voting(train_X, train_y, test_X, test_y)
                # acc_r, p_r, r_r, f1_r = classify_svm_rfe(train_X, train_y, test_X, test_y)

                # Print results for individual subjects
                # for values, name in zip([acc, f1, p, r], ['accuracy', 'F1', 'precision', 'recall']):
                #     print(f"Classification {name}: {subject} {feature_set} result={values}")

                # Collect for overall stats
                all_acc.append(acc)
                all_f1.append(f1)
                all_p.append(p)
                all_r.append(r)
                
                # all_acc_r.append(acc_r)
                # all_f1_r.append(f1_r)
                # all_p_r.append(p_r)
                # all_r_r.append(r_r)
                
                pbar.update(1)

    # Print overall results in requested format
    print("== OVERALL 1 =====")
    print(f"Accuracy: {np.mean(all_acc):.4f}")
    print(f"F1: {np.mean(all_f1):.4f}")
    print(f"Precission: {np.mean(all_p):.4f}")
    print(f"Recall: {np.mean(all_r):.4f}")
    # print("== OVERALL 2 =====")
    # print(f"Accuracy: {np.mean(all_acc_r):.4f}")
    # print(f"F1: {np.mean(all_f1_r):.4f}")
    # print(f"Precission: {np.mean(all_p_r):.4f}")
    # print(f"Recall: {np.mean(all_r_r):.4f}")
    print("== DONE ==")
    
    

    elapsed = (time.time() - start)
    print(f"Elapsed time: {str(timedelta(seconds=elapsed))}")
    
    


if __name__ == '__main__':
    main()
