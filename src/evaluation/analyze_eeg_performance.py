"""
Analyze EEG Performance Scores to Identify Dyslexia Candidates

This script analyzes EEG performance across subjects to identify
the 3 lowest-performing subjects for dyslexia labeling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import config

def load_and_analyze_eeg_features():
    """
    Load EEG features and compute performance metrics for each subject
    """
    eeg_performance = {}
    
    # Get all subjects (training + heldout)
    all_subjects = config.subjects + config.heldout_subjects
    
    for subject in all_subjects:
        try:
            # Load EEG electrode features
            eeg_path = f"../features/{subject}_electrode_features_all.npy"
            eeg_data = np.load(eeg_path, allow_pickle=True).item()
            
            # Extract features and compute performance metrics
            features_list = []
            labels_list = []
            
            for sample_id, data in eeg_data.items():
                # Data is a list where features are all elements except the last (which is label)
                features = np.array(data[:-1])  # All except last element
                label = data[-1]  # Last element is label
                
                features_list.append(features)
                labels_list.append(label)
            
            if features_list:
                features_array = np.array(features_list)
                
                # Compute performance metrics
                # 1. Mean signal strength across all electrodes and samples
                mean_signal = np.mean(np.abs(features_array))
                
                # 2. Signal variance (higher = more active brain)
                signal_variance = np.var(features_array)
                
                # 3. Signal-to-noise ratio approximation
                signal_power = np.mean(features_array ** 2)
                
                # 4. Number of samples (reading efficiency)
                n_samples = len(features_list)
                
                # 5. Feature stability (lower std across samples = more consistent)
                feature_stability = 1 / (1 + np.mean(np.std(features_array, axis=0)))
                
                # Composite EEG performance score
                # Lower score = potentially dyslexic (poor neural efficiency)
                performance_score = (signal_variance * 0.3 + 
                                   mean_signal * 0.25 + 
                                   signal_power * 0.2 + 
                                   (n_samples / 100) * 0.1 +
                                   feature_stability * 0.15)
                
                eeg_performance[subject] = {
                    'performance_score': performance_score,
                    'mean_signal': mean_signal,
                    'signal_variance': signal_variance,
                    'signal_power': signal_power,
                    'n_samples': n_samples,
                    'feature_stability': feature_stability
                }
                
                print(f"Subject {subject}: Performance Score = {performance_score:.6f}, Samples = {n_samples}")
                
        except Exception as e:
            print(f"Error loading data for subject {subject}: {e}")
    
    return eeg_performance

def identify_dyslexia_subjects(eeg_performance):
    """
    Identify the 3 subjects with lowest EEG performance as dyslexic
    """
    # Sort subjects by performance score (ascending - lowest first)
    sorted_subjects = sorted(eeg_performance.items(), 
                           key=lambda x: x[1]['performance_score'])
    
    print("\n" + "="*60)
    print("EEG PERFORMANCE RANKING (Lowest to Highest)")
    print("="*60)
    
    dyslexia_labels = {}
    
    for i, (subject, metrics) in enumerate(sorted_subjects):
        is_dyslexic = i < 3  # First 3 are dyslexic
        dyslexia_labels[subject] = 1 if is_dyslexic else 0
        
        status = "DYSLEXIC" if is_dyslexic else "NORMAL"
        print(f"{i+1:2d}. {subject}: {metrics['performance_score']:.6f} - {status}")
    
    print("\n" + "="*60)
    print("DYSLEXIA LABELING SUMMARY")
    print("="*60)
    
    dyslexic_subjects = [s for s, label in dyslexia_labels.items() if label == 1]
    normal_subjects = [s for s, label in dyslexia_labels.items() if label == 0]
    
    print(f"Dyslexic subjects (3): {dyslexic_subjects}")
    print(f"Normal subjects ({len(normal_subjects)}): {normal_subjects}")
    
    return dyslexia_labels, sorted_subjects

def plot_performance_distribution(eeg_performance, dyslexia_labels):
    """
    Plot EEG performance distribution with dyslexia labels
    """
    subjects = list(eeg_performance.keys())
    scores = [eeg_performance[s]['performance_score'] for s in subjects]
    colors = ['red' if dyslexia_labels[s] == 1 else 'blue' for s in subjects]
    
    plt.figure(figsize=(12, 8))
    
    # Main distribution plot
    plt.subplot(2, 1, 1)
    bars = plt.bar(range(len(subjects)), scores, color=colors, alpha=0.7)
    plt.xlabel('Subjects')
    plt.ylabel('EEG Performance Score')
    plt.title('EEG Performance Scores by Subject\n(Red = Dyslexic, Blue = Normal)')
    plt.xticks(range(len(subjects)), subjects, rotation=45)
    
    # Add threshold line
    threshold_idx = 2.5  # Between 3rd and 4th subject
    plt.axvline(x=threshold_idx, color='black', linestyle='--', 
                label='Dyslexia Threshold')
    plt.legend()
    
    # Box plot comparison
    plt.subplot(2, 1, 2)
    dyslexic_scores = [scores[i] for i, s in enumerate(subjects) if dyslexia_labels[s] == 1]
    normal_scores = [scores[i] for i, s in enumerate(subjects) if dyslexia_labels[s] == 0]
    
    plt.boxplot([dyslexic_scores, normal_scores], 
                labels=['Dyslexic (n=3)', f'Normal (n={len(normal_scores)})'])
    plt.ylabel('EEG Performance Score')
    plt.title('Performance Score Distribution by Group')
    
    plt.tight_layout()
    plt.savefig('../results/eeg_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_dyslexia_labels(dyslexia_labels):
    """
    Save dyslexia labels to files for use in other scripts
    """
    # Save as numpy file
    np.save('../results/dyslexia_labels.npy', dyslexia_labels)
    
    # Save as CSV for human inspection
    df = pd.DataFrame([(subject, label) for subject, label in dyslexia_labels.items()],
                     columns=['Subject', 'Dyslexia_Label'])
    df.to_csv('../results/dyslexia_labels.csv', index=False)
    
    print(f"\nDyslexia labels saved to:")
    print(f"  - ../results/dyslexia_labels.npy")
    print(f"  - ../results/dyslexia_labels.csv")

def main():
    print("Analyzing EEG Performance for Dyslexia Prediction...")
    print("="*60)
    
    # Step 1: Load and analyze EEG features
    eeg_performance = load_and_analyze_eeg_features()
    
    if not eeg_performance:
        print("No EEG data found! Please ensure features are extracted first.")
        return
    
    # Step 2: Identify dyslexia subjects
    dyslexia_labels, sorted_subjects = identify_dyslexia_subjects(eeg_performance)
    
    # Step 3: Plot performance distribution
    plot_performance_distribution(eeg_performance, dyslexia_labels)
    
    # Step 4: Save labels for use in other scripts
    save_dyslexia_labels(dyslexia_labels)
    
    return dyslexia_labels, eeg_performance

if __name__ == "__main__":
    dyslexia_labels, performance_data = main()
