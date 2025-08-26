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

def compute_performance_score(metrics):
    """
    Compute the composite EEG performance score based on updated weights.

    Args:
        metrics (dict): Dictionary containing the following keys:
            - theta_alpha_power
            - connectivity
            - signal_variability
            - complexity
            - erp_markers
            - feature_stability
            - mean_signal

    Returns:
        float: Composite performance score.
    """
    return (
        metrics['theta_alpha_power'] * 0.25 +
        metrics['connectivity'] * 0.20 +
        metrics['signal_variability'] * 0.15 +
        metrics['complexity'] * 0.15 +
        metrics['erp_markers'] * 0.15 +
        metrics['feature_stability'] * 0.10 +
        metrics['mean_signal'] * 0.05
    )

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
            
            # Extract features from all samples for this subject
            all_features = []
            for sample_id, sample_data in eeg_data.items():
                features = np.array(sample_data[:-1])  # Remove label
                all_features.append(features)
            
            all_features = np.array(all_features)  # Shape: (n_samples, n_features)
            
            # Split electrode features into frequency bands (420 features = 105 electrodes x 4 bands)
            band_size = all_features.shape[1] // 4
            theta_features = all_features[:, :band_size]          # 0-104: Theta
            alpha_features = all_features[:, band_size:2*band_size]  # 105-209: Alpha  
            beta_features = all_features[:, 2*band_size:3*band_size]  # 210-314: Beta
            gamma_features = all_features[:, 3*band_size:]        # 315-419: Gamma
            
            # 1. Theta/Alpha Power Ratio (lower ratio indicates better performance)
            theta_power = np.mean(theta_features**2)
            alpha_power = np.mean(alpha_features**2)
            theta_alpha_power = 1.0 / (1.0 + theta_power / (alpha_power + 1e-8))
            
            # 2. Connectivity (measured as correlation between electrodes)
            # Calculate average correlation between electrode pairs within each band
            theta_corr = np.corrcoef(theta_features.T)
            alpha_corr = np.corrcoef(alpha_features.T)
            # Take upper triangle (excluding diagonal) and average
            theta_conn = np.mean(theta_corr[np.triu_indices_from(theta_corr, k=1)])
            alpha_conn = np.mean(alpha_corr[np.triu_indices_from(alpha_corr, k=1)])
            connectivity = (abs(theta_conn) + abs(alpha_conn)) / 2.0
            
            # 3. Signal Variability (coefficient of variation across samples)
            signal_means = np.mean(all_features, axis=0)
            signal_stds = np.std(all_features, axis=0)
            cv = signal_stds / (signal_means + 1e-8)  # Coefficient of variation
            signal_variability = 1.0 / (1.0 + np.mean(cv))  # Inverse for better = higher
            
            # 4. Complexity (using spectral entropy-like measure)
            # Calculate power in each frequency band and use entropy
            band_powers = np.array([
                np.mean(theta_features**2),
                np.mean(alpha_features**2), 
                np.mean(beta_features**2),
                np.mean(gamma_features**2)
            ])
            band_powers_norm = band_powers / (np.sum(band_powers) + 1e-8)
            # Shannon entropy of band powers (higher complexity = more distributed power)
            complexity = -np.sum(band_powers_norm * np.log(band_powers_norm + 1e-8))
            
            # 5. ERP Markers (using beta/gamma activity as cognitive load indicator)
            beta_power = np.mean(beta_features**2)
            gamma_power = np.mean(gamma_features**2) 
            erp_markers = 1.0 / (1.0 + (beta_power + gamma_power) / 2.0)  # Lower = better
            
            # 6. Feature Stability (consistency across samples)
            feature_stability = 1.0 / (1.0 + np.mean(np.std(all_features, axis=0)))
            
            # 7. Mean Signal Strength (overall signal amplitude)
            mean_signal = 1.0 / (1.0 + np.mean(np.abs(all_features)))
            
            # Compute composite performance score
            performance_score = compute_performance_score({
                'theta_alpha_power': theta_alpha_power,
                'connectivity': connectivity,
                'signal_variability': signal_variability,
                'complexity': complexity,
                'erp_markers': erp_markers,
                'feature_stability': feature_stability,
                'mean_signal': mean_signal
            })
            
            eeg_performance[subject] = {
                'performance_score': performance_score,
                'theta_alpha_power': theta_alpha_power,
                'connectivity': connectivity,
                'signal_variability': signal_variability,
                'complexity': complexity,
                'erp_markers': erp_markers,
                'feature_stability': feature_stability,
                'mean_signal': mean_signal
            }
            
            print(f"Subject {subject}: Performance Score = {performance_score:.6f}")
            
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
    
    plt.boxplot([dyslexic_scores, normal_scores])
    plt.xticks([1, 2], ['Dyslexic (n=3)', f'Normal (n={len(normal_scores)})'])
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
