"""
Dyslexia Labeling Module

This module handles the creation and management of dyslexia labels
for the ZuCo benchmark dataset adaptation.
"""

import numpy as np
import config

def load_dyslexia_labels():
    """
    Load pre-computed dyslexia labels from file
    """
    try:
        dyslexia_labels = np.load('../results/dyslexia_labels.npy', allow_pickle=True).item()
        print(f"Loaded dyslexia labels for {len(dyslexia_labels)} subjects")
        return dyslexia_labels
    except FileNotFoundError:
        print("Dyslexia labels not found. Please run analyze_eeg_performance.py first.")
        return None

def get_dyslexia_label_for_subject(subject, dyslexia_labels=None):
    """
    Get dyslexia label for a specific subject
    
    Args:
        subject: Subject ID (e.g., 'YAC', 'XBB')
        dyslexia_labels: Dictionary of labels, will load if None
    
    Returns:
        int: 1 for dyslexic, 0 for normal
    """
    if dyslexia_labels is None:
        dyslexia_labels = load_dyslexia_labels()
    
    if dyslexia_labels is None:
        raise ValueError("Could not load dyslexia labels")
    
    return dyslexia_labels.get(subject, 0)  # Default to normal if not found

def create_sentence_level_dyslexia_labels(features_dict, dyslexia_labels=None):
    """
    Convert subject-level dyslexia labels to sentence-level labels
    
    Args:
        features_dict: Dictionary of features with structure {subject: {sample_id: data}}
        dyslexia_labels: Dictionary mapping subjects to dyslexia labels
    
    Returns:
        Updated features dictionary with dyslexia labels
    """
    if dyslexia_labels is None:
        dyslexia_labels = load_dyslexia_labels()
    
    if dyslexia_labels is None:
        raise ValueError("Could not load dyslexia labels")
    
    updated_features = {}
    
    for subject, subject_data in features_dict.items():
        subject_dyslexia_label = dyslexia_labels.get(subject, 0)
        updated_features[subject] = {}
        
        for sample_id, sample_data in subject_data.items():
            # Copy original data and update label
            updated_sample = sample_data.copy()
            updated_sample['label'] = subject_dyslexia_label
            updated_sample['label_name'] = 'dyslexic' if subject_dyslexia_label == 1 else 'normal'
            updated_sample['original_label'] = sample_data.get('label', None)  # Keep original for reference
            
            updated_features[subject][sample_id] = updated_sample
    
    return updated_features

def get_dyslexia_statistics(dyslexia_labels=None):
    """
    Get statistics about dyslexia labeling
    """
    if dyslexia_labels is None:
        dyslexia_labels = load_dyslexia_labels()
    
    if dyslexia_labels is None:
        return None
    
    dyslexic_count = sum(1 for label in dyslexia_labels.values() if label == 1)
    normal_count = sum(1 for label in dyslexia_labels.values() if label == 0)
    
    dyslexic_subjects = [subject for subject, label in dyslexia_labels.items() if label == 1]
    normal_subjects = [subject for subject, label in dyslexia_labels.items() if label == 0]
    
    stats = {
        'total_subjects': len(dyslexia_labels),
        'dyslexic_count': dyslexic_count,
        'normal_count': normal_count,
        'dyslexic_subjects': dyslexic_subjects,
        'normal_subjects': normal_subjects,
        'dyslexic_percentage': (dyslexic_count / len(dyslexia_labels)) * 100
    }
    
    return stats

def print_dyslexia_summary():
    """
    Print a summary of dyslexia labeling
    """
    stats = get_dyslexia_statistics()
    
    if stats is None:
        print("No dyslexia labels found. Please run analyze_eeg_performance.py first.")
        return
    
    print("\n" + "="*50)
    print("DYSLEXIA LABELING SUMMARY")
    print("="*50)
    print(f"Total subjects: {stats['total_subjects']}")
    print(f"Dyslexic subjects: {stats['dyslexic_count']} ({stats['dyslexic_percentage']:.1f}%)")
    print(f"Normal subjects: {stats['normal_count']}")
    print(f"\nDyslexic subjects: {stats['dyslexic_subjects']}")
    print(f"Normal subjects: {stats['normal_subjects']}")

# For backward compatibility - create default labels if analysis hasn't been run
DEFAULT_DYSLEXIA_LABELS = {
    # Based on typical low performers in EEG studies - these are placeholders
    # Real labels should come from analyze_eeg_performance.py
    'XBD': 1,  # Dyslexic
    'YIS': 1,  # Dyslexic  
    'YSD': 1,  # Dyslexic
}

def get_default_dyslexia_labels():
    """
    Get default dyslexia labels if analysis hasn't been run yet
    """
    all_subjects = config.subjects + config.heldout_subjects
    labels = {}
    
    for subject in all_subjects:
        labels[subject] = DEFAULT_DYSLEXIA_LABELS.get(subject, 0)
    
    return labels

if __name__ == "__main__":
    print_dyslexia_summary()
