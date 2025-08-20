"""
Add Dyslexia Labels to CSV Files

This script processes all CSV files in the feat_csv/ directory and adds
dyslexia labels based on the analysis from analyze_eeg_performance.py.

The 3 subjects with lowest EEG performance scores are labeled as dyslexic:
- XBB: Dyslexic (1)
- YAK: Dyslexic (1) 
- YFR: Dyslexic (1)
All other subjects: Normal (0)
"""

import os
import pandas as pd
from pathlib import Path

# Define dyslexic subjects based on EEG performance analysis
DYSLEXIC_SUBJECTS = {'XBB', 'YAK', 'YFR'}

def get_subject_from_filename(filename):
    """Extract subject ID from filename like 'XBB_electrode_features_all_detailed.csv'"""
    return filename.split('_')[0]

def add_dyslexia_labels_to_csv(csv_file_path):
    """
    Add dyslexia label column to a CSV file
    
    Args:
        csv_file_path: Path to the CSV file
    """
    print(f"Processing: {csv_file_path}")
    
    # Extract subject ID from filename
    filename = os.path.basename(csv_file_path)
    subject_id = get_subject_from_filename(filename)
    
    # Determine dyslexia label
    dyslexia_label = 1 if subject_id in DYSLEXIC_SUBJECTS else 0
    label_name = "Dyslexic" if dyslexia_label == 1 else "Normal"
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
        
        # Check if dyslexia label column already exists
        if 'Dyslexia_Label' in df.columns:
            print(f"  - Dyslexia_Label column already exists, updating...")
        else:
            print(f"  - Adding new Dyslexia_Label column...")
        
        # Add dyslexia label columns
        df['Dyslexia_Label'] = dyslexia_label
        df['Dyslexia_Class'] = label_name
        
        # Save the updated CSV
        df.to_csv(csv_file_path, index=False)
        
        print(f"  - Subject {subject_id}: {label_name} ({dyslexia_label})")
        print(f"  - Updated {len(df)} rows")
        
    except Exception as e:
        print(f"  - Error processing {csv_file_path}: {e}")

def main():
    """Main function to process all CSV files in feat_csv directory"""
    
    print("="*70)
    print("ADDING DYSLEXIA LABELS TO CSV FILES")
    print("="*70)
    
    print(f"Dyslexic subjects: {sorted(DYSLEXIC_SUBJECTS)}")
    
    # Get the feat_csv directory path
    feat_csv_dir = Path("../feat_csv")
    
    if not feat_csv_dir.exists():
        print(f"Error: Directory {feat_csv_dir} does not exist!")
        return
    
    # Find all CSV files
    csv_files = list(feat_csv_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {feat_csv_dir}")
        return
    
    print(f"\nFound {len(csv_files)} CSV files to process:")
    
    # Process each CSV file
    subjects_processed = set()
    dyslexic_files = 0
    normal_files = 0
    
    for csv_file in sorted(csv_files):
        subject_id = get_subject_from_filename(csv_file.name)
        subjects_processed.add(subject_id)
        
        add_dyslexia_labels_to_csv(csv_file)
        
        if subject_id in DYSLEXIC_SUBJECTS:
            dyslexic_files += 1
        else:
            normal_files += 1
        
        print()  # Add blank line between files
    
    # Summary
    print("="*70)
    print("PROCESSING SUMMARY")
    print("="*70)
    print(f"Total files processed: {len(csv_files)}")
    print(f"Unique subjects: {len(subjects_processed)}")
    print(f"Dyslexic subject files: {dyslexic_files}")
    print(f"Normal subject files: {normal_files}")
    
    print(f"\nSubjects processed: {sorted(subjects_processed)}")
    print(f"Dyslexic subjects in data: {sorted(subjects_processed & DYSLEXIC_SUBJECTS)}")
    print(f"Normal subjects in data: {sorted(subjects_processed - DYSLEXIC_SUBJECTS)}")
    
    # Check if all dyslexic subjects were found
    missing_dyslexic = DYSLEXIC_SUBJECTS - subjects_processed
    if missing_dyslexic:
        print(f"\nWarning: Dyslexic subjects not found in data: {sorted(missing_dyslexic)}")
    
    print("\n✅ All CSV files have been updated with dyslexia labels!")

if __name__ == "__main__":
    main()
