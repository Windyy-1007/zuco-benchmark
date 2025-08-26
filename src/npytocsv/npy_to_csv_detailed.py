import os
import numpy as np
import csv
import argparse
import json
from scipy import stats

def calculate_statistics(data, stats_config):
    """Calculate comprehensive statistical measures for a data array."""
    if isinstance(data, (list, np.ndarray)):
        if len(data) == 0:
            return {stat: 0 for stat in stats_config}
        
        # Convert to numpy array and flatten
        arr = np.array(data).flatten()
        # Remove NaN and infinite values
        arr = arr[np.isfinite(arr)]
        
        if len(arr) == 0:
            return {stat: 0 for stat in stats_config}
        
        results = {}
        
        for stat in stats_config:
            try:
                if stat == 'mean':
                    results[stat] = np.mean(arr)
                elif stat == 'median':
                    results[stat] = np.median(arr)
                elif stat == 'std':
                    results[stat] = np.std(arr, ddof=1) if len(arr) > 1 else 0
                elif stat == 'var':
                    results[stat] = np.var(arr, ddof=1) if len(arr) > 1 else 0
                elif stat == 'min':
                    results[stat] = np.min(arr)
                elif stat == 'max':
                    results[stat] = np.max(arr)
                elif stat == 'range':
                    results[stat] = np.max(arr) - np.min(arr)
                elif stat == 'q25':
                    results[stat] = np.percentile(arr, 25)
                elif stat == 'q75':
                    results[stat] = np.percentile(arr, 75)
                elif stat == 'iqr':
                    results[stat] = np.percentile(arr, 75) - np.percentile(arr, 25)
                elif stat == 'skewness':
                    results[stat] = stats.skew(arr) if len(arr) > 2 else 0
                elif stat == 'kurtosis':
                    results[stat] = stats.kurtosis(arr) if len(arr) > 3 else 0
                elif stat == 'cv':
                    mean_val = np.mean(arr)
                    results[stat] = np.std(arr, ddof=1) / mean_val if mean_val != 0 and len(arr) > 1 else 0
                elif stat == 'mad':
                    results[stat] = np.median(np.abs(arr - np.median(arr)))
                elif stat == 'energy':
                    results[stat] = np.sum(arr ** 2)
                elif stat == 'rms':
                    results[stat] = np.sqrt(np.mean(arr ** 2))
                elif stat == 'entropy':
                    # Simple entropy calculation based on histogram
                    hist, _ = np.histogram(arr, bins=min(10, len(arr)))
                    hist = hist[hist > 0]  # Remove zero bins
                    if len(hist) > 1:
                        prob = hist / np.sum(hist)
                        results[stat] = -np.sum(prob * np.log2(prob))
                    else:
                        results[stat] = 0
                elif stat == 'sum':
                    results[stat] = np.sum(arr)
                elif stat == 'count':
                    results[stat] = len(arr)
                else:
                    results[stat] = 0
            except:
                results[stat] = 0
        
        return results
    
    else:
        # Single value
        single_value = data if isinstance(data, (int, float)) and not np.isnan(data) else 0
        results = {}
        for stat in stats_config:
            if stat in ['mean', 'median', 'min', 'max', 'sum']:
                results[stat] = single_value
            elif stat == 'count':
                results[stat] = 1 if single_value != 0 else 0
            else:
                results[stat] = 0
        return results

def get_feature_labels(feature_set_name, num_features):
    """Get proper feature labels based on the feature set type."""
    
    # Electrode names starting from E1
    electrode_names = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20',
                      'E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E32', 'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39',
                      'E40', 'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'E47', 'E48', 'E49', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E56', 'E57', 'E58',
                      'E59', 'E60', 'E61', 'E62', 'E63', 'E64', 'E65', 'E66', 'E67', 'E68', 'E69', 'E70', 'E71', 'E72', 'E73', 'E74', 'E75', 'E76', 'E77',
                      'E78', 'E79', 'E80', 'E81', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E88', 'E89', 'E90', 'E91', 'E92', 'E93', 'E94', 'E95', 'E96',
                      'E97', 'E98', 'E99', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 'E109', 'E110', 'E111', 'E112', 'E113',
                      'E114', 'E115', 'E116', 'E117', 'E118', 'E119', 'E120', 'E121', 'E122', 'E123', 'E124', 'E125', 'E126', 'E127', 'E128']
    
    if 'electrode_features_all' in feature_set_name:
        # Handle electrode features - always start from E1
        electrodes_per_band = num_features // 4
        labels = []
        bands = ['Theta', 'Alpha', 'Beta', 'Gamma']
        
        for band in bands:
            for i in range(electrodes_per_band):
                labels.append(f'{band}_{electrode_names[i]}')
        return labels
    
    elif 'sent_gaze_sacc_eeg_means' in feature_set_name and num_features == 13:
        return ['Omission_Rate', 'Weighted_Fixation_Count', 'Weighted_Reading_Speed',
                'Mean_Saccade_Duration', 'Max_Saccade_Velocity', 'Mean_Saccade_Velocity',
                'Max_Saccade_Duration', 'Mean_Saccade_Amplitude', 'Max_Saccade_Amplitude',
                'Theta_Mean', 'Alpha_Mean', 'Beta_Mean', 'Gamma_Mean']
    
    elif 'sent_gaze_sacc' in feature_set_name and num_features == 9:
        return ['Omission_Rate', 'Weighted_Fixation_Count', 'Weighted_Reading_Speed',
                'Mean_Saccade_Duration', 'Max_Saccade_Velocity', 'Mean_Saccade_Velocity',
                'Max_Saccade_Duration', 'Mean_Saccade_Amplitude', 'Max_Saccade_Amplitude']
    
    elif 'sent_gaze_eeg_means' in feature_set_name and num_features == 11:
        return ['Omission_Rate', 'Weighted_Fixation_Count', 'Weighted_Reading_Speed',
                'Mean_Saccade_Duration', 'Max_Saccade_Velocity', 'Mean_Saccade_Velocity',
                'Max_Saccade_Duration', 'Theta_Mean', 'Alpha_Mean', 'Beta_Mean', 'Gamma_Mean']
    
    elif 'sent_gaze' in feature_set_name and num_features == 4:
        return ['Omission_Rate', 'Weighted_Fixation_Count', 'Weighted_Reading_Speed', 'Mean_Saccade_Duration']
    
    elif 'sent_saccade' in feature_set_name and num_features == 6:
        return ['Mean_Saccade_Duration', 'Max_Saccade_Velocity', 'Mean_Saccade_Velocity',
                'Max_Saccade_Duration', 'Mean_Saccade_Amplitude', 'Max_Saccade_Amplitude']
    
    elif 'electrode_features_theta' in feature_set_name:
        return [f'Theta_{electrode_names[i]}' for i in range(num_features)]
    
    elif 'electrode_features_alpha' in feature_set_name:
        return [f'Alpha_{electrode_names[i]}' for i in range(num_features)]
    
    elif 'electrode_features_beta' in feature_set_name:
        return [f'Beta_{electrode_names[i]}' for i in range(num_features)]
    
    elif 'electrode_features_gamma' in feature_set_name:
        return [f'Gamma_{electrode_names[i]}' for i in range(num_features)]
    
    elif 'eeg_means' in feature_set_name and num_features == 4:
        return ['Theta_Mean', 'Alpha_Mean', 'Beta_Mean', 'Gamma_Mean']
    
    elif num_features == 1:
        # Single feature sets
        if 'omission_rate' in feature_set_name:
            return ['Omission_Rate']
        elif 'fixation_number' in feature_set_name:
            return ['Weighted_Fixation_Count']
        elif 'reading_speed' in feature_set_name:
            return ['Weighted_Reading_Speed']
        elif 'mean_sacc_dur' in feature_set_name:
            return ['Mean_Saccade_Duration']
        elif 'max_sacc_velocity' in feature_set_name:
            return ['Max_Saccade_Velocity']
        elif 'mean_sacc_velocity' in feature_set_name:
            return ['Mean_Saccade_Velocity']
        elif 'max_sacc_dur' in feature_set_name:
            return ['Max_Saccade_Duration']
        elif 'max_sacc_amp' in feature_set_name:
            return ['Max_Saccade_Amplitude']
        elif 'mean_sacc_amp' in feature_set_name:
            return ['Mean_Saccade_Amplitude']
        elif 'theta_mean' in feature_set_name:
            return ['Theta_Mean']
        elif 'alpha_mean' in feature_set_name:
            return ['Alpha_Mean']
        elif 'beta_mean' in feature_set_name:
            return ['Beta_Mean']
        elif 'gamma_mean' in feature_set_name:
            return ['Gamma_Mean']
        elif 'flesch_baseline' in feature_set_name:
            return ['Flesch_Reading_Ease']
    
    # Fallback to generic labels if no match found
    return [f'Feature_{i+1}' for i in range(num_features)]

def load_config(config_file):
    """Load configuration from JSON file."""
    default_config = {
        "statistics": [
            "mean", "median", "std", "var", "min", "max", "range",
            "q25", "q75", "iqr", "skewness", "kurtosis", "cv", 
            "mad", "energy", "rms", "entropy", "sum", "count"
        ],
        "output_format": "detailed"
    }
    
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            # Merge with defaults
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
            return config
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration")
    
    return default_config

def create_default_config(config_file):
    """Create a default configuration file."""
    default_config = {
        "statistics": [
            "mean", "median", "std", "var", "min", "max", "range",
            "q25", "q75", "iqr", "skewness", "kurtosis", "cv", 
            "mad", "energy", "rms", "entropy", "sum", "count"
        ],
        "output_format": "detailed",
        "description": {
            "mean": "Arithmetic mean/average",
            "median": "Middle value (50th percentile)",
            "std": "Standard deviation",
            "var": "Variance", 
            "min": "Minimum value",
            "max": "Maximum value",
            "range": "Max - Min",
            "q25": "25th percentile (first quartile)",
            "q75": "75th percentile (third quartile)",
            "iqr": "Interquartile range (Q3 - Q1)",
            "skewness": "Asymmetry of distribution",
            "kurtosis": "Tail heaviness of distribution",
            "cv": "Coefficient of variation (std/mean)",
            "mad": "Median absolute deviation",
            "energy": "Sum of squares",
            "rms": "Root mean square",
            "entropy": "Information entropy",
            "sum": "Sum of all values",
            "count": "Number of non-zero values"
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"Default configuration created: {config_file}")
    return default_config

def process_features_to_columns(data_dict, stats_config):
    """Process the data to create columns for each feature position with comprehensive statistics."""
    all_features = []
    max_features = 0
    
    # First pass: collect all feature arrays and find max length
    for subject_id, features in data_dict.items():
        if isinstance(features, (list, np.ndarray)):
            # Remove empty strings and labels (last item) - keep only numeric features
            numeric_features = []
            for i, x in enumerate(features):
                # Skip if it's empty string, None, or if it's the last item and it's a string label
                if x == '' or x is None:
                    continue
                # If it's the last item and it's a string (label), skip it
                if i == len(features) - 1 and isinstance(x, str):
                    continue
                numeric_features.append(x)
            
            if numeric_features:
                # Process each feature through statistical calculation
                processed_features = []
                for feature in numeric_features:
                    feature_stats = calculate_statistics(feature, stats_config)
                    # Flatten the statistics dict to a list in consistent order
                    stat_values = [feature_stats[stat] for stat in stats_config]
                    processed_features.extend(stat_values)
                
                all_features.append((subject_id, processed_features))
                max_features = max(max_features, len(processed_features))
    
    return all_features, max_features

def process_npy_to_csv_detailed(npy_file, output_csv, config=None):
    """Convert a .npy file to .csv with comprehensive statistical features."""
    try:
        # Load configuration
        if config is None:
            config = load_config(None)
        
        stats_config = config.get('statistics', ['mean'])
        
        data = np.load(npy_file, allow_pickle=True)
        
        # Handle 0-dimensional arrays containing dictionaries
        if isinstance(data, np.ndarray) and data.ndim == 0:
            data = data.item()
        
        max_features = 1  # Default value
        
        # Extract feature set name from filename
        feature_set_name = os.path.basename(npy_file).replace('.npy', '')
        
        if isinstance(data, dict):
            # Process features to get column format
            all_features, max_features = process_features_to_columns(data, stats_config)
            
            if not all_features:
                print(f"No valid features found in {npy_file}")
                return False
            
            # Calculate number of original features
            num_original_features = max_features // len(stats_config)
            
            # Get proper feature labels based on the feature set type
            feature_labels = get_feature_labels(feature_set_name, num_original_features)
            
            # Create CSV with individual feature columns
            with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                
                # Write header with statistical suffixes
                header = ['Subject_ID']
                for i, feature_label in enumerate(feature_labels):
                    for stat in stats_config:
                        header.append(f'{feature_label}_{stat}')
                writer.writerow(header)
                
                # Write data rows
                for subject_id, features in all_features:
                    row = [subject_id] + features + [0] * (max_features - len(features))  # Pad with zeros
                    writer.writerow(row)
                        
        else:
            # Handle other data formats (fallback)
            num_original_features = 1
            feature_labels = get_feature_labels(feature_set_name, num_original_features)
            feature_stats = calculate_statistics(data, stats_config)
            with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                header = ['Subject_ID'] + [f'{feature_labels[0]}_{stat}' for stat in stats_config]
                writer.writerow(header)
                row = ['Unknown'] + [feature_stats[stat] for stat in stats_config]
                writer.writerow(row)
                
        print(f"Successfully converted {npy_file} to {output_csv}")
        print(f"  - Feature set: {feature_set_name}")
        print(f"  - Original features: {num_original_features}")
        print(f"  - Statistics per feature: {len(stats_config)}")
        print(f"  - Total columns: {max_features} + Subject_ID")
        
    except Exception as e:
        print(f"Error processing {npy_file}: {str(e)}")
        return False
    
    return True

def process_folder(folder_path, config=None):
    """Process all .npy files in a folder."""
    success_count = 0
    total_count = 0
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            total_count += 1
            npy_file = os.path.join(folder_path, file_name)
            output_csv = os.path.splitext(npy_file)[0] + '_detailed.csv'
            
            if process_npy_to_csv_detailed(npy_file, output_csv, config):
                success_count += 1
    
    print(f"Processed {success_count}/{total_count} files successfully from {folder_path}")

def process_sample(folder_path, config=None):
    """Process the first .npy file in a folder."""
    for file_name in sorted(os.listdir(folder_path)):  # Sort for consistent results
        if file_name.endswith('.npy'):
            npy_file = os.path.join(folder_path, file_name)
            output_csv = os.path.splitext(npy_file)[0] + '_detailed_sample.csv'
            
            if process_npy_to_csv_detailed(npy_file, output_csv, config):
                print(f"Sample processing complete: {file_name}")
            return
    
    print("No .npy files found in the specified folder.")

def main():
    parser = argparse.ArgumentParser(description='Convert .npy files to .csv files with comprehensive statistical features.')
    parser.add_argument('--folder', type=str, help='Convert all .npy files in a folder to .csv files.')
    parser.add_argument('--file', type=str, help='Convert a single .npy file to a .csv file.')
    parser.add_argument('--sample', type=str, help='Convert the first .npy file in a folder to a .csv file.')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file.')
    parser.add_argument('--create-config', type=str, help='Create a default configuration file at specified path.')

    args = parser.parse_args()

    # Create default config if requested
    if args.create_config:
        create_default_config(args.create_config)
        return

    # Load configuration
    config = load_config(args.config)
    
    # Display current configuration
    print("Using configuration:")
    print(f"  Statistics: {config['statistics']}")
    print(f"  Total statistics per feature: {len(config['statistics'])}")
    print()

    if args.folder:
        if os.path.exists(args.folder):
            process_folder(args.folder, config)
        else:
            print(f"Folder not found: {args.folder}")
    elif args.file:
        if os.path.exists(args.file):
            output_csv = os.path.splitext(args.file)[0] + '_detailed.csv'
            process_npy_to_csv_detailed(args.file, output_csv, config)
        else:
            print(f"File not found: {args.file}")
    elif args.sample:
        if os.path.exists(args.sample):
            process_sample(args.sample, config)
        else:
            print(f"Folder not found: {args.sample}")
    else:
        print('Please specify --folder, --file, or --sample.')
        print('Examples:')
        print('  python npy_to_csv_detailed.py --file features/XAH_electrode_features_all.npy')
        print('  python npy_to_csv_detailed.py --folder features')
        print('  python npy_to_csv_detailed.py --sample features')
        print('  python npy_to_csv_detailed.py --folder features --config npycsv_config.json')
        print('  python npy_to_csv_detailed.py --create-config npycsv_config.json')

if __name__ == '__main__':
    main()
