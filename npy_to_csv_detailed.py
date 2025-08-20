import os
import numpy as np
import csv
import argparse

def flatten_with_mean(data):
    """Recursively flatten multi-dimensional arrays by taking mean of deepest layers first."""
    if isinstance(data, (list, np.ndarray)):
        if len(data) == 0:
            return 0
        
        # Check if all elements are numbers
        try:
            # Try to convert to numpy array to check if all elements are numeric
            arr = np.array(data)
            if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                return np.mean(arr)
            elif arr.ndim > 1:
                # Multi-dimensional array - take mean of innermost dimension first
                return np.mean(arr)
            else:
                # Mixed types or nested structures - process recursively
                flattened = [flatten_with_mean(item) for item in data]
                numeric_values = [x for x in flattened if isinstance(x, (int, float)) and not np.isnan(x)]
                return np.mean(numeric_values) if numeric_values else 0
        except (ValueError, TypeError):
            # Handle mixed types - process recursively
            flattened = [flatten_with_mean(item) for item in data]
            numeric_values = [x for x in flattened if isinstance(x, (int, float)) and not np.isnan(x)]
            return np.mean(numeric_values) if numeric_values else 0
    elif isinstance(data, (int, float)) and not np.isnan(data):
        return data
    else:
        return 0

def process_features_to_columns(data_dict):
    """Process the data to create columns for each feature position."""
    all_features = []
    max_features = 0
    
    # First pass: collect all feature arrays and find max length
    for subject_id, features in data_dict.items():
        if isinstance(features, (list, np.ndarray)):
            # Remove empty strings and process numeric data
            numeric_features = [x for x in features if x != '' and x is not None]
            
            if numeric_features:
                # Process each feature through mean calculation if it's multi-dimensional
                processed_features = []
                for feature in numeric_features:
                    processed_features.append(flatten_with_mean(feature))
                
                all_features.append((subject_id, processed_features))
                max_features = max(max_features, len(processed_features))
    
    return all_features, max_features

def process_npy_to_csv_detailed(npy_file, output_csv):
    """Convert a .npy file to .csv with each feature as a separate column."""
    try:
        data = np.load(npy_file, allow_pickle=True)
        
        # Handle 0-dimensional arrays containing dictionaries
        if isinstance(data, np.ndarray) and data.ndim == 0:
            data = data.item()
        
        max_features = 1  # Default value
        
        if isinstance(data, dict):
            # Process features to get column format
            all_features, max_features = process_features_to_columns(data)
            
            if not all_features:
                print(f"No valid features found in {npy_file}")
                return False
            
            # Create CSV with individual feature columns
            with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                
                # Write header
                header = ['Subject_ID'] + [f'Feature_{i+1}' for i in range(max_features)]
                writer.writerow(header)
                
                # Write data rows
                for subject_id, features in all_features:
                    row = [subject_id] + features + [0] * (max_features - len(features))  # Pad with zeros
                    writer.writerow(row)
                        
        else:
            # Handle other data formats (fallback)
            mean_value = flatten_with_mean(data)
            with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Subject_ID', 'Feature_1'])
                writer.writerow(['Unknown', mean_value])
                
        print(f"Successfully converted {npy_file} to {output_csv} with {max_features} feature columns")
        
    except Exception as e:
        print(f"Error processing {npy_file}: {str(e)}")
        return False
    
    return True

def process_folder(folder_path):
    """Process all .npy files in a folder."""
    success_count = 0
    total_count = 0
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            total_count += 1
            npy_file = os.path.join(folder_path, file_name)
            output_csv = os.path.splitext(npy_file)[0] + '_detailed.csv'
            
            if process_npy_to_csv_detailed(npy_file, output_csv):
                success_count += 1
    
    print(f"Processed {success_count}/{total_count} files successfully from {folder_path}")

def process_sample(folder_path):
    """Process the first .npy file in a folder."""
    for file_name in sorted(os.listdir(folder_path)):  # Sort for consistent results
        if file_name.endswith('.npy'):
            npy_file = os.path.join(folder_path, file_name)
            output_csv = os.path.splitext(npy_file)[0] + '_detailed_sample.csv'
            
            if process_npy_to_csv_detailed(npy_file, output_csv):
                print(f"Sample processing complete: {file_name}")
            return
    
    print("No .npy files found in the specified folder.")

def main():
    parser = argparse.ArgumentParser(description='Convert .npy files to .csv files with individual feature columns.')
    parser.add_argument('--folder', type=str, help='Convert all .npy files in a folder to .csv files.')
    parser.add_argument('--file', type=str, help='Convert a single .npy file to a .csv file.')
    parser.add_argument('--sample', type=str, help='Convert the first .npy file in a folder to a .csv file.')

    args = parser.parse_args()

    if args.folder:
        if os.path.exists(args.folder):
            process_folder(args.folder)
        else:
            print(f"Folder not found: {args.folder}")
    elif args.file:
        if os.path.exists(args.file):
            output_csv = os.path.splitext(args.file)[0] + '_detailed.csv'
            process_npy_to_csv_detailed(args.file, output_csv)
        else:
            print(f"File not found: {args.file}")
    elif args.sample:
        if os.path.exists(args.sample):
            process_sample(args.sample)
        else:
            print(f"Folder not found: {args.sample}")
    else:
        print('Please specify --folder, --file, or --sample.')
        print('Examples:')
        print('  python npy_to_csv_detailed.py --file features/XAH_electrode_features_all.npy')
        print('  python npy_to_csv_detailed.py --folder features')
        print('  python npy_to_csv_detailed.py --sample features')

if __name__ == '__main__':
    main()
