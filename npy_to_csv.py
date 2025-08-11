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

def process_npy_to_csv(npy_file, output_csv):
    """Convert a .npy file to .csv, handling the dictionary structure with subject data."""
    try:
        data = np.load(npy_file, allow_pickle=True)
        
        # Handle 0-dimensional arrays containing dictionaries
        if isinstance(data, np.ndarray) and data.ndim == 0:
            data = data.item()
        
        if isinstance(data, dict):
            # Dictionary format: keys are subject IDs, values are lists of features
            with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                
                # Write header
                writer.writerow(['Subject_ID', 'Features', 'Mean_Value'])
                
                for subject_id, features in data.items():
                    if isinstance(features, (list, np.ndarray)):
                        # Remove empty strings and process numeric data
                        numeric_features = [x for x in features if x != '' and x is not None]
                        
                        if numeric_features:
                            # Calculate mean of all numeric features
                            mean_value = flatten_with_mean(numeric_features)
                            
                            # Write row with subject ID, original feature count, and mean value
                            writer.writerow([subject_id, len(numeric_features), mean_value])
                        else:
                            writer.writerow([subject_id, 0, 0])
                    else:
                        # Single value
                        mean_value = flatten_with_mean(features)
                        writer.writerow([subject_id, 1, mean_value])
                        
        else:
            # Handle other data formats (fallback)
            mean_value = flatten_with_mean(data)
            with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Mean_Value'])
                writer.writerow([mean_value])
                
        print(f"Successfully converted {npy_file} to {output_csv}")
        
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
            output_csv = os.path.splitext(npy_file)[0] + '.csv'
            
            if process_npy_to_csv(npy_file, output_csv):
                success_count += 1
    
    print(f"Processed {success_count}/{total_count} files successfully from {folder_path}")

def process_sample(folder_path):
    """Process the first .npy file in a folder."""
    for file_name in sorted(os.listdir(folder_path)):  # Sort for consistent results
        if file_name.endswith('.npy'):
            npy_file = os.path.join(folder_path, file_name)
            output_csv = os.path.splitext(npy_file)[0] + '_sample.csv'
            
            if process_npy_to_csv(npy_file, output_csv):
                print(f"Sample processing complete: {file_name}")
            return
    
    print("No .npy files found in the specified folder.")

def main():
    parser = argparse.ArgumentParser(description='Convert .npy files to .csv files.')
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
            output_csv = os.path.splitext(args.file)[0] + '.csv'
            process_npy_to_csv(args.file, output_csv)
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
        print('  python npy_to_csv.py --file features/XAH_electrode_features_all.npy')
        print('  python npy_to_csv.py --folder features')
        print('  python npy_to_csv.py --sample features')

if __name__ == '__main__':
    main()