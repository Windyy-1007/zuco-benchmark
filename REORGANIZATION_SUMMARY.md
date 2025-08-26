# ZuCo Benchmark Project Reorganization Summary

This document summarizes the reorganization of the `src` directory by functionality.

## New Directory Structure

### 📁 src/
- `config.py` - Main configuration file
- `subject_lnorm.json` - Subject normalization data
- `__init__.py` - Package initialization

### 📁 src/data_loaders/
**Purpose**: Data loading and preprocessing functionality
- `data_helpers.py` - Main data loading utilities
- `data_loading_helpers.py` - Low-level data loading functions
- `__init__.py`

### 📁 src/models/
**Purpose**: Machine learning models and classifiers
- `classifier.py` - Main classifier implementation
- `classifier_cnn.py` - Convolutional Neural Network classifier
- `classifier_mlp.py` - Multi-Layer Perceptron classifier
- `classifier_random_forest.py` - Random Forest classifier
- `classifier_rf_clustering.py` - Random Forest with clustering
- `basic_models/` - Directory containing basic model implementations
  - `nn_class.py` - Neural network classes
  - `other_basic_models.py` - Other basic model implementations
  - `svm_class.py` - Support Vector Machine classes
  - `voting_class.py` - Voting classifier implementations
- `__init__.py`

### 📁 src/evaluation/
**Purpose**: Benchmarking and evaluation scripts
- `benchmark.py` - Main benchmark script
- `benchmark_baseline.py` - Baseline benchmark
- `dyslexia_benchmark.py` - Dyslexia prediction benchmark
- `dyslexia_benchmark_simple.py` - Simplified dyslexia benchmark
- `ml_dyslexia_prediction.py` - ML-based dyslexia prediction
- `multi_algorithm_benchmark.py` - Multi-algorithm comparison
- `seec_dyslexia_prediction.py` - SEEC dyslexia prediction
- `analyze_eeg_performance.py` - EEG performance analysis
- `validation.py` - Validation utilities
- `__init__.py`

### 📁 src/labeling/
**Purpose**: Dyslexia labeling functionality
- `dyslexia_labels.py` - Dyslexia labeling utilities
- `__init__.py`

### 📁 src/feature_extraction/
**Purpose**: Feature extraction and processing
- `extract_features.py` - Main feature extraction logic
- `feature_cleaner.py` - Feature cleaning utilities
- `__init__.py`

### 📁 src/npytocsv/
**Purpose**: NPY to CSV conversion utilities
- `npy_to_csv_detailed.py` - Detailed NPY to CSV conversion
- `npy_to_csv.py` - Basic NPY to CSV conversion
- `npyToText.py` - NPY to text conversion
- `npycsv_config.json` - Configuration for NPY/CSV conversion
- `__init__.py`

### 📁 src/visualization/
**Purpose**: Plotting and analysis visualization
- `topoplots_nr_tsr_avg.py` - Topographical plots
- `result-analysis/` - Result analysis and plotting scripts
  - `correlation_analysis.py` - Correlation analysis plots
  - `plot_block_classification.py` - Block classification plots
  - `plot_results.py` - General result plotting
  - `plot_session_classification.py` - Session classification plots
  - `plot_subject_classification.py` - Subject classification plots
  - `plot_task_class_in_blocks.py` - Task classification in blocks
  - `plot_zuco1_session_effect.py` - ZuCo1 session effect plots
- `__init__.py`

### 📁 src/submissions/
**Purpose**: Submission files and outputs (kept as-is)

## Import Updates

All files have been updated with proper import statements to work with the new directory structure:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from data_loaders import data_helpers as dh
from models import classifier as clf
from labeling import dyslexia_labels as dl
from feature_extraction import extract_features as fe
```

## Files Removed

The following unnecessary files were removed:
- `src/nohup.out` - Terminal output file
- `src/.DS_Store` - macOS metadata file

## Benefits of This Reorganization

1. **Clear Separation of Concerns**: Each directory has a specific purpose
2. **Improved Maintainability**: Related functionality is grouped together
3. **Better Code Discovery**: Easier to find specific functionality
4. **Scalability**: Easy to add new files to appropriate categories
5. **Modularity**: Each module can be developed and tested independently

## Usage Notes

- All Python files now include proper path setup for imports
- Each directory has an `__init__.py` file making it a proper Python package
- The original functionality is preserved, just better organized
- Config file remains at the root level for global access

EEG = brain
SEEC