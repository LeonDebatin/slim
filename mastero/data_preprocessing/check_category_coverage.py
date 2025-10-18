"""
Script to check if all categorical values appear in training sets across all Monte Carlo splits.
This validates whether preprocessing on the full dataset vs train-only would make a practical difference.
"""

import pandas as pd
import ast
import sys
import os
sys.path.insert(0, os.path.abspath(".."))

def load_data_info():
    """Load the data info with train/test indices."""
    data_info = pd.read_csv('../data/data_info.csv')
    data_info['test_indices'] = data_info['test_indices'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data_info['train_indices'] = data_info['train_indices'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return data_info

def check_category_coverage(dataset_name, categorical_columns):
    """
    Check if all categorical values appear in training sets across all Monte Carlo runs.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'autism', 'thyroid')
        categorical_columns: List of column names that contain categorical data
    
    Returns:
        Dictionary with coverage analysis results
    """
    
    # Load original dataset (before preprocessing)
    if dataset_name == 'autism':
        # Load autism original data
        import arff
        with open("../data/data_original/autism.arff", "r") as file:
            dataset = arff.load(file)
        df = pd.DataFrame(dataset['data'], columns=[x[0] for x in dataset['attributes']])
        df['Class/ASD'].replace({'NO': 0, 'YES': 1}, inplace=True)
        df.rename(columns={'Class/ASD': 'target'}, inplace=True)
        df.drop(columns=['ethnicity', 'relation', 'age', 'contry_of_res'], inplace=True)
        
    elif dataset_name == 'thyroid':
        # Load thyroid original data
        import arff
        with open("../data/data_original/thyroid.arff", "r") as file:
            dataset = arff.load(file)
        df = pd.DataFrame(dataset['data'], columns=[x[0] for x in dataset['attributes']])
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Load train/test split information
    data_info = load_data_info()
    dataset_info = data_info[data_info['name'] == dataset_name].iloc[0]
    
    # Get all unique values in each categorical column (full dataset)
    full_dataset_categories = {}
    for col in categorical_columns:
        if col in df.columns:
            full_dataset_categories[col] = set(df[col].unique())
    
    print(f"\n=== {dataset_name.upper()} DATASET ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Categorical columns analyzed: {categorical_columns}")
    
    # Analyze each categorical column
    results = {}
    
    for col in categorical_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataset")
            continue
            
        col_results = {
            'full_dataset_categories': full_dataset_categories[col],
            'missing_in_train_runs': [],
            'coverage_stats': {'total_runs': 30, 'runs_with_all_categories': 0}
        }
        
        print(f"\n--- Column: {col} ---")
        print(f"Full dataset categories: {sorted(full_dataset_categories[col])}")
        
        # Check each Monte Carlo run
        runs_with_missing = 0
        for run_idx in range(30):
            train_indices = dataset_info['train_indices'][run_idx]
            train_data = df.iloc[train_indices]
            
            train_categories = set(train_data[col].unique())
            missing_categories = full_dataset_categories[col] - train_categories
            
            if missing_categories:
                runs_with_missing += 1
                col_results['missing_in_train_runs'].append({
                    'run': run_idx + 1,
                    'missing_categories': list(missing_categories),
                    'train_categories': list(train_categories)
                })
            else:
                col_results['coverage_stats']['runs_with_all_categories'] += 1
        
        # Summary statistics
        coverage_percentage = (col_results['coverage_stats']['runs_with_all_categories'] / 30) * 100
        
        print(f"Runs with all categories: {col_results['coverage_stats']['runs_with_all_categories']}/30 ({coverage_percentage:.1f}%)")
        
        if col_results['missing_in_train_runs']:
            print(f"Runs with missing categories: {len(col_results['missing_in_train_runs'])}")
            print("Missing category details:")
            for run_info in col_results['missing_in_train_runs'][:5]:  # Show first 5 examples
                print(f"  Run {run_info['run']}: Missing {run_info['missing_categories']}")
            if len(col_results['missing_in_train_runs']) > 5:
                print(f"  ... and {len(col_results['missing_in_train_runs']) - 5} more runs")
        else:
            print("✅ All categories appear in ALL training sets!")
            
        results[col] = col_results
    
    return results

def main():
    """Main analysis function."""
    
    print("CATEGORICAL COVERAGE ANALYSIS")
    print("=" * 50)
    print("Checking if all categorical values appear in training sets across Monte Carlo runs")
    print("This determines if full-dataset vs train-only preprocessing makes a practical difference")
    
    # Define categorical columns for each dataset
    datasets_to_check = {
        'autism': ['gender', 'jundice', 'austim', 'used_app_before', 'age_desc'],
        'thyroid': None  # Will determine dynamically for thyroid
    }
    
    all_results = {}
    
    # Check Autism dataset
    autism_results = check_category_coverage('autism', datasets_to_check['autism'])
    all_results['autism'] = autism_results
    
    # Check Thyroid dataset (need to determine categorical columns)
    print("\n=== THYROID DATASET ANALYSIS ===")
    print("Loading thyroid data to identify categorical columns...")
    
    import arff
    with open("../data/data_original/thyroid.arff", "r") as file:
        dataset = arff.load(file)
    thyroid_df = pd.DataFrame(dataset['data'], columns=[x[0] for x in dataset['attributes']])
    
    # Identify non-numeric columns (excluding target)
    thyroid_categorical = []
    for col in thyroid_df.columns:
        if col not in ['Age', 'Recurred'] and thyroid_df[col].dtype == 'object':
            thyroid_categorical.append(col)
    
    print(f"Identified categorical columns: {thyroid_categorical}")
    
    if thyroid_categorical:
        thyroid_results = check_category_coverage('thyroid', thyroid_categorical)
        all_results['thyroid'] = thyroid_results
    
    # Overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    total_columns_checked = 0
    columns_with_perfect_coverage = 0
    
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        for col, col_results in results.items():
            total_columns_checked += 1
            coverage_pct = (col_results['coverage_stats']['runs_with_all_categories'] / 30) * 100
            
            if coverage_pct == 100:
                columns_with_perfect_coverage += 1
                status = "✅ PERFECT"
            elif coverage_pct >= 90:
                status = "⚠️  MOSTLY COVERED"
            else:
                status = "❌ FREQUENT MISSING"
                
            print(f"  {col}: {coverage_pct:.1f}% coverage {status}")
    
    print(f"\nCONCLUSION:")
    print(f"Columns with perfect coverage: {columns_with_perfect_coverage}/{total_columns_checked}")
    
    if columns_with_perfect_coverage == total_columns_checked:
        print("✅ ALL categorical values always appear in training sets!")
        print("→ No practical difference between full-dataset vs train-only preprocessing")
        print("→ Data leakage concern is theoretical only - no actual information advantage")
    else:
        print("⚠️  Some categorical values missing from some training sets")
        print("→ Full-dataset preprocessing does provide information advantage")
        print("→ Data leakage concern is valid and should be addressed")

if __name__ == "__main__":
    main()