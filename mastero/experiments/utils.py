import sys
import os
sys.path.insert(0, os.path.abspath("../../.."))

import ast
from slim_gsgp.datasets.data_loader import load_pandas_df
import pandas as pd
from slim_gsgp.main_gp import gp
from slim_gsgp.main_slim import slim
from slim_gsgp.main_gsgp import gsgp
from slim_gsgp.evaluators.fitness_functions import *
import slim_gsgp.evaluators.fitness_functions
from imblearn.over_sampling import SMOTENC, SMOTE, SMOTEN
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings



def load_and_adapt_data_info(filepath):
    #load
    data_info = pd.read_csv(filepath)
    #make dtype list for
    data_info['test_indices'] = data_info['test_indices'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data_info['train_indices'] = data_info['train_indices'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data_info['categoricals'] = data_info['categoricals'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Handle binaries column if it exists (for data2 compatibility)
    if 'binaries' in data_info.columns:
        data_info['binaries'] = data_info['binaries'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    else:
        # For backward compatibility with original data, set empty binaries
        data_info['binaries'] = [[] for _ in range(len(data_info))]
    
    return data_info


def oversample(df, categoricals = []):
    
    #if list is empty
    if not categoricals:
        sm = SMOTE(random_state = 42)
        
    elif len(categoricals) >= 20: #spect dataset
        sm = SMOTEN(random_state = 42)
    
    else:
        sm = SMOTENC(random_state = 42, categorical_features = categoricals)
    
    X,y = sm.fit_resample(df.drop(columns = ['target']), df['target'])
    
    return pd.concat([X,y], axis = 1)

def encode_categoricals(train, test, categoricals):
    """
    Apply one-hot encoding to categorical columns based ONLY on training data.
    This prevents data leakage by not using test set categories during encoding.
    Maintains the exact same column order as the original notebook preprocessing.
    """
    import pandas as pd
    
    if not categoricals:
        return train, test
    
    # Process all categoricals at once like the original notebook
    # Use get_dummies on training data to establish the structure
    train_encoded = pd.get_dummies(train, columns=categoricals, dtype=int)
    
    # For test data, we need to ensure it has the same dummy columns as train
    test_encoded = pd.get_dummies(test, columns=categoricals, dtype=int)
    
    # Get all dummy column names from training data
    non_categorical_cols = [col for col in train.columns if col not in categoricals]
    train_dummy_cols = [col for col in train_encoded.columns if col not in non_categorical_cols]
    
    # Ensure test set has same dummy columns as train set
    for dummy_col in train_dummy_cols:
        if dummy_col not in test_encoded.columns:
            test_encoded[dummy_col] = 0
    
    # Remove any dummy columns from test that don't exist in train (categories not seen in training)
    test_dummy_cols = [col for col in test_encoded.columns if col not in non_categorical_cols]
    extra_test_cols = [col for col in test_dummy_cols if col not in train_dummy_cols]
    if extra_test_cols:
        test_encoded = test_encoded.drop(columns=extra_test_cols)
    
    # Reorder columns to match original preprocessing: non-categoricals (except target), then dummies, then target
    other_cols = [col for col in non_categorical_cols if col != 'target']
    final_order = other_cols + train_dummy_cols + ['target']
    
    train_encoded = train_encoded[final_order]
    test_encoded = test_encoded[final_order]
    
    return train_encoded, test_encoded


def scale(train, test, categoricals):
    warnings.filterwarnings("ignore")
    scaler = StandardScaler()
    features = list(set(train.columns) - set(categoricals) - {'target'})
    train.loc[:, features] = train[features].astype(float)
    test.loc[:, features] = test[features].astype(float)
    
    train.loc[:, features] = scaler.fit_transform(train[features])
    test.loc[:, features] = scaler.transform(test[features])
    
    return train, test




def return_train_test(df, train_indices, test_indices, scaling, oversampling = False, categoricals = [], binaries = []):
    
    train = df.iloc[train_indices]
    test = df.iloc[test_indices]
    
    # Apply one-hot encoding to categoricals based ONLY on training data
    if categoricals:
        train, test = encode_categoricals(train, test, categoricals)
    
    # Get all encoded categorical columns for scaling exclusion
    all_categorical_cols = []
    if categoricals:
        # After encoding, we need to identify the new dummy columns
        for col in categoricals:
            if col in train.columns:
                # Find all dummy columns created from this categorical
                dummy_cols = [c for c in train.columns if c.startswith(f"{col}_")]
                all_categorical_cols.extend(dummy_cols)
        # Add binary columns to exclusion list (they don't need scaling)
        all_categorical_cols.extend(binaries)
    else:
        all_categorical_cols = binaries
    
    if scaling:
        train, test = scale(train, test, all_categorical_cols)
    
    if oversampling:
        train = oversample(train, all_categorical_cols)
    
    X_train, y_train = load_pandas_df(train, X_y=True)
    X_test, y_test = load_pandas_df(test, X_y=True)
        
    return X_train, y_train, X_test, y_test
    
    
def train_model(dataset_name, X_train, y_train, X_test, y_test, model, **model_config):
    
    if model== 'gp':
        best_individual = gp(
                    dataset_name=dataset_name, 
                    X_train=X_train, 
                    y_train = y_train, 
                    X_test = X_test, 
                    y_test = y_test, 
                    **model_config
                    )
    
    if model == 'slim':
        best_individual = slim(  
                    dataset_name=dataset_name, 
                    X_train=X_train, 
                    y_train = y_train, 
                    X_test = X_test, 
                    y_test = y_test, 
                    **model_config
                    )
    #gsgp changed to slim because bug in tree node counter, therefore slim+sig2 is used
    if model == 'gsgp':
        best_individual = slim(
                    dataset_name=dataset_name, 
                    X_train=X_train, 
                    y_train = y_train, 
                    X_test = X_test, 
                    y_test = y_test, 
                    **model_config
                    )
    
    return best_individual


def compute_class_weights(y):
    """
    y: list, numpy array, or 1D torch tensor of class labels
    returns: torch tensor of class weights
    """
    y = torch.tensor(y) if not torch.is_tensor(y) else y
    classes, counts = torch.unique(y, return_counts=True)
    total_samples = y.size(0)

    weights = total_samples / (len(classes) * counts.float())
    return weights

def update_sample_weights(y_train, y_test):
    # Convert tensors to numpy for class weight computation if needed
    if torch.is_tensor(y_train):
        y_train_np = y_train.numpy()
    else:
        y_train_np = y_train
    
    if torch.is_tensor(y_test):
        y_test_np = y_test.numpy()  
    else:
        y_test_np = y_test
        
    class_weights = compute_class_weights(y_train_np)
    
    # Ensure we have the right number of class weights for indexing
    unique_labels = np.unique(np.concatenate([y_train_np, y_test_np]))
    
    # Create a mapping for safe indexing
    label_to_weight = {}
    for i, label in enumerate(sorted(unique_labels)):
        if i < len(class_weights):
            label_to_weight[int(label)] = class_weights[i]
        else:
            label_to_weight[int(label)] = 1.0  # Default weight
    
    train_sample_weights = torch.tensor([label_to_weight[int(label)] for label in y_train_np], dtype=torch.float32)
    test_sample_weights = torch.tensor([label_to_weight[int(label)] for label in y_test_np], dtype=torch.float32)
    
    slim_gsgp.evaluators.fitness_functions.train_sample_weights = train_sample_weights
    slim_gsgp.evaluators.fitness_functions.test_sample_weights = test_sample_weights
    return

def evaluate_prediction(y_true, y_pred):
    rms = sigmoid_rmse(y_true, y_pred)
    wrms = weighted_sigmoid_rmse(y_true, y_pred)
    acc = accuracy(y_true, y_pred)
    roc = roc_auc(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)    
    
    return rms, wrms, acc, roc, f1, prec, rec


def get_evaluation_dictionary(y_true, y_pred):
    
    rms, wrms, acc, roc, f1, prec, rec = evaluate_prediction(y_true, y_pred)
    
    evaluation_dict = {
        'rmse': rms.item(),
        'wrmse': wrms.item(),
        'accuracy': acc.item(),
        'roc_auc': roc.item(),
        'f1_score': f1.item(),
        'precision': prec.item(),
        'recall': rec.item()
    }
    
    return evaluation_dict




def fill_config(model_config, scaling, oversampling, fitness_function, minimization, inflation_rate, ms_upper):
    model_config['scaling'] = scaling
    model_config["oversampling"] = oversampling
    model_config['config']["fitness_function"] = fitness_function
    model_config['config']["minimization"] = minimization
    
    if model_config['name'] == 'slim':
        model_config['config']['p_inflate'] = inflation_rate
        model_config['config']['ms_upper'] = ms_upper
    
    if model_config['name'] == 'gsgp':
        model_config['config']['ms_upper'] = ms_upper
    
    return model_config


