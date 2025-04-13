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
import numpy as np



def load_and_adapt_data_info(filepath):
    #load
    data_info = pd.read_csv(filepath)
    #make dtype list for
    data_info['test_indices'] = data_info['test_indices'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data_info['train_indices'] = data_info['train_indices'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data_info['categoricals'] = data_info['categoricals'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
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


def return_train_test(df, train_indices, test_indices, oversampling = False, categoricals = []):
    
    train = df.iloc[train_indices]
    test = df.iloc[test_indices]
    
    if oversampling:
        train = oversample(df, categoricals)
    
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
    
    if model == 'gsgp':
        best_individual = gsgp(
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
    class_weights = compute_class_weights(y_train)
    # class_weight_dict = {int(label): weight for label, weight in zip(np.unique(y_train), class_weights)}
    train_sample_weights = torch.tensor([class_weights[int(label)] for label in y_train], dtype=torch.float32)
    test_sample_weights = torch.tensor([class_weights[int(label)] for label in y_test], dtype=torch.float32)
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




def fill_config(model_config, oversampling, fitness_function, minimization, inflation_rate, ms_upper):
    model_config["oversampling"] = oversampling
    model_config['config']["fitness_function"] = fitness_function
    model_config['config']["minimization"] = minimization
    
    if model_config['name'] == 'slim':
        model_config['config']['p_inflate'] = inflation_rate
        model_config['config']['ms_upper'] = ms_upper
    
    if model_config['name'] == 'gsgp':
        model_config['config']['ms_upper'] = ms_upper
    
    return model_config


