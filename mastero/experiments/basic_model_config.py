import numpy as np
import copy

basic_config_all = {
    'pop_size': 100, #100
    'n_iter': 1000,  #1000
    'elitism': True,
    'n_elites': 1,
    'init_depth': 6,
    'initializer': 'rhh',
    'tournament_size': 2,
    'prob_const': 0.2,
    'tree_functions': ['add', 'subtract', 'multiply', 'divide'],
    'tree_constants': np.linspace(-10, 10, num=201).round(1).tolist(),
    
    'fitness_function': None, #must be defined
    'minimization' : None, #must be defined
    'seed': None, #must be defined
    
    'log_path' : None, #must be defined
    'verbose': False, 
    'log_level': 0,
    'test_elite': True
}

basic_config_gp = {
    'p_xo': 0.8,
    'max_depth': 17
}

basic_model_config_gp = {
                        'name': 'gp',
                        'scaling': None,
                        'oversampling': None,
                         'config': {**basic_config_all, **basic_config_gp}
                        }

#gsgp changed to slim because bug in tree node counter, therefore slim+sig2 is used with inflationrate of 1.0
basic_config_gsgp = {
    'slim_version': 'SLIM+SIG2',
    'ms_lower': 0,
    'ms_upper': None,
    'p_inflate': 1.0,
    'reconstruct': True,
    'copy_parent': True,
    'max_depth': None
}

basic_model_config_gsgp = {
                        'name': 'gsgp',
                        'scaling': None,
                        'oversampling': None,
                         'config': {**basic_config_all, **basic_config_gsgp}
                        }

basic_config_slim= {
    'slim_version': None,
    'ms_lower': 0,
    'ms_upper': None,
    'p_inflate': None,
    'reconstruct': True,
    'copy_parent': True,
    'max_depth': None
}

basic_model_config_slim = {
                        'name': 'slim',
                        'scaling': None,
                        'oversampling': None,
                         'config': {**basic_config_all, **basic_config_slim}
                        }

basic_model_config_slim_plussig2 = copy.deepcopy(basic_model_config_slim)
basic_model_config_slim_plussig2['config']['slim_version'] = 'SLIM+SIG2'

basic_model_config_slim_mulsig2 = copy.deepcopy(basic_model_config_slim)
basic_model_config_slim_mulsig2['config']['slim_version'] = 'SLIM*SIG2'

basic_model_config_slim_plusabs = copy.deepcopy(basic_model_config_slim)
basic_model_config_slim_plusabs['config']['slim_version'] = 'SLIM+ABS'

basic_model_config_slim_mulabs = copy.deepcopy(basic_model_config_slim)
basic_model_config_slim_mulabs['config']['slim_version'] = 'SLIM*ABS'

basic_model_config_slim_plussig1 = copy.deepcopy(basic_model_config_slim)
basic_model_config_slim_plussig1['config']['slim_version'] = 'SLIM+SIG1'

basic_model_config_slim_mulsig1 = copy.deepcopy(basic_model_config_slim)
basic_model_config_slim_mulsig1['config']['slim_version'] = 'SLIM*SIG1'