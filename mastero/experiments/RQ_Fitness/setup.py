# %%
# how does inflation/deflation rate effect the results?

# %%
import sys
import os
sys.path.insert(0, os.path.abspath(".."))
import copy

from monte_carlo import MonteCarlo
from experiment import Experiment
from basic_model_config import *
from utils import fill_config
import argparse
# %%
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="Run Fitness experiments.")
    parser.add_argument("--experiment_name", type=str, default="RQ_Test", help="Name of the experiment")
    parser.add_argument("--dataset_name", type=str, default="wilt", help="Dataset name")
    return parser.parse_args()

def main():
    args = parse_args()
    rq_configs = []
    base_configs = [basic_model_config_gsgp] #, basic_model_config_slim_mulsig2, basic_model_config_slim_plussig2

    for baseconfig in base_configs:

        for ff in ['sigmoid_rmse', 'weighted_sigmoid_rmse',  'accuracy', 'f1_score']:
            if ff == 'sigmoid_rmse':
                minimization = True
            elif ff == 'weighted_sigmoid_rmse':
                minimization = True
            else:
                minimization = False
            
            for ms_upper in [0.1, 0.5, 1, 5]:
                config = copy.deepcopy(baseconfig)  # Fresh copy for each function
                config = fill_config(config, scaling = True, oversampling= False, fitness_function=ff, minimization=minimization, inflation_rate=None,  ms_upper=ms_upper)
                rq_configs.append(config)

    for config in rq_configs:
        print(config['name'],  config['config']['fitness_function'], config['config']['minimization'], config['config']['ms_upper'])

    n_configs = len(rq_configs)
    print(n_configs)
    with open("n_configs.txt", "w") as f:
        f.write(f"{n_configs}")

    # %%
    mc= MonteCarlo(
                    experiment_name=args.experiment_name,
                    dataset_name=args.dataset_name,
                    data_filepath="../../data/",
                    model_configs=rq_configs,
                    n_runs=30, 
                    log=True, 
                    verbose = True,
                    config_counter_start= 0
                    )
    mc.run()

if __name__ == "__main__":
    main()

