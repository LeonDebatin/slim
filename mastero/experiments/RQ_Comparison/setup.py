# %%
# how does slim perfrom comapred to gp and gsgp?

# %%
import sys
import os
sys.path.insert(0, os.path.abspath(".."))
import copy
import argparse
from monte_carlo import MonteCarlo
from experiment import Experiment
from basic_model_config import *
from utils import fill_config
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="Run Comparison experiments.")
    parser.add_argument("--experiment_name", type=str, default="RQ_Test", help="Name of the experiment")
    parser.add_argument("--dataset_name", type=str, default="wilt", help="Dataset name")
    return parser.parse_args()

def main():
    args = parse_args()
    rq_configs = []
    base_configs = [basic_model_config_gp]

    for baseconfig in base_configs:
            config = copy.deepcopy(baseconfig)
            config = fill_config(config, scaling = True, oversampling= False, fitness_function='sigmoid_rmse', minimization=True, inflation_rate=None, ms_upper=None)
            rq_configs.append(config)

    for config in rq_configs:
            print(config['name'])


    # %%
    with open("../RQ_Fitness/n_configs.txt", "r") as f:
        config_counter1 = int(f.read())
    config_counter1

    with open("../RQ_Inflationrate/n_configs.txt", "r") as f:
        config_counter2 = int(f.read())
        
    config_counter = config_counter1 + config_counter2

    # %%
    mc= MonteCarlo(
                    experiment_name=args.experiment_name,
                    dataset_name=args.dataset_name,
                    data_filepath="../../data/",
                    model_configs=rq_configs,
                    n_runs=30, 
                    log=True, 
                    verbose = True,
                    config_counter_start= config_counter
                    )
    mc.run()

if __name__ == "__main__":
    main()
