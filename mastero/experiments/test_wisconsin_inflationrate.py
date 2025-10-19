# %%
#How does inflation/deflation rate effect the results?

# %%
import sys
import os
sys.path.insert(0, os.path.abspath("../.."))  # Add slim directory to path
sys.path.insert(0, os.path.abspath(".."))     # Add experiments directory to path
import copy
from monte_carlo import MonteCarlo
from experiment import Experiment
from basic_model_config import *
from utils import fill_config
import argparse
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="Run inflation/deflation experiments.")
    parser.add_argument("--experiment_name", type=str, default="RQ_Test_biomed", help="Name of the experiment")
    parser.add_argument("--dataset_name", type=str, default="biomed", help="Dataset name")
    return parser.parse_args()

def main():
    args = parse_args()
    # Define the configurations for the experiment

    rq_configs = []
    base_configs = [basic_model_config_slim_mulsig1, basic_model_config_slim_plussig1] #, basic_model_config_slim_mulsig2, basic_model_config_slim_plussig2

    for baseconfig in base_configs:

            for inflation_rate in [0.1, 0.3, 0.5 , 0.9]:
                for ms_upper in [0.1, 0.5, 1, 5]:
                    config = copy.deepcopy(baseconfig)  # Fresh copy for each function
                    config = fill_config(config, scaling = True, oversampling= False, fitness_function='sigmoid_rmse', minimization=True, inflation_rate=inflation_rate, ms_upper=ms_upper)
                    rq_configs.append(config)

    for config in rq_configs:
        print(config['config']['slim_version'], config['oversampling'], config['config']['fitness_function'], config['config']['p_inflate'], config['config']['ms_upper'])

    n_configs = len(rq_configs)
    print(n_configs)
    with open("n_configs.txt", "w") as f:
        f.write(f"{n_configs}")

    # %%
    mc= MonteCarlo(
                    experiment_name=args.experiment_name,
                    dataset_name=args.dataset_name,
                    data_filepath="../data2/",  # Use data2 folder
                    model_configs=rq_configs,
                    n_runs=1,  # Just 1 run for testing
                    log=True, 
                    verbose = True,
                    config_counter_start= 16  # Start from 0
                    )
    mc.run()

if __name__ == "__main__":
    main()