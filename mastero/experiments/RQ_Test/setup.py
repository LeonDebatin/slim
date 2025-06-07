# %%
#How does inflation/deflation rate effect the results?

# %%
import sys
import os
sys.path.insert(0, os.path.abspath(".."))
import argparse
from monte_carlo import MonteCarlo
from experiment import Experiment
from basic_model_config import *
from utils import fill_config
import copy



def parse_args():
    parser = argparse.ArgumentParser(description="Run inflation/deflation experiments.")
    parser.add_argument("--experiment_name", type=str, default="RQ_Test", help="Name of the experiment")
    parser.add_argument("--dataset_name", type=str, default="wilt", help="Dataset name")
    parser.add_argument("--n_runs", type=int, default=30, help="Number of runs per config")
    parser.add_argument("--log", action="store_true", help="Enable logging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def main():
    args = parse_args()
    rq_configs = []
    base_configs = [basic_model_config_slim_mulsig1, basic_model_config_slim_plussig1]

    for baseconfig in base_configs:
        for inflation_rate in [0.1, 0.3, 0.5 , 0.9]:
            for ms_upper in [0.1, 0.5, 1, 5]:
                config = copy.deepcopy(baseconfig)
                config = fill_config(config, scaling=True, oversampling=False,
                                     fitness_function='sigmoid_rmse', minimization=True,
                                     inflation_rate=inflation_rate, ms_upper=ms_upper)
                rq_configs.append(config)

    with open("../RQ_Fitness/n_configs.txt", "r") as f:
        config_counter = int(f.read())

    
    mc= MonteCarlo(
                    experiment_name=args.experiment_name,
                    dataset_name=args.dataset_name,
                    data_filepath="../../data/",
                    model_configs=rq_configs,
                    n_runs=args.n_runs, 
                    log=True, 
                    verbose = True,
                    config_counter_start= config_counter
                    )
    mc.run()

if __name__ == "__main__":
    main()


    # exp1 = Experiment(
    #     args.experiment_name,
    #     "../../data/",
    #     rq_configs,
    #     n_runs=args.n_runs,
    #     log=args.log,
    #     verbose=args.verbose,
    #     config_counter_start=config_counter
    # )
    # exp1.run()