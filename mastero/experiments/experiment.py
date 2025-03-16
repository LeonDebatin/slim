from monte_carlo import MonteCarlo
from utils import load_and_adapt_data_info

class Experiment():
    
    def __init__ (
        self,
        experiment_name,
        data_filepath,
        model_configs, #list of dictionaries
        n_runs=30,
        log = False,
        verbose = False
    ):
        self.experiment_name = experiment_name
        self.data_filepath = data_filepath
        self.model_configs = model_configs
        self.n_runs = n_runs
        self.data_info = load_and_adapt_data_info(f"{data_filepath}data_info.csv")
        self.verbose = verbose
        self.log = log
        
    
    def run(self):
        
        for data_name in self.data_info['name']:
            
            MonteCarlo(
                experiment_name = self.experiment_name,
                dataset_name = data_name,
                data_filepath = self.data_filepath,
                model_configs = self.model_configs,
                n_runs = self.n_runs,
                log = self.log,
                verbose = self.verbose,
                data_info=self.data_info
            ).run()