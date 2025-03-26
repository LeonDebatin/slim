from monte_carlo import MonteCarlo
from utils import load_and_adapt_data_info
import concurrent.futures




class Experiment():
    
    def __init__ (
        self,
        experiment_name,
        data_filepath,
        model_configs,  # List of dictionaries
        n_runs=30,
        log=False,
        verbose=False
    ):
        self.experiment_name = experiment_name
        self.data_filepath = data_filepath
        self.model_configs = model_configs
        self.n_runs = n_runs
        self.data_info = load_and_adapt_data_info(f"{data_filepath}data_info.csv")
        self.verbose = verbose
        self.log = log

    def _run_montecarlo(self, data_name):
        """Helper function to run a MonteCarlo simulation."""
        MonteCarlo(
            experiment_name=self.experiment_name,
            dataset_name=data_name,
            data_filepath=self.data_filepath,
            model_configs=self.model_configs,
            n_runs=self.n_runs,
            log=self.log,
            verbose=self.verbose,
            data_info=self.data_info
        ).run()

    def run(self, parallel=False):
        """Runs MonteCarlo simulations, optionally in parallel."""
        if parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:  # Use ProcessPoolExecutor for CPU-bound tasks
                executor.map(self._run_montecarlo, self.data_info['name'])
        else:
            for data_name in self.data_info['name']:
                self._run_montecarlo(data_name)
