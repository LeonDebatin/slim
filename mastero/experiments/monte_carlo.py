from utils import *
import pandas as pd
import csv

class MonteCarlo():
    
    def __init__ (
        self,
        experiment_name,
        dataset_name,
        data_filepath,
        model_configs, #list of dictionaries
        n_runs=30,
        log = False,
        verbose = False,
        data_info = None,
        config_counter_start = 0
    ):
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.data_filepath = data_filepath
        self.model_configs = model_configs
        self.n_runs = n_runs
        self.data = pd.read_csv(f"{data_filepath}data_prepared/{dataset_name}.csv")
        self.verbose = verbose
        self.log = log
        self.config_counter_start = config_counter_start
        
        if data_info is not None:
            self.data_info = data_info
        else:
            self.data_info = load_and_adapt_data_info(f"{data_filepath}data_info.csv")
        
        
    
    def run(self):
        
        if self.log:
            
            if self.verbose:
                print('Results are being logged in the following path: ', f'{self.data_filepath}results/{self.experiment_name}/{self.dataset_name}')
            
            if not os.path.exists(f'{self.data_filepath}results/{self.experiment_name}/{self.dataset_name}'):
                os.makedirs(f'{self.data_filepath}results/{self.experiment_name}/{self.dataset_name}')

            # with open(f'{self.data_filepath}results/{self.experiment_name}/{self.dataset_name}/results.csv', mode="w", newline="") as file:
            #     writer = csv.writer(file)
            #     writer.writerow(["config_id", "run_id", "config", "metrics", "nodes_count"])
            
            config_counter = self.config_counter_start
        
            
        for model_config in self.model_configs:
            print(f"Running {model_config['name']} on {self.dataset_name}...")
            if self.log:
                config_counter = config_counter + 1
            
            for i in range(9, self.n_runs+1, 1):
                
                if self.log:
                    config_id = config_counter
                    run_id = i
                    model_config['config']['log_path'] = f'{self.data_filepath}results/{self.experiment_name}/{self.dataset_name}/log_config_id_{config_id}.csv'
                    model_config['config']['log_level'] = 1
                
                
                model_config['config']['seed'] = i
                train_indices = self.data_info.loc[self.data_info['name']== self.dataset_name, 'train_indices'].values[0][i-1]
                test_indices = self.data_info.loc[self.data_info['name']== self.dataset_name, 'test_indices'].values[0][i-1]

                X_train, y_train, X_test, y_test = return_train_test(
                                                                        df = self.data, 
                                                                        train_indices = train_indices, 
                                                                        test_indices = test_indices, 
                                                                        oversampling = model_config['oversampling'], 
                                                                        categoricals = self.data_info.loc[self.data_info['name']== self.dataset_name, 'categoricals'].values[0]
                                                                    )

                update_sample_weights(y_train, y_test)
                
                best_individual = train_model(
                                                dataset_name = self.dataset_name, 
                                                X_train = X_train, 
                                                y_train = y_train, 
                                                X_test = X_test, 
                                                y_test = y_test,
                                                model = model_config['name'], 
                                                **model_config['config']
                                            )
                
                metrics  = {}
                metrics['train'] = get_evaluation_dictionary(y_train, best_individual.predict(X_train))
                metrics['test'] = get_evaluation_dictionary(y_test, best_individual.predict(X_test))
                
                if model_config['name'] == 'slim':
                    node_count = best_individual.nodes_count
                elif model_config['name'] == 'gp':
                    node_count = best_individual.node_count
                else:
                    node_count = best_individual.nodes
                
                if self.verbose:
                    print(
                        f"Run {i} - Accuracy: {round(metrics['train']['accuracy'], 3)} | "
                        f"{round(metrics['test']['accuracy'], 3)} - "
                        f"ROC: {round(metrics['train']['roc_auc'], 3)} | "
                        f"{round(metrics['test']['roc_auc'], 3)} - "
                        f"F1: {round(metrics['train']['f1_score'], 3)} | "
                        f"{round(metrics['test']['f1_score'], 3)} - "
                        f"RMSE: {round(metrics['train']['rmse'], 3)} | "
                        f"{round(metrics['test']['rmse'], 3)} -"
                        f" Weighted RMSE: {round(metrics['train']['wrmse'], 3)} | "
                        f"{round(metrics['test']['wrmse'], 3)}"
                        )

                if self.log:
                    
                    with open(f'{self.data_filepath}results/{self.experiment_name}/{self.dataset_name}/results.csv', mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([config_id, run_id, model_config, metrics, node_count])

        
        return None
        
        