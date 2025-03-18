import pandas as pd
import ast
import os
import seaborn as sns
import matplotlib.pyplot as plt



def expand_dict(df, column):
    """
    Expand a column of a dataframe that contains dictionaries into separate columns.
    
    Args:
        df (pd.DataFrame): The dataframe.
        column (str): The column to expand.
        
    Returns:
        pd.DataFrame: The dataframe with the expanded columns.
    """
    
    df[column] = df[column].apply(ast.literal_eval)
    df_expanded = df.join(pd.json_normalize(df[column]))
    df_expanded = df_expanded.drop(columns=[column])
    
    return df_expanded

def get_results(experiment, dataset_name):
    """
    Load the results of an experiment for a specific dataset.
    
    Args:
        experiment (str): The name of the experiment.
        dataset_name (str): The name of the dataset.
        
    Returns:
        pd.DataFrame: The results.
    """
    
    results = pd.read_csv(f"../../data/results/{experiment}/{dataset_name}/results.csv")
    results = expand_dict(results, 'config')
    results = expand_dict(results, 'metrics')
    
    return results

def get_all_results(experiment):
    """
    Load the results of an experiment for all datasets.
    
    Args:
        experiment (str): The name of the experiment.
        
    Returns:
        pd.DataFrame: The results.
    """
    
    results_list = []  # Store results in a list first

    for dataset_name in os.listdir(f"../../data/results/{experiment}"):
        res_fordataname = get_results(experiment, dataset_name)
        res_fordataname['dataset_name'] = dataset_name
        results_list.append(res_fordataname)  # Store in a list instead of appending to DataFrame

    # Concatenate all results at once (Faster than appending in a loop)
    results = pd.concat(results_list, ignore_index=True)
    
    if 'config.slim_version' in results.columns:
        results.loc[results['config.slim_version'].notna(), 'name'] = results['config.slim_version']

    results['name'] = results['name'].str.upper()
    
    
    return results


def get_average_ranking(results, metric):
    """
    Calculate the average ranking of models based on a metric.
    
    Args:
        results (pd.DataFrame): The results.
        metric (str): The metric to rank by.
        
    Returns:
        pd.DataFrame: The average ranking.
    """
    
    results = results.sort_values(metric, ascending=False)
    results['rank'] = results.groupby('dataset_name')[metric].rank(ascending=False)
    average_ranking = results.groupby('config_id')['rank'].mean().sort_values()
    
    return average_ranking


def get_rankings(results, metric):
    results_avg = results.groupby(["dataset_name", "config_id"], as_index=False)[metric].mean()

    # Rank the averaged scores within each dataset (higher is better)
    results_avg["rank"] = results_avg.groupby("dataset_name")[metric].rank(ascending=False, method="min")


    return results_avg






def get_log(experiment, dataset_name, config_id, add_columns = True):
    log = pd.read_csv(f"../../data/results/{experiment}/{dataset_name}/log_config_id_{config_id}.csv")
    log['config_id'] = config_id
    if add_columns:
        log.columns = ['algorithm', 'id', 'dataset', 'seed', 'generation', 'elite_train_error', 'time', 'population_nodes', 'elite_test_error', 'elite_nodes', 'log_level', 'config_id']
    return log


def get_logs(experiment, dataset_name, add_columns=True):
    logs = []
    for log_file in os.listdir(f"../../data/results/{experiment}/{dataset_name}"):
        if 'log_config_id' in log_file and 'settings' not in log_file:
            config_id = int(log_file.split('_')[-1].split('.')[0])
            logs.append(get_log(experiment, dataset_name, config_id))
    
    logs = pd.concat(logs, ignore_index=True)
    if add_columns:
        logs.columns = ['algorithm', 'id', 'dataset', 'seed', 'generation', 'elite_train_error', 'time', 'population_nodes', 'elite_test_error', 'elite_nodes', 'log_level', 'config_id']
    
    return logs

def get_all_logs(experiment):
    logs = []
    for dataset_name in os.listdir(f"../../data/results/{experiment}"):
        logs.append(get_logs(experiment, dataset_name))
    
    logs = pd.concat(logs, ignore_index=True)
    logs.columns = ['algorithm', 'id', 'dataset', 'seed', 'generation', 'elite_train_error', 'time', 'population_nodes', 'elite_test_error', 'elite_nodes', 'log_level', 'config_id']
    
    logs.drop(columns=['id'], inplace=True)
    logs.loc[logs['algorithm'] == 'StandardGP', 'algorithm'] = 'GP'
    logs.loc[logs['algorithm'] == 'StandardGSGP', 'algorithm'] = 'GSGP'
    #last to first column
    logs.sort_values(by=['dataset', 'config_id', 'seed', 'generation'], inplace=True)
    cols = logs.columns.tolist()
    logs = logs[[cols[-1]] + cols[:-1]]
    logs
    
    return logs

def plot_value_by_generations(logs, value, dataset_name, ax):
    """Plots value over generations for a specific dataset on a given subplot axis."""
    # Group and calculate median
    grp = logs.groupby(['config_id', 'algorithm', 'dataset', 'generation'])[[value]].median().reset_index()
    
    # Plot using seaborn on the given subplot axis
    sns.lineplot(data=grp.loc[grp['dataset'] == dataset_name], x='generation', y=value, hue='algorithm', ax=ax)
    
    # Set labels and title
    ax.set_title(f'{value} by Generation\n({dataset_name})', fontsize=10)
    ax.set_xlabel('Generation')
    ax.set_ylabel(value)
    ax.legend(fontsize=8)


class Analysis():
    def __init__(
        self,
        experiment_name
        ):
        
        self.results = get_all_results(experiment_name)
        self.logs = get_all_logs(experiment_name)
        self.experiment_name = experiment_name
    
    
    def get_ranks_by_metric(self):
        ranks = {}
        for metric in ['test.accuracy', 'test.f1_score', 'test.roc_auc']:
            ranks[metric] = get_rankings(self.results, metric)

        self.ranks_by_metric = ranks
        return ranks
    
    
    
    def plot_value_by_generations_for_experiment(self, value):
        grp = self.results.groupby(['config_id', 'algorithm', 'dataset', 'generation'])[['elite_train_error', 'elite_test_error', 'elite_nodes']].median().reset_index()
        sns.lineplot(data=grp.loc[(grp['dataset'] == 'blood') & (grp['config_id'] < 4)], x='generation', y='elite_train_error', hue='config_id')
        plt.show()