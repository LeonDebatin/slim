import pandas as pd
import ast
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


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


def get_performance_difference_significance(performance1, performance2):
    return stats.wilcoxon(performance1, performance2).pvalue

def get_performance_difference_significance_table(results, config, metric, pivot=False):
    performances = get_aggregated_performance(results, metric)
    significances = []
    for dataset_name in results['dataset_name'].unique():
        results_dataset = results.loc[results['dataset_name'] == dataset_name]
        config_results = results_dataset.loc[results_dataset['config_settings'] == config, metric]
        
        for config2 in results['config_settings'].unique():
            if config2 != config:
                config2_results = results_dataset.loc[results_dataset['config_settings'] == config2, metric]
                performance1 = performances.loc[dataset_name, config]
                performance2 = performances.loc[dataset_name, config2]
                p_value = get_performance_difference_significance(config_results, config2_results)
                sign = ''
                
                if p_value < 0.05:
                    if performance1 > performance2:
                        sign = '+'
                    else:
                        sign = '-'
                else:
                    sign = '≈'
                
                significances.append([dataset_name, config, config2, performance1, performance2, p_value, sign])
                
    sig_df = pd.DataFrame(significances, columns=['dataset_name', 'config1', 'config2',  'performance1', 'performance2', 'p_value', 'sign'])
    
    
    if pivot:
        sig_df['config1_v_config2'] = f'{config}_VS_' + sig_df['config2']
        sig_df['p_value'] = sig_df.apply(
            lambda row: f"{row['p_value']:.3f} ({row['sign']})" ,  axis=1
        )
        
        return sig_df.pivot_table(index='dataset_name', columns='config1_v_config2', values='p_value', aggfunc='first')
    else:
        return sig_df


def sci_notation(x):
    return "{:.3e}".format(x).replace("e", " × 10^")

def get_aggregated_performance(results, metric, agg='mean', algorithm=None):
    if algorithm is not None:
        results = results.loc[results['name'] == algorithm]
    
    if agg == 'mean':
        aggregated_performance = results.groupby(['dataset_name', 'config_settings', 'run_id'])[metric].mean().unstack().mean(axis=1).sort_values(ascending=False).unstack()
    elif agg == 'median':
        aggregated_performance = results.groupby(['dataset_name', 'config_settings', 'run_id'])[metric].median().unstack().median(axis=1).sort_values(ascending=False).unstack()
    
    return aggregated_performance

def get_rankings(results, metric, algorithm=None):
    if algorithm is not None:
        results = results.loc[results['name'] == algorithm]
    #gets ranking for each dataset
    return get_aggregated_performance(results, metric, 'mean').rank(axis=1, ascending=False, method='min')

def get_avg_ranking(results, metric, algorithm=None):
    if algorithm is not None:
        results = results.loc[results['name'] == algorithm]
    #gets averaged ranking accross datasets
    return get_rankings(results, metric).mean(axis=0).sort_values().reset_index().rename(columns={0: 'avg_rank'})

def get_ranking_significance(rankings):
    #used with get_rannkings to test for significance differences for rankings accross datasets
    return stats.friedmanchisquare(*[rankings[col] for col in rankings.columns]).pvalue


def plot_avg_ranking(rankings, show = True):
    import numpy as np
    avg_rank = rankings.set_index('config_settings')
    y_positions = np.random.uniform(0, 0, len(avg_rank))

    plt.figure(figsize=(10, 2))  # Adjust figure size

    # Generate colors using a colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(avg_rank)))

    # Plot points with different colors
    plt.scatter(avg_rank.values, y_positions, c=colors, edgecolors='black', zorder=3)

    # Annotate each point
    xy_texts = [(-10,30), (-20,-30), (-20,30), (0,-30),(0,30), (0,-30), (50,30), (0,30)]
    i = -1
    for model, x, y, color in zip(avg_rank.index, avg_rank.values, y_positions, colors):
        i += 1
        plt.annotate(model, (x, y), textcoords="offset points", xytext=xy_texts[i], ha='center', size=16,
                        bbox=dict(boxstyle="square,pad=0.5", fc="white",  ec="gray", lw=0.5),
                        arrowprops=dict(arrowstyle="-",  edgecolor='black', lw=0.5, facecolor='white'),) #color="black", #connectionstyle="arc3,rad=0.1" arrowstyle wedge


        plt.grid(axis='y',  alpha=0.0)
        plt.ylim(-10, 10)  # Limit y-axis

        # Add a **single x-grid line at y = 0**
        plt.axhline(0, color='gray', linestyle='-', alpha=0.5) 
        # Format x-axis (ranks)
        plt.xticks(np.arange(avg_rank.values.min(), avg_rank.values.max()+0.5, 0.5))  # Set x-ticks at a step of 0.5
        plt.xticks(np.arange(avg_rank.values.min(), avg_rank.values.max()+0.5, 0.25), minor=True) 
        plt.grid(which='both', axis='x',  linewidth=1)
        plt.yticks([])  # Hide y-axis ticks
        plt.xticks(fontsize=14)
    if show:
        plt.show()
    return


# def generations_plot(logs, dataset_name, value):
#     fig, axs = plt.subplots(1, 2, figsize=(12, 4))
#     sns.lineplot








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






def get_config_settings(results):
    settings = results[['config_id', 'name']].drop_duplicates().sort_values('config_id')
    settings['config_settings'] = settings['name']
    settings = settings[['config_id', 'config_settings']]
    results = results.merge(settings, on='config_id')
    return results

class Analysis():
    def __init__(
        self,
        experiment_name
        ):
        
        self.results = get_all_results(experiment_name)
        self.results = get_config_settings(self.results)
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