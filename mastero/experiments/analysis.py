import pandas as pd
import ast
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.ticker as ticker
import re
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

colors ={
            'algorithm' : {
                'SLIM+SIG1': '#3949ab',
                'SLIM*SIG1': '#1e88e5',
                'GP': '#fb8c00',
                'GSGP': '#ffb300',
            },
            'fitness_function':{
                'RMSE': '#3949ab',
                'WRMSE': '#1e88e5',
                'Accuracy': '#fb8c00',
                'F1-Score': '#ffb300',
            },
            'ms_upper': {
                0.1: '#3949ab',   # light but saturated blue
                0.5: '#1e88e5',   # medium blue, noticeably darker
                1:   '#fb8c00',   # dark blue
                5:   '#ffb300'    # nearly blue-black, very deep blue
        }
}
orders = {
    'fitness_function': ['RMSE', 'WRMSE', 'Accuracy', 'F1-Score'],
    'algorithm': ['GP', 'GSGP','SLIM+SIG1', 'SLIM*SIG1',],
}


hue_order_slim = ['SLIM+SIG1', 'SLIM*SIG1']


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

def get_results(experiment, dataset):
    """
    Load the results of an experiment for a specific dataset.
    
    Args:
        experiment (str): The name of the experiment.
        dataset (str): The name of the dataset.
        
    Returns:
        pd.DataFrame: The results.
    """
    
    results = pd.read_csv(f"../../data/results/{experiment}/{dataset}/results.csv")
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

    for dataset in os.listdir(f"../../data/results/{experiment}"):
        res_fordata = get_results(experiment, dataset)
        res_fordata['dataset'] = dataset
        results_list.append(res_fordata)  # Store in a list instead of appending to DataFrame

    # Concatenate all results at once (Faster than appending in a loop)
    results = pd.concat(results_list, ignore_index=True)
    
    if 'config.slim_version' in results.columns:
        results.loc[results['config.slim_version'].notna(), 'name'] = results['config.slim_version']
    results.rename(columns={'name': 'algorithm'}, inplace=True)
    results['algorithm'] = results['algorithm'].str.upper()
    results['dataset'] = results['dataset'].str.capitalize()
    return results

def table_to_latex(table, experiment, name, caption, index=False):
    # Just use escape=True, don't do any manual replacements
    length = len(table.columns) if index else len(table.columns) - 1

    latex_table = table.to_latex(
        column_format="l" + "c" * length,
        escape=True,
        index=index
    )

    latex_code = f"""
    \\begin{{table}}[h]
        \\centering
        \\renewcommand{{\\arraystretch}}{{1.2}}
    {latex_table}
        \\caption{{{caption}}}
        \\label{{tab:{experiment}_{name}}}
    \\end{{table}}
    """
    with open(f"../Latex/Chapters/Tables/Results/{experiment}_{name}.tex", "w") as f:
        f.write(latex_code)

    return


def plot_to_latex(figure, experiment, name, caption):
    figure.savefig(f"../Latex/Chapters/Figures/Results/{experiment}_{name}.png", dpi=500, bbox_inches='tight', transparent=True)
    
    latex_code = f"""
    \\begin{{figure}}[h]
    \\centering
    \\includegraphics[width=\\linewidth]{{../Latex/Chapters/Figures/Results/{experiment}_{name}.png}}
    \\caption{{{caption}}}
    \\label{{fig:{name}}}
    \\end{{figure}}
    """
    with open(f"../Latex/Chapters/Figures/Results/{experiment}_{name}.tex", "w") as f:
        f.write(latex_code)
    return

def get_log(experiment, dataset, config_id, add_columns = True):
    log = pd.read_csv(f"../../data/results/{experiment}/{dataset}/log_config_id_{config_id}.csv")
    log['config_id'] = config_id
    if add_columns:
        log.columns = ['algorithm', 'id', 'dataset', 'seed', 'generation', 'elite_train_error', 'time', 'population_nodes', 'elite_test_error', 'elite_nodes', 'log_level', 'config_id']
    return log


def get_logs(experiment, dataset, add_columns=True):
    logs = []
    for log_file in os.listdir(f"../../data/results/{experiment}/{dataset}"):
        if 'log_config_id' in log_file and 'settings' not in log_file:
            config_id = int(log_file.split('_')[-1].split('.')[0])
            logs.append(get_log(experiment, dataset, config_id))
    
    logs = pd.concat(logs, ignore_index=True)
    if add_columns:
        logs.columns = ['algorithm', 'id', 'dataset', 'seed', 'generation', 'elite_train_error', 'time', 'population_nodes', 'elite_test_error', 'elite_nodes', 'log_level', 'config_id']
    logs['dataset'] = logs['dataset'].str.capitalize()
    return logs

def get_all_logs(experiment):
    logs = []
    for dataset in os.listdir(f"../../data/results/{experiment}"):
        logs.append(get_logs(experiment, dataset))
    
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



        
###mean and standard deviation
# def plot_value_by_generations(logs, value, ax, y_max=None):
#     """Plots mean and standard deviation of a value over generations for a specific dataset on a given subplot axis."""
#     # Group and calculate mean and std
#     stats = logs.groupby(['algorithm', 'dataset', 'generation'])[value].agg(['mean', 'std']).reset_index()

#     # Plot each algorithm separately
#     for algo in stats['algorithm'].unique():
#         data_algo = stats[stats['algorithm'] == algo]
#         ax.plot(data_algo['generation'], data_algo['mean'], label=algo, color=colors_dict.get(algo, None))
#         ax.fill_between(
#             data_algo['generation'],
#             data_algo['mean'] - data_algo['std'],
#             data_algo['mean'] + data_algo['std'],
#             color=colors_dict.get(algo, None),
#             alpha=0.3
#         )

#     # Set labels and title
#     ax.set_title(f'{logs["dataset"].iloc[0]}')
#     ax.set_xlabel('Generation')
#     ax.set_ylabel(value)
#     #ax.legend_.remove()
#     ax.yaxis.set_major_locator(ticker.LinearLocator(5))
#     if value == 'elite_nodes':
#         ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
#     else:
#         ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}")) 
#     if y_max:
#         ax.set_ylim(0, y_max)




def get_min_euclidian_distance(results):
    unique_datasets = results['dataset'].unique()
    unique_models = results['algorithm'].unique()
    
    best_configs = []
    
    for dataset in unique_datasets:
        for model in unique_models:
            subset = results[(results['dataset'] == dataset) & (results['algorithm'] == model)]
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(subset[['test.rmse', 'nodes_count']])
            subset.loc[:, ['test.rmse', 'nodes_count']] = scaled_values
            subset['euclidian_distance'] = (subset['test.rmse']**2 + 2*subset['nodes_count']**2)**0.5

            subset = subset.sort_values('euclidian_distance')
            subset = subset.drop_duplicates(subset=['dataset', 'algorithm'], keep='first')
            best_configs.append(subset)
    
    return pd.concat(best_configs, ignore_index=True)



def get_best_config(filtered_results, metric, minimization):

    # Step 1: median of all runs per config
    median_per_config = (
        filtered_results
        .groupby('config_id')[metric]
        .median()  # median across run_ids
        .reset_index()
    )

    # Step 2: select best config based on minimization or maximization
    best_config = (
        median_per_config
        .sort_values(by=metric, ascending=minimization)
        .iloc[0]
    )

    return best_config



def get_best_config_by_fitness_function(results):
    best_configs = []
    
    for dataset in results['dataset'].unique():
        filtered_by_dataset = results.loc[results['dataset'] == dataset]
        
        for ff in results['config.fitness_function'].unique():
            filtered_by_ff = filtered_by_dataset.loc[filtered_by_dataset['config.fitness_function'] == ff]
            
            if ff == 'sigmoid_rmse':
                best_config = get_best_config(filtered_by_ff, 'test.rmse', minimization=True)
                
            elif ff == 'weighted_sigmoid_rmse':
                best_config = get_best_config(filtered_by_ff, 'test.wrmse', minimization=True)
            
            elif ff == 'accuracy':
                best_config = get_best_config(filtered_by_ff, 'test.accuracy', minimization=False)
            
            elif ff == 'f1_score':
                best_config = get_best_config(filtered_by_ff, 'test.f1_score', minimization=False)

            else:
                raise ValueError(f"Unknown fitness function: {ff}")
            
            best_configs.append([dataset, ff, best_config['config_id']])

    best_configs_df = pd.DataFrame(best_configs, columns=['dataset', 'fitness_function', 'config_id'])
    return best_configs_df




def plot_performance_barplot(results_median, metrics, groupby):

    # Melt the DataFrame to long format
    df_long = results_median.melt(
        id_vars=['dataset', groupby],
        value_vars= metrics,
        var_name='metric',
        value_name='value'
    )

    unique_datasets = df_long['dataset'].unique()
    n_cols = 2
    n_rows = math.ceil(len(unique_datasets) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 2 * n_rows), squeeze=False)

    i=j=0
    for dataset in unique_datasets:
                
        subset = df_long[df_long['dataset'] == dataset]
        
        sns.barplot(
            data=subset,
            x='metric',
            y='value',
            hue= groupby,
            palette=colors[groupby],
            hue_order = orders[groupby],
            ax=axes[i, j],
        )
        
        axes[i,j].set_title(dataset)
        axes[i,j].set_xlabel("")
        axes[i,j].set_ylabel("Test Score")
        axes[i,j].tick_params(axis='x')
        
        # 
        # axes[i,j].set_yticks(np.arange(0, 1.1, 0.2))
        axes[i,j].legend_.remove()
        axes[i,j].yaxis.set_major_locator(ticker.LinearLocator(4))
        axes[i,j].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        min = subset['value'].min() - subset['value'].min() * 0.1
        max = subset['value'].max() + subset['value'].min() * 0.1
        max = 1.0 if max > 1.0 else max
        axes[i,j].set_ylim(min, max)
        
        axes[i, j].spines['right'].set_visible(False)
        axes[i, j].spines['top'].set_visible(False)
        
        
        j = (j + 1) % 2
        i = i +1 if j == 0 else i
        
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.02),  # centered just above the figure
        frameon=False
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at the top
    
    
    
    return fig, axes


def plot_performance_evolution(logs):
    unique_datasets = logs['dataset'].unique()
    fig, ax = plt.subplots(len(unique_datasets), 2, figsize=(8, 3 * int(len(unique_datasets)/2)), squeeze=False)

    for i, dataset in enumerate(unique_datasets):
        dataset_logs = logs[logs['dataset'] == dataset]
        plot_value_by_generations(dataset_logs, 'elite_train_error', ax=ax[i, 0], y_label= 'Train Score')
        plot_value_by_generations(dataset_logs, 'elite_test_error', ax=ax[i, 1], y_label = 'Test Score')

    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.02),
        frameon=False
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    
    return fig, ax

def plot_value_by_generations(logs, value, ax, y_max=None, y_label=''):
    """Plots value over generations for a specific dataset on a given subplot axis."""
    # Group and calculate median
    grp = logs.groupby(['config_id', 'algorithm', 'dataset', 'generation'])[[value]].median().reset_index()

    # Plot using seaborn on the given subplot axis
    sns.lineplot(data=grp, x='generation', y=value, hue='algorithm', ax=ax, palette=colors['algorithm'], hue_order=orders['algorithm'])

    # Set labels and title
    ax.set_title(f'{logs["dataset"].iloc[0]}')
    ax.set_xlabel('Generation')
    ax.set_ylabel(y_label)
    ax.legend_.remove()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.yaxis.set_major_locator(ticker.LinearLocator(4))
    if value == 'elite_nodes':
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    else:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"0" if x < 0 else f"{x:.2f}"))
    #ax.set_xlim(0, 2000)
    if y_max:
        ax.set_ylim(0, y_max)


def plot_tree_size_evolution(logs):
    unique_datasets = logs['dataset'].unique()
    fig, ax = plt.subplots(int(len(unique_datasets)/2), 2, figsize=(8, 1.5 * int(len(unique_datasets)/2)), squeeze=False)

    j = 0
    i = 0
    for dataset in unique_datasets:
        dataset_logs = logs[logs['dataset'] == dataset]
        plot_value_by_generations(dataset_logs, 'elite_nodes', ax=ax[i, j], y_max=1200, y_label='Tree Size')
        j = (j + 1) % 2
        i = i +1 if j == 0 else i
        
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.02),
        frameon=False
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    return fig, ax





def plot_performance_evolution_by_fitness_function(logs):
    unique_datasets = logs['dataset'].unique()
    fig, ax = plt.subplots(len(unique_datasets), 2, figsize=(8, 2 * int(len(unique_datasets)/2)), squeeze=False)

    for i, dataset in enumerate(unique_datasets):
        subset = logs[logs['dataset'] == dataset]
        grouped = subset.groupby(['config_id', 'generation', 'fitness_function'])[['elite_test_error', 'elite_train_error']].median().reset_index()
        grouped.loc[grouped['fitness_function'].isin(['sigmoid_rmse', 'weighted_sigmoid_rmse']), 
            ['elite_test_error', 'elite_train_error']] = 1 - grouped[['elite_test_error', 'elite_train_error']]
        
        sns.lineplot(
            data=grouped,
            x='generation',
            y='elite_train_error',
            hue='fitness_function',
            ax = ax[i, 0],
        )
        ax[i,0].set_title(dataset)
        ax[i,0].set_xlabel("Generation")
        ax[i,0].set_ylabel("Train Score")
        ax[i,0].set_yticks(np.arange(0.4, 1.1, 0.1))
        ax[i,0].set_ylim(0.5, 1.01)
        ax[i,0].legend_.remove()
        ax[i, 0].spines['right'].set_visible(False)
        ax[i, 0].spines['top'].set_visible(False)
        
        sns.lineplot(
            data=grouped,
            x='generation',
            y='elite_test_error',
            hue='fitness_function',
            ax = ax[i, 1],
        )
        ax[i,1].set_title(dataset)
        ax[i,1].set_xlabel("Generation")
        ax[i,1].set_ylabel("Test Score")
        ax[i,1].set_yticks(np.arange(0, 1.1, 0.1))
        ax[i,1].set_ylim(0, 1)
        ax[i,1].legend_.remove()
        ax[i, 0].spines['right'].set_visible(False)
        ax[i, 0].spines['top'].set_visible(False)
        
        
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.01),
        frameon=False
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1]) 

    plt.show()


def plot_tree_size_evolution_by_fitness_function(logs):
    unique_datasets = logs['dataset'].unique()
    fig, ax = plt.subplots(int(len(unique_datasets)/2), 2, figsize=(8, 2 * int(len(unique_datasets)/2)), squeeze=False)

    i=j=0
    for dataset in unique_datasets:
        subset = logs[logs['dataset'] == dataset]
        grouped = subset.groupby(['config_id', 'generation', 'fitness_function'])[['elite_nodes']].median().reset_index()
        sns.lineplot(
            data=grouped,
            x='generation',
            y='elite_nodes',
            hue='fitness_function',
            ax = ax[i, j],
            palette=colors['fitness_function'],
        )
        ax[i,j].set_title(dataset)
        ax[i,j].set_xlabel("Generation")
        ax[i,j].set_ylabel("Tree Size")
        ax[i,j].set_yticks(np.arange(0, 6000, 1000))
        ax[i,j].set_ylim(0, 5000)
        ax[i,j].legend_.remove()
        ax[i, j].spines['right'].set_visible(False)
        ax[i, j].spines['top'].set_visible(False)
        
        j = (j + 1) % 2
        i = i +1 if j == 0 else i
        
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.02),
        frameon=False
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    return fig, ax

def plot_performance_by_p_inflate_with_ms(results):
    unique_datasets = results['dataset'].unique()
    fig, ax = plt.subplots(len(unique_datasets), 2,  figsize=(8, 3 * int(len(unique_datasets)/2)), squeeze=False)
    
    for i, dataset in enumerate(unique_datasets):

        subset = results[results['dataset'] == dataset]

        sns.lineplot(
            data=subset,
            x='config.p_inflate',
            y='train.rmse',
            hue='config.ms_upper',
            style='algorithm',
            markers=True,
            dashes=True,
            palette = colors['ms_upper'],
            ax=ax[i, 0]
        )
        ax[i, 0].set_title(f'{dataset}')
        ax[i, 0].set_xlabel('Inflationrate')
        ax[i, 0].set_ylabel('Train Score')
        ax[i, 0].yaxis.set_major_locator(ticker.LinearLocator(3))
        ax[i, 0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"0" if x < 0 else f"{x:.2f}"))
        ax[i, 0].legend_.remove()
        ax[i, 0].spines['right'].set_visible(False)
        ax[i, 0].spines['top'].set_visible(False)
        
        sns.lineplot(
            data=subset,
            x='config.p_inflate',
            y='test.rmse',
            hue='config.ms_upper',
            style='algorithm',
            markers=True,
            dashes=True,
            palette = colors['ms_upper'],
            ax=ax[i, 1]
        )
        ax[i, 1].set_title(f'{dataset}')
        ax[i, 1].set_xlabel('Inflationrate')
        ax[i, 1].set_ylabel('Test Score')
        ax[i, 1].yaxis.set_major_locator(ticker.LinearLocator(3))
        ax[i, 1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"0" if x < 0 else f"{x:.2f}"))
        ax[i, 1].legend_.remove()
        ax[i, 1].spines['right'].set_visible(False)
        ax[i, 1].spines['top'].set_visible(False)
    
    
    handles, labels = ax[0,0].get_legend_handles_labels()
    handles_row1 = handles[:5]
    labels_row1 = labels[:5]
    labels_row1[0] = 'Upper Mutation Step'
    
    handles_row2 = handles[5:]
    labels_row2 = labels[5:]
    labels_row2[0] = 'Algorithm'
    # First row
    fig.legend(
        handles_row1, labels_row1,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False
    )

    # Second row
    fig.legend(
        handles_row2, labels_row2,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.03),  # slightly below the first
        ncol=3,
        frameon=False
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1]) 
    return fig, ax

def plot_tree_size_by_p_inflate_with_ms(results):
    
    unique_datasets = results['dataset'].unique()
    fig, ax = plt.subplots(int(len(unique_datasets)/2), 2, figsize=(8, 1.5 * int(len(unique_datasets)/2)), squeeze=False)
    
    i=j=0
    for dataset in unique_datasets:
        subset = results[results['dataset'] == dataset]
        sns.lineplot(
            data=subset,
            x='config.p_inflate',
            y='nodes_count',
            hue='config.ms_upper',
            style='algorithm',
            markers=True,
            dashes=True,
            palette = colors['ms_upper'],
            ax=ax[i, j]
        )
        ax[i, j].set_title(f'{dataset}')
        ax[i, j].set_xlabel('Inflationrate')
        ax[i, j].set_ylabel('Tree Size')
        ax[i, j].yaxis.set_major_locator(ticker.LinearLocator(3))
        ax[i, j].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"0" if x < 0 else f"{int(round(x, -2))}"))
        ax[i, j].legend_.remove()
        ax[i, j].spines['right'].set_visible(False)
        ax[i, j].spines['top'].set_visible(False)
        j = (j + 1) % 2
        i = i +1 if j == 0 else i

    handles, labels = ax[0,0].get_legend_handles_labels()
    handles_row1 = handles[:5]
    labels_row1 = labels[:5]
    labels_row1[0] = 'Upper Mutation Step'
    

    handles_row2 = handles[5:]
    labels_row2 = labels[5:]
    labels_row2[0] = 'Algorithm'

    # First row
    fig.legend(
        handles_row1, labels_row1,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=5,
        frameon=False
    )

    # Second row
    fig.legend(
        handles_row2, labels_row2,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),  # slightly below the first
        ncol=3,
        frameon=False
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1]) 
    return fig, ax



def plot_performance_complexity_tradeoff(results, model):
    unique_datasets = results['dataset'].unique()
    fig, axes = plt.subplots(int(len(unique_datasets)/2), 2, figsize=(8, 2.5 * int(len(unique_datasets)/2)))
    axes = axes.flatten()
    
    for i, dataset in enumerate(unique_datasets):
        subset = results[results['dataset'] == dataset]
        subset = subset[subset['algorithm'] == model]
        ax = axes[i]
        sns.scatterplot(
            data=subset,
            x='nodes_count',
            y='test.rmse',
            hue= 'config.ms_upper',
            size= 'config.p_inflate',
            palette = colors['ms_upper'],
            ax=ax
        )
        ax.set_title(f"{dataset}")
        ax.set_xlabel("Tree Size")
        ax.set_ylabel("Test Score")
        ax.legend_.remove()
        
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['top'].set_visible(False)
        axes[i].yaxis.set_major_locator(ticker.LinearLocator(4))
        axes[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"0" if x < 0 else f"{x:.2f}"))
        # min = subset['test.rmse'].min() - subset['test.rmse'].min() * 0.1
        # max = subset['test.rmse'].max() + subset['test.rmse'].min() * 0.1
        # axes[i].set_ylim(min, max)
        
    
        axes[i].xaxis.set_major_locator(ticker.LinearLocator(4))
        axes[i].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"0" if x < 0 else f"{int(round(x, -2))}"))
        # min = subset['nodes_count'].min()- subset['nodes_count'].min() * 0.1
        # max = subset['nodes_count'].max()+ subset['nodes_count'].min() * 0.1
        # axes[i].set_xlim(min, max)    
    
        

    handles, labels = axes[0].get_legend_handles_labels()
    handles_row1 = handles[:5]
    labels_row1 = labels[:5]
    labels_row1[0] = 'Upper Mutation Step'
    
    handles_row2 = handles[5:]
    labels_row2 = labels[5:]
    labels_row2[0] = 'Inflationrate'
    # First row
    fig.legend(
        handles_row1, labels_row1,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.03),
        ncol=5,
        frameon=False
    )

    fig.legend(
        handles_row2, labels_row2,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),  # slightly below the first
        ncol=5,
        frameon=False
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1]) 

    return fig, axes


def plot_ranks(wtl_agg, groupby):
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=wtl_agg, x='metric', y='rank', hue=groupby, marker='o', ax=ax, palette=colors[groupby], hue_order=orders[groupby])
    ax.invert_yaxis()
    ax.set_xlabel('')
    ax.set_ylabel('Average Rank')
    ax.legend_.remove()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.15),
        frameon=False
    )
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    return fig, ax




def get_friedman_rank_pvalues(wtl_detailed, groupby):
    p_values = []
    for metric in wtl_detailed['metric'].unique():
        performances = []
        for group in wtl_detailed[groupby].unique():
            performances.append(wtl_detailed[(wtl_detailed[groupby] == group) & (wtl_detailed['metric'] == metric)]['rank'].values)

        p_value = stats.friedmanchisquare(*performances).pvalue
        
        if p_value < 0.05:
            sig = 'Yes'
        else:
            sig = 'No'
        
        p_values.append([metric, p_value, sig])
    
    return pd.DataFrame(p_values, columns=['Metric', 'P-Value', 'Significant'])


def get_wilcoxon_rank_pvalues(wtl_detailed, groupby):
    from itertools import product
    p_values = []
    unique_groups = wtl_detailed[groupby].unique()
    unique_pairs = [(x, y) for x, y in product(unique_groups, repeat=2) if x != y]
    for metric in wtl_detailed['metric'].unique():
        
        for pair in unique_pairs:
            group1 = pair[0]
            group2 = pair[1]
            
            performances = []
            performances.append(wtl_detailed[(wtl_detailed[groupby] == group1) & (wtl_detailed['metric'] == metric)]['rank'].values)
            performances.append(wtl_detailed[(wtl_detailed[groupby] == group2) & (wtl_detailed['metric'] == metric)]['rank'].values)

            p_value = stats.wilcoxon(*performances).pvalue
            if p_value < 0.05:
                sig = 'True'
            else:
                sig = 'False'
            
            p_values.append([metric, group1, group2, p_value, sig])
    
    return pd.DataFrame(p_values, columns=['metric', f'{groupby}_1', f'{groupby}_2',  'p_value', 'significant'])



def get_win_tie_loss(results, metrics, minimization, groupby):
    wtl_1v1 = []
    memory = []
    unique_datasets = results['dataset'].unique()
    unique_groupby = results[groupby].unique()
    
    for dataset in unique_datasets:
        subset = results[results['dataset'] == dataset]
        
        for config_id1 in subset['config_id'].unique():       
            
            for metric in metrics:
                
                for config_id2 in list( set(subset['config_id'].unique()) - set([config_id1])):
                    if [dataset, metric, config_id1, config_id2] in memory:
                        continue
                    
                    win = tie = loss = 0
                    
                    performance1 = subset[subset['config_id'] == config_id1][metric].values
                    performance2 = subset[subset['config_id'] == config_id2][metric].values
                    
                    if np.all(performance1 == performance2):
                        p_value = 1.0
                    
                    else:
                        p_value = stats.wilcoxon(performance1, performance2).pvalue

                    if p_value >= 0.05:
                        tie = 1
                        
                    else:
                        if np.median(performance1) > np.median(performance2):
                            if minimization:
                                loss = 1
                                
                            else:
                                win = 1
                        else:
                            if minimization:
                                win = 1
                            else:
                                loss = 1
                    
                    memory.append([dataset, metric, config_id1, config_id2])
                    memory.append([dataset, metric, config_id2, config_id1])

                    wtl_1v1.append([dataset, config_id1, config_id2, metric, p_value, win, tie, loss])
                            
    wtl_1v1_df = pd.DataFrame(wtl_1v1, columns=['dataset', 'config_id1', 'config_id2', 'metric', 'p_value', 'win', 'tie', 'loss'])
    wtl_1v1_df = pd.merge(wtl_1v1_df, results[['dataset', 'config_id', groupby]].drop_duplicates().reset_index(drop=True),
                  left_on=['dataset', 'config_id1'], right_on=['dataset', 'config_id'], how='left')
    wtl_1v1_df.rename(columns={groupby: f'{groupby}_1'}, inplace=True)
    wtl_1v1_df = pd.merge(wtl_1v1_df, results[['dataset', 'config_id', groupby]].drop_duplicates().reset_index(drop=True),
                    left_on=['dataset', 'config_id2'], right_on=['dataset', 'config_id'], how='left')
    wtl_1v1_df.rename(columns={groupby: f'{groupby}_2'}, inplace=True)
    wtl_1v1_df.drop(columns=['config_id1', 'config_id2', 'config_id_x', 'config_id_y'], inplace=True)
    wtl_comparison_1v1 = wtl_1v1_df.groupby(['metric',f'{groupby}_1', f'{groupby}_2'])[['win', 'tie', 'loss']].sum()
    
    
    
    c1 = wtl_1v1_df.groupby(['dataset', f'{groupby}_1', 'metric'])[['win', 'tie', 'loss']].sum().reset_index()
    c2 = wtl_1v1_df.groupby(['dataset', f'{groupby}_2', 'metric'])[['win', 'tie', 'loss']].sum().reset_index().rename(columns={f'{groupby}_2': f'{groupby}_1', 'win': 'loss', 'loss':'win'})
    wtl_detailed = pd.concat([c1, c2], axis=0).groupby(['dataset',f'{groupby}_1', 'metric'])[['win', 'tie', 'loss']].sum().reset_index()
    wtl_detailed['sum'] = wtl_detailed['win'] + 0.5*wtl_detailed['tie']
    wtl_detailed['rank'] = (wtl_detailed['sum'] - int(len(unique_groupby)-1)) * -1 + 1
    
    wtl_agg = wtl_detailed.groupby(['metric',f'{groupby}_1'])[['win', 'tie', 'loss','rank']].agg({'win': 'sum', 'tie': 'sum', 'loss': 'sum', 'rank': 'mean'}).reset_index()
    wtl_agg.rename(columns={f'{groupby}_1': f'{groupby}'}, inplace=True)
    wtl_detailed.rename(columns={f'{groupby}_1': f'{groupby}'}, inplace=True)    
    
    return wtl_comparison_1v1, wtl_detailed, wtl_agg 


class Analysis():
    def __init__(
        self,
        experiment_name
        ):
        self.results = get_all_results(experiment_name)
        self.logs = get_all_logs(experiment_name)
        self.experiment_name = experiment_name
class FitnessAnalysis(Analysis):
    def __init__(self, experiment_name):
        super().__init__(experiment_name)
        self.best_configs = get_best_config_by_fitness_function(self.results)
        self.best_config_results = pd.merge(self.results, self.best_configs, on=['dataset', 'config_id'], how='inner')
        self.best_config_results_median = self.best_config_results.groupby(['dataset', 'config_id', 'fitness_function'])[['test.rmse', 'test.accuracy', 'test.f1_score', 'test.roc_auc', 'nodes_count']].median().reset_index()
        
        self.best_config_logs = pd.merge(self.logs, self.best_config_results_median[['dataset', 'config_id', 'fitness_function']], left_on=['dataset', 'config_id'], right_on=['dataset', 'config_id'], how='inner')
        self.wtl_1v1_max, self.wtl_detailed_max, self.wtl_agg_max = get_win_tie_loss(self.best_config_results.sort_values('config.fitness_function'), ['test.accuracy', 'test.f1_score', 'test.roc_auc'], minimization=False, groupby='config.fitness_function')
        self.wtl_1v1_min, self.wtl_detailed_min, self.wtl_agg_min = get_win_tie_loss(self.best_config_results.sort_values('config.fitness_function'), ['nodes_count'], minimization=True, groupby='config.fitness_function')
        self.wtl_1v1 = pd.concat([self.wtl_1v1_max.reset_index(), self.wtl_1v1_min.reset_index()], axis=0).reset_index(drop=True)
        
        #self.wtl_1v1 = self.wtl_1v1.rename(columns={'config.fitness_function': 'fitness_function'}) 
        self.wtl_1v1.rename(columns={'metric': 'Metric', 'win': 'Win', 'tie': 'Tie', 'loss': 'Loss'},inplace=True)
        self.wtl_1v1[['config.fitness_function_1', 'config.fitness_function_2']] = self.wtl_1v1[['config.fitness_function_1', 'config.fitness_function_2']].replace({'sigmoid_rmse': 'RMSE', 'weighted_sigmoid_rmse': 'WRMSE', 'accuracy': 'Accuracy', 'f1_score': 'F1-Score'})
        self.wtl_1v1['Metric'] = self.wtl_1v1['Metric'].replace({'test.rmse': 'RMSE', 'nodes_count': 'Tree Size', 'test.accuracy': 'Accuracy', 'test.f1_score': 'F1-Score', 'test.roc_auc': 'ROC-AUC'})
        self.wtl_1v1['Result'] = self.wtl_1v1['Win'].astype(str) + '-' + self.wtl_1v1['Tie'].astype(str) + '-' + self.wtl_1v1['Loss'].astype(str)
        self.wtl_1v1['Fitness Function'] = self.wtl_1v1['config.fitness_function_1'] + ' vs ' + self.wtl_1v1['config.fitness_function_2']
        self.wtl_1v1 = self.wtl_1v1.pivot(
            index='Fitness Function',
            columns='Metric',
            values='Result'
        ).reset_index()
        
        self.wtl_detailed = pd.concat([self.wtl_detailed_max, self.wtl_detailed_min], axis=0).reset_index(drop=True)
        self.wtl_agg = pd.concat([self.wtl_agg_max, self.wtl_agg_min], axis=0).reset_index(drop=True)
        self.wtl_agg = self.wtl_agg.rename(columns={'config.fitness_function': 'fitness_function'})
        self.wtl_agg['metric'] = self.wtl_agg['metric'].replace({'test.rmse': 'RMSE', 'nodes_count': 'Tree Size', 'test.accuracy': 'Accuracy', 'test.f1_score': 'F1-Score', 'test.roc_auc': 'ROC-AUC'})
        self.wtl_agg['fitness_function'] = self.wtl_agg['fitness_function'].replace({'sigmoid_rmse': 'RMSE', 'weighted_sigmoid_rmse': 'WRMSE', 'accuracy': 'Accuracy', 'f1_score': 'F1-Score'})
        self.wtl_agg = self.wtl_agg.sort_values(['metric', 'rank'])
        self.wilcoxon_pvalues = get_wilcoxon_rank_pvalues(self.wtl_detailed, groupby='config.fitness_function')
        self.friedman_pvalues = get_friedman_rank_pvalues(self.wtl_detailed, groupby='config.fitness_function')
        self.friedman_pvalues['Metric'] = self.friedman_pvalues['Metric'].replace({'test.rmse': 'RMSE', 'nodes_count': 'Tree Size', 'test.accuracy': 'Accuracy', 'test.f1_score': 'F1-Score', 'test.roc_auc': 'ROC-AUC'})
        self.friedman_pvalues = self.friedman_pvalues.sort_values('Metric')
        self.ranks_plot = plot_ranks(self.wtl_agg, groupby='fitness_function')
        self.best_config_results_median['fitness_function'] = self.best_config_results_median['fitness_function'].replace({'sigmoid_rmse': 'RMSE', 'weighted_sigmoid_rmse': 'WRMSE', 'accuracy': 'Accuracy', 'f1_score': 'F1-Score'})
        self.performance_plot = plot_performance_barplot(self.best_config_results_median.rename(columns= {'test.accuracy': 'Accuracy', 'test.f1_score': 'F1-Score', 'test.roc_auc': 'ROC-AUC'}), ['Accuracy', 'F1-Score', 'ROC-AUC'], groupby='fitness_function')
        self.best_config_logs['fitness_function'] = self.best_config_logs['fitness_function'].replace({'sigmoid_rmse': 'RMSE', 'weighted_sigmoid_rmse': 'WRMSE', 'accuracy': 'Accuracy', 'f1_score': 'F1-Score'})
        self.tree_size_evolution_plot = plot_tree_size_evolution_by_fitness_function(self.best_config_logs)
        table_to_latex(self.friedman_pvalues, experiment_name, 'friedman', 'p-Values of the Friedman Test', index=False)
        table_to_latex(self.wtl_1v1, experiment_name, 'wtl', 'Win Tie Loss', index=False)
        plot_to_latex(self.ranks_plot[0], experiment_name, 'ranks', 'Ranks by Fitness Function') #figure, experiment, name, caption
        plot_to_latex(self.performance_plot[0], experiment_name, 'performance', 'Performance by Fitness Function') #figure, experiment, name, caption
        plot_to_latex(self.tree_size_evolution_plot[0], experiment_name, 'tree_size_evolution', 'Tree Size Evolution by Fitness Function') #figure, experiment, name, caption
        
        
class InflationrateAnalysis(Analysis):
    def __init__(self, experiment_name):
        super().__init__(experiment_name)
        self.results_median = self.results.groupby(['dataset', 'config_id', 'algorithm', 'config.ms_upper', 'config.p_inflate'])[['train.rmse', 'test.rmse', 'nodes_count']].median().reset_index()
        self.best_configs = get_min_euclidian_distance(self.results_median)
        self.best_config_results = pd.merge(self.results, self.best_configs[['dataset', 'config_id']], on=['dataset', 'config_id'], how='inner')
        self.best_config_results_median = self.best_config_results.groupby(['dataset', 'algorithm', 'config_id', 'config.ms_upper', 'config.p_inflate'])[['test.rmse', 'test.accuracy', 'test.f1_score', 'test.roc_auc', 'nodes_count']].median().reset_index()
        self.best_config_logs = pd.merge(self.logs, self.best_config_results_median[['dataset', 'config_id', 'config.ms_upper', 'config.p_inflate']], left_on=['dataset', 'config_id'], right_on=['dataset', 'config_id'], how='inner')
        self.performance_by_p_inflate_plot = plot_performance_by_p_inflate_with_ms(self.results_median)
        self.tree_size_by_p_inflate_plot = plot_tree_size_by_p_inflate_with_ms(self.results_median)
        self.performance_complexity_tradeoff_plussig1_plot = plot_performance_complexity_tradeoff(self.results_median, 'SLIM+SIG1')
        self.performance_complexity_tradeoff_mulsig1_plot = plot_performance_complexity_tradeoff(self.results_median, 'SLIM*SIG1')
        
        plot_to_latex(self.performance_by_p_inflate_plot[0], experiment_name, 'performance_by_p_inflate', 'Performance by Inflationrate') #figure, experiment, name, caption
        plot_to_latex(self.tree_size_by_p_inflate_plot[0], experiment_name, 'tree_size_by_p_inflate', 'Tree Size by Inflationrate') #figure, experiment, name, caption
        plot_to_latex(self.performance_complexity_tradeoff_plussig1_plot[0], experiment_name, 'performance_complexity_tradeoff_plussig1', 'Performance-Complexity-Tradeoff SLIM+SIG1') #figure, experiment, name, caption
        plot_to_latex(self.performance_complexity_tradeoff_mulsig1_plot[0], experiment_name, 'performance_complexity_tradeoff_mulsig1', 'Performance-Complexity-Tradeoff SLIM*SIG1') #figure, experiment, name, caption
        
        
        
class ComparisonAnalysis(Analysis):
    def __init__(self, experiment_name):
        super().__init__(experiment_name)
        ana_fitness = FitnessAnalysis('fitness')
        ana_inflate = InflationrateAnalysis('inflationrate')
        self.results = pd.concat([self.results, ana_fitness.best_config_results[ana_fitness.best_config_results['config.fitness_function'] == 'sigmoid_rmse'], ana_inflate.best_config_results], axis=0).reset_index(drop=True)
        self.results_median = self.results.groupby(['dataset', 'config_id', 'algorithm'])[['test.rmse', 'nodes_count', 'test.accuracy', 'test.f1_score', 'test.roc_auc']].median().reset_index()
        
        self.logs = pd.concat([self.logs, ana_fitness.best_config_logs[ana_fitness.best_config_logs['fitness_function'] == 'RMSE'], ana_inflate.best_config_logs], axis=0).reset_index(drop=True)
        self.wtl_1v1_max, self.wtl_detailed_max, self.wtl_agg_max = get_win_tie_loss(self.results, ['test.accuracy', 'test.f1_score', 'test.roc_auc'], minimization=False, groupby='algorithm')
        self.wtl_1v1_min, self.wtl_detailed_min, self.wtl_agg_min = get_win_tie_loss(self.results, ['test.rmse', 'nodes_count'], minimization=True, groupby='algorithm')
        self.wtl_1v1 = pd.concat([self.wtl_1v1_max.reset_index(), self.wtl_1v1_min.reset_index()], axis=0).reset_index(drop=True)

        self.wtl_1v1.rename(columns={'metric': 'Metric', 'win': 'Win', 'tie': 'Tie', 'loss': 'Loss'},inplace=True)
        self.wtl_1v1['Metric'] = self.wtl_1v1['Metric'].replace({'test.rmse': 'RMSE', 'nodes_count': 'Tree Size', 'test.accuracy': 'Accuracy', 'test.f1_score': 'F1-Score', 'test.roc_auc': 'ROC-AUC'})
        self.wtl_1v1['Result'] = self.wtl_1v1['Win'].astype(str) + '-' + self.wtl_1v1['Tie'].astype(str) + '-' + self.wtl_1v1['Loss'].astype(str)
        self.wtl_1v1['Algorithm'] = self.wtl_1v1['algorithm_1'] + ' vs ' + self.wtl_1v1['algorithm_2']
        self.wtl_1v1 = self.wtl_1v1.pivot(
            index='Algorithm',
            columns='Metric',
            values='Result'
        ).reset_index()






        self.wtl_detailed = pd.concat([self.wtl_detailed_max, self.wtl_detailed_min], axis=0).reset_index(drop=True)
        self.wtl_agg = pd.concat([self.wtl_agg_max, self.wtl_agg_min], axis=0).reset_index(drop=True)
        self.wtl_agg['metric'] = self.wtl_agg['metric'].replace({'test.rmse': 'RMSE', 'nodes_count': 'Tree Size', 'test.accuracy': 'Accuracy', 'test.f1_score': 'F1-Score', 'test.roc_auc': 'ROC-AUC'})
        self.wilcoxon_pvalues = get_wilcoxon_rank_pvalues(self.wtl_detailed, groupby='algorithm')
        self.friedman_pvalues = get_friedman_rank_pvalues(self.wtl_detailed, groupby='algorithm')
        self.friedman_pvalues['Metric'] = self.friedman_pvalues['Metric'].replace({'test.rmse': 'RMSE', 'nodes_count': 'Tree Size', 'test.accuracy': 'Accuracy', 'test.f1_score': 'F1-Score', 'test.roc_auc': 'ROC-AUC'})
        self.friedman_pvalues = self.friedman_pvalues.sort_values('Metric')
        self.tree_size_evolution_plot = plot_tree_size_evolution(self.logs)
        self.performance_evolution_plot = plot_performance_evolution(self.logs)
        self.performance_plot = plot_performance_barplot(self.results_median.rename(columns= {'test.accuracy': 'Accuracy', 'test.f1_score': 'F1-Score', 'test.roc_auc': 'ROC-AUC'}), ['Accuracy', 'F1-Score', 'ROC-AUC'], groupby='algorithm')
        self.ranks_plot = plot_ranks(self.wtl_agg.sort_values(['metric']), groupby='algorithm')
        
        
        
        plot_to_latex(self.ranks_plot[0], experiment_name, 'ranks', 'Ranks by Algorithm')
        plot_to_latex(self.performance_plot[0], experiment_name, 'performance', 'Performance by Algorithm')
        plot_to_latex(self.tree_size_evolution_plot[0], experiment_name, 'tree_size_evolution', 'Tree Size Evolution by Algorithm')
        plot_to_latex(self.performance_evolution_plot[0], experiment_name, 'performance_evolution', 'Performance Evolution by Algorithm')
        
        table_to_latex(self.friedman_pvalues, experiment_name, 'friedman', 'p-Values of the Friedman Test', index=False)        
        table_to_latex(self.wtl_1v1, experiment_name, 'wtl', 'Win Tie Loss', index=False)
        