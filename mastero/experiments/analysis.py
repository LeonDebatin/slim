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

colors_dict ={
            'GP': '#050505',
            'GSGP': '#7a7a7a',
            'SLIM+SIG1': '#307b12',
            'SLIM*SIG1': '#cd282c'
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
    results['dataset_name'] = results['dataset_name'].str.capitalize()
    return results




def get_performance_difference_significance(performance1, performance2):
    
    performance1 = subset[subset['config_id'] == config_id][metric].values
    performance2 = subset[subset['config_id'] == config_id2][metric].values
    
    if np.all(performance1 == performance2):
        p_value = 1.0
    
    else:
        p_value = stats.wilcoxon(performance1, performance2).pvalue
    return stats.wilcoxon(performance1, performance2).pvalue

def get_performance_difference_significance_table(results, differentiator, metric, pivot=False):

    
    
    
    
    significances = []
    for dataset_name in results['dataset_name'].unique():
        results_dataset = results.loc[results['dataset_name'] == dataset_name]
        
        config_results = results_dataset.loc[results_dataset['config.fitness_function'] == config, metric]
        
        for config2 in results['config.fitness_function'].unique():
            if config2 != config:
                config2_results = results_dataset.loc[results_dataset['fitness_function'] == config2, metric]
                performance1 = results.loc[dataset_name, config].values
                performance2 = results.loc[dataset_name, config2].values
                p_value = get_performance_difference_significance(config_results, config2_results)
                sign = ''
                
                if ('rmse' not in metric) and ('nodes_count' not in metric):
                    if p_value < 0.05:
                        if performance1 > performance2:
                            sign = '+'
                        else:
                            sign = '-'
                    else:
                        sign = '≈'
                
                else:
                    if p_value < 0.05:
                        if performance1 < performance2:
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


def get_slim_performance_difference_significance_table(results, metric):
    
    slimplussig1 = get_performance_difference_significance_table(results[results['name'].isin(['SLIM+SIG1', 'GP', 'GSGP']) ], config='SLIM+SIG1_', metric=metric, pivot = True)
    slimmulsig1 = get_performance_difference_significance_table(results[results['name'].isin(['SLIM*SIG1', 'GP', 'GSGP']) ], config='SLIM*SIG1_', metric = metric, pivot = True)
    
    df = pd.concat([slimplussig1, slimmulsig1], axis =1)
    return df


def sci_notation(x):
    return "{:.3e}".format(x).replace("e", " × 10^")

def get_aggregated_performance(results, metric, agg='mean', algorithm=None, fitness_function=None, ms = None):
    results = filter_results(results, algorithm, fitness_function, ms)
    
    if agg == 'mean':
        aggregated_performance = results.groupby(['dataset_name', 'config_id', 'run_id'])[metric].mean().unstack().mean(axis=1).sort_values(ascending=False).unstack()
    elif agg == 'median':
        aggregated_performance = results.groupby(['dataset_name', 'config_id', 'run_id'])[metric].median().unstack().median(axis=1).sort_values(ascending=False).unstack()
    
    return aggregated_performance

def get_rankings(results, metric, algorithm=None, fitness_function=None):
    
    results = filter_results(results, algorithm, fitness_function)
    #gets ranking for each dataset
    if any(sub in metric for sub in ('rmse', 'nodes_count')):
        return get_aggregated_performance(results, metric, 'median').rank(axis=1, ascending=True, method='min')
    else:
        return get_aggregated_performance(results, metric, 'median').rank(axis=1, ascending=False, method='max')

def get_avg_ranking(results, metric, algorithm=None, fitness_function=None):
    results = filter_results(results, algorithm, fitness_function)
    #gets averaged ranking accross datasets
    return get_rankings(results, metric).mean(axis=0).sort_values().reset_index().rename(columns={0: 'avg_rank'})

def get_ranking_significance(rankings):
    #used with get_rannkings to test for significance differences for rankings accross datasets
    return stats.friedmanchisquare(*[rankings[col] for col in rankings.columns]).pvalue


def filter_results(results, algorithm = None, fitness_function=None, ms_upper = None):
    if algorithm is not None:
        results = results.loc[results['name'] == algorithm]
    if fitness_function is not None:
        results = results.loc[results['config.fitness_function'] == fitness_function]
    if ms_upper is not None:
        results = results.loc[results['config.ms_upper'] == ms_upper]

    return results


def get_multimetric_ranking_significance(results):
    accuracy = get_ranking_significance(get_rankings(results, 'test.accuracy'))
    f1_score = get_ranking_significance(get_rankings(results, 'test.f1_score'))
    roc_auc = get_ranking_significance(get_rankings(results, 'test.roc_auc'))
    rmse = get_ranking_significance(get_rankings(results, 'test.rmse'))
    
    print(f"P-Value of the Friedman Test for ranks regarding Accuracy: {accuracy:.5f}")
    print(f"P-Value of the Friedman Test for ranks regarding F1-Score: {f1_score:.5f}")
    print(f"P-Value of the Friedman Test for ranks regarding ROC-AUC: {roc_auc:.5f}")
    print(f"P-Value of the Friedman Test for ranks regarding RMSE: {rmse:.5f}")
    return
    

def round_to_nearest_05(num, lower):
    import math
    if lower:
        return math.floor(num * 2) / 2
    else:
        return math.ceil(num * 2) / 2

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
        plt.xticks(np.arange(round_to_nearest_05(avg_rank.values.min(), True), round_to_nearest_05(avg_rank.values.max(), False)+0.5, 0.5))  # Set x-ticks at a step of 0.5
        plt.xticks(np.arange(round_to_nearest_05(avg_rank.values.min(), True), round_to_nearest_05(avg_rank.values.max(), False)+0.5, 0.25), minor=True) 
        plt.grid(which='both', axis='x',  linewidth=1)
        plt.yticks([])  # Hide y-axis ticks
        plt.xticks(fontsize=14)
    if show:
        plt.show()
    return


def plot_avg_ranking_multimetrix(results):
    roc_auc = get_avg_ranking(results, metric='test.roc_auc').rename(columns={'avg_rank': 'roc_auc'}).set_index('config_settings')
    f1_score = get_avg_ranking(results, metric='test.f1_score').rename(columns={'avg_rank': 'f1_score'}).set_index('config_settings')
    accuracy = get_avg_ranking(results, metric='test.accuracy').rename(columns={'avg_rank': 'accuracy'}).set_index('config_settings')
    rmse = get_avg_ranking(results, metric='test.rmse').rename(columns={'avg_rank': 'rmse'}).set_index('config_settings')
    rankings = pd.concat([roc_auc, f1_score, accuracy, rmse], axis=1).T
    rankings.columns = rankings.columns.str.rstrip('_')
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    sns.lineplot(rankings, dashes=False, markers='o', palette=colors_dict)
    plt.gca().invert_yaxis()
    plt.ylabel('Average Rank')
    plt.xlabel('Metric')
    ax.legend_.remove()
    plt.show()

def plot_by_p_inflate(results, metric):
    agg_performances = []
    unique_algorithms = results['name'].unique()
    for algorithm in unique_algorithms:

        agg_performance = get_aggregated_performance(results, metric=metric, agg='median', algorithm=algorithm)
        agg_performance.columns = [re.search(r'inflate([\d.]+)_', col).group(1) for col in agg_performance.columns]
        agg_performances.append(agg_performance)
    df_agg = pd.concat(agg_performances, keys=unique_algorithms).T
    print(df_agg)
    
    unique_datasets = results['dataset_name'].unique()
    fig, ax = plt.subplots(int(len(unique_datasets)/2), 2, figsize=(12, 15), squeeze=False)

    j = 0
    i = 0
    for dataset in unique_datasets:
        sns.lineplot(data=df_agg.loc[:,df_agg.columns.get_level_values(1) == dataset], dashes=False, marker = 'o', palette=colors_dict,  ax=ax[i, j])
        ax[i,j].set_title(f'{dataset}')
        ax[i,j].set_xlabel('p_inflate')
        ax[i,j].set_ylabel(metric)
        ax[i,j].legend_.remove()
        ax[i,j].yaxis.set_major_locator(ticker.LinearLocator(5))
        if metric == 'nodes_count':
            ax[i,j].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
        else:
            ax[i,j].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.3f}")) 
        #ax[i,j].legend_.remove()
        j = (j + 1) % 2
        i = i +1 if j == 0 else i
    
    #fig.legend(unique_algorithms)
    fig.tight_layout()
    plt.show()    


def plot_by_p_inflate2(results, metric):
    agg_performances = []
    unique_algorithms = results['name'].unique()
    unique_ms = results['config.ms_upper'].unique()
    for algorithm in unique_algorithms:
        for ms in unique_ms:
            agg_performance = get_aggregated_performance(results, metric=metric, agg='median', algorithm=algorithm, ms=ms)
            agg_performance.columns = [re.search(r'inflate([\d.]+)_', col).group(1) for col in agg_performance.columns]
            print(ms, algorithm)
            print(agg_performance)
            agg_performances.append(agg_performance)
    df_agg = pd.concat(agg_performances, keys=unique_algorithms).T
    print(df_agg)
    
    unique_datasets = results['dataset_name'].unique()
    fig, ax = plt.subplots(int(len(unique_datasets)/2), 2, figsize=(12, 15), squeeze=False)

    j = 0
    i = 0
    for dataset in unique_datasets:
        for ms in unique_ms:
            sns.lineplot(data=df_agg.loc[:,df_agg.columns.get_level_values(1) == dataset], dashes=False, marker = 'o', palette=colors_dict,  ax=ax[i, j])
        sns.lineplot(data=df_agg.loc[:,df_agg.columns.get_level_values(1) == dataset], dashes=False, marker = 'o', palette=colors_dict,  ax=ax[i, j])
        ax[i,j].set_title(f'{dataset}')
        ax[i,j].set_xlabel('p_inflate')
        ax[i,j].set_ylabel(metric)
        ax[i,j].legend_.remove()
        ax[i,j].yaxis.set_major_locator(ticker.LinearLocator(5))
        if metric == 'nodes_count':
            ax[i,j].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
        else:
            ax[i,j].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.3f}")) 
        #ax[i,j].legend_.remove()
        j = (j + 1) % 2
        i = i +1 if j == 0 else i
    
    #fig.legend(unique_algorithms)
    fig.tight_layout()
    plt.show()    




def error_evolution_plot(logs):
    unique_datasets = logs['dataset'].unique()

    # Create subplots
    fig, ax = plt.subplots(len(unique_datasets), 2, figsize=(10, 15), squeeze=False)

    for i, dataset in enumerate(unique_datasets):
        dataset_logs = logs[logs['dataset'] == dataset]

        # Call the function without assignment
        plot_value_by_generations(dataset_logs, 'elite_train_error', ax=ax[i, 0])
        plot_value_by_generations(dataset_logs, 'elite_test_error', ax=ax[i, 1])

    fig.tight_layout()
    plt.show()
    return fig



def tree_size_evolution_plot(logs):
    unique_datasets = logs['dataset'].unique()
    fig, ax = plt.subplots(int(len(unique_datasets)/2), 2, figsize=(10, 15), squeeze=False)

    j = 0
    i = 0
    for dataset in unique_datasets:
        dataset_logs = logs[logs['dataset'] == dataset]
        plot_value_by_generations(dataset_logs, 'elite_nodes', ax=ax[i, j], y_max=2000)
        j = (j + 1) % 2
        i = i +1 if j == 0 else i

    fig.tight_layout()
    plt.show()



def get_anova_table(results, metric):
    anova = []
    for dataset in results['dataset_name'].unique():
        results_dataset = results.loc[results['dataset_name'] == dataset]
        for name in results['name'].unique():
            results_name = results_dataset.loc[results_dataset['name'] == name]
            p_value = stats.f_oneway(*[results_name.loc[results_name['config.p_inflate'] == p_inflate][metric] for p_inflate in results_name['config.p_inflate'].unique()]).pvalue
            anova.append([dataset, name, p_value])
    
    return pd.DataFrame(anova, columns=['dataset', 'name', 'p_value']).pivot_table(index='dataset', columns='name', values='p_value')

def table_to_latex(table, experiment, name, caption):
        
    latex_table = table.to_latex(column_format="l" + "c" * len(table.columns), escape=False)
    latex_code = f"""
    \\begin{{table}}[h]
        \\centering
        \\renewcommand{{\\arraystretch}}{{1.2}}
        {latex_table}
        \\caption{{{caption}}}
        \\label{{tab:{name}}}
    \\end{{table}}
    """
    with open(f"{experiment}_latex/tables/{name}.tex", "w") as f:
        f.write(latex_code)
    return
    
def plot_countplot(results, metric):
    results.groupby(['dataset_name', 'name', 'config.p_inflate']).agg({metric: 'mean'}).reset_index()
    results.groupby(['dataset_name', 'name']).agg({metric: 'min'}).reset_index()
    results_min = results.loc[results.groupby(['dataset_name', 'name'])[metric].idxmin(), ['dataset_name', 'name', metric, 'config.p_inflate']]
    results_min = results_min.reset_index(drop=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    sns.countplot(results_min, x='config.p_inflate', hue='name', palette=colors_dict, width=0.5, hue_order=hue_order_slim)
    ax.set_xlabel('p_inflate')
    ax.set_ylabel('Best Performance Count')
    ax.legend_.remove()
    plt.show()
    return

def plot_to_latex(figure, experiment, name, caption):
    figure.savefig(f"{experiment}_latex/figures/{name}.png", dpi=500, bbox_inches='tight', transparent=True)
    
    latex_code = f"""
    \\begin{{figure}}[h]
    \\centering
    \\includegraphics[width=\\linewidth]{{figures/{name}.png}}
    \\caption{{{caption}}}
    \\label{{fig:{name}}}
    \\end{{figure}}
    """
    with open(f"{experiment}_latex/figures/{name}.tex", "w") as f:
        f.write(latex_code)

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
    logs['dataset'] = logs['dataset'].str.capitalize()
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






def plot_value_by_generations(logs, value, ax, y_max=None):
    """Plots value over generations for a specific dataset on a given subplot axis."""
    # Group and calculate median
    grp = logs.groupby(['config_id', 'algorithm', 'dataset', 'generation'])[[value]].median().reset_index()

    # Plot using seaborn on the given subplot axis
    sns.lineplot(data=grp, x='generation', y=value, hue='algorithm', ax=ax, palette=colors_dict)

    # Set labels and title
    ax.set_title(f'{logs["dataset"].iloc[0]}')
    ax.set_xlabel('Generation')
    ax.set_ylabel(value)
    ax.legend_.remove()
    ax.yaxis.set_major_locator(ticker.LinearLocator(5))
    if value == 'elite_nodes':
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    else:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.3f}")) 
    #ax.set_xlim(0, 2000)
    if y_max:
        ax.set_ylim(0, y_max)
        
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
#         ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.3f}")) 
#     if y_max:
#         ax.set_ylim(0, y_max)




def get_min_euclidian_distance(results):
    unique_datasets = results['dataset_name'].unique()
    unique_models = results['name'].unique()
    
    best_configs = []
    
    for dataset in unique_datasets:
        for model in unique_models:
            subset = results[(results['dataset_name'] == dataset) & (results['name'] == model)]
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(subset[['test.rmse', 'nodes_count']])
            subset.loc[:, ['test.rmse', 'nodes_count']] = scaled_values
            subset['euclidian_distance'] = (subset['test.rmse']**2 + 2*subset['nodes_count']**2)**0.5

            subset = subset.sort_values('euclidian_distance')
            subset = subset.drop_duplicates(subset=['dataset_name', 'name'], keep='first')
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
    
    for dataset in results['dataset_name'].unique():
        filtered_by_dataset = results.loc[results['dataset_name'] == dataset]
        
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

    best_configs_df = pd.DataFrame(best_configs, columns=['dataset_name', 'fitness_function', 'config_id'])
    return best_configs_df




def plot_performance_barplot(results_median, metrics, groupby, palette):

    # Melt the DataFrame to long format
    df_long = results_median.melt(
        id_vars=['dataset_name', groupby],
        value_vars= metrics,
        var_name='metric',
        value_name='value'
    )

    unique_datasets = df_long['dataset_name'].unique()
    n_cols = 2
    n_rows = math.ceil(len(unique_datasets) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 2 * n_rows), squeeze=False)

    # Plot each dataset
    for idx, dataset in enumerate(unique_datasets):
        i, j = divmod(idx, n_cols)
        ax = axes[i, j]
        
        subset = df_long[df_long['dataset_name'] == dataset]
        
        sns.barplot(
            data=subset,
            x='metric',
            y='value',
            hue= groupby,
            palette=palette,
            ax=ax
        )
        
        ax.set_title(dataset)
        ax.set_xlabel("Evaluation Metric")
        ax.set_ylabel("Test Score")
        ax.tick_params(axis='x')
        ax.legend_.remove()
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
    for k in range(len(unique_datasets), n_rows * n_cols):
        i, j = divmod(k, n_cols)
        fig.delaxes(axes[i, j])

    plt.tight_layout()
    plt.show()


def plot_performance_evolution_by_fitness_function(logs):
    unique_datasets = logs['dataset'].unique()
    fig, ax = plt.subplots(len(unique_datasets), 2, figsize=(8, 10), squeeze=False)

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
        
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend:
            legend.remove()

    fig.tight_layout() 

    plt.show()


def plot_tree_size_evolution_by_fitness_function(logs):
    unique_datasets = logs['dataset'].unique()
    fig, ax = plt.subplots(int(len(unique_datasets)/2), 2, figsize=(10, 2 * int(len(unique_datasets)/2)), squeeze=False)

    i=0
    j=0
    for dataset in unique_datasets:
        subset = logs[logs['dataset'] == dataset]
        grouped = subset.groupby(['config_id', 'generation', 'fitness_function'])[['elite_nodes']].median().reset_index()
        sns.lineplot(
            data=grouped,
            x='generation',
            y='elite_nodes',
            hue='fitness_function',
            ax = ax[i, j],
        )
        ax[i,j].set_title(dataset)
        ax[i,j].set_xlabel("Generation")
        ax[i,j].set_ylabel("Tree Size")
        ax[i,j].set_yticks(np.arange(0, 6000, 1000))
        ax[i,j].set_ylim(0, 5000)
        
        j = (j + 1) % 2
        i = i +1 if j == 0 else i
        
        
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend:
            legend.remove()

    fig.tight_layout() 

    plt.show()

def plot_performance_by_p_inflate_with_ms(results,  colors_dict=None):
    unique_datasets = results['dataset_name'].unique()
    fig, ax = plt.subplots(len(unique_datasets), 2,  figsize=(10, 3 * int(len(unique_datasets)/2)), squeeze=False)
    
    for i, dataset in enumerate(unique_datasets):

        subset = results[results['dataset_name'] == dataset]

        sns.lineplot(
            data=subset,
            x='config.p_inflate',
            y='train.rmse',
            hue='config.ms_upper',
            style='name',
            markers=True,
            dashes=True,
            palette = 'Set1',
            ax=ax[i, 0]
        )
        ax[i, 0].set_title(f'{dataset}')
        ax[i, 0].set_xlabel('config.p_inflate')
        ax[i, 0].set_ylabel('Train Score')
        ax[i, 0].yaxis.set_major_locator(ticker.LinearLocator(3))
        ax[i, 0].legend_.remove()
        
        sns.lineplot(
            data=subset,
            x='config.p_inflate',
            y='test.rmse',
            hue='config.ms_upper',
            style='name',
            markers=True,
            dashes=True,
            palette = 'Set1',
            ax=ax[i, 1]
        )
        ax[i, 1].set_title(f'{dataset}')
        ax[i, 1].set_xlabel('Inflationrate')
        ax[i, 1].set_ylabel('Test Score')
        ax[i, 1].yaxis.set_major_locator(ticker.LinearLocator(3))
        ax[i, 1].legend_.remove()

    fig.tight_layout()
    plt.show()

def plot_tree_size_by_p_inflate_with_ms(results,  colors_dict=None):
    
    unique_datasets = results['dataset_name'].unique()
    fig, ax = plt.subplots(int(len(unique_datasets)/2), 2, figsize=(10, 1.5 * int(len(unique_datasets)/2)), squeeze=False)
    
    i=j=0
    for dataset in unique_datasets:
        subset = results[results['dataset_name'] == dataset]
        sns.lineplot(
            data=subset,
            x='config.p_inflate',
            y='nodes_count',
            hue='config.ms_upper',
            style='name',
            markers=True,
            dashes=True,
            palette = 'Set1',
            ax=ax[i, j]
        )
        ax[i, j].set_title(f'{dataset}')
        ax[i, j].set_xlabel('Inflationrate')
        ax[i, j].set_ylabel('Tree Size')
        ax[i, j].yaxis.set_major_locator(ticker.LinearLocator(3))
        ax[i, j].legend_.remove()
        
        j = (j + 1) % 2
        i = i +1 if j == 0 else i
        
        
    # for ax in fig.axes:
    #     legend = ax.get_legend()
    #     if legend:
    #         legend.remove()

    fig.tight_layout()
    plt.show()

def plot_performance_complexity_tradeoff(results, model):
    unique_datasets = results['dataset_name'].unique()

    fig, axes = plt.subplots(int(len(unique_datasets)/2), 2, figsize=(10, 3 * int(len(unique_datasets)/2)))
    axes = axes.flatten()
    for i, dataset in enumerate(unique_datasets):
        subset = results[results['dataset_name'] == dataset]
        subset = subset[subset['name'] == model]
        ax = axes[i]
        sns.scatterplot(
            data=subset,
            x='nodes_count',
            y='test.rmse',
            hue='config.p_inflate',
            #size= 'config.p_inflate',
            size= 'config.ms_upper',
            palette = 'Set1',
            ax=ax
        )
        #ax.axvline(x=1000, color='black', linestyle='--')
        ax.set_title(f"{dataset}")
        ax.set_xlabel("Tree Size")
        ax.set_ylabel("Test Score")
        ax.legend(loc='best', fontsize='small', title='Config', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    # Hide any unused subplots
    # for j in range(i + 1, len(axes)):
    #     fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_ranks(wtl_agg, groupby):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=wtl_agg, x='metric', y='rank', hue=groupby, marker='o', ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel('Metric')
    ax.set_ylabel('Rank')
    ax.set_title('Rankings of Configurations')
    plt.show()



def get_friedman_rank_pvalues(wtl_detailed, groupby):
    p_values = []
    for metric in wtl_detailed['metric'].unique():
        performances = []
        for group in wtl_detailed[groupby].unique():
            performances.append(wtl_detailed[(wtl_detailed[groupby] == group) & (wtl_detailed['metric'] == metric)]['rank'].values)

        p_value = stats.friedmanchisquare(*performances).pvalue
        
        if p_value < 0.05:
            sig = 'True'
        else:
            sig = 'False'
        
        p_values.append([metric, p_value, sig])
    
    return pd.DataFrame(p_values, columns=['metric', 'p_value', 'significant'])


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
    unique_datasets = results['dataset_name'].unique()
    unique_groupby = results[groupby].unique()
    
    for dataset in unique_datasets:
        subset = results[results['dataset_name'] == dataset]
        
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
                            
    wtl_1v1_df = pd.DataFrame(wtl_1v1, columns=['dataset_name', 'config_id1', 'config_id2', 'metric', 'p_value', 'win', 'tie', 'loss'])
    wtl_1v1_df = pd.merge(wtl_1v1_df, results[['dataset_name', 'config_id', groupby]].drop_duplicates().reset_index(drop=True),
                  left_on=['dataset_name', 'config_id1'], right_on=['dataset_name', 'config_id'], how='left')
    wtl_1v1_df.rename(columns={groupby: f'{groupby}_1'}, inplace=True)
    wtl_1v1_df = pd.merge(wtl_1v1_df, results[['dataset_name', 'config_id', groupby]].drop_duplicates().reset_index(drop=True),
                    left_on=['dataset_name', 'config_id2'], right_on=['dataset_name', 'config_id'], how='left')
    wtl_1v1_df.rename(columns={groupby: f'{groupby}_2'}, inplace=True)
    wtl_1v1_df.drop(columns=['config_id1', 'config_id2', 'config_id_x', 'config_id_y'], inplace=True)
    
    wtl_comparison_1v1 = wtl_1v1_df.groupby(['metric',f'{groupby}_1', f'{groupby}_2'])[['win', 'tie', 'loss']].sum()
    
    
    
    c1 = wtl_1v1_df.groupby(['dataset_name', f'{groupby}_1', 'metric'])[['win', 'tie', 'loss']].sum().reset_index()
    c2 = wtl_1v1_df.groupby(['dataset_name', f'{groupby}_2', 'metric'])[['win', 'tie', 'loss']].sum().reset_index().rename(columns={f'{groupby}_2': f'{groupby}_1', 'win': 'loss', 'loss':'win'})
    wtl_detailed = pd.concat([c1, c2], axis=0).groupby(['dataset_name',f'{groupby}_1', 'metric'])[['win', 'tie', 'loss']].sum().reset_index()
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
        self.best_config_results = pd.merge(self.results, self.best_configs, on=['dataset_name', 'config_id'], how='inner')
        self.best_config_results_median = self.best_config_results.groupby(['dataset_name', 'config_id', 'fitness_function'])[['test.rmse', 'test.accuracy', 'test.f1_score', 'test.roc_auc']].median().reset_index()
        self.best_config_logs = pd.merge(self.logs, self.best_config_results_median[['dataset_name', 'config_id', 'fitness_function']], left_on=['dataset', 'config_id'], right_on=['dataset_name', 'config_id'], how='inner')
        self.wtl_1v1, self.wtl_detailed, self.wtl_agg = get_win_tie_loss(self.best_config_results, ['test.accuracy', 'test.f1_score', 'test.roc_auc'], minimization=False, groupby='config.fitness_function')
        self.wilcoxon_pvalues = get_wilcoxon_rank_pvalues(self.wtl_detailed, groupby='config.fitness_function')
        self.friedman_pvalues = get_friedman_rank_pvalues(self.wtl_detailed, groupby='config.fitness_function')
class InflationrateAnalysis(Analysis):
    def __init__(self, experiment_name):
        super().__init__(experiment_name)
        self.results_median = self.results.groupby(['dataset_name', 'config_id', 'name', 'config.ms_upper', 'config.p_inflate'])[['train.rmse', 'test.rmse', 'nodes_count']].median().reset_index()
        self.best_configs = get_min_euclidian_distance(self.results_median)
        self.best_config_results = pd.merge(self.results, self.best_configs[['dataset_name', 'config_id']], on=['dataset_name', 'config_id'], how='inner')
        self.best_config_results_median = self.best_config_results.groupby(['dataset_name', 'name', 'config_id', 'config.ms_upper', 'config.p_inflate'])[['test.rmse', 'test.accuracy', 'test.f1_score', 'test.roc_auc', 'nodes_count']].median().reset_index()
        self.best_config_logs = pd.merge(self.logs, self.best_config_results_median[['dataset_name', 'config_id', 'config.ms_upper', 'config.p_inflate']], left_on=['dataset', 'config_id'], right_on=['dataset_name', 'config_id'], how='inner')
class ComparisonAnalysis(Analysis):
    def __init__(self, experiment_name):
        super().__init__(experiment_name)
        ana_fitness = FitnessAnalysis('fitness')
        ana_inflate = InflationrateAnalysis('inflationrate')
        self.results = pd.concat([self.results, ana_fitness.best_config_results[ana_fitness.best_config_results['config.fitness_function'] == 'sigmoid_rmse'], ana_inflate.best_config_results], axis=0).reset_index(drop=True)
        self.results_median = self.results.groupby(['dataset_name', 'config_id', 'name'])[['test.rmse', 'nodes_count', 'test.accuracy', 'test.f1_score', 'test.roc_auc']].median().reset_index()
        self.logs = pd.concat([self.logs, ana_fitness.best_config_logs[ana_fitness.best_config_logs['fitness_function'] == 'sigmoid_rmse'], ana_inflate.best_config_logs], axis=0).reset_index(drop=True)
        self.wtl_1v1_max, self.wtl_detailed_max, self.wtl_agg_max = get_win_tie_loss(self.results, ['test.accuracy', 'test.f1_score', 'test.roc_auc'], minimization=False, groupby='name')
        self.wtl_1v1_min, self.wtl_detailed_min, self.wtl_agg_min = get_win_tie_loss(self.results, ['test.rmse', 'nodes_count'], minimization=True, groupby='name')
        self.wtl_1v1 = pd.concat([self.wtl_1v1_max.reset_index(), self.wtl_1v1_min.reset_index()], axis=0).reset_index(drop=True)
        self.wtl_detailed = pd.concat([self.wtl_detailed_max, self.wtl_detailed_min], axis=0).reset_index(drop=True)
        self.wtl_agg = pd.concat([self.wtl_agg_max, self.wtl_agg_min], axis=0).reset_index(drop=True)
        self.wilcoxon_pvalues = get_wilcoxon_rank_pvalues(self.wtl_detailed, groupby='name')
        self.friedman_pvalues = get_friedman_rank_pvalues(self.wtl_detailed, groupby='name')