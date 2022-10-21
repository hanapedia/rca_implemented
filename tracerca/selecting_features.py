import pickle
from collections import defaultdict
from itertools import product
from pathlib import Path

import click
import numpy as np
import seaborn as sns
import pandas as pd
from loguru import logger
from tqdm import tqdm
from pprint import pprint
from trainticket_config import FEATURE_NAMES

DEBUG = False  # very slow


def distribution_criteria(empirical, reference, threshold):
    empirical, reference = np.array(empirical), np.array(reference)
    historical_mean, historical_std = np.mean(reference), np.std(reference)
    ref_ratio = sum(np.abs(reference - historical_mean) > 3 * historical_std) / reference.shape[0]
    emp_ratio = sum(np.abs(empirical - historical_mean) > 3 * historical_std) / empirical.shape[0]
    return (emp_ratio - ref_ratio) > threshold * ref_ratio

# Not used
def fisher_criteria(empirical, reference, side='two-sided'):
    if side == 'two-sided':
        diff_mean = (np.abs(np.mean(empirical) - np.mean(reference)) ** 2)
    elif side == 'less':
        diff_mean = np.maximum(np.mean(empirical) - np.mean(reference), 0) ** 2
    elif side == 'greater':
        diff_mean = np.maximum(np.mean(reference) - np.mean(empirical), 0) ** 2
    else:
        raise RuntimeError(f'invalid side: {side}')
    variance = np.maximum(np.var(empirical) + np.var(reference), 0.1)
    return diff_mean / variance


def stderr_criteria(empirical, reference, threshold):
    empirical, reference = np.array(empirical), np.array(reference)
    historical_mean, historical_std = np.mean(reference), np.std(reference)
    historical_std = np.maximum(historical_std, historical_mean * 0.01 + 0.01)
    ref_ratio = np.mean(np.abs(reference - historical_mean)) / historical_std
    emp_ratio = np.mean(np.abs(empirical - historical_mean)) / historical_std
    return (emp_ratio - ref_ratio) > threshold * ref_ratio + 1.0


# @click.command('invocation feature selection')
# @click.option('-i', '--input', 'input_file', default="*.pkl", type=str)
# @click.option('-o', '--output', 'output_file', default='.', type=str)
# @click.option('-oc', '--output_cache', 'output_cache_file', default='.', type=str)
# @click.option('-h', '--history', default='historical_data.pkl', type=str)
# @click.option("-f", "--fisher", "fisher_threshold", default=1, type=float)
def selecting_feature_main(input_file: str, output_file: str,  output_cache_file: str, history: set, fisher_threshold):
    input_file = Path(input_file)
    output_file = Path(output_file)
    history_df = pd.DataFrame()
    for h in history:
        with open(h, 'rb') as f:
            h_pkl = pickle.load(f)
            history_df = pd.concat([history_df, h_pkl])
    # logger.debug(f'{input_file}')
    with open(str(input_file), 'rb') as f:
        df = pickle.load(f)
    df = df.set_index(keys=['source', 'target'], drop=True).sort_index()
    history = history_df.set_index(keys=['source', 'target'], drop=True).sort_index()
    indices = np.intersect1d(np.unique(df.index.values), np.unique(history.index.values))
    useful_features_dict = defaultdict(list)
    cache_dict = defaultdict(dict)
    if DEBUG:
        plot_dir = output_file.parent / 'selecting_feature.debug'
        plot_dir.mkdir(exist_ok=True)
    for (source, target), feature in tqdm(product(indices, FEATURE_NAMES)):
        empirical = np.sort(df.loc[(source, target), feature].values)
        reference = np.sort(history.loc[(source, target), feature].values)
        token = f"reference-{source}-{target}-{feature}-mean-variance"
        cache_dict[token]['mean'] = np.mean(reference)
        cache_dict[token]['std'] = np.std(reference)
        p_value = -1
        fisher = stderr_criteria(empirical, reference, fisher_threshold)
        if fisher:
            useful_features_dict[(source, target)].append(feature)
        try:
            if DEBUG:
                import matplotlib.pyplot as plt
                from matplotlib.figure import Figure
                fig = Figure(figsize=(4, 3))
                # x = np.sort(np.concatenate([empirical, reference]))
                # print('DEBUG:')
                # print(empirical,reference)
                sns.distplot(empirical, label='Empirical')
                sns.distplot(reference, label='Reference')
                plt.xlabel(feature)
                plt.ylabel('PDF')
                plt.legend()
                plt.title(f"{source}->{target}, ks={p_value:.2f}, fisher={fisher:.2f}")
                plt.savefig(
                    plot_dir / f"{input_file.name.split('.')[0]}_{source}_{target}_{feature}.pdf",
                    bbox_inches='tight', pad_inches=0
                )
        except:
            pass
    with open(output_file, 'w+') as f:
        print(dict(useful_features_dict), file=f)

    with open(output_cache_file, 'wb+') as f:
        pickle.dump(dict(cache_dict), f)


if __name__ == '__main__':
    selecting_feature_main()
