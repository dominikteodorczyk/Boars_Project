import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from pyemd import emd
from tqdm import tqdm

from geo_processor import GeoProcessor


def _compute_flows_individual(trajectory, option, without_self_transitions, container):
    traj = trajectory.copy()
    if without_self_transitions:
        thresh = 2
        traj_temp = traj["tessellation_id"].diff().ne(0).cumsum()
        small_size = traj_temp.groupby(traj_temp).transform('size') < thresh
        first_rows = ~traj_temp.duplicated()
        traj = traj[small_size | first_rows]
    cuted = traj.iloc[1:-1]
    first = traj.iloc[0]["tessellation_id"]
    last = traj.iloc[-1]["tessellation_id"]
    single_flows = cuted.groupby("tessellation_id").count().iloc[:, 0]
    if option == 'all':
        flows = single_flows * 2
        if last in flows.index:
            flows.loc[last] += 1
        elif len(traj) > 1:
            flows[last] = 1
        if first in flows.index:
            flows.loc[first] += 1
        elif len(traj) > 1:
            flows[first] = 1
    elif option == 'incoming':
        if last in single_flows.index:
            single_flows.loc[last] += 1
        elif len(traj) > 1:
            single_flows[last] = 1
        flows = single_flows
    elif option == 'outgoing':
        if first in single_flows.index:
            single_flows.loc[first] += 1
        elif len(traj) > 1:
            single_flows[first] = 1
        flows = single_flows
    container.append(flows)


def compute_flows(trajectory, option, without_self_transitions=True):
    results = []
    trajectory.groupby(level=0).progress_apply(
        lambda x: _compute_flows_individual(x, option, without_self_transitions, results))
    results = pd.concat(results).groupby("tessellation_id").sum()
    results = results.rename('flows')
    return results


def plot_flows(flows_org, flows_syn, box_number, plot_path, file_name, option='logscale'):
    flows_org = pd.Series(flows_org, name="flows").to_frame()
    flows_syn = pd.Series(flows_syn, name="flows").to_frame()

    flows_org.index = flows_org.index.astype(int)
    flows_syn.index = flows_syn.index.astype(int)

    temp = pd.concat([flows_org.add_prefix('org_'), flows_syn.add_prefix('syn_')], axis=1).fillna(0)

    # SP = scipy.stats.spearmanr(temp['org_flows'], temp['syn_flows'])
    pearsonr_corr = scipy.stats.pearsonr(temp['org_flows'], temp['syn_flows'])
    max_value = temp.max().max()
    e = np.log10(max_value).round()

    if option == 'logscale':
        extent = np.logspace(0, e, num=box_number, base=10)
    elif option == 'linescale':
        extent = np.linspace(0, max_value, num=box_number)

    range = (extent[1:] - extent[:-1]) / 2
    means_for_bins = {}
    box_for_bins = {}

    for n in np.arange(len(extent) - 1):
        start = extent[n] - range[n - 1] if n > 0 else extent[n] - range[n]
        end = extent[n] + range[n] if n < len(extent) - 1 else extent[n] + range[n - 1]

        aggregated = temp[(temp['syn_flows'] >= start) & (temp['syn_flows'] < end)]
        means_for_bins[aggregated['syn_flows'].mean()] = aggregated['org_flows'].mean()
        box_for_bins[aggregated['syn_flows'].mean()] = aggregated['org_flows']

        # Box plot categorization
    boxes_black = {k: v for k, v in box_for_bins.items() if v.quantile(.1) <= k <= v.quantile(.9)}
    boxes_red = {k: v for k, v in box_for_bins.items() if k < v.quantile(.1) or k > v.quantile(.9)}

    # Plot
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.scatter(temp['syn_flows'], temp['org_flows'], c='lightgrey', s=8, alpha=.7)
    # spe = SP[0] * 100
    spe = pearsonr_corr[0]
    # te = 'SpearmanR: {spe:.2f} %'
    te = f'Pearson: {spe:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, te, transform=ax.transAxes, fontsize=16, fontweight='bold', verticalalignment='top', bbox=props)

    # Boxplots
    for color, boxes in [('k', boxes_black), ('r', boxes_red)]:
        if boxes:
            widths = [x / 4 * .5 for x in boxes.keys()] if option == 'logscale' else 7
            plt.boxplot([x.dropna() for x in boxes.values()], positions=[round(p, 2) for p in boxes.keys()],
                        sym='', whis=[9, 91], boxprops={'color': color, 'linewidth': 1},
                        widths=widths, medianprops={'linewidth': 1, 'color': color},
                        whiskerprops={'color': color}, capprops={'color': color})

    plt.scatter(means_for_bins.keys(), means_for_bins.values(), s=100)
    plt.plot(np.arange((temp.values.reshape(-1, 1)).max()), c='k')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(1)
    plt.ylim(1)

    if option == 'logscale':
        ax.set_xscale('log', base=10)
        ax.set_yscale('log', base=10)

    plt.xlabel("Travel (model)", fontsize=18)
    plt.ylabel("Travel (data)", fontsize=18)
    plt.grid(False)
    plt.savefig(os.path.join(plot_path, file_name + '.png'), dpi=300)


def normalize(data, column=None, output_column=None):
    if isinstance(data, pd.Series):
        probability = data / data.sum()
        offset = 1.0 - probability.sum()
        if offset != 1:
            probability[probability.idxmax()] += offset
        data = probability
    if isinstance(data, pd.DataFrame):
        probability = data[column] / data[column].sum()
        offset = 1.0 - probability.sum()
        if offset != 1:
            probability[probability.idxmax()] += offset
        data[output_column] = probability
    return data


def compute_emd(org_traj, syn_traj, tessellation):
    org_traj = org_traj.copy()
    syn_traj = syn_traj.copy()

    output = []

    org_traj.index.names = ['animal_id', 'time']
    syn_traj.index.names = ['animal_id', 'time']

    org_traj.reset_index(inplace=True)
    syn_traj.reset_index(inplace=True)

    org_traj['tessellation_id'] = org_traj['tessellation_id'].astype(int)
    syn_traj['tessellation_id'] = syn_traj['tessellation_id'].astype(int)

    org_traj['slot'] = org_traj['time'].dt.hour
    syn_traj['slot'] = syn_traj['time'].dt.hour

    gp = GeoProcessor()
    distance_matrix = gp.compute_dist_matrix(tessellation)

    for slot in tqdm(range(24), total=24):
        temp_org = org_traj[org_traj['slot'] == slot]
        temp_syn = syn_traj[syn_traj['slot'] == slot]

        org = temp_org.groupby('tessellation_id').size().reset_index(name='count')
        syn = temp_syn.groupby('tessellation_id').size().reset_index(name='count')

        unique_values = list(set(org['tessellation_id']).union(set(syn['tessellation_id'])))

        org = tessellation.merge(org, on='tessellation_id', how='left').fillna(0)
        syn = tessellation.merge(syn, on='tessellation_id', how='left').fillna(0)

        org = org[org['tessellation_id'].isin(unique_values)]
        syn = syn[syn['tessellation_id'].isin(unique_values)]

        norm_org = normalize(org, 'count', 'POP_PROB').sort_values(by='tessellation_id')
        norm_syn = normalize(syn, 'count', 'POP_PROB').sort_values(by='tessellation_id')

        selected_cells = distance_matrix.loc[unique_values, unique_values].sort_index().sort_index(axis=1)

        norm_org_values = np.ascontiguousarray(norm_org['POP_PROB'].values)
        norm_syn_values = np.ascontiguousarray(norm_syn['POP_PROB'].values)
        selected_cells_values = np.ascontiguousarray(selected_cells.values)

        score = emd(norm_org_values, norm_syn_values, selected_cells_values)

        output.append([slot, score])

    return pd.DataFrame(output, columns=['slot', 'emd'])


def plot_emd(data_list, label_list, output_dir, file_name):
    colors = {
        "EPR": "blue",
        "RandomWalk": "red",
        "LevyFlight": "green"
    }
    plt.figure(figsize=(15, 15))
    for data, label in zip(data_list, label_list):
        plt.plot(data['slot'], data['emd'], label=label, color=colors[label])
        plt.xlim([-1, 24])
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xticks(np.arange(0, 24, step=1))
        plt.yticks(fontsize=12)
        plt.xlabel("hour", fontsize=18)
        plt.ylabel("error [m]", fontsize=18)
        plt.grid()
        plt.savefig(os.path.join(output_dir, file_name + '_.png'), dpi=300)
