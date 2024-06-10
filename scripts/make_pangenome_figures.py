import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
import metadata_map as mm
from plot_utils import *


# Define plot colors and labels
color_dict = {'A':'tab:orange', 'Bp':'tab:blue', 'C':'tab:green'}
label_dict = {'A':r'$\alpha$', 'Bp':r'$\beta$', 'C':r'$\gamma$'}



def read_og_tables():
    f_core_labeled = '../results/single-cell/sscs_pangenome_v2/filtered_low_copy_clustered_core_mapped_orthogroup_table.tsv'
    og_table = pd.read_csv(f_core_labeled, sep='\t', index_col=0)
    core_og_table = og_table.loc[(og_table['core_A'] == 'Yes') & (og_table['core_Bp'] == 'Yes'), :]
    return og_table, core_og_table


def sort_species_clusters(og_table):
    parent_og_ids = np.sort(og_table['parent_og_id'].unique())

    sorted_idx = []
    for p in parent_og_ids:
        p_idx = og_table.loc[og_table['parent_og_id'] == p, 'num_seqs'].sort_values(ascending=False).index.values
        sorted_idx.append(p_idx)
    sorted_idx = np.concatenate(sorted_idx)

    return og_table.loc[sorted_idx, :]


def plot_species_fraction_distributions(core_og_table, args):
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('species cell fraction', fontsize=14)
    ax.set_ylabel('orthogroup clusters', fontsize=14)
    ax.set_yscale('log')
    x_bins = np.linspace(0, 1, 101)
    for s in label_dict:
        s_freq = core_og_table[f'{s}_sag_abundance'] / core_og_table['num_cells']
        ax.hist(s_freq, bins=x_bins, lw=1, histtype='step', color=color_dict[s], label=label_dict[s])
    ax.legend(fontsize=14, frameon=False)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}orthogroup_species_frequency_distribution.pdf')
    plt.close()


    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('species cell fraction', fontsize=14)
    ax.set_ylabel('orthogroup clusters', fontsize=14)
    ax.set_yscale('log')
    x_bins = np.linspace(0, 1, 101)
    for s in label_dict:
        s_freq = core_og_table.loc[core_og_table['num_cells'] > 1, f'{s}_sag_abundance'] / core_og_table.loc[core_og_table['num_cells'] > 1, 'num_cells']
        ax.hist(s_freq, bins=x_bins, lw=1, histtype='step', color=color_dict[s], label=label_dict[s])
    ax.legend(fontsize=14, frameon=False)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}orthogroup_species_frequency_distribution_nonsingleton.pdf')
    plt.close()
   

def plot_mixed_species_frequency_distribution(core_og_table, args, rng):
    og_counts = core_og_table[['parent_og_id', 'num_seqs', 'num_cells', 'A_sag_abundance', 'Bp_sag_abundance', 'C_sag_abundance']].groupby(['parent_og_id']).sum()
    og_counts['f_max'] = og_counts[['A_sag_abundance', 'Bp_sag_abundance', 'C_sag_abundance']].max(axis=1) / og_counts['num_cells']

    # Get null
    null_arr = []
    num_permutations = 1000
    f_cutoff = 0.8
    for i in range(num_permutations):
        A_null = rng.permutation(og_counts['A_sag_abundance'])
        Bp_null = rng.permutation(og_counts['Bp_sag_abundance'])
        f_null = Bp_null / (A_null + Bp_null)
        null_arr.append(f_null)
    null_arr = np.concatenate(null_arr)

    x_bins = np.linspace(0, 1, 101)
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'fraction $\beta$ sequences', fontsize=14)
    ax.set_ylabel('orthogroups', fontsize=14)
    ax.hist(og_counts['f_max'], bins=x_bins, color='tab:blue', label='data')
    ax.axvline(f_cutoff, color='tab:red', lw=2, label='mixed-species\ncutoff')
    ax.legend(fontsize=10, frameon=False)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}orthogroup_cluster_frequency_distribution.pdf')
    plt.close()

    if args.verbose:
        print('Mixed-species cluster assignment p-value: ', np.sum(null_arr > f_cutoff) / len(null_arr))


def plot_beta_distribution(core_og_table, args, rng):
    f_beta_avg = core_og_table['Bp_sag_abundance'].sum() / np.sum(core_og_table[['A_sag_abundance', 'Bp_sag_abundance', 'C_sag_abundance']].values)
    parent_ids, parent_counts = utils.sorted_unique(core_og_table['parent_og_id'].values)
    single_cluster_idx = core_og_table.index.values[core_og_table['parent_og_id'].isin(parent_ids[parent_counts == 1])]
    f_beta = core_og_table.loc[single_cluster_idx, 'Bp_sag_abundance'] / core_og_table.loc[single_cluster_idx, ['A_sag_abundance', 'Bp_sag_abundance', 'C_sag_abundance']].sum(axis=1)

    n_beta_null = rng.binomial(core_og_table.loc[single_cluster_idx, 'num_cells'], f_beta_avg)
    f_beta_null = n_beta_null / core_og_table.loc[single_cluster_idx, 'num_cells']


    x_bins = np.linspace(0.5, 1, 30)
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'fraction $\beta$ sequences', fontsize=14)
    ax.set_ylabel('orthogroups', fontsize=14)
    ax.hist(f_beta, bins=x_bins, color='tab:blue', label='data')
    ax.hist(f_beta_null, bins=x_bins, color='k', histtype='step', lw=2, label='null model')
    ax.axvline(f_beta_avg, ls='--', color='k', lw=2, label='mean $\\beta$\nfraction')
    ax.legend(fontsize=10, frameon=False)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}single_cluster_orthogroup_beta_distribution.pdf')
    plt.close()


    if args.verbose:
        print(f_beta, f_beta_avg)
        print(np.sum(parent_counts == 1), np.sum(parent_counts > 1))
        print(core_og_table.loc[single_cluster_idx, :])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--figures_dir', default='../figures/analysis/', help='Directory metagenome recruitment files.')
    parser.add_argument('-P', '--pangenome_dir', default='../results/single-cell/sscs_pangenome_v2/', help='Directory with pangenome files.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    parser.add_argument('-r', '--random_seed', type=int, default=12397, help='RNG seed.')
    args = parser.parse_args()

    og_table, core_og_table = read_og_tables()
    rng = np.random.default_rng(args.random_seed)

    metadata_map = mm.MetadataMap()
    sag_ids = np.array([c for c in og_table.columns if 'Uncmic' in c])
    species_sorted_sags = metadata_map.sort_sags(sag_ids, by='species')

    # Add species composition
    for s in ['C', 'Bp', 'A']:
        s_ids = species_sorted_sags[s]
        n_series = core_og_table[s_ids].notna().sum(axis=1)
        core_og_table.insert(7, f'{s}_sag_abundance', n_series)
    core_og_table = sort_species_clusters(core_og_table)

    if args.verbose:
        print(og_table.iloc[:, :10])
        print(core_og_table.iloc[:, :10])
        print('\n\n')

    plot_species_fraction_distributions(core_og_table, args)

    plot_mixed_species_frequency_distribution(core_og_table, args, rng)

    plot_beta_distribution(core_og_table, args, rng)



