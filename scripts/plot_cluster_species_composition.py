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

def sort_species_clusters(og_table):
    parent_og_ids = np.sort(og_table['parent_og_id'].unique())

    sorted_idx = []
    for p in parent_og_ids:
        p_idx = og_table.loc[og_table['parent_og_id'] == p, 'num_seqs'].sort_values(ascending=False).index.values
        sorted_idx.append(p_idx)
    sorted_idx = np.concatenate(sorted_idx)

    return og_table.loc[sorted_idx, :]
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--figures_dir', default='../figures/analysis/', help='Directory metagenome recruitment files.')
    parser.add_argument('-P', '--pangenome_dir', default='../results/single-cell/sscs_pangenome_v2/', help='Directory with pangenome files.')
    parser.add_argument('-g', '--orthogroup_table', required=True, help='File with orthogroup table.')
    parser.add_argument('-o', '--output_file', required=True, help='File with output orthogroup table.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    args = parser.parse_args()

    metadata_map = mm.MetadataMap()
    og_table = pd.read_csv(args.orthogroup_table, sep='\t', index_col=0)
    core_og_table = og_table.loc[(og_table['core_A'] == 'Yes') & (og_table['core_Bp'] == 'Yes'), :]
    sag_ids = np.array([c for c in og_table.columns if 'Uncmic' in c])
    species_sorted_sags = metadata_map.sort_sags(sag_ids, by='species')

    for s in ['C', 'Bp', 'A']:
        s_ids = species_sorted_sags[s]
        n_series = core_og_table[s_ids].notna().sum(axis=1)
        core_og_table.insert(7, f'{s}_sag_abundance', n_series)


    core_og_table = sort_species_clusters(core_og_table)

    print(core_og_table.iloc[:, :10])
    parent_og_ids = core_og_table['parent_og_id'].unique()

    print(core_og_table[['parent_og_id', 'num_seqs', 'A_sag_abundance', 'Bp_sag_abundance', 'C_sag_abundance']].groupby(['parent_og_id']).sum())

    color_dict = {'A':'tab:orange', 'Bp':'tab:blue', 'C':'tab:green'}
    label_dict = {'A':r'$\alpha$', 'Bp':r'$\beta$', 'C':r'$\gamma$'}

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('cell fraction', fontsize=14)
    ax.set_ylabel('orthogroups', fontsize=14)
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

    print(x_bins)
