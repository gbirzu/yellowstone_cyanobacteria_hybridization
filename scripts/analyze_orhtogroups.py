import argparse
import re
import numpy as np
import pandas as pd
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
from pangenome_utils import PangenomeMap
from metadata_map import MetadataMap
from plot_utils import *


def plot_seq_cluster_orthogroups(og_table, args):
    seq_clusters = og_table['seq_cluster'].values
    c_ids, c_counts = utils.sorted_unique(seq_clusters)

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(f'protein cluster rank', fontsize=14)
    ax.set_xscale('log')
    ax.set_ylabel(f'number of orthogroups', fontsize=14)
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax.plot(np.arange(1, len(c_counts) + 1), c_counts, '-o', mfc='none', mec='tab:blue', ms=3)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}sequence_cluster_orthogroup_distribution.pdf')
    plt.close()


def plot_seqs_per_cell(og_table, args):
    r = np.arange(1, og_table.shape[0] + 1)

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(f'orthogroup rank', fontsize=14)
    ax.set_xscale('log')
    ax.set_ylabel(f'seqs per cell', fontsize=14)
    ax.plot(r, og_table['seqs_per_cell'].sort_values(ascending=False), '-o', mfc='none', mec='tab:blue', ms=3)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}og_mean_seqs_per_cell.pdf')
    plt.close()

    seq_clusters = og_table['seq_cluster'].values
    c_ids, c_counts = utils.sorted_unique(seq_clusters)
    single_og_idx = og_table.index.values[og_table['seq_cluster'].isin(c_ids[c_counts <= 1])]
    multi_og_idx = og_table.index.values[og_table['seq_cluster'].isin(c_ids[c_counts > 1])]

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(f'mean copy number', fontsize=14)
    ax.set_ylabel(f'density', fontsize=14)
    ax.set_yscale('log')
    ax.hist(og_table.loc[single_og_idx, 'seqs_per_cell'], bins=50, density=True, alpha=0.5, label='single-OG cluster')
    ax.hist(og_table.loc[multi_og_idx, 'seqs_per_cell'], bins=50, density=True, alpha=0.5, label='multi-OG cluster')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}og_seqs_per_cell_by_cluster_type.pdf')
    plt.close()


def plot_og_rank_prevalence(og_table, args):
    r = np.arange(1, og_table.shape[0] + 1)

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(f'orthogroup rank', fontsize=14)
    ax.set_xscale('log')
    ax.set_ylabel(f'orthogroup prevalence', fontsize=14)
    ax.plot(r, og_table['num_cells'].sort_values(ascending=False), '-o', mfc='none', mec='tab:blue', ms=3)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}og_rank_prevalence.pdf')
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--figures_dir', default='../figures/analysis/', help='Directory metagenome recruitment files.')
    parser.add_argument('-O', '--output_dir', default='../results/tests/', help='Directory in which results are saved.')
    parser.add_argument('-g', '--orthogroup_table', required=True, help='File with orthogroup table.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    args = parser.parse_args()

    pangenome_map = PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    og_table = pangenome_map.og_table
    metadata = MetadataMap()

    # Process OG table
    meta_cols = list(og_table.columns.values[:4]) + ['seq_cluster']
    sag_cols = list(og_table.columns.values[4:])
    og_table['seq_cluster'] = [re.sub(r'[a-z]*', '', s) for s in og_table.index.values]
    og_table = og_table

    plot_seq_cluster_orthogroups(og_table, args)
    plot_seqs_per_cell(og_table, args)



    
