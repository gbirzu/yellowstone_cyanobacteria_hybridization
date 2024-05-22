import argparse
import re
import numpy as np
import pandas as pd
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
from make_main_figures import plot_marginal
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


    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(f'mean seqs per cell', fontsize=14)
    ax.set_ylabel(f'orthogroups', fontsize=14)
    ax.set_yscale('log')
    ax.hist(og_table['seqs_per_cell'], bins=100)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}og_mean_seqs_per_cell_hist.pdf')
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


def plot_trimming_results(og_table, args):
    r = 7
    l_min = 100
    f_trim = args.f_trim

    fig = plt.figure(figsize=(1.25 * single_col_width, 1.25 * single_col_width))
    gspec = gridspec.GridSpec(figure=fig, ncols=2, nrows=2, 
            height_ratios=[1, r], width_ratios=[r, 1], 
            hspace=0.15, wspace=0.15, bottom=0.2, top=0.97, left=0.2, right=0.97)

    ax = plt.subplot(gspec[1, 0])
    ax.set_xlabel(f'pre-trim length', fontsize=14)
    ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_ylabel(f'post-trim length', fontsize=14)
    ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000])

    x, y = og_table.loc[og_table['num_cells'] > 1, ['avg_length', 'trimmed_avg_length']].fillna(0).values.T
    ax.scatter(x, y, s=10, ec='w', fc='tab:blue', alpha=0.3, lw=0.2)

    # Plot cutoff line
    x0 = -25
    x1 = 1.1 * np.max(x)
    ax.set_xlim(x0, x1)
    ax.set_ylim(x0, x1)

    xlim = np.array([l_min / f_trim, np.max(x)])
    ax.plot(xlim, f_trim * xlim, '--k', lw=1)
    ax.plot([x0, l_min], [l_min, l_min], '--k', lw=1)

    # Add marginal distributions
    plot_marginal(gspec[0, 0], y, bins=50, xlim=None)
    plot_marginal(gspec[1, 1], x, bins=50, xlim=None, spines=['left'], orientation='horizontal')

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}aln_trim_scatter.pdf')
    plt.close()
    
    f_trimmed = np.abs(x - y) / x
    f_trimmed2 = np.abs(x[x > 100] - y[x > 100]) / x[x > 100]

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(f'fraction alignment trimmed', fontsize=14)
    ax.set_ylabel(f'density', fontsize=14)
    ax.set_yscale('log')
    ax.set_ylim(1E-2, 3E1)
    ax.hist(f_trimmed2, bins=100, label='$L > 100$', density=True)

    # Plot exponential fit
    hist, _ = np.histogram(f_trimmed2, bins=100, density=True)
    mu = hist[0]
    f_arr = np.linspace(0, 1, 100)
    ax.plot(f_arr, mu * np.exp(-mu * f_arr) / (1 - np.exp(-mu)), '-k', lw=1)

    ax.axvline(1 - f_trim, ls='--', lw=2.0, c='tab:red')
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}f_trimmed_distribution.pdf')
    plt.close()


def plot_copy_number_distributions(og_table, args, f_cn=1.1):
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(f'mean seqs per cell', fontsize=14)
    ax.set_ylabel(f'CCDF', fontsize=14)
    ax.set_yscale('log')
    ax.hist(og_table['seqs_per_cell'], bins=500, density=True, histtype='step', cumulative=-1)
    ax.axvline(f_cn, ls='--', lw=1, c='tab:red')
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}filtered_og_mean_seqs_per_cell_ccdf.pdf')
    plt.close()

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(f'excess copies per cell', fontsize=14)
    ax.set_xscale('log')
    ax.set_ylabel(f'CCDF', fontsize=14)
    ax.set_yscale('log')
    ax.hist(og_table['seqs_per_cell'] - 1, bins=500, density=True, histtype='step', cumulative=-1)
    ax.axvline(f_cn - 1, ls='--', lw=1, c='tab:red')
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}filtered_og_mean_seqs_per_cell_ccdf_loglog.pdf')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--figures_dir', default='../figures/analysis/', help='Directory metagenome recruitment files.')
    parser.add_argument('-O', '--output_dir', default='../results/tests/', help='Directory in which results are saved.')
    parser.add_argument('-g', '--orthogroup_table', required=True, help='File with orthogroup table.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    parser.add_argument('--f_trim', default=0.85, help='Fraction alignment trimmed cutoff.')
    args = parser.parse_args()

    pangenome_map = PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    og_table = pangenome_map.og_table
    metadata = MetadataMap()

    # Process OG table
    meta_cols = ['seq_cluster'] + list(og_table.columns.values[:4])
    sag_cols = list(og_table.columns.values[4:])
    og_table['seq_cluster'] = [re.sub(r'[a-z]*', '', s) for s in og_table.index.values]
    og_table = og_table[meta_cols + sag_cols]

    plot_seq_cluster_orthogroups(og_table, args)
    plot_seqs_per_cell(og_table, args)

    plot_trimming_results(og_table, args)

    # Read filtered OG table
    og_table = pd.read_csv(f'{args.output_dir}filtered_orthogroup_table.tsv', sep='\t', index_col=0)
    plot_copy_number_distributions(og_table, args)

