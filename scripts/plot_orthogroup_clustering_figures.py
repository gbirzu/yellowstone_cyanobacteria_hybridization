import argparse
import pickle
import numpy as np
import pandas as pd
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
from metadata_map import MetadataMap
from plot_utils import *

def plot_pdist_distributions(og_table, args):
    pdist_values = []
    for o in og_table.index:
        pdist_df = pickle.load(open(f'{args.pangenome_dir}pdist/{o}_trimmed_pdist.dat', 'rb'))
        pdist_values.append(utils.get_matrix_triangle_values(pdist_df.values, k=1))

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('pairwise distance, $\pi_{ij}$', fontsize=14)
    ax.set_ylabel('density', fontsize=14)
    ax.hist(np.concatenate(pdist_values), bins=100, density=True)
    ax.axvline(0.075, lw=2, ls='--', c='tab:red')
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}{args.fhead}orthogroup_pdist_distribution.pdf')
    plt.close()

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('pairwise distance, $\pi_{ij}$', fontsize=14)
    ax.set_ylabel('density', fontsize=14)
    ax.set_yscale('log')
    ax.hist(np.concatenate(pdist_values), bins=100, density=True)
    ax.axvline(0.075, lw=2, ls='--', c='tab:red')
    ax.set_yticks([1E-4, 1E-3, 1E-2, 1E-1, 1E0, 1E1])
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}{args.fhead}orthogroup_pdist_distribution_logy.pdf')
    plt.close()

    return pdist_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--figures_dir', default='../figures/analysis/', help='Directory metagenome recruitment files.')
    parser.add_argument('-P', '--pangenome_dir', default='../results/single-cell/sscs_pangenome_v2/', help='Directory with pangenome files.')
    parser.add_argument('-f', '--fhead', default='', help='File name head.')
    parser.add_argument('-g', '--orthogroup_table', required=True, help='File with orthogroup table.')
    parser.add_argument('-s', '--random_seed', default=713, type=int, help='RNG seed.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    args = parser.parse_args()

    og_table = pd.read_csv(args.orthogroup_table, sep='\t', index_col=0)

    #plot_pdist_distributions(og_table, args)
    
    # Plot pdist matrices
    rng = np.random.default_rng(args.random_seed)
    og_id = rng.choice(og_table.index.values)
    print(og_id)
    pdist_df = pickle.load(open(f'{args.pangenome_dir}pdist/{og_id}_trimmed_pdist.dat', 'rb'))
    plot_heatmap(pdist_df, cbar_label='pairwise distance', cmap='inferno', savefig=f'{args.figures_dir}{og_id}_pdist_matrix.pdf')
    plot_pdist_clustermap(pdist_df, cbar_label='pairwise distance', cmap='inferno', savefig=f'{args.figures_dir}{og_id}_pdist_clustermap.pdf')
    print(pdist_df)

    # Get clustering statistics
    f_clustered_table = args.orthogroup_table.replace('orthogroup_table.tsv', 'clustered_orthogroup_table.tsv')
    clustered_og_table = pd.read_csv(f_clustered_table, sep='\t', index_col=0)
    clustered_og_table = clustered_og_table.loc[clustered_og_table['parent_og_id'].isin(og_table.index.values)]
    unclustered_idx = np.array([i for i in clustered_og_table.index if '-' not in i])
    clustered_idx = np.array([i for i in clustered_og_table.index if '-' in i])
    parent_ids, parent_counts = utils.sorted_unique(clustered_og_table['parent_og_id'].values)
    #clustered_parent_ids, clustered_parent_counts = utils.sorted_unique(clustered_og_table.loc[clustered_idx, 'parent_og_id'].values)
    clustered_parent_ids = parent_ids[parent_counts > 1]
    clustered_parent_counts = parent_counts[parent_counts > 1]
    num_multicluster_ogs = np.sum(clustered_parent_counts > 3)
    print(og_table)
    print(clustered_og_table)
    print(len(unclustered_idx), len(clustered_idx))
    print(clustered_parent_ids[:num_multicluster_ogs])
    print(clustered_parent_counts[:num_multicluster_ogs])
    print(len(clustered_parent_ids), num_multicluster_ogs)

    # Plot OG species clusters
    x, y = utils.sorted_unique(parent_counts, sort='ascending')
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('species clusters', fontsize=14)
    ax.set_ylabel('number of orthogroups', fontsize=14)
    ax.set_yscale('log')
    ax.set_ylim(8E-1, 1.5 * np.max(y))
    ax.plot(x, y, '-o', lw=2, ms=6, mfc='none', mec='tab:blue')
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}{args.fhead}species_clusters_distribution.pdf')
    plt.close()


    multi_cluster_parent_ids = clustered_parent_ids[clustered_parent_counts > 3]
    clustered_og_table.loc[clustered_og_table['parent_og_id'].isin(multi_cluster_parent_ids), :].to_csv(f'../results/tests/pangenome_construction/{args.fhead}multi_cluster_og_table.tsv', sep='\t')




