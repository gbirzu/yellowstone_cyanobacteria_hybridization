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
    plt.savefig(f'{args.figures_dir}orthogroup_pdist_distribution.pdf')
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
    plt.savefig(f'{args.figures_dir}orthogroup_pdist_distribution_logy.pdf')
    plt.close()

    return pdist_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--figures_dir', default='../figures/analysis/', help='Directory metagenome recruitment files.')
    parser.add_argument('-P', '--pangenome_dir', default='../results/single-cell/sscs_pangenome_v2/', help='Directory with pangenome files.')
    parser.add_argument('-g', '--orthogroup_table', required=True, help='File with orthogroup table.')
    parser.add_argument('-s', '--random_seed', default=None, type=int, help='RNG seed.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    args = parser.parse_args()

    og_table = pd.read_csv(args.orthogroup_table, sep='\t', index_col=0)

    #plot_pdist_distributions(og_table, args)
    
    rng = np.random.default_rng(args.random_seed)
    og_id = rng.choice(og_table.index.values)
    print(og_id)

    pdist_df = pickle.load(open(f'{args.pangenome_dir}pdist/{og_id}_trimmed_pdist.dat', 'rb'))

    plot_heatmap(pdist_df, cbar_label='pairwise distance', cmap='inferno', savefig=f'{args.figures_dir}{og_id}_pdist_matrix.pdf')
    plot_pdist_clustermap(pdist_df, cbar_label='pairwise distance', cmap='inferno', savefig=f'{args.figures_dir}{og_id}_pdist_clustermap.pdf')
    print(pdist_df)





