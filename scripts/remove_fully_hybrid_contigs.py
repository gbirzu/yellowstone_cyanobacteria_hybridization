import argparse
import pandas as pd
import numpy as np
import pickle
import os
import utils
import pangenome_utils as pg_utils
from metadata_map import MetadataMap



if __name__ == '__main__':
    # Define default variables
    f_og_table = '../results/single-cell/sscs_pangenome_v2/filtered_low_copy_clustered_core_mapped_labeled_orthogroup_table.tsv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--contigs_file', required=True, help='File with list of contigs to exclude.')
    parser.add_argument('-g', '--orthogroup_table', default=f_og_table, help='File with orthogroup table.')
    parser.add_argument('-o', '--output_file', default='../results/tests/cleaned_orthogroup_table.tsv', help='Output file.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    parser.add_argument('--random_seed', default=12345, type=int, help='Random seed for reproducibility.')
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    fully_hybrid_contigs = np.loadtxt(args.contigs_file, dtype=str)
    og_table = pd.read_csv(args.orthogroup_table, sep='\t', index_col=0)

    sag_ids = [c for c in og_table.columns if 'Uncmic' in c]

    cleaned_og_table = og_table.fillna('')
    for s in sag_ids:
        single_copy_genes = cleaned_og_table.loc[~cleaned_og_table[s].str.contains(';'), s]
        single_copy_contigs = single_copy_genes.str.split('_').str[:2].str.join('_')
        filtered_idx = single_copy_contigs.index.values[single_copy_contigs.isin(fully_hybrid_contigs)]
        cleaned_og_table.loc[filtered_idx, s] = ''

        multicopy_genes = cleaned_og_table.loc[cleaned_og_table[s].str.contains(';'), s]
        if len(multicopy_genes) > 0:
            gene_ids = multicopy_genes.str.split(';').values
            filtered_gene_ids = []
            for l in gene_ids:
                filtered_gene_ids.append(';'.join([g for g in l if '_'.join(g.split('_')[:2]) not in fully_hybrid_contigs]))
            cleaned_og_table.loc[multicopy_genes.index, s] = filtered_gene_ids

    # Update abundance stats
    metadata = MetadataMap()
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    for species in ['A', 'Bp', 'C']:
        s_ids = species_sorted_sags[species]
        n_series = (cleaned_og_table[s_ids].values != '').sum(axis=1)
        cleaned_og_table.loc[:, f'{species}_sag_abundance'] = n_series
    cleaned_og_table.loc[:, 'num_cells'] = cleaned_og_table[['A_sag_abundance', 'Bp_sag_abundance', 'C_sag_abundance']].sum(axis=1)

    for i in cleaned_og_table.index:
        temp = np.concatenate(cleaned_og_table.loc[i, sag_ids].str.split(';'))
        gene_ids = temp[temp != '']
        cleaned_og_table.loc[i, 'num_seqs'] = len(gene_ids)
        cleaned_og_table.loc[i, 'avg_length'] = pg_utils.calculate_mean_gene_length(gene_ids)
    cleaned_og_table.loc[:, 'seqs_per_cell'] = cleaned_og_table['num_seqs'] / cleaned_og_table['num_cells']

    # Save output
    cleaned_og_table.to_csv(args.output_file, sep='\t')

    if args.verbose:
        print(cleaned_og_table)

