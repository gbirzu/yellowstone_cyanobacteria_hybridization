import argparse
import re
import numpy as np
import pandas as pd
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
from pangenome_utils import PangenomeMap


def partition_trimmed_alignments(og_table, args):
    f_trim = args.f_trim

    # Filter highly trimmed alignments
    og_table['f_trimmed'] = np.abs(og_table['avg_length'] - og_table['trimmed_avg_length']) / og_table['avg_length']
    highly_trimmed_ogs = og_table.index.values[(og_table['f_trimmed'] > 1 - f_trim) & (og_table['num_cells'] > 1)]
    high_confidence_ogs = og_table.index.values[(og_table['f_trimmed'] < 1 - f_trim) & (og_table['num_cells'] > 1)]
    singleton_ogs = og_table.index.values[og_table['num_cells'] <= 1]

    cols = og_table.columns.values[:-1] # remove 'f_trimmed'
    filtered_og_table = og_table.loc[high_confidence_ogs, cols]
    filtered_og_table.to_csv(f'{args.output_dir}filtered_orthogroup_table.tsv', sep='\t')

    trimmed_og_table = og_table.loc[highly_trimmed_ogs, cols]
    trimmed_og_table.to_csv(f'{args.output_dir}high_trim_orthogroup_table.tsv', sep='\t')

    singleton_og_table = og_table.loc[singleton_ogs, cols]
    singleton_og_table.to_csv(f'{args.output_dir}singleton_orthogroup_table.tsv', sep='\t')

    if args.verbose:
        print('Filtered OG table')
        print(filtered_og_table)
        print('\n\n')

        print('Highly-trimmed OG table')
        print(trimmed_og_table)
        print('\n\n')

        print('Singleton OG table')
        print(singleton_og_table)
        print('\n\n')

    return filtered_og_table


def partition_og_copy_numbers(og_table, args):
    f_cn = args.f_copy
    single_copy_og_table = og_table.loc[og_table['seqs_per_cell'] == 1, :]
    single_copy_og_table.to_csv(f'{args.output_dir}filtered_single_copy_orthogroup_table.tsv', sep='\t')
    low_copy_og_table = og_table.loc[og_table['seqs_per_cell'] < f_cn, :]
    low_copy_og_table.to_csv(f'{args.output_dir}filtered_low_copy_orthogroup_table.tsv', sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-O', '--output_dir', default='../results/tests/', help='Directory in which results are saved.')
    parser.add_argument('-g', '--orthogroup_table', required=True, help='File with orthogroup table.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    parser.add_argument('--f_trim', default=0.85, help='Fraction alignment trimmed cutoff.')
    parser.add_argument('--f_copy', default=1.1, help='Average copy number cutoff.')
    args = parser.parse_args()

    pangenome_map = PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    og_table = pangenome_map.og_table

    # Process OG table
    meta_cols = ['seq_cluster'] + list(og_table.columns.values[:4])
    sag_cols = list(og_table.columns.values[4:])
    og_table['seq_cluster'] = [re.sub(r'[a-z]*', '', s) for s in og_table.index.values]
    og_table = og_table[meta_cols + sag_cols]

    filtered_og_table = partition_trimmed_alignments(og_table, args)
    partition_og_copy_numbers(filtered_og_table, args)
