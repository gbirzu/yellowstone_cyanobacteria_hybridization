import argparse
import numpy as np
import pandas as pd
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
from metadata_map import MetadataMap
from plot_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--figures_dir', default='../figures/analysis/', help='Directory metagenome recruitment files.')
    parser.add_argument('-P', '--pangenome_dir', default='../results/single-cell/sscs_pangenome_v2/', help='Directory with pangenome files.')
    parser.add_argument('-g', '--orthogroup_table', required=True, help='File with orthogroup table.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    args = parser.parse_args()

    og_table = pd.read_csv(args.orthogroup_table, sep='\t', index_col=0)
    sag_cols = np.array([c for c in og_table.columns if 'Uncmic' in c])

    og_prevalence = pd.DataFrame(0, index=og_table.index.values, columns=sag_cols)
    for c in sag_cols:
        og_prevalence[c] = og_table[c].str.split(';').str.len()
    print(og_prevalence)

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('orthogroups', fontsize=14)
    ax.set_ylabel('SAGs', fontsize=14)
    ax.hist(og_prevalence.sum(axis=0), bins=20)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}sag_og_number_distribution.pdf')
    plt.close()


