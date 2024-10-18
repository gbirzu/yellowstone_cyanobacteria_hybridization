import argparse
import pandas as pd
import numpy as np
import os
import utils
import pangenome_utils as pg_utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
from metadata_map import MetadataMap


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-A', '--alignment_dir', default='../results/single-cell/sscs_pangenome_v2/trimmed_aln/', help='Directory with initial alignment files.')
    parser.add_argument('-O', '--output_dir', default='../results/single-cell/alignments/core_ogs_cleaned/', help='Directory in which cleaned alignments are saved.')
    parser.add_argument('-g', '--orthogroup_table', help='File with orthogroup table.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    args = parser.parse_args()

    og_table = pd.read_csv(args.orthogroup_table, sep='\t', index_col=0)
    sag_ids = [c for c in og_table.columns if 'Uncmic' in c]

    for og_id in og_table['parent_og_id'].unique():
        f_aln = f'{args.alignment_dir}{og_id}_trimmed_aln.fna'
        if os.path.exists(f_aln):
            aln = seq_utils.read_alignment(f_aln)

            # Get gene IDs
            gene_ids = []
            for i, row in og_table.loc[og_table['parent_og_id'] == og_id, sag_ids].iterrows():
                row_ids = row.dropna().str.split(';')
                gene_ids.append(np.concatenate(row_ids))
            gene_ids = np.concatenate(gene_ids)

            cleaned_aln = align_utils.get_subsample_alignment(aln, gene_ids)

            align_utils.write_alignment(cleaned_aln, f'{args.output_dir}{og_id}_cleaned_aln.fna')

            print(og_id)
            print(aln)
            print(cleaned_aln)
            print('\n')
        else:
            print(f'{f_aln} does not exist!')
