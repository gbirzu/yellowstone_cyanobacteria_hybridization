import argparse
import pandas as pd
import numpy as np
import os
import utils
import pangenome_utils as pg_utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
from metadata_map import MetadataMap


def write_full_core_alignments(og_table, args):
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

            if args.verbose:
                print(og_id)
                print(aln)
                print(cleaned_aln)
                print('\n')
        else:
            print(f'{f_aln} does not exist!')

def write_species_core_alignments(og_table, args, sites='all'):
    metadata = MetadataMap()
    sag_ids = [c for c in og_table.columns if 'Uncmic' in c]
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')

    out_ext_dict = {'A':f'_{sites}_A_aln.fna', 'Bp':f'_{sites}_B_aln.fna'}
    for og_id in og_table['parent_og_id'].unique():
        f_aln = f'{args.alignment_dir}{og_id}{args.extension}'
        if os.path.exists(f_aln):
            aln = seq_utils.read_alignment(f_aln)

            # Get gene IDs
            for species in ['A', 'Bp']:
                species_sag_ids = species_sorted_sags[species]
                gene_ids = []
                for i, row in og_table.loc[og_table['parent_og_id'] == og_id, species_sag_ids].iterrows():
                    row_ids = row.dropna().str.split(';')
                    if len(row_ids) > 1:
                        gene_ids.append(np.concatenate(row_ids))
                    else:
                        gene_ids.append(row_ids.values)
                gene_ids = np.concatenate(gene_ids)

                species_aln = align_utils.get_subsample_alignment(aln, gene_ids)

                if sites == '4D':
                    output_aln, x_sites = seq_utils.get_synonymous_sites(species_aln, return_x=True)
                    if len(x_sites) > 0:
                        f_aln_out = f'{args.output_dir}{og_id}{out_ext_dict[species]}'
                        align_utils.write_alignment(output_aln, f_aln_out)

                        # Save site positions
                        f_sites = f_aln_out.replace('_aln.fna', '_sites.txt')
                        np.savetxt(f_sites, np.column_stack(x_sites), fmt='%d', delimiter=',')
                    else:
                        print(og_id, species, output_aln)
                else:
                    align_utils.write_alignment(species_aln, f'{args.output_dir}{og_id}{out_ext_dict[species]}')

                if args.verbose:
                    print(og_id, species)
                    print(aln)
                    print(species_aln)
                    print('\n')
            if args.verbose:
                print('\n')
        else:
            print(f'{f_aln} does not exist!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-A', '--alignment_dir', default='../results/single-cell/sscs_pangenome_v2/trimmed_aln/', help='Directory with initial alignment files.')
    parser.add_argument('-O', '--output_dir', default='../results/single-cell/alignments/core_ogs_cleaned/', help='Directory in which cleaned alignments are saved.')
    parser.add_argument('-e', '--extension', default='_trimmed_aln.fna', help='Input files extension.')
    parser.add_argument('-g', '--orthogroup_table', help='File with orthogroup table.')
    parser.add_argument('-s', '--sites', default='all', help='["all", "4D"].')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    parser.add_argument('--split_species', action='store_true')
    args = parser.parse_args()

    og_table = pd.read_csv(args.orthogroup_table, sep='\t', index_col=0)


    '''
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
 cleaned_aln   '''

    if args.split_species:
        write_species_core_alignments(og_table, args, sites=args.sites)
    else:
        write_full_core_alignments(og_table, args)

