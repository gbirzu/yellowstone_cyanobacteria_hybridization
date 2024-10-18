import argparse
import subprocess
import glob
import pickle
import utils
import numpy as np
import pandas as pd
import seq_processing_utils as seq_utils
import pangenome_utils as pg_utils
import time
from pangenome_utils import PangenomeMap
from metadata_map import MetadataMap
from Bio import SeqIO
from Bio import AlignIO
from Bio.SeqRecord import SeqRecord


def update_orthogroup_table(og_table, subcluster_dict, output_file=None):
    for cluster_id in subcluster_dict:
        subclusters = subcluster_dict[cluster_id]
        for og_id in subclusters:
            gene_ids = subclusters[og_id]
            og_table.loc[og_id, 'num_seqs'] = len(gene_ids)
            og_table.loc[og_id, 'avg_length'] = pg_utils.calculate_mean_gene_length(gene_ids)

            sag_grouped_ids = pg_utils.group_og_table_genes(gene_ids, og_table)
            og_table.loc[og_id, 'num_cells'] = len(sag_grouped_ids)

            for sag_id in sag_grouped_ids:
                og_table.loc[og_id, sag_id] = ';'.join(sag_grouped_ids[sag_id])

        og_table['seqs_per_cell'] = og_table['num_seqs'] / og_table['num_cells']

    # Remove old clusters
    og_index = list(og_table.index)
    old_cluster_ids = [cluster_id for cluster_id in list(subcluster_dict.keys()) if cluster_id in og_index]
    og_table = og_table.drop(index=old_cluster_ids)

    if output_file is not None:
        og_table.to_csv(output_file, sep='\t')


def add_fine_scale_clusters(og_table, fine_scale_clusters_dir, ext='fna'):
    table_updates = {}
    for og_id in og_table.index:
        print(og_id)
        fine_scale_cluster_files = glob.glob(f'{fine_scale_clusters_dir}{og_id}-*.fna')
        if len(fine_scale_cluster_files) > 0:
            table_rows = []
            for f_seqs in fine_scale_cluster_files:
                cluster_records = utils.read_fasta(f_seqs)
                gene_ids = list(cluster_records.keys())
                cluster_id = f_seqs.split('/')[-1].replace(f'.{ext}', '')
                table_rows.append(pg_utils.make_subcluster_og_table_row(cluster_id, gene_ids, og_table))
            table_updates[og_id] = table_rows

    old_og_ids = list(table_updates.keys())
    for og_id in table_updates:
        print(og_id)
        for table_row in table_updates[og_id]:
            og_table = append_og_table_row(og_table, table_row)
    og_table = og_table.drop(old_og_ids)

    return og_table

def append_rows(og_table, table_rows_dict):
    for ssog_id in table_rows_dict:
        og_table.loc[ssog_id, :] = table_rows_dict[ssog_id]
    return og_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--data_dir', help='Main directory containing pangenome output.')
    parser.add_argument('-i', '--input_table', default=None, help='Path to input OG table.')
    parser.add_argument('-n', '--num_seqs', type=int, default=100, help='Presence threshold for including OG in high-confidence table.')
    parser.add_argument('-o', '--output_file', default=None, help='Path to output OG table.')
    parser.add_argument('-p', '--prefix', default='sscs', help='Output files prefix.')
    parser.add_argument('-u', '--updates_file', default=None, help='Path to pickle with OG table updates.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode.')
    parser.add_argument('--get_high_confidence_table', action='store_true', help='Only keep single-copy OGs with high presence.')
    parser.add_argument('--add_fine_scale_clusters', action='store_true', help='Search for OG clusters in ``data_dir``.')
    parser.add_argument('--update_orthogroup_assignments', action='store_true', help='Old analysis for compatibility.')
    parser.add_argument('--postprocessing_clean_up', action='store_true', help='Clean up problematic SAGs and OGs.')
    parser.add_argument('--update_mixed_orthogroups', action='store_true', help='Reassign mixed OG subclusters.')
    args = parser.parse_args()

    pangenome_map = PangenomeMap(f_orthogroup_table=args.input_table)
    if args.update_orthogroup_assignments == True:
        if args.get_high_confidence_table:
            og_table = pangenome_map.get_high_confidence_og_table(high_presence_threshold=args.num_seqs)
        else:
            og_table = pangenome_map.og_table
    
        subcluster_dict = {}
        update_files = sorted(glob.glob(f'{args.data_dir}filtered_orthogroups/*_og_updates.dat'))
        for f_update in update_files:
            og_id = f_update.split('/')[-1].strip('_og_updates.dat')
            update_dict = pickle.load(open(f_update, 'rb'))
            subcluster_dict[og_id] = update_dict

        if args.output_file is not None:
            update_orthogroup_table(og_table, subcluster_dict, output_file=args.output_file)
        else:
            update_orthogroup_table(og_table, subcluster_dict, output_file=args.input_table)

        if args.verbose:
            print(f'OGs updated: {[og_id for og_id in subcluster_dict]}')

    else:
        if args.get_high_confidence_table:
            og_table = pangenome_map.get_high_confidence_og_table(high_presence_threshold=args.num_seqs)
        else:
            og_table = pangenome_map.og_table

        if args.add_fine_scale_clusters:
            og_table = add_fine_scale_clusters(og_table, f'{args.data_dir}fine_scale_og_clusters/')
        if args.updates_file:
            update_dict = pickle.load(open(args.updates_file, 'rb'))
            old_clusters = []
            for cluster_id in update_dict:
                if args.verbose:
                    print(f'Updating {cluster_id}...')

                if cluster_id in og_table.index:
                    old_clusters.append(cluster_id)
                for table_rows_dict in update_dict[cluster_id]:
                    og_table = append_rows(og_table, table_rows_dict)
            og_table = og_table.drop(old_clusters)

            if args.update_mixed_orthogroups:
                metadata = MetadataMap()
                subcluster_label_dict = {'A':'a', 'Bp':'b', 'C':'c', 'O':'o', 'M':'M'}
                
                # Get parent OG IDs for updated OGs
                updated_og_ids = np.concatenate([np.concatenate([list(table_rows_dict.keys()) for table_rows_dict in update_dict[cluster_id]]) for cluster_id in update_dict])
                mixed_orthogroup_parent_ids = og_table.loc[updated_og_ids, 'parent_og_id'].unique()

                # Add sequence cluster labels to table
                pangenome_map.og_table = og_table
                og_table = pangenome_map.bin_ogs_by_species_composition(metadata, parent_og_ids=mixed_orthogroup_parent_ids, label_dict=subcluster_label_dict, reorder_columns=False)


    if args.postprocessing_clean_up == True:
        og_table = pangenome_map.og_table
        metadata = MetadataMap()

        # Remove SAGs with fewer than 100 hits to both OS-A and OS-B', and SAGs with ambiguous species assignment
        high_confidence_sag_ids = metadata.get_high_confidence_sag_ids()
        data_columns = [col for col in og_table.columns if 'Uncmic' not in col]
        out_columns = np.concatenate((data_columns, high_confidence_sag_ids))
        filtered_og_table = og_table.reindex(columns=out_columns)

        # Remove mislabeled SOGs
        mislabeled_sog_ids = ['clpP-1', 'clpP-2']
        filtered_og_table = filtered_og_table.drop(mislabeled_sog_ids)

        # Update stats and remove OGs with zero sequences
        filtered_og_table = pg_utils.update_og_table_stats(filtered_og_table)
        filtered_og_table = filtered_og_table.loc[filtered_og_table['num_seqs'] > 0, :]

        # Annotate clusters based on species composition
        pangenome_map.og_table = filtered_og_table
        pangenome_map.add_parent_og_ids()
        og_table = pangenome_map.bin_ogs_by_species_composition(metadata)

    if args.output_file is not None:
        og_table.to_csv(args.output_file, sep='\t')
