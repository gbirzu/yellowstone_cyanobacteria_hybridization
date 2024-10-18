import argparse
import glob
import pickle
import utils
import numpy as np
import pandas as pd
import pangenome_utils as pg_utils


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

def update_species_clusters(og_table, subcluster_dict, output_file=None):
    old_columns = og_table.columns.values
    og_table.insert(0, 'parent_og_id', og_table.index.values)
    for og_id in subcluster_dict:
        for subclusters in subcluster_dict[og_id]:
            for sog_id in subclusters:
                og_table.loc[sog_id, old_columns] = subclusters[sog_id]
                og_table.loc[sog_id, 'parent_og_id'] = og_id

    # Remove old clusters
    og_index = list(og_table.index)
    old_og_ids = [og_id for og_id in list(subcluster_dict.keys()) if og_id in og_index]
    og_table = og_table.drop(index=old_og_ids)

    if output_file is not None:
        og_table.to_csv(output_file, sep='\t')


def add_fine_scale_clusters(og_table, fine_scale_clusters_dir, ext='fna'):
    table_updates = {}
    for og_id in og_table.index:
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
    parser.add_argument('-U', '--update_files_dir', help='Directory containing update files.')
    parser.add_argument('-i', '--input_table', default=None, help='Path to input OG table.')
    parser.add_argument('-o', '--output_table', default=None, help='Path to output OG table.')
    parser.add_argument('-t', '--updates_type', default='ogs', help='["ogs", "og_clusters"]')
    parser.add_argument('-u', '--updates_file', default=None, help='Path to pickle with OG table updates.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode.')
    args = parser.parse_args()

    pangenome_map = pg_utils.PangenomeMap(f_orthogroup_table=args.input_table)
    og_table = pangenome_map.og_table

    subcluster_dict = {}

    # Process all update files from args.update_files_dir
    update_files = sorted(glob.glob(f'{args.update_files_dir}*_updates.dat'))

    if args.updates_type == 'ogs':
        for f_update in update_files:
            update_dict = pickle.load(open(f_update, 'rb'))
            og_id = '_'.join(f_update.split('/')[-1].split('_')[:2])
            subcluster_dict[og_id] = update_dict
        if args.output_table is not None:
            update_orthogroup_table(og_table, subcluster_dict, output_file=args.output_table)

    elif args.updates_type == 'og_clusters':
        for f_update in update_files:
            update_dict = pickle.load(open(f_update, 'rb'))
            subcluster_dict = {**subcluster_dict, **update_dict}
        if args.output_table is not None:
            update_species_clusters(og_table, subcluster_dict, output_file=args.output_table)

    if args.verbose:
        print(f'OGs updated: {[og_id for og_id in subcluster_dict]}')

