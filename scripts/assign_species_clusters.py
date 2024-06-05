import argparse
import re
import numpy as np
import pandas as pd
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
import metadata_map as mm

def bin_ogs_by_species_composition(og_table, metadata, label_dict={'A':'A', 'Bp':'Bp', 'C':'C', 'O':'X', 'M':'M'}, reorder_columns=True):
    og_table = self.og_table
    og_table = og_table.reindex(index=np.sort(np.unique(og_table.index)))
    sag_ids = [col for col in og_table if 'Uncmic' in col]
    species_sorted_sags = metadata_map.sort_sags(sag_ids, by='species')

    og_table = sort_species_clusters(og_table)
    for s in ['C', 'Bp', 'A']:
        s_ids = species_sorted_sags[s]
        n_series = og_table[s_ids].notna().sum(axis=1)
        og_table.insert(7, f'{s}_sag_abundance', n_series)

    parent_og_ids = np.unique(og_table['parent_og_id'].values)
    for pid in parent_og_ids:
        og_num_seqs = og_table.loc[og_table['parent_og_id'] == pid, 'num_seqs'].sort_values(ascending=False)

        if len(og_num_seqs) == 1:
            sid = og_num_seqs.index[0]
            og_table.loc[sid, 'sequence_cluster'] = self.label_major_og_cluster(sid, label_dict=label_dict)
        else:
            labeled_major_clusters = {'A':False, 'Bp':False}
            for i, sid in enumerate(og_num_seqs.index):
                are_major_clusters_labeled = labeled_major_clusters['A'] & labeled_major_clusters['Bp']
                if are_major_clusters_labeled == False:
                    cluster_label = self.label_major_og_cluster(sid, label_dict=label_dict)
                    og_table.loc[sid, 'sequence_cluster'] = cluster_label
                    labeled_major_clusters[cluster_label] = True
                else:
                    cluster_label = self.label_minor_og_cluster(sid, label_dict=label_dict)
                    og_table.loc[sid, 'sequence_cluster'] = cluster_label

    # Reorder columns and return table
    data_columns = [col for col in og_table if 'Uncmic' not in col]
    sag_columns = [col for col in og_table if 'Uncmic' in col]
    if reorder_columns:
        new_columns = data_columns[:3] + [data_columns[-1]] + data_columns[3:-1] + sag_columns
    else:
        new_columns = data_columns + sag_columns

    return og_table.reindex(columns=new_columns)


def sort_species_clusters(og_table):
    parent_og_ids = np.sort(og_table['parent_og_id'].unique())

    sorted_idx = []
    for p in parent_og_ids:
        p_idx = og_table.loc[og_table['parent_og_id'] == p, 'num_seqs'].sort_values(ascending=False).index.values
        sorted_idx.append(p_idx)
    sorted_idx = np.concatenate(sorted_idx)

    return og_table.loc[sorted_idx, :]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-O', '--output_dir', default='../results/tests/', help='Directory in which results are saved.')
    parser.add_argument('-g', '--orthogroup_table', required=True, help='File with orthogroup table.')
    parser.add_argument('-o', '--output_file', required=True, help='File with output orthogroup table.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    args = parser.parse_args()

    metadata_map = mm.MetadataMap()
    og_table = pd.read_csv(args.orthogroup_table, sep='\t', index_col=0)
    core_og_table = og_table.loc[(og_table['core_A'] == 'Yes') & (og_table['core_Bp'] == 'Yes'), :]
    sag_ids = np.array([c for c in og_table.columns if 'Uncmic' in c])
    species_sorted_sags = metadata_map.sort_sags(sag_ids, by='species')

    '''
    core_og_table = sort_species_clusters(core_og_table)
    for s in ['C', 'Bp', 'A']:
        s_ids = species_sorted_sags[s]
        n_series = core_og_table[s_ids].notna().sum(axis=1)
        core_og_table.insert(7, f'{s}_sag_abundance', n_series)
    '''
    pangenome_map = pg_utils.PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    labeled_og_table = pangenome_map.bin_ogs_by_species_composition(metadata_map)
    print(labeled_og_table.iloc[:, :10])
