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
    # Double sort table: alphabetically by parent OG ID and in order of decreasing abundance within each parent OG
    og_table = sort_species_clusters(og_table)
    og_table.insert(3, 'sequence_cluster', None)

    # Calculate species abundances for each OG cluster
    sag_ids = [col for col in og_table if 'Uncmic' in col]
    species_sorted_sags = metadata_map.sort_sags(sag_ids, by='species')
    sag_species = ['A', 'Bp', 'C']
    for s in sag_species[::-1]:
        s_ids = species_sorted_sags[s]
        n_series = og_table[s_ids].notna().sum(axis=1)
        og_table.insert(7, f'{s}_sag_abundance', n_series)

    parent_og_ids = np.unique(og_table['parent_og_id'].values)
    for pid in parent_og_ids:
        abundance_cols = ['num_cells', 'A_sag_abundance', 'Bp_sag_abundance', 'C_sag_abundance']
        og_abundance = og_table.loc[og_table['parent_og_id'] == pid, abundance_cols].sort_values('num_cells', ascending=False)

        if len(og_abundance) == 1:
            # Assign single cluster OGs to A, Bp, or M
            sid = og_abundance.index[0]
            og_table.loc[sid, 'sequence_cluster'] = label_single_og_cluster(sid, og_table, label_dict=label_dict)
        else:
            # Guess multi-cluster OG labels based on cluster composition
            f_max = og_abundance['num_cells'].values[0] / np.sum(og_abundance['num_cells'])
            if f_max > args.f_single_species:
                # Assign largest cluster to M if fraction of sequences is too high to be explained by Bp cluster 
                og_table.loc[og_abundance.index.values[0], 'sequence_cluster'] = label_dict['M']
                og_table.loc[og_abundance.index.values[1:], 'sequence_cluster'] = label_dict['O']
            else:
                assigned_labels = []
                for i, sid in enumerate(og_abundance.index):
                    if og_abundance.loc[sid, 'C_sag_abundance'] > 0:
                        # Assign clusters containing gamma sequence to C
                        og_table.loc[sid, 'sequence_cluster'] = label_dict['C']
                        assigned_labels.append(label_dict['C'])
                    else:
                        # Assign other clusters based on majority species
                        majority_label = sag_species[np.argmax(og_abundance.loc[sid, [f'{s}_sag_abundance' for s in sag_species]])]
                        if majority_label in assigned_labels:
                            # If majority species already assign, automatically label O
                            og_table.loc[sid, 'sequence_cluster'] = label_dict['O']
                            assigned_labels.append(label_dict['O'])
                        else:
                            og_table.loc[sid, 'sequence_cluster'] = label_dict[majority_label]
                            assigned_labels.append(label_dict[majority_label])

    # Reorder columns and return table
    '''
    data_columns = [col for col in og_table if 'Uncmic' not in col]
    sag_columns = [col for col in og_table if 'Uncmic' in col]
    if reorder_columns:
        new_columns = data_columns[:2] + [data_columns[-1]] + data_columns[2:-1] + sag_columns
    else:
        new_columns = data_columns + sag_columns
    return og_table.reindex(columns=new_columns).rename(columns={'seq_cluster':'protein_sequence_cluster'})
    '''
    return og_table.rename(columns={'seq_cluster':'protein_sequence_cluster'})


def sort_species_clusters(og_table):
    parent_og_ids = np.sort(og_table['parent_og_id'].unique())
    sorted_idx = []
    for p in parent_og_ids:
        p_idx = og_table.loc[og_table['parent_og_id'] == p, 'num_seqs'].sort_values(ascending=False).index.values
        sorted_idx.append(p_idx)
    sorted_idx = np.concatenate(sorted_idx)
    return og_table.loc[sorted_idx, :]


def label_single_og_cluster(sog_id, og_table, label_dict={'A':'A', 'Bp':'Bp', 'M':'M'}, f_single_species_threshold=0.9):
    if og_table.loc[sog_id, 'C_sag_abundance'] > 0:
        s_label = label_dict['M']
    else:
        f_A = og_table.loc[sog_id, 'A_sag_abundance'] / og_table.loc[sog_id, 'num_cells']
        f_Bp = og_table.loc[sog_id, 'Bp_sag_abundance'] / og_table.loc[sog_id, 'num_cells']
        if f_A > f_single_species_threshold:
            s_label = label_dict['A']
        elif f_Bp > f_single_species_threshold:
            s_label = label_dict['Bp']
        else:
            s_label = label_dict['M']
    return s_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-O', '--output_dir', default='../results/tests/', help='Directory in which results are saved.')
    parser.add_argument('-g', '--orthogroup_table', required=True, help='File with orthogroup table.')
    parser.add_argument('-o', '--output_file', required=True, help='File with output orthogroup table.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    parser.add_argument('-f', '--f_single_species', default=0.8, type=float, help='Maximum fraction of sequences in single-species cluster.')
    args = parser.parse_args()

    metadata_map = mm.MetadataMap()
    og_table = pd.read_csv(args.orthogroup_table, sep='\t', index_col=0)
    sag_ids = np.array([c for c in og_table.columns if 'Uncmic' in c])
    species_sorted_sags = metadata_map.sort_sags(sag_ids, by='species')
    core_og_table = og_table.loc[(og_table['core_A'] == 'Yes') & (og_table['core_Bp'] == 'Yes'), :]


    # New method
    labeled_og_table = bin_ogs_by_species_composition(core_og_table, metadata_map)
    labeled_og_table = labeled_og_table.drop(columns=['trimmed_avg_length'])
    labeled_og_table.to_csv(args.output_file, sep='\t')

    if args.verbose:
        cluster_labels, label_counts = utils.sorted_unique(labeled_og_table['sequence_cluster'])
        print(labeled_og_table.iloc[:, :10])
        print(cluster_labels)
        print(label_counts)


