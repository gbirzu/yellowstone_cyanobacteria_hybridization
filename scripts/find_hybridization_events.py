import argparse
import pandas as pd
import numpy as np
import pickle
import os
import sys
import re
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
from pangenome_utils import PangenomeMap
from metadata_map import MetadataMap
from Bio import SeqIO
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment


'''
This script searches for all suspected hybridization events.
'''

def find_multicluster_ogs(pangenome_map, max_num_subclusters):
    og_cluster_dict = pangenome_map.make_og_clusters_dict(max_num_subclusters=max_num_subclusters)
    multicluster_ogs_dict = {}
    for oid in og_cluster_dict:
        if len(og_cluster_dict[oid]) > 1:
            multicluster_ogs_dict[oid] = og_cluster_dict[oid]
    return multicluster_ogs_dict

def pick_test_ids(og_table, metadata):
    parent_og_ids = np.sort(np.unique(og_table['parent_og_id'].values))
    parent_og_table = pd.DataFrame(index=parent_og_ids, columns=['num_seqs', 'num_clusters'])
    for pid in parent_og_ids:
        parent_og_table.loc[pid, 'num_seqs'] = og_table.loc[og_table['parent_og_id'] == pid, 'num_seqs'].sum()
        parent_og_table.loc[pid, 'num_clusters'] = len(og_table.loc[og_table['parent_og_id'] == pid, :])

    np.random.seed(12345)
    two_cluster_pids = np.array(parent_og_table.loc[(parent_og_table['num_seqs'] < 150) & (parent_og_table['num_seqs'] > 100) & (parent_og_table['num_clusters'] == 2), :].index)
    higher_num_cluster_pids = np.array(parent_og_table.loc[(parent_og_table['num_seqs'] < 150) & (parent_og_table['num_seqs'] > 100) & (parent_og_table['num_clusters'] > 2), :].index)

    n2_test_pids = np.random.choice(two_cluster_pids, size=5)
    n3_test_pids = np.random.choice(higher_num_cluster_pids, size=5)
    pangenome_map.count_og_species_composition(metadata)
    og_species_counts_table = pangenome_map.og_species_counts_table
    print(f'n=2 clusters: {n2_test_pids}')
    print(og_table.loc[og_table['parent_og_id'].isin(n2_test_pids), :].iloc[:, :8])
    print(og_species_counts_table.loc[og_table.index[og_table['parent_og_id'].isin(n2_test_pids)], :])
    print('\n')

    print(f'n>2 clusters: {n3_test_pids}')
    print(og_table.loc[og_table['parent_og_id'].isin(n3_test_pids), :].iloc[:, :8])
    print(og_species_counts_table.loc[og_table.index[og_table['parent_og_id'].isin(n3_test_pids)], :])

def find_gene_cell_cluster_mismatches(og_table, metadata):
    #relevant_cluster_labels = ['A', 'Bp', 'C', 'O']
    relevant_cluster_labels = ['A', 'Bp', 'C', 'X']

    # Make mismatched SAG IDs dict
    #   maps each SOG cluster to SAGs that would be mismatched if they contained the SOG
    sag_ids = [col for col in og_table.columns if 'Uncmic' in col]
    sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')
    mismatched_sags_dict = {}
    for cluster_label in relevant_cluster_labels:
        mismatched_sags_dict[cluster_label] = np.concatenate([sorted_sag_ids[l] for l in sorted_sag_ids if l != cluster_label])

    # Make mismatch table
    relevant_sog_ids = list(og_table.loc[og_table['sequence_cluster'].isin(relevant_cluster_labels), :].index)
    og_cluster_mismatch_table = pd.DataFrame(index=relevant_sog_ids, columns = sag_ids)
    for cluster_label in mismatched_sags_dict:
        mismatched_sag_ids = mismatched_sags_dict[cluster_label]
        subtable = og_table.loc[og_table['sequence_cluster'] == cluster_label, mismatched_sag_ids]
        og_cluster_mismatch_table.loc[list(subtable.index), mismatched_sag_ids] = subtable.notna()

    return og_cluster_mismatch_table

def construct_hybridization_candidates_contig_cluster_sequences(gene_cluster_mismatches, contig_seqs_df, pangenome_map):
    og_table = pangenome_map.og_table

    # Get cluster sequences for contigs with mismatches
    contig_cluster_sequences = {}
    sag_ids = gene_cluster_mismatches.columns.values
    contig_seqs_sag_ids = np.unique([pangenome_map.get_gene_sag_id(s) for s in contig_seqs_df.index])
    joint_sag_ids = sag_ids[np.isin(sag_ids, contig_seqs_sag_ids)]

    for sag_id in joint_sag_ids:
        # Loop through SAGs and find contigs with genes from different species clusters
        hybrid_genes_og_ids = list(gene_cluster_mismatches.loc[gene_cluster_mismatches[sag_id] == True, sag_id].index)
        hybrid_genes_gene_ids = og_table.loc[hybrid_genes_og_ids, sag_id].values
        unique_contigs, num_hybrid_genes = utils.sorted_unique(['_'.join(gid.split('_')[:2]) for gid in hybrid_genes_gene_ids])

        for i, contig_id in enumerate(unique_contigs):
            # Loop through contigs with hybrid genes and record OG IDs for hybrid genes
            #   A bit convoluted, but resulting table is grouped by contings which will be useful later
            contig_seq = contig_seqs_df.loc[contig_id, 'cluster_labels_sequence']
            if contig_seq == contig_seq: # Filter NaN's

                # Construct mismatched OG IDs str
                og_ids = [hybrid_genes_og_ids[j] for j in range(len(hybrid_genes_gene_ids)) if contig_id in hybrid_genes_gene_ids[j]]
                if len(og_ids) == 1:
                    hybrid_ogs_ids_str = og_ids[0]
                else:
                    hybrid_ogs_ids_str = ';'.join(og_ids) 
                contig_cluster_sequences[contig_id] = (contig_seq, num_hybrid_genes[i], hybrid_ogs_ids_str)

    # Make hybridization table
    contig_ids = sorted(list(contig_cluster_sequences.keys()))
    hybridized_contigs_df = pd.DataFrame(index=contig_ids, columns=['cluster_labels_sequence', 'sag_id', 'sag_species', 'hybrid_og_ids', 'num_genes', 'num_hybrid_genes', 'num_matches'])
    for contig_id in contig_cluster_sequences:
        contig_seq, num_hybrid_genes, hybrid_ogs_ids_str = contig_cluster_sequences[contig_id]
        hybridized_contigs_df.loc[contig_id, 'cluster_labels_sequence'] = contig_seq
        hybridized_contigs_df.loc[contig_id, 'hybrid_og_ids'] = hybrid_ogs_ids_str
        hybridized_contigs_df.loc[contig_id, 'num_genes'] = len(contig_seq)
        hybridized_contigs_df.loc[contig_id, 'num_hybrid_genes'] = num_hybrid_genes
    return hybridized_contigs_df


def add_hybridized_contigs_stats(hybridized_contigs_df, pangenome_map, metadata):
    out_df = hybridized_contigs_df
    for cid in out_df.index:
        sid = pangenome_map.get_gene_sag_id(cid)
        sag_species = metadata.get_sag_species(sid)
        out_df.loc[cid, 'sag_id'] = sid
        out_df.loc[cid, 'sag_species'] = sag_species

        if sag_species is not None:
            seq_arr = np.array(list(out_df.loc[cid, 'cluster_labels_sequence']))
            out_df.loc[cid, 'num_matches'] = np.sum(seq_arr == sag_species[0])
    return out_df


def find_hybridization_breakpoints(pvalue_cutoff=0.01):
    annotations_dir = '../data/single-cell/filtered_annotations/sscs/'
    pangenome_dir = '../results/single-cell/sscs_pangenome/'
    f_orthogroup_table = f'{pangenome_dir}filtered_orthogroups/sscs_annotated_single_copy_orthogroup_presence.tsv'
    f_hybridization_table = '../results/single-cell/hybridization/sscs_hybridization_events.tsv'
    pangenome_map = PangenomeMap(gff_files_dir=annotations_dir, f_orthogroup_table=f_orthogroup_table)

    #----------------------#
    # Get recent transfer breakpoints
    #----------------------#
    hybridization_table = pd.read_csv(f_hybridization_table, sep='\t', index_col=0)
    og_table = pangenome_map.og_table

    # Find clear breakpoints between species OG clusters within contigs
    species_aliases = {'A':['A', 'a'], 'Bp':['B', 'b'], 'C':['C'], 'O':['O']}
    foreign_species = {'A':['B', 'b', 'C', 'O'], 'Bp':['A', 'a', 'C', 'O']}
    cluster_species_map = {'A':'A', 'a':'A', 'B':'Bp', 'b':'Bp', 'C':'C', 'O':'O'}

    results_df = pd.DataFrame(columns=['contig_id', 'host_species', 'donor_species', 'alignment_length', 'x_breakpoint', 'p-value'])
    for species in ['A', 'Bp']:
    #for species in ['Bp']:
        species_hybridizations = hybridization_table.loc[hybridization_table['sag_species'] == species, :]

        species_cluster_labels = species_aliases[species]
        nonspecies_cluster_labels = foreign_species[species]

        # Make filtration index
        contig_ids = np.array(species_hybridizations.index)
        cluster_label_seqs = species_hybridizations['cluster_labels_sequence'].values

        # Search for all possible hybrid substrings
        hybridization_contig_dict = {}
        has_hybrid_substring = np.zeros(len(cluster_label_seqs), dtype=bool)
        for l1 in species_cluster_labels:
            for l2 in nonspecies_cluster_labels:
                forward_substring = f'{l1}{l2}'
                reverse_substring = f'{l2}{l1}'
                ##has_substring = np.array([(forward_substring in seq) or (reverse_substring in seq) for seq in cluster_label_seqs])
                #has_hybrid_substring = has_hybrid_substring | has_substring

                hybridization_contig_dict[forward_substring] = contig_ids[[forward_substring in seq for seq in cluster_label_seqs]]
                hybridization_contig_dict[reverse_substring] = contig_ids[[reverse_substring in seq for seq in cluster_label_seqs]]

        gene_ids_map = pangenome_map.make_gene_id_to_orthogroup_map()
        metadata = MetadataMap()
        sag_ids = pangenome_map.get_sag_ids()
        species_sorted_sags = metadata.sort_sags(sag_ids, by='species')

        print(hybridization_contig_dict.keys())
        for transfer_type in hybridization_contig_dict:
        #for transfer_type in ['AB']:
            print(transfer_type)
            candidate_breakpoint_genes = get_candidate_breakpoint_genes(transfer_type, hybridization_contig_dict, hybridization_table, pangenome_map)
            for breakpoint_candidates in candidate_breakpoint_genes:
            #for breakpoint_candidates in candidate_breakpoint_genes[-1:]:
                for candidate_gene_id in breakpoint_candidates:
                    sog_id = gene_ids_map[candidate_gene_id]
                    og_id = og_table.loc[sog_id, 'parent_og_id']
                    f_aln = f'{pangenome_dir}_aln_results/{og_id}_aln.fna'
                    aln = seq_utils.read_alignment(f_aln)
                    aln_rec_ids = [rec.id for rec in aln]

                    hybridizing_species = list(transfer_type)
                    for label in hybridizing_species:
                        if label in foreign_species[species]:
                            donor_species = cluster_species_map[label]
                            break

                    host_index = og_table.loc[(og_table['parent_og_id'] == og_id) & (og_table['sequence_cluster'].isin(species_aliases[species])), :].index
                    if len(host_index) > 0:
                        host_species_sog_id = host_index[0]
                        host_species_gene_ids = pg_utils.read_gene_ids(og_table.loc[host_species_sog_id, species_sorted_sags[species]])
                    else:
                        host_species_sog_id = og_id
                        host_species_gene_ids = pangenome_map.get_og_gene_ids(og_id, sag_ids=species_sorted_sags[species])
                    host_species_gene_ids = [gene_id for gene_id in host_species_gene_ids if gene_id in aln_rec_ids]

                    if len(host_species_gene_ids) > 0:
                        host_species_aln = align_utils.get_subsample_alignment(aln, host_species_gene_ids)
                        host_species_consensus_seq = Seq(''.join(seq_utils.get_consensus_seq(host_species_aln, seq_type='codons', keep_gaps=True)))
                    else:
                        continue
                    
                    donor_index = og_table.loc[(og_table['parent_og_id'] == og_id) & (og_table['sequence_cluster'].isin(species_aliases[donor_species])), :].index
                    if donor_species != 'O':
                        if len(donor_index) > 0:
                            donor_species_sog_id = donor_index[0]
                            donor_species_gene_ids = pg_utils.read_gene_ids(og_table.loc[donor_species_sog_id, species_sorted_sags[donor_species]])
                        else:
                            donor_species_sog_id = og_id
                            donor_species_gene_ids = pangenome_map.get_og_gene_ids(og_id, sag_ids=species_sorted_sags[donor_species])
                    elif len(donor_index) == 1:
                        donor_species_sog_id = donor_index[0]
                        donor_species_gene_ids = pg_utils.read_gene_ids(og_table.loc[donor_species_sog_id, sag_ids])
                    else:
                        print(f'Found {len(donor_index)} "O" clusters in {og_id}')
                        donor_species_gene_ids = []
                    donor_species_gene_ids = [gene_id for gene_id in donor_species_gene_ids if gene_id in aln_rec_ids]

                    if len(donor_species_gene_ids) > 0:
                        donor_species_aln = align_utils.get_subsample_alignment(aln, donor_species_gene_ids)
                        donor_species_consensus_seq = Seq(''.join(seq_utils.get_consensus_seq(donor_species_aln, seq_type='codons', keep_gaps=True)))
                    else:
                        continue

                    # Create triplet alignment
                    if candidate_gene_id in aln_rec_ids:
                        rec_idx = aln_rec_ids.index(candidate_gene_id)
                        triplet_list = [aln[rec_idx]]
                        triplet_list.append(SeqRecord(host_species_consensus_seq, id=f'{host_species_sog_id}_{species}_consensus', description=''))
                        triplet_list.append(SeqRecord(donor_species_consensus_seq, id=f'{donor_species_sog_id}_{donor_species}_consensus', description=''))
                        triplet_aln = MultipleSeqAlignment(triplet_list)

                        candidate_contig_id = '_'.join(candidate_gene_id.split('_')[:2])
                        breakpoint_label = f'{og_id}_hybrid_{candidate_contig_id}'
                        align_utils.write_alignment(triplet_aln, f'../results/tests/hybridization/{breakpoint_label}_aln.fna')

                        aln_max_chi2 = align_utils.calculate_triplet_maxchi2(np.array(triplet_aln), dx=5)
                        test_results = aln_max_chi2[0]
                        
                        if test_results[2] < pvalue_cutoff:
                            x_breakpoint = test_results[0]
                        else:
                            x_breakpoint = ''

                        results_df.loc[breakpoint_label, :] = [candidate_contig_id, species, donor_species, triplet_aln.get_alignment_length(), x_breakpoint, test_results[2]]
                        print(sog_id, og_id, x_breakpoint, test_results[2])
                    else:
                        print(f'{candidate_gene_id} not found in {og_id} alignment!')

                print('\n')
                #break
            #break

    print(results_df)
    results_df.to_csv(f'../results/single-cell/hybridization/sscs_hybridization_breakpoints.tsv', sep='\t')

def get_candidate_breakpoint_genes(transfer_type, hybridization_contig_dict, hybridization_table, pangenome_map):
    contig_annotations = pangenome_map.annotations
    candidate_gene_ids = []
    for contig_id in hybridization_contig_dict[transfer_type]:
        contig_cluster_seq = hybridization_table.loc[contig_id, 'cluster_labels_sequence']
        hybrid_match = re.search(transfer_type, contig_cluster_seq)
        candidates_idx = [hybrid_match.start(), hybrid_match.end() - 1]
        contig_gene_ids = np.array(contig_annotations.loc[contig_annotations['contig'] == contig_id, :].index)
        candidate_gene_ids.append(contig_gene_ids[candidates_idx])
    return np.array(candidate_gene_ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-A', '--annotations_dir', default=None, help='Directory with SAG GFF annotations.')
    parser.add_argument('-O', '--output_dir', default='../results/single-cell/hybridization/', help='Directory in which intermediate results are saved.')
    #parser.add_argument('-c', '--contig_cluster_sequences', default=None, help='File with cluster label sequences for each contig.')
    parser.add_argument('-c', '--contig_cluster_sequences', default='../results/single-cell/hybridization/contig_sequence_clusters.tsv', help='File with cluster label sequences for each contig.')
    parser.add_argument('-g', '--orthogroup_table', help='File with orthogroup table.')
    parser.add_argument('-m', '--max_num_subclusters', default=5, help='Maximum number of subclusters per parent orthogroup.')
    parser.add_argument('-o', '--output_file', default=None, help='Path to output file.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    parser.add_argument('--pick_test_ids', action='store_true', help='Choose IDs to verify SOG species assignments.')
    parser.add_argument('--search_targets', default='hybrid_genes', help='Search for hybridization breakpoint.')
    args = parser.parse_args()

    if args.search_targets == 'hybridization_breakpoints':
        find_hybridization_breakpoints()

    elif args.search_targets == 'hybrid_genes':
        metadata = MetadataMap()
        pangenome_map = PangenomeMap(gff_files_dir=args.annotations_dir, f_orthogroup_table=args.orthogroup_table)
        og_table = pangenome_map.og_table

        if os.path.exists(args.contig_cluster_sequences):
            contig_seqs_df = pd.read_csv(args.contig_cluster_sequences, sep='\t', index_col=0)
        else:
            contig_seqs_df = pangenome_map.label_contig_sequence_clusters()
            contig_seqs_df.to_csv(args.contig_cluster_sequences, sep='\t')

        cluster_labels, label_counts = utils.sorted_unique(og_table['sequence_cluster'].values)
        gene_cluster_mismatches = find_gene_cell_cluster_mismatches(og_table, metadata)
        temp_df = construct_hybridization_candidates_contig_cluster_sequences(gene_cluster_mismatches, contig_seqs_df, pangenome_map)
        hybridized_contigs_df = add_hybridized_contigs_stats(temp_df, pangenome_map, metadata)

        # Save results
        if args.output_file is not None:
            hybridized_contigs_df.to_csv(args.output_file, sep='\t')

        # Print results info
        if args.pick_test_ids:
            pick_test_ids(og_table, metadata)

        if args.verbose:
            print(contig_seqs_df)
            print('\n')

            print(hybridized_contigs_df)
            print('\n')

