import argparse
import subprocess
import glob
import pickle
import utils
import numpy as np
import pandas as pd
import seq_processing_utils as seq_utils
import time
import ete3
import pangenome_utils as pg_utils
import pg_update_og_table as pg_update
import scipy.cluster.hierarchy as hclust
import scipy.spatial.distance as distance
from pangenome_utils import PangenomeMap
from Bio import Phylo
from Bio import SeqIO
from Bio import AlignIO
from Bio.SeqRecord import SeqRecord


def write_subcluster_seqs(subclusters, parent_id, seqs_dir, output_dir, seq_type='prot'):
    if seq_type == 'prot':
        ext = 'faa'
    else:
        ext = 'fna'
    cluster_records = utils.read_fasta(f'{seqs_dir}{parent_id}.{ext}')
    for subcluster_label in subclusters:
        subcluster_gene_ids = subclusters[subcluster_label]
        subcluster_records = []
        for gene_id in subcluster_gene_ids:
            subcluster_records.append(cluster_records[gene_id])
        SeqIO.write(subcluster_records, f'{output_dir}{subcluster_label}.{ext}', 'fasta')

def cluster_sequences(pdist_df, linkage_method='average'):
    gene_ids = np.array(list(pdist_df.index), dtype=object)
    pdist_squareform = distance.squareform(pdist_df.fillna(0).values)
    Z = hclust.linkage(pdist_squareform, method=linkage_method, optimal_ordering=True)
    return Z, gene_ids

def calculate_cluster_distances(pdist_df, gene_ids, clusters):
    num_clusters = np.max(clusters)
    D_clusters = np.zeros((num_clusters, num_clusters))
    for j in range(num_clusters):
        for i in range(j + 1):
            pdist_i = pdist_df.reindex(index=gene_ids[clusters == i + 1], columns=gene_ids[clusters == i + 1])
            pdist_j = pdist_df.reindex(index=gene_ids[clusters == j + 1], columns=gene_ids[clusters == j + 1])
            pdist_ij = pdist_df.reindex(index=gene_ids[clusters == i + 1], columns=gene_ids[clusters == j + 1])
            D_clusters[i, i] = np.max(pdist_i.values)
            D_clusters[j, j] = np.max(pdist_j.values)
            D_clusters[i, j] = np.min(pdist_ij.values)
            D_clusters[j, i] = np.min(pdist_ij.values)
    return D_clusters

def check_cluster_separation(D_clusters, min_ratio=1.5):
    check = True
    for j in range(len(D_clusters)):
        for i in range(j):
            dw = max(D_clusters[i, i], D_clusters[j, j])
            if dw > 0 and D_clusters[i, j] / dw < min_ratio:
                check = False
                break
    return check

def label_split_symmetry(clusters, symmetry_threshold):
    unique_ids, id_counts = utils.sorted_unique(clusters)
    if min(id_counts) / len(clusters) > symmetry_threshold:
        symmetry = 'S'
    else:
        symmetry = 'A'
    return symmetry

def label_subclusters(cluster_id, subcluster_idx, symmetry=None, label_type='number_symmetry'):
    if label_type == 'alphabetic':
        # Use ASCII table to convert `subcluster_idx` to letter
        return f'{cluster_id}{chr(97 + subcluster_idx)}'
    elif label_type == 'numeric':
        return f'{cluster_id}-{subcluster_idx:d}'
    else:
        return f'{cluster_id}-{subcluster_idx:d}{symmetry}'


def find_orthogroup_subclusters(og_table, divergence_matrices, args):
    scog_ids_list = [scog_id for scog_id in scog_table.index if scog_id in divergence_matrices]
    scog_updates = {}
    for scog_id in scog_ids_list:
        if args.verbose:
            print(f'Clustering {scog_id}...')

        pdist_df = divergence_matrices[scog_id]
        Z, gene_ids = cluster_sequences(pdist_df)
        clusters = hclust.fcluster(Z, args.branch_cutoff, criterion='distance')
        num_clusters = np.max(clusters)

        if num_clusters > 1:
            D_clusters = calculate_cluster_distances(pdist_df, gene_ids, clusters)
            clusters_are_separated = check_cluster_separation(D_clusters, min_ratio=args.min_distance_ratio)
            if clusters_are_separated:
                splitting_symmetry = label_split_symmetry(clusters, args.symmetry_threshold)
                subclusters = {}
                for i in range(num_clusters):
                    subcluster_ids = gene_ids[clusters == i + 1]
                    subcluster_label = label_subclusters(scog_id, i + 1, splitting_symmetry, label_type=args.label_type)
                    subclusters[subcluster_label] = subcluster_ids
                write_subcluster_seqs(subclusters, scog_id, f'{args.data_dir}filtered_orthogroups/', args.output_dir, seq_type='nucl')
                #scog_updates[scog_id] = [pg_utils.make_og_table_row(subcluster_id, subclusters[subcluster_id], scog_table) for subcluster_id in subclusters]
                scog_updates[scog_id] = make_updated_table_rows(scog_id, subclusters, scog_table, reclustering_flag=args.recluster_mixed_scogs)

    return scog_updates

def make_updated_table_rows(scog_id, subclusters, scog_table, reclustering_flag=False):
    if reclustering_flag == False:
        new_table_rows = [pg_utils.make_og_table_row(subcluster_id, subclusters[subcluster_id], scog_table) for subcluster_id in subclusters]
    else:
        new_table_rows = []
        for subcluster_id in subclusters:
            numeric_subcluster_id = subcluster_id[:-1]
            new_row = pg_utils.make_og_table_row(numeric_subcluster_id, subclusters[subcluster_id], scog_table)
            subcluster_term = subcluster_id.split('-')[-1]
            new_row[numeric_subcluster_id]['locus_tag'] = scog_table.loc[scog_id, 'locus_tag']
            new_row[numeric_subcluster_id]['og_id'] = scog_table.loc[scog_id, 'og_id'] + f'-{subcluster_term}'
            new_row[numeric_subcluster_id]['parent_og_id'] = scog_table.loc[scog_id, 'og_id']
            new_table_rows.append(new_row)
    return new_table_rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--data_dir', help='Main directory containing pangenome output.')
    parser.add_argument('-O', '--output_dir', help='Directory in which subcluster seqs are saved.')
    parser.add_argument('-o', '--output_file', default=None, help='Output file for updated OG table.')
    parser.add_argument('-b', '--branch_cutoff', default=0.075, type=float, help='Cutoff length for long branches.')
    parser.add_argument('-d', '--divergence_matrices_file', help='Input file with pre-computed divergences between cluster sequences.')
    parser.add_argument('-g', '--orthogroup_table', default=None, help='File with orthogroup table.')
    parser.add_argument('-l', '--label_type', default='number_symmetry', help='Output files prefix.')
    parser.add_argument('-p', '--prefix', default='sscs', help='Output files prefix.')
    parser.add_argument('-r', '--min_distance_ratio', default=1.5, type=float, help='Min db/dw ratio for accepting cluster split.')
    parser.add_argument('-s', '--symmetry_threshold', default=0.1, type=float, help='Min relative size of smallest subcluster for splitting to be labeled symmetric.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    parser.add_argument('--recluster_mixed_scogs', action='store_true', help='Recluster OGs labeled as mixed ("M") in previous round.')
    parser.add_argument('--test_mixed_subclustering', action='store_true', help='Run mixed clusters subclustering tests.')
    args = parser.parse_args()


    if args.test_mixed_subclustering:
        test_mixed_subclustering(args)
        exit()


    # Read inputs
    divergence_matrices = pickle.load(open(args.divergence_matrices_file, 'rb'))
    if args.orthogroup_table is None:
        pangenome_map = PangenomeMap(f_orthogroup_table=f'{args.output_dir}{args.prefix}_single_copy_orthogroup_presence.tsv')
    else:
        pangenome_map = PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    scog_table = pangenome_map.og_table
    scog_updates = find_orthogroup_subclusters(scog_table, divergence_matrices, args)

    # Save SCOG table updates
    with open(args.output_file, 'wb') as out_handle:
        pickle.dump(scog_updates, out_handle)
