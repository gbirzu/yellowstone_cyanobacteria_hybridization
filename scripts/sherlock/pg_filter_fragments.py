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
import scipy.sparse as sparse
from Bio import SeqIO
from Bio import AlignIO
from Bio.SeqRecord import SeqRecord

def make_length_graph(seq_records, length_threshold=None, edge_threshold=0.7):
    seq_ids = np.array(list(seq_records.keys()))
    length_arr = np.array([len(seq_records[seq_id]) for seq_id in seq_ids])
    seq_ids = seq_ids[np.argsort(length_arr)]
    length_arr = length_arr[np.argsort(length_arr)]
    if length_threshold is not None:
        w = np.abs(length_arr[:, None] - length_arr[None, :])
        print(length_threshold)
        print(w)
        w_binary = (w <= length_threshold).astype(int)
        print(w_binary)
    else:
        w = 1 - (np.abs(length_arr[:, None] - length_arr[None, :]) / length_arr)
        w_symmetric = np.triu(w, k=0) + np.triu(w, k=1).T
        w_binary = (w_symmetric > edge_threshold).astype(int)
    return sparse.csr_matrix(w_binary), seq_ids

def format_abc_graph(edge_matrix, ordered_ids):
    graph = []
    for j in range(edge_matrix.shape[0]):
        for i in range(j + 1):
            graph.append([ordered_ids[i], ordered_ids[j], f'{edge_matrix[i, j]:.4f}'])
            graph.append([ordered_ids[j], ordered_ids[i], f'{edge_matrix[i, j]:.4f}'])
    return graph

def find_length_clusters(seq_records, length_threshold=None, edge_threshold=0.9, savefig=None):
    edge_matrix, sorted_ids = make_length_graph(seq_records, length_threshold=length_threshold, edge_threshold=edge_threshold)
    num_components, components = sparse.csgraph.connected_components(edge_matrix)
    sorted_lengths = np.array([len(seq_records[seq_id]) for seq_id in sorted_ids])
    #if savefig:
    #    plot_length_clusters_distribution(edge_matrix, sorted_lengths, savefig=savefig)
    return sorted_ids, components

def filter_length_clusters(seq_records, sorted_ids, length_clusters, size_threshold=0.8):
    cluster_index, cluster_sizes = utils.sorted_unique(length_clusters)
    num_seqs = sum(cluster_sizes)
    if len(cluster_sizes) == 0:
        filtered_ids = sorted_ids
    elif cluster_sizes[0] / num_seqs >= size_threshold:
        filtered_ids = sorted_ids[length_clusters == cluster_index[0]]
    elif sum(cluster_sizes[:2]) / num_seqs >= size_threshold:
        filtered_ids = np.concatenate([sorted_ids[length_clusters == cluster_index[0]], sorted_ids[length_clusters == cluster_index[1]]])
    else:
        filtered_ids = []
    chaff_ids = [gene_id for gene_id in sorted_ids if gene_id not in filtered_ids]
    return filtered_ids, chaff_ids

def update_orthogroup_table(filtered_orthogroups, args):
    og_table = pd.read_csv(f'{args.input_dir}{args.prefix}_orthogroup_presence.tsv', sep='\t', index_col=0, low_memory=False)
    
    for og_id in filtered_orthogroups:
        gene_ids = filtered_orthogroups[og_id]
        og_table.loc[og_id, 'num_seqs'] = len(gene_ids)
        og_table.loc[og_id, 'avg_length'] = pg_utils.calculate_mean_gene_length(gene_ids)

        sag_grouped_ids = pg_utils.group_og_table_genes(gene_ids, og_table)
        og_table.loc[og_id, 'num_cells'] = len(sag_grouped_ids)

        for sag_id in sag_grouped_ids:
            og_table.loc[og_id, sag_id] = ';'.join(sag_grouped_ids[sag_id])
        og_table['seqs_per_cell'] = og_table['num_seqs'] / og_table['num_cells']

    # Remove old clusters
    og_index = list(og_table.index)
    fragment_ids = [og_id for og_id in og_index if og_id not in filtered_orthogroups]
    og_table = og_table.drop(fragment_ids)
    og_table.to_csv(f'{args.output_dir}{args.prefix}_filtered_orthogroup_presence.tsv', sep='\t')

    if args.verbose:
        print(f'Finished updated orthogroup table!\n{og_table}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', default=None, help='FASTA file with cluster sequences.')
    parser.add_argument('-I', '--input_dir', default=None, help='Input dir with FASTA file for each protein cluster.')
    parser.add_argument('-O', '--output_dir', help='Output directory.')
    parser.add_argument('-e', '--ext', default='fna', help='FASTA file extension.')
    parser.add_argument('-p', '--prefix', default='sscs', help='Output files prefix.')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--edge_threshold', default=0.95, help='Minimum similarity to assign edge to graph.')
    args = parser.parse_args()

    start_time = time.time()
    if args.input_dir:
        cluster_files = sorted(glob.glob(f'{args.input_dir}*_????.{args.ext}'))
        filtered_orthogroups = {}
        for f_cluster in cluster_files:
            seq_records = utils.read_fasta(f_cluster)
            orthogroup_id = f_cluster.split('/')[-1].replace(f'.{args.ext}', '')
            if args.verbose:
                print(f'Filtering {orthogroup_id} sequence fragments...')

            sorted_ids, length_clusters = find_length_clusters(seq_records, edge_threshold=args.edge_threshold)
            filtered_seq_ids, fragment_seq_ids = filter_length_clusters(seq_records, sorted_ids, length_clusters)
            if len(filtered_seq_ids) > 0:
                filtered_orthogroups[orthogroup_id] = filtered_seq_ids
                filtered_records = [seq_records[seq_id] for seq_id in filtered_seq_ids]
                SeqIO.write(filtered_records, f'{args.output_dir}{orthogroup_id}.{args.ext}', 'fasta')
            if len(fragment_seq_ids) > 0:
                fragment_records = [seq_records[seq_id] for seq_id in fragment_seq_ids]
                SeqIO.write(fragment_records, f'{args.input_dir}{orthogroup_id}.fragments.{args.ext}', 'fasta')

        update_orthogroup_table(filtered_orthogroups, args)

    else:
        print('Both -i and -I options not set. Please give either input file (-i) or input directory (-I).')

    run_time = utils.timeit(start_time)
    if args.verbose:
        print(f'Done! Total pg_filter_fragments.py runtime: {run_time}')
