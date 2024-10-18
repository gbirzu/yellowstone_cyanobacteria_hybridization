import argparse
import numpy as np
import pandas as pd
import pickle
import glob
import copy
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.cluster.hierarchy as hclust
import scipy.spatial.distance as distance
import scipy.sparse as sparse
import natsort
import string
import pangenome_utils as pg_utils
#import ete3
import matplotlib.patches as patches
import networkx as nx
from pangenome_utils import PangenomeMap
from networkx.drawing.nx_pydot import graphviz_layout
from matplotlib.sankey import Sankey
from matplotlib.path import Path
from alignment_tools import SequenceTable
from calculate_linkage_disequilibria import calculate_rsquared
from metadata_map import MetadataMap
from seq_processing_utils import AlleleFreqTable
from pg_filter_fragments import find_length_clusters
from plot_utils import *
from Bio import Phylo
from Bio import AlignIO
from Bio import SearchIO
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

    

def find_strongly_linked_sites(snps_aln, seed_idx=None, r2_c=0.999, L_c=2, f_c=0, x0=0):
    r2_matrix = calculate_rsquared(snps_aln)

    if seed_idx:
        candidate_sites = seed_idx
    else:
        candidate_sites = list(np.arange(1, r2_matrix.shape[0] - 1))

    # Find linked regions
    i = 0
    linked_segments = []
    haplotype_freqs = []
    r2_avg = []
    while i < len(candidate_sites):
        test_site_idx = candidate_sites[i]
        linkage_results = get_linked_region(r2_matrix, test_site_idx, r2_c)

        if linkage_results['is_linked']:
            j_left, j_right = linkage_results['segment_edges']

            # Check segment length
            if linkage_results['length'] >= L_c:
                # Check haploytpe frequencies
                segment_aln = np.array(snps_aln)[:, j_left:j_right + 1]
                haplotypes, counts = np.unique(segment_aln, return_counts=True, axis=0)
                f0 = min(counts) / sum(counts)
                
                if f0 >= f_c:
                    linked_segments.append(list(linkage_results['segment_edges']))
                    haplotype_freqs.append(f0)
                    r2_avg.append(np.mean(r2_matrix[test_site_idx, j_left:j_right + 1]))

            # Move to next possible block
            if j_right in candidate_sites:
                i = candidate_sites.index(j_right) + 1
            else:
                i = len(candidate_sites)
        else:
            i += 1

    return {'segments':np.array(linked_segments), 'haplotype_frequencies':np.array(haplotype_freqs), 'segment_r2':np.array(r2_avg)}


def get_linked_region(r2_matrix, i, r2_c):
    j_left = i
    while r2_matrix[i, j_left - 1] >= r2_c:
        j_left -= 1
        if j_left == 0:
            break

    j_right = i
    while r2_matrix[i, j_right + 1] >= r2_c:
        j_right += 1
        if j_right == len(r2_matrix) - 1:
            break
    
    segment_length = j_right - j_left + 1
    found_linked_segment = (j_left < i) or (j_right > i)

    return {'is_linked':found_linked_segment, 'segment_edges':(j_left, j_right), 'length':segment_length}



def make_sag_gene_id_map(data_tables):
    gene_id_map = {'CYA':'CP000239', 'CYB':'CP000240'}
    for sag_id in data_tables:
        gene_table = data_tables[sag_id]['genes']
        for gene_id in gene_table.index:
            gene_id_map[gene_id] = sag_id
        id_stem = gene_id.split('_')[0]
        gene_id_map[id_stem] = sag_id
    return gene_id_map

def get_gene_sag_id(gene_id, gene_id_map):
    stem = gene_id.split('_')[0]
    return gene_id_map[stem]

def read_roary_clusters(f_in):
    cluster_dict = {}
    with open(f_in, 'r') as handle:
        for line in handle.readlines():
            head, tail = line.split(':')
            cluster_id = head.strip()
            gene_ids = tail.strip().split('\t')
            cluster_dict[cluster_id] = gene_ids
    return cluster_dict

def read_roary_gene_presence(f_in):
    gene_presence_absence = pd.read_csv(f_in, sep=',', index_col=0)
    return gene_presence_absence

def make_locus_map(gene_clusters, gene_id_map, sag_ids):
    cluster_ids = natsort.natsorted(list(gene_clusters.keys()))
    locus_map = pd.DataFrame(index=cluster_ids, columns=sag_ids)
    for cluster_id in cluster_ids:
        locus_map.loc[cluster_id, :] = [[] for sag_id in sag_ids]
        for gene_id in gene_clusters[cluster_id]:
            if gene_id in gene_id_map:
                locus_map.at[cluster_id, get_gene_sag_id(gene_id, gene_id_map)].append(gene_id)
    return locus_map

def calculate_gene_copy_numbers(locus_map):
    gene_copy_numbers = pd.DataFrame(index=locus_map.index, columns=locus_map.columns)
    for sag_id in gene_copy_numbers.columns:
        gene_copy_numbers[sag_id] = locus_map[sag_id].str.len()
    return gene_copy_numbers

def calculate_custom_pipeline_metrics():
    input_dir = '../results/tests/pangenome_construction/n4/'
    # Pre-clustering
    f_blastp = f'{input_dir}_blastp_results/sag_protein_seqs_blast_results.tab'
    blastp_results = seq_utils.read_blast_results(f_blastp, extra_columns=['qseqlen', 'sseqlen'])
    blastp_results = calculate_normalized_bitscore(blastp_results)

    print(blastp_results)
    '''
    # Post-clustering
    tree_files = glob.glob(f'{input_dir}_aln_results/*.nwk')
    cluster_ids = sorted([f_tree.split('/')[-1].replace('_aln.nwk', '') for f_tree in tree_files])
    cluster_stats = pd.DataFrame(index=cluster_ids, columns=['n', 'D', 'pi_aa', 'L_aa'])

    for cluster_id in cluster_ids:
        tree = Phylo.read(f'{input_dir}_aln_results/{cluster_id}_aln.nwk', 'newick')
        n = tree.count_terminals()
        cluster_stats.loc[cluster_id, 'n'] = n
        depths = tree.depths()
        cluster_stats.loc[cluster_id, 'D'] = np.max(list(depths.values()))
        leafs = tree.get_terminals()
        pi_aa = []
        for j in range(n):
            for i in range(j):
                pi_aa.append(tree.distance(leafs[i], leafs[j]))
        cluster_stats.at[cluster_id, 'pi_aa'] = pi_aa

        seq_records = SeqIO.parse(f'{input_dir}_mcl_results/{cluster_id}.faa', 'fasta')
        lengths = []
        for seq_rec in seq_records:
            lengths.append(len(seq_rec))
        cluster_stats.at[cluster_id, 'L_aa'] = lengths

    pickle.dump({'blastp':blastp_results, 'cluster_stats':cluster_stats}, open(f'{input_dir}clustering_eval_metrics.dat', 'wb'))
    '''

def test_custom_pipeline():
    input_file = '../results/tests/pangenome_construction/n4/clustering_eval_metrics.dat'
    output_dir = '../figures/analysis/tests/gene_clustering/'
    prefix = 'n4'
    clustering_metrics = pickle.load(open(input_file, 'rb'))

    blastp_results = clustering_metrics['blastp']
    for metric in ['pident', 'bitscore', 'normbitscore', 'mismatch']:
        plot_distribution(blastp_results.loc[blastp_results['qseqid'] != blastp_results['sseqid'], metric], xlabel=metric, savefig=f'{output_dir}{prefix}_{metric}_noselfhits.pdf')

    ax = plot_correlation(blastp_results['pident'], 100 * blastp_results['normbitscore'], xlabel='pident', ylabel=r'BS/BS_self')
    ax.plot([0, 100], [0, 100], c='r')
    plt.savefig(f'{output_dir}{prefix}_pident_vs_normbitscore.pdf')

    cluster_stats = clustering_metrics['cluster_stats']
    print(cluster_stats)
    num_clusters = 2295
    plot_rank_size_distribution(cluster_stats, num_clusters, savefig=f'{output_dir}{prefix}_cluster_rank_size.pdf')

    plot_distribution(cluster_stats['D'], xlabel='cluster diameter, D', bins=50, savefig=f'{output_dir}{prefix}_cluster_diameter.pdf')

    pi_w = [np.array(pi_aa) for pi_aa in cluster_stats['pi_aa']]
    x_w = np.concatenate([pi_aa / np.mean(pi_aa) for pi_aa in pi_w if (len(pi_aa) > 1 and np.mean(pi_aa) > 0 and np.mean(pi_aa) < 0.5)])

    bins = 50
    plot_distribution(np.concatenate(pi_w), xlabel=r'$\pi_{aa}$', bins=bins, savefig=f'{output_dir}{prefix}_cluster_divergence_distribution.pdf')
    plot_distribution(x_w, xlabel=r'$\frac{\pi_{aa}}{\langle \pi_{aa} \rangle _c}$', bins=bins, savefig=f'{output_dir}{prefix}_exp_normalized_cluster_divergences_bins{bins}.pdf')

    l_w = [np.array(l_aa) for l_aa in cluster_stats['L_aa']]
    z_w = np.concatenate([(l_aa - np.mean(l_aa)) / np.std(l_aa) for l_aa in l_w if (len(l_aa) > 2 and np.std(l_aa) > 0)])

    bins = 50
    plot_distribution(np.concatenate(l_w), xlabel=r'$L_{aa}$', bins=bins, savefig=f'{output_dir}{prefix}_cluster_seqlen_distribution.pdf')
    plot_distribution(z_w, xlabel=r'$\frac{L_{aa} - \langle L_{aa} \rangle}{\sqrt{Var(L_{aa})}}$', bins=bins, savefig=f'{output_dir}{prefix}_cluster_gaussian_norm_seqlen_distribution.pdf')

    clusters = []
    with open('../results/tests/pangenome_construction/n4/_mcl_results/sag_protein_clusters.tsv', 'r') as in_handle:
        for line in in_handle.readlines():
            clusters.append(line.strip().split('\t'))

    d_b = []
    for cluster in clusters:
        pident_b = blastp_results.loc[blastp_results['qseqid'].isin(cluster) * (~blastp_results['sseqid'].isin(cluster)), 'pident']
        if len(pident_b) > 0:
            d_b.append(pident_b)
    d_b = np.concatenate(d_b)
    plot_distribution(d_b, xlabel=r'$\pi_{aa}$', bins=bins, savefig=f'{output_dir}{prefix}_between_cluster_divergence_distribution.pdf')


def calculate_normalized_bitscore(blastp_results):
    query_ids = np.unique(blastp_results['qseqid'])
    blastp_results['normbitscore'] = 0
    for query_id in query_ids:
        BS0 = blastp_results.loc[blastp_results['qseqid'] == query_id, :].loc[blastp_results['sseqid'] == query_id, 'bitscore'].values[0]
        blastp_results.loc[blastp_results['qseqid'] == query_id, 'normbitscore'] = blastp_results.loc[blastp_results['qseqid'] == query_id, 'bitscore'] / BS0
    return blastp_results

def plot_distribution(values, xlabel='', xscale='linear', bins=100, savefig=None):
    fig = plt.figure(figsize=(double_col_width, 0.9 * single_col_width))
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel(xlabel)
    ax1.set_xscale(xscale)
    ax1.set_ylabel('counts')
    ax1.set_yscale('log')
    ax1.hist(values, bins=bins)

    ax2 = fig.add_subplot(122)
    ax2.set_xlabel(xlabel)
    ax2.set_xscale('log')
    ax2.set_ylabel('CCDF')
    ax2.set_yscale('log')

    x, cuml = utils.cumulative_distribution(values, epsilon=0.1)
    ax2.plot(x, 1 - cuml)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    else:
        return ax1, ax2

def plot_correlation(x, y, xlabel='', ylabel='', savefig=None):
    fig = plt.figure(figsize=(single_col_width, 0.9 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.scatter(x, y, s=10, lw=0.5, fc='none', ec='k')
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    else:
        return ax

def plot_rank_size_distribution(cluster_stats, num_clusters, savefig=None):
    fig = plt.figure(figsize=(single_col_width, 0.9 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('rank')
    ax.set_xscale('log')
    ax.set_ylabel('cluster size')

    cluster_sizes = list(cluster_stats['n']) + [1] * (num_clusters - len(cluster_stats))
    x, cuml = utils.cumulative_distribution(cluster_sizes, normalized=False)
    size = x
    rank = max(cuml) - cuml + 1
    ax.plot(rank, size)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    else:
        return ax

def test_all():
    '''
    f_data_tables = '../results/tests/closely_related_sag_data_tables.dat'
    data_tables = pickle.load(open(f_data_tables, 'rb'))
    gene_id_map = make_sag_gene_id_map(data_tables)

    for ident_c in [90, 70, 50, 30]:
        print(ident_c)
        gene_clusters = read_roary_clusters(f'../results/tests/closely_related_sags_pangenome/i{ident_c}/clustered_proteins')
        if ident_c == 50:
            sag_ids = ['CP000239', 'CP000240'] + list(data_tables.keys())
        else:
            sag_ids = list(data_tables.keys())
        locus_map = make_locus_map(gene_clusters, gene_id_map, sag_ids)
        gene_copy_numbers = calculate_gene_copy_numbers(locus_map)
        print(gene_copy_numbers)
        for sag_id in gene_copy_numbers.columns:
            print(sag_id, np.unique(gene_copy_numbers[sag_id], return_counts=True))
        print(gene_copy_numbers.loc[gene_copy_numbers.sum(axis=1) > 1, :])
        print(gene_copy_numbers.loc[gene_copy_numbers.sum(axis=1) == 1, :])
        print(gene_copy_numbers.loc[gene_copy_numbers.sum(axis=1) == 5, :])
        #sags_seq_table = SequenceTable(f_data_tables)

        print('\n\n')

    '''
    ident_c = 95
    #pairwise_divergences = pickle.load(open(f'../results/tests/closely_related_sags_pangenome/i{ident_c}/closely_related_sags_i{ident_c}_locus_divergences.dat', 'rb'))
    pairwise_divergences = pickle.load(open(f'../results/tests/closely_related_sags_pangenome/i{ident_c}/closely_related_sags_i{ident_c}_synbp_locus_divergences.dat', 'rb'))
    print(pairwise_divergences)
    pN_aggregate = []
    pN_max_close = []
    pN_max_all = []
    pS_aggregate = []
    pS_max_close = []
    pS_max_all = []
    for locus in pairwise_divergences:
        pN = pairwise_divergences[locus]['pN']
        sag_ids = list(pN.index)
        close_cells = [sag_id for sag_id in sag_ids if 'CY' not in sag_id]
        pN_aggregate.append(utils.get_matrix_triangle_values(pN.values, k=1))
        pN_max_close.append(np.max(pN.loc[close_cells, close_cells]))
        pN_max_all.append(np.max(pN))
        pS = pairwise_divergences[locus]['pS']
        pS_aggregate.append(utils.get_matrix_triangle_values(pS.values, k=1))
        pS_max_close.append(np.max(pS.loc[close_cells, close_cells]))
        pS_max_all.append(np.max(pS))

    pN_aggregate = np.concatenate(pN_aggregate)
    pN_max_close = np.concatenate(pN_max_close)
    pN_max_all = np.concatenate(pN_max_all)
    pS_aggregate = np.concatenate(pS_aggregate)
    pS_max_close = np.concatenate(pS_max_close)
    pS_max_all = np.concatenate(pS_max_all)

    pi_est = 7 * pN_aggregate / 9 + 2 * pS_aggregate / 9
    pi_max_close = 7 * pN_max_close / 9 + 2 * pS_max_close / 9
    pi_max_all = 7 * pN_max_all / 9 + 2 * pS_max_all / 9

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('pairwise diversity')
    ax.set_ylabel('histogram')
    ax.set_yscale('log')
    ax.hist(pi_est, bins=50, alpha=0.5, label='$\pi$')
    ax.hist(pN_aggregate, bins=50, alpha=0.5, label='$\pi_N$')
    ax.hist(pS_aggregate, bins=50, alpha=0.5, label='$\pi_S$')
    ax.legend(fontsize=8)
    plt.tight_layout()
    #plt.savefig(f'../figures/analysis/tests/gene_clustering/roary_i{ident_c}_cluster_pairwise_divergence.pdf')
    plt.savefig(f'../figures/analysis/tests/gene_clustering/roary_i{ident_c}_synbp_cluster_pairwise_divergence.pdf')

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('pairwise diversity')
    ax.set_ylabel('histogram')
    ax.set_yscale('log')
    ax.hist(pi_max_close, bins=50, alpha=0.6, label='$r_{close}$')
    ax.hist(pi_max_all, bins=50, alpha=0.6, label='$r_{all}$')
    ax.hist(pN_max_all, bins=50, alpha=0.6, label='$r_{all}^{N}$')
    ax.legend(fontsize=8)
    plt.tight_layout()
    #plt.savefig(f'../figures/analysis/tests/gene_clustering/roary_i{ident_c}_cluster_close_cell_diameter.pdf')
    plt.savefig(f'../figures/analysis/tests/gene_clustering/roary_i{ident_c}_synbp_cluster_close_cell_diameter.pdf')

    '''
    #pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth', 200)

    ident_c = 50
    #gene_presence = read_roary_gene_presence(f'../results/tests/closely_related_sags_pangenome/i{ident_c}/gene_presence_absence.csv')
    gene_presence = read_roary_gene_presence(f'../results/tests/closely_related_sags_pangenome/i{ident_c}_prokka/gene_presence_absence.csv')
    #sag_ids = ['CP000239', 'CP000240'] + list(data_tables.keys())
    sag_columns = ['CP000239', 'CP000240', 'UncmicMRedA02K13_3_FD_prokka', 'UncmicMRedA02N14_2_FD_prokka', 'UncmicMuRedA1H13_FD_prokka']
    print_columns = ['Annotation', 'No. sequences'] + sag_columns
    print(len(gene_presence))
    print(np.unique(gene_presence['No. sequences'], return_counts=True))
    print(gene_presence.loc[gene_presence['CP000240'] == 'CYB_2287.p01', print_columns])
    print(gene_presence.loc[gene_presence['UncmicMRedA02K13_3_FD_prokka'] == 'LPPNFIOK_00119', print_columns])
    print(gene_presence.loc[gene_presence['UncmicMRedA02N14_2_FD_prokka'] == 'NMBNCMHM_00975', print_columns])
    print(gene_presence.loc[gene_presence['UncmicMRedA02N14_2_FD_prokka'] == 'NMBNCMHM_01342', print_columns])
    print(gene_presence.loc[gene_presence['UncmicMRedA02N14_2_FD_prokka'] == 'NMBNCMHM_01042', print_columns])


    gene_presence_jgi = read_roary_gene_presence(f'../results/tests/closely_related_sags_pangenome/i{ident_c}/gene_presence_absence.csv')
    sag_columns = ['CP000239', 'CP000240', 'UncmicMRedA02K13_3_FD_jgi_annotation', 'UncmicMRedA02N14_2_FD_jgi_annotation', 'UncmicMuRedA1H13_FD_jgi_annotation']
    print_columns = ['Annotation', 'No. sequences'] + sag_columns
    #print(gene_presence_jgi[sag_columns].notna().prod(axis=1).astype(bool))
    jgi_filtered = gene_presence_jgi.loc[gene_presence_jgi[sag_columns].notna().sum(axis=1).astype(bool), :]
    print(len(jgi_filtered))
    print(np.unique(jgi_filtered['No. sequences'], return_counts=True))
    print(jgi_filtered.loc[jgi_filtered['No. sequences'] > 7, print_columns])
    print(jgi_filtered.loc[jgi_filtered['CP000240'] == 'CYB_2287.p01', print_columns])
    print(jgi_filtered.loc[jgi_filtered['UncmicMuRedA1H13_FD_jgi_annotation'] == 'Ga0374792_028_7348_7941', print_columns])
    #sag_ids = ['CP000239', 'CP000240'] + list(data_tables.keys())
    #sag_id = 'UncmicMRedA02E12_2_FD'
    #sag_column = sag_id + '_jgi_annotation'

    #print(gene_presence.loc[gene_presence[sag_column] == 'Ga0393552_003_23510_24046', :])
    #print(gene_presence.loc[gene_presence[sag_column] == 'Ga0393552_003_24871_25407', :])
    '''

    singleton_lengths = pickle.load(open('../results/tests/closely_related_sags_pangenome/i95/singleton_lengths.dat', 'rb'))
    print(singleton_lengths)
    gene_lengths, length_bin_counts = utils.sorted_unique(singleton_lengths.values)
    print(gene_lengths, length_bin_counts)
    print(np.sum(length_bin_counts[length_bin_counts > 1]), np.sum(length_bin_counts))
    x_multiple_gene_bins = gene_lengths[length_bin_counts > 1]

    bins = 100
    fig = plt.figure(figsize=(single_col_width, 0.9 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('gene length')
    ax.set_ylabel('counts')
    ax.hist(singleton_lengths.values, bins=bins)
    plt.tight_layout()
    plt.savefig('../figures/analysis/tests/gene_clustering/roary_i95_singleton_lengths.pdf')

    fig = plt.figure(figsize=(single_col_width, 0.9 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('gene length')
    ax.set_xscale('log')
    ax.set_ylabel('counts')
    x_bins = np.geomspace(10, 7000, bins)
    ax.hist(singleton_lengths.values, bins=x_bins)
    plt.tight_layout()
    plt.savefig('../figures/analysis/tests/gene_clustering/roary_i95_singleton_lengths_logx.pdf')


def test_orthogroup_divergences():
    print('Making othogroup distance matrices...')
    pangenome_dir = '../results/tests/pangenome_construction/n4/'
    figures_dir = '../figures/analysis/tests/gene_clustering/clustering_methods/'

    clustering_metrics = pickle.load(open(f'{pangenome_dir}clustering_eval_metrics.dat', 'rb'))
    blastp_results = clustering_metrics['blastp']

    # Analyze MCL gene clusters
    cluster_dicts, idx = pg_utils.process_mcl_clusters(f'{pangenome_dir}_mcl_results/sag_protein_clusters.tsv', idx=1)
    cluster_sample = [cluster_dicts[cluster_id] for cluster_id in cluster_dicts if len(cluster_dicts[cluster_id]) > 3]
    gene_ids = np.concatenate(cluster_sample)
    print(gene_ids, len(gene_ids))

    calculate_orthogroups_pdist(blastp_results, gene_ids, output_file=f'{pangenome_dir}mcl_orthogroup_normbitscore_matrix.dat')
    pdist_df = pickle.load(open(f'{pangenome_dir}mcl_orthogroup_normbitscore_matrix.dat', 'rb'))
    plot_pdist_clustermap(-1 * pdist_df.apply(np.log10), grid=False, savefig=f'{figures_dir}mcl_orthogroups_neg_log_normbitscore_matrix.pdf')
    print('\n\n')

    # Analyze RBH gene clusters
    rbh_pangenome_dir = '../results/tests/pangenome_construction/n4_rbh/'
    cluster_dicts, idx = pg_utils.process_mcl_clusters(f'{rbh_pangenome_dir}sag_protein_clusters.tsv', idx=1)
    cluster_sample = [cluster_dicts[cluster_id] for cluster_id in cluster_dicts if len(cluster_dicts[cluster_id]) > 3]
    gene_ids = np.concatenate(cluster_sample)
    print(gene_ids, len(gene_ids))

    calculate_orthogroups_pdist(blastp_results, gene_ids, output_file=f'{rbh_pangenome_dir}rbh_orthogroup_normbitscore_matrix.dat')
    pdist_df = pickle.load(open(f'{rbh_pangenome_dir}rbh_orthogroup_normbitscore_matrix.dat', 'rb'))
    plot_pdist_clustermap(-1 * pdist_df.apply(np.log10), grid=False, savefig=f'{figures_dir}rbh_orthogroups_neg_log_normbitscore_matrix.pdf')


def calculate_orthogroups_pdist(blastp_results, gene_ids, output_file=None):
    pdist_df = pd.DataFrame(np.nan, index=gene_ids, columns=gene_ids)

    for gene_id, gene_pdist in pdist_df.iterrows():
        gene_blastp_results = blastp_results.reindex(blastp_results.loc[blastp_results['qseqid'] == gene_id, :].index)

        # Get unique indices
        index_values, index_counts = utils.sorted_unique(gene_blastp_results['sseqid'])
        multihit_idx = index_values[index_counts > 1]
        gene_index = list(gene_blastp_results.loc[gene_blastp_results['sseqid'].isin(index_values[index_counts == 1]), :].index)
        for idx in multihit_idx:
            gene_index.append(gene_blastp_results.loc[gene_blastp_results['sseqid'] == idx, :].iloc[0, :].name)

        gene_blastp_results = gene_blastp_results.loc[gene_index, :].set_index('sseqid')
        gene_idx = [col for col in pdist_df.columns if col in (gene_blastp_results.index)]
        pdist_df.loc[gene_id, gene_idx] = gene_blastp_results.loc[gene_idx, 'normbitscore'].values

    if output_file is not None:
        pickle.dump(pdist_df, open(output_file, 'wb'))
    else:
        return pdist_df


def test_locus_seqs_clustering(random_seed=12345):
    data_dir = '../results/tests/gene_clustering/'
    figures_dir = '../figures/analysis/tests/gene_clustering/clustering_methods/'
    locus = 'CYB_2798'

    #make_test_data(data_dir, locus)

    #aln = AlignIO.read(f'{data_dir}{locus}_codon_aln_MUSCLE.fasta', 'fasta')
    #pdist_df = pickle.load(open(f'{data_dir}raw_nucl_snp_pdist.dat', 'rb'))
    #plot_pdist_clustermap(pdist_df, savefig=f'{figures_dir}raw_alignment_snp_distance.pdf')
    #print(aln)
    #print(pdist_df)
    #print('\n\n')

    #make_clustered_pdist_figures(pdist_df, figures_dir, prefix='raw_alignment')

    # Gap-filtered alignments
    filtered_pdist_df = pickle.load(open(f'{data_dir}filtered_nucl_snp_pdist.dat', 'rb'))
    #make_clustered_pdist_figures(filtered_pdist_df, figures_dir, linkage_methods=['ward', 'weighted'], prefix='filtered_alignment')

    metadata = MetadataMap()
    sag_ids = np.array(list(filtered_pdist_df.index))
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    species_sorted_sags['syna'] = species_sorted_sags.pop('A')
    species_sorted_sags['synbp'] = species_sorted_sags.pop('Bp')

    '''
    # MCL clustering
    pg_utils.write_protein_graph(f'{data_dir}CYB_2798_blast_results.tab', f'{data_dir}CYB_2798.abc', weights='bitscore')
    make_finescale_seq_graph(f'{data_dir}CYB_2798_blast_results.tab', f'{data_dir}CYB_2798_finescale.abc')

    # RBH clustering

    rbh_graph = make_rbh_graph(f'{data_dir}CYB_2798_blast_results.tab', sag_ids)
    num_clusters, cluster_labels = sparse.csgraph.connected_components(rbh_graph)
    unique_labels, label_counts = utils.sorted_unique(cluster_labels)
    print(cluster_labels)
    print(unique_labels, label_counts)
    print('\n\n')

    sort_idx = []
    idx = np.arange(len(cluster_labels))
    for label in unique_labels:
        cluster_idx = idx[cluster_labels == label]
        sort_idx += list(cluster_idx)

    sorted_sag_ids = sag_ids[sort_idx]
    plot_pdist_clustermap(filtered_pdist_df, sags_sorting=sorted_sag_ids, species_sort=species_sorted_sags, savefig=f'{figures_dir}filtered_alignment_seq_rbh_ccomponents.pdf')

    rbh_distance = rbh_graph.toarray()
    diag_idx = np.arange(len(rbh_distance))
    rbh_distance[diag_idx, diag_idx] = 1
    rbh_distance = 1 - rbh_distance
    rbh_linkage = hclust.linkage(distance.squareform(rbh_distance), method='single', optimal_ordering=True)
    dn = hclust.dendrogram(rbh_linkage)
    sags_sorting = list(sag_ids[dn['leaves']])
    plot_pdist_clustermap(filtered_pdist_df, linkage=rbh_linkage, sags_sorting=sags_sorting, species_sort=species_sorted_sags, savefig=f'{figures_dir}filtered_alignment_seq_rbh_single-linkage.pdf')

    # FastTree NJ clustering
    tree = ete3.PhyloTree(f'{data_dir}CYB_2798_codon_aln_MUSCLE.nwk', format=0)
    postordered_leaves = [leaf.name for leaf in tree.get_leaves()]
    plot_pdist_clustermap(filtered_pdist_df, sags_sorting=postordered_leaves, species_sort=species_sorted_sags, savefig=f'{figures_dir}filtered_alignment_nj_clustering.pdf')

    plot_tree(tree, savefig=f'{figures_dir}filtered_aln_nj_tree.pdf')

    synbp_tree = ete3.PhyloTree(f'{data_dir}CYB_2798_synbp_seqs_aln.nwk', format=0)
    plot_tree(synbp_tree, savefig=f'{figures_dir}filtered_aln_synbp_nj_tree.pdf')

    # Use average linkage to look for sub-species clusters
    # Look in Syn. B' since it's more diverse
    synbp_sag_ids = np.array(species_sorted_sags['synbp'])
    synbp_pdist_df = filtered_pdist_df.reindex(index=synbp_sag_ids, columns=synbp_sag_ids)
    print(synbp_pdist_df)

    linkage_method = 'average'
    pdist = distance.squareform(synbp_pdist_df.values)
    Z = hclust.linkage(pdist, method=linkage_method, optimal_ordering=True)

    clusters = hclust.fcluster(Z, 4, criterion='distance')
    print(clusters)

    dn = hclust.dendrogram(Z, no_plot=True)
    sags_sorting = list(synbp_sag_ids[dn['leaves']])
    plot_pdist_clustermap(synbp_pdist_df, linkage=Z, sags_sorting=sags_sorting, savefig=f'{figures_dir}synbp_seq_pdist_{linkage_method}-linkage.pdf')

    print(sags_sorting)
    f_aln = f'{data_dir}CYB_2798_synbp_seqs_aln.fna'
    aln = AlignIO.read(f_aln, 'fasta')
    print(aln)
    aln_nogaps = align_utils.filter_alignment_gap_columns(aln, max_gap_frequency=0.1)
    print(aln_nogaps)
    f_aln_nogaps = f'{data_dir}CYB_2798_synbp_nogaps_aln.fna'
    AlignIO.write(aln_nogaps, f_aln_nogaps, 'fasta')
    aln_snps = seq_utils.get_snps(aln_nogaps)
    f_aln_snps = f'{data_dir}CYB_2798_synbp_snps_aln.fna'
    AlignIO.write(aln_snps, f_aln_snps, 'fasta')

    snps_arr = np.array(aln_snps)
    sorted_snp_arr = snps_arr[dn['leaves']]
    sag_idx = []
    for rec in aln_snps:
        sag_idx.append(sags_sorting.index(rec.id))
    aln_snps_sorted = []
    for i in sag_idx:
        aln_snps_sorted.append(aln_snps[i])
    aln_snps_sorted = MultipleSeqAlignment(aln_snps_sorted)

    print(sags_sorting)
    aln_sags = []
    for rec in aln_nogaps:
        aln_sags.append(rec.id)
    sorted_idx = []
    for sag_id in sags_sorting:
        if sag_id in aln_sags:
            sorted_idx.append(aln_sags.index(sag_id))
    print('\n\n')
    aln_nogaps_sorted = []
    for i in sorted_idx:
        aln_nogaps_sorted.append(aln_nogaps[i])
        print(aln_nogaps[i].id)
    aln_nogaps_sorted = MultipleSeqAlignment(aln_nogaps_sorted)

    draw_alignment_sequences(aln_nogaps, savefig=f'{figures_dir}synbp_seq_alignment.pdf')
    draw_alignment_sequences(aln_nogaps_sorted, savefig=f'{figures_dir}synbp_seq_alignment_clustered.pdf')
    '''


    # Plot alignment for all samples to check for possible interspecies recombination
    print(filtered_pdist_df)

    linkage_method = 'average'
    pdist = distance.squareform(filtered_pdist_df.values)
    Z = hclust.linkage(pdist, method=linkage_method, optimal_ordering=True)
    clusters = hclust.fcluster(Z, 4, criterion='distance')
    print(clusters)

    dn = hclust.dendrogram(Z, no_plot=True)
    sags_sorting = list(sag_ids[dn['leaves']])


    f_aln = f'{data_dir}CYB_2798_codon_aln_MUSCLE.fasta'
    aln = AlignIO.read(f_aln, 'fasta')
    aln_sags = []
    for rec in aln:
        aln_sags.append(rec.id)
    sorted_idx = []
    for sag_id in sags_sorting:
        if sag_id in aln_sags:
            sorted_idx.append(aln_sags.index(sag_id))
    aln_sorted = []
    for i in sorted_idx:
        aln_sorted.append(aln[i])
    aln_sorted = MultipleSeqAlignment(aln_sorted)

    draw_alignment_sequences(aln_sorted, savefig=f'{figures_dir}CYB_2798_synecho_alignment.pdf')


    '''
    # Analyze simulations data
    coalescent_pdist_df = pickle.load(open(f'{data_dir}coalescent_seq_pdist.dat', 'rb'))
    sample_ids = np.array(list(coalescent_pdist_df.index))
    pdist = distance.squareform(coalescent_pdist_df.values)
    Z = hclust.linkage(pdist, method=linkage_method, optimal_ordering=True)
    print(Z)

    clusters = hclust.fcluster(Z, 6.5, criterion='distance')
    print(clusters)

    dn = hclust.dendrogram(Z, no_plot=True)
    sags_sorting = list(sample_ids[dn['leaves']])
    plot_pdist_clustermap(coalescent_pdist_df, linkage=Z, sags_sorting=sags_sorting, savefig=f'{figures_dir}coalescent_seq_pdist_{linkage_method}-linkage.pdf')

    coalescent_aln = AlignIO.read('../results/tests/gene_clustering/coalescent_aln.fna', 'fasta')
    print(coalescent_aln)
    coalescent_sorted = sort_alnmt(coalescent_aln, sags_sorting)
    print(coalescent_sorted)
    draw_alignment_sequences(coalescent_sorted, savefig=f'{figures_dir}coalescent_seq_alignment_clustered.pdf')
    '''


def make_rbh_graph(in_blast_results, sag_ids):
    blast_results = seq_utils.read_blast_results(in_blast_results, extra_columns=['qlen', 'slen'])
    rbh_results = pg_utils.filter_reciprocal_best_hits(blast_results)
    print(rbh_results)

    num_seqs = len(sag_ids)
    adjacency_matrix = sparse.csr_matrix((num_seqs, num_seqs))
    query_idx, hit_idx = map_gene_index(rbh_results, sag_ids)
    adjacency_matrix[query_idx, hit_idx] = 1
    return adjacency_matrix

def sort_alnmt(aln, sorted_ids):
    aln_sags = []
    for rec in aln:
        aln_sags.append(rec.id)

    sorted_idx = []
    for sag_id in sorted_ids:
        if sag_id in aln_sags:
            sorted_idx.append(aln_sags.index(sag_id))

    aln_sorted = []
    for i in sorted_idx:
        aln_sorted.append(aln[i])

    return MultipleSeqAlignment(aln_sorted)


def map_gene_index(rbh_results, gene_ids):
    idx_map = {}
    for i, gene_id in enumerate(gene_ids):
        idx_map[gene_id] = i

    query_idx = rbh_results['qseqid'].map(idx_map)
    hit_idx = rbh_results['sseqid'].map(idx_map)
    return query_idx, hit_idx


def make_finescale_seq_graph(in_blast_results, output_file):
    blast_results = seq_utils.read_blast_results(in_blast_results, extra_columns=['qlen', 'slen'])
    print(blast_results)

    pseudocount = 0.5 / blast_results['qlen']
    blast_results['pseudocount'] = 0.5 / blast_results['qlen']
    blast_results['edge_weight'] = -np.log10(blast_results['pseudocount'])
    blast_results.loc[blast_results['pident'] != 100, 'edge_weight'] = -np.log10(1 - blast_results['pident'] / 100)

    with open(output_file, 'w') as out_handle:
        for hit in blast_results[['qseqid', 'sseqid', 'edge_weight']].values:
            out_handle.write('\t'.join(hit.astype(str)) + '\n')

def make_test_data(data_dir, locus, random_seed=12345):
    metadata = MetadataMap()

    nucl_seqs = utils.read_fasta(f'{data_dir}{locus}_seqs.fna')
    prot_seqs = seq_utils.translate_seqs_dict(nucl_seqs)
    seq_utils.write_seqs_dict(prot_seqs, f'{data_dir}{locus}_seqs.faa')

    sag_ids = list(nucl_seqs.keys())
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    print(species_sorted_sags)

    synbp_nucl_seqs = {}
    for sag_id in species_sorted_sags['Bp']:
        synbp_nucl_seqs[sag_id] = nucl_seqs[sag_id]
    seq_utils.write_seqs_dict(synbp_nucl_seqs, f'{data_dir}{locus}_synbp_seqs.fna')

    aln = AlignIO.read(f'{data_dir}{locus}_codon_aln_MUSCLE.fasta', 'fasta')

    # Shuffle sequence order
    aln_records = [record for record in aln]
    np.random.seed(random_seed)
    idx_shuffled = np.random.permutation(len(aln))
    aln_shuffled = MultipleSeqAlignment([aln_records[i] for i in idx_shuffled])

    # Calculate pairwise SNP distances
    pdist_df = align_utils.calculate_pairwise_snp_distance(aln_shuffled)
    pickle.dump(pdist_df, open(f'{data_dir}raw_nucl_snp_pdist.dat', 'wb'))

    filtered_aln = align_utils.filter_alignment_gap_columns(aln_shuffled)
    filtered_pdist_df = align_utils.calculate_pairwise_snp_distance(filtered_aln)
    pickle.dump(filtered_pdist_df, open(f'{data_dir}filtered_nucl_snp_pdist.dat', 'wb'))

    coalescent_aln = AlignIO.read(f'{data_dir}coalescent_simulations.fna', 'fasta')
    coalescent_pdist_df = align_utils.calculate_pairwise_snp_distance(coalescent_aln)
    pickle.dump(coalescent_pdist_df, open(f'{data_dir}coalescent_seq_pdist.dat', 'wb'))


def make_clustered_pdist_figures(pdist_df, figures_dir,
        linkage_methods=['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'],
        prefix='alignment'):
    metadata = MetadataMap()
    sag_ids = np.array(list(pdist_df.index))
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    species_sorted_sags['syna'] = species_sorted_sags.pop('A')
    species_sorted_sags['synbp'] = species_sorted_sags.pop('Bp')

    # Custom clustering
    linkage = hclust.linkage(pdist_df.values, method='average', metric='euclidean', optimal_ordering=True)
    dn = hclust.dendrogram(linkage)
    sags_sorting = list(sag_ids[dn['leaves']])
    #plot_pdist_clustermap(pdist_df, linkage=linkage, sags_sorting=sags_sorting, species_sort=species_sorted_sags, savefig=f'{figures_dir}{prefix}_euclidean_snp_pdist_average-linkage.pdf')

    # Standard HACs
    for linkage_method in linkage_methods:
        pdist = distance.squareform(pdist_df.values)
        linkage = hclust.linkage(pdist, method=linkage_method, optimal_ordering=True)
        dn = hclust.dendrogram(linkage)
        sags_sorting = list(sag_ids[dn['leaves']])
        plot_pdist_clustermap(pdist_df, linkage=linkage, sags_sorting=sags_sorting, species_sort=species_sorted_sags, savefig=f'{figures_dir}{prefix}_seq_pdist_{linkage_method}-linkage.pdf')


def plot_pdist_clustermap(pdist_df, linkage=None, sags_sorting=None, location_sort=None, sample_sort=None, species_sort=None, temperature_sort=None, log=False, pseudocount=1E-1, cmap='Blues', grid=True, cbar_label='SNPs', savefig=None):
    # Set up a colormap:
    # use copy so that we do not mutate the global colormap instance
    palette = copy.copy(plt.get_cmap(cmap))
    palette.set_bad((0.5, 0.5, 0.5))

    if sags_sorting is None:
        ordered_sags = list(pdist_df.index)
    else:
        ordered_sags = list(sags_sorting)

    plot_df = pdist_df.reindex(index=ordered_sags, columns=ordered_sags)
    num_seqs = plot_df.shape[0]

    #fig = plt.figure(figsize=(double_col_width, double_col_width))
    fig = plt.figure(figsize=(single_col_width, single_col_width))
    ax_dendro = plt.subplot2grid((10, 10), (0, 0), rowspan=1, colspan=9)
    ax_dendro.axis('off')
    ax_cbar = plt.subplot2grid((10, 10), (2, 9), rowspan=8, colspan=1)
    ax = plt.subplot2grid((10, 10), (1, 0), rowspan=9, colspan=9)
    #ax = fig.add_subplot(111)
    if log == True:
        im = ax.imshow(np.log10(pseudocount + plot_df), cmap=palette, aspect='equal')
    else:
        im = ax.imshow(plot_df.astype(float), cmap=palette, aspect='equal')

    ax.set_xlim(-1, num_seqs)
    ticks = []
    tick_labels = []
    for ref in ["OS-A", "OS-B'"]:
        if ref in ordered_sags:
            ticks.append(ordered_sags.index(ref))
            tick_labels.append(ref)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(-1, num_seqs)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(plot_df.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(plot_df.shape[0] + 1) - 0.5, minor=True)
    ax.tick_params(which='minor', bottom=False, left=False, right=False, top=False)
    if grid == True:
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.1)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    #fig.colorbar(im, shrink=0.75)
    cbar = fig.colorbar(im, shrink=0.75, cax=ax_cbar, label=cbar_label)

    if linkage is not None:
        mpl.rcParams['lines.linewidth'] = 0.5
        hclust.dendrogram(linkage, ax=ax_dendro, color_threshold=0, above_threshold_color='k')

    # Add annotations
    if location_sort is not None:
        # Annotate locations
        x_marks = np.arange(num_seqs)
        y_marks = -1.5 * np.ones(num_seqs)
        colors = []
        for sag_id in ordered_sags:
            if sag_id in location_sort['OS']:
                colors.append('cyan')
            elif sag_id in location_sort['MS']:
                colors.append('magenta')
            else:
                colors.append('none')
        ax.scatter(x_marks, y_marks, s=10, ec='none', fc=colors, clip_on=False)
        ax.scatter(y_marks, x_marks, s=10, ec='none', fc=colors, clip_on=False)

    elif species_sort is not None:
        # Annotate species
        x_marks = np.arange(num_seqs)
        y_marks = -1.5 * np.ones(num_seqs)
        cmap = plt.get_cmap('tab10')
        colors = []
        for sag_id in ordered_sags:
            if sag_id in species_sort['syna']:
                colors.append(cm(0))
            elif sag_id in species_sort['synbp']:
                colors.append(cm(1))
            else:
                colors.append('none')
        ax.scatter(x_marks, y_marks, s=10, ec='none', fc=colors, clip_on=False)
        ax.scatter(y_marks, x_marks, s=10, ec='none', fc=colors, clip_on=False)

    elif sample_sort is not None:
        # Annotate sample
        x_marks = np.arange(num_seqs)
        y_marks = -1.5 * np.ones(num_seqs)
        cmap = plt.get_cmap('tab10')
        colors = []
        sag_samples = sorted(list(sample_sort.keys()))
        sample_dict = {}
        for i, sample_id in enumerate(sag_samples):
            sample_dict[sample_id] = i

        samples = []
        for sag_id in ordered_sags:
            color = 'none'
            for sample_id in sample_sort:
                if sag_id in sample_sort[sample_id]:
                    color = cmap(sample_dict[sample_id])
                    samples.append(sample_id)
                    break
            colors.append(color)

        print('\n')
        print([(sample_id, sample_dict[sample_id]) for sample_id in sag_samples])
        for sample_id in sag_samples:
            print(sample_id, len(sample_sort[sample_id]), sample_sort[sample_id])
        print('\n\n')
        ax.scatter(x_marks, y_marks, s=10, ec='none', fc=colors, clip_on=False)
        ax.scatter(y_marks, x_marks, s=10, ec='none', fc=colors, clip_on=False)

    elif temperature_sort is not None:
        # Annotate locations
        x_marks = np.arange(num_seqs)
        y_marks = -1.5 * np.ones(num_seqs)
        colors = []
        if 'T55' in temperature_sort:
            for sag_id in ordered_sags:
                if sag_id in location_sort['T55.0']:
                    colors.append('blue')
                elif sag_id in location_sort['T59.0'] or sag_id in location_sort['T60.0']:
                    colors.append('red')
                else:
                    colors.append('none')
        ax.scatter(x_marks, y_marks, s=10, ec='none', fc=colors, clip_on=False)
        ax.scatter(y_marks, x_marks, s=10, ec='none', fc=colors, clip_on=False)


    plt.tight_layout(h_pad=-1.5, pad=0.2)
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    else:
        return ax

def convert_alnmt2num(aln):
    letters = list(string.ascii_uppercase)
    letters_dict = {'-':0}
    for i, lett in enumerate(letters):
        letters_dict[lett] = i + 1

    return np.array([[letters_dict[nucl] for nucl in seq] for seq in np.array(aln)])

def check_alignment_lengths():
    data_dir = '../results/single-cell/sscs_pangenome/_aln_results/'
    figures_dir = '../figures/analysis/tests/pangenome_construction/sequence_lengths/'
    locus = 'YSG_0200'

    aln = AlignIO.read(f'{data_dir}{locus}_aln.fna', 'fasta')

    pdist_df = align_utils.calculate_pairwise_snp_distance(aln)
    pickle.dump(pdist_df, open(f'../results/tests/failed_alignments/{locus}_pdist.dat', 'wb'))
    pdist_df = pickle.load(open(f'../results/tests/failed_alignments/{locus}_pdist.dat', 'rb'))

    sample_ids = np.array(list(pdist_df.index))
    pdist = distance.squareform(pdist_df.values)
    Z = hclust.linkage(pdist, method='average', optimal_ordering=True)
    print(Z)

    dn = hclust.dendrogram(Z, no_plot=True)
    sags_sorting = list(sample_ids[dn['leaves']])
    aln_sorted = sort_alnmt(aln, sags_sorting)

    print(aln_sorted)

    #draw_alignment_sequences(aln_nogaps, savefig=f'{figures_dir}synbp_seq_alignment.pdf')
    #draw_alignment_sequences(aln_nogaps_sorted, savefig=f'{figures_dir}synbp_seq_alignment_clustered.pdf')


def plot_cluster_gene_lengths():
    input_file='../results/single-cell/sscs_pangenome/_mcl_results/YSG_0200.fna'
    figures_dir='../figures/analysis/tests/pangenome_construction/sequence_lengths/'
    seq_records = utils.read_fasta(input_file)
    cluster_id = input_file.split('/')[-1].replace('.fna', '')

    cluster_sizes = []
    for length_threshold in [0, 1, 5, 10, 20, 50, 100, 200, 500]:
        print(length_threshold)
        sorted_ids, length_clusters = find_length_clusters(seq_records, length_threshold=length_threshold, savefig=f'{figures_dir}{cluster_id}_length_clusters_lcutoff{length_threshold}.pdf')
        cluster_sizes.append([length_threshold, utils.sorted_unique(length_clusters)])

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('rank')
    ax.set_xscale('log')
    ax.set_ylabel('cluster size')
    ax.set_yscale('log')
    for ldata in cluster_sizes:
        lcutoff = ldata[0]
        cluster_ids, cluster_counts = ldata[1]
        ax.plot(np.arange(len(cluster_counts)) + 1, cluster_counts, '-o', mfc='none', label=f'lc={lcutoff}')
    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}{cluster_id}_cluster_sizes_lcutoff.pdf')


    cluster_sizes = []
    for edge_threshold in [0.5, 0.75, 0.9, 0.95, 0.99]:
        sorted_ids, length_clusters = find_length_clusters(seq_records, edge_threshold=edge_threshold, savefig=f'{figures_dir}{cluster_id}_length_clusters_ecutoff{edge_threshold}.pdf')
        cluster_sizes.append([edge_threshold, utils.sorted_unique(length_clusters)])

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('rank')
    ax.set_xscale('log')
    ax.set_ylabel('cluster size')
    ax.set_yscale('log')
    for edata in cluster_sizes:
        ecutoff = edata[0]
        cluster_ids, cluster_counts = edata[1]
        ax.plot(np.arange(len(cluster_counts)) + 1, cluster_counts, '-o', mfc='none', label=f'ec={ecutoff}')
    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}{cluster_id}_cluster_sizes_ecutoff.pdf')

    seq_records = utils.read_fasta('../results/single-cell/sscs_pangenome/filtered_orthogroups/YSG_0200.fna')
    length_arr = np.array([len(seq_records[seq_id]) for seq_id in seq_records])
    plot_length_distribution(length_arr)

def plot_length_distribution(seq_lengths, savefig=None):
    fig = plt.figure(figsize=(double_col_width, 0.3 * double_col_width))
    ax = fig.add_subplot(131)
    ax.set_xlabel('sequence length')
    ax.set_ylabel('counts')
    ax.hist(seq_lengths, bins=40, range=(min(seq_lengths) - 100, max(seq_lengths) + 100))

    ax = fig.add_subplot(132)
    ax.set_xlabel('rank')
    ax.set_ylabel('sequence length')

    num_seqs = len(seq_lengths)
    rank = np.arange(1, num_seqs + 1)
    length_arr = np.array(sorted(seq_lengths))
    avg_slope = (length_arr[-1] - length_arr[0]) / (rank[-1] - rank[0])
    ax.plot(rank, length_arr)
    ax.plot(rank, length_arr[0] + avg_slope * rank, '--k')

    ax = fig.add_subplot(133)
    ax.set_xlabel('rank')
    ax.set_ylabel('length-rank slope')

    ax.plot(rank, avg_slope * np.ones(num_seqs), '--k')

    for step in [1, 2, 5, 10]:
        local_slope = [(length_arr[i + step] - length_arr[i]) / (rank[i + step] - rank[i]) for i in range(0, num_seqs - step)]
        ax.plot(rank[:-step], local_slope, lw=1, label=f'step={step}')
    ax.legend(fontsize=6)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()


def check_hitchhiking_blocks(random_seed=12345):
    data_dir = '../results/tests/gene_clustering/'
    figures_dir = '../figures/analysis/tests/hitchhiking/'
    locus = 'CYB_2798'


    # Gap-filtered alignments
    filtered_pdist_df = pickle.load(open(f'{data_dir}filtered_nucl_snp_pdist.dat', 'rb'))
    #make_clustered_pdist_figures(filtered_pdist_df, figures_dir, linkage_methods=['ward', 'weighted'], prefix='filtered_alignment')

    metadata = MetadataMap()
    sag_ids = np.array(list(filtered_pdist_df.index))
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    species_sorted_sags['syna'] = species_sorted_sags.pop('A')
    species_sorted_sags['synbp'] = species_sorted_sags.pop('Bp')


    # Use average linkage to look for sub-species clusters
    # Look in Syn. B' since it's more diverse
    synbp_sag_ids = np.array(species_sorted_sags['synbp'])
    synbp_pdist_df = filtered_pdist_df.reindex(index=synbp_sag_ids, columns=synbp_sag_ids)
    print(synbp_pdist_df)

    linkage_method = 'average'
    pdist = distance.squareform(synbp_pdist_df.values)
    Z = hclust.linkage(pdist, method=linkage_method, optimal_ordering=True)

    clusters = hclust.fcluster(Z, 4, criterion='distance')
    cluster_tags = np.unique(clusters)
    print(clusters)

    cluster_dict = {}
    for tag in cluster_tags:
        cluster_dict[tag] = synbp_sag_ids[clusters == tag]
    print(cluster_dict)

    dn = hclust.dendrogram(Z, no_plot=True)
    sags_sorting = list(synbp_sag_ids[dn['leaves']])
    location_sort = metadata.sort_sags(synbp_sag_ids, by='location')
    print(location_sort)
    plot_pdist_clustermap(synbp_pdist_df, linkage=Z, sags_sorting=sags_sorting, location_sort=location_sort, savefig=f'{figures_dir}synbp_rplC_location.pdf')
    print('\n\n\n')

    sample_sort = metadata.sort_sags(synbp_sag_ids, by='sample')
    print(sample_sort)
    plot_pdist_clustermap(synbp_pdist_df, linkage=Z, sags_sorting=sags_sorting, sample_sort=sample_sort, savefig=f'{figures_dir}synbp_rplC_sample.pdf')
    print('\n\n\n')

    temperature_sort = metadata.sort_sags(synbp_sag_ids, by='temperature')
    print(temperature_sort)

def plot_haplotype_rsb():
    data_dir = '../results/tests/gene_clustering/'
    figures_dir = '../figures/analysis/tests/hitchhiking/'
    locus = 'CYB_2798'


    # Gap-filtered alignments
    #filtered_pdist_df = pickle.load(open(f'{data_dir}filtered_nucl_snp_pdist.dat', 'rb'))
    #make_clustered_pdist_figures(filtered_pdist_df, figures_dir, linkage_methods=['ward', 'weighted'], prefix='filtered_alignment')

    aln = AlignIO.read(f'{data_dir}{locus}_codon_aln_MUSCLE.fasta', 'fasta')
    synbp_aln = seq_utils.filter_species_alignment(aln, 'osbp')
    print(synbp_aln)

    trimmed_aln, x_trim = align_utils.trim_alignment_gaps(synbp_aln, start_gap_perc=0.05, return_edges=True)
    print(locus, x_trim)

    nonsynonymous_snps = seq_utils.get_nonsynonymous_snps(trimmed_aln)
    print(nonsynonymous_snps)
    print(np.array(trimmed_aln)[:, nonsynonymous_snps[0]])
    protein_aln_arr = np.array(seq_utils.translate_nucl_alnmt(trimmed_aln))

    aln_snps, x_snps = seq_utils.get_snps(trimmed_aln, return_x=True)
    print(x_snps)
    r_sq = calculate_rsquared(aln_snps)
    nonsynonymous_idx = [i for i in range(len(x_snps)) if x_snps[i] in nonsynonymous_snps]
    nonsynonymous_perfect_linkage_regions = find_strongly_linked_sites(r_sq, nonsynonymous_idx, min_length=2)

    # Plot S / pi
    cmap = plt.get_cmap('tab10')
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('segregating sites, $S_n$')
    ax.set_ylabel(r'$\langle \pi \rangle / S_n$')
    ax.set_ylim(0, 1)

    x0 = 5
    Sn = []
    pi = []
    for i in range(5):
        pdist_df = align_utils.calculate_pairwise_snp_distance(aln_snps[:, x0 - i:x0 + i + 1])
        pdist_values = utils.get_matrix_triangle_values(pdist_df.values, k=1)
        Sn.append(2 * i + 1)
        pi.append(np.mean(pdist_values))
    Sn = np.array(Sn)
    pi = np.array(pi)

    ax.scatter(Sn, pi / Sn, s=10, fc='none', ec=cmap(0))
    n = len(aln_snps)
    an = np.sum([1 / i for i in range(1, n)])
    print(an, np.log(n))
    x_theory = np.arange(11)
    ax.plot(x_theory, np.ones(len(x_theory)) / an, ls='--', c='k')

    plt.tight_layout()
    plt.savefig(f'{figures_dir}segregating_sites_diversity_ratio.pdf')

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('segregating sites, $S_n$')
    ax.set_ylabel(r'average pair SNPs, $\langle \pi \rangle$')
    ax.scatter(Sn, pi, s=20, fc='none', ec=cmap(0), label='CYB_2798')
    ax.plot(x_theory[1:], x_theory[1:] / an, ls='--', c='k')
    ax.plot(x_theory[1:], x_theory[1:] / 2, ls='--', c='gray')

    np.random.seed(12345)
    aln = AlignIO.read(f'../results/tests/hitchhiking_search/coalescent_simulations/coalescent_theta0.05_rho0_run10_seqs.fna', 'fasta')
    aln = subsample_alignment(aln, n)
    x0 = get_core_site(aln, min_length=4)
    Sn, pi = calculate_snp_diversity(aln, x0, L=5)
    ax.scatter(Sn, pi, s=20, marker='s', fc='none', ec=cmap(1), label=r'$\theta=0.05$, $\rho=0$')

    aln = AlignIO.read(f'../results/tests/hitchhiking_search/coalescent_simulations/coalescent_theta0.05_rho0.15_run10_seqs.fna', 'fasta')
    aln = subsample_alignment(aln, n)
    x0 = get_core_site(aln, min_length=2)
    Sn, pi = calculate_snp_diversity(aln, x0, L=5)
    ax.scatter(Sn, pi, s=20, marker='v', fc='none', ec=cmap(2), label=r'$\theta=0.05$, $\rho=0.15$')

    aln = AlignIO.read(f'../results/tests/hitchhiking_search/coalescent_simulations/coalescent_theta0.05_rho0.5_run10_seqs.fna', 'fasta')
    aln = subsample_alignment(aln, n)
    x0 = get_core_site(aln, min_length=2)
    print(x0)
    Sn, pi = calculate_snp_diversity(aln, x0, L=5)
    ax.scatter(Sn, pi, s=20, marker='x', fc='none', ec=cmap(3), label=r'$\theta=0.05$, $\rho=0.5$')

    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}segregating_sites_vs_diversity.pdf')

    print('\n\n')

    print(nonsynonymous_perfect_linkage_regions)


    fig = plt.figure(figsize=(1.5 * double_col_width, single_col_width))
    palette = plt.get_cmap('Blues')
    palette.set_bad((0.5, 0.5, 0.5))

    pdist_df = align_utils.calculate_pairwise_snp_distance(aln_snps)
    sag_ids = np.array(list(pdist_df.index))
    pdist = distance.squareform(pdist_df.values)
    Z = hclust.linkage(pdist, method='average', optimal_ordering=True)
    dn = hclust.dendrogram(Z, no_plot=True)
    ordered_sags = list(sag_ids[dn['leaves']])

    ns_snp_idx = 5
    for i, L in enumerate([1, 2, 3, 4, 5]):
        ax = fig.add_subplot(1, 5, i + 1)
        pdist_df = align_utils.calculate_pairwise_snp_distance(aln_snps[:, ns_snp_idx - L:ns_snp_idx + L + 1])
        plot_df = pdist_df.reindex(index=ordered_sags, columns=ordered_sags) / (2 * L + 1)
        num_seqs = plot_df.shape[0]

        ax.set_title(f'{2 * L + 1} SNPs')
        ax.set_xlim(-1, num_seqs)
        ticks = []
        tick_labels = []
        for ref in ["OS-A", "OS-B'"]:
            if ref in ordered_sags:
                ticks.append(ordered_sags.index(ref))
                tick_labels.append(ref)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(-1, num_seqs)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels)

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(plot_df.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(plot_df.shape[0] + 1) - 0.5, minor=True)
        ax.tick_params(which='minor', bottom=False, left=False, right=False, top=False)
        #ax.grid(which='minor', color='w', linestyle='-', linewidth=0.1)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        im = ax.imshow(plot_df, cmap=palette, vmin=0, vmax=1, aspect='equal', origin='upper')

    plt.tight_layout()
    plt.savefig(f'{figures_dir}{locus}_rsb.pdf')

    '''
    '''

    '''
    bifurcation_tree = seq_utils.BifurcationTree(aln_snps, 5)
    hitchhiking_dir = '../results/tests/hitchhiking_search/'
    #bifurcation_tree.export_graphs(f'{hitchhiking_dir}CYB_2798_bifurcation_diagram', format='dat')
    bifurcation_tree.export_graphs(output_file=f'{hitchhiking_dir}CYB_2798_bifurcation_diagram_new_format.dat', format='dat')

    bifurcation_tree = nx.read_gpickle(f'{hitchhiking_dir}CYB_2798_bifurcation_diagram_new_format.dat')
    plot_bifurcation_tree(bifurcation_tree, savefig=f'{figures_dir}CYB_2798_bifurcation_diagram_new_format.pdf')

    pos = graphviz_layout(bifurcation_tree, prog='dot')
    nx.draw(bifurcation_tree, pos=pos, arrows=False, node_size=0, connectionstyle='arc3,rad=0.2')
    for edge in bifurcation_tree.edges(data='weight'):
        nx.draw_networkx_edges(bifurcation_tree, pos, edgelist=[edge], width=edge[2]/3, arrows=False, node_size=0, connectionstyle='arc3,rad=0.2')
    core_haplotypes = bifurcation_tree.successors('root')
    cmap = plt.get_cmap('tab10')
    for i, core_hapl in enumerate(core_haplotypes):
        nx.draw_networkx_nodes(bifurcation_tree, pos, nodelist=[core_hapl], node_size=300, node_color=[cmap(i)])

    nx.draw_networkx_nodes(bifurcation_tree, pos, nodelist=['root'], node_size=500, node_color='k')
    plt.savefig(f'{figures_dir}CYB_2798_bifurcation_diagram_new_format.pdf')
    '''

    '''
    plot_num_haplotypes(bifurcation_tree, savefig=f'{figures_dir}CYB_2798_num_haplotypes_vs_distance.pdf')

    # Plot sigma^2 vs distance from core
    x0_simul = []


    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('genome distance')
    ax.set_ylabel('$r^2$')
    ax.set_yscale('log')
    ax.set_ylim(1E-5, 5E1)

    r_sq = calculate_rsquared(aln_snps)
    nonsynonymous_idx = [i for i in range(len(x_snps)) if x_snps[i] in nonsynonymous_snps]
    perfect_linkage_regions = find_strongly_linked_sites(r_sq, nonsynonymous_idx, min_length=2)
    perfect_linkage_lengths = [t[1] - t[0] for t in perfect_linkage_regions.values()]
    max_run = max(perfect_linkage_lengths)
    max_run_idx = [i for i in perfect_linkage_regions if (perfect_linkage_regions[i][1] - perfect_linkage_regions[i][0]) == max_run]
    print(max_run_idx)
    x0 = max_run_idx[0]
    ax.plot(np.arange(r_sq.shape[0]) - x0, r_sq[x0, :], c='k', ls='-', lw=2, label=r'CYB_2798', zorder=3)


    aln = AlignIO.read(f'../results/tests/hitchhiking_search/coalescent_simulations/coalescent_theta0.05_rho0_run10_seqs.fna', 'fasta')
    aln_snps, x_snps = seq_utils.get_snps(aln, return_x=True)
    r_sq = calculate_rsquared(aln_snps)
    snps_idx = [i for i in range(len(x_snps))]
    perfect_linkage_regions = find_strongly_linked_sites(r_sq, snps_idx, min_length=4, verbose=False)
    perfect_linkage_lengths = [t[1] - t[0] for t in perfect_linkage_regions.values()]
    max_run = max(perfect_linkage_lengths)
    max_run_idx = [i for i in perfect_linkage_regions if (perfect_linkage_regions[i][1] - perfect_linkage_regions[i][0]) == max_run]
    x0 = max_run_idx[max_run // 2]
    x0_simul.append(x0)
    ax.plot(np.arange(r_sq.shape[0])[x0-25:x0+25] - x0, r_sq[x0, x0-25:x0+25], alpha=0.5, label=r'$\theta=0.05$, $\rho=0$')

    aln = AlignIO.read(f'../results/tests/hitchhiking_search/coalescent_simulations/coalescent_theta0.05_rho0.15_run10_seqs.fna', 'fasta')
    aln_snps, x_snps = seq_utils.get_snps(aln, return_x=True)
    r_sq = calculate_rsquared(aln_snps)
    snps_idx = [i for i in range(len(x_snps))]
    perfect_linkage_regions = find_strongly_linked_sites(r_sq, snps_idx, min_length=2, verbose=False)
    perfect_linkage_lengths = [t[1] - t[0] for t in perfect_linkage_regions.values()]
    max_run = max(perfect_linkage_lengths)
    max_run_idx = [i for i in perfect_linkage_regions if (perfect_linkage_regions[i][1] - perfect_linkage_regions[i][0]) == max_run]
    x0 = max_run_idx[max_run // 2]
    x0_simul.append(x0)
    ax.plot(np.arange(r_sq.shape[0])[x0-25:x0+25] - x0, r_sq[x0, x0-25:x0+25], alpha=0.5, label=r'$\rho=0.15$')

    aln = AlignIO.read(f'../results/tests/hitchhiking_search/coalescent_simulations/coalescent_theta0.05_rho0.5_run10_seqs.fna', 'fasta')
    aln_snps, x_snps = seq_utils.get_snps(aln, return_x=True)
    r_sq = calculate_rsquared(aln_snps)
    snps_idx = [i for i in range(len(x_snps))]
    perfect_linkage_regions = find_strongly_linked_sites(r_sq, snps_idx, min_length=2, verbose=False)
    perfect_linkage_lengths = [t[1] - t[0] for t in perfect_linkage_regions.values()]
    max_run = max(perfect_linkage_lengths)
    max_run_idx = [i for i in perfect_linkage_regions if (perfect_linkage_regions[i][1] - perfect_linkage_regions[i][0]) == max_run]
    x0 = max_run_idx[max_run // 2]
    x0_simul.append(x0)
    ax.plot(np.arange(r_sq.shape[0])[x0-25:x0+25] - x0, r_sq[x0, x0-25:x0+25], alpha=0.5, label=r'$\rho=0.5$')

    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}sigma2_around_core.pdf')
    plt.close()

    # Bifurcation trees in simulations
    aln = AlignIO.read(f'../results/tests/hitchhiking_search/coalescent_simulations/coalescent_theta0.05_rho0_run10_seqs.fna', 'fasta')
    #x0 = x0_simul[0]
    #bifurcation_tree = make_bifurcation_tree(aln[:, x0 - 150:x0 + 150])
    bifurcation_tree = make_bifurcation_tree(aln[:, 350:650])
    g = bifurcation_tree.graph
    plot_bifurcation_tree(g, savefig=f'{figures_dir}coalescent_theta0.05_rho0_bifurcation_diagram.pdf')


    aln = AlignIO.read(f'../results/tests/hitchhiking_search/coalescent_simulations/coalescent_theta0.05_rho0.15_run10_seqs.fna', 'fasta')
    #x0 = x0_simul[1]
    bifurcation_tree = make_bifurcation_tree(aln[:, 100:400], i_offset=-1)
    g = bifurcation_tree.graph
    plot_bifurcation_tree(g, savefig=f'{figures_dir}coalescent_theta0.05_rho0.15_bifurcation_diagram.pdf')

    aln = AlignIO.read(f'../results/tests/hitchhiking_search/coalescent_simulations/coalescent_theta0.05_rho0.5_run10_seqs.fna', 'fasta')
    bifurcation_tree = make_bifurcation_tree(aln[:, 200:500], i_offset=-1)
    g = bifurcation_tree.graph
    plot_bifurcation_tree(g, savefig=f'{figures_dir}coalescent_theta0.05_rho0.5_bifurcation_diagram.pdf')



    right_tree = nx.read_gpickle(f'{hitchhiking_dir}CYB_2798_bifurcation_diagram_right.dat')
    right_tree.remove_node('root')
    subset_nodes = []
    for node in right_tree.nodes():
        if node not in ['H1', 'H2']:
            if '_left' in node:
                hapl = node.replace('_left', '')
                if hapl[-1] == 'A':
                    right_tree.nodes[node]['subset'] = 0
                    subset_nodes.append(node)
                else:
                    right_tree.nodes[node]['subset'] = 1
            else:
                if node[0] == 'A':
                    right_tree.nodes[node]['subset'] = 0
                    subset_nodes.append(node)
                else:
                    right_tree.nodes[node]['subset'] = 1
        else:
            if node == 'H1':
                right_tree.nodes[node]['subset'] = 0
                subset_nodes.append(node)
            else:
                right_tree.nodes[node]['subset'] = 1

    pos = graphviz_layout(right_tree, prog='dot')
    #pos = graphviz_layout(right_tree, prog='twopi')
    #pos = nx.multipartite_layout(right_tree, align='horizontal', center=['H1', 'H2'])
    #pos = nx.multipartite_layout(right_tree, align='horizontal')
    #pos = nx.bipartite_layout(right_tree, nodes=subset_nodes, align='horizontal')

    nx.draw(right_tree, pos=pos, arrows=False, node_size=0, connectionstyle='arc3,rad=0.2')
    for edge in right_tree.edges(data='weight'):
        nx.draw_networkx_edges(right_tree, pos, edgelist=[edge], width=edge[2]/3, arrows=False, node_size=0, connectionstyle='arc3,rad=0.2')
    nx.draw_networkx_nodes(right_tree, pos, nodelist=['H1'], node_color='blue')
    nx.draw_networkx_nodes(right_tree, pos, nodelist=['H2'], node_color='red')
    plt.savefig(f'{figures_dir}CYB_2798_bifurcation_diagram.pdf')

    aln = AlignIO.read(f'../results/tests/hitchhiking_search/coalescent_simulations/coalescent_theta0.05_rho0_run10_seqs.fna', 'fasta')
    aln_snps, x_snps = seq_utils.get_snps(aln, return_x=True)
    r_sq = calculate_rsquared(aln_snps)

    #plot_num_haplotypes(bifurcation_tree, savefig=f'{figures_dir}num_haplotypes_vs_distance.pdf')
    ax = plot_num_haplotypes(bifurcation_tree, label='CYB_2798')
    cmap = plt.get_cmap('tab10')

    aln = AlignIO.read(f'../results/tests/hitchhiking_search/coalescent_simulations/coalescent_theta0.05_rho0_run10_seqs.fna', 'fasta')
    bifurcation_tree = make_bifurcation_tree(aln)
    #plot_num_haplotypes(bifurcation_tree, ax=ax, label=r'$\rho=0$', alpha=0.6, color=cmap(0), xlim=(-10, 10), ylim=(0, 10), savefig=f'{figures_dir}num_haplotypes_vs_distance.pdf')
    ax = plot_num_haplotypes(bifurcation_tree, ax=ax, label=r'$\rho=0$', alpha=0.6, color=cmap(0), xlim=(-10, 10), ylim=(0, 10))

    aln = AlignIO.read(f'../results/tests/hitchhiking_search/coalescent_simulations/coalescent_theta0.05_rho0.15_run10_seqs.fna', 'fasta')
    bifurcation_tree = make_bifurcation_tree(aln[:, 100:400], i_offset=-1)
    #plot_num_haplotypes(bifurcation_tree, ax=ax, label=r'$\rho=0.15$', alpha=0.6, color=cmap(1), xlim=(-10, 10), ylim=(0, 10), savefig=f'{figures_dir}num_haplotypes_vs_distance.pdf')
    ax = plot_num_haplotypes(bifurcation_tree, ax=ax, label=r'$\rho=0.15$', alpha=0.6, color=cmap(1), xlim=(-10, 10), ylim=(0, 10))

    aln = AlignIO.read(f'../results/tests/hitchhiking_search/coalescent_simulations/coalescent_theta0.05_rho0.5_run10_seqs.fna', 'fasta')
    bifurcation_tree = make_bifurcation_tree(aln[:, 200:500], i_offset=-1)
    plot_num_haplotypes(bifurcation_tree, ax=ax, label=r'$\rho=0.5$', alpha=0.6, color=cmap(2), xlim=(-10, 10), ylim=(0, 10), savefig=f'{figures_dir}num_haplotypes_vs_distance.pdf')
    '''


def plot_num_haplotypes(bifurcation_tree, label='', alpha=1, color='k', xlim=None, ylim=None, ax=None, savefig=None):
    num_haplotypes = []
    for direction in ['left', 'right']:
        direction_K = []
        for core_node in bifurcation_tree.successors('root'):
            K = []
            nodes = [f'{core_node}_{direction}']
            while len(nodes) > 0:
                new_nodes = []
                for node in nodes:
                    new_nodes += bifurcation_tree.successors(node)
                nodes = new_nodes
                K.append(len(nodes))
            direction_K.append(np.array(K[:-1]))
        num_haplotypes.append(np.sum(direction_K, axis=0))


    if ax is None:
        fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
        ax = fig.add_subplot(111)
        ax.set_xlabel('genome distance (SNPs)')
        ax.set_ylabel(r'number of haplotypes, $K$')

    direction = -1
    x_full = []
    K_full = []
    for i, K in enumerate(num_haplotypes):
        if i == 0:
            K_full += list(K[::-1])
            x_full += list(-np.arange(1, len(K) + 1)[::-1])
        else:
            K_full.append(len(list(bifurcation_tree.successors('root'))))
            K_full += list(K)
            x_full += list(np.arange(len(K) + 1))

    print(x_full, K_full)
    ax.plot(x_full, K_full, '-o', lw=1, ms=2, alpha=alpha, c=color, label=label)
    '''
    for K in num_haplotypes:
        x = direction * np.arange(len(K))
        if direction == -1:
            ax.plot(x, K, '-o', lw=1, ms=2, alpha=alpha, c=color, label=label)
        else:
            ax.plot(x, K, '-o', lw=1, ms=2, alpha=alpha, c=color)
        direction = 1
        x_full += list(x)
        K_full += K
    '''
    ax.legend()

    #ax.set_xticks(np.unique(x_full))
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, max(K_full) + 1)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        return ax

def make_bifurcation_tree(aln, i_offset=0):
    aln_snps, x_snps = seq_utils.get_snps(aln, return_x=True)
    r_sq = calculate_rsquared(aln_snps)
    snps_idx = [i for i in range(len(x_snps))]
    perfect_linkage_regions = find_strongly_linked_sites(r_sq, snps_idx, min_length=2, verbose=False)

    perfect_linkage_lengths = [t[1] - t[0] for t in perfect_linkage_regions.values()]
    max_run = max(perfect_linkage_lengths)
    max_run_idx = [i for i in perfect_linkage_regions if (perfect_linkage_regions[i][1] - perfect_linkage_regions[i][0]) == max_run]
    if i_offset >= 0:
        i_max = max_run_idx[i_offset + max_run // 2]
    else:
        i_max = max_run_idx[len(max_run_idx) // 2]
    print(max_run)
    bifurcation_tree = seq_utils.BifurcationTree(aln_snps, i_max)

    return bifurcation_tree

def plot_bifurcation_tree(bifurcation_tree, savefig=None):
    cmap = plt.get_cmap('tab10')
    pos = graphviz_layout(bifurcation_tree, prog='dot')
    nx.draw(bifurcation_tree, pos=pos, arrows=False, node_size=0, connectionstyle='arc3,rad=0.2')
    for edge in bifurcation_tree.edges(data='weight'):
        nx.draw_networkx_edges(bifurcation_tree, pos, edgelist=[edge], width=edge[2]/3, arrows=False, node_size=0, connectionstyle='arc3,rad=0.2')
    core_haplotypes = bifurcation_tree.successors('root')
    for i, core_hapl in enumerate(core_haplotypes):
        nx.draw_networkx_nodes(bifurcation_tree, pos, nodelist=[core_hapl], node_size=300, node_color=[cmap(i)])

    nx.draw_networkx_nodes(bifurcation_tree, pos, nodelist=['root'], node_size=500, node_color='k')
    if savefig:
        plt.savefig(savefig)
        plt.close()

def get_core_site(aln, min_length=2):
    aln_snps, x_snps = seq_utils.get_snps(aln, return_x=True)
    r_sq = calculate_rsquared(aln_snps)
    snps_idx = [i for i in range(len(x_snps))]
    perfect_linkage_regions = find_strongly_linked_sites(r_sq, snps_idx, min_length=min_length, verbose=False)
    perfect_linkage_lengths = [t[1] - t[0] for t in perfect_linkage_regions.values()]
    if len(perfect_linkage_lengths) > 0:
        max_run = max(perfect_linkage_lengths)
        if max_run > 0:
            max_run_idx = [i for i in perfect_linkage_regions if (perfect_linkage_regions[i][1] - perfect_linkage_regions[i][0]) == max_run]
            x0 = max_run_idx[max_run // 2]
            print(f'Max run length: {max_run}')
        else:
            print(f'No perfectly linked region longer than {min_length} SNPs found.')
            x0 = r_sq.shape[0] // 2
    else:
        print(f'No perfectly linked region longer than {min_length} SNPs found.')
        x0 = r_sq.shape[0] // 2
    return x0

def subsample_alignment(aln, n):
    idx = np.random.choice(len(aln), n, replace=False)
    rec_list = list(aln)
    sampled_aln = MultipleSeqAlignment([rec_list[i] for i in idx])
    return sampled_aln

def calculate_snp_diversity(aln, x0, L=10):
    Sn = []
    pi = []
    aln_snps, x_snps = seq_utils.get_snps(aln, return_x=True)
    for i in range(L):
        pdist_df = align_utils.calculate_pairwise_snp_distance(aln_snps[:, x0 - i:x0 + i + 1])
        pdist_values = utils.get_matrix_triangle_values(pdist_df.values, k=1)
        Sn.append(2 * i + 1)
        pi.append(np.mean(pdist_values))
    return np.array(Sn), np.array(pi)

def make_sweep_test_files():
    test_loci = ['CYB_0762', 'YSG_0398-2S', 'lnt-2']
    input_dir = '../results/single-cell/sscs_pangenome/fine_scale_og_clusters/'
    output_dir = '../results/tests/hitchhiking_search/'
    f_orthogroup_table = '../results/single-cell/sscs_pangenome/fine_scale_og_clusters/sscs_mapped_single_copy_orthogroup_presence.tsv'
    metadata = MetadataMap()
    pangenome_map = PangenomeMap(f_orthogroup_table=f_orthogroup_table)

    for locus in test_loci:
        f_seqs = f'{input_dir}{locus}.fna'
        seq_records = seq_utils.read_seqs_and_map_sag_ids(f_seqs, pangenome_map)
        sag_ids = [rec.id for rec in seq_records]
        species_sag_ids = metadata.sort_sags(sag_ids, by='species')
        synbp_sag_ids = species_sag_ids['Bp']
        filtered_records = [rec for rec in seq_records if rec.id in synbp_sag_ids]
        SeqIO.write(filtered_records, f'{output_dir}{locus}_synbp_seqs.fna', 'fasta')


if __name__ == '__main__':
    #test_all()
    #test_custom_pipeline()
    #calculate_custom_pipeline_metrics()
    #test_orthogroup_divergences()
    #test_locus_seqs_clustering()
    #check_alignment_lengths()
    #plot_cluster_gene_lengths()
    #check_hitchhiking_blocks()
    #plot_haplotype_rsb()
    make_sweep_test_files()

