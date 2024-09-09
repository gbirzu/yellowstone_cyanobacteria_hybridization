import argparse
import glob
import os
import numpy as np
import pandas as pd
import pickle
import utils
import scipy.sparse as sparse
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
import calculate_linkage_disequilibria as ld
import pg_find_orthogroup_clusters as cluster_utils
import scipy.cluster.hierarchy as hclust
from pangenome_utils import PangenomeMap
from metadata_map import MetadataMap
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def get_linkage_blocks(aln, x_snps=None, rsq_threshold=0.6, seed_size_threshold=3, min_nn_perc=0.75, min_length=6):
    rsq = ld.calculate_rsquared(aln)
    #g_dense = (rsq > rsq_threshold).astype(int)
    g_dense = (rsq >= rsq_threshold).astype(int)
    np.fill_diagonal(g_dense, 0)
    g = sparse.csr_matrix(g_dense)

    num_components, labels = sparse.csgraph.connected_components(g)
    cc, cc_counts = utils.sorted_unique(labels)
    cluster_labels = cc[cc_counts > seed_size_threshold]

    # Map locations to alignment
    cluster_dict = {}
    if x_snps is None:
        x_snps = np.arange(len(aln[0]))
    for cluster_idx, cluster_label in enumerate(cluster_labels):
        #cluster_dict[cluster_label] = x_snps[labels == cluster_label]
        x_cluster = x_snps[labels == cluster_label]
        cluster_length = np.max(x_cluster) - np.min(x_cluster) + 1

        # Filter clusters with low % of nearest neighbor SNPs or short length
        nn_perc = calculate_nearest_neighbor_percentage(x_cluster, x_snps)
        if (nn_perc >= min_nn_perc) and (cluster_length >= min_length):
            cluster_dict[cluster_idx] = x_cluster
    return cluster_dict


def summarize_linkage_blocks(f_aln, og_id, sampled_sag_ids, pangenome_map, metadata, args, out_sample_sag_ids=None, consensus_id_head='consensus'):
    aln = align_utils.read_main_cloud_alignment(f_aln, pangenome_map, metadata, dc_dict={'A':args.main_cloud_cutoff, 'Bp':args.main_cloud_cutoff, 'C':0.})
    species_gene_ids = pangenome_map.get_og_gene_ids(og_id, sag_ids=sampled_sag_ids)
    linkage_blocks, x_species_snps = find_linkage_blocks(aln, species_gene_ids, snp_frequency_cutoff=args.snp_frequency_cutoff, rsq_min=args.rsq_min, min_block_length=args.min_block_length)

    # Construct haplotype block alignments
    aln_species = align_utils.get_subsample_alignment(aln, species_gene_ids)
    if out_sample_sag_ids is None:
        sag_ids = pangenome_map.get_sag_ids()
        out_sample_sag_ids = sag_ids[~np.isin(sag_ids, sampled_sag_ids)]
    out_sample_gene_ids = pangenome_map.get_og_gene_ids(og_id, sag_ids=out_sample_sag_ids)
    aln_out_sample = align_utils.get_subsample_alignment(aln, out_sample_gene_ids)

    # Make consensus alignment for other species
    out_sample_species_sags = metadata.sort_sags(out_sample_sag_ids, by='species')
    consensus_list = []
    for species in ['A', 'Bp', 'C']:
        out_sample_gene_ids = pangenome_map.get_og_gene_ids(og_id, sag_ids=out_sample_species_sags[species])
        aln_out_sample_species = align_utils.get_subsample_alignment(aln, out_sample_gene_ids)
        if len(aln_out_sample_species) > 0:
            consensus_seq_arr = seq_utils.get_consensus_seq(np.array(aln_out_sample_species), seq_type='nucl')
            consensus_list.append(SeqRecord(Seq(''.join(consensus_seq_arr)), id=f'{species}_main_cloud_consensus', description=''))
    aln_consensus = MultipleSeqAlignment(consensus_list)

    block_stats, haplotype_clustered_genomes = make_block_summary_tables(linkage_blocks, aln_species, x_species_snps, aln_out_sample, og_id, pangenome_map, metadata, args.snp_frequency_cutoff, aln_consensus=aln_consensus, donor_match_cutoff=args.match_cutoff, consensus_id_head=consensus_id_head)
    
    return block_stats, haplotype_clustered_genomes


'''
def read_main_cloud_alignment(f_aln, pangenome_map, metadata, dc_dict={'A':0.15, 'Bp':0.5, 'C':0.}):
    aln = seq_utils.read_alignment(f_aln)
    species_sorted_gene_ids = align_utils.sort_aln_rec_ids(aln, pangenome_map, metadata)
    species_main_cloud_alns = []
    for species in species_sorted_gene_ids:
        if len(species_sorted_gene_ids[species]) > 0:
            aln_species = align_utils.get_subsample_alignment(aln, species_sorted_gene_ids[species])
            seq_ids = np.array([rec.id for rec in aln_species])
            d_consensus = align_utils.calculate_consensus_distance(aln_species, method='jc69')
            filtered_seq_ids = seq_ids[d_consensus <= dc_dict[species]]
            species_main_cloud_alns.append(align_utils.get_subsample_alignment(aln_species, filtered_seq_ids))
    return align_utils.stack_alignments(species_main_cloud_alns)
'''

def find_linkage_blocks(aln, subsample_gene_ids, snp_frequency_cutoff=3, rsq_min=1., min_block_length=0):
    aln_species = align_utils.get_subsample_alignment(aln, subsample_gene_ids)

    if len(aln_species) > 0:
        aln_species_hf_snps, x_species_snps = align_utils.get_high_frequency_snp_alignment(aln_species, snp_frequency_cutoff, counts_filter=True)
        if len(aln_species_hf_snps) > 1:
            rsq = ld.calculate_rsquared(aln_species_hf_snps)
            linkage_blocks = get_linkage_runs(rsq, rsq_min)
            long_blocks = [block for block in linkage_blocks if len(block) >= min_block_length]
        else:
            long_blocks = []
            x_species_snps = []
    else:
        long_blocks = []
        x_species_snps = []

    return long_blocks, x_species_snps

def get_linkage_runs(d_matrix, d_min):
    '''
    Partitions index of `d_matrix` into segments of consecutive entries if d_ij >= d_min.
    '''

    runs = [[0]]
    for i in range(len(d_matrix) - 1):
        d_ij = d_matrix[i, i + 1]
        if d_ij >= d_min:
            runs[-1].append(i + 1)
        else:
            runs.append([i + 1])
    return runs


def make_block_summary_tables(linkage_blocks, aln_species, x_species_snps, aln_donors, og_id, pangenome_map, metadata, snp_frequency_cutoff, aln_consensus=None, donor_match_cutoff=1, consensus_id_head='consensus_haplotype'):
    #block_stats = pd.DataFrame(columns=['og_id', 'x_start', 'x_end', 'num_snps', 'haplotype_frequencies', 'dS_b', 'dS_1', 'dS_2', 'other_species_hits', 'donor_species'])
    #block_stats = pd.DataFrame(columns=['og_id', 'x_start', 'x_end', 'num_snps', 'haplotype_frequencies', 'db', 'pi1', 'pi2', 'd1A', 'd1Bp', 'd1C', 'd1_closest', 'd2A', 'd2Bp', 'd2C', 'd2_closest', 'other_species_hits', 'donor_species'])
    block_stats = pd.DataFrame(columns=['og_id', 'x_start', 'x_end', 'num_snps', 'haplotype_frequencies', 'db', 'dS_b', 'dN_b', 'pi1', 'piS_1', 'piN_1', 'pi2', 'piS_2', 'piN_2', 'd1A', 'dS_1A', 'dN_1A', 'd1Bp', 'dS_1Bp', 'dN_1Bp', 'd1C', 'dS_1C', 'dN_1C', 'd1_closest', 'dS1_closest', 'dN1_closest', 'd2A', 'dS_2A', 'dN_2A', 'd2Bp', 'dS_2Bp', 'dN_2Bp', 'd2C', 'dS_2C', 'dN_2C', 'd2_closest', 'dS2_closest', 'dN2_closest', 'other_species_hits', 'donor_species'])

    sag_ids = pangenome_map.get_sag_ids()
    haplotype_clustered_genomes = pd.DataFrame(index=sag_ids, columns=[])
    for i, block_idx in enumerate(linkage_blocks):
        aln_haplotypes, haplotype_seq_ids = get_linkage_block_consensus_haplotypes(aln_species, linkage_blocks[i], x_species_snps, snp_frequency_cutoff, rec_id_stem=f'{consensus_id_head}{i}')

        if len(haplotype_seq_ids) != 2:
            # Invalid number of haplotypes; can be due to large number of gaps
            continue

        block_stats.loc[i, 'og_id'] = og_id
        block_stats.loc[i, 'x_start'] = x_species_snps[np.min(block_idx)]
        block_stats.loc[i, 'x_end'] = x_species_snps[np.max(block_idx)] + 1
        block_stats.loc[i, 'num_snps'] = len(block_idx)
        #block_stats.loc[i, 'haplotype_frequencies'] = ';'.join([str(len(np.concatenate(haplotype_seq_ids[label]))) for label in haplotype_seq_ids])
        block_stats.loc[i, 'haplotype_frequencies'] = ';'.join([str(len(haplotype_seq_ids[label])) for label in haplotype_seq_ids])

        # Calculate between and within block divergences
        aln_block_segment = extract_block_segment(aln_species, x_species_snps, block_stats.loc[i, ['x_start', 'x_end']].values)
        pdist_segment = align_utils.calculate_fast_pairwise_divergence(aln_block_segment)

        # Get haplotype seq IDs
        db, d1, d2 = calculate_subpopulation_divergences(pdist_segment, haplotype_seq_ids[1], haplotype_seq_ids[2])
        block_stats.loc[i, 'db'] = db
        block_stats.loc[i, 'pi1'] = d1
        block_stats.loc[i, 'pi2'] = d2
        pN_segment, pS_segment = seq_utils.calculate_pairwise_pNpS(aln_block_segment)
        dN_b, dN_1, dN_2 = calculate_subpopulation_divergences(pN_segment, haplotype_seq_ids[1], haplotype_seq_ids[2])
        block_stats.loc[i, 'dN_b'] = dN_b
        block_stats.loc[i, 'piN_1'] = dN_1
        block_stats.loc[i, 'piN_2'] = dN_2
        dS_b, dS_1, dS_2 = calculate_subpopulation_divergences(pS_segment, haplotype_seq_ids[1], haplotype_seq_ids[2])
        block_stats.loc[i, 'dS_b'] = dS_b
        block_stats.loc[i, 'piS_1'] = dS_1
        block_stats.loc[i, 'piS_2'] = dS_2

        aln_block = construct_consensus_haplotype_alignments(aln_haplotypes, aln_donors, block_idx, x_species_snps)
        block_pdist = align_utils.calculate_fast_pairwise_divergence(aln_block)
        block_pN, block_pS = seq_utils.calculate_pairwise_pNpS(aln_block)
        p_match = donor_match_cutoff / aln_haplotypes.get_alignment_length()
        pdist_idx = np.array(block_pdist.index)
        hit_genes_str = []
        for i_hapl, rec in enumerate(aln_haplotypes):
            h = rec.id
            h_index = pdist_idx[len(aln_haplotypes):]
            h_pdist = block_pdist.loc[h, h_index]
            hit_genes_str.append(','.join(h_pdist[h_pdist <= p_match].index))
            block_stats.loc[i, f'd{i_hapl+1}_closest'] = np.min(h_pdist)
            block_stats.loc[i, f'dN{i_hapl+1}_closest'] = np.min(block_pN.loc[h, h_index])
            block_stats.loc[i, f'dS{i_hapl+1}_closest'] = np.min(block_pS.loc[h, h_index])
        block_stats.loc[i, 'other_species_hits'] = ';'.join(hit_genes_str)
        block_stats.loc[i, 'donor_species'] = ';'.join(get_donor_species([gene_str.split(',') for gene_str in hit_genes_str], pangenome_map, metadata))

        # Assign haplotype clusters
        ih = 1
        for key in haplotype_seq_ids:
            #for j, seq_ids in enumerate(haplotype_seq_ids[key]):
            #    gene_sag_ids = [pangenome_map.get_gene_sag_id(g) for g in seq_ids]
            #    haplotype_clustered_genomes.loc[gene_sag_ids, f'{og_id}_block{i}'] = ih * 10 + j
            seq_ids = haplotype_seq_ids[key]
            gene_sag_ids = [pangenome_map.get_gene_sag_id(g) for g in seq_ids]
            haplotype_clustered_genomes.loc[gene_sag_ids, f'{og_id}_block{i}'] = 10 * ih
            ih += 1

        if aln_consensus is not None:
            aln_block_consensus = align_utils.stack_alignments([aln_haplotypes[:2], aln_consensus[:, block_stats.loc[i, 'x_start']:block_stats.loc[i, 'x_end']]])
            block_consensus_pdist = align_utils.calculate_fast_pairwise_divergence(aln_block_consensus)
            block_consensus_pN, block_consensus_pS = seq_utils.calculate_pairwise_pNpS(aln_block_consensus)
            pdist_idx = np.array(block_consensus_pdist.index)
            consensus_species = [rec.id.strip('_main_cloud_consensus') for rec in aln_block_consensus[2:]]
            for i_hapl in range(2):
                for i_species in range(len(pdist_idx) - 2):
                    key = f'd{i_hapl+1}{consensus_species[i_species]}'
                    block_stats.loc[i, key] = block_consensus_pdist.loc[pdist_idx[i_hapl], pdist_idx[i_species + 2]]
                    block_stats.loc[i, f'dN_{i_hapl+1}{consensus_species[i_species]}'] = block_consensus_pN.loc[pdist_idx[i_hapl], pdist_idx[i_species + 2]]
                    block_stats.loc[i, f'dS_{i_hapl+1}{consensus_species[i_species]}'] = block_consensus_pS.loc[pdist_idx[i_hapl], pdist_idx[i_species + 2]]


    return block_stats, haplotype_clustered_genomes


def get_linkage_block_consensus_haplotypes(aln_species, block_idx, x_species_snps, snp_frequency_cutoff, rec_id_stem='consensus_haplotype'):
    x0 = x_species_snps[min(block_idx)]
    x1 = x_species_snps[max(block_idx)] + 1

    aln_arr = []
    haplotype_alleles = get_linkage_block_haplotype_seq_ids(aln_species[:, x0:x1], snp_frequency_cutoff=snp_frequency_cutoff)
    for haplotype_id in haplotype_alleles:
        haplotype_seq_ids = haplotype_alleles[haplotype_id]
        aln_haplotype = align_utils.get_subsample_alignment(aln_species[:, x0:x1], haplotype_seq_ids)
        haplotype_consensus_seq = seq_utils.get_consensus_seq(aln_haplotype)
        aln_arr.append(haplotype_consensus_seq)

    aln_arr = np.array(aln_arr)
    rec_ids = [f'{rec_id_stem}{i}' for i in range(len(haplotype_alleles))]
    return align_utils.convert_array_to_alignment(aln_arr, id_index=rec_ids), haplotype_alleles


def get_linkage_block_haplotype_seq_ids(aln_block, snp_frequency_cutoff=0, num_haplotypes=2):
    aln_hf_snps, _ = align_utils.get_high_frequency_snp_alignment(aln_block, snp_frequency_cutoff, counts_filter=True)
    block_pdist = align_utils.calculate_fast_pairwise_divergence(aln_hf_snps)
    linkage_matrix, filtered_idx = align_utils.cluster_divergence_matrix(block_pdist.values.astype(float))
    if len(filtered_idx) > 1:
        pdist_idx = np.array(block_pdist.index)[filtered_idx]
        clusters = hclust.fcluster(linkage_matrix, num_haplotypes, criterion='maxclust')
        haplotype_map = {}
        unique_clusters, _ = utils.sorted_unique(clusters)
        for c in unique_clusters:
            haplotype_map[c] = pdist_idx[clusters == c]
    else:
        haplotype_map = {}

    return haplotype_map


def construct_consensus_haplotype_alignments(aln_haplotypes, aln_nonspecies, block_idx, x_snps):
    x0 = x_snps[min(block_idx)]
    x1 = x_snps[max(block_idx)] + 1
    return align_utils.stack_alignments([aln_haplotypes, aln_nonspecies[:, x0:x1]])
   

def extract_block_segment(aln, x_snps, x_segment):
    '''
    Extracts a segment of `aln` with edges at `x_segment`. If edges are within codons 
        nearest codon start and end positions containing `x_segment` are used instead.

    OLD: Extracts segment between halfway distance from edges given by `x_segment`
    and nearest SNPs.
    '''

    '''
    # Get edges
    xl_arr = x_snps[x_snps < x_segment[0]]
    if len(xl_arr) > 0:
        xl = xl_arr[-1]
        x0 = int(np.ceil((x_segment[0] + xl) / 2))
    else:
        x0 = x_segment[0]

    xr_arr = x_snps[x_snps > x_segment[1]]
    if len(xr_arr) > 0:
        xr = xr_arr[0]
        x1 = int(np.floor((x_segment[1] + xr) / 2))
    else:
        x1 = x_segment[1]
    '''

    if (x_segment[0] % 3) != 0:
        x0 = x_segment[0] - (x_segment[0] % 3)
    else:
        x0 = x_segment[0]

    if (x_segment[1] % 3) != 2:
        dx1 = 2 - (x_segment[1] % 3)
        x1 = x_segment[1] + dx1
    else:
        x1 = x_segment[1]

    return aln[:, x0:x1 + 1]


def calculate_subpopulation_divergences(pdist, pop1_idx, pop2_idx):
    db = np.mean(pdist.loc[pop1_idx, pop2_idx].values)
    d1 = np.mean(utils.get_matrix_triangle_values(pdist.loc[pop1_idx, pop1_idx].values, k=1))
    d2 = np.mean(utils.get_matrix_triangle_values(pdist.loc[pop2_idx, pop2_idx].values, k=1))
    return db, d1, d2


def get_donor_species(hit_gene_ids, pangenome_map, metadata):
    donor_species = []
    for id_list in hit_gene_ids:
        hit_species = get_gene_species(id_list, pangenome_map, metadata)
        if 'C' in hit_species:
            donor_species.append('C')
        elif len(hit_species) > 0:
            donors, donor_counts = utils.sorted_unique(hit_species)
            donor_species.append(donors[0])
        else:
            donor_species.append('')
    return donor_species


def get_gene_species(gene_ids, pangenome_map, metadata):
    species = []
    for g in gene_ids:
        s = pangenome_map.get_gene_sag_id(g)
        s_species = metadata.get_sag_species(s)
        if s_species is None:
            species.append('')
        else:
            species.append(s_species)
    return species

def read_og_id_from_fname(f_aln):
    id_number = f_aln.split('YSG_')[-1].split('_')[0]
    return f'YSG_{id_number}'


#def calculate_linkage_runs(f_aln, pangenome_map, metadata, snp_frequency_cutoff, rsq_min, min_block_length):
def calculate_linkage_runs(f_aln, pangenome_map, metadata, args):
    aln = align_utils.read_main_cloud_alignment(f_aln, pangenome_map, metadata, dc_dict={'A':args.main_cloud_cutoff, 'Bp':args.main_cloud_cutoff, 'C':0.})
    species_sorted_gene_ids = align_utils.sort_aln_rec_ids(aln, pangenome_map, metadata)

    if args.species in species_sorted_gene_ids:
        species_gene_ids = species_sorted_gene_ids[args.species]
        linkage_blocks, _ = find_linkage_blocks(aln, species_gene_ids, snp_frequency_cutoff=args.snp_frequency_cutoff, rsq_min=args.rsq_min, min_block_length=args.min_block_length)
    else:
        linkage_blocks = []

    return linkage_blocks


def test_block_finder(args):
    pangenome_map = PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    metadata = MetadataMap()

    '''
    og_id = 'YSG_0556'
    species = 'Bp'
    sites = '4D'
    f_aln = f'../results/single-cell/snp_blocks/synbp/{og_id}_4D_aln.fna'

    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    sampled_sag_ids = species_sorted_sags[species]

    block_stats, haplotype_clustered_genomes = summarize_linkage_blocks(f_aln, og_id, sampled_sag_ids, pangenome_map, metadata, args)
    og_id = 'YSG_0622b'
    species = 'Bp'
    sites = 'all'
    f_aln = f'../results/single-cell/alignments/core_ogs_cleaned/{og_id}_cleaned_aln.fna'
    '''
    og_id = 'YSG_0608b'
    species = 'Bp'
    sites = 'all'
    f_aln = f'../results/single-cell/alignments/v2/core_ogs_cleaned/{og_id}_cleaned_aln.fna'

    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    sampled_sag_ids = species_sorted_sags[species]

    block_stats, haplotype_clustered_genomes = summarize_linkage_blocks(f_aln, og_id, sampled_sag_ids, pangenome_map, metadata, args)
    print(block_stats)
    print(haplotype_clustered_genomes)

    print(block_stats.iloc[0, :])


if __name__ == '__main__':
    pangenome_dir = '../results/single-cell/sscs_pangenome/'
    f_orthogroup_table = f'{pangenome_dir}filtered_orthogroups/sscs_annotated_single_copy_orthogroup_presence.tsv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--input_dir', default=None, help='Directory with alignment files.')
    parser.add_argument('-a', '--alignments_file', default=None, help='File with paths to input alignments.')
    parser.add_argument('-f', '--snp_frequency_cutoff', type=int, default=3, help='Cutoff for frequency of SNPs included in blocks.')
    parser.add_argument('-g', '--orthogroup_table', default=f_orthogroup_table, help='File with orthogroup table.')
    parser.add_argument('-i', '--input_alignment', default=None, help='Input alignment file.')
    parser.add_argument('-l', '--min_block_length', type=int, default=5, help='Minimum number of SNPs in significant blocks.')
    parser.add_argument('-o', '--output_file', default=None, help='Path to results file.')
    parser.add_argument('-r', '--rsq_min', type=float, default=0.99999, help='Minimum r^2 between consecutive SNPs in same block.')
    parser.add_argument('-s', '--sampled_sags_file', default=None, help='File containing subsample of SAG IDs to include in analysis.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    parser.add_argument('--out_sample_sags_file', default=None, help='File containing SAG IDs for candidate donors.')
    parser.add_argument('--species', default='A', help='Species in which to look for blocks.')
    parser.add_argument('--og_id', help='OG ID of input alignment.')
    parser.add_argument('--match_cutoff', type=int, default=1, help='Max number of SNPs away for block matches.')
    parser.add_argument('--main_cloud_cutoff', type=float, default=0.05, help='Divergence cutoff from consensus for species main cloud definition.')
    parser.add_argument('--get_linkage_run_lengths', action='store_true', help='Calculate concordant run lengths for all alignments.')
    parser.add_argument('--test_all', action='store_true', help='Run tests.')
    args = parser.parse_args()

    pangenome_map = PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    metadata = MetadataMap()

    if args.sampled_sags_file is not None:
        with open(args.sampled_sags_file, 'r') as in_handle:
            sampled_sag_ids = [line.strip() for line in in_handle.readlines()]
    else:
        # Use species SAG IDs
        sag_ids = pangenome_map.get_sag_ids()
        species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
        sampled_sag_ids = species_sorted_sags[args.species]

    if args.out_sample_sags_file is not None:
        with open(args.out_sample_sags_file, 'r') as in_handle:
            out_sample_sag_ids = [line.strip() for line in in_handle.readlines()]
    else:
        out_sample_sag_ids = None

    if args.test_all == True:
        test_block_finder(args)

    if args.input_alignment is not None:
        block_stats, haplotype_clustered_genomes = summarize_linkage_blocks(args.input_alignment, args.og_id, sampled_sag_ids, pangenome_map, metadata, args, out_sample_sag_ids=out_sample_sag_ids, consensus_id_head=f'{args.og_id}_{args.species}_consensus')

        if args.verbose:
            print(block_stats)
            print(haplotype_clustered_genomes)

    elif args.alignments_file is not None:
        # Read input files
        input_files = []
        with open(args.alignments_file, 'r') as in_handle:
            for line in in_handle.readlines():
                input_files.append(line.strip())

        sag_ids = pangenome_map.get_sag_ids()
        species_sorted_sags = metadata.sort_sags(sag_ids, by='species')

        if args.get_linkage_run_lengths:
            run_lengths = []
            for f_aln in input_files:
                if args.verbose:
                    print(f_aln)

                linkage_blocks = calculate_linkage_runs(f_aln, pangenome_map, metadata, args)
                run_lengths.append([len(block) for block in linkage_blocks])
            pickle.dump(run_lengths, open(args.output_file, 'wb'))

        else:
            block_stats_tables = []
            coarse_grained_genomes = []
            for f_aln in input_files:
                og_id = read_og_id_from_fname(f_aln)
                block_stats, haplotype_clustered_genomes = summarize_linkage_blocks(f_aln, og_id, sampled_sag_ids, pangenome_map, metadata, args, out_sample_sag_ids=out_sample_sag_ids, consensus_id_head=f'{og_id}_{args.species}_consensus')
                block_stats_tables.append(block_stats)
                coarse_grained_genomes.append(haplotype_clustered_genomes.loc[species_sorted_sags[args.species], :])

                if args.verbose:
                    print(og_id, block_stats, '\n')

            # Save results
            merged_block_stats = pd.concat(block_stats_tables, ignore_index=True)
            f_block_stats = args.output_file.replace('.tsv', '_block_stats.tsv')
            merged_block_stats.to_csv(f_block_stats, sep='\t', index=False)

            block_haplotypes_df = pd.concat(coarse_grained_genomes, axis=1)
            f_block_haplotypes = args.output_file.replace('.tsv', '_block_haplotypes.tsv')
            block_haplotypes_df.to_csv(f_block_haplotypes, sep='\t')


