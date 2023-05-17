import argparse
import numpy as np
import scipy.stats as spstats
import pandas as pd
import pickle
import utils
import os
import glob
import alignment_tools as align_utils
import re
import networkx as nx
import scipy.cluster.hierarchy as hclust
import scipy.spatial.distance as distance
#from pangenome_utils import PangenomeMap
from syn_homolog_map import SynHomologMap
from metadata_map import MetadataMap
from Bio import SeqIO
from Bio import AlignIO
from Bio import SearchIO
from Bio import Align
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq, translate
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.SeqRecord import SeqRecord
from Bio.Blast import NCBIXML
from Bio.Align import substitution_matrices


class AlleleFreqTable:
    def __init__(self, random_seed=12345):
        np.random.seed(random_seed)

    def __repr__(self):
        return f'{self.locus_alleles}\n{self.allele_counts}'

    def make_tables(self, contig_genes_dict, locus_gene_tag_map):
        self.get_loci_geneIDs(locus_gene_tag_map)
        self.extract_gene_seqs(contig_genes_dict, locus_gene_tag_map)
        self.calculate_allele_freqs()

    def get_loci_geneIDs(self, locus_map):
        self.locus_geneIDs = {}
        for locus, data in locus_map.iterrows():
            self.locus_geneIDs[locus] = [np.random.choice(id_list)  if len(id_list) > 0 else 'none' for id_list in data]
            if locus == 'CYA_1867':
                print(data)
                print(self.locus_geneIDs[locus])

    def extract_gene_seqs(self, contig_genes_dict, locus_gene_tag_map):
        self.locus_gene_seqs = {}
        self.sag_ids = list(locus_gene_tag_map.columns)
        for locus, gene_ids in self.locus_geneIDs.items():
            self.locus_gene_seqs[locus] = []
            for i, gene_id in enumerate(gene_ids):
                if gene_id != 'none':
                    contig_table = contig_genes_dict[self.sag_ids[i]]['contigs']
                    genes_table = contig_genes_dict[self.sag_ids[i]]['genes']
                    self.locus_gene_seqs[locus].append(get_gene_seq(contig_table, gene_id, genes_table.loc[gene_id, 'strand_direction']))
                else:
                    self.locus_gene_seqs[locus].append('')

    def calculate_allele_freqs(self):
        self.locus_alleles = {}
        self.allele_counts = {}
        for locus, gene_seqs in self.locus_gene_seqs.items():
            if len(gene_seqs) > 2:
                alleles, allele_counts = utils.sorted_unique(gene_seqs)
            elif len(gene_seqs) == 2:
                # length 2 lists give strange output so have to do manually
                if gene_seqs[0] == gene_seqs[1]:
                    alleles = np.array([gene_seqs[0]])
                    allele_counts = 2 * np.ones(1)
                else:
                    alleles = gene_seqs
                    allele_counts = np.ones(2)
            elif len(gene_seqs) == 1:
                alleles = gene_seqs
                allele_counts = np.ones(1)
            else:
                alleles = []
                allele_counts = []
            self.locus_alleles[locus] = alleles
            self.allele_counts[locus] = allele_counts

    def get_sagID(self, i):
        return self.sag_ids[i]


class BifurcationTree:
    def __init__(self, aln=None, x0=None):
        self.graph = nx.DiGraph()
        self.graph.add_node('root')

        if aln and x0:
            self.build_bifurcation_tree(aln, x0)

    def __repr__(self):
        return f'{self.left_tree.name}-Left Tree: {self.left_tree}\n{self.right_tree.name}-right Tree: {self.right_tree}\n'

    def build_bifurcation_tree(self, aln, x0):
        aln_arr = np.array(aln)
        haplotuples = make_haplotype_tuples(aln_arr[:, [x0]])

        for hapl in haplotuples:
            self.graph.add_node(hapl[0])
            self.graph.add_node(f'{hapl[0]}_left')
            self.graph.add_node(f'{hapl[0]}_right')
            self.graph.add_edge('root', hapl[0], weight=hapl[1])
            self.graph.add_edge(hapl[0], f'{hapl[0]}_left', weight=hapl[1])
            self.graph.add_edge(hapl[0], f'{hapl[0]}_right', weight=hapl[1])

        for direction in ['left', 'right']:
            level_nodes = [f'{node}_{direction}'for node in self.graph.successors('root')]

            while len(level_nodes) > 0:
                for parent_node in level_nodes:
                    parent_haplotype = parent_node.strip(f'_{direction}')
                    haplotuples = self.get_child_haplotype_tuples(aln_arr, parent_haplotype, x0, direction=direction)

                    for hapl in haplotuples:
                        if hapl[0] != parent_haplotype:
                            node_name = f'{hapl[0]}_{direction}'
                            self.graph.add_node(node_name)
                            self.graph.add_edge(parent_node, node_name, weight=hapl[1])
                level_nodes = self.update_level_nodes(level_nodes)

    def get_child_haplotype_tuples(self, aln_arr, parent_haplotype, x0, direction='right'):
        parent_aln = self.filter_alignment_haplotype(aln_arr, parent_haplotype, x0, direction=direction)
        if direction == 'right':
            if x0 + len(parent_haplotype) + 1 <= aln_arr.shape[1]:
                haplotuples = make_haplotype_tuples(parent_aln[:, x0:x0 + len(parent_haplotype) + 1])
            else:
                haplotuples = []
        else:
            if x0 - len(parent_haplotype) >= 0:
                haplotuples = make_haplotype_tuples(parent_aln[:, x0 - len(parent_haplotype):x0 + 1])
            else:
                haplotuples = []
        return haplotuples

    def filter_alignment_haplotype(self, aln_arr, haplotype, x0, direction='right'):
        if direction == 'right':
            x = x0 + len(haplotype)
            initial_haplotypes = np.array([''.join(seq) for seq in aln_arr[:, x0:x]])
        else:
            x = x0 - len(haplotype) + 1
            initial_haplotypes = np.array([''.join(seq) for seq in aln_arr[:, x:x0 + 1]])
        return aln_arr[initial_haplotypes == haplotype, :]

    def update_level_nodes(self, level_nodes):
        new_nodes = []
        for node in level_nodes:
            new_nodes += self.graph.successors(node)
        return new_nodes

    def export_graphs(self, output_file=None, output_name_stem=None, left_right_fnames=None, format='adjlist'):
        if output_file:
            if format == 'adjlist':
                nx.write_adjlist(self.graph, output_file)
            elif format == 'dat':
                pickle.dump(self.graph, open(output_file, 'wb'))
        else:
            if left_right_fnames is None:
                # Make separate files for left and right bifurcation graphs
                f_left_tree = re.sub(f'.{format}$', '', output_name_stem) + f'_left.{format}'
                f_right_tree = re.sub(f'.{format}$', '', output_name_stem) + f'_right.{format}'
            else:
                f_left_tree, f_right_tree = left_right_fnames

            if format == 'adjlist':
                nx.write_adjlist(self.graph, f_right_tree)
            elif format == 'dat':
                pickle.dump(self.graph, open(f_right_tree, 'wb'))

def make_haplotype_tuples(aln_arr):
    haplotypes = [''.join(seq) for seq in aln_arr]
    unique_haplotypes, haplotype_counts = utils.sorted_unique(haplotypes)
    return [(unique_haplotypes[i], haplotype_counts[i]) for i in range(len(unique_haplotypes))]


def get_gene_seq(contigs_table, gene_id, strand):
    contig_id, location = split_gene_id(gene_id)
    feature = SeqFeature(FeatureLocation(location[0] - 1, location[1]), type='gene')
    contig_seq = contigs_table.loc[contig_id, 'sequence']
    gene_seq = feature.extract(contig_seq)

    # Check if gene on reverse DNA strand
    if strand == '-':
        gene_seq = gene_seq.reverse_complement()
    return gene_seq

def split_gene_id(gene_id):
    id_comps = gene_id.split('_')
    return '_'.join(id_comps[:-2]), (int(id_comps[-2]), int(id_comps[-1]))


class AlignmentStats:
    def __init__(self, aln=None, locus='none', max_frac_gaps=0.25):
        self.locus = locus
        self.max_frac_gaps = max_frac_gaps
        if aln is not None:
            self.length = len(aln[0])
            self.read_sample_ids(aln)
            self.calculate_pairwise_divergences(aln)
        else:
            self.length = 0
            self.sample_ids = []
            self.pairwise_divergences = {}


    def __repr__(self):
        return f'AlignmentStats:\n\talignment length={self.length}\n\tsamples={self.sample_ids}\n\tpairwise divergences={self.pairwise_divergences}'

    def read_sample_ids(self, aln):
        self.sample_ids = []
        for record in aln:
            gap_frac = np.sum(np.array(record.seq) == '-') / len(record.seq)
            if gap_frac < self.max_frac_gaps:
                self.sample_ids.append(record.id)

    def calculate_pairwise_divergences(self, aln, max_frac_gaps=0.25):
        self.pairwise_divergences = {}
        for i, rec_i in enumerate(aln):
            # Skip seqs with large number of gaps
            gap_frac_i = np.sum(np.array(rec_i.seq) == '-') / len(rec_i.seq)
            if gap_frac_i >= max_frac_gaps:
                continue
            for j in range(i):
                rec_j = aln[j]
                # Skip seqs with large number of gaps
                gap_frac_j = np.sum(np.array(rec_j.seq) == '-') / len(rec_j.seq)
                if gap_frac_j >= max_frac_gaps:
                    continue

                pair = tuple(sorted((rec_i.id, rec_j.id)))
                self.pairwise_divergences[pair] = self.calculate_pair_divergence(rec_i.seq, rec_j.seq)
                '''
                if self.pairwise_divergences[pair] == 0:
                    print(pair)
                    print(aln[i, :].seq)
                    print(aln[j, :].seq)
                    print('\n')
                '''

    def calculate_pair_divergence(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        seq1_arr = np.array(seq1)
        seq2_arr = np.array(seq2)
        non_gaps = (seq1_arr != '-') * (seq2_arr != '-')
        matches = np.sum(seq1_arr[non_gaps] == seq2_arr[non_gaps])
        #matches = np.sum(np.array(seq1) == np.array(seq2))
        #return 1 - matches / len(seq1)
        return 1 - matches / np.sum(non_gaps)

def get_pdist_values(pdist_df):
    triu_indices = np.triu_indices(pdist_df.shape[0], k=1)
    divergences = pdist_df.values[triu_indices]
    return divergences

def build_visual_contig(contig_table_row, gene_strands=None, polarize=False):
    gene_ids = contig_table_row['lowest_tagged_hits']
    num_genes = len(gene_ids)
    gene_locations = contig_table_row['gene_locations']
    if gene_strands is None:
        gene_strands = contig_table_row['lowest_tagged_hit_strands']
    if polarize == True:
        global_polarization = sum(gene_strands)
        print(gene_strands, global_polarization)
        if global_polarization < 0:
            gene_locations = [(-x[1], -x[0]) for x in gene_locations]
            gene_strands = -np.array(gene_strands)

    visual_contig = []
    for i in range(num_genes):
        gene = VisualGene(gene_ids[i], strand=gene_strands[i], location=gene_locations[i])
        visual_contig.append(gene)
    return visual_contig

def construct_artificial_reference_genomes(syn_homolog_map):
    os_cds = list(syn_homolog_map.cds.values())
    osa_gene_tags = [cds.qualifiers['locus_tag'][0] for cds in os_cds if 'CYA' in cds.qualifiers['locus_tag'][0]]
    osa_artificial_genome = map_gene_families(osa_gene_tags, syn_homolog_map)
    osbp_gene_tags = [cds.qualifiers['locus_tag'][0] for cds in os_cds if 'CYB' in cds.qualifiers['locus_tag'][0]]
    osbp_artificial_genome = map_gene_families(osbp_gene_tags, syn_homolog_map)
    return {'osa':osa_artificial_genome, 'osbp':osbp_artificial_genome}

def map_gene_families(gene_tags, syn_homolog_map):
    gene_families = []
    for tag in gene_tags:
        gene_families.append(syn_homolog_map.find_lowest_tagged_homolog(tag))
    return gene_families

def make_test_data(output_file=f'../results/tests/genome_divergence_tables.dat'):
    metadata = MetadataMap()
    typical_sags = ['UncmicORedA02I17_FD', 'UncmicORedA02J13_FD', 'UncmicOcRedA1L21_FD', 'UncmicOcRedA3C21_FD']
    sag_data_tables = pickle.load(open(f'../results/single-cell/contig_and_genes_tables.sscs.dat', 'rb'))
    test_sags = {}
    for sag_id in typical_sags:
        test_sags[sag_id] = sag_data_tables[sag_id]
    pickle.dump(test_sags, open(f'../results/tests/typical_sag_data_tables.dat', 'wb'))

def merge_species_alleles(allele_dict):
    loci = get_merged_allele_dict_loci(allele_dict)
    merged_allele_dict = {}

def get_merged_allele_dict_loci(allele_dict):
    loci = set()
    for species in allele_dict:
        loci = loci.union(set(list(allele_dict[species].keys())))
    return list(loci)

def calculate_ng86_pdist(seq1, seq2):
    '''
    Uses method II form Nei & Gojobori (1986) to calculate fraction of substituions
    between two sequences.

    '''

    codon_dict = CodonSitesDict()

    num_codons = 0
    num_nongapped_codons = 0
    S = 0 # num synonymous sites
    N = 0 # num non-synonymous sites
    m_s = np.zeros(3) # num substitution sites at codons with single difference
    m_ss = np.zeros(3) # num synonymous sites at codons with single difference
    m_m = np.zeros(3) # num substitution sites at codons with multiple differences

    if len(seq1) == len(seq2):
        # Read sequence codon by codon
        for i in range(0, len(seq1), 3):
            codon1 = seq1.seq[i:i + 3]
            codon2 = seq2.seq[i:i + 3]

            if '-' in codon1 or '-' in codon2:
                #print(i, codon1, codon2)
                # Ignore codon if gaps are present
                continue
            elif 'N' in codon1 or 'N' in codon2:
                # Ignore codons with 'N's
                continue
            elif len(codon1) == 3 and len(codon2):
                # Gaps that are not multiples of 3 can lead to bad reading frame
                # Ignoring for now. TODO: find workaround
                num_nongapped_codons += 1
                f1 = codon_dict.get_synonymous_substitutions_fraction(codon1)
                f2 = codon_dict.get_synonymous_substitutions_fraction(codon2)
                site_diffs = np.array(codon1) != np.array(codon2)
                num_diffs = np.sum(site_diffs, dtype=np.int64)
                if num_diffs == 1:
                    m_s[site_diffs] += 1
                    m_ss[site_diffs] += (f1[site_diffs] + f2[site_diffs]) / 2
                    #print(codon1, codon2)
                elif num_diffs > 1:
                    m_m[site_diffs] += 1
                    #print(site_diffs, codon1, codon2)
                S += np.sum((f1 + f2) / 2)
                N += 3 - np.sum((f1 + f2) / 2)

    #print(m_ss, m_s, num_nongapped_codons)
    pi_s = m_ss / (m_s + (m_s == 0)) # avoid division by zero
    S_d = np.dot(m_s + m_m, pi_s)
    N_d = np.dot(m_s + m_m, 1 - pi_s)

    #print(S_d, S, N_d, N, S_d + N_d, S + N)

    return N_d / N, S_d / S


class CodonSitesDict:
    def __init__(self):
        self.initialize_synonymous_fraction_dict()

    def initialize_synonymous_fraction_dict(self):
        '''
        Manually define fraction of synonymous substitutions at each
        codon site from bacterial genetic code (table 11).
        '''
        self.codon_fraction_synonymous = {}
        self.codon_fraction_synonymous['TTT'] = np.array([0, 0, 1./3]) # Phenylalanine (F)
        self.codon_fraction_synonymous['TTC'] = np.array([0, 0, 1./3]) # Phenylalanine (F)
        self.codon_fraction_synonymous['TTA'] = np.array([1./3, 0, 1./3]) # Leucine (L)
        self.codon_fraction_synonymous['TTG'] = np.array([1./3, 0, 1./3]) # Leucine (L)
        self.codon_fraction_synonymous['TCT'] = np.array([0, 0, 1]) # Serine (S)
        self.codon_fraction_synonymous['TCC'] = np.array([0, 0, 1]) # Serine (S)
        self.codon_fraction_synonymous['TCA'] = np.array([0, 0, 1]) # Serine (S)
        self.codon_fraction_synonymous['TCG'] = np.array([0, 0, 1]) # Serine (S)
        self.codon_fraction_synonymous['TAT'] = np.array([0, 0, 1./3]) # Tyrosine (Y)
        self.codon_fraction_synonymous['TAC'] = np.array([0, 0, 1./3]) # Tyrosine (Y)
        self.codon_fraction_synonymous['TAA'] = np.array([0, 1./3, 1./3]) # Stop (Ochre)
        self.codon_fraction_synonymous['TAG'] = np.array([0, 0, 1./3]) # Stop (Amber)
        self.codon_fraction_synonymous['TGT'] = np.array([0, 0, 1./3]) # Cysteine (C)
        self.codon_fraction_synonymous['TGC'] = np.array([0, 0, 1./3]) # Cysteine (C)
        self.codon_fraction_synonymous['TGA'] = np.array([0, 1./3, 0]) # Stop (Opal)
        self.codon_fraction_synonymous['TGG'] = np.array([0, 0, 0]) # Tryptophan (W)

        self.codon_fraction_synonymous['CTT'] = np.array([0, 0, 1]) # Leucine (L)
        self.codon_fraction_synonymous['CTC'] = np.array([0, 0, 1]) # Leucine (L)
        self.codon_fraction_synonymous['CTA'] = np.array([1./3, 0, 1]) # Leucine (L)
        self.codon_fraction_synonymous['CTG'] = np.array([1./3, 0, 1]) # Leucine (L)
        self.codon_fraction_synonymous['CCT'] = np.array([0, 0, 1]) # Proline (P)
        self.codon_fraction_synonymous['CCC'] = np.array([0, 0, 1]) # Proline (P)
        self.codon_fraction_synonymous['CCA'] = np.array([0, 0, 1]) # Proline (P)
        self.codon_fraction_synonymous['CCG'] = np.array([0, 0, 1]) # Proline (P)
        self.codon_fraction_synonymous['CAT'] = np.array([0, 0, 1./3]) # Histidine (H)
        self.codon_fraction_synonymous['CAC'] = np.array([0, 0, 1./3]) # Histidine (H)
        self.codon_fraction_synonymous['CAA'] = np.array([0, 0, 1./3]) # Glutamine (Q)
        self.codon_fraction_synonymous['CAG'] = np.array([0, 0, 1./3]) # Glutamine (Q)
        self.codon_fraction_synonymous['CGT'] = np.array([0, 0, 1]) # Arginine (R)
        self.codon_fraction_synonymous['CGC'] = np.array([0, 0, 1]) # Arginine (R)
        self.codon_fraction_synonymous['CGA'] = np.array([1./3, 0, 1]) # Arginine (R)
        self.codon_fraction_synonymous['CGG'] = np.array([1./3, 0, 1]) # Arginine (R)

        self.codon_fraction_synonymous['ATT'] = np.array([0, 0, 2./3]) # Isoleucine (I)
        self.codon_fraction_synonymous['ATC'] = np.array([0, 0, 2./3]) # Isoleucine (I)
        self.codon_fraction_synonymous['ATA'] = np.array([0, 0, 2./3]) # Isoleucine (I)
        self.codon_fraction_synonymous['ATG'] = np.array([0, 0, 0]) # Methionine (M)
        self.codon_fraction_synonymous['ACT'] = np.array([0, 0, 1]) # Threonine (T)
        self.codon_fraction_synonymous['ACC'] = np.array([0, 0, 1]) # Threonine (T)
        self.codon_fraction_synonymous['ACA'] = np.array([0, 0, 1]) # Threonine (T)
        self.codon_fraction_synonymous['ACG'] = np.array([0, 0, 1]) # Threonine (T)
        self.codon_fraction_synonymous['AAT'] = np.array([0, 0, 1./3]) # Asparagine (N)
        self.codon_fraction_synonymous['AAC'] = np.array([0, 0, 1./3]) # Asparagine (N)
        self.codon_fraction_synonymous['AAA'] = np.array([0, 0, 1./3]) # Lysine (K)
        self.codon_fraction_synonymous['AAG'] = np.array([0, 0, 1./3]) # Lysine (K)
        self.codon_fraction_synonymous['AGT'] = np.array([0, 0, 1./3]) # Serine (S)
        self.codon_fraction_synonymous['AGC'] = np.array([0, 0, 1./3]) # Serine (S)
        self.codon_fraction_synonymous['AGA'] = np.array([1./3, 0, 1./3]) # Arginine (R)
        self.codon_fraction_synonymous['AGG'] = np.array([1./3, 0, 1./3]) # Arginine (R)

        self.codon_fraction_synonymous['GTT'] = np.array([0, 0, 1]) # Valine (V)
        self.codon_fraction_synonymous['GTC'] = np.array([0, 0, 1]) # Valine (V)
        self.codon_fraction_synonymous['GTA'] = np.array([0, 0, 1]) # Valine (V)
        self.codon_fraction_synonymous['GTG'] = np.array([0, 0, 1]) # Valine (V)
        self.codon_fraction_synonymous['GCT'] = np.array([0, 0, 1]) # Alanine (A)
        self.codon_fraction_synonymous['GCC'] = np.array([0, 0, 1]) # Alanine (A)
        self.codon_fraction_synonymous['GCA'] = np.array([0, 0, 1]) # Alanine (A)
        self.codon_fraction_synonymous['GCG'] = np.array([0, 0, 1]) # Alanine (A)
        self.codon_fraction_synonymous['GAT'] = np.array([0, 0, 1./3]) # Aspartic acid (D)
        self.codon_fraction_synonymous['GAC'] = np.array([0, 0, 1./3]) # Aspartic acid (D)
        self.codon_fraction_synonymous['GAA'] = np.array([0, 0, 1./3]) # Glutamic acid (E)
        self.codon_fraction_synonymous['GAG'] = np.array([0, 0, 1./3]) # Glutamic acid (E)
        self.codon_fraction_synonymous['GGT'] = np.array([0, 0, 1]) # Glycine (G)
        self.codon_fraction_synonymous['GGC'] = np.array([0, 0, 1]) # Glycine (G)
        self.codon_fraction_synonymous['GGA'] = np.array([0, 0, 1]) # Glycine (G)
        self.codon_fraction_synonymous['GGG'] = np.array([0, 0, 1]) # Glycine (G)

    def get_synonymous_substitutions_fraction(self, codon):
        return self.codon_fraction_synonymous[codon]

    def get_site_degeneracies(self, codon):
        f_synonymous = self.get_synonymous_substitutions_fraction(codon)
        degeneracies = []
        for f in f_synonymous:
            if f == 0:
                degeneracies.append('1D')
            elif f > 0 and f < 2./3:
                degeneracies.append('2D')
            elif f > 1./3 and f < 1:
                degeneracies.append('3D')
            else:
                degeneracies.append('4D')
        return degeneracies

    def is_codon(self, string):
        return string in self.codon_fraction_synonymous

def calculate_pairwise_pNpS(aln):
    sag_ids = [record.id for record in aln]
    pN_df = pd.DataFrame(index=sag_ids, columns=sag_ids)
    pS_df = pd.DataFrame(index=sag_ids, columns=sag_ids)
    for i in range(len(aln)):
        for j in range(i):
            seq1 = aln[i]
            seq2 = aln[j]

            pN, pS = calculate_ng86_pdist(seq1, seq2)
            pN_df.at[seq1.id, seq2.id] = pN
            pN_df.at[seq2.id, seq1.id] = pN
            pS_df.at[seq1.id, seq2.id] = pS
            pS_df.at[seq2.id, seq1.id] = pS
    return pN_df, pS_df

#def read_mafft_alignment(f_in, file_format='fasta', alphabet=IUPAC.unambiguous_dna):
def read_mafft_alignment(f_in, file_format='fasta', alphabet='generic_dna'):
    aln = AlignIO.read(f_in, file_format)
    #for record in aln:
    #    record.seq = validate_and_convert_aln_seq(record.seq, alphabet)
    for rec in aln:
        rec.seq = rec.seq.upper()
    return aln

def validate_and_convert_aln_seq(in_seq, alphabet):
    # Make sure upper case letters only
    out_seq = in_seq.upper()

    # Check for letters not in alphabet
    seq_letters = set(str(out_seq))
    if seq_letters - set(alphabet.letters):
        # Replace ambiguous letters with gaps
        seq_arr = np.array(out_seq)
        replacement_idx = [n not in alphabet.letters for n in seq_arr]
        seq_arr[replacement_idx] = '-'
        out_seq = Seq(''.join(seq_arr), alphabet)

    return out_seq

#def read_alignment_and_map_sag_ids(locus_id, alignments_dir, pangenome_map, assign_unique_ids=False):
#    f_aln = f'{alignments_dir}{locus_id}_aln.fna'
def read_alignment_and_map_sag_ids(f_aln, pangenome_map, file_format='fasta_ambiguous', assign_unique_ids=False):
    if os.path.exists(f_aln):
        aln = read_alignment(f_aln, file_format=file_format)
        aln_mapped = map_alignment_to_sags(aln, pangenome_map, assign_unique_ids)
    else:
        aln_mapped = MultipleSeqAlignment([])
    return aln_mapped


def read_alignment(f_in, file_format='fasta_unambiguous', seq_type='nucl'):
    '''
    Assumes input is FASTA. Converts alphabet to upper case if necessary and replaces
        all 'N' with '-'.
    '''

    aln = AlignIO.read(f_in, 'fasta')
    if file_format == 'mafft':
        for record in aln:
            record.seq = record.seq.upper()

    if file_format != 'fasta_unambiguous':
        for record in aln:
            record.seq = replace_ambiguous_chars(record.seq, seq_type)

    return aln

def replace_ambiguous_chars(in_seq, seq_type):
    if seq_type == 'nucl':
        alphabet = ['A', 'T', 'G', 'C']
    else:
        alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K',
                'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    seq_letters = set(str(in_seq))
    if seq_letters - set(alphabet):
        # Replace ambiguous letters with gaps
        seq_arr = np.array(in_seq)
        replacement_idx = [n not in alphabet for n in seq_arr]
        seq_arr[replacement_idx] = '-'
        out_seq = Seq(''.join(seq_arr))
    else:
        out_seq = in_seq
    return out_seq

def map_alignment_to_sags(aln, pangenome_map, assign_unique_ids):
    mapped_aln = []
    mapped_ids = []
    for rec in aln:
        mapped_id = map_gene_to_sag_id(rec.id, pangenome_map, mapped_ids, assign_unique_ids)
        mapped_rec = SeqRecord(rec.seq, id=mapped_id, description=rec.description)
        mapped_aln.append(mapped_rec)
        mapped_ids.append(mapped_id)
    return MultipleSeqAlignment(mapped_aln)

def map_gene_to_sag_id(gene_id, pangenome_map, mapped_ids, assign_unique_ids):
    naive_map = pangenome_map.get_gene_sag_id(gene_id)
    
    # Check if ID already present in list
    if assign_unique_ids == True and naive_map in mapped_ids:
        current_count = sum([naive_map in rec_id for rec_id in mapped_ids])
        mapped_id = f'{naive_map}-{current_count + 1}'
    else:
        mapped_id = naive_map
    return mapped_id

def read_seqs_and_map_sag_ids(f_seqs, pangenome_map, assign_unique_ids=False):
    with open(f_seqs, 'r') as in_handle:
        seq_records = SeqIO.parse(in_handle, 'fasta')
        mapped_seqs = []
        mapped_ids = []
        for rec in seq_records:
            mapped_id = map_gene_to_sag_id(rec.id, pangenome_map, mapped_ids, assign_unique_ids)
            mapped_rec = SeqRecord(rec.seq, id=mapped_id, description=rec.description)
            mapped_seqs.append(mapped_rec)
            mapped_ids.append(mapped_id)
        return mapped_seqs

def read_seqs(f_seqs, format='list'):
    with open(f_seqs, 'r') as in_handle:
        recs_handler = SeqIO.parse(in_handle, 'fasta')

        if format == 'list':
            seq_records = []
            for rec in recs_handler:
                seq_records.append(rec)
        elif format == 'dict':
            seq_records = {}
            for rec in recs_handler:
                seq_records[rec.id] = rec

        return seq_records

def read_blast_results(f_blast_tab, extra_columns=[]):
    blast_columns = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'] + extra_columns
    blastp_results = pd.read_csv(f_blast_tab, sep='\t', header=None, names=blast_columns)
    return blastp_results

def translate_seqs_dict(nucl_seqs):
    prot_dict = {}
    for gene_id in nucl_seqs:
        record = nucl_seqs[gene_id]
        prot_record = SeqRecord(record.seq.translate(table=11), id=record.id, description=record.description)
        prot_dict[gene_id] = prot_record
    return prot_dict

def write_seqs_dict(seq_dict, output_file, format='fasta'):
    records = list(seq_dict.values())
    SeqIO.write(records, output_file, 'fasta')


def write_seqs(seq_records, output_file, format='fasta'):
    SeqIO.write(seq_records, output_file, 'fasta')

def make_Seq(seq_str):
    return Seq(seq_str)


def merge_alignments(aln_list):
    common_ids = find_common_sag_ids(aln_list)
    sorted_alnmts = []
    for aln in aln_list:
        aln_dict = {}
        for rec in aln:
            aln_dict[rec.id] = rec
        sorted_aln = MultipleSeqAlignment([aln_dict[sag_id] for sag_id in common_ids])
        sorted_alnmts.append(sorted_aln)
    merged_alnmt = sorted_alnmts[0]
    for aln in sorted_alnmts[1:]:
        merged_alnmt = merged_alnmt + aln
    return merged_alnmt

def find_common_sag_ids(aln_list):
    sag_ids = [[rec.id for rec in aln] for aln in aln_list]
    common_ids = set(np.concatenate(sag_ids))
    for id_list in sag_ids:
        common_ids = common_ids.intersection(set(id_list))
    return list(common_ids)

def filter_alignment_gaps(alignment, gap_threshold=0):
    '''
    Removes sequences from ``alignment`` that have fraction of gaps
    higher than ``gap_threshold``.
    '''
    filtered = []
    len_aligned = len(alignment[0])
    for i, seq in enumerate(alignment):
        num_gaps = count_gaps(seq)
        if num_gaps / len_aligned <= gap_threshold:
            filtered.append(seq)
    return MultipleSeqAlignment(filtered)

def count_gaps(seq):
    return seq.seq.count('-')

def remove_seq_gaps(seq):
    out_seq = Seq(str(seq).replace('-', ''))
    return out_seq

def filter_species_alignment(merged_aln, species):
    metadata_map = MetadataMap()
    sag_ids = [rec.id for rec in merged_aln]
    species_sorted_sags = metadata_map.sort_sags(sag_ids, by='species')
    if species in ['osa', 'syna', 'A']:
        species_records = [rec for rec in merged_aln if rec.id in species_sorted_sags['A']]
    elif species in ['osbp', 'synbp', 'Bp']:
        species_records = [rec for rec in merged_aln if rec.id in species_sorted_sags['Bp']]
    else:
        print(f'Species {species} not recognized.')
        species_records = []
    return MultipleSeqAlignment(species_records)

def filter_main_cloud(aln, anchor_id="OS-B'", radius=0.1):
    anchor_seq = None
    for record in aln:
        if record.id == anchor_id:
            anchor_seq = record.seq
            break

    if anchor_seq:
        filtered_aln = []
        for record in aln:
            snps, aln_len = align_utils.count_snps(anchor_seq, record.seq)
            if snps / aln_len <= radius:
                filtered_aln.append(record)
        filtered_aln = MultipleSeqAlignment(filtered_aln)
    else:
        print(f'Anchor ID {anchor_id} not found in alignment. Returning unfiltered alignment.')
        filtered_aln = aln

    return filtered_aln

def back_translate(aa_aln, original_seqs, replace_ambiguous_chars=False):
    nucl_aln = []
    for rec in aa_aln:
        num_gaps = 0
        aln_seq = []
        nucl_seq = original_seqs[rec.id]
        for i in range(len(rec.seq)):
            if rec[i] == '-':
                aln_seq.append('---')
                num_gaps += 1
            else:
                idx = 3 * (i - num_gaps)
                site_codon_str = str(nucl_seq[idx:(idx + 3)].seq)
                if replace_ambiguous_chars == True and 'N' in site_codon_str:
                    aln_seq.append('---')
                else:
                    aln_seq.append(site_codon_str)
        nucl_aln.append(SeqRecord(Seq(''.join(aln_seq)), id=rec.id, description=rec.description))
    return MultipleSeqAlignment(nucl_aln)


def get_alignment_site_degeneracies(aln):
    codon_dict = CodonSitesDict()
    nucl_aln_arr = np.array(aln)
    prot_aln_arr = np.array(translate_nucl_alnmt(aln))
    consensus_prot = get_consensus_seq(prot_aln_arr)

    site_degeneracies = {'1D':[], '2D':[], '3D':[], '4D':[], 'gaps':[]}
    if len(prot_aln_arr.shape) > 1:
        num_sites = prot_aln_arr.shape[1]
    else:
        num_sites = 0

    for sa in range(num_sites):
        sn = 3 * sa 

        # Find consensus codon conditioned on consensus AA
        consensus_aa = consensus_prot[sa]
        if consensus_aa == '-':
            site_degeneracies['gaps'] += list(np.arange(sn, sn + 3))
        else:
            site_codons = np.array([''.join(codon_lett) for codon_lett in nucl_aln_arr[:, sn:sn + 3]])
            consensus_aa_codons = site_codons[prot_aln_arr[:, sa] == consensus_aa]
            unique_codons, codon_counts = utils.sorted_unique(consensus_aa_codons)
            consensus_codon = unique_codons[0]

            codon_degeneracies = codon_dict.get_site_degeneracies(consensus_codon)
            for i, degeneracy in enumerate(codon_degeneracies):
                site_degeneracies[degeneracy].append(sn + i)

    return site_degeneracies


def get_synonymous_sites(nucl_aln, return_x=False):
    '''
    This mostly includes 4-fold degenerate sites, but also adds 2-fold degenerate sites in codons coding for L and R
    '''
    degenerate_aa = ['L', 'V', 'P', 'T', 'A', 'R', 'G'] # 4-fold degenerate AAs; treat 'S' separately

    nucl_arr = np.array(nucl_aln)
    prot_aln = translate_nucl_alnmt(nucl_aln)
    aa_arr = np.array(prot_aln)
    syn_sites = []
    x_syn_sites = []
    for i in range(len(prot_aln[0])):
        site_aa = aa_arr[:, i]
        aa_unique, aa_counts = utils.sorted_unique(site_aa[site_aa != '-'])
        if len(aa_unique) > 0:
            most_common_aa = aa_unique[0]
            if most_common_aa in degenerate_aa:
                syn_sites.append(nucl_arr[:, 3*i+2])
                x_syn_sites.append(3 * i + 2)
            elif most_common_aa == 'S':
                # Check if first codon letter is T
                s1_unique, s1_counts = utils.sorted_unique(nucl_arr[:, 3*i])
                if s1_unique[0] == 'T':
                    syn_sites.append(nucl_arr[:, 3*i+2])
                    x_syn_sites.append(3 * i + 2)

    # Convert back to alignment
    syn_sites = np.array(syn_sites).T
    records = [SeqRecord(Seq(''.join(syn_sites[i, :])), id=nucl_aln[i].id, description=nucl_aln[i].description) for i in range(len(nucl_aln))]

    if return_x == False:
        return MultipleSeqAlignment(records)
    else:
        return MultipleSeqAlignment(records), np.array(x_syn_sites, dtype=int)


def get_nonsynonymous_sites(aln, return_x=False):
    codon_dict = CodonSitesDict()
    nucl_aln_arr = np.array(aln)
    prot_aln_arr = np.array(translate_nucl_alnmt(aln))

    # Find consensus protein sequence
    consensus_prot = get_consensus_seq(prot_aln_arr)

    nonsynonymous_sites = []
    for j in range(len(consensus_prot)):
        i = 3 * j

        # Find consensus codon conditioned on consensus AA
        consensus_aa = consensus_prot[j]
        site_codons = np.array([''.join(codon_lett) for codon_lett in nucl_aln_arr[:, i:i + 3]])
        consensus_aa_codons = site_codons[prot_aln_arr[:, j] == consensus_aa]
        unique_codons, codon_counts = utils.sorted_unique(consensus_aa_codons)
        consensus_codon = unique_codons[0]

        # Define nonsynonymous sites as sites that are 1D in non-gap consensus codons
        if consensus_codon != '---':
            # Skip gap codons
            is_nonsynonymous = (codon_dict.get_synonymous_substitutions_fraction(consensus_codon) == 0)
            nonsynonymous_sites += list(np.arange(i, i + 3)[is_nonsynonymous])

    nonsynonymous_aln = align_utils.get_alignment_sites(aln, nonsynonymous_sites)

    if return_x == False:
        return nonsynonymous_aln
    else:
        return nonsynonymous_aln, np.array(nonsynonymous_sites)


def get_snps(aln, return_x=False):
    '''
    Takes alignment and returns polymorphic sites.
    '''
    snp_index = []
    aln_arr = np.array(aln)
    gaps = aln_arr == '-'
    match_arr = np.zeros(aln_arr.shape[1], dtype=np.int32)
    for b in ['A', 'T', 'C', 'G']:
        match_arr += np.prod((aln_arr == b) + gaps, axis=0)
    match_arr -= np.prod(gaps, axis=0)
    snp_arr = aln_arr[:, ~match_arr.astype(bool)]

    # Filter all gap columns
    nongap_cols = np.sum(snp_arr != '-', axis=0, dtype=bool)
    snp_nogaps_arr = snp_arr[:, nongap_cols]

    snp_records = []
    for i in range(len(aln)):
        #snp_records.append(SeqRecord(Seq(''.join(snp_nogaps_arr[i]), generic_dna), id=aln[i].id, description=aln[i].description))
        snp_records.append(SeqRecord(Seq(''.join(snp_nogaps_arr[i])), id=aln[i].id, description=aln[i].description))
    if return_x:
        x = np.arange(aln_arr.shape[1], dtype=int)
        x_snps = x[~match_arr.astype(bool)]
        return MultipleSeqAlignment(snp_records), x_snps[nongap_cols]
    else:
        return MultipleSeqAlignment(snp_records)

def get_amino_acid_substitutions(prot_aln, return_x=False):
    '''
    Takes alignment and returns amino acid substitutions.
    '''
    aln_arr = np.array(prot_aln)
    is_gap = (aln_arr == '-')
    ref_seq = aln_arr[0] # pick first sequence as reference
    is_same_as_ref = (aln_arr == ref_seq)
    is_substitution = ((~is_same_as_ref) * (~is_gap)) # different from reference but not gap

    # Get substitutions index
    x = np.arange(aln_arr.shape[1], dtype=int)
    substitution_idx = x[np.sum(is_substitution, axis=0) > 0]

    substitution_records = []
    for i in range(len(prot_aln)):
        substitution_records.append(SeqRecord(Seq(''.join(aln_arr[i, substitution_idx])), id=prot_aln[i].id, description=prot_aln[i].description))
    if return_x:
        return MultipleSeqAlignment(substitution_records), substitution_idx
    else:
        return MultipleSeqAlignment(substitution_records)


def get_nonsynonymous_snps(aln):
    codon_dict = CodonSitesDict()
    nucl_aln_arr = np.array(aln)
    prot_aln_arr = np.array(translate_nucl_alnmt(aln))
    num_aas = np.array([len(np.unique(column[column != '-'])) for column in prot_aln_arr.T])

    # Find consensus protein sequence
    consensus_prot = get_consensus_seq(prot_aln_arr)
    variable_aa_idx = np.arange(len(num_aas))[num_aas > 1]

    nonsynonymous_snps = []
    for j in variable_aa_idx:
        i = 3 * j

        # Find consensus codon conditioned on consensus AA
        consensus_aa = consensus_prot[j]
        site_codons = np.array([''.join(codon_lett) for codon_lett in nucl_aln_arr[:, i:i + 3]])
        consensus_aa_codons = site_codons[prot_aln_arr[:, j] == consensus_aa]
        unique_codons, codon_counts = utils.sorted_unique(consensus_aa_codons)
        consensus_codon = unique_codons[0]

        # Define nonsynonymous SNPs as variable nucleotide sites that are 1D in consensus codon
        is_nonsynonymous = (codon_dict.get_synonymous_substitutions_fraction(consensus_codon) == 0)
        is_snp = np.array([len(np.unique(nucl_aln_arr[:, i + k])) > 1 for k in range(3)])

        # Append results
        for k in range(3):
            if is_nonsynonymous[k] * is_snp[k]:
                nonsynonymous_snps.append(i + k)

    return nonsynonymous_snps


def mask_alignment(aln, masked_sites):
    '''
    Replaces nucleotides at `masked_sites` with consensus.

    Params
    -------

    aln : alignment
    masked_sites : Sites at which to apply mask ['1D', '4D', 'non-4D']
    '''

    consensus_seq_arr = get_consensus_seq(aln)
    masked_arr = np.array(aln)
    if '4D' in masked_sites:
        aln_syn, x_syn = get_synonymous_sites(aln, return_x=True)
        if masked_sites == '4D':
            # Mask 4-fold degenerate sites
            x_mask = x_syn
        elif masked_sites == 'non-4D':
            # Mask 1-, 2-, and 3-fold degenerate sites
            x_mask = np.arange(masked_arr.shape[1])
            x_mask = x_mask[~np.isin(x_mask, x_syn)]

    elif masked_sites == '1D':
        x_mask = get_nonsynonymous_snps(aln)

    for x in x_mask:
        masked_arr[:, x] = consensus_seq_arr[x]

    masked_aln = []
    for i, rec in enumerate(aln):
        rec_copy = align_utils.copy_SeqRecord(rec)
        rec_copy.seq = Seq(''.join(masked_arr[i, :]))
        masked_aln.append(rec_copy)

    return MultipleSeqAlignment(masked_aln)
            

def translate_nucl_alnmt(nucl_aln):
    '''
    Converts nucleotide alignment to protein alignment. Assumes gaps are
    aligned to maintain reading frame.
    '''
    prot_records = []
    for rec in nucl_aln:
        prot_seq = []
        for i in range(0, len(rec.seq), 3):
            if '-' in rec.seq[i:i+3]:
                prot_seq.append('-')
            else:
                prot_seq.append(str(rec.seq[i:i+3].translate()))
        prot_records.append(SeqRecord(Seq(''.join(prot_seq)), id=rec.id, description=rec.description))
    return MultipleSeqAlignment(prot_records)

def add_consensus_to_alignment(aln, seq_type='codons', consensus_id='consensus'):
    consensus_seq = Seq(''.join(get_consensus_seq(aln, seq_type=seq_type)))
    consensus_rec = SeqRecord(consensus_seq, id=consensus_id, description='Alignment consensus sequence')
    aln_recs_out = [consensus_rec] + [rec for rec in aln]
    return MultipleSeqAlignment(aln_recs_out)

def get_consensus_seq(aln, seq_type='protein', keep_gaps=True):
    aln_arr = np.array(aln)
    consensus_seq = []
    if seq_type == 'codons':
        for i in range(0, aln_arr.shape[1] - 1, 3):
            codons = [''.join(codon_lett) if '-' not in codon_lett else '---' for codon_lett in aln_arr[:, i:i + 3]]
            unique_codons, codon_counts = utils.sorted_unique(codons)
            if keep_gaps:
                consensus_seq += list(unique_codons[0])
            elif unique_codons[0] != '---':
                consensus_seq += list(unique_codons[0])
    else:
        for column in aln_arr.T:
            unique_aas, aa_counts = utils.sorted_unique(column[column != '-'])
            if len(unique_aas) > 0:
                consensus_seq.append(unique_aas[0])
            elif keep_gaps:
                consensus_seq.append('-')
    seq_arr = np.array(consensus_seq)
    return seq_arr

def parse_blast_alignments(f_aln):
    with open(f_aln, 'r') as handle:
        #blast_records = NCBIXML.parse(handle)
        blast_qresults = SearchIO.parse(handle, 'blast-xml')

        #for blast_record in blast_records:
        for qresult in blast_qresults:
            if len(qresult) == 0:
                continue
            #for aln in blast_record.alignments:
            #    hsp = aln.hsps[0]
            #    print(hsp, hsp.gap_num)
            blast_hsp = qresult[0][0]
            print(blast_hsp)
            print(blast_hsp.gap_num)
            print(blast_hsp.aln[0].seq)
            print(blast_hsp.hit_range)
            print(blast_hsp.hit_start % 3)

        #return blast_records
        return blast_qresults

def generate_outlier_test_data(n=100, k=5, theta1=0.02, theta2=0.05, theta12=0.2):
    '''
    Generates pairwise distances from bimodal distribution with mode means at ``theta1``
    and ``theta2`` and mean distance between modes at ``theta12``. Total sample size is
    ``n`` and number of outliers is ``k``.
    '''

    pi_matrix = np.zeros((n, n))
    pi_matrix[:n-k, :n-k] = np.random.exponential(scale=theta1, size=(n - k, n - k))
    pi_matrix[:n-k, :n-k] = (pi_matrix[:n - k, :n - k] + pi_matrix[:n - k, :n - k].T ) / 2

    pi_matrix[n - k:, n - k:] = np.random.exponential(scale=theta2, size=(k, k))
    pi_matrix[n-k:, n-k:] = (pi_matrix[n - k:, n - k:] + pi_matrix[n - k:, n - k:].T ) / 2
    pi_matrix[n - k:, :n - k] = np.random.exponential(scale=theta12, size=(k, n - k))
    pi_matrix[:n - k, n - k:] = pi_matrix[n - k:, :n - k].T
    pi_matrix[np.arange(n), np.arange(n)] = 0
    return pi_matrix

def test_all():
    #gene_copy_numbers = pd.read_csv('../results/tests/SAG_gene_copy_number_table.csv', sep=',', index_col=0)
    #make_test_data()
    #highq_geneID_map = pickle.load(open('../results/tests/highq_sag_locus_geneID_map.dat', 'rb'))
    #export_allele_freqs(highq_sag_tables, highq_geneID_map, output_file=f'../results/tests/highq_allele_freq_table.dat')
    #random_seed = 12345
    #allele_table = AlleleFreqTable(random_seed)
    #allele_table.make_tables(highq_sag_tables, highq_geneID_map)
    #syn_homolog_map = SynHomologMap(build_maps=True)
    #export_locus_sequences(allele_table, '../results/tests/locus_seqs/', syn_homolog_map)


    #alleles_dict = pickle.load(open('../results/tests/sag_filtered_allele_dict.dat', 'rb'))
    '''
    test_locus = 'CYB_0022'

    alleles_dict = pickle.load(open('../results/single-cell/sscs/sag_os-merged_alnmts_allele_dict.dat', 'rb'))
    num_loci = 100
    np.random.seed(12345)
    #loci = np.unique(np.concatenate([np.random.choice(list(alleles_dict['A'].keys()), num_loci // 2), np.random.choice(list(alleles_dict['Bp'].keys()), num_loci // 2)]))
    loci_dict = {'A':np.random.choice(list(alleles_dict['A'].keys()), num_loci // 2), 'Bp':np.random.choice(list(alleles_dict['Bp'].keys()), num_loci // 2)}

    pdist_dict = {}
    for species in alleles_dict:
        loci_list = []
        pN_list = []
        pS_list = []
        for locus in loci_dict[species]:
            if locus not in alleles_dict[species]:
                continue
            locus_dict = alleles_dict[species][locus]
            allele_seqs = locus_dict['allele_seqs']
            if len(allele_seqs) > 1:
                pN, pS = calculate_ng86_pdist(allele_seqs[0], allele_seqs[1])
                #print(allele_seqs[0])
                #print(allele_seqs[1])
                #print(pN, pS, pN / (pS + (pS == 0)))
                loci_list.append(locus)
                pN_list.append(pN)
                pS_list.append(pS)
        pdist_dict[species] = [np.array(loci_list), np.array(pN_list), np.array(pS_list)]
    print(pdist_dict)

    cmap = plt.get_cmap('tab10')
    fig = plt.figure(figsize=(9, 5))
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('pS')
    ax1.set_ylabel('pN')

    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('pS')
    #ax2.set_xlim(0, 0.1)
    #ax2.set_ylabel('pN')
    #ax2.set_ylim(0, 0.1)
    ax2.set_xlim(1E-2, 1.5)
    ax2.set_ylim(1E-2, 15)
    ax2.set_xscale('log')
    ax2.set_ylabel('pN/pS')
    ax2.set_yscale('log')

    ax1.scatter(pdist_dict['A'][2], pdist_dict['A'][1], marker='o', s=20, fc='none', ec=cmap(0), label="A, joint alignment")
    ax1.scatter(pdist_dict['Bp'][2], pdist_dict['Bp'][1], marker='o', s=20, fc='none', ec=cmap(1), label="B', joint alignment")
    ax2.scatter(pdist_dict['A'][2], pdist_dict['A'][1] / (pdist_dict['A'][2] + np.isinf(pdist_dict['A'][2])), marker='o', s=20, fc='none', ec=cmap(0), label="A, separate alignment")
    ax2.scatter(pdist_dict['Bp'][2], pdist_dict['Bp'][1] / (pdist_dict['Bp'][2] + np.isinf(pdist_dict['Bp'][2])), marker='o', s=20, fc='none', ec=cmap(1), label="B', separate alignment")
    #ax2.scatter(pdist_dict['A'][2], pdist_dict['A'][1], marker='o', s=20, fc='none', ec=cmap(0), label="A, joint alignment")
    #ax2.scatter(pdist_dict['Bp'][2], pdist_dict['Bp'][1], marker='o', s=20, fc='none', ec=cmap(1), label="B', joint alignment")

    alleles_dict = pickle.load(open('../results/single-cell/sscs/sag_osa_alnmts_allele_dict.dat', 'rb'))
    species = 'A'
    syna_loci = np.random.choice(list(alleles_dict[species]), 50)
    loci_list = []
    pN_list = []
    pS_list = []
    #for locus in loci_dict[species]:
    for locus in syna_loci:
        if locus not in alleles_dict[species]:
            continue
        locus_dict = alleles_dict[species][locus]
        allele_seqs = locus_dict['allele_seqs']
        if len(allele_seqs) > 1:
            pN, pS = calculate_ng86_pdist(allele_seqs[0], allele_seqs[1])
            #print(allele_seqs[0])
            #print(allele_seqs[1])
            #print(pN, pS, pN / (pS + (pS == 0)))
            loci_list.append(locus)
            pN_list.append(pN)
            pS_list.append(pS)
    pdist_syna = [np.array(loci_list), np.array(pN_list), np.array(pS_list)]
    print(pdist_syna)
    print(np.nanmean(pdist_dict['A'][2]), np.nanmean(pdist_dict['A'][1]))
    print(np.nanmean(pdist_syna[2]), np.nanmean(pdist_syna[1]))

    ax1.scatter(pdist_syna[2], pdist_syna[1], marker='x', s=20, fc='none', ec=cmap(0), label="A, separate alignment")
    ax2.scatter(pdist_syna[2], pdist_syna[1] / (pdist_syna[2] + np.isinf(pdist_syna[2])), marker='x', s=20, fc='none', ec=cmap(0), label="A, separate alignment")
    #ax2.scatter(pdist_syna[2], pdist_syna[1], marker='x', s=20, fc='none', ec=cmap(0), label="A, separate alignment")

    alleles_dict = pickle.load(open('../results/single-cell/sscs/sag_osbp_alnmts_allele_dict.dat', 'rb'))
    species = 'Bp'
    loci_list = []
    pN_list = []
    pS_list = []
    for locus in loci_dict[species]:
        if locus not in alleles_dict[species]:
            continue
        locus_dict = alleles_dict[species][locus]
        allele_seqs = locus_dict['allele_seqs']
        if len(allele_seqs) > 1:
            pN, pS = calculate_ng86_pdist(allele_seqs[0], allele_seqs[1])
            #print(allele_seqs[0])
            #print(allele_seqs[1])
            #print(pN, pS, pN / (pS + (pS == 0)))
            loci_list.append(locus)
            pN_list.append(pN)
            pS_list.append(pS)
    pdist_synbp = [np.array(loci_list), np.array(pN_list), np.array(pS_list)]
    print(pdist_synbp)
    print(np.nanmean(pdist_dict['Bp'][2]), np.nanmean(pdist_dict['Bp'][1]))
    print(np.nanmean(pdist_synbp[2]), np.nanmean(pdist_synbp[1]))
    ax1.scatter(pdist_synbp[2], pdist_synbp[1], marker='x', s=20, fc='none', ec=cmap(1), label="B', separate alignment")
    ax2.scatter(pdist_synbp[2], pdist_synbp[1] / (pdist_synbp[2] + np.isinf(pdist_synbp[2])), marker='x', s=20, fc='none', ec=cmap(1), label="B', separate alignment")
    #ax2.scatter(pdist_synbp[2], pdist_synbp[1], marker='x', s=20, fc='none', ec=cmap(1), label="B', separate alignment")
    ax1.legend()

    plt.tight_layout()
    plt.savefig('../figures/analysis/tests/pairwise_divergences/pN_pS_MSA_test.pdf')

    muscle_aln = AlignIO.read('../results/single-cell/sscs/osbp_alignments/CYB_2597_aln_MUSCLE.fasta', 'fasta')


    mafft_aln = AlignIO.read('../results/single-cell/sscs/osbp_alignments/CYB_2597_aln_MAFFT.fasta', 'fasta')
    print(mafft_aln)

    pdist_dict = {}
    pN_list = []
    pS_list = []
    for i in range(len(muscle_aln)):
        for j in range(i):
            seq1 = muscle_aln[i]
            seq2 = muscle_aln[j]

            pN, pS = calculate_ng86_pdist(seq1, seq2)
            pN_list.append(pN)
            pS_list.append(pS)
            omega = pN / (pS + (pS == 0))
            if omega > 0.3:
                print(seq1, seq2, pS, pN, omega)
    pdist_dict['muscle'] = [np.array(pS_list), np.array(pN_list)]

    pN_list = []
    pS_list = []
    for i in range(len(mafft_aln)):
        for j in range(i):
            seq1 = muscle_aln[i].upper()
            seq2 = muscle_aln[j].upper()

            pN, pS = calculate_ng86_pdist(seq1, seq2)
            pN_list.append(pN)
            pS_list.append(pS)

            omega = pN / (pS + (pS == 0))
            if omega > 0.3:
                print(seq1, seq2, pS, pN, omega)

    pdist_dict['mafft'] = [np.array(pS_list), np.array(pN_list)]

    cmap = plt.get_cmap('tab10')
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('pS')
    ax.set_ylabel('pN')
    ax.scatter(pdist_dict['muscle'][0], pdist_dict['muscle'][1], marker='o', ec=cmap(2), fc='none', lw=0.5, label='MUSCLE v3.8.31')
    ax.scatter(pdist_dict['mafft'][0], pdist_dict['mafft'][1], marker='x', ec=cmap(3), fc='none', lw=0.5, label='MAFFT v7.453')
    ax.legend()
    plt.tight_layout()
    plt.savefig('../figures/analysis/tests/pairwise_divergences/CYB_2597_muscle_v_mafft.pdf')


    # MUSCLE + custom pN/pS vs NG86 from codeml

    filtered_muscle_aln = []
    for record in muscle_aln:
        f_gaps = sum(np.array(record.seq) == '-') / len(record.seq)
        if f_gaps < 0.25:
            filtered_muscle_aln.append(record)
    filtered_muscle_aln = MultipleSeqAlignment(filtered_muscle_aln)
    '''


    '''
    filtered_muscle_aln = AlignIO.read('../results/tests/CYB_2597_aln_MUSCLE_filtered.phy', 'phylip-sequential')
    print(filtered_muscle_aln)
    pN_list = []
    pS_list = []
    for i in range(len(filtered_muscle_aln)):
        for j in range(i):
            seq1 = filtered_muscle_aln[i]
            seq2 = filtered_muscle_aln[j]

            pN, pS = calculate_ng86_pdist(seq1, seq2)
            pN_list.append(pN)
            pS_list.append(pS)
            omega = pN / (pS + (pS == 0))
            if omega > 0.5:
                print(seq1, seq2, pS, pN, omega)
                print('\n')
    muscle_omega = [np.array(pS_list), np.array(pN_list)]

    with open('../results/tests/CYB_2597_codeml_dNdS_matrix.out', 'r') as f_in:
        omega_list = []
        skip_first = True
        for line in f_in.readlines():
            if skip_first:
                skip_first = False
                continue
            values_str = line[21:]
            values_str = values_str.replace('(', '').replace(')-', ') -').replace(')\n', '')
            values_list = values_str.split(') ')
            for triplet in values_list:
                triplet_list = triplet.strip().split(' ')
                omega_list.append([float(x) for x in triplet_list])
    codeml_omega = np.array(omega_list)
    print(codeml_omega)

    pickle.dump({'custom_pipeline':muscle_omega, 'codeml':codeml_omega}, open('../results/tests/codeml_dNdS_comparison_data.dat', 'wb'))


    #syn_aln = AlignIO.read('../results/single-cell/sscs/os-merged_alignments/CYB_0652_aln_MUSCLE.fasta', 'fasta')
    syn_aln = AlignIO.read('../results/single-cell/sscs/os-merged_alignments/CYB_1493_aln_MUSCLE.fasta', 'fasta')
    sag_ids = [record.id for record in syn_aln]

    metadata_map = MetadataMap()
    species_sorted_sags = metadata_map.sort_sags(sag_ids, by='species')

    synbp_aln_merged = MultipleSeqAlignment([record for record in syn_aln if record.id in species_sorted_sags['Bp']])
    pN_merged_df, pS_merged_df = calculate_pairwise_pNpS(synbp_aln_merged)
    print(f'Merged alignment...\npN\n{pN_merged_df}\npS\n{pS_merged_df}')

    synbp_aln_isolated = AlignIO.read('../results/single-cell/sscs/osbp_alignments/CYB_1493_aln_MUSCLE.fasta', 'fasta')
    pN_isolated_df, pS_isolated_df = calculate_pairwise_pNpS(synbp_aln_isolated)
    print(f'Bp alignment...\npN\n{pN_isolated_df}\npS\n{pS_isolated_df}')

    pickle.dump({'merged_alignment':(pN_merged_df, pS_merged_df), 'Bp_alignment':(pN_isolated_df, pS_isolated_df)}, open('../results/tests/CYB_1493_pNpS_merged_v_synbp_alignments.dat', 'wb'))
    '''

    synbp_merged_aln_comparison = pickle.load(open('../results/tests/CYB_1493_pNpS_merged_v_synbp_alignments_mafft.dat', 'rb'))
    pN_merged, pS_merged = synbp_merged_aln_comparison['merged_alignment']
    pN_synbp, pS_synbp = synbp_merged_aln_comparison['Bp_alignment']
    print(pN_merged)

    syn_aln = AlignIO.read('../results/single-cell/sscs/os-merged_alignments/CYB_1493_aln_MUSCLE.fasta', 'fasta')
    syn_aln_filtered = filter_alignment_gaps(syn_aln, gap_threshold=0.25)
    filtered_sag_ids = [record.id for record in syn_aln_filtered]
    print(syn_aln)
    print(filtered_sag_ids, len(filtered_sag_ids))
    filtered_index = [sag_id for sag_id in pN_merged.index if sag_id in filtered_sag_ids]

    pN_merged_filtered = pN_merged.reindex(index=filtered_index, columns=filtered_index)
    pS_merged_filtered = pS_merged.reindex(index=filtered_index, columns=filtered_index)
    print(pN_merged_filtered)
    print(pS_merged_filtered)

    pN_synbp_filtered = pN_synbp.reindex(index=filtered_index, columns=filtered_index)
    pS_synbp_filtered = pS_synbp.reindex(index=filtered_index, columns=filtered_index)
    print(pN_synbp_filtered["OS-B'"])
    print(pS_synbp_filtered["OS-B'"])
    print('\n\n')

    pN_merged_values = utils.get_matrix_triangle_values(pN_merged_filtered.values, triangle='upper', k=1)
    pS_merged_values = utils.get_matrix_triangle_values(pS_merged_filtered.values, triangle='upper', k=1)
    pN_synbp_values = utils.get_matrix_triangle_values(pN_synbp_filtered.values, triangle='upper', k=1)
    pS_synbp_values = utils.get_matrix_triangle_values(pS_synbp_filtered.values, triangle='upper', k=1)

    cmap = plt.get_cmap('tab10')
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('pS')
    ax.set_ylabel('pN')
    ax.scatter(pS_merged_values, pN_merged_values, marker='o', ec=cmap(2), fc='none', lw=0.5, label='merged alignment')
    ax.scatter(pS_synbp_values, pN_synbp_values, marker='+', ec=cmap(4), fc='none', lw=0.5, label="B' alignment")
    ax.legend()
    plt.tight_layout()
    plt.savefig('../figures/analysis/tests/pairwise_divergences/CYB_1493_merged_vs_synbp_alignment.pdf')

    pN_mafft_merged_filtered, pS_mafft_merged_filtered = synbp_merged_aln_comparison['mafft_merged_filtered']
    pN_mafft_merged_values = utils.get_matrix_triangle_values(pN_mafft_merged_filtered.values, triangle='upper', k=1)
    pS_mafft_merged_values = utils.get_matrix_triangle_values(pS_mafft_merged_filtered.values, triangle='upper', k=1)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('pS')
    ax.set_ylabel('pN')
    ax.scatter(pS_merged_values, pN_merged_values, marker='o', ec=cmap(2), fc='none', lw=0.5, label='merged MUSCLE')
    ax.scatter(pS_mafft_merged_values, pN_mafft_merged_values, marker='x', ec=cmap(3), fc='none', lw=0.5, label="merged MAFFT")
    ax.scatter(pS_synbp_values, pN_synbp_values, marker='+', ec=cmap(4), fc='none', lw=0.5, label="B' MUSCLE alignment")
    ax.legend()
    plt.tight_layout()
    plt.savefig('../figures/analysis/tests/pairwise_divergences/CYB_1493_merged_muscle_vs_mafft_alignment.pdf')


    # Codon-aware alignment test

    #data_tables = pickle.load(open('../results/tests/contig_genes_summary.sscs.dat', 'rb'))
    #genes_table = data_tables[list(data_tables.keys())[0]]['genes']
    #print(list(genes_table.columns))
    #print(genes_table[['partial', 'start_type', 'shortened']])

    '''
    seq_dir = '../results/single-cell/sscs/os-merged_seqs/'
    aln_dir = '../results/single-cell/sscs/os-merged_alignments/'
    f_nucl = 'CYB_1493_seqs.fasta'
    seq_records = SeqIO.parse(f'{seq_dir}{f_nucl}', 'fasta')

    nucl_seqs = {}
    for rec in seq_records:
        nucl_seqs[rec.id] = rec
    records = [rec.translate(table='Bacterial', id=rec.id, description=rec.description) for rec in nucl_seqs.values()]
    f_aa = f_nucl.replace('seqs.fasta', 'aa.fasta')
    SeqIO.write(records, f'{seq_dir}{f_aa}', 'fasta')

    f_aa_aln = f_aa.replace('aa.fasta', 'aa_aln_MUSCLE.fasta')
    aa_aln = AlignIO.read(f'{aln_dir}{f_aa_aln}', 'fasta')
    nucl_aln = back_translate(aa_aln, nucl_seqs)

    filtered_codon_aln = filter_alignment_gaps(nucl_aln, gap_threshold=0.25)
    pN_codon_df, pS_codon_df = calculate_pairwise_pNpS(filtered_codon_aln)
    filtered_sag_ids = [record.id for record in filtered_codon_aln]
    filtered_index = [sag_id for sag_id in pN_codon_df.index if sag_id in filtered_sag_ids]

    pN_codon_filtered = pN_codon_df.reindex(index=filtered_index, columns=filtered_index)
    pS_codon_filtered = pS_codon_df.reindex(index=filtered_index, columns=filtered_index)
    pN_codon_values = utils.get_matrix_triangle_values(pN_codon_filtered.values, triangle='upper', k=1)
    pS_codon_values = utils.get_matrix_triangle_values(pS_codon_filtered.values, triangle='upper', k=1)

    f_aa_aln = f_aa.replace('aa.fasta', 'aa_aln_MAFFT.fasta')
    aa_aln = AlignIO.read(f'{aln_dir}{f_aa_aln}', 'fasta')
    nucl_aln = back_translate(aa_aln, nucl_seqs)

    filtered_codon_aln = filter_alignment_gaps(nucl_aln, gap_threshold=0.25)
    pN_codon_mafft_df, pS_codon_mafft_df = calculate_pairwise_pNpS(filtered_codon_aln)
    filtered_sag_ids = [record.id for record in filtered_codon_aln]
    filtered_index = [sag_id for sag_id in pN_codon_mafft_df.index if sag_id in filtered_sag_ids]

    pN_codon_mafft_filtered = pN_codon_mafft_df.reindex(index=filtered_index, columns=filtered_index)
    pS_codon_mafft_filtered = pS_codon_mafft_df.reindex(index=filtered_index, columns=filtered_index)
    pN_codon_mafft_values = utils.get_matrix_triangle_values(pN_codon_mafft_filtered.values, triangle='upper', k=1)
    pS_codon_mafft_values = utils.get_matrix_triangle_values(pS_codon_mafft_filtered.values, triangle='upper', k=1)
    '''

    nucl_aln = AlignIO.read('../results/single-cell/sscs/os-merged_alignments/CYB_1493_codon_aln_MUSCLE.fasta', 'fasta')
    filtered_codon_aln = filter_alignment_gaps(nucl_aln, gap_threshold=0.25)
    pN_codon_df, pS_codon_df = calculate_pairwise_pNpS(filtered_codon_aln)
    filtered_sag_ids = [record.id for record in filtered_codon_aln]
    filtered_index = [sag_id for sag_id in pN_codon_df.index if sag_id in filtered_sag_ids]

    pN_codon_filtered = pN_codon_df.reindex(index=filtered_index, columns=filtered_index)
    pS_codon_filtered = pS_codon_df.reindex(index=filtered_index, columns=filtered_index)
    pN_codon_values = utils.get_matrix_triangle_values(pN_codon_filtered.values, triangle='upper', k=1)
    pS_codon_values = utils.get_matrix_triangle_values(pS_codon_filtered.values, triangle='upper', k=1)

    #pN_codon_filtered, pS_codon_filtered = synbp_merged_aln_comparison['merged_filtered_codon']
    #pN_codon_values = utils.get_matrix_triangle_values(pN_codon_filtered.values, triangle='upper', k=1)
    #pS_codon_values = utils.get_matrix_triangle_values(pS_codon_filtered.values, triangle='upper', k=1)

    pN_codon_mafft_filtered, pS_codon_mafft_filtered = synbp_merged_aln_comparison['merged_filtered_codon_mafft']
    pN_codon_mafft_values = utils.get_matrix_triangle_values(pN_codon_mafft_filtered.values, triangle='upper', k=1)
    pS_codon_mafft_values = utils.get_matrix_triangle_values(pS_codon_mafft_filtered.values, triangle='upper', k=1)

    #synbp_merged_aln_comparison['merged_filtered_codon'] = (pN_codon_filtered, pS_codon_filtered)
    #synbp_merged_aln_comparison['merged_filtered_codon_mafft'] = (pN_codon_mafft_filtered, pS_codon_mafft_filtered)
    #pickle.dump(synbp_merged_aln_comparison, open('../results/tests/CYB_1493_pNpS_merged_v_synbp_alignments_mafft.dat', 'wb'))

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('pS')
    ax.set_ylabel('pN')
    ax.scatter(pS_merged_values, pN_merged_values, marker='o', ec=cmap(2), fc='none', lw=0.5, label='merged MUSCLE')
    ax.scatter(pS_codon_mafft_values, pN_codon_mafft_values, marker='x', ec=cmap(3), fc='none', lw=0.5, label="merged codon MAFFT")
    ax.scatter(pS_codon_values, pN_codon_values, marker='s', ec=cmap(5), fc='none', lw=0.5, label="merged codon MUSCLE")
    ax.scatter(pS_synbp_values, pN_synbp_values, marker='+', ec=cmap(4), fc='none', lw=0.5, label="B' MUSCLE alignment")
    ax.legend()
    plt.tight_layout()
    plt.savefig('../figures/analysis/tests/pairwise_divergences/CYB_1493_merged_vs_codon_alignment.pdf')

    '''
    pN_merged_osbp = pN_merged_filtered["OS-B'"].values
    pS_merged_osbp = pS_merged_filtered["OS-B'"].values
    pN_synbp_osbp = pN_synbp_filtered["OS-B'"].values
    pS_synbp_osbp = pS_synbp_filtered["OS-B'"].values
    pN_mafft_merged_osbp = pN_mafft_merged_filtered["OS-B'"].values
    pS_mafft_merged_osbp = pS_mafft_merged_filtered["OS-B'"].values

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('pS')
    ax.set_ylabel('pN')
    ax.scatter(pS_merged_osbp, pN_merged_osbp, marker='o', ec=cmap(2), fc='none', lw=0.5, label='merged MUSCLE')
    ax.scatter(pS_mafft_merged_osbp, pN_mafft_merged_osbp, marker='x', ec=cmap(3), fc='none', lw=0.5, label="merged MAFFT")
    ax.scatter(pS_synbp_osbp, pN_synbp_osbp, marker='+', ec=cmap(4), fc='none', lw=0.5, label="B' MUSCLE alignment")
    ax.legend()
    plt.tight_layout()
    plt.savefig('../figures/analysis/tests/pairwise_divergences/CYB_1493_osbp_divergences_merged_muscle_vs_mafft_alignment.pdf')

    # Make test seqs for manual alignment
    #seq_records = SeqIO.parse('../results/single-cell/sscs/os-merged_seqs/CYB_1493_seqs.fasta', 'fasta')
    #filtered_seqs = [record for record in seq_records if record.id in filtered_sag_ids]
    #SeqIO.write(filtered_seqs, '../results/tests/CYB_1493_filtered_seqs.fasta', 'fasta')

    # Process blast alignments
    #blast_aln = parse_blast_alignments('../results/tests/CYB_1493_filtered_blastn.xml')
    #print('\n\n')
    #blast_aln = parse_blast_alignments('../results/tests/CYB_1493_filtered_blastn_qcov100.xml')
    #print('\n\n')
    #blast_aln = parse_blast_alignments('../results/tests/CYB_1493_filtered_tblastx.xml')

    '''


    #################################

    #syn_mafft_aln = read_mafft_alignment('../results/single-cell/sscs/os-merged_alignments/CYB_1493_aln_MAFFT.fasta')
    #syn_mafft_filtered_aln = MultipleSeqAlignment([record for record in syn_mafft_aln if record.id in list(pN_merged_filtered.index)])
    #print(syn_mafft_filtered_aln)

    #pN_mafft_merged_filtered, pS_mafft_merged_filtered = calculate_pairwise_pNpS(syn_mafft_filtered_aln)
    #synbp_merged_aln_comparison['mafft_merged_filtered'] = (pN_mafft_merged_filtered, pS_mafft_merged_filtered)
    #print(synbp_merged_aln_comparison)
    #pickle.dump(synbp_merged_aln_comparison, open('../results/tests/CYB_1493_pNpS_merged_v_synbp_alignments_mafft.dat', 'wb'))
    '''
    codeml_comparison = pickle.load(open('../results/tests/codeml_dNdS_comparison_data.dat', 'rb'))
    muscle_omega = codeml_comparison['custom_pipeline']
    codeml_omega = codeml_comparison['codeml']

    cmap = plt.get_cmap('tab10')
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('pS')
    ax.set_ylabel('pN')
    ax.scatter(muscle_omega[0], muscle_omega[1], marker='o', ec=cmap(2), fc='none', lw=0.5, label='MUSCLE v3.8.31')
    ax.scatter(codeml_omega[:, 2], codeml_omega[:, 1], marker='+', ec=cmap(4), fc='none', lw=0.5, label='codeml')
    ax.legend()
    plt.tight_layout()
    plt.savefig('../figures/analysis/tests/pairwise_divergences/CYB_2597_codeml_v_custom_script.pdf')

    alleles_dict = pickle.load(open('../results/single-cell/sscs/sag_osa_alnmts_allele_dict.dat', 'rb'))
    print(alleles_dict['A'].keys())
    locus_dict = alleles_dict['A']['CYA_2521']
    allele_seqs = locus_dict['allele_seqs']
    if len(allele_seqs) > 1:
        print(allele_seqs[0])
        print(allele_seqs[1])
        pN, pS = calculate_ng86_pdist(allele_seqs[0], allele_seqs[1])
        print(pN, pS, pN / (pS + (pS == 0)))


    alleles_dict = pickle.load(open('../results/single-cell/sscs/sag_osbp_alnmts_allele_dict.dat', 'rb'))
    print(alleles_dict['Bp'].keys())
    locus_dict = alleles_dict['Bp'][test_locus]
    allele_seqs = locus_dict['allele_seqs']
    if len(allele_seqs) > 1:
        print(allele_seqs[0])
        print(allele_seqs[1])
        pN, pS = calculate_ng86_pdist(allele_seqs[0], allele_seqs[1])
        print(pN, pS, pN / (pS + (pS == 0)))

    # Save to PHYLIP format
    for record in muscle_aln:
        record.id = utils.strip_sample_name(record.id, replace=True)
    AlignIO.write([muscle_aln], '../results/tests/CYB_2597_aln_MUSCLE.phy', 'phylip-sequential')

    AlignIO.write([filtered_muscle_aln], '../results/tests/CYB_2597_aln_MUSCLE_filtered.phy', 'phylip-sequential')
    print(muscle_aln)
    print(filtered_muscle_aln)


    n = 50
    k = 5
    theta1 = 0.03
    theta2 = 0.05
    theta12 = 0.2
    pi_matrix_test = generate_outlier_test_data(n=n, k=k, theta1=theta1, theta2=theta2, theta12=theta12)
    pi_bulk = utils.get_matrix_triangle_values(pi_matrix_test[:n - k, :n - k], k=1)
    pi_bulk_mean = np.mean(pi_bulk)
    pi_bulk_std = np.std(pi_bulk)
    sigma_bulk_i = theta1 / np.sqrt(n - k)
    pi_between = pi_matrix_test[:n - k, n - k:].flatten()

    pi_mean = np.mean(pi_matrix_test)
    print(pi_mean, np.median(pi_matrix_test), np.mean(utils.get_matrix_triangle_values(pi_matrix_test, k=1)), np.median(utils.get_matrix_triangle_values(pi_matrix_test, k=1)))
    t_test_results = spstats.ttest_1samp(pi_matrix_test, pi_mean, axis=1)
    print(t_test_results)


    fig = plt.figure(figsize=(8, 2.5))
    ax = fig.add_subplot(131)
    ax.set_xlabel('x')
    ax.set_yscale('log')
    ax.hist(pi_bulk, bins=50, color='gray')
    ax.hist(pi_between, bins=50, color='red', alpha=0.5)

    bins = 10
    ax = fig.add_subplot(132)
    ax.set_xlabel('x')
    ax.set_yscale('log')
    pi_bulk_mean_row = np.mean(pi_matrix_test[:n - k, :n - k], axis=1)
    ax.hist((pi_bulk_mean_row - pi_bulk_mean) / (pi_bulk_std / np.sqrt(n - k)), bins=bins, alpha=1.0, color='gray', density=True, label='bulk only')

    pi_mean = np.mean(pi_matrix_test)
    pi_median = np.median(pi_matrix_test)
    pi_mean_row = np.mean(pi_matrix_test, axis=1)
    pi_std_row = np.std(pi_matrix_test, axis=1) / np.sqrt(n)
    ax.hist((pi_mean_row - pi_mean) / pi_std_row, bins=bins, alpha=0.5, color='blue', density=True, label='all samples')

    x = np.linspace(-4, 4, 100)
    ax.plot(x, spstats.norm.pdf(x), c='k')
    ax.legend()


    pi_mean = np.mean(pi_matrix_test)
    print(pi_mean, np.median(pi_matrix_test), np.mean(utils.get_matrix_triangle_values(pi_matrix_test, k=1)), np.median(utils.get_matrix_triangle_values(pi_matrix_test, k=1)))
    ttest_stats, ttest_pvalues = spstats.ttest_1samp(pi_matrix_test, pi_mean, axis=1)

    alpha = 1E-3 / n
    keep_index = np.arange(n)[ttest_pvalues > alpha]
    filtered_matrix = pi_matrix_test[keep_index, :][:, keep_index]
    pi_mean = np.mean(filtered_matrix)
    print(pi_mean, np.median(filtered_matrix), np.mean(utils.get_matrix_triangle_values(filtered_matrix, k=1)), np.median(utils.get_matrix_triangle_values(filtered_matrix, k=1)))
    ttest_stats, ttest_pvalues = spstats.ttest_1samp(filtered_matrix, pi_mean, axis=1)
    print(ttest_pvalues, len(ttest_pvalues))

    ax = fig.add_subplot(133)
    ax.set_xlabel('p-value')
    ax.set_xscale('log')
    ax.set_yscale('log')
    bin_edges = np.geomspace(min(t_test_results[1]) / 2, 1, bins + 1)
    ax.hist(t_test_results[1], bins=bin_edges, color='gray', label='all samples')
    ax.hist(ttest_pvalues, bins=bins, color='red', alpha=0.5, label='filtered_samples')
    ax.legend()

    plt.show()

    '''

def calculate_loci_pairwise_divergences(f_loci, output_file, alnmts_dir='../results/single-cell/sscs/os-merged_alignments/', term='_codon_aln_MUSCLE.fasta'):
    candidate_loci = np.loadtxt(f_loci, dtype='U10')
    pairwise_divergences = {}
    for locus in candidate_loci:
        aln = AlignIO.read(f'{alnmts_dir}{locus}{term}', 'fasta')
        sag_ids = [rec.id for rec in aln]
        filtered_aln = filter_alignment_gaps(aln, gap_threshold=0.25)
        pN, pS = calculate_pairwise_pNpS(filtered_aln)
        pairwise_divergences[locus] = [pN, pS]
    pickle.dump(pairwise_divergences, open(output_file, 'wb'))

def calculate_paralog_pairwise_divergences(data_dir, output_file, alnmt_term='_aln.fasta'):
    seq_files = glob.glob(f'{data_dir}*_seqs.fasta')
    sag_ids = np.unique([fname.split('/')[-1].replace('_paralog_seqs.fasta', '')[9:] for fname in seq_files])

    results = {}
    for sag_id in sag_ids:
        aln_files = glob.glob(f'{data_dir}*{sag_id}*{alnmt_term}')
        sag_dict = {}
        for f_aln in aln_files:
            locus = '_'.join(f_aln.split('/')[-1].split('_')[:2])
            aln = AlignIO.read(f_aln, 'fasta')
            filtered_aln = filter_alignment_gaps(aln, gap_threshold=0.25)
            pN, pS = calculate_pairwise_pNpS(filtered_aln)
            sag_dict[locus] = (pN, pS)
        results[sag_id] = sag_dict

    with open(output_file, 'wb') as out_handle:
        pickle.dump(results, out_handle)

def make_clustered_divergence_matrix(aln, clusters):
    allele_aln = align_utils.construct_allele_alignment(aln)
    allele_pdist = align_utils.calculate_pairwise_distances(allele_aln, metric='SNP')
    allele_ids = np.array(allele_pdist.index)
    alleles, allele_counts = align_utils.group_alignment_alleles(aln)
    cluster_ids = np.unique(clusters)
    cluster_sizes = np.array([np.sum(allele_counts[clusters == k]) for k in cluster_ids])

    # Sort clusters by sizes
    sorted_allele_ids = []
    for k in cluster_ids[np.argsort(cluster_sizes)[::-1]]:
        cluster_allele_ids = allele_ids[clusters == k]
        cluster_allele_sizes = allele_counts[clusters == k]
        mc_dist = allele_pdist.loc[cluster_allele_ids[0], cluster_allele_ids].values
        dw_values, _ = utils.sorted_unique(mc_dist, sort='ascending', sort_by='tag')
        subcluster_sorted_ids = np.concatenate([cluster_allele_ids[mc_dist == d] for d in dw_values])
        sorted_allele_ids.append(subcluster_sorted_ids)
    sorted_allele_ids = np.concatenate(sorted_allele_ids)

    clustered_allele_pdist = allele_pdist.reindex(index=sorted_allele_ids, columns=sorted_allele_ids)
    clustered_seqs_pdist = convert_alleles_pdist_matrix_to_seqs(aln, clustered_allele_pdist)
    return clustered_seqs_pdist


def cluster_sweep_alleles(alleles, allele_counts, d_cutoff=3, linkage_method='single'):
    allele_aln = MultipleSeqAlignment(alleles)
    allele_pdist = align_utils.calculate_pairwise_distances(allele_aln, metric='SNP')
    pdist = distance.squareform(allele_pdist.values)
    linkage = hclust.linkage(pdist, method=linkage_method)
    clusters = hclust.fcluster(linkage, d_cutoff, criterion='distance')
    cluster_ids = np.unique(clusters)
    cluster_sizes = np.array([np.sum(allele_counts[clusters == k]) for k in cluster_ids])
    return clusters, cluster_sizes


'''
def cluster_sweep_segments(aln, d_cutoff=3, linkage_method='single'):
    allele_aln = align_utils.construct_allele_alignment(aln)
    allele_pdist = align_utils.calculate_pairwise_distances(allele_aln, metric='SNP')
    allele_ids = np.array(allele_pdist.index)
    alleles, allele_counts = align_utils.group_alignment_alleles(aln)

    # Find broad clusters
    pdist = distance.squareform(allele_pdist.values)
    linkage = hclust.linkage(pdist, method=linkage_method)
    clusters = hclust.fcluster(linkage, d_cutoff, criterion='distance')
    cluster_ids = np.unique(clusters)
    cluster_sizes = np.array([np.sum(allele_counts[clusters == k]) for k in cluster_ids])

    # Sort clusters by sizes
    sorted_allele_ids = []
    for k in cluster_ids[np.argsort(cluster_sizes)[::-1]]:
        cluster_allele_ids = allele_ids[clusters == k]
        cluster_allele_sizes = allele_counts[clusters == k]
        mc_dist = allele_pdist.loc[cluster_allele_ids[0], cluster_allele_ids].values
        dw_values, _ = utils.sorted_unique(mc_dist, sort='ascending', sort_by='tag')
        subcluster_sorted_ids = np.concatenate([cluster_allele_ids[mc_dist == d] for d in dw_values])
        sorted_allele_ids.append(subcluster_sorted_ids)
    sorted_allele_ids = np.concatenate(sorted_allele_ids)

    clustered_allele_pdist = allele_pdist.reindex(index=sorted_allele_ids, columns=sorted_allele_ids)
    clustered_seqs_pdist = convert_alleles_pdist_matrix_to_seqs(aln, clustered_allele_pdist)
    return clustered_seqs_pdist
'''

def convert_alleles_pdist_matrix_to_seqs(sample_reads_aln, alleles_pdist):
    # Make allele dictionary
    allele_ids = list(alleles_pdist.index)
    allele_grouped_seqs = align_utils.get_alignment_alleles(sample_reads_aln)
    allele_rec_dict = {}
    for allele_recs in allele_grouped_seqs:
        for rec in allele_recs:
            if rec.id in allele_ids:
                allele_rec_dict[rec.id] = [rec.id for rec in allele_recs]
                break
    #seq_ids = np.concatenate([allele_rec_dict[allele_id] for allele_id in allele_rec_dict])

    # Sort seq ids
    sorted_seq_ids = []
    for allele_id in alleles_pdist.index:
        sorted_seq_ids += allele_rec_dict[allele_id]

    # Fill in pdist matrix

    seq_pdist = pd.DataFrame(0, index=sorted_seq_ids, columns=sorted_seq_ids)
    for i, i_allele in enumerate(allele_ids):
        for j in range(i):
            j_allele = allele_ids[j]
            seq_pdist.loc[allele_rec_dict[i_allele], allele_rec_dict[j_allele]] = alleles_pdist.loc[i_allele, j_allele]
            seq_pdist.loc[allele_rec_dict[j_allele], allele_rec_dict[i_allele]] = alleles_pdist.loc[j_allele, i_allele]

    return seq_pdist


def test_locus_alignment_input():
    f_og_table = '../results/single-cell/sscs_pangenome/fine_scale_og_clusters/sscs_mapped_single_copy_orthogroup_presence.tsv'
    alignments_dir = '../results/single-cell/sscs_pangenome/_aln_results/'
    pangenome_map = PangenomeMap(f_orthogroup_table=f_og_table)
    aln_mapped = read_alignment_and_map_sag_ids('XXX', alignments_dir, pangenome_map)
    print(aln_mapped)
    aln_mapped = read_alignment_and_map_sag_ids('psaL-1', alignments_dir, pangenome_map)
    print(aln_mapped)

def test_alignment_reader():
    f_aln = '../results/tests/pangenome_construction/sscs/_aln_results/YSG_1376b_nucl_aln.fna'
    print(f_aln)
    aln = read_mafft_alignment(f_aln)
    print(aln)


def test_alana_alignments():
    f_aln = '../results/tests/other_tests/MIT9303_G1046_Subsample.fasta'
    aln = read_alignment(f_aln)
    print(aln)
    pN, pS = calculate_ng86_pdist(aln[0], aln[1])
    print(pN, pS)


def test_pNpS():
    f_test = '../results/tests/YSG_0947_alpha_block1_seqs_v1.fna'
    aln = read_alignment(f_test)
    print(aln)

    pN_segment, pS_segment = calculate_pairwise_pNpS(aln)
    print(pN_segment)
    print(pS_segment)

    f_test = '../results/tests/YSG_0947_alpha_block1_seqs_v2.fna'
    aln = read_alignment(f_test)
    pN_segment, pS_segment = calculate_pairwise_pNpS(aln)
    print(aln)
    print(pN_segment)
    print(pS_segment)


if __name__ == '__main__':
    #test_all()
    #calculate_loci_pairwise_divergences('../results/tests/interspecies_recombination_test_loci.txt',
    #        '../results/tests/interspecies_recombination_test_loci_divergences.dat',
    #        #alnmts_dir='/oak/stanford/projects/qmicdiv/cyanobacteria/results/single-cell/sscs/os-merged_alignments/')
    #        alnmts_dir='../results/single-cell/sscs/os-merged_alignments/')
    #data_dir = '../results/tests/gene_duplication/contig_filtered_paralog_seqs/'
    #calculate_paralog_pairwise_divergences(data_dir, '../results/tests/gene_duplication/sag_paralog_divergences.dat')

    #test_locus_alignment_input()
    #test_alignment_reader()

    #test_alana_alignments()
    test_pNpS()

