import numpy as np
import pandas as pd
import scipy.stats as stats
import utils
import pickle
import seq_processing_utils as seq_utils
import pangenome_utils as pg_utils
import scipy.cluster.hierarchy as hclust
import scipy.spatial.distance as distance
#from pangenome_utils import PangenomeMap
from metadata_map import MetadataMap
#from make_locus_geneID_maps import make_locus_geneID_map
from Bio.Seq import Seq
from Bio import SeqFeature
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio import SeqIO
from Bio import AlignIO
from Bio.Phylo.PAML import codeml
#from Bio.Alphabet import IUPAC


def calculate_ld_matrices_vectorized(alignment, reference=0, unbiased=True, convert_to_numeric=True):
    if convert_to_numeric == True:
        allele_matrix = utils.convert_alignment_to_numeric(alignment, reference)
    else:
        allele_matrix = alignment
    num_samples = ((allele_matrix[:, None, :] != 0) * (allele_matrix[:, :, None] != 0)).sum(axis=0)
    n00_matrix = ((allele_matrix[:, None, :] == 1) * (allele_matrix[:, :, None] == 1)).sum(axis=0)
    n01_matrix = ((allele_matrix[:, None, :] == 1) * (allele_matrix[:, :, None] == 2)).sum(axis=0)
    n10_matrix = ((allele_matrix[:, None, :] == 2) * (allele_matrix[:, :, None] == 1)).sum(axis=0)
    n11_matrix = ((allele_matrix[:, None, :] == 2) * (allele_matrix[:, :, None] == 2)).sum(axis=0)

    if unbiased == True:
        Dsq = n00_matrix * (n00_matrix - 1) * n11_matrix * (n11_matrix - 1)
        Dsq -= 2 * n00_matrix * n01_matrix * n10_matrix * n11_matrix
        Dsq += n01_matrix * (n01_matrix - 1) * n10_matrix * (n10_matrix - 1)
        Dsq = np.divide(Dsq * (num_samples >= 4),
                           num_samples * (num_samples - 1) * (num_samples - 2) * (num_samples - 3) + (num_samples < 4)) # hack to ensure we don't divide by zero when num_samples < 4
    else:
        Dsq = (n00_matrix / (num_samples + (num_samples == 0)) - (n00_matrix + n01_matrix) * (n00_matrix + n10_matrix) / (num_samples**2 + (num_samples == 0)))**2

    if unbiased == True:
        #1
        rsq_denominator = n10_matrix * (n10_matrix - 1) * n01_matrix * (n01_matrix - 1)
        #2
        rsq_denominator += n10_matrix * n01_matrix * (n01_matrix - 1) * n00_matrix
        #3
        rsq_denominator += n10_matrix * (n10_matrix - 1) * n01_matrix * n11_matrix
        #4
        rsq_denominator += n10_matrix * n01_matrix * n11_matrix * n00_matrix
        #5
        rsq_denominator += n10_matrix * (n10_matrix - 1) * n01_matrix * n00_matrix
        #6
        rsq_denominator += n10_matrix * n01_matrix * n00_matrix * (n00_matrix - 1)
        #7
        rsq_denominator += n10_matrix * (n10_matrix - 1) * n11_matrix * n00_matrix
        #8
        rsq_denominator += n10_matrix * n11_matrix * n00_matrix * (n00_matrix - 1)
        #9
        rsq_denominator += n10_matrix * n01_matrix * (n01_matrix - 1) * n11_matrix
        #10
        rsq_denominator += n01_matrix * (n01_matrix - 1) * n11_matrix * n00_matrix
        #11
        rsq_denominator += n10_matrix * n01_matrix * n11_matrix * (n11_matrix - 1)
        #12
        rsq_denominator += n01_matrix * n11_matrix * (n11_matrix - 1) * n00_matrix
        #13
        rsq_denominator += n10_matrix * n01_matrix * n11_matrix * n00_matrix
        #14
        rsq_denominator += n01_matrix * n11_matrix * n00_matrix * (n00_matrix - 1)
        #15
        rsq_denominator += n10_matrix * n11_matrix * (n11_matrix - 1) * n00_matrix
        #16
        rsq_denominator += n11_matrix * (n11_matrix - 1) * n00_matrix * (n00_matrix - 1)

        rsq_denominator = np.divide(rsq_denominator * (num_samples >= 4),
                            num_samples * (num_samples - 1) * (num_samples - 2) * (num_samples - 3) + (num_samples < 4))  # hack to ensure we don't divide by zero when num_samples < 4
    else:
        rsq_denominator = (n00_matrix + n01_matrix) * (n00_matrix + n10_matrix) * (n01_matrix + n11_matrix) * (n10_matrix + n11_matrix) / (num_samples**4 + (num_samples == 0))

    return Dsq, rsq_denominator

def calculate_Dprime_vectorized(alignment, reference=0, unbiased=True):
    '''
    Calculates Lewontin's D' linkage metric. Uses the index of `reference` haplotype
        in alignment to polarize mutations.
    '''

    allele_matrix = utils.convert_alignment_to_numeric(alignment, reference)
    num_samples = ((allele_matrix[:, None, :] != 0) * (allele_matrix[:, :, None] != 0)).sum(axis=0)
    n00_matrix = ((allele_matrix[:, None, :] == 1) * (allele_matrix[:, :, None] == 1)).sum(axis=0)
    n01_matrix = ((allele_matrix[:, None, :] == 1) * (allele_matrix[:, :, None] == 2)).sum(axis=0)
    n10_matrix = ((allele_matrix[:, None, :] == 2) * (allele_matrix[:, :, None] == 1)).sum(axis=0)
    n11_matrix = ((allele_matrix[:, None, :] == 2) * (allele_matrix[:, :, None] == 2)).sum(axis=0)

    f0i_matrix = (n00_matrix + n01_matrix) / (num_samples + (num_samples == 0))
    fi0_matrix = (n00_matrix + n10_matrix) / (num_samples + (num_samples == 0))

    D00 = n00_matrix / (num_samples + (num_samples == 0)) - f0i_matrix * fi0_matrix
    maxD00 = (np.fmin(f0i_matrix * fi0_matrix, (1 - f0i_matrix) * (1 - fi0_matrix))) * (D00 <= 0) + \
            (np.fmin(f0i_matrix * (1 - fi0_matrix), (1 - f0i_matrix) * fi0_matrix)) * (D00 > 0)

    return D00 / (maxD00 + (maxD00 == 0))


def calculate_ld_matrix(alignment, metric='r^2', method='snps_only', reference=0):
    alignment_length = len(alignment[0])
    snp_frequencies = calculate_snp_frequencies(alignment, reference=reference)
    if method == 'snps_only':
        ld_index = np.arange(len(snp_frequencies), dtype=int)[np.where(snp_frequencies > 0)[0]]
    else:
        ld_index = np.arange(len(snp_frequencies))
    if metric == 'r^2':
        ld = [[] for i in range(alignment_length)]
        for i, x_a in enumerate(ld_index):
            for x_b in ld_index[:i]:
                x_a = int(x_a)
                x_b = int(x_b)
                fab = calculate_snp_pair_frequencies(alignment, x_a, x_b, reference=reference)
                ld[x_a - x_b].append(calculate_ld_metric(snp_frequencies[x_a], snp_frequencies[x_b], fab, metric=metric))
    elif metric == 'sigma^2':
        sigma_sq_numerator = [[] for i in range(alignment_length)]
        sigma_sq_denominator = [[] for i in range(alignment_length)]
        for i, x_a in enumerate(ld_index):
            for x_b in ld_index[:i]:
                x_a = int(x_a)
                x_b = int(x_b)
                fab = calculate_snp_pair_frequencies(alignment, x_a, x_b, reference=reference)
                result_numerator, result_denominator = calculate_ld_metric(snp_frequencies[x_a], snp_frequencies[x_b], fab, metric=metric)
                sigma_sq_numerator[x_a - x_b].append(result_numerator)
                sigma_sq_denominator[x_a - x_b].append(result_denominator)
        ld = (sigma_sq_numerator, sigma_sq_denominator)
    else:
        ld = None
    return ld

def calculate_snp_frequencies(alignment, reference=None, filter_nans=True):
    alignment_length = len(alignment[0])
    freq = np.zeros(alignment_length)
    for site in range(alignment_length):
        locus_alleles = np.array(list(alignment[:, site]))
        alleles_no_gaps = locus_alleles[np.where(locus_alleles != '-')[0]]

        if reference is None:
            '''
            Calculate frequencies based on dominant allele.
            '''
            #alleles, counts = np.unique(alleles_no_gaps, return_counts=True)
            alleles, counts = utils.sorted_unique(alleles_no_gaps)
            if len(alleles) == 0:
                freq[site] = np.nan
            elif (len(alleles) <= 2) and (len(alleles) > 0):
                freq[site] = 1 - max(counts) / len(alleles_no_gaps)
            else:
                # Renormalize frequencies based on most common 2 alleles
                freq[site] = counts[1] / np.sum(counts[:2])
        else:
            ancestral_allele = alignment[reference, site]
            alleles, counts = np.unique(alleles_no_gaps, return_counts=True)
            if len(alleles) > 1:
                freq[site] = counts[np.where(alleles == ancestral_allele)[0]][0] / len(alleles_no_gaps)
    if filter_nans:
        freq = freq[~np.isnan(freq)]
    return freq


def calculate_ld_metric(fa, fb, fab, metric='r^2'):
    if (fa == 0) | (fa == 1) | (fb == 0) | (fb == 1):
        ld = 0
    else:
        if metric == 'r^2':
            ld = (fa*fb - fab)**2/(fa*(1 - fa)*fb*(1 - fb))
        elif metric == 'sigma^2':
            ld = ((fa*fb - fab)**2, (fa*(1 - fa)*fb*(1 - fb)))
    return ld


def calculate_snp_pair_frequencies(alignment, x_a, x_b, reference=None):
    alignment_length = len(alignment[0])
    alignment_array = np.array([list(sample_aln) for sample_aln in alignment])
    alleles = alignment_array[:, [x_a, x_b]]
    alleles_no_gaps = [sample_locus for sample_locus in alleles if (sample_locus[0] != '-' and sample_locus[1] != '-')]
    haplotypes = np.array([''.join(loci) for loci in alleles_no_gaps])

    if reference is None:
        '''
        Calculate frequencies based on dominant allele.
        '''
        alleles, counts = np.unique(haplotypes, return_counts=True)
        if len(alleles) > 1:
            f = max(counts) / len(alleles_no_gaps)
        else:
            f = 1
    else:
        ancestral_allele = f'{alignment[reference, x_a]}{alignment[reference, x_b]}'
        alleles, counts = np.unique(haplotypes, return_counts=True)
        if len(alleles) > 1:
            f = counts[np.where(alleles == ancestral_allele)[0]][0] / len(alleles_no_gaps)
        else:
            f = 1
    return f


def merge_ld_matrices(ld_matrix1, ld_matrix2):
    if len(ld_matrix1) <= len(ld_matrix2):
        ld_merged = [[] for i in range(len(ld_matrix2))]
        for i in range(len(ld_matrix1)):
            ld_merged[i] = ld_matrix1[i] + ld_matrix2[i]
        for i in range(len(ld_matrix1), len(ld_matrix2)):
            ld_merged[i] = ld_matrix2[i]
        return ld_merged
    else:
        merge_ld_matrices(ld_matrix2, ld_matrix1)

def calculate_gene_pair_snp_linkage(aln_list, metric="r^2", min_depth=0):
    snps_aln_list = [seq_utils.get_snps(aln) for aln in aln_list]
    boundary_idx = len(snps_aln_list[0][0])
    merged_snps_aln = seq_utils.merge_alignments(snps_aln_list)
    if len(merged_snps_aln) > min_depth:
        if metric == "D'":
            linkage_matrix = calculate_Dprime_vectorized(merged_snps_aln)
        else:
            Dsq, denom = calculate_ld_matrices_vectorized(merged_snps_aln, unbiased=False)
            linkage_matrix = Dsq / (denom + (denom == 0))
    else:
        linkage_matrix = None
    return linkage_matrix, boundary_idx

def calculate_pairwise_snp_distance(locus_alnmt):
    num_seqs = len(locus_alnmt)
    sag_ids = [seq.id for seq in locus_alnmt]
    pdist_df = pd.DataFrame(index=sag_ids, columns=sag_ids)
    for j in range(num_seqs):
        for i in range(j + 1):
            pdist_df.loc[sag_ids[i], sag_ids[j]] = calculate_pair_snps(locus_alnmt[i], locus_alnmt[j])
            pdist_df.loc[sag_ids[j], sag_ids[i]] = calculate_pair_snps(locus_alnmt[j], locus_alnmt[i])
    return pdist_df


def calculate_pairwise_distances(locus_alnmt, metric='divergence'):
    num_seqs = len(locus_alnmt)
    pdist_arr = np.zeros((num_seqs, num_seqs))
    for j in range(num_seqs):
        for i in range(j):
            if metric == 'SNP':
                dij = calculate_pair_snps(locus_alnmt[i], locus_alnmt[j])
            else:
                dij = calculate_pair_divergence(locus_alnmt[i], locus_alnmt[j])
            pdist_arr[i, j] = dij
            pdist_arr[j, i] = dij

    sag_ids = [seq.id for seq in locus_alnmt]
    pdist_df = pd.DataFrame(pdist_arr, index=sag_ids, columns=sag_ids)
    return pdist_df

def calculate_pair_divergence(seq1, seq2):
    assert len(seq1) == len(seq2)
    seq1_arr = np.array(seq1)
    seq2_arr = np.array(seq2)
    non_gaps = (seq1_arr != '-') * (seq2_arr != '-')
    matches = np.sum(seq1_arr[non_gaps] == seq2_arr[non_gaps])
    return 1 - matches / np.sum(non_gaps)

def calculate_pair_snps(seq1, seq2):
    assert len(seq1) == len(seq2)
    seq1_arr = np.array(seq1)
    seq2_arr = np.array(seq2)
    non_gaps = (seq1_arr != '-') * (seq2_arr != '-')
    return np.sum(seq1_arr[non_gaps] != seq2_arr[non_gaps])

def calculate_fast_pairwise_divergence(aln):
    seq_ids = [record.id for record in aln]
    #pdist_df = pd.DataFrame(0, index=seq_ids, columns=seq_ids)
    pdist_df = pd.DataFrame(index=seq_ids, columns=seq_ids)

    aln_arr = np.array(aln)
    nongap_arr = aln_arr[:, np.sum(aln_arr == '-', axis=0) == 0]

    if nongap_arr.shape[1] > 0:
        consensus_seq = seq_utils.get_consensus_seq(nongap_arr, seq_type='nucl')
        digitized_aln = (consensus_seq[None, :] == nongap_arr).astype(int)
        digitized_aln[np.where(digitized_aln == 0)] = -1
        for j in range(digitized_aln.shape[0]):
            for i in range(j):
                seq_i = seq_ids[i]
                seq_j = seq_ids[j]
                d = np.sum(digitized_aln[i] + digitized_aln[j] == 0) / digitized_aln.shape[1]
                pdist_df.loc[seq_i, seq_j] = d
                pdist_df.loc[seq_j, seq_i] = d

    np.fill_diagonal(pdist_df.values, 0)

    return pdist_df

def calculate_consensus_distance(aln, method='pdist'):
    aln_arr = np.array(aln)
    nongap_arr = aln_arr[:, np.sum(aln_arr == '-', axis=0) == 0]
    
    if nongap_arr.shape[1] > 0:
        consensus_seq = seq_utils.get_consensus_seq(nongap_arr, seq_type='nucl')
        pdist = np.sum(nongap_arr != consensus_seq[None, :], axis=1) / aln_arr.shape[1]
    else:
        pdist = np.nan * np.ones(aln_arr.shape[0])

    if method == 'pdist':
        return pdist
    elif method == 'jc69':
        return calculate_divergence(pdist)


def sample_alignment(aln, sample_size, num_samples, random_seed=None):
    np.random.seed(random_seed)
    samples = []
    seq_record_list = list(aln)
    for i in range(num_samples):
        random_idx = np.random.randint(0, len(aln), size=sample_size)
        samples.append(MultipleSeqAlignment([seq_record_list[k] for k in random_idx]))
    return samples

def randomize_alignment_snps(aln, random_seed=None):
    aln_arr = np.array(aln)
    shuffled_arr = []
    for j in range(aln_arr.shape[1]):
        shuffled_arr.append(np.random.permutation(aln_arr[:, j]))
    shuffled_arr = np.array(shuffled_arr).T

    shuffled_aln = []
    for i, record in enumerate(aln):
        shuffled_aln.append(SeqRecord(Seq(''.join(shuffled_arr[i])), id=record.id, name=record.name, description=record.description))
    return MultipleSeqAlignment(shuffled_aln)

def calculate_FST(pairwise_snps, clade_ids_list):
    n = pairwise_snps.shape[0]
    dij_values = utils.get_matrix_triangle_values(pairwise_snps.values, k=1)
    d_mean = np.mean(dij_values, dtype=np.float64)

    d_clade_sum = 0
    num_pairs = 0
    for clade_ids in clade_ids_list:
        n_clade = len(clade_ids)
        dij_clade = pairwise_snps.reindex(index=clade_ids, columns=clade_ids)
        dij_clade_values = utils.get_matrix_triangle_values(dij_clade.values, k=1)
        d_clade_sum += np.sum(dij_clade_values)
        num_pairs += n_clade * (n_clade - 1) / 2
    d_clade_mean = d_clade_sum / num_pairs

    return 1 - d_clade_mean / d_mean

def calculate_homozygosity(freqs, epsilon=1E-6):
    n = np.sum(freqs)
    if abs(n - 1) >= epsilon:
        relative_freqs = np.array(freqs) / n
    else:
        relative_freqs = np.array(freqs)
    return np.sum(relative_freqs**2)

def calculate_snp_heterozygosity(aln):
    f = calculate_snp_frequencies(aln)
    H = 2 * np.mean(f * (1 - f))
    return H


def remove_paralogs(aln):
    rec_ids = [rec.id for rec in aln]
    unique_ids, id_counts = np.unique(rec_ids, return_counts=True)

    if max(id_counts) == 1:
        return aln
    else:
        filtered_recs = []
        added_ids = []
        for rec in aln:
            if rec.id not in added_ids:
                filtered_recs.append(rec)
                added_ids.append(rec.id)
        return MultipleSeqAlignment(filtered_recs)


def get_subsample_alignment(aln, sample_ids):
    subsample_aln = []
    for rec in aln:
        if rec.id in sample_ids:
            subsample_aln.append(rec)
    return MultipleSeqAlignment(subsample_aln)

def get_alignment_sites(aln, x_sites):
    '''
    Returns alignment only at sites given by `x_sites`.
    '''
    out_aln = []
    for record in aln:
        if len(x_sites) > 1:
            filtered_seq = Seq(''.join(np.array(record.seq)[x_sites]))
        else:
            filtered_seq = record.seq[int(x_sites[0])]
        out_aln.append(SeqRecord(filtered_seq, id=record.id, name=record.name, description=record.description))
    return MultipleSeqAlignment(out_aln)

def stack_alignments(aln_list):
    '''
    Merges MultipleSeqAlignment objects from `aln_list` vertically, i.e., sequences are stacked on top of each other.
    '''

    merged_aln_recs = []
    for aln in aln_list:
        for rec in aln:
            merged_aln_recs.append(rec)
    return MultipleSeqAlignment(merged_aln_recs)

def mutate_sites(aln, x_mutations, k_mutations, rng=None):
    '''
    Randomly mutates sites in an alignment.

    Params
    ------------
        x_mutation : location of new SNPs
        k_mutation : absolute frequencies for each SNP
        rng : random number generator
    '''

    if rng is None:
        rng = np.random.default_rng()

    nucleotides = np.array(['A', 'T', 'C', 'G'])
    aln_arr = np.array(aln)
    for i, x in enumerate(x_mutations):
        # Pick new nucleotide and mutate sites
        dominant_allele = np.unique(aln_arr[:, x])[0]
        mutation = rng.choice(nucleotides[nucleotides != dominant_allele])
        mutants_idx = rng.choice(len(aln_arr), size=k_mutations[i], replace=False)
        aln_arr[mutants_idx, x] = mutation

    new_aln = []
    for i, seq_arr in enumerate(aln_arr):
        new_rec = SeqRecord(Seq(''.join(aln_arr[i])), id=aln[i].id, name=aln[i].name, description=aln[i].description)
        new_aln.append(new_rec)

    return MultipleSeqAlignment(new_aln)



def extract_mixed_species_snps(aln, og_gene_presence, species_sorted_sags, snp_accuracy_cutoff=1.0, min_species_seqs=4, return_accuracies=False):
    # Get species sorted gene IDs
    rec_ids = [rec.id for rec in aln]
    species_sorted_gene_ids = {}
    for species in ['A', 'Bp']:
        species_gene_ids = np.array(pg_utils.read_gene_ids(og_gene_presence[species_sorted_sags[species]], drop_none=True))
        species_sorted_gene_ids[species] = species_gene_ids[np.isin(species_gene_ids, rec_ids)]

    # Remove seqs from non-A or B' cells
    species_aln = get_subsample_alignment(aln, np.concatenate(list(species_sorted_gene_ids.values())))

    # Find mixed species SNPs
    if min([len(gene_ids) for gene_ids in species_sorted_gene_ids.values()]) >= min_species_seqs:
        x_mixed_species_snps, snp_accuracies = find_mixed_species_snps(species_aln, species_sorted_gene_ids, snp_accuracy_cutoff=snp_accuracy_cutoff)
        mixed_species_snps_aln = get_alignment_sites(species_aln, x_mixed_species_snps)
    else:
        mixed_species_snps_aln = MultipleSeqAlignment([])
        x_mixed_species_snps = []
        snp_accuracies = np.array([])

    if return_accuracies:
        return mixed_species_snps_aln, x_mixed_species_snps, snp_accuracies
    else:
        return mixed_species_snps_aln, x_mixed_species_snps


def find_mixed_species_snps(aln, species_sorted_gene_ids, snp_accuracy_cutoff=1.0):
    aln_snps, x_snps = seq_utils.get_snps(aln, return_x=True)

    # Get species gene IDs sorted by species abundance
    species_list = np.array(list(species_sorted_gene_ids.keys()))
    species_abundance = [len(species_sorted_gene_ids[species]) for species in species_list]
    species_gene_ids = [species_sorted_gene_ids[species_list[i]] for i in np.argsort(species_abundance)[::-1]]

    x_mixed_species = []
    snp_accuracies = []
    for i, xi in enumerate(x_snps):
        # Group gene IDs by site allele
        site_alleles = get_site_alleles(aln, xi)
        top_alleles = [allele for allele in site_alleles][:2]
        allele_gene_ids = [site_alleles[allele] for allele in top_alleles]

        # Sort site based on species classification accuracy
        contingency_table = utils.make_contingency_table(species_gene_ids, allele_gene_ids) # use top two alleles only
        snp_accuracy = (contingency_table[0, 0] + contingency_table[1, 1]) / np.sum(contingency_table)
        if snp_accuracy < snp_accuracy_cutoff:
            x_mixed_species.append(xi)
            snp_accuracies.append(snp_accuracy)

    return np.array(x_mixed_species), np.array(snp_accuracies)


def calculate_individual_snps_FST(aln, species_sorted_gene_ids, min_FST=None):
    aln_gene_ids = np.concatenate(species_sorted_gene_ids)
    species_aln = get_subsample_alignment(aln, aln_gene_ids)
    aln_snps, x_snps = seq_utils.get_snps(species_aln, return_x=True)

    snp_FSTs = []
    filtered_idx = []
    for i, xi in enumerate(x_snps):
        # Group gene IDs by site allele
        site_alleles = get_site_alleles(aln_snps, i)

        # Make pdist DataFrame
        pdist_df = pd.DataFrame(index=aln_gene_ids, columns=aln_gene_ids)
        alleles = [al for al in site_alleles.keys() if al != '-']
        for j, al1 in enumerate(alleles):
            pdist_df.loc[site_alleles[al1], site_alleles[al1]] = 0
            for k in range(j):
                al2 = alleles[k]
                pdist_df.loc[site_alleles[al1], site_alleles[al2]] = 1
                pdist_df.loc[site_alleles[al2], site_alleles[al1]] = 1

        FST = calculate_FST(pdist_df, species_sorted_gene_ids)
        if min_FST is not None and FST < min_FST:
            filtered_idx.append(i)
        else:
            snp_FSTs.append(FST)

    if len(filtered_idx) > 0:
        aln_idx = [i for i in range(len(x_snps)) if i not in filtered_idx]
        aln_snps = get_alignment_sites(aln_snps, aln_idx)
        x_snps = x_snps[aln_idx]

    return aln_snps, x_snps, np.array(snp_FSTs)

def get_site_alleles(aln, x):
    rec_ids = np.array([rec.id for rec in aln])
    site_nucleotides = np.array(aln)[:, x]
    alleles, allele_counts = utils.sorted_unique(site_nucleotides)

    # Group site alleles
    site_alleles = {}
    for al in alleles:
        site_alleles[al] = rec_ids[site_nucleotides == al]

    return site_alleles


def trim_alignment_and_remove_gaps(aln, max_edge_gaps=0.05, max_gap_column_freq=0.5):
    trimmed_aln, x_edges = trim_alignment_gaps(aln, start_gap_perc=max_edge_gaps, return_edges=True)
    if trimmed_aln.get_alignment_length() > 0:
        x_trimmed = np.arange(x_edges[0], x_edges[1] + 1, dtype=int)
        filtered_aln, x_gaps = filter_alignment_gap_columns(trimmed_aln, max_gap_frequency=max_gap_column_freq, return_x=True)
        x_filtered = x_trimmed[[i for i in range(len(x_trimmed)) if i not in x_gaps]] # remove gap coordinates
    else:
        filtered_aln = trimmed_aln
        x_filtered = None
    return filtered_aln, x_filtered

def trim_alignment_gaps(aln, start_gap_perc=0.0, return_edges=False):
    if len(aln) > 0 and aln.get_alignment_length() > 0:
        gap_perc = np.sum(np.array(aln) == '-', axis=0) / len(aln)
        i_start, i_end = find_alignment_endpoints(gap_perc, start_gap_perc)
        filtered_aln = []
        for record in aln:
            if i_end > 0:
                filtered_seq = Seq(''.join(np.array(record.seq)[i_start:i_end + 1]))
            else:
                filtered_seq = Seq(''.join(np.array(record.seq)[i_start:]))
            filtered_aln.append(SeqRecord(filtered_seq, id=record.id, name=record.name, description=record.description))
        if return_edges:
            return MultipleSeqAlignment(filtered_aln), (i_start, i_end)
        else:
            return MultipleSeqAlignment(filtered_aln)
    else:
        if return_edges:
            return aln, None
        else:
            return aln


def clean_alignment(aln, start_gap_perc=0.05, gap_threshold=0.25, filter_gap_columns=False):
    trimmed_aln = trim_alignment_gaps(aln, start_gap_perc=start_gap_perc)
    if filter_gap_columns:
        gap_columns_filtered_aln = filter_alignment_gap_columns(trimmed_aln)
        filtered_aln = seq_utils.filter_alignment_gaps(gap_columns_filtered_aln, gap_threshold)
    else:
        filtered_aln = seq_utils.filter_alignment_gaps(trimmed_aln, gap_threshold)
    return filtered_aln


def filter_alignment_gap_columns(aln, max_gap_frequency=0.1, return_x=False):
    gap_counts = np.sum(np.array(aln) == '-', axis=0)
    gap_freq = gap_counts / len(aln)
    filtered_aln = []
    for record in aln:
        filtered_seq = Seq(''.join(np.array(record.seq)[gap_freq < max_gap_frequency]))
        filtered_aln.append(SeqRecord(filtered_seq, id=record.id, name=record.name, description=record.description))
    if return_x == True:
        x = np.arange(len(gap_freq), dtype=int)
        x_gap_columns = x[gap_freq >= max_gap_frequency]
        return MultipleSeqAlignment(filtered_aln), x_gap_columns
    else:
        return MultipleSeqAlignment(filtered_aln)


def remove_gap_codons(aln, max_gap_frequency=0.1, return_x=False):
    # Get codons with gaps
    gap_codon_sites = []
    aln_arr = np.array(aln)
    for s in range(0, aln.get_alignment_length(), 3):
        num_nongapped_codons = np.sum(np.sum(aln_arr[:, s:s+3] != '-', axis=1) == 3)
        if (num_nongapped_codons / aln_arr.shape[0]) < 1.0 - max_gap_frequency:
            gap_codon_sites += [s, s + 1, s + 2]

    x_filtered = np.arange(aln.get_alignment_length())
    x_filtered = x_filtered[~np.isin(x_filtered, gap_codon_sites)]
    filtered_aln = []
    for i, rec in enumerate(aln):
        new_rec = copy_SeqRecord(rec)
        new_rec.seq = Seq(''.join(aln_arr[i, x_filtered]))
        filtered_aln.append(new_rec)

    if return_x:
        return MultipleSeqAlignment(filtered_aln), gap_codon_sites
    else:
        return MultipleSeqAlignment(filtered_aln)

def copy_SeqRecord(rec):
    rec_copy = SeqRecord(Seq(str(rec.seq)), id=rec.id, name=rec.name, description=rec.description)
    return rec_copy

def get_strict_biallelic_matrix(aln, polarization='major_allele'):
    aln_arr = np.array(aln)
    aln_arr = aln_arr[:, np.sum(aln_arr == '-', axis=0) == 0] # remove gap columns

    biallelic_matrix = [] 
    for j in range(aln_arr.shape[1]):
        site_alleles, allele_counts = utils.sorted_unique(aln_arr[:, j])
        if len(site_alleles) == 2:
            if polarization == 'minor_allele':
                is_minor_allele = (aln_arr[:, j] == site_alleles[1]).astype(float)
                biallelic_matrix.append(is_minor_allele)
            else:
                is_major_allele = (aln_arr[:, j] == site_alleles[0]).astype(float)
                biallelic_matrix.append(is_major_allele)

    return np.array(biallelic_matrix).T

def trim_alignment_and_remove_gap_codons(aln, max_edge_gaps=0.05, max_gap_column_freq=0.1, max_final_gap_percentage=0.05):
    '''
    Uses a two-step filtering approach to clean mixed orthogroup alignments.
    '''

    # Coarse filtering
    coarse_edge_gap_fraction = 0.15
    coarse_trimmed_aln, x_edges = trim_alignment_gaps(aln, start_gap_perc=coarse_edge_gap_fraction, return_edges=True)
    if coarse_trimmed_aln.get_alignment_length() > 0:
        x_trimmed = np.arange(x_edges[0], x_edges[1] + 1, dtype=int)
        filtered_aln, x_gaps = remove_gap_codons(coarse_trimmed_aln, max_gap_frequency=max_gap_column_freq, return_x=True)
        x_filtered = x_trimmed[[i for i in range(len(x_trimmed)) if i not in x_gaps]] # remove gap coordinates

        # Filter remaining high gap percentage sequences
        rec_ids = np.array([rec.id for rec in filtered_aln])
        gap_perc = calculate_alignment_gap_percentages(filtered_aln, axis=1)
        filtered_rec_ids = rec_ids[gap_perc <= max_final_gap_percentage]
        coarse_filtered_aln = sort_alnmt(filtered_aln, filtered_rec_ids)

        # Fine filtering
        trimmed_aln, edges_idx = trim_alignment_gaps(coarse_filtered_aln, start_gap_perc=0.0, return_edges=True)
        x_trimmed = x_filtered[edges_idx[0]:edges_idx[1] + 1]
    else:
        # Revert back to standard trimming
        trimmed_aln, x_trimmed = trim_alignment_and_remove_gaps(aln)

    return trimmed_aln, x_trimmed


def clean_mixed_orthogroup_alignments(aln, max_edge_gaps=0.05, max_gap_column_freq=0.1, max_final_gap_percentage=0.05):
    '''
    Uses a two-step filtering approach to clean mixed orthogroup alignments.
    '''

    # Coarse filtering
    coarse_edge_gap_fraction = 0.15
    coarse_trimmed_aln, x_edges = trim_alignment_gaps(aln, start_gap_perc=coarse_edge_gap_fraction, return_edges=True)
    if coarse_trimmed_aln.get_alignment_length() > 0:
        x_trimmed = np.arange(x_edges[0], x_edges[1] + 1, dtype=int)
        filtered_aln, x_gaps = filter_alignment_gap_columns(coarse_trimmed_aln, max_gap_frequency=max_gap_column_freq, return_x=True)
        x_filtered = x_trimmed[[i for i in range(len(x_trimmed)) if i not in x_gaps]] # remove gap coordinates

        # Filter remaining high gap percentage sequences
        rec_ids = np.array([rec.id for rec in filtered_aln])
        gap_perc = calculate_alignment_gap_percentages(filtered_aln, axis=1)
        filtered_rec_ids = rec_ids[gap_perc <= max_final_gap_percentage]
        coarse_filtered_aln = sort_alnmt(filtered_aln, filtered_rec_ids)

        # Fine filtering
        trimmed_aln, edges_idx = trim_alignment_gaps(coarse_filtered_aln, start_gap_perc=0.0, return_edges=True)
        x_trimmed = x_filtered[edges_idx[0]:edges_idx[1] + 1]
    else:
        # Revert back to standard trimming
        trimmed_aln, x_trimmed = trim_alignment_and_remove_gaps(aln)

    return trimmed_aln, x_trimmed


def get_high_frequency_snp_alignment(aln, freq_threshold, counts_filter=False):
    if counts_filter == True:
        freq_threshold /= len(aln)
    aln_snps, x_snps = remove_low_frequency_snps(aln, freq_threshold, return_x=True)
    return aln_snps, x_snps

def remove_low_frequency_snps(aln, f_threshold, return_x=False):
    snp_frequencies = calculate_snp_frequencies(aln, filter_nans=False)
    filtered_list = []
    for record in aln:
        filtered_seq = Seq(''.join(np.array(record.seq)[snp_frequencies > f_threshold]))
        filtered_list.append(SeqRecord(filtered_seq, id=record.id, name=record.name, description=record.description))
    filtered_aln = MultipleSeqAlignment(filtered_list)
    if return_x:
        x = np.arange(len(aln[0]))
        return filtered_aln, x[snp_frequencies > f_threshold]
    else:
        return filtered_aln


def read_and_process_alignment(f_aln, sites='all'):
    aln_raw = seq_utils.read_alignment(f_aln)
    trimmed_aln, _ = trim_alignment_and_remove_gap_codons(aln_raw)

    # Pick alignment columns
    if sites != 'all':
        site_degeneracies = seq_utils.get_alignment_site_degeneracies(trimmed_aln)
        x_sites = site_degeneracies[sites]
    else:
        x_sites = np.arange(species_cluster_aln.get_alignment_length())
    aln_sites = get_alignment_sites(trimmed_aln, x_sites)

    return aln_sites, x_sites


def read_species_og_alignment(og_id, f_aln, pangenome_map, species_sag_ids):
    aln = seq_utils.read_alignment(f_aln)
    filtered_gene_ids = pangenome_map.get_og_gene_ids(og_id, sag_ids=species_sag_ids)
    return get_subsample_alignment(aln, filtered_gene_ids)


def find_alignment_endpoints(gap_perc, endpoint_gap_threshold):
    i_start = 0
    i_end = len(gap_perc) - 1
    while gap_perc[i_start] > endpoint_gap_threshold or gap_perc[i_end] > endpoint_gap_threshold:
        if gap_perc[i_start] > endpoint_gap_threshold:
            i_start += 3
        if gap_perc[i_end] > endpoint_gap_threshold:
            i_end -= 3
        if i_start == len(gap_perc) or abs(i_end) > len(gap_perc):
            break
    return i_start, i_end

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


def filter_alignment(alignment, gap_threshold=0):
    '''
    Removes sequences from ``alignment`` that have fraction of gaps
    higher than ``gap_threshold``.
    '''
    filtered = []
    if len(alignment) > 0:
        len_aligned = len(alignment[0])
        for i, seq in enumerate(alignment):
            num_gaps = count_gaps(seq)
            if num_gaps / len_aligned <= gap_threshold:
                filtered.append(seq)
    return MultipleSeqAlignment(filtered)

def calculate_alignment_gap_percentages(aln, axis=0):
    if aln.get_alignment_length() > 0 and len(aln) > 0:
        if axis == 1:
            gap_percs = np.sum(np.array(aln) == '-', axis=1) / aln.get_alignment_length()
        else:
            gap_percs = np.sum(np.array(aln) == '-', axis=0) / len(aln)
    else:
        gap_percs = []
    return gap_percs


def remove_alignment_codons(in_aln, x_filter, return_x=False):
    x_start_codons = np.unique([x - (x % 3) for x in x_filter])
    x_remove = np.concatenate([x + np.arange(3) for x in x_start_codons])
    aln_arr = np.delete(np.array(in_aln), x_remove, axis=1)

    filtered_arr = []
    for i, record in enumerate(in_aln):
        filtered_seq = Seq(''.join(aln_arr[i]))
        filtered_arr.append(SeqRecord(filtered_seq, id=record.id, name=record.name, description=record.description))
    filtered_aln = MultipleSeqAlignment(filtered_arr)

    if return_x:
        x_filtered = np.array([x for x in range(len(in_aln[0])) if x not in x_remove])
        return filtered_aln, x_filtered
    else:
        return filtered_aln

def extract_codon_segment(in_aln, x_segment, include_edges=False):
    if include_edges == True:
        x_start = x_segment[0] - (x_segment[0] % 3)
        x_end = x_segment[1] + (2 - (x_segment[1] % 3))
    else:
        x_start = x_segment[0] + (3 - (x_segment[0] % 3))
        x_end = x_segment[1] - (1 + (x_segment[1] % 3))
    return in_aln[:, x_start:x_end + 1]


def site_frequency_spectrum(aln, ref_seq=None):
    aln_snps = seq_utils.get_snps(aln)
    minor_allele_frequency = calculate_snp_frequencies(aln_snps)
    f_bins = np.linspace(0, 1, len(aln) + 1)
    snp_bins = np.digitize(minor_allele_frequency, f_bins, right=True)
    unique_bins, bin_counts = utils.sorted_unique((snp_bins - 1).astype(int)) # shift bin indexing to start at 0
    counts = np.zeros(len(f_bins) - 1)
    counts[unique_bins] = bin_counts
    return counts


def calculate_sfs(aln, f_bins, x_max=33):
    aln_snps, x_snps = seq_utils.get_snps(aln, return_x=True)
    minor_allele_frequency = calculate_snp_frequencies(aln_snps)
    snp_bins = np.digitize(minor_allele_frequency, f_bins, right=True)
    unique_bins, bin_counts = utils.sorted_unique((snp_bins - 1).astype(int)) # shift bin indexing to start at 0
    counts = np.zeros(len(f_bins) - 1)
    counts[unique_bins] = bin_counts
    return counts


def calculate_2sfs(aln, f_bins, x_max=33):
    aln_snps, x_snps = seq_utils.get_snps(aln, return_x=True)
    minor_allele_frequency = calculate_snp_frequencies(aln_snps)
    snp_bins = np.digitize(minor_allele_frequency, f_bins, right=True)
    i_close, j_close = find_close_snp_pairs(x_snps, x_max)

    i_bin_idx = (snp_bins[i_close] - 1).astype(int)
    j_bin_idx = (snp_bins[j_close] - 1).astype(int)
    counts = np.zeros((len(f_bins) - 1, len(f_bins) - 1))
    for k, i in enumerate(i_bin_idx):
        j = j_bin_idx[k]
        counts[i, j] += 1
        counts[j, i] += 1
    return counts

def find_close_snp_pairs(x_snps, x_max):
    snp_distances = calculate_snp_distances(x_snps)
    i_arr, j_arr = np.where(snp_distances <= x_max)
    return i_arr[i_arr < j_arr], j_arr[i_arr < j_arr]


def calculate_snp_distances(x_snps):
    num_snps = len(x_snps)
    X = np.array([x_snps for i in range(num_snps)])
    return np.abs(X - X.T)

def count_gaps(seq):
    return seq.seq.count('-')

def count_snps(seq1, seq2):
    snps = 0
    aligned = 0
    for i, nucl in enumerate(seq1):
        if nucl != '-' and seq2[i] != '-':
            aligned += 1
            if nucl != seq2[i]:
                snps += 1
    return snps, aligned

def calculate_alignment_minpvalue_break(aln, dx=1, random_seed=None):
    '''
    Looks for recombinant breakpoint between all possible triplets in
    alignment by identifying point with least likely split in SNPs
    between putative recombinant and putative parent sequences. Algorithm
    is similar to Chimera (Posada & Crandall, 2001), but uses Fisher's
    exact test instead of chi2 test.
    '''
    if random_seed:
        np.random.seed(random_seed)

    num_alleles = len(aln)
    aln_arr = np.array(aln)
    results_dict = {}
    min_pvalues = []
    for i in range(num_alleles):
        for j in range(i):
            for k in range(j):
                triplet_results = calculate_triplet_maxchi2(aln_arr[[i, j, k], :], dx=dx)
                results_dict[(aln[i].id, aln[j].id, aln[k].id)] = triplet_results
                min_pvalues.append(min([t[2] for t in triplet_results]))
    results_dict['pvalues'] = min_pvalues
    return results_dict

def calculate_triplet_maxchi2(triplet_arr, x0=5, dx=1):
    num_sites = triplet_arr.shape[1]

    # Find endpoints for breakpoint search
    nogap_index = np.arange(num_sites)[np.sum((triplet_arr == '-'), axis=0) == 0]
    if nogap_index[0] > x0:
        x_min = nogap_index[0] + x0
    else:
        x_min = x0
    if nogap_index[-1] < num_sites - x0 - 1:
        x_max = nogap_index[-1] - x0
    else:
        x_max = num_sites - x0 - 1

    idx = np.arange(3)
    test_results = []
    # Loop through putative recombinants
    for i in range(3):
        rec = idx[0]
        p1, p2 = idx[1:]
        or_minp = 1
        x_minp = 0
        table_minp = [[0, 0], [0, 0]]
        p_min = 1
        #for x_break in range(x_min, x_max, dx):
        for x_break in nogap_index[::dx]:
            #p1_nogaps = np.sum((triplet_arr[[rec, p1], :] == '-'), axis=0) == 0
            #p1_matches = triplet_arr[rec, p1_nogaps] == triplet_arr[p1, p1_nogaps]
            #p2_nogaps = np.sum((triplet_arr[[rec, p2], :] == '-'), axis=0) == 0
            #p2_matches = triplet_arr[rec, p2_nogaps] == triplet_arr[p2, p2_nogaps]

            # Compare matches in non-gap regions only
            p1_matches = triplet_arr[rec, nogap_index] == triplet_arr[p1, nogap_index]
            p2_matches = triplet_arr[rec, nogap_index] == triplet_arr[p2, nogap_index]

            # Construct contingency table and calculate p-value
            p1b = np.sum(p1_matches[:x_break], dtype=np.int32)
            p1a = np.sum(p1_matches[x_break:], dtype=np.int32)
            p2b = np.sum(p2_matches[:x_break], dtype=np.int32)
            p2a = np.sum(p2_matches[x_break:], dtype=np.int32)
            contingency_table = [[p1b, p2b], [p1a, p2a]]
            oddsratio, pvalue = stats.fisher_exact(contingency_table)

            # Update minimum p-value
            if pvalue < p_min:
                p_min = pvalue
                x_minp = x_break
                or_minp = oddsratio
                table_minp = contingency_table

        test_results.append((x_minp, or_minp, p_min, table_minp))
        idx = np.roll(idx, -1)
    return test_results

def calculate_snp_run_null(aln_snps, num_permutations=1000, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    snp_array = np.array(aln_snps)
    stats_permutation = []
    for i in range(num_permutations):
        shuffled_arr = snp_array[:, np.random.permutation(snp_array.shape[1])]
        sscf, mcf = calculate_snp_run_stats(shuffled_arr)
        stats_permutation.append([sscf, mcf])
    return np.array(stats_permutation)

def calculate_snp_run_stats(aln_snps):
    snp_array = np.array(aln_snps)
    num_seqs = len(aln_snps)
    condensed_fragment_lengths = []
    for i in range(num_seqs):
        for j in range(i):
            seq1 = snp_array[i]
            seq2 = snp_array[j]
            x12 = calculate_snp_runs(seq1, seq2)
            condensed_fragment_lengths += list(x12)
    return np.sum(np.array(condensed_fragment_lengths)**2), max(condensed_fragment_lengths)

def calculate_snp_runs(seq1, seq2):
    run_length, diffs, run_value = rle(seq1 == seq2)
    if sum(run_value) > 0:
        xi = run_length[run_value]
    else:
        xi = np.zeros(1, dtype=np.int64)
    return xi

def rle(inarray):
    '''
    Run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values)
    '''
    ia = np.asarray(inarray)                # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return (z, p, ia[i])

def convert_pdist_tuples_to_divergences(pdist):
    pdiv = []
    for snp_tuple in pdist:
        pdiv.append(snp_tuple[0] / snp_tuple[1])
    return pdiv

def calculate_divergence(pdist):
    '''
    Apply Jukes-Cantor correction to pdist values.
    '''
    return np.abs(3. * np.log(1 - 4. * pdist / 3) / 4)

def write_alignment(aln, output_file, aln_format='fasta'):
    # Wrapper for AlignIO.write
    AlignIO.write(aln, output_file, aln_format)
    #with open(output_file, 'w') as out_handle:
    #    AlignIO.write(aln, out_handle, aln_format)

def write_alignment_array(aln_arr, output_file, id_index=None, aln_format='fasta'):
    if id_index is None:
        id_index = np.arange(aln_arr.shape[0]).astype(str)

    aln = []
    for i, seq_arr in enumerate(aln_arr):
        rec = SeqRecord(Seq(''.join(seq_arr)), id=id_index[i], description='')
        aln.append(rec)

    AlignIO.write(MultipleSeqAlignment(aln), output_file, aln_format)


def convert_array_to_alignment(aln_arr, id_index=None):
    if id_index is None:
        id_index = np.arange(aln_arr.shape[0]).astype(str)

    aln = []
    for i, seq_arr in enumerate(aln_arr):
        rec = SeqRecord(Seq(''.join(seq_arr)), id=id_index[i], description='')
        aln.append(rec)

    return MultipleSeqAlignment(aln)


def convert_list_to_alignment(rec_list):
    return MultipleSeqAlignment(rec_list)

def write_alleles_alignment(aln, output_file, condensed=True, anchor_seq_index=0, aln_range='all'):
    anchor_id = aln[anchor_seq_index].id
    anchor_seq = aln[anchor_seq_index].seq
    grouped_alignments, allele_counts = group_alignment_alleles(aln)
    for i, n in enumerate(allele_counts):
        grouped_alignments[i].description = f'{n} seqs'
        if grouped_alignments[i].seq == anchor_seq:
            anchor_allele_index = i
            anchor_allele_record = grouped_alignments[anchor_allele_index]
            anchor_seq = np.array(anchor_seq)

    with open(output_file, 'w') as handle:
        handle.write(f'>{anchor_allele_record.id} {anchor_allele_record.description}\n')
        if aln_range == 'all':
            handle.write(f'{"".join(anchor_seq)}\n')
        else:
            xi, xf = aln_range
            handle.write(f'{"".join(anchor_seq[xi:xf])}\n')

        for i in range(len(grouped_alignments)):
            if i != anchor_allele_index:
                record = grouped_alignments[i]
                record_seq = np.array(record.seq)
                if condensed == True:
                    for j in range(len(record_seq)):
                        record_seq[record_seq == anchor_seq] = '.'
                    handle.write(f'>{record.id} {record.description}\n')
                    if aln_range == 'all':
                        handle.write(f'{"".join(record_seq)}\n')
                    else:
                        handle.write(f'{"".join(record_seq[xi:xf])}\n')
                else:
                    handle.write(f'>{record.id} {record.description}\n')
                    if aln_range == 'all':
                        handle.write(f'{"".join(record_seq)}\n')
                    else:
                        handle.write(f'{"".join(record_seq[xi:xf])}\n')


def get_alignment_alleles(aln, output_type='list'):
    allele_index = list(range(len(aln)))[::-1]
    alleles = []
    while allele_index:
        i = allele_index.pop()
        alleles.append([aln[i]])
        visited_indices = []
        for j in allele_index:
            if aln[i].seq == aln[j].seq:
                alleles[-1].append(aln[j])
                visited_indices.append(j)
        for j in visited_indices:
            allele_index.remove(j)

    # Convert to output type
    if output_type == 'list':
        output = alleles
    elif output_type == 'alignment':
        output = MultipleSeqAlignment([recs[0] for recs in alleles])

    return output

def group_alignment_alleles(aln):
    unique_seqs = []
    seq_counts = []
    seq_ids = []
    for record in aln:
        if record.seq not in unique_seqs:
            unique_seqs.append(record.seq)
            seq_counts.append(1)
            seq_ids.append(record.id)
        else:
            i = unique_seqs.index(record.seq)
            seq_counts[i] += 1
    return [SeqRecord(unique_seqs[i], id=seq_ids[i]) for i in np.argsort(seq_counts)[::-1]], np.array(seq_counts)[np.argsort(seq_counts)[::-1]]

def calculate_allele_frequencies(alleles, relative_frequencies=True):
    allele_freqs = []
    for allele_records in alleles:
        allele_freqs.append(len(allele_records))
    if relative_frequencies:
        allele_freqs = np.array(allele_freqs) / sum(allele_freqs)
    else:
        allele_freqs = np.array(allele_freqs)
    return allele_freqs

def construct_allele_alignment(aln):
    allele_grouped_seqs, allele_counts = group_alignment_alleles(aln)
    return MultipleSeqAlignment(allele_grouped_seqs)


def concatenate_alignments(aln_list, how='all'):
    # Get concatenated rec IDs
    if how == 'all':
        temp = []
        for aln in aln_list:
            temp += [rec.id for rec in aln]
        unique_rec_ids = np.unique(temp)
    elif how == 'common':
        temp = []
        for aln in aln_list:
            temp.append(set([rec.id for rec in aln]))
        intersection_ids = temp[0]
        for idx in range(1, len(temp)):
            intersection_ids = intersection_ids.intersection(temp[idx])
        unique_rec_ids = np.array(list(intersection_ids))

    concatenated_aln = MultipleSeqAlignment([SeqRecord(Seq(''), name=rid, id=rid) for rid in unique_rec_ids])
    for aln in aln_list:
        aln_records = []
        aln_rec_ids = [rec.id for rec in aln]

        for rec_id in unique_rec_ids:
            if rec_id in aln_rec_ids:
                i = aln_rec_ids.index(rec_id)
                aln_records.append(aln[i])
            else:
                # Replace missing sequence with gaps
                gap_seq = Seq(''.join(aln.get_alignment_length() * ['-']))
                empty_rec = SeqRecord(gap_seq, name=rec_id, id=rec_id)
                aln_records.append(empty_rec)
        concatenated_aln = concatenated_aln + MultipleSeqAlignment(aln_records)

    return concatenated_aln


def cluster_divergence_matrix(divergence_matrix, linkage_method='average', return_sorted_indices=False, verbose=False):
    # Hierarchically remove NaN values until matrix is complete
    filtered_idx = np.arange(len(divergence_matrix))
    filtered_matrix = divergence_matrix.copy()
    num_nans = np.sum(np.isnan(filtered_matrix), axis=1)
    filtering_rounds = 0
    while np.sum(num_nans) > 0:
        nan_sorted_idx = np.argsort(num_nans)
        filtered_matrix = np.delete(filtered_matrix, nan_sorted_idx[-1], axis=0)
        filtered_matrix = np.delete(filtered_matrix, nan_sorted_idx[-1], axis=1)
        filtered_idx = np.delete(filtered_idx, nan_sorted_idx[-1])
        num_nans = np.sum(np.isnan(filtered_matrix), axis=1)
        filtering_rounds += 1
    if verbose:
        print(f'Number of filtered entries: {filtering_rounds}')

    # Calculate linkage matrix
    pdist = distance.squareform(filtered_matrix)
    if len(pdist) > 0:
        linkage = hclust.linkage(pdist, method=linkage_method, optimal_ordering=True)
    else:
        linkage = None

    if return_sorted_indices == True:
        dn = hclust.dendrogram(linkage, no_plot=True)
        sorted_idx = filtered_idx[dn['leaves']]
        return linkage, sorted_idx
    else:
        return linkage, filtered_idx


class SequenceTable:
    def __init__(self, f_data_tables, homolog_map='../results/reference_genomes/syn_homolog_map.dat'):
        self.ref_homolog_map = pickle.load(open(homolog_map, 'rb'))
        data_tables = pickle.load(open(f_data_tables, 'rb'))
        self.extract_contig_tables(data_tables)
        #self.make_locus_maps(data_tables)

    def extract_contig_tables(self, data_tables):
        contig_tables = self.make_ref_contig_tables()
        for sag_id, sag_data_tables in data_tables.items():
            sag_table = self.format_data_tables(sag_data_tables)
            sag_table['sag_id'] = sag_id
            contig_tables.append(sag_table)
        self.contig_table = pd.concat(contig_tables)

    def make_ref_contig_tables(self, ref_files={'osa_001':'../data/reference_genomes/CP000239.genbank', 'osbp_001':'../data/reference_genomes/CP000240.genbank'}):
        id_dict = {'osa_001':"OS-A", 'osbp_001':"OS-B'"}
        table_columns = ['sag_id', 'sequence', 'locus_tags', 'gene_locations', 'gene_strands']
        ref_table = pd.DataFrame(index=list(ref_files.keys()), columns=table_columns)
        for ref in ref_files:
            gb_record = SeqIO.read(ref_files[ref], 'genbank')
            ref_table.at[ref, 'sag_id'] = id_dict[ref]
            ref_table.at[ref, 'sequence'] = gb_record._seq
            locus_tags = []
            gene_locations = []
            gene_strands = []
            for record in gb_record.features[1:]:
                if record.type == 'CDS':
                    gene_tag = record.qualifiers['locus_tag'][0]
                    locus_tags.append(self.ref_homolog_map.find_lowest_tagged_homolog(gene_tag))
                    gene_locations.append((int(record.location.start), int(record.location.end)))
                    gene_strands.append(record.location.strand)
            ref_table.at[ref, 'locus_tags'] = locus_tags
            ref_table.at[ref, 'gene_locations'] = gene_locations
            ref_table.at[ref, 'gene_strands'] = gene_strands
        return [ref_table]

    def format_data_tables(self, sag_data_tables):
        table_columns = ['sag_id', 'sequence', 'locus_tags', 'gene_locations', 'gene_strands']
        sag_contig_table = sag_data_tables['contigs']
        contig_table = pd.DataFrame(index=sag_contig_table.index, columns=table_columns)
        contig_table[['sequence', 'gene_locations']] = sag_contig_table[['sequence', 'gene_locations']]
        contig_table['locus_tags'] = sag_contig_table['lowest_tagged_hits']

        contig_gene_strands = self.extract_gene_strands(sag_data_tables)
        for contig_id in contig_gene_strands:
            contig_table.at[contig_id, 'gene_strands'] = contig_gene_strands[contig_id]
        return contig_table

    def extract_gene_strands(self, sag_data_tables):
        contig_tables = sag_data_tables['contigs']
        gene_tables = sag_data_tables['genes']
        strand_dict = {'+':1, '-':-1}
        if 'strand_direction' in list(gene_tables.columns):
            contig_gene_strands = {}
            for contig in contig_tables.index:
                contig_gene_strands[contig] = [strand_dict[strand] if strand in strand_dict else 0 for strand in gene_tables.loc[gene_tables['contig'] == contig, 'strand_direction']]
        else:
            contig_gene_strands = contig_tables.loc[:, 'lowest_tagged_hit_strands'].to_dict()
        return contig_gene_strands

    '''
    def make_locus_maps(self, data_tables):
        locus_contig_map = make_locus_geneID_map(data_tables, self.ref_homolog_map)
        ref_contig_dict = {"OS-A":'osa_001', "OS-B'":'osbp_001'}
        for ref in ["OS-A", "OS-B'"]:
            locus_contig_map[ref] = [[] for i in range(len(locus_contig_map))]
            ref_contig = ref_contig_dict[ref]
            for i in range(len(self.contig_table.loc[ref_contig, 'locus_tags'])):
                locus_tag = self.contig_table.loc[ref_contig, 'locus_tags'][i]
                gene_loc = self.contig_table.loc[ref_contig, 'gene_locations'][i]
                if locus_tag in list(locus_contig_map.index):
                    locus_contig_map.loc[locus_tag, ref].append(f'{ref_contig}_{gene_loc[0]}_{gene_loc[1]}')
        self.locus_contig_map = locus_contig_map
    '''

    def extract_contig_seqs(self, anchor_locus, sag_ids=None, left_ext=0, right_ext=0):
        gene_id_series = self.extract_locus_gene_ids(anchor_locus, sag_ids)
        gene_records = []
        for sag_id, gene_ids in gene_id_series.iteritems():
            if len(gene_ids) > 0:
                sag_records = self.extract_contig_segment(gene_id_series[sag_id], left_ext, right_ext)
                for record in sag_records:
                    record.id = sag_id
                gene_records += sag_records
        return gene_records

    def extract_locus_gene_ids(self, locus_tag, sag_ids):
        gene_id_series = self.locus_contig_map.loc[locus_tag, :]
        if sag_ids:
            gene_id_series = gene_id_series[[sag_id for sag_id in gene_id_series.index if sag_id in sag_ids]]
        return gene_id_series

    def extract_contig_segment(self, gene_ids, left_ext=0, right_ext=0):
        gene_records = []
        for gene_id in gene_ids:
            contig_id, location = seq_utils.split_gene_id(gene_id)
            xl = location[0] - left_ext - 1
            xr = location[1] + right_ext
            segment_seq = self.contig_table.loc[contig_id, 'sequence'][xl:xr]
            gene_strand = self.get_gene_strand(contig_id, location)
            if gene_strand == -1:
                gene_records.append(SeqRecord(segment_seq.reverse_complement(), id='', description=f'{contig_id}:{xl + 1}-{xr}_complement'))
            else:
                gene_records.append(SeqRecord(segment_seq, id='', description=f'{contig_id}:{xl + 1}-{xr}'))
        return gene_records

    def get_gene_strand(self, contig_id, location):
        gene_index = [gene_location == location for gene_location in self.contig_table.loc[contig_id, 'gene_locations']]
        gene_strands = np.array(self.contig_table.loc[contig_id, 'gene_strands'])
        return gene_strands[gene_index][0]

    def is_gene_in_sag(self, sag_id, gene):
        return gene in np.concatenate(self.sag_contigs[sag_id])

    def make_ref_segment(self, ref_genome, anchor_gene, lim, hatched_loci=[]):
        ref_chromosome = seq_utils.construct_artificial_reference_genomes(self.ref_homolog_map)[ref_genome]
        ref_cds_map = self.ref_homolog_map.cds
        if ref_genome == 'osa':
            tag_prefix = 'CYA'
        else:
            tag_prefix = 'CYB'
        ref_gene_tags = [gene_tag for gene_tag in list(ref_cds_map.keys()) if tag_prefix in gene_tag]
        anchor_index = ref_chromosome.index(anchor_gene)
        ref_contig = []
        for i in range(anchor_index + lim[0], anchor_index + lim[1]):
            cds_record = ref_cds_map[ref_gene_tags[i]]
            if ref_chromosome[i] in hatched_loci:
                ref_contig.append(VisualGene(ref_chromosome[i], strand=cds_record.location.strand, location=[cds_record.location.start, cds_record.location.end], hatch='//////'))
            else:
                ref_contig.append(VisualGene(ref_chromosome[i], strand=cds_record.location.strand, location=[cds_record.location.start, cds_record.location.end]))
        return ref_contig


def run_codeml(alignment_file, tree_file='species.tree', out_file='results.out', run_mode='pairwise'):
    cml = codeml.Codeml(alignment=alignment_file,
                        tree=tree_file,
                        out_file='results.out',
                        working_dir='./scratch')

    # Set up default options
    cml.set_options(NSsites=[0, 1, 2])
    cml.set_options(model=0)
    cml.set_options(fix_kappa=0)
    cml.set_options(kappa=2)
    cml.set_options(fix_omega=0)
    cml.set_options(omega=0.4)
    cml.set_options(fix_alpha=1)
    cml.set_options(alpha=0)

    cml.set_options(noisy=9)
    cml.set_options(verbose=1)
    cml.set_options(seqtype=1)
    cml.set_options(CodonFreq=2)
    cml.set_options(clock=0)
    cml.set_options(aaDist=0)

    if run_mode == 'automatic':
        cml.set_options(runmode=2)
    else:
        cml.set_options(runmode=-2)

    results = cml.run()
    return results

def test_sequence_table():
    seq_table = SequenceTable('../results/tests/closely_related_sag_data_tables.dat')
    #seq_records = seq_table.extract_contig_seqs('CYB_0649', sag_ids=['UncmicMRedA02J17_2_FD', 'UncmicMRedA02N14_2_FD', 'UncmicMuRedA1H13_FD', "OS-B'"], left_ext=300, right_ext=-1577)
    #SeqIO.write(seq_records, '../results/tests/recombination_breakpoints/CYB_0649_intergenic_seq.fasta', 'fasta')
    #aln = AlignIO.read('../results/tests/recombination_breakpoints/CYB_0649_intergenic_aln.fasta', 'fasta')
    #write_alleles_alignment(aln, '../results/tests/recombination_breakpoints/CYB_0649_intergenic_condensed_aln.fasta', anchor_seq_index=3)

    seq_recomb_recs = seq_table.extract_contig_seqs('CYB_0674', sag_ids=['UncmicMuRedA1H13_FD'], left_ext=-1001, right_ext=209)
    seq_recomb = seq_recomb_recs[0]
    seq_recomb.description = 'CYB_0674_end+intergenic+CYB_0649'
    #seq_p1 = seq_table.extract_contig_seqs('CYB_0674', sag_ids=['UncmicMRedA02N14_2_FD'], left_ext=-908, right_ext=0)
    seq_p1_recs = seq_table.extract_contig_seqs('CYB_0674', sag_ids=['UncmicMRedA02N14_2_FD'], left_ext=50, right_ext=-1001)
    seq_p1 = seq_p1_recs[0]
    seq_p1.description = 'CYB_0674_end+intergenic'
    seq_records = [seq_recomb, seq_p1]
    print('CYB_0674...')
    for seq in seq_records:
        print(f'{seq.id}\n\t{seq.seq}')

    #seq_records = seq_table.extract_contig_seqs('CYB_0649', sag_ids=['UncmicMuRedA1H13_FD', 'UncmicMRedA02N14_2_FD'], left_ext=300, right_ext=-1500)
    #seq_records = seq_table.extract_contig_seqs('CYB_0649', sag_ids=['UncmicMRedA02N14_2_FD'], left_ext=0, right_ext=-1479)
    #seq_recomb2 = seq_table.extract_contig_seqs('CYB_0649', sag_ids=['UncmicMuRedA1H13_FD'], left_ext=0, right_ext=0)
    seq_p2_recs = seq_table.extract_contig_seqs('CYB_0649', sag_ids=['UncmicMRedA02N14_2_FD'], left_ext=251, right_ext=-1382)
    seq_p2 = seq_p2_recs[0]
    seq_p2.description = 'CYB_0649_start+intergenic'
    #seq_records = [seq_recomb2[0], seq_p2[0]]
    seq_records = [seq_p2]
    print('CYB_0649...')
    for seq in seq_records:
        print(f'{seq.id}\n\t{seq.seq}')

    seq_records = [seq_p1, seq_recomb, seq_p2]
    SeqIO.write(seq_records, '../results/tests/recombination_breakpoints/CYB_0674-0649_triple_seqs.fasta', 'fasta')

    aln = AlignIO.read('../results/tests/recombination_breakpoints/CYB_0674-0649_triple_manual_alnmt.fasta', 'fasta')
    write_alleles_alignment(aln, '../results/tests/recombination_breakpoints/CYB_0674-0649_triple_manual_alnmt_condensed.fasta', condensed=True, anchor_seq_index=1)

    #seq_table = SequenceTable('../results/single-cell/contig_and_genes_tables.sscs.dat')
    #triplet_ids = ['UncmicMuRedA1H18_FD', 'UncmicMRedA02K20_3_FD', 'UncmicMRedA02I10_FD']
    #triplet_ids = ['UncmicMRedA02I10_FD', 'UncmicMRedA02E17_2_FD', 'UncmicOcRedA2M20_FD']

    #seq_records = seq_table.extract_contig_seqs('CYB_2600', sag_ids=triplet_ids, left_ext=0, right_ext=0)
    #print('CYB_2600...')
    #for seq in seq_records:
    #    print(f'{seq.id}\n\t{seq.seq}')
    #SeqIO.write(seq_records, '../results/tests/recombination_breakpoints/CYB_2600_triplet_seqs.fasta', 'fasta')
    #SeqIO.write(seq_records, '../results/tests/recombination_breakpoints/CYB_2600_triplet2_seqs.fasta', 'fasta')

    #aln = AlignIO.read('../results/tests/recombination_breakpoints/CYB_2600_triplet_aln.fasta', 'fasta')
    #write_alleles_alignment(aln, '../results/tests/recombination_breakpoints/CYB_2600_triplet_aln_condensed.fasta', condensed=True, anchor_seq_index=1)

    aln = AlignIO.read('../results/tests/recombination_breakpoints/CYB_2600_triplet2_aln.fasta', 'fasta')
    write_alleles_alignment(aln, '../results/tests/recombination_breakpoints/CYB_2600_triplet2_aln_condensed.fasta', condensed=True, anchor_seq_index=1)

def test_linkage_metrics():
    test_aln1 = MultipleSeqAlignment([SeqRecord(Seq('AA'), id='sample1'), SeqRecord(Seq('AA'), id='sample2'),
        SeqRecord(Seq('TA'), id='sample3'), SeqRecord(Seq('TT'), id='sample4')])
    test_aln2 = MultipleSeqAlignment([SeqRecord(Seq('AA'), id='sample1'), SeqRecord(Seq('AA'), id='sample2'),
        SeqRecord(Seq('TA'), id='sample3'), SeqRecord(Seq('AT'), id='sample4'), SeqRecord(Seq('TT'), id='sample5')])
    test_aln3 = MultipleSeqAlignment([SeqRecord(Seq('AA'), id='sample1'), SeqRecord(Seq('TA'), id='sample2'),
        SeqRecord(Seq('AT'), id='sample3'), SeqRecord(Seq('AT'), id='sample4'), SeqRecord(Seq('AT'), id='sample5')])

    print(test_aln1)
    Dsq, denom = calculate_ld_matrices_vectorized(test_aln1, unbiased=False)
    Dprime = calculate_Dprime_vectorized(test_aln1)
    print(Dprime, '\n')
    print(Dsq, '\n\n', denom, '\n\n', Dsq / (denom + (denom == 0)), '\n\n\n')

    print(test_aln2)
    Dsq, denom = calculate_ld_matrices_vectorized(test_aln2, unbiased=False)
    Dprime = calculate_Dprime_vectorized(test_aln2)
    print(Dprime, '\n')
    print(Dsq, '\n\n', denom, '\n\n', Dsq / (denom + (denom == 0)), '\n\n\n')

    print(test_aln3)
    Dsq, denom = calculate_ld_matrices_vectorized(test_aln3, unbiased=False)
    Dprime = calculate_Dprime_vectorized(test_aln3)
    print(Dprime, '\n')
    print(Dsq, '\n\n', denom, '\n\n', Dsq / (denom + (denom == 0)), '\n\n\n')

def test_alignment_processing():
    empty_aln = MultipleSeqAlignment([])
    print(empty_aln)

    trimmed_aln = trim_alignment_gaps(empty_aln, start_gap_perc=0.05)
    print(trimmed_aln)
    filtered_aln = filter_alignment(trimmed_aln, gap_threshold=0.25)
    print(filtered_aln)
    subsample_aln = get_subsample_alignment(filtered_aln, ['UncmicMuRedA1H13_FD', 'UncmicMRedA02N14_2_FD'])
    print(subsample_aln)

def test_diversity_statistics():
    #f_aln = '../results/single-cell/sscs_pangenome/_aln_results/hisF-2_aln.fna'
    og_id = 'CYA_0743'
    f_aln = f'../results/single-cell/sscs_pangenome/_aln_results/{og_id}_aln.fna'
    aln = seq_utils.read_alignment(f_aln)
    print(aln)
    #trimmed_aln = trim_alignment_gaps(aln, start_gap_perc=0.05)
    trimmed_aln, x_trimmed = trim_alignment_and_remove_gaps(aln)

    print(trimmed_aln)
    f = calculate_snp_frequencies(trimmed_aln)
    print(f, np.sum((f > 0) & (f < 1)), len(f))
    H = calculate_snp_heterozygosity(trimmed_aln)
    print(H)
    utils.print_break()

    # Test separate calculation for each species
    pangenome_map = PangenomeMap(f_orthogroup_table='../results/single-cell/sscs_pangenome/filtered_orthogroups/sscs_annotated_single_copy_orthogroup_presence.tsv')
    og_table = pangenome_map.og_table
    metadata = MetadataMap()

    sag_ids = [col for col in og_table.columns if 'Uncmic' in col]
    print(sag_ids)
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    print(species_sorted_sags)
    for species in ['A', 'Bp']:
        species_gene_ids = pg_utils.read_gene_ids(og_table.loc[og_id, species_sorted_sags[species]], drop_none=True)
        print(species, species_gene_ids)
        if len(species_gene_ids) > 0:
            species_aln = get_subsample_alignment(trimmed_aln, species_gene_ids)
            print(species, species_aln)
            H = calculate_snp_heterozygosity(species_aln)
            print(H)

def average_rsq_matrices(r2_list):
    max_sites = 0
    max_index = 0
    for i in range(len(r2_list)):
        r2 = r2_list[i]
        if r2.shape[0] > max_sites:
            max_sites = r2.shape[0]
            max_index = i
    r2_avg = np.zeros((max_sites, max_sites))
    counts = np.zeros((max_sites, max_sites))
    for r2 in r2_list:
        num_sites = r2.shape[0]
        r2_avg[:num_sites, :][:, :num_sites] += r2
        counts[:num_sites, :][:, :num_sites] += 1
    r2_avg /= counts
    return r2_avg

def get_empty_alignment():
    return MultipleSeqAlignment([])

def is_nonempty_alignment(aln):
    return (aln is not None) and (len(aln) > 0) and (aln.get_alignment_length() > 0)

def sort_aln_rec_ids(aln, pangenome_map, metadata, by='species'):
    rec_ids = [rec.id for rec in aln]

    # Make mapping from SAG IDs to rec IDs
    sag_rec_map = {}
    for r in rec_ids:
        s = pangenome_map.get_gene_sag_id(r)
        sag_rec_map[s] = r

    sag_ids = [s for s in sag_rec_map]
    sorted_sag_ids = metadata.sort_sags(sag_ids, by=by)

    # Map species back to gene IDs
    sorted_rec_ids = {}
    for species in sorted_sag_ids:
        if len(sorted_sag_ids[species]) > 0:
            sorted_rec_ids[species] = [sag_rec_map[s] for s in sorted_sag_ids[species]]

    return sorted_rec_ids


def test_pairwise_distances(og_id='CYB_1073', species='Bp', f_orthogroup_table='../results/single-cell/sscs_pangenome/filtered_orthogroups/sscs_annotated_single_copy_orthogroup_presence.tsv'):
    #f_aln = '../results/tests/snp_blocks/multiple_haplotypes/CYB_1073_trimmed_aln.fna'
    f_aln = f'../results/single-cell/sscs_pangenome/_aln_results/{og_id}_aln.fna'
    aln = seq_utils.read_alignment(f_aln)

    metadata = MetadataMap()
    pangenome_map = PangenomeMap(f_orthogroup_table=f_orthogroup_table)
    species_sorted_sags = metadata.sort_sags(pangenome_map.get_sag_ids(), by='species')
    species_sag_ids = species_sorted_sags[species]
    og_table = pangenome_map.og_table
    gene_ids = pg_utils.read_gene_ids(og_table.loc[og_id, species_sag_ids], drop_none=True)
    species_aln = get_subsample_alignment(aln, gene_ids)
    aln, x_filtered = trim_alignment_and_remove_gaps(species_aln)

    print(aln)
    pdist_df = calculate_pairwise_distances(aln, metric='SNP')
    print(pdist_df)
    pdist = distance.squareform(pdist_df.values)
    print(pdist)
    Z = hclust.linkage(pdist, method='average', optimal_ordering=True)
    print('\n\n')

    '''
    pdist_df = calculate_pairwise_distances(aln, metric='SNP')
    pdist = distance.squareform(pdist_df.values)
    print(pdist)
    #Z = hclust.linkage(pdist, method='average', optimal_ordering=True)
    '''

    #new_pdist_df = calculate_pairwise_distances(aln, metric='SNP')
    #print(new_pdist_df)
    #new_pdist = distance.squareform(pdist_df.values)
    #print(new_pdist)
    #Z = hclust.linkage(pdist, method='average', optimal_ordering=True)
    #print(Z)

    x_sites = np.arange(100, 200)
    segment_aln = get_alignment_sites(aln, x_sites)
    print(segment_aln)
    pdist_df = calculate_pairwise_distances(aln, metric='SNP')
    print(pdist_df)
    pdist = distance.squareform(pdist_df.values)
    print(pdist)
    Z = hclust.linkage(pdist, method='average', optimal_ordering=True)

    x_sites = np.arange(100, 200)
    segment_aln = get_alignment_sites(aln, x_sites)
    print(segment_aln)
    pdist_df = calculate_pairwise_distances(aln, metric='SNP')
    print(pdist_df)
    pdist = distance.squareform(pdist_df.values)
    print(pdist)
    Z = hclust.linkage(pdist, method='average', optimal_ordering=True)


if __name__ == '__main__':
    #test_sequence_table()
    #test_linkage_metrics()
    #test_alignment_processing()
    #test_diversity_statistics()
    test_pairwise_distances()

