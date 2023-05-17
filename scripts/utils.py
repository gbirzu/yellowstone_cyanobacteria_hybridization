import numpy as np
import pandas as pd
import string
import time
import re
import skbio
import scipy.cluster.hierarchy as hclust
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

osa_genome_size = 2932766
osbp_genome_size = 3046682


# Input/Output
#-----------------------------------------------------------------------------#

def read_genome_regions(file_path, num_loci=None, rename_samples=False):
    if rename_samples == True:
        alignment_df = pd.read_csv(file_path, sep='\t', index_col=1, header=None, low_memory=False)
        columns_dict = {0:'genome', 2:'ref_nucl', 3:'alt_nucl'}
        for sample_column in alignment_df.columns[3:]:
            columns_dict[sample_column] = f'sample_{sample_column - 3}'
        alignment_df = alignment_df.rename(columns=columns_dict)
    else:
        alignment_df = read_loci_table(file_path)
    if num_loci is not None:
        alignment_df = alignment_df.loc[alignment_df.index[:num_loci], :]
    return alignment_df

def read_loci_table(f_aligned_loci):
    alignment = pd.read_csv(f_aligned_loci, sep='\t', low_memory=False)

    # Remove potential empty last column
    columns = [column for column in alignment.columns if 'Unnamed' not in column]
    return alignment[columns]

def read_annotation_file(f_gff, format='gff'):
    '''
    Reads gene functional annotation GFF3 file from SAG assembly.
    Returns DataFrame with table
    '''
    gff_columns = ['contig', 'source', 'type', 'start_coord', 'end_coord', 'score', 'strand', 'phase', 'attributes']
    if format == 'gff-fasta':
        with open(f_gff, 'r') as in_handle:
            # Read GFF part
            gff_df = pd.DataFrame(columns=gff_columns)
            gff_rows = []
            while True:
                line = in_handle.readline()
                if not line or '##FASTA' in line:
                    break
                elif line[0] == '#':
                    continue
                row_values = line.strip().split('\t')
                gff_rows.append(row_values)
            gff_index = make_gff_index(gff_rows)
            #gff_index = [f'{row[0]}_{row[3]}_{row[4]}' for row in gff_rows]
            gff_df = pd.DataFrame(gff_rows, index=gff_index, columns=gff_columns)

            # Read FASTA part
            fasta_handle = SeqIO.parse(in_handle, 'fasta')
            fasta_records = {}
            for record in fasta_handle:
                fasta_records[record.id] = record
        return gff_df, fasta_records
    else:
        # Assume GFF format
        gff_data = pd.read_csv(f_gff, sep='\t', header=None, names=gff_columns)
        gff_index = gff_data['contig'] + '_' + gff_data['start_coord'].map(str) + '_' + gff_data['end_coord'].map(str)
        indexed_gff = gff_data.set_index(gff_index.values)
        return indexed_gff

def make_gff_index(gff_rows):
    gff_index = []
    for row in gff_rows:
        attr_dict = make_gff_attribute_dict(row[-1])
        if 'ID' in attr_dict:
            gff_index.append(attr_dict['ID'])
        else:
            gff_index.append(f'{row[0]}_{row[3]}_{row[4]}')
    return gff_index

def make_gff_attribute_dict(attr_str):
    attributes = attr_str.split(';')
    attr_dict = {}
    for attr in attributes:
        attr_comps = attr.split('=')
        attr_dict[attr_comps[0]] = attr_comps[1]
    return attr_dict

def read_fasta(in_fasta, return_seq_order=False):
    with open(in_fasta, 'r') as in_handle:
        fasta_handle = SeqIO.parse(in_handle, 'fasta')
        fasta_records = {}
        for record in fasta_handle:
            fasta_records[record.id] = record

        if return_seq_order == True:
            rec_ids = []
            with open(in_fasta, 'r') as in_handle:
                for line in in_handle.readlines():
                    line = line.rstrip()
                    if line.startswith('>'):
                        rec_id = line.split(' ')[0].strip('>')
                        rec_ids.append(rec_id)
            return fasta_records, rec_ids
        else:
            return fasta_records

def read_genbank_cds(f_genbank):
    genbank_record = SeqIO.read(f_genbank, 'genbank')
    genbank_cds_dict = {}
    for feature in genbank_record.features:
        if feature.type == 'CDS':
            locus_tag = feature.qualifiers['locus_tag'][0]
            genbank_cds_dict[locus_tag] = feature
    return genbank_cds_dict

def extract_genbank_cds_seqs(f_genbank, seq_type='nucl', add_descriptions=False):
    genbank_records = read_genbank_genomes(f_genbank)
    name, genome_seq = genbank_records.popitem()

    genbank_cds_dict = read_genbank_cds(f_genbank)
    cds_seqs = {}
    for locus, location_feature in genbank_cds_dict.items():
        if seq_type == 'prot':
            seq = location_feature.extract(genome_seq).translate()
        else:
            seq = location_feature.extract(genome_seq)

        if 'product' in location_feature.qualifiers and add_descriptions == True:
            description = location_feature.qualifiers['product'][0]
        else:
            description = ''
        cds_seqs[locus] = SeqRecord(seq, id=locus, description=description)
    return cds_seqs

def read_genbank_genomes(f_genbank):
    genbank_records = SeqIO.parse(open(f_genbank, 'r'), 'genbank')
    genomes = {}
    for record in genbank_records:
        genomes[record.name] = record.seq
    return genomes

def print_break(width=120):
    print('\n')
    print(''.join(width * ['-']))
    print(''.join(width * ['-']))
    print('\n\n')

def read_text_file(f_in):
    '''
    Returns list with each line of `f_in`.
    '''

    f_content = []
    with open(f_in, 'r') as in_handle:
        for line in in_handle.readlines():
            f_content.append(line.strip())
    return f_content

def write_linkage_to_newick(linkage_matrix, id_list, output_file):
    #tree = sktree.TreeNode.from_linkage_matrix(linkage_matrix, id_list)
    tree = skbio.tree.TreeNode.from_linkage_matrix(linkage_matrix, id_list)
    skbio.io.write(tree, 'newick', output_file)


def read_hybridization_table(f_table, length_cutoff=0):
    hybridization_table = pd.read_csv(f_table, sep='\t', index_col=0)
    filtered_hybridization_table = hybridization_table.loc[hybridization_table['num_genes'] > length_cutoff, :]
    return filtered_hybridization_table


#-----------------------------------------------------------------------------#

def split_alphanumeric_string(s):
    return re.findall(r'[^\W\d_]+|\d+', s)


def timeit(start_time):
    t = time.time() - start_time
    return f'{round(t/60):d} min {t%60:.2f} s'

def fill_zeros(x, y):
    '''
    Fills `y` with zeros at missing integers in `x`. Assumes `x` and `y` have same dimension.
    '''
    x_fill = np.arange(np.min(x), np.max(x) + 1)
    y_fill = np.zeros(len(x_fill))
    y_fill[(np.array(x) - np.min(x)).astype(int)] = y
    return x_fill, y_fill


def convert_snps_to_alleles(alignment_df):
    '''
    Converts .vcf table to numerical table, with following mapping:
    '.' : 0.0 = locus not present
    '0' : 1.0 = ancestral allele
    'i' : i + 1 = derived allele i
    '''
    alt_nucleotides = get_alt_nucls(alignment_df)
    samples = get_sample_columns(alignment_df)
    alleles = alignment_df[samples].values
    for alt in alt_nucleotides:
        alleles[np.where(alleles == alt)] = float(alt) + 1
    alleles[np.where(alleles == '.')] = 0.0
    alleles = alleles.astype(float)
    allele_table = alignment_df.copy()
    allele_table[samples] = alleles
    return allele_table

def get_alt_nucls(alignment_df):
    return ['0', '1', '2', '3']

def get_sample_columns(alignment_df):
    sample_columns = []
    for col in alignment_df.columns:
        if ('sample' in col) or ('Uncmic' in col):
            sample_columns.append(col)
    return sample_columns

def convert_alignment_to_numeric(multiseq_alignment, reference_sample):
    '''
    Converts Biopython MultipleSeqAlignment object into numerical representation using
    the following mapping:

    0 : missing nucleotide
    1 : reference allele
    2 : derived allele
    '''
    alignment_array = np.array(multiseq_alignment)
    alignment_numeric = np.zeros(alignment_array.shape, dtype=np.int)

    # Set all sites identical to reference to 1
    alignment_numeric += (alignment_array == alignment_array[reference_sample])

    # Set sites different from reference to 2
    alignment_numeric += 2*(alignment_array != alignment_array[reference_sample])

    # Set gaps to 0
    alignment_numeric[np.where(alignment_array == '-')] = 0

    return alignment_numeric


def strip_sample_names(sample_list):
    return [strip_sample_name(sample) for sample in sample_list]

def strip_sample_name(sample_str, replace=False):
    striped_str = sample_str.strip('Uncmic').strip('_FD')
    if replace == True:
        striped_str = striped_str.replace('Red', '')
    return striped_str

def calculate_locus_depth(allele_table_input):
    allele_table = allele_table_input.copy()
    samples = get_sample_columns(allele_table)
    depth = allele_table[samples].values
    depth[np.where(depth > 0)] = 1
    allele_table['locus_depth'] = np.sum(depth, axis=1)
    return allele_table

def calculate_alignement_gaps(depth_data, min_depth=0):
    '''
    Calculates gap sizes in input array. A gap of length n is a sequence of n consecutive zeros in the array.
    '''
    # Get index of non-zero elements
    index_contigs = np.where(depth_data[:, 1] > min_depth)[0]
    if len(index_contigs) == 0:
        # Return zero
        return np.array([0])
    else:
        # Subtract index assuming one contig
        cumulative_jumps = index_contigs - np.arange(len(index_contigs))
        cumulative_jumps = np.unique(cumulative_jumps)

        if len(cumulative_jumps) == 1:
            # Found just one jump
            return cumulative_jumps
        else:
            # Return jump sizes
            pos = np.unique(nonzero_jumps)
            return np.array([pos[i + 1] - pos[i] for i in range(len(pos) - 1)])

def read_alignment_depth(fname):
    return np.loadtxt(fname, delimiter='\t', usecols=(1, 2))

def coarse_grain_alignment(allele_table, l, gap_threshold=0.5):
    '''
    Coarse grains 'allele_table' over length 'l'. Segments of length 'l' with fraction of
    aligned loci less than 'gap_threshold' are assigned value 0. Rest of segments are assigned
    mean score (1 for reference allele, 2 for derived allele) in parts of segment without gaps.
    '''
    samples = get_sample_columns(allele_table)
    coarse_grained_table = pd.DataFrame(columns=samples, index=coarse_grain_distribution(allele_table['pos'], l).astype(int))
    for sample in samples:
        # Convert aligned loci to 'ref' (1) or 'SNP' (2).
        snp_list = allele_table[sample].values
        snp_list[np.where(snp_list > 2)[0]] = 2

        coarse_grained_table[sample] = coarse_grain_distribution(snp_list, l, nonzero=True)
        alignment_coverage = coarse_grain_distribution(snp_list.astype(bool), l)
        coarse_grained_table.loc[alignment_coverage < gap_threshold, sample] = 0

    # Recalculate 'locus_depth' return table
    return calculate_locus_depth(coarse_grained_table)

def coarse_grain_distribution(raw_data, l, nonzero=False):
    if l > 1:
        coarse_grain = []
        for k in range(0, len(raw_data), l):
            if nonzero == False:
                coarse_grain.append(np.mean(raw_data[k:k + l], axis=0))
            else:
                nonzero_values = raw_data[k:k + l][np.nonzero(raw_data[k:k + l])]
                coarse_grain.append(np.mean(nonzero_values, axis=0))
    else:
        coarse_grain = raw_data
    return np.array(coarse_grain)

def normalize_distribution(counts):
    return counts / np.sum(counts)

def cumulative_distribution(data_points, x_min='auto', x_max='auto', epsilon=1, normalized=True):
    if x_min == 'auto':
        x_min = min(data_points)
    if x_max == 'auto':
        x_max = max(data_points)
    #unique, counts = np.unique(data_points, return_counts=True)
    unique, counts = sorted_unique(data_points, sort='ascending', sort_by='tag')
    print(unique, counts, '\n')
    x = np.array([x_min - epsilon] + list(np.sort(unique)) + [x_max + epsilon])
    cumulative = np.zeros(len(x))
    for i, x0 in enumerate(unique):
        cumulative[i + 1] = cumulative[i] + counts[i]
    cumulative[-1] = np.sum(counts)
    if normalized == True:
        cumulative /= cumulative[-1]
    return x, cumulative

def extract_file_stem(fpath):
    fname = fpath.split('/')[-1]
    stem_comps = fname.split('.')
    if len(stem_comps) <= 2:
        stem = stem_comps[0]
    else:
        stem = '.'.join(stem_comps)
    return stem

def segments_union(segments_list):
    '''
    Performs the unions segments on real line using algorithm from Klee (1977).
    Segments should be given as list of tuples with endpoints.
    '''
    endpoints = []
    tags = []
    for s in segments_list:
        endpoints += [s[0], s[1]]
        tags += [1, -1]
    endpoints, tags = sort_endpoints(endpoints, tags)
    tags = tags[np.argsort(endpoints)]
    endpoints = sorted(endpoints)
    c = []
    d = []
    excess = 0
    m = 0

    for i, e in enumerate(endpoints):
        excess += tags[i]
        if excess == 1 and tags[i] == 1:
            m += 1
            c.append(e)
        elif excess == 0:
            d.append(e)

    return [(c[i], d[i]) for i in range(m)]

def sort_endpoints(endpoints, tags):
    n = len(endpoints)
    point_tuples = [(endpoints[i], tags[i]) for i in range(n)]
    sorted_tuples = sorted(point_tuples, key=lambda t:(t[0], -t[1]))
    return np.array([tpl[0] for tpl in sorted_tuples]), np.array([tpl[1] for tpl in sorted_tuples])

def extract_syn_gene_tag_from_fasta_header(header):
    return header.split('|')[-1]

def sorted_unique(tags, sort='descending', sort_by='counts'):
    unique_tags, tag_counts = np.unique(tags, return_counts=True)
    if sort_by == 'counts':
        index = np.argsort(tag_counts)
    else:
        # Sort by tags
        index = np.argsort(unique_tags)
    if sort == 'descending':
        index = index[::-1]
    return unique_tags[index], tag_counts[index]

def merge_tables(contig_genes_dict, table_type='genes', cell_ids=None):
    if cell_ids is None:
        #cell_ids = list(contig_genes_dict.keys())
        cell_ids = find_cells_w_table_type(contig_genes_dict, table_type)

    if len(cell_ids) > 0:
        merged_table = contig_genes_dict[cell_ids[0]][table_type]
        merged_table['cell_id'] = cell_ids[0]

        for i in range(1, len(cell_ids)):
            cell_table = contig_genes_dict[cell_ids[i]][table_type]
            cell_table['cell_id'] = cell_ids[i]
            merged_table = merged_table.append(cell_table, sort=False)
    else:
        merged_table = None
    return merged_table

def find_cells_w_table_type(contig_genes_dict, table_type):
    cell_ids = []
    for cell_id in contig_genes_dict.keys():
        if table_type in contig_genes_dict[cell_id]:
            cell_ids.append(cell_id)
    return cell_ids

def jaccard_set_distance(u, v):
    # Make sure inputs are sets
    s1 = set(u)
    s2 = set(v)
    return 1 - len(s1.intersection(s2)) / len(s1.union(s2))

def calculate_run_lengths(integer_sequence):
    '''
    Calculates length of consecutive subsequences in sequence of integers.

    --------
    integer_sequence : numpy array with sequence
    '''
    afine = np.arange(len(integer_sequence))
    runs = integer_sequence - afine
    edge_values, run_lengths = np.unique(runs, return_counts=True)
    return run_lengths

def sort_sag_species(sag_genes):
    '''
    Takes DataFrame with SAG genes and sorts SAGs into species
        index : gene triplet pattern ['XA', 'XB', 'XAB', ...]
        columns: SAG IDs
    '''
    sag_num_patterns = pd.DataFrame(index=sag_genes.index, columns=sag_genes.columns)
    for sag_id in sag_genes.columns:
        sag_num_patterns[sag_id] = [len(list(x)) for x in sag_genes[sag_id]]

    sag_fingerprints = sag_num_patterns / sag_num_patterns.sum(axis=0)
    species_sorted_sags = {'osa':[], 'osbp':[]}
    for sag_id in sag_fingerprints.columns:
        if sag_fingerprints.loc['XA-B', sag_id] > sag_fingerprints.loc['XB-A', sag_id]:
            species_sorted_sags['osa'].append(sag_id)
        else:
            species_sorted_sags['osbp'].append(sag_id)
    return species_sorted_sags

def calculate_homozygosity(counts):
    n_i = np.array(counts)
    n = np.sum(n_i)
    return np.sum((n_i / n)**2)

def get_matrix_triangle_values(matrix, triangle='upper', k=0):
    '''
    Returns values in upper or lower triangle of (n x n) matrix. Use
    k>0 to start from kth diagonal.
    '''
    if triangle == 'upper':
        tri_indices = np.triu_indices(len(matrix), k=k)
    elif triangle == 'lower':
        tri_indices = np.tril_indices(len(matrix), k=k)
    else:
        print('"triangle" variable value not recognized. Choose between ["upper", "lower"].')
        return None
    return matrix[tri_indices]

def find_longest_palindrome(s):
    '''
    A simple algorithm for finding the lengths of palindromic subsequences from given string s.
    Algorithm is quadratic in string length, which is good enough for our use.
    More sophisticated algorithms that are linear in string length can be found at
    https://en.wikipedia.org/wiki/Longest_palindromic_substring.
    '''

    # Add character breaks to string
    s_prime = ['|']
    for i in range(len(s)):
        s_prime += [s[i], '|']
    s_prime = np.array(s_prime)
    palindrome_lengths = np.zeros(len(s_prime)) # lengths of palindromes centered at each point in s_prime

    i_max = 0
    l_max = 0
    for i in range(1, len(s_prime) - 1):
        l = 0
        e = 1
        while (s_prime[i - e] == s_prime[i + e]) and (e < i) and ((i + e) < len(s_prime) - 1):
            if s_prime[i - e] != '|':
                l += 1
            e += 1
        if l > l_max:
            i_max = i
            l_max = l
    return l_max, i_max

def get_alphabetic_index(i):
    '''
    Converts integer ``i`` to character. Works for i < 26^2.
    '''
    alphabet = list(string.ascii_lowercase)
    if i < 26:
        idx = alphabet[i]
    else:
        j1 = i // 26 - 1
        j2 = i % 26
        idx = alphabet[j1] + alphabet[j2]
    return idx

def benjamini_hochberg_correction(pvalues, alpha):
    '''
    Returns index of tests where null hypothesis is rejected with FDR `alpha`
    using Bejamini-Hochberg correction for multiple hypothesis testing.
    '''

    idx = np.argsort(pvalues) # index of sorted p-values
    k = np.arange(1, len(pvalues) + 1) # rank of sorted p-values
    m = len(pvalues) # number of tests

    test_results = (pvalues[idx] <= alpha * k / m)
    if sum(test_results) > 0:
        k_max = max(k[test_results])
        is_rejected = idx[:k_max]
    else:
        is_rejected = []
    return is_rejected

def calculate_zero_run_lengths(x):
    '''
    Calculates length of runs of 0 values in input array.
    '''
    run_lengths = []
    i = 0
    l = 0
    is_run = False
    while i < len(x):
        if x[i] == 0:
            l += 1
            if is_run == False:
                is_run = True
        elif is_run == True:
            run_lengths.append(l)
            l = 0
            is_run = False
        i += 1
    if l > 0:
        run_lengths.append(l)
    return np.array(run_lengths)


def make_contingency_table(true_split, predicted_split):
    '''
    Takes two list of observations and returns 2x2 contingency table.

    --------

    true_split : contains arrays of true observations, grouped into two conditions (called here 1 and 2)
    predicted_split : predicted grouping of observations by condition; assumes first array is condition 1
                        and second array condition 2
    '''

    xt1, xt2 = true_split # true split = (true condition 1) + (true condition 2)
    xp1, xp2 = predicted_split # predicted split = (predicted condition 1) + (predicted condition 2)
    contingency_table = np.zeros((2, 2), dtype=int) # assume first index for true and second index for predicted
    contingency_table[0, 0] = np.sum(np.isin(xt1, xp1)) # true positives
    contingency_table[0, 1] = np.sum(np.isin(xt1, xp2)) # false negatives
    contingency_table[1, 0] = np.sum(np.isin(xt2, xp1)) # false positives
    contingency_table[1, 1] = np.sum(np.isin(xt2, xp2)) # true negatives
    return contingency_table

def fill_matrix_zeros(input_matrix, zeros_idx):
    '''
    Takes input matrix and adds zeros to rows and columns from given index.
        Size of new matrix will be sum of the original matrix and the zeros 
        index.
    '''
    matrix_length = len(input_matrix) + len(zeros_idx)
    output_matrix = np.zeros((matrix_length, matrix_length))
    idx = np.arange(matrix_length)
    input_idx = [index for index in idx if index not in zeros_idx]
    output_matrix[np.ix_(input_idx, input_idx)] = input_matrix
    return output_matrix


def cluster_matrix():
    dn = hclust.dendrogram(linkage, no_plot=True)
    sample_ids = np.array(pdist_df.index)
    ordered_sags = list(sample_ids[dn['leaves']])


def test_all():
    segments = [(3, 4), (1, 3), (2, 5), (3, 6), (9, 12), (8, 10)]
    print(segments)
    union = segments_union(segments)
    print(union)

    segments = [(1, 3), (2, 3), (4, 6), (9, 12), (8, 10)]
    print(segments)
    union = segments_union(segments)
    print(union)

    print(1, get_alphabetic_index(1))
    print(10, get_alphabetic_index(10))
    print(26, get_alphabetic_index(26))
    print(500, get_alphabetic_index(500))

if __name__ == '__main__':
    test_all()
