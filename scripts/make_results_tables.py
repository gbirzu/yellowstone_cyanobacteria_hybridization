import argparse
import numpy as np
import pandas as pd
import pickle
import glob
import os
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import matplotlib.pyplot as plt
import pangenome_utils as pg_utils
import plot_linkage_figures as plt_linkage
import matplotlib.tri as tri
import matplotlib.transforms as mtransforms
import matplotlib.lines as mlines
import matplotlib.patheffects as mpe
from Bio.Align import MultipleSeqAlignment
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from metadata_map import MetadataMap
from analyze_metagenome_reads import strip_sample_id
from plot_utils import *

mpl.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


###########################################################
# Fig. 2 panels: Gene-level analysis
###########################################################

def make_gene_tables(pangenome_map, args):
    species_cluster_genomes = pd.read_csv(f'{args.data_dir}labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)
    metadata = MetadataMap()

    print(species_cluster_genomes)

    species_hybrid_donor_frequency = {}
    for s in ['A', 'Bp']:
        donor_freq_table = make_donor_frequency_table(species_cluster_genomes, s, pangenome_map, metadata)
        donor_freq_table.to_csv(f'{args.output_dir}{s}_hybrid_donor_frequency.tsv', sep='\t')
        species_hybrid_donor_frequency[s] = donor_freq_table
        print(s)
        print(donor_freq_table)
        print('\n')

    pdist_dir = f'{args.pangenome_dir}pdist/'
    diversity_table = make_gene_nucleotide_diversity_table(pangenome_map, species_cluster_genomes, metadata, pdist_dir)
    print(diversity_table)
    diversity_table.to_csv(f'{args.output_dir}gene_diversity_table.tsv', sep='\t')


def make_donor_frequency_table(species_cluster_genomes, species, pangenome_map, metadata):
    if species == 'A':
        species_core_genome_clusters = species_cluster_genomes.loc[species_cluster_genomes['osa_location'].dropna().index, :].sort_values('osa_location')
        species_core_genome_clusters = species_core_genome_clusters.loc[species_core_genome_clusters['core_A'] == 'Yes', :]
        #species_core_genome_clusters = species_cluster_genomes.loc[species_cluster_genomes.index[species_cluster_genomes[species_sorted_sags[species]].notna().sum(axis=1).values > 0], :].copy()
    elif species == 'Bp':
        species_core_genome_clusters = species_cluster_genomes.loc[species_cluster_genomes['osbp_location'].dropna().index, :].sort_values('osbp_location')
        species_core_genome_clusters = species_core_genome_clusters.loc[species_core_genome_clusters['core_Bp'] == 'Yes', :]

    # Initialize frequency table
    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    donor_freq_table = pd.DataFrame(index=species_core_genome_clusters.index, columns=['CYA_tag', 'CYB_tag', 'osa_location', 'osbp_location', 'A', 'Bp', 'C', 'O'])
    donor_freq_table[['CYA_tag', 'CYB_tag', 'osa_location', 'osbp_location']] = species_core_genome_clusters[['CYA_tag', 'CYB_tag', 'osa_location', 'osbp_location']].values
    donor_freq_table[['A', 'Bp', 'C', 'O']] = 0

    #Fill table
    for o in donor_freq_table.index:
        gene_cluster_str = species_core_genome_clusters.loc[o, species_sorted_sags[species]].dropna().replace({'a':'A', 'b':'Bp'})
        gene_clusters = [utils.split_alphanumeric_string(c)[0] for c in np.concatenate(gene_cluster_str.str.split(','))]
        unique_clusters, cluster_counts = utils.sorted_unique(gene_clusters)
        donor_freq_table.loc[o, unique_clusters] = cluster_counts
    
    return donor_freq_table

def make_gene_nucleotide_diversity_table(pangenome_map, species_cluster_genomes, metadata, pdist_dir, species=['A', 'Bp', 'C']):
    # Make data columns
    data_cols =[]
    for sites in ['pS', 'pN']:
        for j, s2 in enumerate(species):
            data_cols.append(f'{s2}_{sites}_mean')
        for j, s2 in enumerate(species):
            for i in range(j):
                s1 = species[i]
                for metric in ['mean', 'std', 'min', 'max']:
                    data_cols.append(f'{s1}-{s2}_{sites}_{metric}')
    
    #diversity_table = pd.DataFrame(index=species_cluster_genomes.index.values, columns=['osa_location', 'osbp_location', 'A_piS', 'Bp_piS', 'C_piS', 'A-Bp_pS_mean', 'A-Bp_pS_std', 'A-C_pS_mean', 'A-C_pS_std', 'Bp-C_pS_mean', 'Bp-C_pS_std'])
    diversity_table = pd.DataFrame(index=species_cluster_genomes.index.values, columns=['osa_location', 'osbp_location'] + data_cols)
    diversity_table.loc[:, ['osa_location', 'osbp_location']] = species_cluster_genomes[['osa_location', 'osbp_location']]

    og_table = pangenome_map.og_table
    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')

    for og_id in diversity_table.index:
        for table in ['pS', 'pN']:
            f_pdist = f'{pdist_dir}{og_id}_cleaned_{table}.dat'
            if os.path.exists(f_pdist):
                pdist = pickle.load(open(f_pdist, 'rb'))
                for j, s2 in enumerate(species):
                    s2_gene_ids = get_gene_ids(og_table, og_id, species_sorted_sag_ids[s2])
                    if len(s2_gene_ids) == 0:
                        continue
                    for i in range(j):
                        s1 = species[i]
                        s1_gene_ids = get_gene_ids(og_table, og_id, species_sorted_sag_ids[s1])
                        if len(s1_gene_ids) == 0:
                            continue

                        pdist_values = pdist.loc[s1_gene_ids, s2_gene_ids].values
                        diversity_table.loc[og_id, f'{s1}-{s2}_{table}_mean'] = np.mean(pdist_values)
                        diversity_table.loc[og_id, f'{s1}-{s2}_{table}_std'] = np.std(pdist_values)
                        diversity_table.loc[og_id, f'{s1}-{s2}_{table}_min'] = np.min(pdist_values)
                        diversity_table.loc[og_id, f'{s1}-{s2}_{table}_max'] = np.max(pdist_values)
                    if len(s2_gene_ids) > 1:
                        pdist_values = utils.get_matrix_triangle_values(pdist.loc[s2_gene_ids, s2_gene_ids].values, k=1)
                        diversity_table.loc[og_id, f'{s2}_{table}_mean'] = np.mean(pdist_values)
                    #print(s2, s2_gene_ids, pdist_values)
            else:
                print(f_pdist)
        #print(diversity_table.loc[og_id, :])
        #print('\n')

        #if og_id == 'YSG_0050b':
        #    break

    return diversity_table


def get_gene_ids(og_table, og_id, sag_ids):
    subtable = og_table.loc[og_table['parent_og_id'] == og_id, sag_ids]
    gene_ids = []
    for i, row in subtable.iterrows():
        id_list = row.dropna().str.split(';')
        if len(id_list) > 0:
            gene_ids.append(np.concatenate(id_list))
    if len(gene_ids) > 0:
        gene_ids = np.concatenate(gene_ids)
    else:
        gene_ids = np.array(gene_ids)
    return gene_ids


###########################################################
# Single site statistics
###########################################################

def make_single_site_tables(pangenome_map, metadata, alignments_dir, args, species='A', main_cloud=False, sites='4D', aln_ext='cleaned_aln.fna'):
    og_table = pangenome_map.og_table
    species_sorted_sags = metadata.sort_sags(pangenome_map.get_sag_ids(), by='species')

    for species in ['A', 'Bp']:
        species_core_ogs = og_table.loc[og_table[f'core_{species}'] == 'Yes', 'parent_og_id'].unique()
        species_sag_ids = species_sorted_sags[species]
        num_site_alleles, mutation_spectra = calculate_single_site_statistics(pangenome_map, alignments_dir, species_sag_ids, species_core_ogs, main_cloud=main_cloud, sites=sites, aln_ext=aln_ext)

        # Save results
        f_num_site_alleles = f'{args.output_dir}{species}_num_site_alleles_{sites}.tsv'
        num_site_alleles.to_csv(f_num_site_alleles, sep='\t')
        f_mutation_spectra = f'{args.output_dir}{species}_mutation_spectra_{sites}.tsv'
        mutation_spectra.to_csv(f_mutation_spectra, sep='\t')

        if args.verbose:
            print(species)
            print(num_site_alleles)
            print(mutation_spectra)
            print('\n\n')


def calculate_single_site_statistics(pangenome_map, alignments_dir, species_sag_ids, species_core_ogs, 
        main_cloud=False, sites='4D', aln_ext='cleaned_aln.fna'):
    num_site_alleles = pd.DataFrame(index=species_core_ogs, columns=['1', '2', '3', '4', 'n'])
    mutation_spectra = pd.DataFrame(index=species_core_ogs, columns=['A', 'C', 'G', 'T', 'A<>C', 'A<>G', 'A<>T', 'C<>G', 'C<>T', 'G<>T'])
    for o in species_core_ogs:
        f_aln = f'{alignments_dir}{o}_{aln_ext}'
        if os.path.exists(f_aln):
            if main_cloud == False:
                species_aln = pangenome_map.read_sags_og_alignment(f_aln, o, species_sag_ids)
            else:
                aln_main_cloud = align_utils.read_main_cloud_alignment(f_aln, pangenome_map, metadata)
                filtered_gene_ids = pangenome_map.get_og_gene_ids(o, sag_ids=species_sag_ids)
                species_aln = align_utils.get_subsample_alignment(aln_main_cloud, filtered_gene_ids)

            if sites == '4D':
                species_aln = seq_utils.get_synonymous_sites(species_aln)

            if len(species_aln) > 0:
                # Check that alignment is not empty
                num_site_alleles.loc[o, ['1', '2', '3', '4']] = calculate_site_alleles_histogram(species_aln)
                num_site_alleles.loc[o, 'n'] = len(species_aln)
                mutation_spectra.loc[o, :] = calculate_allele_mutation_frequencies(species_aln)
        else:
            print(f'{f_aln} not found!')
    num_site_alleles['L'] = num_site_alleles[['1', '2', '3', '4']].sum(axis=1)
    num_site_alleles['fraction_polymorphic'] = num_site_alleles[['2', '3', '4']].sum(axis=1) / num_site_alleles['L']

    return num_site_alleles, mutation_spectra


def calculate_site_alleles_histogram(aln, site_type='nucl'):
    aln_arr = np.array(aln)
    if site_type == 'nucl':
        hist = np.zeros(4)
        num_alleles = count_site_alleles(aln)
        n, n_counts = utils.sorted_unique(num_alleles, sort='ascending', sort_by='tag')
        hist[n - 1] += n_counts
    return hist

def count_site_alleles(aln, excluded_alleles=['-', 'N']):
    aln_arr = np.array(aln)
    num_alleles = []
    for s in range(aln_arr.shape[1]):
        unique_alleles = np.unique(aln_arr[~np.isin(aln_arr[:, s], excluded_alleles), s])
        num_alleles.append(len(unique_alleles))
    return np.array(num_alleles)

def calculate_allele_mutation_frequencies(aln, nucleotides=['A', 'C', 'G', 'T']):
    mutation_spectrum = pd.Series(0, index=nucleotides + ['A<>C', 'A<>G', 'A<>T', 'C<>G', 'C<>T', 'G<>T'])

    # Get frequecy of fixed alleles
    sites_idx = np.arange(aln.get_alignment_length())
    num_alleles = count_site_alleles(aln)
    monoallelic_idx = sites_idx[num_alleles == 1]
    if len(monoallelic_idx) > 0:
        aln_monoallelic_arr = np.array(align_utils.get_alignment_sites(aln, monoallelic_idx))
        fixed_alleles, num_sites = utils.sorted_unique(aln_monoallelic_arr[0])
        mutation_spectrum[fixed_alleles[np.isin(fixed_alleles, nucleotides)]] = num_sites[np.isin(fixed_alleles, nucleotides)]

    # Get biallelic sites
    biallelic_idx = sites_idx[num_alleles == 2]
    
    if len(biallelic_idx) > 0:
        aln_biallelic_arr = np.array(align_utils.get_alignment_sites(aln, biallelic_idx))
        for s in range(aln_biallelic_arr.shape[1]):
            unique_alleles = np.sort(np.unique(aln_biallelic_arr[np.isin(aln_biallelic_arr[:, s], nucleotides), s]))
            mutation_spectrum['<>'.join(unique_alleles)] += 1

    return mutation_spectrum


if __name__ == '__main__':
    # Default variables
    alignments_dir = '../results/single-cell/reference_alignment/'
    pangenome_dir = '../results/single-cell/sscs_pangenome_v2/'
    output_dir = '../results/single-cell//main_figures_data/'
    data_dir = '../results/tests/main_figures_data/'
    metagenome_dir = '../results/metagenome/'
    f_orthogroup_table = f'{pangenome_dir}filtered_low_copy_clustered_core_mapped_labeled_cleaned_orthogroup_table.tsv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-A', '--alignments_dir', default=alignments_dir, help='Directory BLAST alignments against refs.')
    parser.add_argument('-D', '--data_dir', default=data_dir, help='Directory with data for main figures.')
    parser.add_argument('-M', '--metagenome_dir', default=metagenome_dir, help='Directory with results for metagenome.')
    parser.add_argument('-P', '--pangenome_dir', default=pangenome_dir, help='Pangenome directory.')
    parser.add_argument('-O', '--output_dir', default=output_dir, help='Main results directory.')
    parser.add_argument('-g', '--orthogroup_table', default=f_orthogroup_table, help='File with orthogroup table.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    args = parser.parse_args()

    random_seed = 12345
    rng = np.random.default_rng(random_seed)

    pangenome_map = pg_utils.PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    metadata = MetadataMap()
    #make_gene_tables(pangenome_map, args)

    make_single_site_tables(pangenome_map, metadata, '../results/single-cell/alignments/v2/core_ogs_cleaned/', args)
    #make_single_site_tables(pangenome_map, metadata, '../results/single-cell/alignments/v2/core_ogs_cleaned/', args, species='A')
    #make_single_site_tables(pangenome_map, metadata, '../results/single-cell/alignments/v2/core_ogs_cleaned/', args, species='Bp')
