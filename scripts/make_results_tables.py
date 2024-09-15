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

def make_gene_tables(pangenome_map, args, dS_trough=0.2):
    species_cluster_genomes = pd.read_csv(f'{args.results_dir}main_figures_data/labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)
    metadata = MetadataMap()

    pdist_dir = f'{args.pangenome_dir}pdist/'
    gene_diversity_table = make_gene_nucleotide_diversity_table(pangenome_map, species_cluster_genomes, metadata, pdist_dir)
    gene_diversity_table.to_csv(f'{args.output_dir}gene_diversity_table.tsv', sep='\t')
    

    og_table = pangenome_map.og_table
    pS_ABp = gene_diversity_table['A-Bp_pS_mean'].values.astype(float)
    dS_ABp = align_utils.calculate_divergence(pS_ABp)
    #genomic_trenches_ogs = gene_diversity_table.index.values[gene_diversity_table['A-Bp_pS_mean'] < dS_trough, :]
    genomic_trenches_ogs = gene_diversity_table.index.values[dS_ABp < dS_trough]
    raw_osa_cds_dict = utils.read_genbank_cds('../data/reference_genomes/CP000239.genbank')
    osa_cds_dict = add_gene_id_map(raw_osa_cds_dict)
    raw_osbp_cds_dict = utils.read_genbank_cds('../data/reference_genomes/CP000240.genbank')
    osbp_cds_dict = add_gene_id_map(raw_osbp_cds_dict)
    genomic_trench_loci_table = construct_trench_loci_table(genomic_trenches_ogs, og_table, metadata, osa_cds_dict, osbp_cds_dict)
    genomic_trench_loci_table.to_csv(f'{args.output_dir}genomic_trench_loci_annotations.tsv', sep='\t')

    if args.verbose:
        print('Gene diversity table:\n', gene_diversity_table)
        print('\n\n')
        print('Diversity trough table:\n', genomic_trench_loci_table)



def make_gene_nucleotide_diversity_table(pangenome_map, species_cluster_genomes, metadata, pdist_dir, species=['A', 'Bp', 'C']):
    # Make data columns
    data_cols =[]
    for sites in ['pS', 'pN']:
        for j, s2 in enumerate(species):
            for metric in ['mean', 'std', 'min', 'max']:
                data_cols.append(f'{s2}_{sites}_{metric}')
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
                        diversity_table.loc[og_id, f'{s2}_{table}_std'] = np.std(pdist_values)
                        diversity_table.loc[og_id, f'{s2}_{table}_min'] = np.min(pdist_values)
                        diversity_table.loc[og_id, f'{s2}_{table}_max'] = np.max(pdist_values)
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


def add_gene_id_map(cds_dict):
    updated_dict = copy.copy(cds_dict)
    for cds in cds_dict:
        annot = cds_dict[cds]
        if 'gene' in annot.qualifiers:
            gene_id = annot.qualifiers['gene'][0]
            updated_dict[gene_id] = cds
    return updated_dict


def construct_trench_loci_table(parent_og_ids, og_table, metadata, osa_cds_dict, osbp_cds_dict):
    sites_df = pd.DataFrame(index=parent_og_ids, columns=['locus_tag', 'CYB_tag', 'genome_position', 'aa_length', 'annotation'])

    for pid in sites_df.index:
        candidate_og_ids = og_table.loc[og_table['parent_og_id'] == pid, 'locus_tag'].fillna('').astype(str).values
        annotated_og_ids = [oid for oid in candidate_og_ids if 'YSG' not in oid]
        if len(annotated_og_ids) > 0:
            sites_df.loc[pid, 'locus_tag'] = annotated_og_ids[0]
        else:
            sites_df.loc[pid, 'locus_tag'] = candidate_og_ids[0]

    # Add CYB locus tags
    cyb_tags = []
    for locus_id in sites_df['locus_tag']:
        if 'CYB' in locus_id:
            # Remove subspecies index if necessary
            if '-' in locus_id:
                locus_id = '-'.join(locus_id.split('-')[:-1])
            cyb_tags.append(locus_id)
        else: 
            # Remove subspecies index if necessary
            root_id = '-'.join(locus_id.split('-')[:-1])
            if locus_id in osbp_cds_dict:
                cyb_tags.append(osbp_cds_dict[locus_id])
            elif root_id in osbp_cds_dict:
                cyb_tags.append(osbp_cds_dict[root_id])
            else:
                cyb_tags.append(None)
    sites_df['CYB_tag'] = cyb_tags

    for idx in sites_df.index:
        cyb_tag = sites_df.loc[idx, 'CYB_tag']

        # Skip loci without annotation
        if cyb_tag is None:
            og_id = sites_df.loc[idx, 'locus_tag']
            if 'CYA' in og_id:
                locus_tag = og_id
                annotation = osa_cds_dict[locus_tag]
            else:
                aln_length = og_table.loc[og_table['parent_og_id'] == idx, 'avg_length'].mean()
                sites_df.loc[idx, 'aa_length'] = int(aln_length / 3)
                continue
        else:
            locus_tag = cyb_tag
            annotation = osbp_cds_dict[locus_tag]

        loc = annotation.location
        pos_str = f'{loc.start}-{loc.end}'
        sites_df.loc[idx, 'genome_position'] = pos_str

        qualifiers = annotation.qualifiers
        aa_length = len(qualifiers['translation'][0])
        sites_df.loc[idx, 'aa_length'] = aa_length

        if 'product' in qualifiers:
            sites_df.loc[idx, 'annotation'] = qualifiers['product'][0]

    return sites_df


###########################################################
# Single site statistics
###########################################################

def make_single_site_tables(pangenome_map, metadata, alignments_dir, args, species='A', main_cloud=False, sites='4D', aln_ext='cleaned_aln.fna'):
    og_table = pangenome_map.og_table
    species_sorted_sags = metadata.sort_sags(pangenome_map.get_sag_ids(), by='species')

    # Add tag for main cloud analysis
    if main_cloud:
        main_cloud_tag = '_main_cloud'
    else:
        main_cloud_tag = ''

    for species in ['A', 'Bp']:
        species_core_ogs = og_table.loc[og_table[f'core_{species}'] == 'Yes', 'parent_og_id'].unique()
        species_sag_ids = species_sorted_sags[species]
        num_site_alleles, mutation_spectra = calculate_single_site_statistics(pangenome_map, alignments_dir, species_sag_ids, species_core_ogs, main_cloud=main_cloud, sites=sites, aln_ext=aln_ext)

        # Save results
        f_num_site_alleles = f'{args.output_dir}{species}_num_site_alleles{main_cloud_tag}_{sites}.tsv'
        num_site_alleles.to_csv(f_num_site_alleles, sep='\t')
        f_mutation_spectra = f'{args.output_dir}{species}_mutation_spectra{main_cloud_tag}_{sites}.tsv'
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

###########################################################
# Hybridization tables
###########################################################

def make_hybridization_tables(pangenome_map, metadata, args):
    hybridization_dir = '../results/single-cell/hybridization/'
    hybridization_table = utils.read_hybridization_table(f'{hybridization_dir}hybridization_events.tsv', length_cutoff=args.contig_length_cutoff)
    fully_hybrid_contigs = np.array(hybridization_table.loc[hybridization_table['num_matches'] == 0, :].index) # filter out contigs without species alleles

    # Save hybrid contigs
    with open(f'{hybridization_dir}fully_hybrid_contigs.txt', 'w') as f_out:
        for c in fully_hybrid_contigs:
            f_out.write(f'{c}\n')

    species_cluster_genomes = initialize_sequence_cluster_haplotypes(pangenome_map, metadata)
    species_cluster_genomes = annotate_haplotype_gene_clusters(species_cluster_genomes, pangenome_map, metadata, excluded_contigs=fully_hybrid_contigs)
    species_cluster_genomes.to_csv(f'{args.output_dir}labeled_sequence_cluster_genomes.tsv', sep='\t')

    og_table = pangenome_map.og_table
    filtered_hybridization_table = hybridization_table.loc[hybridization_table.index[~hybridization_table.index.isin(fully_hybrid_contigs)], :].copy()
    hybridization_counts = make_hybridization_counts_table(filtered_hybridization_table, species_cluster_genomes, pangenome_map, metadata)
    hybridization_counts.to_csv(f'{args.output_dir}hybridization_counts_table.tsv', sep='\t')

    if args.verbose:
        print('Full hybridization table:')
        print(hybridization_table)
        print('\n')
        print('Hybridization table excluding contings without same species genes:')
        print(filtered_hybridization_table)
        print('\n')
        print('OGs with simple transfers:')
        print(hybridization_counts.loc[hybridization_counts['total_transfers'] > 0, :])
        print('\n\n')

    if args.verbose:
        print(species_cluster_genomes)
        print('\n\n')

    if args.validate_counts:
        # Validate hybridization counts
        sag_ids = pangenome_map.get_sag_ids()
        species_sorted_sags = sort_sags_by_species(sag_ids, metadata)
        counts_validation = count_gene_hybrids(species_cluster_genomes, species_sorted_sags)
        counts_validation.to_csv(f'{args.output_dir}hybridization_counts_table_validation.tsv', sep='\t')

        if args.verbose:
            print('Counts validation:')
            print(hybridization_counts)
            print(counts_validation)
            print(counts_validation)


def make_hybridization_counts_table(hybridization_table, species_core_genome_clusters, pangenome_map, metadata, min_og_frequency=0.2, og_type='parent_og_id'):
    og_table = pangenome_map.og_table
    og_ids = np.sort(og_table['parent_og_id'].dropna().unique())
    hybridization_counts_df = pd.DataFrame(index=og_ids, columns=['CYA_tag', 'CYB_tag', 'core_A', 'core_Bp', 'osa_location', 'osbp_location', 'Bp->A', 'C->A', 'O->A', 'A->Bp', 'C->Bp', 'O->Bp', 'total_transfers', 'A', 'Bp', 'M'])

    # Add core OGs
    hybridization_counts_df = add_core_og_labels(hybridization_counts_df, og_table)

    # Add locus tags
    hybridization_counts_df = add_locus_tags(hybridization_counts_df, og_table)
    hybridization_counts_df = sort_hybridization_events_by_type(hybridization_counts_df, pangenome_map, metadata)

    # Add genome positions
    osa_scale_factor = 1E-6 * 2932766 / 2905 # approximate gene position in Mb
    hybridization_counts_df.loc[:, 'osa_location'] = hybridization_counts_df['CYA_tag'].str.split('_').str[-1].astype(float) * osa_scale_factor
    osbp_scale_factor = 1E-6 * 3046682 / 2942  # approximate gene position in Mb
    hybridization_counts_df.loc[:, 'osbp_location'] = hybridization_counts_df['CYB_tag'].str.split('_').str[-1].astype(float) * osbp_scale_factor

    # Add other cluster counts
    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    for o in hybridization_counts_df.index:
        for species in ['A', 'Bp']:
            gene_cluster_str = species_core_genome_clusters.loc[o, species_sorted_sags[species]].dropna().replace({'a':'A', 'b':'Bp'})
            gene_clusters = [utils.split_alphanumeric_string(c)[0] for c in np.concatenate(gene_cluster_str.str.split(','))]
            unique_clusters, cluster_counts = utils.sorted_unique(gene_clusters)
            species_clusters = [species, 'M']
            idx = np.isin(unique_clusters, species_clusters)
            hybridization_counts_df.loc[o, unique_clusters[idx]] = cluster_counts[idx]

    #hybridization_counts_df = hybridization_counts_df.fillna(0)

    return hybridization_counts_df

def add_core_og_labels(locus_table, og_table):
    core_df = og_table[['parent_og_id', 'core_A', 'core_Bp']].replace('Yes', 1).groupby('parent_og_id').sum()
    core_df.loc[core_df['core_A'] > 0, 'core_A'] = 'Yes'
    core_df.loc[core_df['core_Bp'] > 0, 'core_Bp'] = 'Yes'
    locus_table.loc[:, ['core_A', 'core_Bp']] = core_df.loc[locus_table.index.values, ['core_A', 'core_Bp']]
    return locus_table

def add_locus_tags(locus_table, og_table):
    for og_id in locus_table.index:
        for col in ['CYA_tag', 'CYB_tag']:
            tags = og_table.loc[og_table['parent_og_id'] == og_id, col].dropna().values
            if len(tags) > 0:
                locus_table.loc[og_id, col] = tags[0]
    return locus_table

def sort_hybridization_events_by_type(hybridization_counts_df, pangenome_map, metadata):
    og_table = pangenome_map.og_table
    gene_cluster_mismatches = find_gene_cell_cluster_mismatches(og_table, metadata)
    mismatches_idx = list(gene_cluster_mismatches.index)
    transfer_columns = [col for col in hybridization_counts_df.columns if '->' in col] + ['total_transfers']
    hybridization_counts_df[transfer_columns] = 0 # initialize transfer type columns

    for i, og_id in enumerate(og_table['parent_og_id'].unique()):
        sog_ids = [idx for idx in og_table.loc[og_table['parent_og_id'] == og_id, :].index if idx in mismatches_idx]
        if len(sog_ids) == 1:
            mismatches_subtable = gene_cluster_mismatches.loc[sog_ids[0], :].dropna()
            mismatched_sag_ids = mismatches_subtable.index[mismatches_subtable == True].values
            donor_cluster = og_table.loc[sog_ids[0], 'sequence_cluster']
            host_clusters = [metadata.get_sag_species(sag_id) for sag_id in mismatched_sag_ids]
            for host_species in host_clusters:
                transfer_type = f'{donor_cluster}->{host_species}'
                hybridization_counts_df.loc[og_id, transfer_type] += 1
                hybridization_counts_df.loc[og_id, 'total_transfers'] += 1

        else:
            mismatches_subtable = gene_cluster_mismatches.loc[sog_ids, :].dropna(axis=1, how='all')
            mismatched_sag_ids = mismatches_subtable.columns[(mismatches_subtable == True).sum(axis=0) > 0].values
            donor_cluster = og_table.loc[sog_ids, 'sequence_cluster']
            host_clusters = [[metadata.get_sag_species(sag_id) for sag_id in mismatched_sag_ids[mismatches_subtable.loc[sog_id, mismatched_sag_ids] == True]] for sog_id in sog_ids] 
            for j, donor in enumerate(donor_cluster):
                if len(host_clusters[j]) > 0:
                    for host_species in host_clusters[j]:
                        transfer_type = f'{donor}->{host_species}'
                        hybridization_counts_df.loc[og_id, transfer_type] += 1
                        hybridization_counts_df.loc[og_id, 'total_transfers'] += 1

    return hybridization_counts_df


def find_gene_cell_cluster_mismatches(og_table, metadata):
    relevant_cluster_labels = ['A', 'Bp', 'C', 'O']

    # Make mismatched SAG IDs dict
    #   maps each SOG cluster to SAGs that would be mismatched if they contained the SOG
    sag_ids = [col for col in og_table.columns if 'Uncmic' in col]
    sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')
    mismatched_sags_dict = {}
    for cluster_label in relevant_cluster_labels:
        mismatched_sags_dict[cluster_label] = np.concatenate([sorted_sag_ids[l] for l in sorted_sag_ids if l != cluster_label])

    # Add mixed SOG subclusters
    mismatched_sags_dict['a'] = mismatched_sags_dict['A']
    mismatched_sags_dict['b'] = mismatched_sags_dict['Bp']
    relevant_cluster_labels += ['a', 'b']

    # Make mismatch table
    relevant_sog_ids = list(og_table.loc[og_table['sequence_cluster'].isin(relevant_cluster_labels), :].index)
    og_cluster_mismatch_table = pd.DataFrame(index=relevant_sog_ids, columns = sag_ids)
    for cluster_label in mismatched_sags_dict:
        mismatched_sag_ids = mismatched_sags_dict[cluster_label]
        subtable = og_table.loc[og_table['sequence_cluster'] == cluster_label, mismatched_sag_ids]
        og_cluster_mismatch_table.loc[list(subtable.index), mismatched_sag_ids] = subtable.notna()

    return og_cluster_mismatch_table


def initialize_sequence_cluster_haplotypes(pangenome_map, metadata):
    og_table = pangenome_map.og_table
    og_ids = np.sort(og_table['parent_og_id'].dropna().unique())
    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    synabp_sag_ids = np.concatenate((species_sorted_sags['A'], species_sorted_sags['Bp']))
    species_cluster_genomes = pd.DataFrame(index=og_ids, columns=np.concatenate((['CYA_tag', 'CYB_tag', 'core_A', 'core_Bp', 'osa_location', 'osbp_location'], sag_ids)) )

    # Add species core OGs
    species_cluster_genomes = add_core_og_labels(species_cluster_genomes, og_table)

    # Add locus tags
    species_cluster_genomes = add_locus_tags(species_cluster_genomes, og_table)

    # Add genome positions
    osa_scale_factor = 1E-6 * 2932766 / 2905 # approximate gene position in Mb
    species_cluster_genomes['osa_location'] = species_cluster_genomes['CYA_tag'].str.split('_').str[-1].astype(float) * osa_scale_factor
    osbp_scale_factor = 1E-6 * 3046682 / 2942  # approximate gene position in Mb
    species_cluster_genomes['osbp_location'] = species_cluster_genomes['CYB_tag'].str.split('_').str[-1].astype(float) * osbp_scale_factor

    return species_cluster_genomes

def annotate_haplotype_gene_clusters(species_cluster_genomes, pangenome_map, metadata, excluded_contigs=[]):
    og_table = pangenome_map.og_table.copy()
    sag_ids = pangenome_map.get_sag_ids()
    sag_species = ['A', 'Bp', 'C']
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')

    # Map species cluster labels
    og_table['sequence_cluster'] = og_table['sequence_cluster'].map({'A':'A', 'Bp':'Bp', 'M':'M', 'C':'C', 'O':'O', 'a':'A', 'b':'Bp'}, na_action='ignore')
    core_og_ids = np.array([o for o in species_cluster_genomes.index[(species_cluster_genomes[['core_A', 'core_Bp']] == 'Yes').any(axis=1)] if 'rRNA' not in o])

    num_multiple_alleles = 0
    num_filtered_multiple_alleles = 0
    for o in core_og_ids:
        og_idx = np.array(og_table.loc[og_table['parent_og_id'] == o, :].index)
        og_clusters = og_table.loc[og_idx, 'sequence_cluster'].copy()

        # Assign unique ID for multiple alleles from same OG
        if len(og_clusters.unique()) < len(og_clusters):
            cluster_labels, label_count = utils.sorted_unique(og_clusters.values)
            for c in cluster_labels[label_count > 1]:
                new_labels = [f'{c}{i+1}' for i in range(label_count[cluster_labels == c][0])]
                og_clusters[og_clusters == c] = new_labels

        # Check if any SAGs have more than one cluster
        num_genes = np.array([len(og_table.loc[og_idx, s].dropna()) for s in sag_ids])
        for s in sag_ids[num_genes > 1]:
            # Get index for covered OGs on valid contigs
            sag_allele_contigs = og_table.loc[og_idx, s].dropna().str.split('_').str[:2].str.join('_')
            filtered_idx = sag_allele_contigs.index[~sag_allele_contigs.isin(excluded_contigs)]

            # Assign cluster label
            s_alleles = og_clusters[filtered_idx].values
            species_cluster_genomes.loc[o, s] = ','.join(np.sort(s_alleles))

            num_multiple_alleles += 1
            if len(s_alleles) > 1:
                num_filtered_multiple_alleles += 1

        #for idx, c in og_clusters.iteritems():
        for idx, c in og_clusters.items():
            cluster_sag_ids = np.array(og_table.loc[idx, sag_ids[num_genes == 1]].dropna().index)
            species_cluster_genomes.loc[o, cluster_sag_ids] = c

            # Filter out OGs on excluded contigs
            covered_sag_ids = np.array(og_table.loc[idx, sag_ids[num_genes == 1]].dropna().index)
            sag_contig_ids = og_table.loc[idx, covered_sag_ids].str.split('_').str[:2].str.join('_') # Get contig IDs for covered SAGs
            filtered_contig_sag_ids = sag_contig_ids.index[sag_contig_ids.isin(excluded_contigs)]
            if len(filtered_contig_sag_ids) > 0:
                species_cluster_genomes.loc[o, filtered_contig_sag_ids] = np.nan

    return species_cluster_genomes


def sort_sags_by_species(sag_ids, metadata):
    '''
    Sorts SAG IDs by species and returns dict with just species present.
    '''
    temp = metadata.sort_sags(sag_ids, by='species')
    species_sorted_sags = {}
    for s in temp.keys():
        # Remove empty categories
        if len(temp[s]) > 0:
            species_sorted_sags[s] = temp[s]
    return species_sorted_sags


def count_gene_hybrids(species_cluster_genomes, species_sorted_sags, host_species=['A', 'Bp'], donor_species=['A', 'Bp', 'C', 'O']):
    '''
    Calculates distribution of hybridization events from species cluster labeled genomes.
    '''

    # Initialize table
    metadata_columns = ['CYA_tag', 'CYB_tag', 'core_A', 'core_Bp']
    transfer_columns = []
    for h in host_species:
        for d in donor_species:
            if h != d:
                transfer_columns.append(f'{d}->{h}')
    columns = metadata_columns + transfer_columns
    hybridization_counts_df = pd.DataFrame(0, index=species_cluster_genomes.index.values, columns=columns)
    hybridization_counts_df.loc[:, metadata_columns] = species_cluster_genomes.loc[:, metadata_columns]

    # Count hybrids
    species_cluster_genomes = species_cluster_genomes.fillna('')
    for host in host_species:
        host_sags = species_sorted_sags[host]
        host_donors = [s for s in donor_species if s != host]

        for donor in host_donors:
            col = f'{donor}->{host}'
            for s in host_sags:
                hybrids = species_cluster_genomes[s].str.contains(donor).astype(int)
                hybridization_counts_df.loc[:, col] += hybrids

    hybridization_counts_df['total_transfers'] = hybridization_counts_df[transfer_columns].sum(axis=1)

    return hybridization_counts_df


def make_pairwise_divergence_tables(pangenome_map, metadata, args):
    pdist_dir = f'{args.pangenome_dir}pdist/'
    core_og_table = pangenome_map.og_table
    core_og_ids = core_og_table['parent_og_id'].unique()
    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    sorted_sag_ids = np.concatenate([species_sorted_sags[s] for s in ['A', 'Bp', 'C']])
    
    d_arr = initialize_pdist_array(core_og_ids, sorted_sag_ids)
    pS_arr = initialize_pdist_array(core_og_ids, sorted_sag_ids)
    pN_arr = initialize_pdist_array(core_og_ids, sorted_sag_ids)
    d_idx = np.arange(len(sorted_sag_ids))

    for i, g in enumerate(core_og_ids):
        f_pdist = f'{pdist_dir}{g}_trimmed_pdist.dat'
        if os.path.exists(f_pdist):
            pdist = read_pdist_df(f_pdist)

            # Filter SAGs with more than one gene copy
            pdist_filtered = filter_pdist_df(pdist, pangenome_map)

            # Copy values
            copy_pdist_values(pdist_filtered, d_arr, i, sorted_sag_ids, d_idx)

        f_pS = f'{pdist_dir}{g}_cleaned_pS.dat'
        if os.path.exists(f_pS):
            pS = read_pdist_df(f_pS)
            pS_filtered = filter_pdist_df(pS, pangenome_map)
            copy_pdist_values(pS_filtered, pS_arr, i, sorted_sag_ids, d_idx)

        f_pN = f'{pdist_dir}{g}_cleaned_pN.dat'
        if os.path.exists(f_pN):
            pN = read_pdist_df(f_pN)
            pN_filtered = filter_pdist_df(pN, pangenome_map)
            copy_pdist_values(pN_filtered, pN_arr, i, sorted_sag_ids, d_idx)

        if args.verbose:
            print(g)
            print(d_arr[i])
            print(pS_arr[i])
            print(pN_arr[i])
            print('\n\n')

    pickle.dump((d_arr, core_og_ids, sorted_sag_ids), open(f'{args.output_dir}pdist_array.dat', 'wb'))
    pickle.dump((pS_arr, core_og_ids, sorted_sag_ids), open(f'{args.output_dir}pS_array.dat', 'wb'))
    pickle.dump((pN_arr, core_og_ids, sorted_sag_ids), open(f'{args.output_dir}pN_array.dat', 'wb'))

    d, x, y = pickle.load(open(f'{args.output_dir}pdist_matrix.dat', 'rb'))
    print('Saved array:')
    print(d, d.shape)
    print(x, len(x))
    print(y, len(y))
    print('\n\n')


def read_pdist_df(f_name):
    pdist = pickle.load(open(f_name, 'rb'))
    return pdist.fillna(0)

def initialize_pdist_array(core_og_ids, sorted_sag_ids):
    L = len(core_og_ids)
    n = len(sorted_sag_ids)
    d_arr = np.zeros((L, n, n))
    d_arr[:, :, :] = np.nan
    return d_arr

def filter_pdist_df(pdist, pangenome_map):
    g_sag_ids = np.array([pangenome_map.get_gene_sag_id(s) for s in pdist.index])
    g_idx = np.arange(pdist.shape[0])

    # Filter any SAGs with more than one gene copy
    s_unique, s_counts = utils.sorted_unique(g_sag_ids)
    filtered_idx = pdist.index.values[np.isin(g_sag_ids, s_unique[s_counts <= 1])]
    g_idx = g_idx[np.isin(g_sag_ids, s_unique[s_counts <= 1])]
    pdist_filtered = pdist.loc[filtered_idx, filtered_idx]

    sag_idx = g_sag_ids[np.isin(g_sag_ids, s_unique[s_counts <= 1])] 
    pdist_filtered.columns = sag_idx
    pdist_filtered.index = sag_idx
    return pdist_filtered

def copy_pdist_values(pdist_filtered, d_arr, i, sorted_sag_ids, d_idx):
    g_idx = d_idx[np.isin(sorted_sag_ids, pdist_filtered.index.values)]
    sag_idx = sorted_sag_ids[g_idx]
    pdist_filtered = pdist_filtered.loc[sag_idx, sag_idx]
    
    n = len(sorted_sag_ids)
    idx_1d = np.arange(n)[:, None] * n + np.arange(n)[None, :]
    g_idx_1d = idx_1d[g_idx, :][:, g_idx].flatten()

    d_arr[i].flat[g_idx_1d] = pdist_filtered.values.flatten()


if __name__ == '__main__':
    # Default variables
    alignments_dir = '../results/single-cell/reference_alignment/'
    pangenome_dir = '../results/single-cell/sscs_pangenome_v2/'
    output_dir = '../results/single-cell/main_figures_data/'
    results_dir = '../results/single-cell/'
    metagenome_dir = '../results/metagenome/'
    f_orthogroup_table = f'{pangenome_dir}filtered_low_copy_clustered_core_mapped_labeled_cleaned_orthogroup_table.tsv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-A', '--alignments_dir', default=alignments_dir, help='Directory BLAST alignments against refs.')
    parser.add_argument('-M', '--metagenome_dir', default=metagenome_dir, help='Directory with results for metagenome.')
    parser.add_argument('-P', '--pangenome_dir', default=pangenome_dir, help='Pangenome directory.')
    parser.add_argument('-R', '--results_dir', default=results_dir, help='Directory with analysis results.')
    parser.add_argument('-O', '--output_dir', default=output_dir, help='Main results directory.')
    parser.add_argument('-g', '--orthogroup_table', default=f_orthogroup_table, help='File with orthogroup table.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    parser.add_argument('--contig_length_cutoff', type=int, default=0, help='Filter hybrid contigs with this or fewer number of genes.')
    parser.add_argument('--validate_counts', action='store_true', help='Recalculate transfer counts from cluster labeled genomes for validation.')
    args = parser.parse_args()

    random_seed = 12345
    rng = np.random.default_rng(random_seed)

    pangenome_map = pg_utils.PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    metadata = MetadataMap()
    #make_gene_tables(pangenome_map, args)
    #make_single_site_tables(pangenome_map, metadata, '../results/single-cell/alignments/v2/core_ogs_cleaned/', args, sites='4D')
    #make_single_site_tables(pangenome_map, metadata, '../results/single-cell/alignments/v2/core_ogs_cleaned/', args, sites='all_sites')
    #make_hybridization_tables(pangenome_map, metadata, args)
    make_pairwise_divergence_tables(pangenome_map, metadata, args)


