import argparse
import numpy as np
import pandas as pd
import glob 
import os
import pickle
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import matplotlib.pyplot as plt
import pangenome_utils as pg_utils
import plot_linkage_figures as plt_linkage
import make_main_figures as main_figs
import er2
import scipy.stats as stats
import matplotlib.transforms as mtransforms
from analyze_metagenome_reads import strip_sample_id, strip_target_id, plot_abundant_target_counts
from syn_homolog_map import SynHomologMap
from metadata_map import MetadataMap
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from plot_utils import *



def make_og_diversity_figures(pangenome_map, args, fig_count, contig_length_cutoff=0, min_msog_fraction=0.5):
    metadata = MetadataMap()
    merged_donor_frequency_table = read_merged_donor_frequency_table(pangenome_map, metadata, args)
    plot_hybridization_pie_chart(merged_donor_frequency_table, savefig=f'{args.figures_dir}S{fig_count}_hybridization_pie.pdf')
    fig_count += 1

    plot_species_og_abundances(pangenome_map, savefig=f'{args.figures_dir}S{fig_count}_og_cluster_abundance_hist.pdf', verbose=args.verbose)
    fig_count += 1
    plot_species_og_abundances(pangenome_map, yscale='linear', savefig=f'{args.figures_dir}S{fig_count}_og_cluster_abundance_cuml.pdf', verbose=args.verbose, cumulative=True, density=True)

    return fig_count + 1


def read_merged_donor_frequency_table(pangenome_map, metadata, args):
    #species_cluster_genomes = pd.read_csv(f'{args.results_dir}supplement/sscs_labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)
    species_cluster_genomes = pd.read_csv(f'{args.results_dir}main_figures_data/labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)

    # Read donor frequncy tables
    species_donor_frequency_tables = {}
    temp = []
    for species in ['A', 'Bp']:
        #donor_frequency_table = main_figs.make_donor_frequency_table(species_cluster_genomes, species, pangenome_map, metadata)
        donor_frequency_table = make_donor_frequency_table(species_cluster_genomes, species, pangenome_map, metadata)
        donor_frequency_table['fraction_mixed_clusters'] = donor_frequency_table['M'] / donor_frequency_table[['A', 'Bp', 'C', 'O', 'M']].sum(axis=1)
        species_donor_frequency_tables[species] = donor_frequency_table
        temp.append(set(donor_frequency_table.index.values))
    common_og_ids = np.array(sorted(list(set.intersection(*temp))))

    # Merge tables
    merged_donor_frequency_table = pd.DataFrame(0, index=common_og_ids, columns=['non-hybrid', 'A simple hybrid', 'Bp simple hybrid', 'mosaic hybrid'])
    for species in ['A', 'Bp']:
        donor_freq_table = species_donor_frequency_tables[species]
        merged_donor_frequency_table.loc[common_og_ids, 'non-hybrid'] += donor_freq_table.loc[common_og_ids, species]
        donor_species = [s for s in ['A', 'Bp', 'C', 'O'] if s != species] 
        merged_donor_frequency_table.loc[:, f'{species} simple hybrid'] += donor_freq_table.loc[common_og_ids, donor_species].sum(axis=1)
        merged_donor_frequency_table.loc[:, f'mosaic hybrid'] += donor_freq_table.loc[common_og_ids, 'M'].fillna(0)

    return merged_donor_frequency_table

def make_donor_frequency_table(species_cluster_genomes, species, pangenome_map, metadata):
    if species == 'A':
        species_core_genome_clusters = species_cluster_genomes.loc[species_cluster_genomes['core_A'] == 'Yes', :].sort_values('osa_location')
    elif species == 'Bp':
        species_core_genome_clusters = species_cluster_genomes.loc[species_cluster_genomes['core_Bp'] == 'Yes', :].sort_values('osa_location')

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


def plot_hybridization_pie_chart(merged_donor_frequency_table, savefig=None):

    # Calculate bin numbers
    nonhybrid_og_ids = merged_donor_frequency_table.loc[merged_donor_frequency_table[['A simple hybrid', 'Bp simple hybrid', 'mosaic hybrid']].sum(axis=1) < 1, :].index.values
    merged_donor_frequency_table['fraction mosaic'] = merged_donor_frequency_table['mosaic hybrid'] / merged_donor_frequency_table[['non-hybrid', 'A simple hybrid', 'Bp simple hybrid', 'mosaic hybrid']].sum(axis=1)
    mosaic_og_ids = merged_donor_frequency_table.loc[merged_donor_frequency_table['mosaic hybrid'] > 0, :].index.values
    nonmosaic_og_ids = merged_donor_frequency_table.loc[merged_donor_frequency_table['mosaic hybrid'] < 1, :].index.values
    singleton_hybrid_og_ids = merged_donor_frequency_table.loc[merged_donor_frequency_table[['A simple hybrid', 'Bp simple hybrid']].sum(axis=1) == 1, :].index.values 
    singleton_hybrid_og_ids = singleton_hybrid_og_ids[np.isin(singleton_hybrid_og_ids, nonmosaic_og_ids)]
    nonsingleton_hybrid_og_ids = merged_donor_frequency_table.loc[merged_donor_frequency_table[['A simple hybrid', 'Bp simple hybrid']].sum(axis=1) > 1, :].index.values
    nonsingleton_hybrid_og_ids = nonsingleton_hybrid_og_ids[np.isin(nonsingleton_hybrid_og_ids, nonmosaic_og_ids)]

    bins = [len(nonhybrid_og_ids), len(mosaic_og_ids), len(singleton_hybrid_og_ids), len(nonsingleton_hybrid_og_ids)]
    #bin_labels = ['no hybrids', 'mixed\nclusters', 'singleton\nhybrids', 'non-singleton\nhybrids']
    bin_labels = [f'no gene\nhybrids ({bins[0]})', f'mosaic hybrids\nand other\nmixed clusters({bins[1]})', f'singleton\nhybrids\n({bins[2]})', f'non-singleton\nhybrids ({bins[3]})']
    #bin_labels = [f'no hybrids', f'mixed-species', f'singleton hybrids', f'non-singleton hybrids']

    #text_props = {'weight':'bold', 'size':12}
    text_props = {'size':10, 'color':'k'}
    text_fmt = r'%1.0f\%%'
    #text_fmt = r'{.0f}%'
    fig = plt.figure(figsize=(single_col_width, single_col_width))
    ax = fig.add_subplot(111)
    #ax.set_title(f'n = {len(np.concatenate(list(core_parent_ogs_dict.values())))}', fontweight='bold')
    ax.pie(bins, labels=bin_labels, autopct=text_fmt, textprops=text_props, labeldistance=1.2)
    #wedges, texts, autotexts = ax.pie(bins, autopct=lambda pct: label_func(pct, bins), textprops=dict(color='w'))

    #ax.legend(wedges, bin_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), frameon=False, fontsize=10)
    #plt.setp(autotexts, size=8, weight='bold')

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()


def plot_species_og_abundances(pangenome_map, savefig, yscale='log', verbose=False, **hist_kwargs):
    og_table = pangenome_map.og_table

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('orthogroup abundance', fontsize=14)
    ax.set_ylabel('orthogroups', fontsize=14)
    ax.set_yscale(yscale)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    x_bins = np.arange(og_table['num_seqs'].max() + 2)

    y_min = 1E-3
    color_dict = {'A':'tab:orange', 'Bp':'tab:blue', 'C':'tab:green'}
    label_dict = {'A':r'$\alpha$', 'Bp':r'$\beta$', 'C':r'$\gamma$'}
    #k_max_dict = {'A':37, 'Bp':74} # manual fit to mode of distribution
    #species_core_ogs = {}
    for species in ['A', 'Bp', 'C']:
        n = og_table.loc[og_table['sequence_cluster'] == species, 'num_seqs'].values
        #ax.hist(n, bins=x_bins, histtype='step', lw=2.5, label=label_dict[species], alpha=1.0, color=color_dict[species])
        ax.hist(n, bins=x_bins, histtype='step', lw=1.5, label=label_dict[species], alpha=1.0, color=color_dict[species], **hist_kwargs)

        if verbose:
            print(f'{len(n)} {species} clusters')
            if species == 'C':
                n_values, hist = utils.sorted_unique(n, sort_by='tag', sort='ascending')
                print(f'{species} cluster sizes:')
                print(n_values)
                print(hist)
                print(np.cumsum(hist))
            print('\n')

    ax.legend(fontsize=14, frameon=False)
    plt.tight_layout()
    plt.savefig(savefig)
    plt.close()


###########################################################
# Genetic diversity analysis
###########################################################

def make_genetic_diversity_figures(pangenome_map, args, fig_count, low_diversity_cutoff=0.05):
    metadata = MetadataMap()
    alignments_dir = '../results/single-cell/alignments/core_ogs_cleaned_4D_sites/'

    # Get single-site statistics tables
    syna_num_site_alleles, syna_mutation_spectra = get_single_site_statistics(pangenome_map, metadata, alignments_dir, args, species='A', main_cloud=False)
    low_diversity_ogs = np.array(syna_num_site_alleles.loc[syna_num_site_alleles['fraction_polymorphic'] < low_diversity_cutoff, :].index)
    high_diversity_ogs = np.array([o for o in syna_num_site_alleles.index if o not in low_diversity_ogs])
    syna_num_site_alleles['num_snps'] = syna_num_site_alleles[['2', '3', '4']].sum(axis=1)

    syna_mean_diversity = syna_num_site_alleles.loc[low_diversity_ogs, "piS"].mean()

    print(f'Low-diversity OGs: {syna_num_site_alleles.loc[low_diversity_ogs, "num_snps"].sum():.0f} 4D SNPs; {syna_num_site_alleles.loc[low_diversity_ogs, "L"].sum():.0f} 4D sites; piS = {syna_num_site_alleles.loc[low_diversity_ogs, "piS"].mean()}; {len(low_diversity_ogs)} loci')
    print(f'High-diversity OGs: {syna_num_site_alleles.loc[high_diversity_ogs, "num_snps"].sum():.0f} 4D SNPs; {syna_num_site_alleles.loc[high_diversity_ogs, "L"].sum():.0f} 4D sites; piS = {syna_num_site_alleles.loc[high_diversity_ogs, "piS"].mean()}; {len(high_diversity_ogs)} loci')

    syna_num_all_site_alleles, syna_all_site_mutation_spectra = get_single_site_statistics(pangenome_map, metadata, f'{args.results_dir}alignments/core_ogs_cleaned/', args, species='A', main_cloud=False, sites='all_sites', aln_ext='cleaned_aln.fna')
    syna_num_all_site_alleles['num_snps'] = syna_num_all_site_alleles[['2', '3', '4']].sum(axis=1)

    f_block_stats = f'{args.results_dir}supplement/A_all_sites_hybrid_linkage_block_stats.tsv'
    block_diversity_stats = pd.read_csv(f_block_stats, sep='\t')
    num_high_diversity_og_snps = syna_num_all_site_alleles.loc[high_diversity_ogs, "num_snps"].sum()
    num_block_snps = block_diversity_stats.loc[block_diversity_stats.index.values[np.isin(block_diversity_stats['og_id'].values, high_diversity_ogs)], 'num_snps'].sum()
    print(f'\t{num_high_diversity_og_snps:.0f} SNPs; {num_block_snps:.0f} SNPs in blocks; {syna_num_all_site_alleles.loc[high_diversity_ogs, "L"].sum():.0f} sites; fraction of block SNPs {num_block_snps / num_high_diversity_og_snps:.3f}; {len(high_diversity_ogs)} loci')

    rng = np.random.default_rng(args.random_seed)
    fig_count = plot_gene_polymorphisms_figure(syna_num_site_alleles, low_diversity_ogs, high_diversity_ogs, alignments_dir, metadata, rng, args, fig_count, low_diversity_cutoff=low_diversity_cutoff)

    #synbp_num_site_alleles, synbp_mutation_spectra = get_single_site_statistics(pangenome_map, metadata, alignments_dir, species='Bp', main_cloud=False)
    synbp_num_site_alleles, synbp_mutation_spectra = get_single_site_statistics(pangenome_map, metadata, alignments_dir, args, species='Bp', main_cloud=False)
    synbp_num_site_alleles['num_snps'] = synbp_num_site_alleles[['2', '3', '4']].sum(axis=1)
    #low_diversity_ogs = np.array(synbp_num_site_alleles.loc[synbp_num_site_alleles['fraction_polymorphic'] < low_diversity_cutoff, :].index)
    synbp_low_diversity_ogs = synbp_num_site_alleles.index.values
    synbp_high_diversity_ogs = np.array([o for o in synbp_num_site_alleles.index if o not in synbp_low_diversity_ogs])
    fig_count = plot_gene_polymorphisms_figure(synbp_num_site_alleles, synbp_low_diversity_ogs, synbp_high_diversity_ogs, alignments_dir, metadata, rng, args, fig_count, species='Bp', low_diversity_cutoff=low_diversity_cutoff, inset=False, fit='mean')

    fig_count = plot_genomic_trench_diversity(syna_num_site_alleles, synbp_num_site_alleles, low_diversity_ogs, rng, args, fig_count)

    print(f'Beta OGs: {synbp_num_site_alleles["num_snps"].sum()} 4D SNPs; {synbp_num_site_alleles["L"].sum()} 4D sites; piS = {synbp_num_site_alleles["piS"].mean()}; {len(synbp_num_site_alleles)} loci')
    synbp_mean_diversity = synbp_num_site_alleles["piS"].mean()

    #synbp_num_all_site_alleles, synbp_all_site_mutation_spectra = get_single_site_statistics(pangenome_map, metadata, f'{args.results_dir}alignments/core_ogs_cleaned/', species='Bp', main_cloud=False, sites='all_sites', aln_ext='cleaned_aln.fna')
    synbp_num_all_site_alleles, synbp_all_site_mutation_spectra = get_single_site_statistics(pangenome_map, metadata, f'{args.results_dir}alignments/core_ogs_cleaned/', args, species='Bp', main_cloud=False, sites='all_sites', aln_ext='cleaned_aln.fna')
    synbp_num_all_site_alleles['num_snps'] = synbp_num_all_site_alleles[['2', '3', '4']].sum(axis=1)

    print(synbp_num_all_site_alleles)

    f_block_stats = f'{args.results_dir}supplement/Bp_all_sites_hybrid_linkage_block_stats.tsv'
    block_diversity_stats = pd.read_csv(f_block_stats, sep='\t')
    num_total_snps = synbp_num_all_site_alleles.loc[:, "num_snps"].sum()
    num_block_snps = block_diversity_stats.loc[block_diversity_stats.index.values[np.isin(block_diversity_stats['og_id'].values, synbp_num_all_site_alleles.index.values)], 'num_snps'].sum()   
    print(f'\t{num_total_snps:.0f} SNPs; {num_block_snps:.0f} SNPs in blocks; {synbp_num_all_site_alleles.loc[:, "L"].sum():.0f} sites; fraction of block SNPs {num_block_snps / num_total_snps:.3f}')

    fig_count = plot_hybrid_gene_diversity(pangenome_map, metadata, syna_num_site_alleles, syna_mean_diversity, synbp_num_site_alleles, synbp_mean_diversity, rng, args, fig_count)

    print('\n\n')
    print(fig_count)

    fig_count = plot_alpha_spring_low_diversity(pangenome_map, metadata, low_diversity_ogs, rng, args, fig_count, num_bins=30, legend_fs=8)

    fig_count = plot_diversity_along_genome(pangenome_map, args, fig_count)

    fig_count = plot_pairwise_divergences(pangenome_map, metadata, args, fig_count)

    return fig_count


def get_single_site_statistics(pangenome_map, metadata, alignments_dir, args, species='A', hybridization_dir='../results/single-cell/hybridization/', main_cloud=False, sites='4D', aln_ext='4D_aln.fna'):
    f_num_site_alleles = f'{args.output_dir}{species}_num_site_alleles_{sites}_all_seqs.tsv'
    f_mutation_spectra = f'{args.output_dir}{species}_mutation_spectra_{sites}_all_seqs.tsv'

    if os.path.exists(f_num_site_alleles) and os.path.exists(f_mutation_spectra):
        num_site_alleles = pd.read_csv(f_num_site_alleles, sep='\t', index_col=0)
        mutation_spectra = pd.read_csv(f_mutation_spectra, sep='\t', index_col=0)
    else:
        species_cluster_genomes = pd.read_csv(f'{hybridization_dir}sscs_labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)
        temp = np.array(species_cluster_genomes.loc[species_cluster_genomes[f'core_{species}'] == 'Yes', :].index)
        species_core_ogs = temp[['rRNA' not in o for o in temp]]
        species_sorted_sags = metadata.sort_sags(pangenome_map.get_sag_ids(), by='species')
        species_sag_ids = species_sorted_sags[species]

        num_site_alleles = pd.DataFrame(index=species_core_ogs, columns=['1', '2', '3', '4', 'n'])
        mutation_spectra = pd.DataFrame(index=species_core_ogs, columns=['A', 'C', 'G', 'T', 'A<>C', 'A<>G', 'A<>T', 'C<>G', 'C<>T', 'G<>T'])
        min_num_seqs = 20
        for o in species_core_ogs:
            f_aln = f'{alignments_dir}{o}_{aln_ext}'
            if os.path.exists(f_aln):
                if main_cloud == False:
                    species_aln = pangenome_map.read_sags_og_alignment(f_aln, o, species_sag_ids)
                else:
                    aln_main_cloud = align_utils.read_main_cloud_alignment(f_aln, pangenome_map, metadata)
                    filtered_gene_ids = pangenome_map.get_og_gene_ids(o, sag_ids=species_sag_ids)
                    species_aln = align_utils.get_subsample_alignment(aln_main_cloud, filtered_gene_ids)
                if len(species_aln) > min_num_seqs:
                    num_site_alleles.loc[o, ['1', '2', '3', '4']] = calculate_site_alleles_histogram(species_aln)
                    num_site_alleles.loc[o, 'n'] = len(species_aln)
                    mutation_spectra.loc[o, :] = calculate_allele_mutation_frequencies(species_aln)
        num_site_alleles['L'] = num_site_alleles[['1', '2', '3', '4']].sum(axis=1)
        num_site_alleles['fraction_polymorphic'] = num_site_alleles[['2', '3', '4']].sum(axis=1) / num_site_alleles['L']

        # Add synonymous diversity
        piS, piS_idx = calculate_synonymous_diversity(num_site_alleles.index.values, species_sag_ids, pangenome_map, args)
        num_site_alleles.loc[piS_idx, 'piS'] = piS

        num_site_alleles.to_csv(f_num_site_alleles, sep='\t')
        mutation_spectra.to_csv(f_mutation_spectra, sep='\t')

    num_site_alleles_filtered = num_site_alleles.dropna()


    return num_site_alleles_filtered, mutation_spectra


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


def calculate_alpha_synonymous_diversity(pangenome_map, metadata, low_diversity_ogs, high_diversity_ogs, args):
    divergence_files = [f'{args.pangenome_dir}_aln_results/sscs_orthogroups_{j}_pS_matrices.dat' for j in range(10)]
    pangenome_map.read_pairwise_divergence_results(divergence_files)
    species_sorted_sags = metadata.sort_sags(pangenome_map.get_sag_ids(), by='species')
    syna_sag_ids = species_sorted_sags['A']
    piS_low, ld_index = pangenome_map.calculate_mean_pairwise_divergence(low_diversity_ogs, syna_sag_ids)
    piS_high, hd_index = pangenome_map.calculate_mean_pairwise_divergence(high_diversity_ogs, syna_sag_ids)
    piS_all, wg_index = pangenome_map.calculate_mean_pairwise_divergence(np.concatenate([low_diversity_ogs, high_diversity_ogs]), syna_sag_ids)

    return {'all':piS_all, 'low_diversity':piS_low, 'high_diversity':piS_high}


def calculate_synonymous_diversity(og_ids, sag_ids, pangenome_map, args):
    divergence_files = [f'{args.pangenome_dir}_aln_results/sscs_orthogroups_{j}_pS_matrices.dat' for j in range(10)]
    pangenome_map.read_pairwise_divergence_results(divergence_files)
    return pangenome_map.calculate_mean_pairwise_divergence(og_ids, sag_ids)


def plot_gene_polymorphisms_figure(num_site_alleles, low_diversity_ogs, high_diversity_ogs, alignments_dir, metadata, rng, args, fig_count, species='A', low_diversity_cutoff=0.05, ms=3, inset=True, fit='zero', label_fs=14):
    # Get alpha SAG IDs
    species_sorted_sags = metadata.sort_sags(pangenome_map.get_sag_ids(), by='species')
    species_sag_ids = species_sorted_sags[species]
    label_dict = {'A':r'$\alpha$ core genes', 'Bp':r'$\beta$ core genes'}

    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(121)
    ax.set_xlabel('fraction polymorphic 4D sites', fontsize=label_fs)
    ax.set_ylabel('counts', fontsize=label_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plot_polymorphic_sites_null_comparison(ax, num_site_alleles, low_diversity_ogs, species_sag_ids, alignments_dir, rng, xmax=1., num_bins=100, add_null=True, density=False, label=label_dict[species], low_diversity_cutoff=low_diversity_cutoff, inset=inset, fit=fit)

    ax = fig.add_subplot(122)
    ax.set_xlabel('synonymous diversity, $\pi_S$', fontsize=label_fs)
    ax.set_ylabel('counts', fontsize=label_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if species == 'Bp':
        legend = False
    else:
        legend = True
    plot_alpha_loci_diversity(ax, num_site_alleles, low_diversity_ogs, high_diversity_ogs, legend=legend)
    ax.set_xticks([0, 1E-4, 1E-3, 1E-2, 1E-1])

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_{species}_diversity_partition.pdf')
    plt.close()

    return fig_count + 1


def plot_polymorphic_sites_null_comparison(ax, num_site_alleles, low_diversity_ogs, syna_sag_ids, alignments_dir, rng, og_ids=None, label='data', xmax=1., num_bins=30, low_diversity_cutoff=0.05, ms=3, add_null=True, density=True, null_color='k', inset=True, fit='zero'):
    if og_ids is None:
        # Use all loci
        og_ids = num_site_alleles.index.values

    # Estimate effective mutation rates
    p_fixed = num_site_alleles.loc[og_ids, '1'].sum() / num_site_alleles.loc[og_ids, 'L'].sum()
    theta = -np.log(p_fixed)
    x_bins = np.linspace(0, xmax, num_bins)
    ax.hist(num_site_alleles.loc[og_ids, 'fraction_polymorphic'], bins=x_bins, density=density, label=label)

    # Add low-diversity inset
    if inset:
        ax_inset = ax.inset_axes([0.3, 0.45, 0.6, 0.5])
        ax_inset.set_xlim(0, low_diversity_cutoff)
        ax_inset.spines['right'].set_visible(False)
        ax_inset.spines['top'].set_visible(False)
        ax.indicate_inset_zoom(ax_inset, edgecolor='k')
        x_bins_inset = np.linspace(0, low_diversity_cutoff, 30)
        ax_inset.hist(num_site_alleles.loc[low_diversity_ogs, 'fraction_polymorphic'], bins=x_bins_inset, density=True, color='tab:blue', label=r'$\alpha$')

    # Add null
    if add_null:
        if fit == 'zero':
            p_fixed = num_site_alleles.loc[low_diversity_ogs, '1'].sum() / num_site_alleles.loc[low_diversity_ogs, 'L'].sum()
            theta = -np.log(p_fixed)
        elif fit == 'mean':
            theta = num_site_alleles.loc[low_diversity_ogs, 'fraction_polymorphic'].mean()
        #y_null, x_null = generate_polymorphic_sites_null(low_diversity_ogs, syna_sag_ids, alignments_dir, theta, rng, x_bins=x_bins)
        y_null, x_null = generate_polymorphic_sites_null(low_diversity_ogs, syna_sag_ids, pangenome_map, alignments_dir, theta, rng, x_bins=x_bins)
        y_null *= len(low_diversity_ogs) / np.sum(y_null)
        ax.plot(x_null, y_null, '-o', c=null_color, label=f'Poisson fit', ms=ms, lw=1)

        if inset:
            y_null, x_null = generate_polymorphic_sites_null(low_diversity_ogs, syna_sag_ids, pangenome_map, alignments_dir, theta, rng, x_bins=x_bins_inset)
            ax_inset.plot(x_null, y_null, '-o', c=null_color, ms=ms, lw=1)

    ax.legend(fontsize=10, frameon=False)



def generate_polymorphic_sites_null(og_ids, sag_ids, pangenome_map, alignments_dir, theta, rng, x_bins=None):
    if x_bins is None:
        x_bins = np.linspace(0, 1, 100)

    f_null = []
    for og_id in og_ids:
        f_aln = f'{alignments_dir}{og_id}_4D_aln.fna'
        aln_sags = pangenome_map.read_sags_og_alignment(f_aln, og_id, sag_ids)
        n = len(aln_sags)
        L = aln_sags.get_alignment_length()
        k = rng.poisson(lam=theta * L)
        f_null.append(k / L)
    hist_null, x_null = np.histogram(f_null, bins=x_bins, density=True)
    x_null = [np.mean(x_null[i:i + 2]) for i in range(len(x_null) - 1)]
    return hist_null, np.array(x_null)


def plot_alpha_loci_diversity(ax, num_site_alleles, low_diversity_ogs, high_diversity_ogs, epsilon=2E-5, xlim=(2E-5, 0.25), num_bins=50, legend=True):
    ax.set_xscale('symlog', linthresh=epsilon, linscale=0.1)
    ax.set_xlim(0., xlim[-1])
    temp = np.geomspace(epsilon, 0.25, num_bins)
    x_bins = np.concatenate([[0], temp])
    ax.hist(num_site_alleles.loc[low_diversity_ogs, 'piS'], bins=x_bins, density=False, label='non-hybrid loci', alpha=0.5)
    ax.hist(num_site_alleles.loc[high_diversity_ogs, 'piS'], bins=x_bins, density=False, label='hybrid loci', alpha=0.5)
    ax.axvline(epsilon, ls='--', color='k', lw=1)

    if legend:
        ax.legend(fontsize=10, frameon=False)


def plot_genomic_trench_diversity(syna_num_site_alleles, synbp_num_site_alleles, control_loci, rng, args, fig_count, num_bins=20, label_fs=14, epsilon=2E-5):
    genomic_troughs_df = pd.read_csv(f'{args.results_dir}supplement/genomic_trench_loci_annotations.tsv', sep='\t', index_col=0)
    genomic_troughs_df.sort_values('CYB_tag', inplace=True)
    genomic_troughs_df['CYB_id'] = genomic_troughs_df['CYB_tag'].str.split('_').str[1].astype(float)
    nif_gt_ogs = genomic_troughs_df.loc[(genomic_troughs_df['CYB_id'] >= 385) & (genomic_troughs_df['CYB_id'] <= 427), :].index.values
    syna_nif_gt_idx = [g for g in nif_gt_ogs if g in syna_num_site_alleles.index.values]
    synbp_nif_gt_idx = [g for g in nif_gt_ogs if g in synbp_num_site_alleles.index.values]

    print('Alpha nif trough pi_S:', syna_num_site_alleles.loc[syna_nif_gt_idx, 'piS'].mean())
    print('Beta nif trough pi_S', synbp_num_site_alleles.loc[synbp_nif_gt_idx, 'piS'].mean())
    print('\n\n')

    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(121)
    ax.set_xlim(0, 1)
    #ax.set_xscale('log')
    ax.set_xscale('symlog', linthresh=epsilon, linscale=0.1)
    ax.set_xlabel('synonymous diversity, $\pi_S$', fontsize=label_fs)
    #ax.set_xticks([1E-5, 1E-4, 1E-3, 1E-2, 1E-1])
    ax.set_xticks([0, 1E-4, 1E-3, 1E-2, 1E-1, 1])
    ax.set_ylabel('counts', fontsize=label_fs)
    ax.axvline(epsilon, ls='--', color='k', lw=1)

    syna_gt_idx = [g for g in genomic_troughs_df.index if g in syna_num_site_alleles.index.values]
    syna_control_og_list = [g for g in control_loci if (g not in syna_gt_idx) and (g in syna_num_site_alleles.index.values)]
    syna_control_idx = rng.choice(syna_control_og_list, size=len(syna_gt_idx), replace=False)

    #x_bins = np.geomspace(1E-5, 1, num_bins)
    x_bins = np.concatenate([[0], np.geomspace(epsilon, 1, num_bins)])
    #ax.hist(syna_num_site_alleles.loc[syna_gt_idx, 'piS'] + epsilon, bins=x_bins, color='tab:purple', alpha=0.5, label=r'$\alpha$ troughs')
    #ax.hist(syna_num_site_alleles.loc[syna_control_idx, 'piS'] + epsilon, bins=x_bins, color='tab:orange', alpha=0.5, label=r'$\alpha$ non-hybrid')
    ax.hist(syna_num_site_alleles.loc[syna_gt_idx, 'piS'], bins=x_bins, color='tab:purple', alpha=0.5, label=r'$\alpha$ troughs')
    ax.hist(syna_num_site_alleles.loc[syna_control_idx, 'piS'], bins=x_bins, color='tab:orange', alpha=0.5, label=r'$\alpha$ non-hybrid')
    ax.legend(loc='upper left', frameon=False)

    print(syna_num_site_alleles.loc[syna_gt_idx, 'piS'].mean(), syna_num_site_alleles.loc[syna_control_idx, 'piS'].mean())

    synbp_gt_idx = [g for g in genomic_troughs_df.index if g in synbp_num_site_alleles.index.values]
    synbp_control_og_list = [g for g in synbp_num_site_alleles.index.values if g not in syna_gt_idx]
    synbp_control_idx = rng.choice(synbp_control_og_list, size=len(synbp_gt_idx), replace=False)

    ax = fig.add_subplot(122)
    #ax.set_xscale('log')
    ax.set_xlim(0, 1)
    ax.set_xscale('symlog', linthresh=epsilon, linscale=0.1)
    ax.set_xlabel('synonymous diversity, $\pi_S$', fontsize=label_fs)
    #ax.set_xticks([1E-5, 1E-4, 1E-3, 1E-2, 1E-1])
    ax.set_xticks([0, 1E-4, 1E-3, 1E-2, 1E-1, 1])
    ax.set_ylabel('counts', fontsize=label_fs)
    ax.axvline(epsilon, ls='--', color='k', lw=1)

    #ax.hist(synbp_num_site_alleles.loc[synbp_gt_idx, 'piS'] + epsilon, bins=x_bins, color='tab:purple', alpha=0.5, label=r'$\beta$ troughs')
    #ax.hist(synbp_num_site_alleles.loc[synbp_control_idx, 'piS'] + epsilon, bins=x_bins, color='tab:blue', alpha=0.5, label=r'$\beta$ core')
    ax.hist(synbp_num_site_alleles.loc[synbp_gt_idx, 'piS'] + epsilon, bins=x_bins, color='tab:purple', alpha=0.5, label=r'$\beta$ troughs')
    ax.hist(synbp_num_site_alleles.loc[synbp_control_idx, 'piS'] + epsilon, bins=x_bins, color='tab:blue', alpha=0.5, label=r'$\beta$ core')
    ax.legend(loc='upper left', frameon=False)

    print(synbp_num_site_alleles.loc[synbp_gt_idx, 'piS'].mean(), synbp_num_site_alleles.loc[synbp_control_idx, 'piS'].mean())

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_genomic_trough_diversity.pdf')
    plt.close()

    return fig_count + 1


def plot_hybrid_gene_diversity(pangenome_map, metadata, syna_num_site_alleles, syna_mean_diversity, synbp_num_site_alleles, synbp_mean_diversity, rng, args, fig_count, x0=0, dx1=0.25, dx2=0.2):
    species_cluster_genomes = pd.read_csv(f'{args.results_dir}supplement/sscs_labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)
    syna_hybrid_donor_frequency_table = main_figs.make_donor_frequency_table(species_cluster_genomes, 'A', pangenome_map, metadata)
    synbp_hybrid_donor_frequency_table = main_figs.make_donor_frequency_table(species_cluster_genomes, 'Bp', pangenome_map, metadata)

    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    print('\n\n')

    label_fs = 14
    epsilon = 2E-5
    linscale = 0.25
    ms = 6**2
    lw = 0.1
    colors_dict = {'A':'tab:orange', 'Bp':'tab:blue', 'C':'tab:green', 'O':'tab:purple'}

    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(121)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([r'$\beta \rightarrow \alpha$', r'$\gamma \rightarrow \alpha$', r'$X \rightarrow \alpha$'], fontsize=label_fs)
    #ax.set_xticklabels([r'$\beta$', r'$\gamma$', r'$X$'], fontsize=label_fs)
    #ax.set_xticklabels([r'Bp', r'C', 'X'], fontsize=label_fs)
    ax.set_ylabel(r'synonymous diversity, $\pi_S$', fontsize=label_fs)
    #ax.set_ylim(8E-6, 1)
    #ax.set_yscale('log')
    ax.set_yscale('symlog', linthresh=epsilon, linscale=linscale)
    ax.set_ylim(-epsilon, 1)
    ax.set_yticks([0, 1E-4, 1E-3, 1E-2, 1E-1, 1])
    ax.axhline(syna_mean_diversity, lw=2, color=colors_dict['A'], alpha=0.5)
    ax.axhline(epsilon, ls='--', color='k', lw=1)

    counter = np.array([0, 0])
    for hybrid_cluster in ['Bp', 'C', 'O']:
        #print(hybrid_cluster, 'hybrids')
        for g in syna_hybrid_donor_frequency_table.loc[syna_hybrid_donor_frequency_table[hybrid_cluster] > 1, :].index:
            locus_clusters = species_cluster_genomes.loc[g, species_sorted_sags['A']]
            hybrid_cluster_sag_ids = locus_clusters[locus_clusters == hybrid_cluster].index.values
            f_aln = f'{args.results_dir}alignments/core_ogs_cleaned/{g}_cleaned_aln.fna'
            aln = seq_utils.read_alignment_and_map_sag_ids(f_aln, pangenome_map)
            aln_hybrids = align_utils.get_subsample_alignment(aln, hybrid_cluster_sag_ids)
            pN, pS = seq_utils.calculate_pairwise_pNpS(aln_hybrids)
            pS_values = utils.get_matrix_triangle_values(pS.values, k=1)
            #y = np.mean(pS_values) + 1E-5
            y = np.mean(pS_values)
            x = np.array([x0, x0 + dx1 / 2 + dx2]) + rng.uniform(-dx1 / 2, dx1 / 2, size=2)
            #ax.scatter(x[0], y, 16, marker='o', color=colors_dict[hybrid_cluster])
            ax.scatter(x[0], y, ms, marker='o', fc=colors_dict[hybrid_cluster], ec='w', lw=lw, alpha=0.6)
            if (g in synbp_num_site_alleles.index.values) and (hybrid_cluster == 'Bp'):
                #print(g, np.mean(pS_values), synbp_num_site_alleles.loc[g, 'piS'], len(aln_hybrids))
                yc = synbp_num_site_alleles.loc[g, 'piS']
                #ax.scatter(x[1], yc, 16, marker='s', color=colors_dict[hybrid_cluster])
                ax.scatter(x[1], yc, ms, marker='s', fc=colors_dict[hybrid_cluster], ec='w', lw=lw, alpha=0.6)
                #ax.plot(x, [y, yc], c='k', lw=0.75, alpha=0.5)
                ax.plot(x, [y, yc], c=colors_dict[hybrid_cluster], lw=0.75, alpha=0.6)
            else:
                #print(g, np.mean(pS_values), len(aln_hybrids))
                pass

            counter[0] += 1
            if y < 1E-4:
                counter[1] += 1

        x0 += 1
        #print('\n')

    print(counter)
    print('\n\n')
    #ax.set_yticks([1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1])

    counter = np.array([0, 0])
    ax = fig.add_subplot(122)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([r'$\alpha \rightarrow \beta$', r'$\gamma \rightarrow \beta$', r'$X \rightarrow \beta$'], fontsize=label_fs)
    #ax.set_xticklabels([r'$\alpha$', r'$\gamma$', 'X'], fontsize=label_fs)
    #ax.set_xticklabels([r'A', r'C', 'X'], fontsize=label_fs)
    ax.set_ylabel(r'synonymous diversity, $\pi_S$', fontsize=label_fs)
    #ax.set_ylim(8E-6, 1)
    #ax.set_yscale('log')
    ax.set_yscale('symlog', linthresh=epsilon, linscale=linscale)
    ax.set_ylim(-epsilon, 1)
    ax.set_yticks([0, 1E-4, 1E-3, 1E-2, 1E-1, 1])
    ax.axhline(epsilon, ls='--', color='k', lw=1)
    ax.axhline(synbp_mean_diversity, lw=2, color=colors_dict['Bp'], alpha=0.5)

    x0 = 0
    for hybrid_cluster in ['A', 'C', 'O']:
        #print(hybrid_cluster, 'hybrids')
        for g in synbp_hybrid_donor_frequency_table.loc[synbp_hybrid_donor_frequency_table[hybrid_cluster] > 1, :].index:
            locus_clusters = species_cluster_genomes.loc[g, species_sorted_sags['Bp']]
            hybrid_cluster_sag_ids = locus_clusters[locus_clusters == hybrid_cluster].index.values
            f_aln = f'{args.results_dir}alignments/core_ogs_cleaned/{g}_cleaned_aln.fna'
            aln = seq_utils.read_alignment_and_map_sag_ids(f_aln, pangenome_map)
            aln_hybrids = align_utils.get_subsample_alignment(aln, hybrid_cluster_sag_ids)
            pN, pS = seq_utils.calculate_pairwise_pNpS(aln_hybrids)
            pS_values = utils.get_matrix_triangle_values(pS.values, k=1)
            #y = np.mean(pS_values) + 1E-5
            y = np.mean(pS_values)
            x = np.array([x0, x0 + dx1 / 2 + dx2]) + rng.uniform(-dx1 / 2, dx1 / 2, size=2)
            #ax.scatter(x[0], y, 16, marker='o', color=colors_dict[hybrid_cluster])
            ax.scatter(x[0], y, ms, marker='o', fc=colors_dict[hybrid_cluster], ec='w', lw=lw, alpha=0.6)
            if (g in syna_num_site_alleles.index.values) and (hybrid_cluster == 'A'):
                #print(g, np.mean(pS_values), syna_num_site_alleles.loc[g, 'piS'], len(aln_hybrids))
                yc = syna_num_site_alleles.loc[g, 'piS']
                #ax.scatter(x[1], yc, 16, marker='s', color=colors_dict[hybrid_cluster])
                ax.scatter(x[1], yc, ms, marker='s', fc=colors_dict[hybrid_cluster], ec='w', lw=lw, alpha=0.6)
                ax.plot(x, [y, yc], c='k', lw=0.75, alpha=0.5)
            else:
                #print(g, np.mean(pS_values), len(aln_hybrids))
                pass
            counter[0] += 1
            if y < 1E-4:
                counter[1] += 1

        x0 += 1
        print('\n')

    print(counter)
    #ax.set_yticks([1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1])

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_hybrid_gene_diversity.pdf')
    plt.close()

    return fig_count + 1


def plot_alpha_spring_low_diversity(pangenome_map, metadata, og_ids, rng, args, fig_count, label_fs=14, num_bins=50, legend_fs=10, ms=4):
    divergence_files = [f'{args.output_dir}sscs_orthogroups_{j}_cleaned_divergence_matrices.dat' for j in range(10)]

    pangenome_map.read_pairwise_divergence_results(divergence_files)

    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')
    syna_sag_ids = np.array(species_sorted_sag_ids['A'])

    pdist_matrix = pangenome_map.construct_pairwise_divergence_across_ogs(og_ids, syna_sag_ids)
    syna_sorted_sag_ids = metadata.sort_sags(syna_sag_ids, by='location')

    # Get mean pdist
    pdist_mean = pd.DataFrame(index=syna_sag_ids, columns=syna_sag_ids)
    for i, s1 in enumerate(syna_sag_ids):
        pdist_mean.loc[s1, s1] = 0
        for j in range(i):
            s2 = syna_sag_ids[j]
            pdist_mean.loc[s1, s2] = np.nanmean(pdist_matrix[i, j, :])
            pdist_mean.loc[s2, s1] = np.nanmean(pdist_matrix[j, i, :])

    # Reverse cumulative distribution
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('mean pair divergence, $\pi_{ij}$', fontsize=label_fs)
    ax.set_xscale('log')
    ax.set_ylabel('reverse cumulative', fontsize=label_fs)
    #ax.set_yscale('log')

    xlim = (1E-4, 2E-1)
    pdist_values = utils.get_matrix_triangle_values(pdist_mean.values, k=1)
    x_bins = np.geomspace(*xlim, num_bins)
    #y = np.array([np.sum(pdist_values > x) / len(pdist_values) for x in x_bins])
    #y = np.array([np.sum(pdist_values <= x) for x in x_bins])
    #ax.plot(x_bins, y, f'-o', lw=1, ms=3, color='tab:gray', alpha=0.5, mfc='none', label=r'all $\alpha$')

    spring_colors = {'OS':'tab:cyan', 'MS':'tab:purple'}
    for spring in syna_sorted_sag_ids:
        spring_sag_ids = syna_sorted_sag_ids[spring]
        pdist_values = utils.get_matrix_triangle_values(pdist_mean.loc[spring_sag_ids, spring_sag_ids].values, k=1)
        y = np.array([np.sum(pdist_values > x) / len(pdist_values) for x in x_bins])
        n = len(syna_sorted_sag_ids[spring])
        ax.plot(x_bins, y, f'-s', lw=1, ms=ms, color=spring_colors[spring], alpha=0.5, mfc='none', label=f'{spring} (n={n:d})')

        if spring == 'OS':
            subsampled_sag_ids = rng.choice(spring_sag_ids, size=len(syna_sorted_sag_ids['MS']))
            pdist_values = utils.get_matrix_triangle_values(pdist_mean.loc[subsampled_sag_ids, subsampled_sag_ids].values, k=1)
            y = np.array([np.sum(pdist_values > x) / len(pdist_values) for x in x_bins])
            ax.plot(x_bins, y, f'-^', lw=1, ms=ms, color=spring_colors[spring], alpha=0.5, mfc='none', label=f'{spring} subsampled')

    # Between springs comparison
    pdist_values = pdist_mean.loc[syna_sorted_sag_ids['OS'],syna_sorted_sag_ids['MS']].values.flatten()
    y = np.array([np.sum(pdist_values > x) / len(pdist_values) for x in x_bins])
    ax.plot(x_bins, y, f'-D', lw=1, ms=3, color='tab:red', alpha=0.5, mfc='none', label='OS vs MS')

    ax.legend(frameon=False, fontsize=legend_fs)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_alpha_low_diversity_location_rev_cumul.pdf')
    plt.close()

    return fig_count + 1


def plot_gamma_alignment_results(pangenome_map, metadata, args, fig_count, num_bins=50):
    alignment_files = glob.glob(f'{args.results_dir}blast_alignment/Ga*_gamma_blast.tab')
    x_bins = np.linspace(0, 0.5, num_bins)

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'divergence from $\gamma$')
    ax.set_ylabel('histogram')

    for f_in in alignment_files:
        blast_results = seq_utils.read_blast_results(f_in)
        query_ids = blast_results['qseqid'].unique()

        gene_divergences = []
        for q in query_ids:
            sorted_hits = blast_results.loc[blast_results['qseqid'] == q, :].sort_values('bitscore', ascending=False)
            d = 1 - sorted_hits.loc[:, 'pident'].values[0] / 100
            gene_divergences.append(d)
        hist, x_bins = np.histogram(gene_divergences, bins=x_bins, density=True)
        x = [np.mean(x_bins[i:i + 2]) for i in range(num_bins - 1)]
        #ax.hist(gene_divergences, bins=x_bins, alpha=0.5, histtype='step', density=True)
        ax.plot(x, hist, alpha=0.5, lw=1, c='tab:gray')

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_gamma_blast.pdf')
    plt.close()

    return fig_count + 1


def plot_diversity_along_genome(pangenome_map, args, fig_count):
    metadata = MetadataMap()
    species_cluster_genomes = pd.read_csv(f'{args.results_dir}supplement/sscs_labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)
    #divergence_files = [f'{args.results_dir}supplement/sscs_orthogroup_{j}_divergence_matrices.dat' for j in range(10)]
    divergence_files = [f'{args.results_dir}supplement/sscs_orthogroups_{j}_cleaned_divergence_matrices.dat' for j in range(10)]
    pangenome_map.read_pairwise_divergence_results(divergence_files)

    fig = plt.figure(figsize=(double_col_width, 1.1 * single_col_width))
    for i, species in enumerate(['A', 'Bp']):
        ax = fig.add_subplot(2, 1, i + 1)
        plot_species_diversity_along_genome(ax, species_cluster_genomes, pangenome_map, metadata, species=species)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_core_gene_diversity.pdf')
    return fig_count + 1


def plot_species_diversity_along_genome(ax, species_cluster_genomes, pangenome_map, metadata, species='A', dx=2, w=5, min_og_presence=0.2, min_length=200):
    core_og_ids = pangenome_map.get_core_og_ids(metadata, min_og_frequency=min_og_presence, og_type='parent_og_id')
    syn_homolog_map = SynHomologMap(build_maps=True)

    if species == 'A':
        sorted_mapped_og_ids = np.array(species_cluster_genomes.loc[core_og_ids, :].sort_values('osa_location').index)
    elif species == 'Bp':
        sorted_mapped_og_ids = np.array(species_cluster_genomes.loc[core_og_ids, :].sort_values('osbp_location').index)

    # Filter short genes
    og_table = pangenome_map.og_table
    filtered_idx = []
    for og_id in sorted_mapped_og_ids:
        avg_length = og_table.loc[og_table['parent_og_id'] == og_id, 'avg_length'].mean()
        if avg_length > min_length + 100:
            filtered_idx.append(og_id)

    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')

    # Calculate divergence between species
    species_divergence_table = pd.DataFrame(index=filtered_idx, columns=['genome_position', 'species_diversity'])

    # Choose random high coverage SAGs for illustration
    sample_size = 10
    gene_presence_cutoff = 1000
    if species == 'A':
        species_divergence_table['genome_position'] = species_cluster_genomes.loc[filtered_idx, 'osa_location']
        xlabel = "OS-A genome position (Mb)"
        high_coverage_sag_ids = list(np.array(species_sorted_sag_ids['A'])[(og_table[species_sorted_sag_ids['A']].notna().sum(axis=0) > gene_presence_cutoff).values])
        mean_color = 'tab:orange'
        sample_color = open_colors['orange'][2]
    elif species == 'Bp':
        species_divergence_table['genome_position'] = species_cluster_genomes.loc[filtered_idx, 'osbp_location']
        xlabel = "OS-B' genome position (Mb)"
        high_coverage_sag_ids = list(np.array(species_sorted_sag_ids['Bp'])[(og_table[species_sorted_sag_ids['Bp']].notna().sum(axis=0) > gene_presence_cutoff).values])
        mean_color = 'tab:blue'
        sample_color = open_colors['blue'][2]

    sampled_sag_ids = np.random.choice(high_coverage_sag_ids, size=sample_size)

    mean_divergence, og_ids = pangenome_map.calculate_mean_pairwise_divergence(filtered_idx, species_sorted_sag_ids[species])
    species_divergence_table.loc[og_ids, 'species_diversity'] = mean_divergence
    species_divergence_table = species_divergence_table.loc[species_divergence_table['species_diversity'].notnull(), :]
    print(species, species_divergence_table)

    # Get divergences between A and B' pairs across sites
    dijk_dict = pangenome_map.get_sags_pairwise_divergences(sampled_sag_ids, input_og_ids=filtered_idx)
    pair_divergence_values = []
    x_pair_divergences = []
    for og_id in species_divergence_table.index:
        if og_id in dijk_dict:
            dij = dijk_dict[og_id]
            pair_divergence_values.append(utils.get_matrix_triangle_values(dij.astype(float), k=1))
        else:
            empty_array = np.empty(int(sample_size * (sample_size - 1) / 2))
            empty_array[:] = np.nan
            pair_divergence_values.append(empty_array.astype(float))
        x_pair_divergences.append(species_divergence_table.loc[og_id, 'genome_position'])
    pair_divergence_values = np.array(pair_divergence_values)
    x_pair_divergences = np.array(x_pair_divergences)


    # Plot results
    ylabel = r'nucleotide diversity, $\pi$'

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_ylim(0, 0.15)
    ax.set_ylim(1E-3, 3E-1)
    ax.set_yscale('log')

    for i, pair_divergences in enumerate(pair_divergence_values.T):
        y_smooth = np.array([np.nanmean(pair_divergences[j:j + w]) if np.sum(np.isfinite(pair_divergences[j:j + w])) > 0 else np.nan for j in range(0, len(pair_divergences) - w, dx)])
        x_smooth = np.array([np.mean(x_pair_divergences[j:j + w]) for j in range(0, len(species_divergence_table) - w, dx)])
        ax.plot(x_smooth, y_smooth, lw=0.25, c=sample_color, alpha=0.4)

    y_smooth = np.array([np.mean(species_divergence_table['species_diversity'].values[j:j + w]) for j in range(0, len(species_divergence_table) - w, dx)])
    x_smooth = np.array([np.mean(species_divergence_table['genome_position'].values[j:j + w]) for j in range(0, len(species_divergence_table) - w, dx)])
    ax.plot(x_smooth, y_smooth, c=mean_color, lw=1.5)


def plot_pairwise_divergences(pangenome_map, metadata, args, fig_count):
    og_table = pangenome_map.og_table
    color_dict = {'A':'tab:orange', 'Bp':'tab:blue', 'C':'tab:green'}
    label_dict = {'A':r'$\alpha$', 'Bp':r'$\beta$', 'C':r'$\gamma$'}

    pdist_values = {'A':[], 'Bp':[]}
    for og_id in og_table['parent_og_id'].unique():
        f_aln = f'{args.results_dir}alignments/v2/core_ogs_cleaned/{og_id}_cleaned_aln.fna'
        aln = seq_utils.read_alignment(f_aln)
        species_grouping = align_utils.sort_aln_rec_ids(aln, pangenome_map, metadata)
        pdist = pickle.load(open(f'{args.pangenome_dir}pdist/{og_id}_cleaned_pS.dat', 'rb'))
        print(og_id)
        for species in ['A', 'Bp']:
            if species in species_grouping:
                species_idx = species_grouping[species]
                pdist_values[species].append(utils.get_matrix_triangle_values(pdist.loc[species_idx, species_idx].values, k=1))

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'synonymous divergence, $d_S$')
    ax.set_ylabel('histogram')
    ax.set_yscale('log')

    for species in ['A', 'Bp']:
        d_values = np.concatenate(pdist_values[species])
        ax.hist(d_values, bins=100, lw=1.5, histtype='step', color=color_dict[species], label=label_dict[species])
    ax.legend(fontsize=14, frameon=False)

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_pdist_histogram.pdf')
    plt.close()
    fig_count += 1

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'synonymous divergence, $d_S$')
    ax.set_ylabel('cumulative')

    for species in ['A', 'Bp']:
        d_values = np.concatenate(pdist_values[species])
        ax.hist(d_values, bins=100, lw=1.5, histtype='step', density=True, cumulative=True, color=color_dict[species], label=label_dict[species])
        print(species, len(d_values), np.sum(d_values > 0.01), np.sum(d_values > 0.1))
    ax.axhline(0.99, lw=1.0, ls='--', color='k')
    ax.axhline(0.9, lw=1.0, ls='--', color='k')
    ax.legend(fontsize=14, frameon=False)

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_pdist_cumulative.pdf')
    plt.close()

    return fig_count + 1



###########################################################
# Linkage disequilibrium analysis
###########################################################


def make_revised_linkage_figures(pangenome_map, args, fig_count, avg_length_fraction=0.75, ms=5, low_diversity_cutoff=0.05, ax_label_size=12, tick_size=12):
    color_dict = {'A':'tab:orange', 'Bp':'tab:blue', 'Bp_subsampled':'gray', 'population':'k'}
    label_dict = {'A':r'$\alpha$', 'Bp':r'$\beta$', 'Bp_subsampled':r'$\beta$ (subsampled)', 'population':r'whole population'}
    cloud_dict = {'A':0.05, 'Bp':0.05, 'Bp_subsampled':0.05, 'population':0.1}
    marker_dict = {'A':'o', 'Bp':'s', 'Bp_subsampled':'x', 'population':'D'}

    rng = np.random.default_rng(args.random_seed)
    random_gene_linkage = calculate_random_gene_linkage(args, rng, cloud_dict)

    metadata = MetadataMap()

    # Comparison to neutral model
    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax1 = fig.add_subplot(121)
    set_up_linkage_curve_axis(ax1, ax_label='A', xlim=(0.8, 7E3), xticks=[1E0, 1E1, 1E2, 1E3], ylim=(5E-3, 1.5E0), ylabel='linkage disequilibrium', x_ax_label=1.5E-1)
    ax2 = fig.add_subplot(122)
    set_up_linkage_curve_axis(ax2, ax_label='B', xlim=(0.02, 3E2), xticks=[1E-1, 1E0, 1E1, 1E2], ylim=(1E-2, 1.5E0), xlabel=r'rescaled separation, $\rho x$', ylabel='linkage disequilibrium', x_ax_label=3E-3)
    rho_fit = {'A':0.03, 'Bp':0.12}
    theta = 0.03
    lmax = 2000
    x_theory = np.arange(1, lmax)


    #ms = 5
    for species in ['A', 'Bp']:
        cloud_radius = cloud_dict[species]
        #linkage_results = pickle.load(open(f'{args.output_dir}sscs_core_ogs_cleaned_{species}_linkage_curves_c{cloud_radius}.dat', 'rb'))
        #sigmad2 = plt_linkage.average_sigmad_sq(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
        #sigmad2_cg, x_cg = plt_linkage.coarse_grain_linkage_array(sigmad2)
        linkage_results = pickle.load(open(f'{args.output_dir}sscs_core_ogs_cleaned_{species}_linkage_curves_c{cloud_radius}_all_sites.dat', 'rb'))
        x_arr, sigmad2 = average_linkage_curves(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
        x_cg, sigmad2_cg = coarse_grain_distances(x_arr, sigmad2)
        y0 = sigmad2_cg[1]
        ax1.plot(x_cg[:-5], sigmad2_cg[:-5], f'-{marker_dict[species]}', ms=ms, mfc='none', mew=1.5, lw=0, alpha=1.0, c=color_dict[species], label=label_dict[species])

        # Plot theory
        rho = rho_fit[species]
        y_theory = er2.sigma2_theory(rho * x_theory, theta)
        ax1.plot(x_theory, (y0 / y_theory[0]) * y_theory, lw=1.5, ls='--', c=color_dict[species], label=f'fit ($\\rho={rho}$)')

        ax2.plot(rho * x_cg[:-5], sigmad2_cg[:-5], f'-{marker_dict[species]}', ms=ms, mfc='none', mew=1.5, lw=1.0, alpha=1.0, c=color_dict[species], label=label_dict[species])
        #ax2.scatter(rho * x_cg[:-5], sigmad2_cg[:-5], marker=marker_dict[species], s=ms**2, fc='none', ec=color_dict[species], lw=1, alpha=1.0, label=label_dict[species])
    x_theory = np.geomspace(0.01, 200, 100)
    y_theory = er2.sigma2_theory(x_theory, theta)
    ax2.plot(x_theory, y_theory, lw=1.5, ls='-', c='k', label=f'neutral theory')

    ax1.legend(fontsize=10, frameon=False, loc='lower left')
    ax2.legend(fontsize=10, frameon=False, loc='lower left')
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_linkage_collapse.pdf')
    fig_count += 1



def calculate_random_gene_linkage(args, rng, cloud_dict, sites_ext='_all_sites', min_sample_size=20, min_coverage=0.9, sample_size=1000):
    random_gene_linkage = {}
    for species in ['A', 'Bp', 'Bp_subsampled', 'population']:
        c = cloud_dict[species]
        if species != 'population':
            gene_pair_results = pickle.load(open(f'{args.output_dir}sscs_core_ogs_cleaned_{species}_gene_pair_linkage_c{c}{sites_ext}.dat', 'rb'))
        else:
            gene_pair_results = pickle.load(open(f'{args.output_dir}sscs_core_ogs_cleaned_{species}_gene_pair_linkage{sites_ext}.dat', 'rb'))

        gene_pair_linkage = gene_pair_results['sigmad_sq']
        sample_sizes = gene_pair_results['sample_sizes']

        # Draw random gene pairs
        g1_idx, g2_idx = np.where((sample_sizes.values >= min_sample_size) & (gene_pair_linkage.notna().values))
        og_ids = sample_sizes.index.values
        random_sample_idx = rng.choice(len(g1_idx), size=sample_size, replace=False)
        gene_pair_array = np.array([og_ids[g1_idx[random_sample_idx]], og_ids[g2_idx[random_sample_idx]]]).T
        random_gene_linkage[species] = (np.mean(gene_pair_linkage.values[g1_idx, g2_idx]), gene_pair_array)

    return random_gene_linkage

def set_up_linkage_curve_axis(ax, xlim=(8E-1, 1E4), ylim=(5E-3, 1.5E0), linkage_metric='$\sigma_d^2$', ax_label='', ax_label_fs=14, xticks=[1, 1E1, 1E2, 1E3, 1E4], xlabel=r'separation, $x$', ylabel='linkage', x_ax_label=1E-1, yticks=[1E-2, 1E-1, 1]):
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    #ax.set_ylabel(f'linkage disequilibrium, {linkage_metric}', fontsize=14)
    if ylabel is not None:
        ax.set_ylabel(f'{ylabel}, {linkage_metric}', fontsize=14)
    ax.set_yscale('log')
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.text(x_ax_label, 1.05 * ylim[1], ax_label, fontweight='bold', fontsize=ax_label_fs)


def average_linkage_curves(linkage_results, metric='sigmad_sq', min_sample_size=20, average_length_fraction=1, x_max=5000, min_depth=1000):
    Dsq_list = []
    denom_list = []
    x_arr = np.arange(x_max, dtype=int)

    if metric == 'sigmad_sq':
        avg = np.zeros((2, x_max))
        total_depth = np.zeros((2, x_max))
        for og_id in linkage_results:
            Dsq_tuple, denom_tuple, rsq_tuple, sample_size = linkage_results[og_id]

            if sample_size >= min_sample_size:
                x_Dsq, Dsq, depth_Dsq = Dsq_tuple
                #length_cutoff_idx = int(average_length_fraction * x_Dsq[-1]) + 1
                length_cutoff_idx = np.argmin(np.abs(x_Dsq - (average_length_fraction * x_Dsq[-1]))) + 1
                filtered_idx = depth_Dsq[:length_cutoff_idx] > 0
                #avg[0, x_Dsq[:length_cutoff_idx][depth_Dsq[:length_cutoff_idx] > 0]] += Dsq[:length_cutoff_idx][depth_Dsq[:length_cutoff_idx] > 0]
                #total_depth[0, x_Dsq[:length_cutoff_idx][depth_Dsq[:length_cutoff_idx] > 0]] += depth_Dsq[:length_cutoff_idx][depth_Dsq[:length_cutoff_idx] > 0]
                avg[0, x_Dsq[:length_cutoff_idx][filtered_idx]] += Dsq[:length_cutoff_idx][filtered_idx] * depth_Dsq[:length_cutoff_idx][filtered_idx]
                total_depth[0, x_Dsq[:length_cutoff_idx][filtered_idx]] += depth_Dsq[:length_cutoff_idx][filtered_idx]

                x_denom, denom, depth_denom = denom_tuple
                #avg[1, x_denom[:length_cutoff_idx][depth_denom[:length_cutoff_idx] > 0]] += denom[:length_cutoff_idx][depth_denom[:length_cutoff_idx] > 0]
                #total_depth[1, x_denom[:length_cutoff_idx][depth_denom[:length_cutoff_idx] > 0]] += depth_denom[:length_cutoff_idx][depth_denom[:length_cutoff_idx] > 0]
                avg[1, x_denom[:length_cutoff_idx][filtered_idx]] += denom[:length_cutoff_idx][filtered_idx] * depth_denom[:length_cutoff_idx][filtered_idx]
                total_depth[1, x_denom[:length_cutoff_idx][filtered_idx]] += depth_denom[:length_cutoff_idx][filtered_idx]

        idx = (total_depth[0, :] >= min_depth) & (total_depth[1, :] >= min_depth)
        avg[:, idx] /= total_depth[:, idx]
        linkage = avg[0, idx] / (avg[1, idx] + (avg[1, idx] <= 0.))

    elif metric == 'r_sq':
        avg = np.zeros(x_max)
        total_depth = np.zeros(x_max)
        for og_id in linkage_results:
            Dsq_tuple, denom_tuple, rsq_tuple, sample_size = linkage_results[og_id]

            if sample_size >= min_sample_size:
                x_rsq, rsq, depth_rsq = rsq_tuple
                #length_cutoff_idx = int(average_length_fraction * x_rsq[-1]) + 1
                length_cutoff_idx = np.argmin(np.abs(x_rsq - (average_length_fraction * x_rsq[-1]))) + 1
                filtered_idx = depth_rsq[:length_cutoff_idx] > 0
                #avg[x_rsq[:length_cutoff_idx][depth_rsq[:length_cutoff_idx] > 0]] += rsq[:length_cutoff_idx][depth_rsq[:length_cutoff_idx] > 0]
                #total_depth[x_rsq[:length_cutoff_idx][depth_rsq[:length_cutoff_idx] > 0]] += depth_rsq[:length_cutoff_idx][depth_rsq[:length_cutoff_idx] > 0]
                avg[x_rsq[:length_cutoff_idx][filtered_idx]] += rsq[:length_cutoff_idx][filtered_idx] * depth_rsq[:length_cutoff_idx][filtered_idx]
                total_depth[x_rsq[:length_cutoff_idx][filtered_idx]] += depth_rsq[:length_cutoff_idx][filtered_idx]
        idx = total_depth >= min_depth
        linkage = avg[idx] / total_depth[idx]

    x_out = x_arr[idx]
    output = (x_out, linkage)
    return output


def coarse_grain_distances(x, y, num_cg_points=20):
    x_max = x[-1] + 1
    x_bin = (np.log10(2 * x_max) - np.log10(11)) / num_cg_points
    x_log = np.geomspace(11, 2 * x_max, num_cg_points)
    y_cg = np.zeros(len(x_log) + 11)
    y_cg[x[x <= 10]] = y[x[x <= 10]]
    for i, xi in enumerate(x_log):
        idx = i + 11
        jl = np.argmin(np.abs(x - np.floor(xi)))
        jr = np.argmin(np.abs(x - np.ceil(10**(np.log10(xi) + x_bin))))
        y_cg[idx] = np.mean(y[jl:jr])
    x_cg = np.concatenate([np.arange(11), x_log])
    return x_cg, y_cg

if __name__ == '__main__':
    # Default variables
    figures_dir = '../figures/supplement/'
    pangenome_dir = '../results/single-cell/sscs_pangenome_v2/'
    results_dir = '../results/single-cell/'
    output_dir = '../results/single-cell/supplement/'
    metagenome_dir = '../data/metagenome/recruitment_v4/'
    annotations_dir = '../data/single-cell/filtered_annotations/sscs/'
    #f_orthogroup_table = f'{pangenome_dir}filtered_orthogroups/sscs_annotated_single_copy_orthogroup_presence.tsv'
    f_orthogroup_table = f'{pangenome_dir}filtered_low_copy_clustered_core_mapped_labeled_cleaned_orthogroup_table.tsv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--figures_dir', default=figures_dir, help='Directory where figures are saved.')
    parser.add_argument('-M', '--metagenome_dir', default=metagenome_dir, help='Directory with results for metagenome.')
    parser.add_argument('-N', '--annotations_dir', default=annotations_dir, help='Directory with annotation files.')
    parser.add_argument('-P', '--pangenome_dir', default=pangenome_dir, help='Pangenome directory.')
    parser.add_argument('-R', '--results_dir', default=results_dir, help='Main results directory.')
    parser.add_argument('-O', '--output_dir', default=output_dir, help='Directory in which supplemental data is saved.')
    parser.add_argument('-g', '--orthogroup_table', default=f_orthogroup_table, help='File with orthogroup table.')
    parser.add_argument('-r', '--random_seed', default=12345, type=int, help='Seed for RNG.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    args = parser.parse_args()


    pangenome_map = pg_utils.PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    make_og_diversity_figures(pangenome_map, args, 1)
    #fig_count = make_genetic_diversity_figures(pangenome_map, args, 22)
    #fig_count = make_revised_linkage_figures(pangenome_map, args, fig_count)
