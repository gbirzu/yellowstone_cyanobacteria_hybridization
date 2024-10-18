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
from syn_homolog_map import SynHomologMap
from metadata_map import MetadataMap
from analyze_metagenome_reads import strip_sample_id
from plot_utils import *

mpl.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


###########################################################
# Fig. 1 panels: Genome-level analysis
###########################################################
def make_genome_level_figure(pangenome_map, args, rng, fig_number='', ax_label_size=12, tick_size=12):
    print('Plotting Fig. 1...\n\n')

    metadata = MetadataMap()
    ncols = 2
    nrows = 1
    ratio = 0.5
    fig = plt.figure(figsize=(intermediate_col_width, 0.6 * intermediate_col_width))
    gspec = gridspec.GridSpec(figure=fig,
            ncols=ncols, nrows=nrows, width_ratios=[1, ratio], wspace=0)

    # Reference divergences
    ax = plt.subplot(gspec[0, 0])
    sag_ids = pangenome_map.get_sag_ids()
    mean_pident, pident_sag_ids, _ = calculate_reference_pident(alignment_dir, metric='mean')
    mean_pident_filtered = mean_pident[[s in sag_ids for s in pident_sag_ids]]
    plot_ref_divergence_scatter(ax, mean_pident_filtered, fig, annotate=True, distance='divergence', xlabel_stem='Mean OS-A', ylabel_stem="Mean OS-B'", label_size=ax_label_size, tick_size=tick_size, ax_label='A', xticks=[0, 0.05, 0.1, 0.15, 0.2], yticks=[0, 0.05, 0.1, 0.15, 0.2])

    ax = plt.subplot(gspec[0, 1])
    species_abundances = read_species_abundances(args, 100)
    species_abundances.loc[['A', 'Bp', 'C'], :] = species_abundances.loc[['A', 'Bp', 'C'], :] / species_abundances.loc['total_average_depth', :]
    species_relative_abundances = process_species_abundance_table(species_abundances)
    plot_species_frequency_across_samples(ax, species_relative_abundances, fig, ms=6, ax_label='B', label_size=ax_label_size, tick_size=tick_size)

    gspec.tight_layout(fig)
    plt.savefig(f'{args.figures_dir}fig{fig_number}_genome_level.pdf')


def calculate_reference_pident(alignment_dir, metric='median', d_species_cutoff=0.075, verbose=False):
    ref_sorted_sag_ids = {'A':[], 'Bp':[], 'C':[], 'ambiguous':[]}
    pident = []
    sag_ids = []
    blast_results_files = sorted(glob.glob(f'{alignment_dir}*_blast_results.tab'))
    for f_blast in blast_results_files:
        sag_id = f_blast.split('/')[-1].replace('_ref_blast_results.tab', '')
        blast_results = seq_utils.read_blast_results(f_blast)
        blast_best_hits = get_best_hits(blast_results)
        osa_hits = blast_results.loc[blast_results['sseqid'].str.contains('CYA'), :]
        osa_best_hits = get_best_hits(osa_hits)
        osbp_hits = blast_results.loc[blast_results['sseqid'].str.contains('CYB'), :]
        osbp_best_hits = get_best_hits(osbp_hits)
        if len(osa_best_hits) > 100 and len(osbp_best_hits) > 100:
            if metric == 'median':
                osa_pident = osa_best_hits['pident'].median() / 100.0
                osbp_pident = osbp_best_hits['pident'].median() / 100.0
            elif metric == 'mean':
                osa_pident = osa_best_hits['pident'].mean() / 100.0
                osbp_pident = osbp_best_hits['pident'].mean() / 100.0

            pident.append([osa_pident, osbp_pident])
            sag_ids.append(sag_id)

            if osa_pident < 0.99 and osbp_pident < 0.95 and verbose:
                print(sag_id, f'{1 - osa_pident:.4f}', f'{1 - osbp_pident:.4f}')

            if osa_pident > 1 - d_species_cutoff and osbp_pident < 1 - d_species_cutoff:
                ref_sorted_sag_ids['A'].append(sag_id)
            elif osa_pident < 1 - d_species_cutoff and osbp_pident > 1 - d_species_cutoff:
                ref_sorted_sag_ids['Bp'].append(sag_id)
            elif osa_pident < 1 - d_species_cutoff and osbp_pident < 1 - d_species_cutoff:
                ref_sorted_sag_ids['C'].append(sag_id)
            else:
                ref_sorted_sag_ids['ambiguous'].append(sag_id)

    pident = np.array(pident)
    sag_ids = np.array(sag_ids)
    return pident, sag_ids, ref_sorted_sag_ids

def get_best_hits(blast_results):
    return blast_results.sort_values('pident').drop_duplicates('qseqid', keep='last')

def plot_ref_divergence_scatter(ax, pident, fig, annotate=False, distance='divergence', 
        xlabel_stem='median OS-A', ylabel_stem="median OS-B'", xticks=[0, 0.1, 0.2], yticks=[0, 0.1, 0.2], ytick_position='left', 
        label_size=16, tick_size=14, clean_borders=True, ax_label=None, panel_label_fs=10):
    ax.set_xlabel(f'{xlabel_stem.split(" ")[0]} {distance} from {xlabel_stem.split(" ")[1]}', fontsize=label_size)
    ax.set_ylabel(f'{ylabel_stem.split(" ")[0]} {distance} from {ylabel_stem.split(" ")[1]}', fontsize=label_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)

    if ytick_position == 'right':
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
    if clean_borders:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


    # Make colors
    colors = []
    for p in pident:
        if p[0] > 0.95 and p[1] < 0.95:
            colors.append('tab:orange')
        elif p[0] < 0.95 and p[1] > 0.95:
            colors.append('tab:blue')
        elif p[0] < 0.95 and p[1] < 0.95:
            colors.append('tab:green')
        else:
            colors.append('gray')
    
    if distance == 'divergence':
        ax.set_xlim(-0.0, 0.2)
        ax.set_xticks(xticks)
        ax.set_ylim(-0.0, 0.2)
        ax.set_yticks(yticks)
        ax.scatter(1 - pident[:, 0], 1 - pident[:, 1], s=10, fc=colors, alpha=0.6)

        if annotate:
            ax.text(0.04, 0.16, r'$\boldsymbol{\alpha}$', ha='center', va='center', fontsize=20, fontweight='bold', c='tab:orange')
            ax.text(0.17, 0.02, r'$\boldsymbol{\beta}$', ha='center', va='center', fontsize=20, fontweight='bold', c='tab:blue')
            ax.text(0.15, 0.15, r'$\boldsymbol{\gamma}$', ha='center', va='center', fontsize=20, fontweight='bold', c='tab:green')

    else:
        ax.set_xlim(0.8, 1.0)
        ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1])
        ax.set_ylim(0.8, 1.0)
        ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1])
        ax.scatter(pident[:, 0], pident[:, 1], s=10, fc=colors, alpha=0.6)
        
        if annotate:
            ax.text(0.99, 0.84, r'$\boldsymbol{\alpha}$', ha='center', va='center', fontsize=20, fontweight='bold', c='tab:orange')
            ax.text(0.83, 0.98, r'$\boldsymbol{\beta}$', ha='center', va='center', fontsize=20, fontweight='bold', c='tab:blue')
            ax.text(0.86, 0.84, r'$\boldsymbol{\gamma}$', ha='center', va='center', fontsize=20, fontweight='bold', c='tab:green')

    if ax_label is not None:
        trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
        ax.text(-0.02, 1.02, ax_label, transform=ax.transAxes + trans, fontsize=panel_label_fs, fontweight='bold', va='bottom', usetex=False)



def read_species_abundances(args, num_loci, ext='_pident95.xlsx'):
    # Get most abundant loci
    for i, species in enumerate(['A', 'B', 'C']):
        f_recruitment = f'{args.metagenome_dir}{species}_Allele{ext}'
        alleles_df = pd.read_excel(f_recruitment, sheet_name=0, index_col=0)
        sample_ids = np.array([c for c in alleles_df.columns if 'Hot' in c])
        alleles_df['average_depth'] = [np.nan, np.nan] + [np.nanmean(row.astype(float)) for row in alleles_df[sample_ids].values[2:]]
        alleles_df = alleles_df.sort_values('average_depth', ascending=False)
        loci_depth_df = alleles_df.loc[alleles_df.index[:-2], np.concatenate([sample_ids, ['average_depth']])]

        if i == 0:
            species_abundances = pd.DataFrame(index=['A', 'Bp', 'C'], columns=sample_ids)

        if i == 1:
            species = 'Bp' # change species label

        species_abundances.loc[species, sample_ids] = loci_depth_df.loc[loci_depth_df.index[:num_loci], sample_ids].mean(axis=0) 

    # Remove _FD from sample IDs
    renamed_sample_ids = np.array([s.strip('_FD')for s in sample_ids])
    species_abundances.columns = renamed_sample_ids
    species_abundances.loc['total_average_depth', :] = species_abundances.loc[['A', 'Bp', 'C'], :].sum(axis=0)
    species_abundances.loc['loci_average_depth', :] = species_abundances.loc[['A', 'Bp', 'C'], :].mean(axis=0)
    species_abundances.loc['min_average_depth', :] = species_abundances.loc[['A', 'Bp', 'C'], :].min(axis=0)

    return species_abundances


def process_species_abundance_table(abundance_df):
    abundance_df = abundance_df.rename({'Bp':'B'}, axis='rows')

    # Automatic sample relabeling
    column_dict = {}
    for c in abundance_df.columns:
        column_dict[c] = strip_sample_id(c)
    abundance_df = abundance_df.rename(column_dict, axis='columns')

    # Manual sample relabeling
    column_dict = {'t1':'OSt1', 'm2':'OSm2', 'P4':'MSP4', 'T8':'MST8', 'T9':'MST9', 'R4cd':'MSR4cd', 'lt10cd':'MSt10cd', 'ottomLayer':'OSBottom', 'ottomLayer_2':'OSBottom2'}
    for c in abundance_df.columns:
        if '2Sample' in c:
            column_dict[c] = c.replace('2Sample', 'MS2S')
    abundance_df = abundance_df.rename(column_dict, axis='columns')
    abundance_df = abundance_df[np.sort(abundance_df.columns.values)]

    # Add SAG sample markers
    abundance_df = abundance_df.rename({'MSe4':'*MSe4', 'OSM3':'*OSM3'}, axis='columns')

    return abundance_df


def plot_species_frequency_across_samples(ax, sample_relative_abundances, fig, lw=0.8, tick_size=12, label_size=14, legend_size=10, ms=4, sag_samples=['HotsprSampleMSe4', 'HotsprSampleOSM3'], clean_borders=True, ax_label=None, yticks=[1E-4, 1E-3, 1E-2, 1E-1, 1], panel_label_fs=10):
    metagenome_dir = '../data/metagenome/recruitment_v3/'
    sample_columns = sample_relative_abundances.columns.values
    
    # Choose MS temperature series
    sample_columns = ['MS50', 'MS55', 'MS60', 'MS65']
    plot_abundances = sample_relative_abundances[sample_columns]

    num_loci = 100
    markers = ['o', 's', 'D']
    colors = ['tab:orange', 'tab:blue', 'tab:green']


    x = np.array([50, 55, 60, 65])
    for i, species in enumerate(['A', 'B', 'C']):
        y = plot_abundances.loc[species, :].values
        ax.plot(x, y, f'-{markers[i]}', c=colors[i], label=species)

    ax.set_xlim(48, 67)
    ax.set_xticks(x)
    ax.set_xlabel('Temperature ($^\circ$C)', fontsize=label_size)
    ax.set_ylim(1E-4, 1.5)
    ax.set_yticks(yticks)
    ax.set_yscale('log')
    ax.set_ylabel('Relative abundance', fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.plot(x, np.ones(len(x)) / 48, ls='--', c=open_colors['gray'][8])
    y_min = 1. / plot_abundances.loc['total_average_depth', :].values
    ax.plot(x, y_min, ls='--', c=open_colors['gray'][5])

    ax.fill_between([55, 60], 1, alpha=0.5, lw=0, color='tab:gray', zorder=3)
    ax.annotate('SAG samples', xy=(57.5, 1.2), xycoords='data', xytext=(57.5, 3), textcoords='data',
            ha='center', va='bottom', fontsize=10,
            arrowprops=dict(arrowstyle='-['))

    if clean_borders:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    if ax_label is not None:
        trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
        ax.text(-0.02, 1.02, ax_label, transform=ax.transAxes + trans, fontsize=panel_label_fs, fontweight='bold', va='bottom', usetex=False)


###########################################################
# Fig. 2 panels: Gene-level analysis
###########################################################

def make_gene_level_figure(pangenome_map, args, rng, fig_number=''):
    print('Plotting Fig. 2 panels...\n\n')

    hybridization_dir = f'{args.results_dir}hybridization/'
    species_cluster_genomes = pd.read_csv(f'{hybridization_dir}sscs_labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)
    metadata = MetadataMap()

    syna_hybrid_donor_frequency_table = make_donor_frequency_table(species_cluster_genomes, 'A', pangenome_map, metadata)
    synbp_hybrid_donor_frequency_table = make_donor_frequency_table(species_cluster_genomes, 'Bp', pangenome_map, metadata)
    ncols = 2
    nrows = 3
    ratio = 4
    ratio2 = 1.5
    fig = plt.figure(figsize=(double_col_width, 1.0 * single_col_width))
    gspec = gridspec.GridSpec(figure=fig,
            ncols=ncols, nrows=nrows, 
            height_ratios=[ratio, ratio, ratio2], hspace=0.4,
            top=0.95, bottom=0.15)

    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    

    inner_grid = gspec[:2, :].subgridspec(ncols=2, nrows=2, wspace=0, width_ratios=[1, 69], hspace=0.65)
    axs_merged = plt.subplot(inner_grid[:, 0])
    for side in ['left', 'top', 'right', 'bottom']:
        axs_merged.spines[side].set_color('none')
    axs_merged.set_xticks([])
    axs_merged.set_yticks([])
    axs_merged.set_ylabel(r'Hybrid gene frequency', fontsize=14, labelpad=18)

    ax = plt.subplot(inner_grid[0, 1])
    ax.text(0.0, 0.95, 'A', transform=ax.transAxes + trans, fontsize=10, fontweight='bold', va='center', usetex=False)
    plot_hybrid_gene_frequencies(ax, syna_hybrid_donor_frequency_table, 'A', xlabel='OS-A genome position (Mb)', lw=0.75, ms=5, ylabel='', legend_loc='upper left', legend_size=8, tick_size=10)
    ax = plt.subplot(inner_grid[1, 1])
    ax.text(0.0, 0.95, 'B', transform=ax.transAxes + trans, fontsize=10, fontweight='bold', va='center', usetex=False)
    plot_hybrid_gene_frequencies(ax, synbp_hybrid_donor_frequency_table, 'Bp', xlabel="", lw=0.75, ms=5, ylabel='', legend_loc='upper left', legend_size=8, tick_size=10)


    # Read divergence table
    divergence_files = [f'{args.pangenome_dir}_aln_results/sscs_orthogroups_{j}_pS_matrices.dat' for j in range(10)]
    pangenome_map.read_pairwise_divergence_results(divergence_files)
    f_species_divergence = f'{args.data_dir}species_divergence_table.tsv'
    if os.path.exists(f_species_divergence):
        species_divergence_table = pd.read_csv(f_species_divergence, sep='\t', index_col=0)
    else:
        species_divergence_table = calculate_species_divergence_along_genome(species_cluster_genomes, pangenome_map, metadata)
        species_divergence_table.to_csv(f_species_divergence, sep='\t')
    species_divergence_table = species_divergence_table.loc[species_divergence_table['mean_divergence'].notnull(), :]

    example_ogs = ['YSG_0928', 'YSG_0703']
    aln_segments = {'YSG_0928':(50, 251),'YSG_0703':(250, 451)}
    ax_gt = plt.subplot(gspec[2, :])
    ax_gt.text(-0.03, 0.8, 'C', transform=ax_gt.transAxes + trans, fontsize=10, fontweight='bold', va='bottom', usetex=False)
    plot_genomic_trenches_panel(ax_gt, species_divergence_table, species_cluster_genomes, pangenome_map, metadata, rng, pos_column='osbp_location', ylim=(0, 1.0), yticks=[0, 0.5, 1], d_low=0.1, random_sample=False, highlight=example_ogs, xlabel_size=14, tick_size=10)
    plt.savefig(f'{args.figures_dir}fig{fig_number}A-C.pdf', dpi=1000)


    for o in example_ogs:
        plot_species_divergence_blowup(o, species_divergence_table, savefig=f'{args.figures_dir}fig{fig_number}C_{o}.pdf', ylabel='', yticks=[0, 0.5, 1.], num_xticks=4, tick_size=10)


    subsample_size = 40
    axes_labels = ['D', 'E']
    for i, og_id in enumerate(example_ogs):
        # Make new figures for alignments
        fig = plt.figure(figsize=(single_col_width, 0.55 * single_col_width))
        ax = fig.add_subplot(111)

        # Plot alignment
        f_aln = f'{args.results_dir}alignments/core_ogs_main_cloud/{og_id}_main_cloud_aln.fna'
        aln = seq_utils.read_alignment(f_aln)
        species_grouping = align_utils.sort_aln_rec_ids(aln, pangenome_map, metadata)

        # Subsample alignments
        temp = []
        for species  in ['A', 'Bp', 'C']:
            if species in species_grouping:
                n_species = int(np.ceil(subsample_size * len(species_grouping[species]) / len(aln)))
                temp.append(rng.choice(species_grouping[species], size=n_species, replace=False))
        subsampled_ids = np.concatenate(temp)
        aln_plot = align_utils.get_subsample_alignment(aln, subsampled_ids)
        species_grouping = align_utils.sort_aln_rec_ids(aln_plot, pangenome_map, metadata)

        xi, xf = aln_segments[og_id]
        plot_alignment(aln_plot[:, xi:xf], annotation=species_grouping, annotation_style='lines', marker_size=4, reference='closest_to_consensus', fig_dpi=1000, ax=ax)

        # Add labels
        xticklabels = np.arange(xi, xf, 50)
        xticks = np.arange(0, xf - xi + 1, 50)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'${x}$' for x in xticklabels])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(f'{args.figures_dir}fig{fig_number}{axes_labels[i]}.pdf', dpi=1000)


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


def plot_hybrid_gene_frequencies(ax, hybrid_freq_table, species, xlabel='', ylabel='hybrid gene frequency', transform_counts=True, y_ref=0, lw=1.0, label_size=14, tick_size=12, legend_size=9, ms=3, legend_loc='best', min_msog_fraction=0.5):

    # Define variables
    species_list = ['A', 'Bp', 'C', 'O']
    donor_list = [s for s in species_list if s != species]
    color_dict = {'A':'#E69F00', 'Bp':'#56B4E9', 'C':'#009E73', 'O':'#CC79A7', 'M':'#D55E00'}
    donor_label_dict = {'A':r'$\alpha$', 'Bp':r'$\beta$', 'C':r'$\gamma$', 'O':'X'}
    if species == 'A':
        x_column = 'osa_location'
    elif species == 'Bp':
        x_column = 'osbp_location'

    # Set up axis
    if species == 'A':
        ax.set_xlim(0, 1E-6 * utils.osa_genome_size)
    elif species == 'Bp':
        ax.set_xlim(0, 1E-6 * utils.osbp_genome_size)
    ax.set_xlabel(xlabel, fontsize=label_size)
    if ylabel != '':
        ax.set_ylabel(ylabel, fontsize=label_size)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['$0$', '$1$', '$2$','$5$','$15$'], fontsize=tick_size)
    if transform_counts:
        ymax = height_transform(hybrid_freq_table[donor_list].sum(axis=1).max()) + 2
    else:
        ymax = hybrid_freq_table[donor_list].sum(axis=1).max() + 2
    ax.set_ylim(0, ymax)

    ax.tick_params(labelsize=tick_size)

    single_donor_idx = hybrid_freq_table.loc[(hybrid_freq_table[donor_list] > 0).sum(axis=1) == 1, :].index

    offset = 0.01
    legend_handles = {}
    total_hybrids = 0
    for s in donor_list:
        x, y = hybrid_freq_table.loc[single_donor_idx, [x_column, s]].values.T
        x = x[y > 0]
        y = y[y > 0]
        if transform_counts:
            y = np.array([height_transform(yi) for yi in y])
        total_hybrids += len(y)

        ps = ax.scatter(x[0], y[0] + offset, marker='o', s=ms**2, fc=color_dict[s], ec='white', lw=0.05, zorder=3, label=f'{donor_label_dict[s]}')
        ax.plot([x[0], x[0]], [y_ref, y_ref + y[0]], lw=lw, c=color_dict[s])
        legend_handles[donor_label_dict[s]] = ps
        for i in range(1, len(x)):
            ax.plot([x[i], x[i]], [y_ref, y_ref + y[i]], lw=lw, c=color_dict[s], zorder=2)
            ax.scatter(x[i], y[i] + offset, marker='o', s=ms**2, fc=color_dict[s], ec='white', lw=0.05, zorder=3)

    # Add MSOGs
    hybrid_freq_table['fraction_mixed_clusters'] = hybrid_freq_table['M'] / hybrid_freq_table[['A', 'Bp', 'C', 'O', 'M']].sum(axis=1)
    x_msog = hybrid_freq_table.loc[hybrid_freq_table['fraction_mixed_clusters'] >= min_msog_fraction, x_column].values
    y_msog = ymax - 0.25 * np.ones(len(x_msog))
    ps = ax.scatter(x_msog, y_msog, marker='|', s=100, ec=color_dict['M'], lw=0.5, zorder=3, label='M')
    legend_handles['M'] = ps 
    ax.legend(handles=[legend_handles[k] for k in legend_handles], labels=[k for k in legend_handles], loc=legend_loc, fontsize=legend_size, frameon=False, ncol=2, borderaxespad=1.0)


def height_transform(y, y_steps=np.array([0, 1, 2, 5, 15])):
    '''
    Maps input `y` to piecewise linear range from `steps[0]` to `steps[-1]`, with values in
        `steps` mapped to consecutive integers.
    '''

    i0 = np.where(y - y_steps >= 0)[0][-1]
    if i0 < len(y_steps) - 1:
        dy = (y - y_steps[i0]) / (y_steps[i0 + 1] - y_steps[i0])
    else:
        dy = 0

    return i0 + dy

def calculate_species_divergence_along_genome(species_cluster_genomes, pangenome_map, metadata, min_length=200, metric='jc69'):
    og_table = pangenome_map.og_table
    core_og_ids = species_cluster_genomes.index[(species_cluster_genomes[['core_A', 'core_Bp']] == 'Yes').all(axis=1).values]
    sorted_mapped_og_ids = np.array(species_cluster_genomes.loc[core_og_ids, :].sort_values('osbp_location').index)
    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')

    # Filter short OGs
    filtered_idx = []
    for og_id in sorted_mapped_og_ids:
        avg_length = og_table.loc[og_table['parent_og_id'] == og_id, 'avg_length'].mean()
        if avg_length > min_length + 100:
            filtered_idx.append(og_id)

    species_divergence_table = pd.DataFrame(index=filtered_idx, columns=['osa_location', 'osbp_location', 'mean_divergence', 'median_divergence', 'min_divergence', 'max_divergence', 'std_divergence'])
    species_divergence_table[['osa_location', 'osbp_location']] = species_cluster_genomes.loc[filtered_idx, ['osa_location', 'osbp_location']]

    for distance_metric in ['mean', 'median', 'min', 'max', 'std']:
        pdist, og_ids = pangenome_map.calculate_group_divergence(filtered_idx, species_sorted_sag_ids['A'], species_sorted_sag_ids['Bp'], metric=distance_metric)
        if metric == 'jc69':
            div = align_utils.calculate_divergence(pdist)
        else:
            div = pdist
        species_divergence_table.loc[og_ids, f'{distance_metric}_divergence'] = div

    return species_divergence_table


def plot_genomic_trenches_panel(ax, species_divergence_table, species_cluster_genomes, pangenome_map, metadata, rng, pos_column='genome_position', xlim=(-0.02, 3.05), ylim=(0, 0.35), yticks=[0, 0.15, 0.3, 0.45], xlabel_size=12, ylabel_size=18, dx=2, w=5, min_og_presence=0.2, min_length=200, d_low=0.03, random_sample=False, xlabel="OS-B' genome position (Mb)", highlight=[], tick_size=12):

    # Set up axes
    ax.set_xlabel(xlabel, fontsize=xlabel_size)
    ax.set_ylabel(r'$d_S$', fontsize=ylabel_size, rotation=0, labelpad=10, va='center')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=tick_size)

    # Prepare data
    core_og_ids = pangenome_map.get_core_og_ids(metadata, min_og_frequency=min_og_presence, og_type='parent_og_id')
    syn_homolog_map = SynHomologMap(build_maps=True)
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
    if random_sample:
        # Choose random high-coverage A and B' SAGs for individual pair comparisons
        sample_size = 10
        gene_presence_cutoff = 1000
        high_coverage_syna_sag_ids = list(np.array(species_sorted_sag_ids['A'])[(og_table[species_sorted_sag_ids['A']].notna().sum(axis=0) > gene_presence_cutoff).values])
        syna_sample_sag_ids = rng.choice(high_coverage_syna_sag_ids, size=sample_size)
        high_coverage_synbp_sag_ids = list(np.array(species_sorted_sag_ids['Bp'])[(og_table[species_sorted_sag_ids['Bp']].notna().sum(axis=0) > gene_presence_cutoff).values])
        synbp_sample_sag_ids = rng.choice(high_coverage_synbp_sag_ids, size=sample_size)
        sampled_sag_ids = np.concatenate([syna_sample_sag_ids, synbp_sample_sag_ids])

        # Get divergences between A and B' pairs across sites
        dijk_dict = pangenome_map.get_sags_pairwise_divergences(sampled_sag_ids, input_og_ids=filtered_idx)
        pair_divergence_values = []
        x_pair_divergences = []
        for og_id in species_divergence_table.index:
            if og_id in dijk_dict:
                dij = dijk_dict[og_id]
                pair_divergence_values.append(dij[:sample_size, sample_size:].flatten().astype(float))
            else:
                empty_array = np.empty(sample_size**2)
                empty_array[:] = np.nan
                pair_divergence_values.append(empty_array.astype(float))
            x_pair_divergences.append(species_divergence_table.loc[og_id, pos_column])
        pair_divergence_values = np.array(pair_divergence_values)
        x_pair_divergences = np.array(x_pair_divergences)

        for i, pair_divergences in enumerate(pair_divergence_values.T):
            y_smooth = np.array([np.nanmean(pair_divergences[j:j + w]) if np.sum(np.isfinite(pair_divergences[j:j + w])) > 0 else np.nan for j in range(0, len(pair_divergences) - w, dx)])
            x_smooth = np.array([np.mean(x_pair_divergences[j:j + w]) for j in range(0, len(species_divergence_table) - w, dx)])
            ax.plot(x_smooth, y_smooth, lw=0.25, c='gray', alpha=0.3)

    y_smooth = np.array([np.mean(species_divergence_table['mean_divergence'].values[j:j + w]) for j in range(0, len(species_divergence_table) - w, dx)])
    x_smooth = np.array([np.mean(species_divergence_table[pos_column].values[j:j + w]) for j in range(0, len(species_divergence_table) - w, dx)])
    ax.plot(x_smooth, y_smooth, c='k', lw=1.5)

    x = species_divergence_table[pos_column].values
    y_min = species_divergence_table['min_divergence'].values
    y_max = species_divergence_table['max_divergence'].values
    ax.plot(x, y_min, c=open_colors['blue'][5], lw=0.25, ls='-', zorder=1)
    ax.plot(x, y_max, c=open_colors['red'][5], lw=0.25, ls='-', zorder=1)


    dx_highlight = 0.05
    colors = ['gray', 'tab:red']
    for i, og_id in enumerate(highlight):
        x_og = species_divergence_table.loc[og_id, pos_column]
        #ax.fill_between([x_og - dx_highlight, x_og + dx_highlight], ylim[0], ylim[1], alpha=0.5, color=colors[i])
        ax.indicate_inset([x_og - dx_highlight, ylim[0], 2 * dx_highlight, ylim[1] - ylim[0]], ec='k', lw=2.)


def plot_species_divergence_blowup(og_id, species_divergence_table, savefig=None, dx=2, dx_window=0.05, dx_highlight=0.005, pos_column='osbp_location', label_size=14, xlabel='', ylabel=r'$d_S$', yticks=[0, 0.25, 0.5, 0.5, 0.75, 1], ylim=(0, 1), num_xticks=4, tick_size=12):
    species_divergence_table = species_divergence_table.loc[species_divergence_table['mean_divergence'].notnull(), :]

    x_og = species_divergence_table.loc[og_id, pos_column]
    divergence_table_slice = species_divergence_table.loc[(species_divergence_table[pos_column] >= x_og - dx_window) & (species_divergence_table[pos_column] < x_og + dx_window), :]

    fig = plt.figure(figsize=(0.7 * single_col_width, 0.3 * single_col_width))
    ax = fig.add_subplot(111)

    # Set up axes
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.set_ylim(ylim)
    xticks = np.linspace(x_og - dx_window, x_og + dx_window, num_xticks)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=tick_size)


    x = divergence_table_slice[pos_column].values
    y = divergence_table_slice['mean_divergence'].values
    ax.scatter(x, y, s=1.0**2, c='k')

    w = 5
    y_smooth = np.array([np.mean(divergence_table_slice['mean_divergence'].values[j:j + w]) for j in range(0, len(divergence_table_slice) - w, dx)])
    x_smooth = np.array([np.mean(divergence_table_slice[pos_column].values[j:j + w]) for j in range(0, len(divergence_table_slice) - w, dx)])
    ax.plot(x_smooth, y_smooth, c='k', lw=1.5, alpha=0.7)

    x = divergence_table_slice[pos_column].values
    y_min = divergence_table_slice['min_divergence'].values
    y_max = divergence_table_slice['max_divergence'].values
    ax.plot(x, y_min, c=open_colors['blue'][5], lw=0.5, ls='-', zorder=1)
    ax.plot(x, y_max, c=open_colors['red'][5], lw=0.5, ls='-', zorder=1)

    dx_highlight = 0.002
    x_og = species_divergence_table.loc[og_id, pos_column]
    ax.indicate_inset([x_og - dx_highlight, ylim[0], 2 * dx_highlight, ylim[1] - ylim[0]], ec='k', lw=2.)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)



###########################################################
# Fig. 3 panels: Nucleotide-level analysis
###########################################################

def make_snp_level_panels(pangenome_map, args, rng, fig_number='3', fig_dpi=1000, panel_label_fs=14, tick_fs=14):
    print(f'Plotting Fig. {fig_number} panels...\n\n')

    metadata = MetadataMap()
    core_og_ids = pangenome_map.get_core_og_ids(metadata, min_og_frequency=0.2, og_type='parent_og_id', output_type='dict')

    # Block divergences joint distribution
    syna_block_stats = pd.read_csv(f'{args.results_dir}snp_blocks/A_all_sites_hybrid_linkage_block_stats.tsv', sep='\t', index_col=None)
    syna_block_haplotypes = pd.read_csv(f'{args.results_dir}snp_blocks/A_all_sites_hybrid_linkage_block_haplotypes.tsv', sep='\t', index_col=0)
    synbp_block_stats = pd.read_csv(f'{args.results_dir}snp_blocks/Bp_all_sites_hybrid_linkage_block_stats.tsv', sep='\t', index_col=None)
    synbp_block_haplotypes = pd.read_csv(f'{args.results_dir}snp_blocks/Bp_all_sites_hybrid_linkage_block_haplotypes.tsv', sep='\t', index_col=0)


    # Plot alignment blocks
    example_og_id = 'YSG_0947'
    fig = plt.figure(figsize=(double_col_width, 0.7 * single_col_width))
    ax = fig.add_subplot(111)
    x0 = 380
    x1 = 944
    plot_alignment_blocks_panel(ax, example_og_id, pangenome_map, metadata, syna_block_stats, syna_block_haplotypes, args, annot_lw=8, yticks=[], x0=x0, x1=x1)

    xticks = np.linspace(0, x1 - x0 + 1, 6, dtype=int)
    xticklabels = x0 + xticks
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'${x}$' for x in xticklabels], fontsize=tick_fs)
    ax.set_yticks([])
    ax.text(-0.05, 1, 'A', transform=ax.transAxes, fontsize=panel_label_fs, fontweight='bold', va='center', usetex=False)

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}fig{fig_number}A1.pdf', dpi=800)
    plt.close()


    # Plot zoomed-in blocks
    fig = plt.figure(figsize=(single_col_width, 0.6 * single_col_width))
    ax = fig.add_subplot(111)
    aln_consensus = plot_block_alignment_panel(ax, 1, example_og_id, 'A', pangenome_map, metadata, syna_block_stats, syna_block_haplotypes, args, grid=True, label_fs=tick_fs, aspect=5)
    plot_nucleotide_legend(fig, ax, fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}fig{fig_number}A2.pdf', dpi=1000)
    plt.close()


    # Plot consensus pdist
    block_consensus_pdist = align_utils.calculate_fast_pairwise_divergence(aln_consensus)
    for i1 in block_consensus_pdist.index:
        for i2 in block_consensus_pdist.index:
            block_consensus_pdist.loc[i1, i2] = align_utils.calculate_divergence(block_consensus_pdist.loc[i1, i2])

    cmap = 'YlOrBr'
    vrange = [0, np.around(np.max(block_consensus_pdist.values), decimals=1)]
    fig = plt.figure(figsize=(0.6 * single_col_width, 0.6 * single_col_width))
    ax = fig.add_subplot(111)
    plot_block_pdist_panel(ax, block_consensus_pdist, vrange=vrange, matrix_labels=True, tick_fs=10, cmap=cmap, label_fs=10, fontcolor='k', highlight_blocks=True, glw=1.0)
    cmap = generate_color_map(plt.get_cmap(cmap), vrange)
    cax = fig.colorbar(cmap, ax=ax, ticks=np.linspace(vrange[0], vrange[1], 3), fraction=0.046, pad=0.04)
    cax.ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}fig{fig_number}A3.pdf', dpi=1000)
    plt.close()


    # Plot block joint divergence distributions
    ratio = 5
    dlim = (-0.02, 1)
    num_ticks = 5
    num_bins = 31
    x_bins = np.linspace(*dlim, num_bins)

    fig = plt.figure(figsize=(double_col_width / 3, double_col_width / 3.2))
    gspec = gridspec.GridSpec(figure=fig, ncols=2, nrows=2, 
            height_ratios=[1, ratio], width_ratios=[ratio, 1], 
            hspace=0.15, wspace=0.15, bottom=0.27, top=0.93, left=0.29, right=0.97)
    ax = setup_joint_divergence_axes(gspec[1, 0], ylabel=r'$d_{\alpha_1 \gamma}$, $d_{\alpha_2 \gamma}$', xlabel=r'$d_{\alpha_1 \beta}$, $d_{\alpha_2 \beta}$', label_size=tick_fs, xlim=dlim, ylim=dlim, num_ticks=num_ticks)
    x_arr, y_arr, c_arr = get_block_divergences(syna_block_stats[['db', 'd1Bp', 'd1C', 'd2Bp', 'd2C']], species='A')
    ax.scatter(y_arr, x_arr, s=2.5**2, ec='none', lw=0.5, alpha=0.2, color='k', zorder=1)

    # Add highlighted blocks
    df_divergences = syna_block_stats[['og_id', 'db', 'd1Bp', 'd1C', 'd2Bp', 'd2C']]
    idx = np.arange(len(x_arr) // 2)
    i_example = idx[df_divergences.dropna()['og_id'].values == example_og_id][1]
    idx_filter = [i_example, i_example + len(x_arr) // 2]
    ax.scatter(y_arr[idx_filter], x_arr[idx_filter], s=4.5**2, ec='w', lw=1., alpha=1.0, color=['tab:orange', 'tab:blue'], zorder=2)

    # Add marginal distributions
    ax, _ = plot_marginal(gspec[0, 0], y_arr, xlim=dlim, bins=x_bins)
    ax.text(-0.25, 1, 'B', transform=ax.transAxes, fontsize=panel_label_fs, fontweight='bold', va='center', usetex=False)
    plot_marginal(gspec[1, 1], x_arr, xlim=None, ylim=dlim, bins=x_bins, spines=['left'], orientation='horizontal')

    plt.savefig(f'{args.figures_dir}fig{fig_number}B.pdf', dpi=fig_dpi)
    plt.close()


    fig = plt.figure(figsize=(double_col_width / 3, double_col_width / 3.2))
    gspec = gridspec.GridSpec(figure=fig, ncols=2, nrows=2, 
            height_ratios=[1, ratio], width_ratios=[ratio, 1], 
            hspace=0.15, wspace=0.15, bottom=0.27, top=0.93, left=0.29, right=0.97)
    ax = setup_joint_divergence_axes(gspec[1, 0], ylabel=r'$d_{\beta_1 \gamma}$, $d_{\beta_2 \gamma}$', xlabel=r'$d_{\beta_1 \alpha}$, $d_{\beta_2 \alpha}$', label_size=tick_fs, xlim=dlim, ylim=dlim)
    x_arr, y_arr, c_arr = get_block_divergences(synbp_block_stats[['db', 'd1A', 'd1C', 'd2A', 'd2C']], species='Bp')
    ax.scatter(y_arr, x_arr, s=2.5**2, ec='none', lw=0.5, alpha=0.2, color='k', zorder=1)

    # Add marginal distributions
    ax, _ = plot_marginal(gspec[0, 0], y_arr, xlim=dlim, bins=x_bins)
    ax.text(-0.25, 1, 'C', transform=ax.transAxes, fontsize=panel_label_fs, fontweight='bold', va='center', usetex=False)
    plot_marginal(gspec[1, 1], x_arr, xlim=None, ylim=dlim, bins=x_bins, spines=['left'], orientation='horizontal')

    plt.savefig(f'{args.figures_dir}fig{fig_number}C.pdf', dpi=fig_dpi)
    plt.close()

    # Plot block linkage
    fig = plt.figure(figsize=(double_col_width / 3, double_col_width / 3.5))
    ax = fig.add_subplot(111)
    plot_block_haplotype_linkage(ax, syna_block_haplotypes, synbp_block_haplotypes, label_size=12, legend_size=7, xticks=[1E-4, 1E-2, 1])
    ax.set_ylim(2E-6, 1)
    ax.set_ylim(0, 0.08)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}fig{fig_number}D.pdf', dpi=fig_dpi)
    plt.close()


def plot_alignment_blocks_panel(ax, og_id, pangenome_map, metadata, syna_block_stats, syna_block_haplotypes, args, annot_lw=8, yticks=[], i_ref=0, x0=0, x1=1000):
    f_aln = f'{args.results_dir}alignments/core_ogs_cleaned/{og_id}_cleaned_aln.fna'
    aln = seq_utils.read_alignment(f_aln)
    species_grouping = align_utils.sort_aln_rec_ids(aln, pangenome_map, metadata)
    aln_syna = align_utils.get_subsample_alignment(aln, species_grouping['A'])

    plot_alignment(aln_syna[:, x0:x1], annotation={'A':species_grouping['A']}, reference=i_ref, yticks=yticks, marker_size=4, ax=ax, annotation_style='lines', annot_lw=annot_lw)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ref_gene_id = aln_syna[i_ref].id
    ref_sag_id = pangenome_map.get_gene_sag_id(ref_gene_id)

    # Get alignment SAG IDs
    temp = np.array([pangenome_map.get_gene_sag_id(rec.id) for rec in aln_syna])
    aln_sag_ids = [temp[i_ref]]
    for i in range(len(temp)):
        if i != i_ref:
            aln_sag_ids.append(temp[i]) # convoluted but maintains ordering from figure
    aln_sag_ids = np.array(aln_sag_ids)

    annotate_blocks_talk(ax, aln_syna, og_id, ref_sag_id, aln_sag_ids, syna_block_stats, syna_block_haplotypes, x0)

    xticks = np.linspace(0, x1 - x0 + 1, 6, dtype=int)
    xticklabels = x0 + xticks
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'${x}$' for x in xticklabels])


def annotate_blocks_talk(ax, aln, og_id, ref_sag_id, aln_sag_ids, block_stats, block_haplotypes, x0, box_lw=0.1, dx=2, dy=-0.5, fill_color=open_colors['blue'][5], fill_color_alt=open_colors['orange'][5]):
    idx = 0

    for i_block, block_row in block_stats.loc[block_stats['og_id'] == og_id, :].iterrows():
        xi, xf = block_row[['x_start', 'x_end']].astype(int) - x0
        xi -= dx
        xf += dx

        ref_haplotype = int(block_haplotypes.loc[ref_sag_id, f'{og_id}_block{idx}'])
        if ref_haplotype == 10:
            hybrid_haplotype = 20
        else:
            hybrid_haplotype = 10

        block_id = f'{og_id}_block{idx}'

        # Highlight hybrid haplotype
        block_sag_ids = np.array(block_haplotypes.loc[(block_haplotypes[block_id] // 10) == (hybrid_haplotype // 10), block_id].index)
        block_y_index = np.arange(len(aln))[np.isin(aln_sag_ids, block_sag_ids)]
        highlight_block_haplotype(ax, block_y_index, xi, xf, dy, fill_color, box_lw)

        # Highlight ancestral haplotype
        block_sag_ids = np.array(block_haplotypes.loc[(block_haplotypes[block_id] // 10) != (hybrid_haplotype // 10), block_id].index)
        block_y_index = np.arange(len(aln))[np.isin(aln_sag_ids, block_sag_ids)]
        highlight_block_haplotype(ax, block_y_index, xi, xf, dy, fill_color_alt, box_lw)

        idx += 1

def highlight_block_haplotype(ax, block_y_index, xi, xf, dy, fill_color, box_lw):
    j = 0
    while j < len(block_y_index):
        y_list = [block_y_index[j] + dy]
        j += 1
        while (j < len(block_y_index)) and (block_y_index[j] == block_y_index[j - 1] + 1):
            y_list.append(block_y_index[j] + dy)
            j += 1
        yi = np.min(y_list)
        yf = np.max(y_list) + 1
        ax.fill_between([xi, xf], yi, yf, alpha=0.3, fc=fill_color, ec='none')
        ax.plot([xi, xf], [yi, yi], lw=box_lw, c='k')
        ax.plot([xi, xi], [yi, yf], lw=box_lw, c='k')
        ax.plot([xi, xf], [yf, yf], lw=box_lw, c='k')
        ax.plot([xf, xf], [yi, yf], lw=box_lw, c='k')


def plot_block_alignment_panel(ax, i_block, og_id, species, pangenome_map, metadata, block_stats, block_haplotypes, args, ref_id=None, yticklabels=[r'$\alpha_1$', r'$\alpha_2$', r'$\beta$', r'$\gamma$'], grid=False, label_fs=8, aspect='auto'):

    # Read alignment
    f_aln = f'{args.results_dir}alignments/core_ogs_main_cloud/{og_id}_main_cloud_aln.fna'
    aln = seq_utils.read_alignment(f_aln)
    species_grouping = align_utils.sort_aln_rec_ids(aln, pangenome_map, metadata)
    aln_species = align_utils.get_subsample_alignment(aln, species_grouping[species])

    # Get reference index
    rec_ids = [rec.id for rec in aln_species]
    if ref_id is None:
        reference = 0
        ref_gene_id = rec_ids[0] # fill-in for real value
    else:
        reference = rec_ids.index(ref_id)
        ref_gene_id = rec_ids[reference]

    ref_sag_id = pangenome_map.get_gene_sag_id(ref_gene_id)
    ref_haplotype = int(block_haplotypes.loc[ref_sag_id, f'{og_id}_block{i_block}'])
    if ref_haplotype == 10:
        hybrid_haplotype = 20
    else:
        hybrid_haplotype = 10

    # Get alignment SAG IDs
    temp = np.array([pangenome_map.get_gene_sag_id(rec.id) for rec in aln_species])
    aln_sag_ids = [temp[reference]]
    for i in range(len(temp)):
        if i != reference:
            aln_sag_ids.append(temp[i]) # convoluted but maintains ordering from figure
    aln_sag_ids = np.array(aln_sag_ids)

    block_id = f'{og_id}_block{i_block}'
    block_sag_ids = np.array(block_haplotypes.loc[(block_haplotypes[block_id] // 10) != (hybrid_haplotype // 10), block_id].index)
    block_y_index = np.arange(len(aln_species))[np.isin(aln_sag_ids, block_sag_ids)]
    hapl1_gene_ids = [rec_ids[i] for i in block_y_index]
    hapl1_aln = align_utils.get_subsample_alignment(aln_species, hapl1_gene_ids)
    hapl1_seq = seq_utils.get_consensus_seq(hapl1_aln)
    hapl2_gene_ids = [rec_ids[i] for i in range(len(aln_species)) if i not in block_y_index]
    hapl2_aln = align_utils.get_subsample_alignment(aln_species, hapl2_gene_ids)
    hapl2_seq = seq_utils.get_consensus_seq(hapl2_aln)

    # Get x-span
    j = 0
    for _, stats_row in block_stats.loc[block_stats['og_id'] == og_id, :].iterrows():
        if j == i_block:
            x0, x1 = stats_row[['x_start', 'x_end']]
            break
        j += 1

    aln_consensus = [hapl1_seq, hapl2_seq]
    consensus_idx = [f'{species}_1', f'{species}_2']
    for s in species_grouping:
        if s != species:
            aln_donors = align_utils.get_subsample_alignment(aln, species_grouping[s])
            aln_consensus.append(seq_utils.get_consensus_seq(aln_donors))
            consensus_idx.append(s)

    # Set edges to codon boundaries
    if (x0 % 3) != 0:
        x_start = x0 - (x0 % 3)
    if (x1 % 3) != 2:
        dx1 = 2 - (x1 % 3)
        x_end = x1 + dx1

    aln_consensus = align_utils.convert_array_to_alignment(np.array(aln_consensus)[:, x_start:x_end], id_index=consensus_idx)

    plot_alignment(aln_consensus, reference=0, ax=ax, aspect=aspect)
    xticks = np.linspace(0, aln_consensus.get_alignment_length() - 1, 4, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'${x0 + x}$' for x in xticks], fontsize=label_fs)
    ax.set_yticks(np.arange(len(aln_consensus)))
    ax.set_yticklabels([f'${y}$' for y in yticklabels], fontsize=label_fs)

    # Set up axis
    ax.tick_params(which='minor', bottom=False, left=False, right=False, top=False)
    if grid == True:
        data = np.array(aln_consensus)
        ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.25)

    return aln_consensus


def plot_nucleotide_legend(fig, ax, fontsize=10):
    cmap = generate_color_map(nucl_aln_cmap, [-0.5, 4.5])
    cax = fig.colorbar(cmap, ax=ax, ticks=[1, 2, 3, 4], shrink=0.595)
    temp_seq = np.array(['A', 'C', 'G', 'T'])
    seq_numeric = convert_nucleotides_to_numbers(temp_seq, None)
    ticklabels = temp_seq[np.argsort(seq_numeric)]

    cax.ax.set_ylim(4.5, 0.5) # sets alphabetical order from top to bottom
    cax.ax.set_yticklabels(ticklabels, fontsize=fontsize)
    cax.ax.tick_params(labelsize=fontsize)


def plot_block_pdist_panel(ax, block_consensus_pdist, cmap='YlOrRd', grid=True, glw=0.5, vrange=None, matrix_labels=True, label_fs=9, tick_fs=9, legend_fs=9, fontcolor='0.5', highlight_blocks=False, textlw=1.0):
    # Set up a colormap:
    # use copy so that we do not mutate the global colormap instance
    palette = copy.copy(plt.get_cmap(cmap))
    palette.set_bad((1., 1., 1.))

    mask = np.tri(block_consensus_pdist.shape[0], k=-1)
    pdist_arr = np.ma.array(block_consensus_pdist.astype(float), mask=mask)

    if vrange is None:
        im = ax.imshow(pdist_arr, cmap=palette, aspect='equal')
    else:
        im = ax.imshow(pdist_arr, cmap=palette, aspect='equal', vmin=vrange[0], vmax=vrange[1])

    # Turn spines off and create white grid.
    ax.xaxis.set_ticks_position('top')
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(block_consensus_pdist.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(block_consensus_pdist.shape[0] + 1) - 0.5, minor=True)
    ax.tick_params(which='minor', bottom=False, left=False, right=False, top=False)
    if grid == True:
        ax.grid(which='minor', color='w', linestyle='-', linewidth=glw)

    ticks = np.arange(len(block_consensus_pdist))
    ax.set_xticks(ticks, minor=False)
    ax.set_xticklabels([r'$\alpha_1$', r'$\alpha_2$', r'$\beta$', r'$\gamma$'], fontsize=tick_fs)
    ax.set_yticks(ticks, minor=False)
    ax.set_yticklabels([r'$\alpha_1$', r'$\alpha_2$', r'$\beta$', r'$\gamma$'], fontsize=tick_fs)

    if matrix_labels:
        text = ax.text(1, 0, r'$d_{\alpha_1 \alpha_2}$', ha='center', va='center', fontsize=label_fs, fontweight='bold', color=fontcolor)
        text.set_path_effects([mpe.Stroke(linewidth=textlw, foreground='1.0'), mpe.Normal()])

        text = ax.text(2, 0, r'$d_{\alpha_1 \beta}$', ha='center', va='center', fontsize=label_fs, fontweight='bold', color=fontcolor)
        text.set_path_effects([mpe.Stroke(linewidth=textlw, foreground='1.0'), mpe.Normal()])

        text = ax.text(3, 0, r'$d_{\alpha_1 \gamma}$', ha='center', va='center', fontsize=label_fs, fontweight='bold', color=fontcolor)
        text.set_path_effects([mpe.Stroke(linewidth=textlw, foreground='1.0'), mpe.Normal()])

        text = ax.text(2, 1, r'$d_{\alpha_2 \beta}$', ha='center', va='center', fontsize=label_fs, fontweight='bold', color=fontcolor)
        text.set_path_effects([mpe.Stroke(linewidth=textlw, foreground='1.0'), mpe.Normal()])

        text = ax.text(3, 1, r'$d_{\alpha_2 \gamma}$', ha='center', va='center', fontsize=label_fs, fontweight='bold', color=fontcolor)
        text.set_path_effects([mpe.Stroke(linewidth=textlw, foreground='1.0'), mpe.Normal()])

        text = ax.text(3, 2, r'$d_{\beta \gamma}$', ha='center', va='center', fontsize=label_fs, fontweight='bold', color=fontcolor)
        text.set_path_effects([mpe.Stroke(linewidth=textlw, foreground='1.0'), mpe.Normal()])


    if highlight_blocks:
        hlw = 3
        block_colors = ['tab:orange', 'tab:blue']
        for i, xy in enumerate([(1.5, -0.55), (1.5, 0.55)]):
            x0, y0 = xy
            ax.plot([x0, x0 + 2], [y0, y0], lw=hlw, c=block_colors[i], zorder=3)
            ax.plot([x0 + 2, x0 + 2], [y0, y0 + 1], lw=hlw, c=block_colors[i], zorder=3, clip_on=False)
            ax.plot([x0 + 2, x0], [y0 + 1, y0 + 1], lw=hlw, c=block_colors[i], zorder=3)
            ax.plot([x0, x0], [y0 + 1, y0], lw=hlw, c=block_colors[i], zorder=3)


def get_block_divergences(df_divergences, species):
    ref1 = 'C'
    if species == 'A':
        ref2 = 'Bp'
    elif species == 'Bp':
        ref2 = 'A'

    temp = []
    for i_hapl in [1, 2]:
        x = align_utils.calculate_divergence(df_divergences.dropna()[f'd{i_hapl}{ref1}'])
        y = align_utils.calculate_divergence(df_divergences.dropna()[f'd{i_hapl}{ref2}'])
        z = align_utils.calculate_divergence(df_divergences.dropna()['db'])
        temp.append([x.values, y.values, z.values])

    x, y, z = np.concatenate(temp, axis=1)

    return x, y, z


def setup_joint_divergence_axes(gspec_ax, xlabel=r'', ylabel=r'', label_size=14, xlim=(0, 1), ylim=(0, 1), num_ticks=5, xticks_loc='bottom'):
    # Get tick positions
    x0 = max(min(xlim), 0)
    x1 = min(max(xlim), 1)
    xticks = np.linspace(x0, x1, num_ticks)

    y0 = max(min(ylim), 0)
    y1 = min(max(ylim), 1)
    yticks = np.linspace(y0, y1, num_ticks)

    ax = plt.subplot(gspec_ax)
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)

    ax.xaxis.set_ticks_position(xticks_loc)

    return ax


def plot_marginal(gspec_ax, data, dlim=(0, 1), xlim=(0, 1), ylim=None, num_ticks=5, bins=None, num_bins=31, spines=['bottom'], orientation='vertical', xtickpositions='bottom', ytickpositions='left'):
    ax = plt.subplot(gspec_ax)

    # Set tick positions
    ax.xaxis.set_ticks_position(xtickpositions)
    ax.yaxis.set_ticks_position(ytickpositions)
    ticks = np.linspace(*dlim, num_ticks)

    if xlim is not None:
        ax.set_xlim(xlim) 
        ax.set_xticks(ticks)
        ax.set_xticklabels([])
    else:
        ax.set_xticks([])

    if ylim is not None:
        ax.set_ylim(ylim) 
        ax.set_yticks(ticks)
        ax.set_yticklabels([])
    else:
        ax.set_yticks([])

    for s in ax.spines:
        if s not in spines:
            ax.spines[s].set_visible(False)

    if bins is None:
        bins = np.linspace(dlim[0], dlim[1], num_bins)

    hist, _, _ = ax.hist(data, bins=bins, facecolor='0.75', edgecolor='white', linewidth=0.5, orientation=orientation)
    return ax, hist


def plot_block_haplotype_linkage(ax, syna_block_haplotypes, synbp_block_haplotypes, min_num_samples=20, label_size=14, xlim=(9E-6, 1.5), num_bins=50, ms=6, legend_size=11, alpha=0.4, xticks=[1E-4, 1E-3, 1E-2, 1E-1, 1]):
    ax.set_xlabel(r'Block haplotype linkage', fontsize=label_size)
    ax.set_xscale('log')
    ax.set_xticks(xticks)
    ax.set_xlim(xlim)
    ax.set_ylabel(r'Fraction block pairs', fontsize=label_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    x_bins = np.geomspace(xlim[0], 1, num_bins)
    x = np.array([np.sqrt(x_bins[i] * x_bins[i + 1]) for i in range(len(x_bins) - 1)]) # Get mean location for plot
    widths = np.array([x_bins[i + 1] - x_bins[i] for i in range(len(x_bins) - 1)]) # widths for bar plots
    for k, two_allele_block_haplotypes in enumerate([syna_block_haplotypes, synbp_block_haplotypes]):
        block_rsq_values = {'data':[], 'control':[]}
        rsq, num_samples = calculate_rsq(two_allele_block_haplotypes)
        rsq_values = utils.get_matrix_triangle_values(rsq, k=1)
        hist, x_hist = np.histogram(rsq_values, bins=x_bins, density=False)

        num_sample_values = utils.get_matrix_triangle_values(num_samples, k=1)
        block_rsq_values['data'].append(rsq_values[num_sample_values >= min_num_samples])

        rsq_control, num_samples = calculate_rsq(two_allele_block_haplotypes, randomize=True, rng=rng)
        rsq_control_values = utils.get_matrix_triangle_values(rsq_control, k=1)
        num_sample_values = utils.get_matrix_triangle_values(num_samples, k=1)
        block_rsq_values['control'].append(rsq_control_values[num_sample_values >= min_num_samples])
        
        rsq_values = np.concatenate(block_rsq_values['data'])
        rsq_control_values = np.concatenate(block_rsq_values['control'])
        hist_control, x_control = np.histogram(rsq_control_values, bins=x_bins, density=False)

        if k == 0:
            ax.bar(x, hist / np.sum(hist), color='tab:orange', width=widths, ec='white', lw=0.5, alpha=alpha, label=r'$\alpha$ blocks', zorder=0)
            ax.plot(x, hist_control / np.sum(hist_control), ls='-', lw=1.5, c='tab:orange', alpha=1.0, label=r'$\alpha$ unlinked', zorder=1)

        else:
            ax.bar(x, hist / np.sum(hist), color='tab:blue', width=widths, ec='white', lw=0.5, alpha=alpha, label=r'$\beta$ blocks', zorder=0)
            ax.plot(x, hist_control / np.sum(hist_control), ls='-', lw=1.5, c='tab:blue', alpha=1.0, label=r'$\beta$ unlinked', zorder=1)
   
    ax.legend(loc='upper left', fontsize=legend_size, frameon=False)


def calculate_rsq(block_haplotypes, randomize=False, rng=None):
    block_alleles = (block_haplotypes.values // 10).astype(float)
    allele_matrix = np.nan_to_num(block_alleles).astype(int)

    raw_idx = np.arange(allele_matrix.shape[0])
    if randomize:
        for i in range(allele_matrix.shape[1]):
            covered_idx = raw_idx[allele_matrix[:, i] > 0]
            if rng is None:
                shuffled_idx = np.random.permutation(covered_idx)
            else:
                shuffled_idx = rng.permutation(covered_idx)
            allele_matrix[covered_idx, i] = allele_matrix[shuffled_idx, i]

    Dsq, Z = align_utils.calculate_ld_matrices_vectorized(allele_matrix, reference=0, unbiased=False, convert_to_numeric=False)
    num_samples = ((allele_matrix[:, None, :] != 0) * (allele_matrix[:, :, None] != 0)).sum(axis=0)
    return Dsq / (Z + (Z <= 0.0)), num_samples


if __name__ == '__main__':
    # Default variables
    alignment_dir = '../results/single-cell/reference_alignment/'
    figures_dir = '../figures/main_text/'
    pangenome_dir = '../results/single-cell/sscs_pangenome/'
    results_dir = '../results/single-cell/'
    data_dir = '../results/single-cell/main_figures_data/'
    metagenome_dir = '../results/metagenome/'
    f_orthogroup_table = f'{pangenome_dir}filtered_orthogroups/sscs_annotated_single_copy_orthogroup_presence.tsv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-A', '--alignment_dir', default=alignment_dir, help='Directory BLAST alignments against refs.')
    parser.add_argument('-D', '--data_dir', default=data_dir, help='Directory with data for main figures.')
    parser.add_argument('-F', '--figures_dir', default=figures_dir, help='Directory where figures are saved.')
    parser.add_argument('-M', '--metagenome_dir', default=metagenome_dir, help='Directory with results for metagenome.')
    parser.add_argument('-P', '--pangenome_dir', default=pangenome_dir, help='Pangenome directory.')
    parser.add_argument('-R', '--results_dir', default=results_dir, help='Main results directory.')
    parser.add_argument('-g', '--orthogroup_table', default=f_orthogroup_table, help='File with orthogroup table.')
    args = parser.parse_args()

    random_seed = 12345
    rng = np.random.default_rng(random_seed)
    pangenome_map = pg_utils.PangenomeMap(f_orthogroup_table=f_orthogroup_table)
    make_genome_level_figure(pangenome_map, args, rng, fig_number='1')
    make_gene_level_figure(pangenome_map, args, rng, fig_number='2')
    make_snp_level_panels(pangenome_map, args, rng, panel_label_fs=10)
