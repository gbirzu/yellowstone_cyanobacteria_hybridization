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
import quantify_genomewide_linkage as qlink
import make_main_figures as main_figs
import er2
import scipy.stats as stats
import matplotlib.transforms as mtransforms
from clean_orthogroup_alignments import read_main_cloud_alignment
from analyze_metagenome_reads import strip_sample_id, strip_target_id, plot_abundant_target_counts
from syn_homolog_map import SynHomologMap
from metadata_map import MetadataMap
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from plot_utils import *


###########################################################
# Genomic clusters figures
###########################################################


def make_genome_clusters_figures(pangenome_map, args, cloud_radius=0.1, fig_count=1):
    metadata = MetadataMap()
    num_triple_patterns = 5

    # Plot gene triplet example patterns
    fig = plt.figure(figsize=(double_col_width, 0.75 * double_col_width))
    #gs = gridspec.GridSpec(3, 6, width_ratios=[1, 1, 1, 1, 1, 2.5])
    gs = gridspec.GridSpec(3, 5, hspace=0.35, wspace=0.05)

    example_sag_ids = ['UncmicOctRedA1J9_FD', 'UncmicMRedA02H14_2_FD', 'UncmicOcRedA3L13_FD']
    gene_triplet_divergences_dict = pickle.load(open(f'{args.results_dir}reference_alignment/sscs_gene_triple_divergence_tables.dat', 'rb'))
    ax_labels = ['A', 'B', 'C']
    for i, sag_id in enumerate(example_sag_ids):
        ax_objs = []
        for j in range(num_triple_patterns):
            ax_objs.append(fig.add_subplot(gs[i, j]))
        plot_sag_gene_triplet_aggregates(ax_objs, gene_triplet_divergences_dict[sag_id], ax_label=ax_labels[i])
    #plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_gene_triplet_examples.pdf')
    fig_count += 1

    # Plot SAG fingerprints
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    triplet_pattern_table = pd.read_csv(f'{args.results_dir}reference_alignment/gene_triplet_table_cloud_radius{cloud_radius}.tsv', sep='\t', index_col=0)
    #ax = fig.add_subplot(gs[0:, 5])
    plot_sag_fingerprints(ax, triplet_pattern_table, metadata, lw=0.5, cloud_radius=cloud_radius)
    #ax.text(-0.75, 1.2, 'D', fontweight='bold', fontsize=14, ha='center')
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_gene_triplet_fingerprints.pdf')
    fig_count += 1

    plot_16s_tree(pangenome_map, metadata, args, fig_count=fig_count)
    fig_count += 1

    return fig_count


def plot_sag_gene_triplet_aggregates(ax_objs, sag_triplet_divergences_df, triplet_patterns=['XA-B', 'XAB', 'X-A-B', 'X-AB', 'XB-A'], R=0.3, ax_label=None, label_fs=14):
    triplet_divergences_df = sag_triplet_divergences_df.loc[sag_triplet_divergences_df['CYA_vs_CYB_divergence'].notna(), ['CYA_divergence', 'CYB_divergence', 'CYA_vs_CYB_divergence']]
    sag_sorted_triplets = sort_gene_triplet_divergences(triplet_divergences_df)

    #triplet_patterns = [key for key in sag_sorted_triplets if key != 'other']
    for i, pattern in enumerate(triplet_patterns):
        #ax = fig.add_subplot(1, len(sag_sorted_triplets), i + 1)
        ax = ax_objs[i]
        ax.set_title(f'{pattern} (n={len(sag_sorted_triplets[pattern])})', fontsize=8, fontweight='bold')


        divergence_values = triplet_divergences_df.loc[sag_sorted_triplets[pattern], :].values
        if len(divergence_values) > 0:
            #plot_triplets(divergence_values, ax, center='A-B', R=0.25, ms=20)
            plot_triplets(divergence_values, ax, center='X', R=R, ms=20)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if i == 0 and ax_label is not None:
            ax.text(-1.5 * R, 1.0 * R, ax_label, fontweight='bold', fontsize=label_fs)

def sort_gene_triplet_divergences(triplet_divergences_df, cloud_cutoff=0.1):
    sorted_triplets = {'XA':[], 'XB':[], 'XAB':[], 'XA-B':[], 'XB-A':[], 'X-AB':[], 'X-A-B':[], 'other':[]}
    for oid, triplet_row in triplet_divergences_df.iterrows():
        triplet = triplet_row.values
        if triplet[0] < cloud_cutoff:
            if triplet[1] < cloud_cutoff:
                pattern = 'XAB'
            elif triplet[1] < 1:
                pattern = 'XA-B'
            else:
                pattern = 'XA'
        elif triplet[0] < 1:
            if triplet[1] < cloud_cutoff:
                pattern = 'XB-A'
            elif triplet[2] < cloud_cutoff:
                pattern = 'X-AB'
            else:
                pattern = 'X-A-B'
        else:
            if triplet[1] < cloud_cutoff:
                pattern = 'XB'
            else:
                pattern = 'other'
        sorted_triplets[pattern].append(oid)

    return sorted_triplets


def plot_triplets(triplets, ax, center='X', cloud_radius=0.1, ms=40, R=0.4, alpha=0.4):
    ax.set_xlim(-R, R)
    ax.set_xticks([])
    ax.set_ylim(-R, R)
    ax.set_yticks([])
    #ax.set_axis_off()

    if center == 'X':
        circle = plt.Circle((0,0), cloud_radius, fc='none', ec='k', lw=0.5, zorder=0)
        ax.add_patch(circle)
        ax.scatter([0], [0], s=0.75*ms, marker='o', edgecolor='none', facecolor='k', zorder=1)

        for triplet in triplets:
            #print(triplet)
            if (0 not in triplet) and (1 not in triplet):
                triangle_coordinates = calculate_triangle_coords(triplet)
                if ~np.isnan(triangle_coordinates[2][1]):
                    triangle = plt.Polygon(triangle_coordinates, facecolor='none', edgecolor='gray', lw=0.5, alpha=0.35, zorder=2)
                    ax.add_patch(triangle)
                    ax.scatter([triangle_coordinates[1][0]], [triangle_coordinates[1][1]], s=ms, marker='v', edgecolor='none', facecolor='tab:blue', alpha=alpha, zorder=3)
                    ax.scatter([triangle_coordinates[2][0]], [triangle_coordinates[2][1]], s=ms, marker='^', edgecolor='none', facecolor='tab:orange', alpha=alpha, zorder=3)
                else:
                    ax.scatter([-triplet[0]], [0], s=ms, marker='^', edgecolor='none', facecolor='tab:orange', alpha=alpha, zorder=3)
                    ax.scatter([triplet[1]], [0], s=ms, marker='v', edgecolor='none', facecolor='tab:blue', alpha=alpha, zorder=3)
            elif 0 not in triplet:
                if triplet[0] == 1:
                    ax.scatter([triplet[1]], [0], s=ms, marker='v', edgecolor='none', facecolor='tab:blue', alpha=alpha, zorder=3)
                elif triplet[1] == 1:
                    ax.scatter([-triplet[0]], [0], s=ms, marker='^', edgecolor='none', facecolor='tab:orange', alpha=alpha, zorder=3)
                elif triplet[2] == 1:
                    ax.scatter([-triplet[0]], [0], s=ms, marker='^', edgecolor='none', facecolor='tab:orange', alpha=alpha, zorder=3)
                    ax.scatter([triplet[1]], [0], s=ms, marker='v', edgecolor='none', facecolor='tab:blue', alpha=alpha, zorder=3)
            else:
                if triplet[0] == 0 and triplet[1] > 0:
                    ax.scatter([0], [0], s=ms, marker='^', edgecolor='none', facecolor='tab:orange', alpha=alpha, zorder=3)
                    ax.scatter([triplet[1]], [0], s=ms, marker='v', edgecolor='none', facecolor='tab:blue', alpha=alpha, zorder=3)
                elif triplet[1] == 0 and triplet[0] > 0:
                    ax.scatter([-triplet[0]], [0], s=ms, marker='^', edgecolor='none', facecolor='tab:orange', alpha=alpha, zorder=3)
                    ax.scatter([0], [0], s=ms, marker='v', edgecolor='none', facecolor='tab:blue', alpha=alpha, zorder=3)
                elif triplet[2] == 0:
                    ax.scatter([triplet[1]], [0], s=ms, marker='^', edgecolor='none', facecolor='tab:orange', alpha=alpha, zorder=3)
                    ax.scatter([triplet[1]], [0], s=ms, marker='v', edgecolor='none', facecolor='tab:blue', alpha=alpha, zorder=3)

    elif center == 'A-B':
        circles = False
        for triplet in triplets:
            triangle_coordinates = calculate_triangle_coords(triplet, center=center)
            if ~np.isnan(triangle_coordinates[2][1]):
                triangle = plt.Polygon(triangle_coordinates, facecolor='none', edgecolor='gray', lw=0.5, alpha=0.35, zorder=1)
                ax.add_patch(triangle)
                ax.scatter([triangle_coordinates[0][0]], [triangle_coordinates[0][1]], s=ms, marker='^', edgecolor='none', facecolor='tab:orange', alpha=1, zorder=2)
                ax.scatter([triangle_coordinates[1][0]], [triangle_coordinates[1][1]], s=ms, marker='v', edgecolor='none', facecolor='tab:blue', alpha=1, zorder=2)
                #ax.scatter(triangle_coordinates[2][0], triangle_coordinates[2][1], s=0.75*ms, marker='o', edgecolor='none', facecolor='gray', alpha=alpha, zorder=1)
                ax.scatter(triangle_coordinates[2][0], triangle_coordinates[2][1], s=0.25*ms, marker='o', edgecolor='none', facecolor='gray', alpha=alpha, zorder=3)

                if circles == False:
                    circleA = plt.Circle((triangle_coordinates[0][0], triangle_coordinates[0][1]), cloud_radius, fc='none', ec='k', lw=0.5, zorder=0)
                    ax.add_patch(circleA)
                    circleB = plt.Circle((triangle_coordinates[1][0], triangle_coordinates[1][1]), cloud_radius, fc='none', ec='k', lw=0.5, zorder=0)
                    ax.add_patch(circleB)
                    circles = True
            else:
                ax.plot([-triplet[2] / 2, -triplet[2] / 2 + triplet[1], triplet[2] / 2], [0, 0, 0], '-', lw=0.5, color='gray', alpha=0.35, zorder=1)
                ax.scatter([-triplet[2] / 2 + triplet[0]], [0], s=0.25*ms, marker='o', edgecolor='none', facecolor='gray', alpha=alpha, zorder=3)


def calculate_triangle_coords(triplet, center='X'):
    if center == 'X':
        coords = [[0, 0], [triplet[1], 0]]
        cosa = (triplet[0]**2 + triplet[1]**2 - triplet[2]**2) / (2 * triplet[0] * triplet[1])
        cosb = (triplet[1]**2 + triplet[2]**2 - triplet[0]**2) / (2 * triplet[1] * triplet[2])
        sina = np.sqrt(1 - cosa**2)
        coords.append([triplet[0] * cosa, triplet[0] * sina])
    else:
        coords = [[-triplet[2] / 2, 0], [triplet[2] / 2, 0]]
        cosa = (triplet[0]**2 + triplet[2]**2 - triplet[1]**2) / (2 * triplet[0] * triplet[2])
        sina = np.sqrt(1 - cosa**2)
        coords.append([ - triplet[2] / 2 + triplet[0] * cosa, triplet[0] * sina])
    return coords


def plot_sag_fingerprints(ax, triplet_pattern_table, metadata, sag_ids=None, x_labels = ['XA-B', 'XAB', 'X-A-B', 'X-AB', 'XB-A'], lw=1, alpha=0.5, cloud_radius=10):
    color_dict = {'A':'tab:orange', 'Bp':'tab:blue', 'C':'tab:green'}
    species_added = {'A':False, 'Bp':False, 'C':False}
    species_labels = {'A':r'$\alpha$', 'Bp':r'$\beta$', 'C':r'$\gamma$'}
    x = np.arange(len(x_labels))

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(-0.02, 1.2)
    ax.set_ylabel('gene fraction')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

    if sag_ids is None:
        sag_ids = np.array(triplet_pattern_table.columns)[::-1] # sort in A-B-C order

    for i, sag_id in enumerate(sag_ids):
        pattern_counts = np.array([(triplet_pattern_table[sag_id] == pattern).sum() for pattern in x_labels])
        y = pattern_counts / np.sum(pattern_counts)
        sag_species = metadata.get_sag_species(sag_id)
        if species_added[sag_species] == False:
            ax.plot(x, y, lw=lw, alpha=alpha, c=color_dict[sag_species], label=species_labels[sag_species])
            species_added[sag_species] = True
        else:
            ax.plot(x, y, lw=lw, alpha=alpha, c=color_dict[sag_species])

    ax.legend(fontsize=10, loc='upper center')


def plot_16s_tree(pangenome_map, metadata, args, fig_count=1):
    # Set parameters
    branch_thickness=1.5

    # Set node styles for species
    nst_alpha = NodeStyle()
    nst_alpha["bgcolor"] = "Orange"
    nst_alpha['fgcolor'] = 'black'
    nst_alpha['size'] = 0

    nst_beta = NodeStyle()
    nst_beta["bgcolor"] = "DeepSkyBlue"
    nst_beta['fgcolor'] = 'black'
    nst_beta['size'] = 0

    nst_gamma = NodeStyle()
    nst_gamma["bgcolor"] = "PaleGreen"
    nst_gamma['fgcolor'] = 'black'
    nst_gamma['size'] = 0

    #rrna_tree = Tree(f'{args.results_dir}locus_diversity/rrna_UPGMA_tree.nwk')
    rrna_tree = Tree(f'{args.results_dir}supplement/rrna_UPGMA_tree.nwk')

    # Get leaf names sorted by species
    leaf_ids = [leaf.name for leaf in rrna_tree.iter_leaves()]
    leaf_names = [leaf.name.strip('\'') for leaf in rrna_tree.iter_leaves()]
    species_sorted_sags = metadata.sort_sags(pangenome_map.get_sag_ids(), by='species')
    sorted_leaves = {}
    for species in ['A', 'Bp', 'C']:
        temp_list = [utils.strip_sample_name(sid, replace=True) for sid in species_sorted_sags[species]]
        sorted_leaves[species] = [leaf_ids[leaf_names.index(lid)] for lid in temp_list if lid in leaf_names]

    ts = ete3.TreeStyle()
    ts.mode = 'c' # Make tree circular
    ts.draw_guiding_lines = False
    ts.scale_length = 0.005 # Set scale

    # Set background colors
    syna_ancestor = rrna_tree.get_common_ancestor(sorted_leaves['A'])
    syna_ancestor.set_style(nst_alpha)
    synbp_ancestor = rrna_tree.get_common_ancestor(sorted_leaves['Bp'])
    synbp_ancestor.set_style(nst_beta)
    sync = rrna_tree.search_nodes(name=sorted_leaves['C'][0])[0]
    sync.set_style(nst_gamma)
    set_nodes = [syna_ancestor, synbp_ancestor, sync]

    # Modified from Kat Holt package
    for node in rrna_tree.traverse():
        if node in set_nodes:
            continue

        nstyle = NodeStyle()
        nstyle['fgcolor'] = 'black'
        nstyle['size'] = 0
        node.set_style(nstyle)

        node.img_style['hz_line_width'] = branch_thickness
        node.img_style['vt_line_width'] = branch_thickness

    rrna_tree.dist = 0 # Set root distance to zero
    rrna_tree.render(f'{args.figures_dir}S{fig_count}_16s_tree.pdf', w=110, units='mm', dpi=300, tree_style=ts)




###########################################################
# Linkage figures
###########################################################

def make_linkage_figures(pangenome_map, args, cloud_radius=0.1, avg_length_fraction=0.75, fig_count=4, ax_label_size=14, tick_size=14):
    color_dict = {'A':'tab:orange', 'Bp':'tab:blue', 'Bp_subsampled':'gray', 'population':'k'}
    label_dict = {'A':r'$\alpha$', 'Bp':r'$\beta$', 'Bp_subsampled':r'$\beta$ (subsampled)', 'population':r'whole population'}

    random_gene_linkage = pickle.load(open(f'{args.results_dir}linkage_disequilibria/sscs_core_ogs_random_gene_linkage_c{cloud_radius}.dat', 'rb'))

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    set_up_linkage_curve_axis(ax, xlim=(1, 2E4), ylim=(5E-3, 1.5E0), ax_label='A', ax_label_fs=16)
    for species in ['A', 'Bp', 'population']:
        linkage_results = pickle.load(open(f'{args.results_dir}linkage_disequilibria/sscs_core_ogs_{species}_linkage_curves_c{cloud_radius}.dat', 'rb'))
        sigmad2 = plt_linkage.average_sigmad_sq(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
        sigmad2_cg, x_cg = plt_linkage.coarse_grain_linkage_array(sigmad2)
        ax.plot(x_cg[:-5], sigmad2_cg[:-5], '-o', ms=3, mec='none', lw=1, alpha=1.0, c=color_dict[species], label=label_dict[species]) # exclude last points with low depth
        if species in random_gene_linkage:
            ax.scatter(1.5E4, random_gene_linkage[species], s=20, ec='none', fc=color_dict[species])
    ax.axvline(1E4, ls='--', c='k')
    ax.legend(fontsize=10)

    '''
    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(121)
    set_up_linkage_curve_axis(ax, xlim=(1, 2E4), ylim=(5E-3, 1.5E0), ax_label='A', ax_label_fs=16)
    for species in ['A', 'Bp', 'population']:
        linkage_results = pickle.load(open(f'{args.results_dir}linkage_disequilibria/sscs_core_ogs_{species}_linkage_curves_c{cloud_radius}.dat', 'rb'))
        sigmad2 = plt_linkage.average_sigmad_sq(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
        sigmad2_cg, x_cg = plt_linkage.coarse_grain_linkage_array(sigmad2)
        ax.plot(x_cg[:-5], sigmad2_cg[:-5], '-o', ms=3, mec='none', lw=1, alpha=1.0, c=color_dict[species], label=label_dict[species]) # exclude last points with low depth
        if species in random_gene_linkage:
            ax.scatter(1.5E4, random_gene_linkage[species], s=20, ec='none', fc=color_dict[species])
    ax.axvline(1E4, ls='--', c='k')
    ax.legend(fontsize=10)


    ax = fig.add_subplot(122)
    divergence_array, og_ids, sag_ids = pickle.load(open(f'{args.results_dir}locus_diversity/genomewide_pairwise_divergence_array.dat', 'rb'))

    og_table = pangenome_map.og_table
    sag_prefix_dict = pg_utils.make_prefix_dict(og_table)
    f_rrna_aln = f'../results/single-cell/rRNA/16S_rRNA_manual_aln.fna'
    rrna_aln = seq_utils.read_alignment(f_rrna_aln)
    trimmed_aln, x_trimmed = align_utils.trim_alignment_and_remove_gaps(rrna_aln, max_edge_gaps=0.0)

    # Rename seqs using SAG IDs
    for rec in trimmed_aln:
        rec.id = sag_prefix_dict[rec.id.split('_')[0]]

    # Calculate pairwise divergences
    rrna_pairwise_divergences = align_utils.calculate_fast_pairwise_divergence(trimmed_aln)
    #plot_genomewide_vs_rrna_correlation(ax, rrna_pairwise_divergences, divergence_array, sag_ids, xlim=(-0.0005, 0.015), ylim=(0, 0.05), genomewide_stat='mean', ax_label='B', ax_label_fs=16)
    '''
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_linkage_decay.pdf')
    fig_count += 1


    # Compare A with B' and subsampled B'
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    set_up_linkage_curve_axis(ax, xlim=(1, 2E4), ylim=(5E-3, 1.5E0))
    for species in ['A', 'Bp', 'Bp_subsampled']:
        linkage_results = pickle.load(open(f'{args.results_dir}linkage_disequilibria/sscs_core_ogs_{species}_linkage_curves_c{cloud_radius}.dat', 'rb'))
        sigmad2 = plt_linkage.average_sigmad_sq(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
        sigmad2_cg, x_cg = plt_linkage.coarse_grain_linkage_array(sigmad2)
        ax.plot(x_cg[:-5], sigmad2_cg[:-5], '-o', ms=3, mec='none', lw=1, alpha=1.0, c=color_dict[species], label=label_dict[species]) # exclude last points with low depth
        if species in random_gene_linkage:
            ax.scatter(1.5E4, random_gene_linkage[species], s=20, ec='none', fc=color_dict[species])
    ax.axvline(1E4, ls='--', c='k')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_linkage_depth_validation.pdf')
    fig_count += 1


    # Make 16S vs WG correlation
    metadata = MetadataMap()
    f_divergences = f'../results/single-cell/locus_diversity/core_ogs_species_consensus_divergence_table.tsv'
    consensus_divergence_table = pd.read_csv(f_divergences, sep='\t', index_col=0)
    consensus_divergence_table['average'] = consensus_divergence_table.mean(axis=1)
    rrna_aln = main_figs.read_rrna_alignment(pangenome_map)
    rrna_consensus_divergences = main_figs.calculate_locus_consensus_divergence(rrna_aln, metadata)
    rrna_sag_ids = np.array(rrna_consensus_divergences.index)

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    main_figs.plot_consensus_divergence_loci_comparisons(ax, rrna_consensus_divergences.values, consensus_divergence_table.loc[rrna_sag_ids, 'average'].values, rrna_sag_ids, metadata, fig, ax_label='', label_size=ax_label_size, tick_size=tick_size)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_16S_WG_correlation.pdf')
    fig_count += 1

    return fig_count


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


def plot_genomewide_vs_rrna_correlation(ax, rrna_pairwise_divergences, divergence_array, sag_ids, xlim=(-0.0005, 0.055), ylim=(-0.005, 0.2), genomewide_stat='median', fs=14, ax_label='', ax_label_fs=16):
    sag_ids_arr = np.array(sag_ids)
    rrna_sag_ids = np.array(rrna_pairwise_divergences.index)
    rrna_idx = [np.where(sag_ids_arr == s)[0][0] for s in rrna_sag_ids]

    if genomewide_stat == 'median':
        genome_divergence = np.nanmedian(divergence_array[:, rrna_idx, :][:, :, rrna_idx], axis=0)
    elif genomewide_stat == 'mean':
        genome_divergence = np.nanmean(divergence_array[:, rrna_idx, :][:, :, rrna_idx], axis=0)
    genome_divergence_values = utils.get_matrix_triangle_values(genome_divergence, k=1)
    rrna_divergence_values = utils.get_matrix_triangle_values(rrna_pairwise_divergences.values, k=1)

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    #ax.set_xlabel('16S divergence')
    ax.set_xlabel('16S divergence', fontsize=fs)
    ax.set_xlim(xlim)
    #ax.set_ylabel(f'{genomewide_stat} genomewide divergence')
    ax.set_ylabel(f'genomewide divergence', fontsize=fs)
    ax.set_ylim(ylim)
    #ax.scatter(genome_divergence_values, rrna_divergence_values, s=5, fc='gray', ec='none', alpha=0.7)
    ax.scatter(rrna_divergence_values, genome_divergence_values, s=8, fc='k', ec='none', alpha=0.05)
    ax.text(-0.003, 0.05, ax_label, fontweight='bold', fontsize=ax_label_fs)

    print(rrna_divergence_values, len(rrna_divergence_values))
    print(utils.sorted_unique(rrna_divergence_values))

    r_all, pvalue_all = stats.pearsonr(genome_divergence_values, rrna_divergence_values)

    close_idx = (genome_divergence_values < 0.05) & (rrna_divergence_values < 0.015)
    print(len(close_idx))
    print(rrna_divergence_values[close_idx], len(rrna_divergence_values[close_idx]))
    print(genome_divergence_values[close_idx], len(genome_divergence_values[close_idx]))
    slope, intercept, r_close, pvalue_close, stderr, intercept_stderr = stats.linregress(rrna_divergence_values[close_idx], genome_divergence_values[close_idx])
    if ylim[1] < 0.075:
        ax.plot(xlim, intercept + slope * np.array(xlim), c='k')
        xy_text = (np.mean(xlim) / 2, intercept + slope * np.array(np.mean(xlim)) + 0.02)
        ax.annotate(f'$R^2={r_close**2:.2f}$', xy_text, fontsize=14)
    print(f'All pairs: rho={r_all:.3f}, p-value={pvalue_all:.1e}')
    print(f'Close pairs: rho={r_close:.3f}, R^2={r_close**2:.3f}, p-value={pvalue_close:.1e}, slope={slope:.4f}')
    print('\n\n')




###########################################################
# Recent transfers
###########################################################

def make_recent_transfer_figures(pangenome_map, args, contig_length_cutoff=0, fig_count=6, min_msog_fraction=0.5):
    metadata = MetadataMap()

    merged_donor_frequency_table = read_merged_donor_frequency_table(pangenome_map, metadata, args)
    plot_hybridization_pie_chart(merged_donor_frequency_table, savefig=f'{args.figures_dir}S{fig_count}_hybridization_pie.pdf')

    '''
    hybridization_table = utils.read_hybridization_table(f'{args.results_dir}hybridization/sscs_hybridization_events.tsv', length_cutoff=contig_length_cutoff)

    # Get species sorted SAG IDs
    metadata = MetadataMap()
    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')

    # Calculate hybridization counts for each hybrid OG
    hybridized_og_ids, og_hybridization_counts = utils.sorted_unique(np.concatenate(hybridization_table['hybrid_og_ids'].str.split(';').values))

    # Plot breakdown of hybrids per OG
    og_table = pangenome_map.og_table
    hybridized_parent_og_ids, parent_og_hybridization_counts = calculate_parent_og_hybridization_counts(hybridized_og_ids, og_hybridization_counts, og_table)
    #core_parent_ogs_dict = get_core_og_ids(og_table, species_sorted_sags, args, og_type='parent_og_id', return_dict=True)
    core_parent_ogs_dict = pangenome_map.get_core_og_ids(metadata, og_type='parent_og_id', output_type='dict')
    plot_hybridization_pie_chart(hybridized_parent_og_ids, parent_og_hybridization_counts, core_parent_ogs_dict, savefig=f'{args.figures_dir}S{fig_count}_hybridization_pie.pdf')
    '''


def read_merged_donor_frequency_table(pangenome_map, metadata, args):
    species_cluster_genomes = pd.read_csv(f'{args.results_dir}hybridization/sscs_labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)

    # Read donor frequncy tables
    species_donor_frequency_tables = {}
    temp = []
    for species in ['A', 'Bp']:
        donor_frequency_table = main_figs.make_donor_frequency_table(species_cluster_genomes, species, pangenome_map, metadata)
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
    bin_labels = [f'no hybrids ({bins[0]})', f'mosaic hybrids\nand other\nmixed clusters({bins[1]})', f'singleton\nhybrids\n({bins[2]})', f'non-singleton\nhybrids ({bins[3]})']
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


def label_func(pct, allvals):
    absolute = int(np.round(pct / 100. * np.sum(allvals)))
    return f'{pct:.1f}\%\n({absolute:d})'



def calculate_parent_og_hybridization_counts(hybridized_og_ids, og_hybridization_counts, og_table):
    temp = og_table.loc[hybridized_og_ids, 'parent_og_id'].values
    hybridized_parent_og_ids = []
    parent_og_hybridization_counts = []
    for i, og_id in enumerate(hybridized_og_ids):
        parent_og_id = temp[i]
        if parent_og_id in hybridized_parent_og_ids:
            j = hybridized_parent_og_ids.index(parent_og_id)
            parent_og_hybridization_counts[j] += og_hybridization_counts[i]
        else:
            hybridized_parent_og_ids.append(parent_og_id)
            parent_og_hybridization_counts.append(og_hybridization_counts[i])
    return np.array(hybridized_parent_og_ids), np.array(parent_og_hybridization_counts)


def plot_hybridization_pie_chart_v1(hybridized_parent_og_ids, parent_og_hybridization_counts, core_parent_ogs_dict, savefig=None):
    # Get pure mixed clusters
    temp = core_parent_ogs_dict['M']
    pure_mixed_filter = ~(np.isin(temp, core_parent_ogs_dict['A']) | np.isin(temp, core_parent_ogs_dict['Bp']))
    pure_mixed_og_ids = temp[pure_mixed_filter]
    core_parent_og_ids = np.unique(np.concatenate(list(core_parent_ogs_dict.values())))

    temp_arr = np.unique(np.concatenate(list(core_parent_ogs_dict.values())))
    core_nonmixed_og_ids = temp_arr[~np.isin(temp_arr, pure_mixed_og_ids)]

    bins = [np.sum(~np.isin(core_parent_og_ids, hybridized_parent_og_ids)), len(pure_mixed_og_ids), 
            np.sum(np.isin(core_nonmixed_og_ids, hybridized_parent_og_ids[parent_og_hybridization_counts <= 1])),
            np.sum(np.isin(core_nonmixed_og_ids, hybridized_parent_og_ids[parent_og_hybridization_counts > 1]))]
    #bin_labels = ['no hybrids', 'mixed\nclusters', 'singleton\nhybrids', 'non-singleton\nhybrids']
    bin_labels = [f'no hybrids ({bins[0]})', f'mixed\nclusters\n({bins[1]})', f'singleton\nhybrids\n({bins[2]})', f'non-singleton\nhybrids ({bins[3]})']

    #text_props = {'weight':'bold', 'size':12}
    text_props = {'size':12}
    text_fmt = r'%1.0f\%%'
    #text_fmt = r'{.0f}%'
    fig = plt.figure(figsize=(single_col_width, single_col_width))
    ax = fig.add_subplot(111)
    #ax.set_title(f'n = {len(np.concatenate(list(core_parent_ogs_dict.values())))}', fontweight='bold')
    ax.pie(bins, labels=bin_labels, autopct=text_fmt, textprops=text_props, labeldistance=1.1)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()



###########################################################
# Full sweeps
###########################################################

def make_species_full_sweep_figures(pangenome_map, args, fig_count=9):
    species_cluster_genomes = pd.read_csv(f'{args.results_dir}hybridization/sscs_labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)
    metadata = MetadataMap()
    divergence_files = [f'{args.results_dir}sscs_pangenome/_aln_results/sscs_orthogroup_{j}_divergence_matrices.dat' for j in range(10)]
    pangenome_map.read_pairwise_divergence_results(divergence_files)

    fig = plt.figure(figsize=(double_col_width, 1.1 * single_col_width))
    for i, species in enumerate(['A', 'Bp']):
        ax = fig.add_subplot(2, 1, i + 1)
        plot_species_diversity_along_genome(ax, species_cluster_genomes, pangenome_map, metadata, species=species)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_core_gene_diversity.pdf')
    fig_count += 1
    return fig_count


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



###########################################################
# Hybridization quality control
###########################################################

def make_hybridization_qc_figures(pangenome_map, args, low_diversity_cutoff=0.05, fig_count=14):
    metadata = MetadataMap()
    sag_ids = pangenome_map.get_sag_ids()
    syna_sags = metadata.sort_sags(sag_ids, by='species')['A']
    pure_syna_sample_sags, mixed_syna_sample_sags = make_syna_test_samples(pangenome_map, metadata)

    # Split OGs by diversity
    syna_4d_diversity = pd.read_csv(f'../results/single-cell/snp_blocks/syna/A_core_ogs_4D_diversity_filtered.tsv', sep='\t', index_col=0)
    low_diversity_ogs = np.unique(np.array(syna_4d_diversity.loc[syna_4d_diversity['snps_per_site'] < low_diversity_cutoff].index))
    high_diversity_ogs = np.unique(np.array(syna_4d_diversity.loc[syna_4d_diversity['snps_per_site'] >= low_diversity_cutoff].index))
    og_ids = np.concatenate([low_diversity_ogs, high_diversity_ogs])

    plot_species_composition_timeseries(metadata, f'{args.figures_dir}S{fig_count}_species_composition.pdf')
    fig_count += 1

    plot_og_presence_model_validation(pangenome_map, low_diversity_ogs, pure_syna_sample_sags, f'{args.figures_dir}S{fig_count}_og_presence_model_fit.pdf')
    fig_count += 1

    plot_og_presence_tests(pangenome_map, (low_diversity_ogs, high_diversity_ogs), (pure_syna_sample_sags, mixed_syna_sample_sags), f'{args.figures_dir}S{fig_count}_og_presence_tests.pdf')
    fig_count += 1

    plot_block_distributions_sample_comparisons('../results/single-cell/qc/', f'{args.figures_dir}S{fig_count}_block_distribution_comparison.pdf')
    fig_count += 1

    return fig_count

def make_syna_test_samples(pangenome_map, metadata):
    sag_ids = pangenome_map.get_sag_ids()
    syna_sag_ids = metadata.sort_sags(sag_ids, by='species')['A']
    sample_sorted_sag_ids = metadata.sort_sags(syna_sag_ids, by='sample')
    pure_syna_sample_sags = np.array(sample_sorted_sag_ids['OS2005'])
    mixed_syna_sample_sags = np.concatenate([sample_sorted_sag_ids[sample] for sample in ['MS2004', 'MS2006', 'OS2009']])
    return pure_syna_sample_sags, mixed_syna_sample_sags

def plot_species_composition_timeseries(metadata, savefig, label_fs=14, legend_fs=12):
    os_t, os_num = make_spring_timeseries(metadata, 'Octopus Spring')
    ms_t, ms_num = make_spring_timeseries(metadata, 'Mushroom Spring')
    xticks = np.arange(2004, 2010)
    yticks = [0, 0.25, 0.5, 0.75, 1.]

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('year', fontsize=label_fs)
    ax.set_xticks(xticks)
    ax.set_ylabel(r'$\alpha$ relative abundance', fontsize=label_fs)
    ax.set_yticks(yticks)

    p, p_err = estimate_binomial_fraction(os_num)
    ax.errorbar(os_t, 1 - p[:, 1], yerr=p_err[::-1], fmt='-o', mec='none', mfc='cyan', c='cyan', elinewidth=1, capsize=3, label='OS')
    p, p_err = estimate_binomial_fraction(ms_num)
    ax.errorbar(ms_t, 1 - p[:, 1], yerr=p_err[::-1], fmt='-o', mec='none', mfc='magenta', c='magenta', elinewidth=1, capsize=3, label='MS')
    ax.legend(fontsize=legend_fs)
    plt.tight_layout()
    plt.savefig(savefig)
    plt.close()


def make_spring_timeseries(metadata, spring_name):
    metadata_df = metadata.metadata
    years = np.sort(metadata_df.loc[metadata_df['Spring_Name'] == spring_name, 'Year'].values)
    num_sags = []
    for t in years:
        sample_ids = metadata.get_sag_ids({'spring_name':spring_name, 'year':t})
        num_a_sags = sum([sag_id in sample_ids for sag_id in metadata.sag_species['A']])
        num_bp_sags = sum([sag_id in sample_ids for sag_id in metadata.sag_species['Bp']])
        num_sags.append([num_a_sags, num_bp_sags])
    return (years, np.array(num_sags))

def estimate_binomial_fraction(counts):
    N = np.sum(counts, axis=1)
    p = counts / N[:, None]
    err_u = 0.5 * np.sqrt(np.prod(p, axis=1) / N)
    err_l = 0.5 * np.sqrt(np.prod(p, axis=1) / N)
    err_u += 1.0 * (p[:, 0] == 0) / N # set error for p = 0 to 1/N
    err_l += 1.0 * (p[:, 0] == 1) / N # set error for p = 0 to 1/N
    return p, np.array([err_l, err_u])


def plot_og_presence_model_validation(pangenome_map, og_ids, sag_ids, savefig, bins=100, epsilon=0.001):
    # Define variables
    N = len(sag_ids)
    L = len(og_ids)
    presence_df, covariance_df = calculate_og_presence_covariance(pangenome_map, og_ids, sag_ids)

    # Fit OG coverage mean and variance
    sag_coverage = np.sum(presence_df.values, axis=1) / L 
    mu_coverage = np.mean(sag_coverage)
    sigma_coverage = np.sqrt(np.sum(sag_coverage * (1 - sag_coverage))) / N

    sorted_og_ids = og_ids[np.argsort(np.sum(presence_df, axis=0))[::-1]]
    sorted_sag_ids = sag_ids[np.argsort(np.sum(presence_df, axis=1))[::-1]]
    
    nrows = 2
    ncols = 3
    h_ratio = 1.5
    w_ratio = 10
    #ratio = 1
    fig = plt.figure(figsize=(double_col_width, (2 / (1 + h_ratio)) * double_col_width), constrained_layout=True)
    gspec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig, height_ratios=[h_ratio, 1], width_ratios=[w_ratio, w_ratio, 1])

    ax = plt.subplot(gspec[0, 0])
    ax.text(-120, -2, 'A', fontweight='bold', fontsize=16)
    ax.imshow(presence_df.loc[sorted_sag_ids, sorted_og_ids].values, aspect='auto', vmin=0, vmax=1, cmap='Greys', interpolation='nearest')
    ax.set_xlabel('OG index', fontsize=12)
    ax.set_ylabel('SAG index', fontsize=12)

    ax = plt.subplot(gspec[1, 0])
    ax.text(-0.2, 7, 'C', fontweight='bold', fontsize=16)
    og_coverage = np.sum(presence_df.values, axis=0) / N
    plot_presence_distribution(ax, og_coverage, fit_params=(mu_coverage, sigma_coverage))
    ax.legend(fontsize=10)

    ax = plt.subplot(gspec[0, 1])
    ax.text(-150, -30, 'B', fontweight='bold', fontsize=16)
    cmap = plt.get_cmap('coolwarm')
    #cmap = plt.get_cmap('bwr')
    norm = mpl.colors.TwoSlopeNorm(vmin=np.min(covariance_df.values), vcenter=0, vmax=np.max(covariance_df.values))
    im = ax.imshow(covariance_df.loc[sorted_og_ids, sorted_og_ids].values, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    ticks = np.arange(0, 801, 200)
    ax.set_xlabel('OG index', fontsize=12)
    ax.set_xticks(ticks)
    ax.set_ylabel('OG index', fontsize=12)
    ax.set_yticks(ticks)

    ax = plt.subplot(gspec[0, 2])
    plt.colorbar(im, cax=ax)

    ax = plt.subplot(gspec[1, 1])
    ax.text(-0.2, 33, 'D', fontweight='bold', fontsize=16)
    ax.set_xlim(-0.1, 0.5)
    ax.set_ylim(-0.02, 30) 
    ax.set_xlabel('OG presence covariance')
    ax.set_ylabel('probability density')


    # Compare diagonal and off-diagonal to prediction from independent presence
    C_aa_values = covariance_df.values.diagonal()
    C_ab_values = utils.get_matrix_triangle_values(covariance_df.values, k=1)
    C_values = np.concatenate([C_aa_values, C_ab_values])
    pbar_i = np.mean(presence_df.values, axis=1)

    mu_aa = np.mean(pbar_i * (1 - pbar_i))
    sigma_aa = np.sqrt(np.sum(pbar_i - 4 * pbar_i**2 + 6 * pbar_i**3 - 3 * pbar_i**4) / N**2)
    sigma_ab = np.sqrt(np.sum(pbar_i**2 - 2 * pbar_i**3 + pbar_i**4) / N**2)

    x_bins = np.linspace(np.min(C_values) - epsilon, np.max(C_values) + epsilon, bins)

    ax.hist(C_aa_values, bins=x_bins, color='tab:blue', alpha=0.6, density=True, label='data')
    y_aa = stats.norm.pdf(x_bins, loc=mu_aa, scale=sigma_aa)
    ax.plot(x_bins, y_aa, color='tab:orange', label='$C_{aa}$ (idependent OGs)')

    ax.hist(C_ab_values, bins=x_bins, color='tab:blue', alpha=0.6, density=True)
    y_ab = stats.norm.pdf(x_bins, loc=0, scale=sigma_ab)
    ax.plot(x_bins, y_ab, color='tab:green', label='$C_{ab}$ (idependent OGs)')
    ax.legend(fontsize=10)

    #plt.tight_layout()
    plt.savefig(savefig)
    plt.close()

    return presence_df, covariance_df

def calculate_og_presence_covariance(pangenome_map, og_ids, sag_ids):
    og_subtable = pangenome_map.og_table.loc[pangenome_map.og_table['parent_og_id'].isin(og_ids), :]
    og_presence_matrix = []
    for o in og_ids:
        og_presence = np.sum(og_subtable.loc[og_subtable['parent_og_id'] == o, sag_ids].notna().values, axis=0)
        og_presence_matrix.append(og_presence)
    p_matrix = np.array(og_presence_matrix).T
    presence_df = pd.DataFrame(p_matrix, index=sag_ids, columns=og_ids)

    N = p_matrix.shape[0]
    L = p_matrix.shape[1]
    pbar_i = np.mean(p_matrix, axis=1)
    pbar_matrix = np.dot(pbar_i[:, None], np.ones(L)[None, :])
    covariance_df = pd.DataFrame(np.dot((p_matrix - pbar_matrix).T, (p_matrix - pbar_matrix)) / N, index=og_ids, columns=og_ids)
    return presence_df, covariance_df

def plot_presence_distribution(ax, og_coverage, fit_params=None, x_bins=None, num_bins=47, alpha=1.0, xlabel='fraction SAGs present', ylabel='probability density', label='', fit_label='fit', label_fs=12, legend_fs=12, **hist_kwargs):
    ax.set_xlabel(xlabel, fontsize=label_fs)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fs)

    if x_bins is None:
        x_bins = np.linspace(0, 1, num_bins)

    ax.hist(og_coverage, bins=x_bins, density=True, label=label, alpha=alpha, **hist_kwargs)

    if fit_params is not None:
        mu, sigma = fit_params
        if 'cumulative' not in hist_kwargs:
            y_theory = stats.norm.pdf(x_bins, loc=mu, scale=sigma)
        else:
            y_theory = stats.norm.cdf(x_bins, loc=mu, scale=sigma)
        ax.plot(x_bins, y_theory, label=fit_label)

    if label != '':
        ax.legend(fontsize=legend_fs)


def plot_og_presence_tests(pangenome_map, og_ids_tuple, sag_ids_tuple, savefig):
    # Partition test OGs
    low_diversity_ogs, high_diversity_ogs = og_ids_tuple
    high_diversity_og_cluster_map = partition_ogs_by_clusters(high_diversity_ogs, pangenome_map)
    high_diversity_msogs = high_diversity_og_cluster_map['M']
    low_diversity_og_cluster_map = partition_ogs_by_clusters(low_diversity_ogs, pangenome_map)
    low_diversity_species_cluster_ogs = np.concatenate([low_diversity_og_cluster_map['A,Bp'], low_diversity_og_cluster_map['A,Bp,C']])

    # Set up figure
    nrows = 1
    ncols = 2
    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    gspec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    # MSOG vs pure species clusters in mixed-sample SAGs
    pure_syna_sample_sags, mixed_syna_sample_sags = sag_ids_tuple
    mixed_syna_msog_presence, _ = calculate_og_presence_covariance(pangenome_map, high_diversity_msogs, mixed_syna_sample_sags)
    mixed_syna_species_cluster_presence, _ = calculate_og_presence_covariance(pangenome_map, low_diversity_species_cluster_ogs, mixed_syna_sample_sags)
    group1_og_coverage = np.sum(mixed_syna_msog_presence.values, axis=0) / len(mixed_syna_msog_presence)
    group2_og_coverage = np.sum(mixed_syna_species_cluster_presence.values, axis=0) / len(mixed_syna_species_cluster_presence)
    ks_stat, ks_pvalue = stats.kstest(group1_og_coverage, group2_og_coverage)
    print(f'MSOG vs pure species: {ks_stat:.4f}, {ks_pvalue:.1e}')
    print(np.mean(group1_og_coverage), np.std(group1_og_coverage))
    print(np.mean(group2_og_coverage), np.std(group2_og_coverage))
    print('\n')

    pure_syna_msog_presence, _ = calculate_og_presence_covariance(pangenome_map, high_diversity_msogs, pure_syna_sample_sags)
    pure_syna_species_cluster_presence, _ = calculate_og_presence_covariance(pangenome_map, low_diversity_species_cluster_ogs, pure_syna_sample_sags)
    group3_og_coverage = np.sum(pure_syna_msog_presence.values, axis=0) / len(pure_syna_msog_presence)
    group4_og_coverage = np.sum(pure_syna_species_cluster_presence.values, axis=0) / len(pure_syna_species_cluster_presence)
    ks_stat, ks_pvalue = stats.kstest(group3_og_coverage, group4_og_coverage)
    print(f'MSOG vs pure species (OS05): {ks_stat:.4f}, {ks_pvalue:.1e}')
    print(np.mean(group3_og_coverage), np.std(group3_og_coverage))
    print(np.mean(group4_og_coverage), np.std(group4_og_coverage))
    print('\n\n')

    epsilon = 0.01
    num_bins = 100

    ax = plt.subplot(gspec[0, 0])
    #ymax = 8
    ymax = 1.2
    ax.text(-0.1, 1.1 * ymax, 'A', fontweight='bold', fontsize=16)
    #plot_presence_distribution(ax, group1_og_coverage, alpha=0.5, label='mixed species OGs')
    #plot_presence_distribution(ax, group2_og_coverage, alpha=0.5, label='pure species OGs', label_fs=14, legend_fs=10)
    coverage_values = np.concatenate([group1_og_coverage, group2_og_coverage])
    x_bins = np.linspace(np.min(coverage_values) - epsilon, np.max(coverage_values) + epsilon, num_bins)

    plot_presence_distribution(ax, group1_og_coverage, alpha=0.5, label='mixed species OGs', x_bins=x_bins, cumulative=True, histtype='step', lw=2)
    plot_presence_distribution(ax, group2_og_coverage, alpha=0.5, label='pure species OGs', ylabel='cumulative', label_fs=16, legend_fs=10, x_bins=x_bins, cumulative=True, histtype='step', lw=2)
    #ax.set_xlim(0, 1.1)
    ax.set_ylim(-0.02, ymax)
    ax.set_xlim(x_bins[0] - 0.1, x_bins[-1])
    #ax.set_ylim(0.0, ymax)
    ax.legend(loc='upper left', fontsize=10)

    ax = plt.subplot(gspec[0, 1])
    ymax = 1.2
    ax.text(-0.40, 1.1 * ymax, 'B', fontweight='bold', fontsize=16)
    #z_group1 = group1_og_coverage - np.mean(group2_og_coverage)
    #z_group3 = group3_og_coverage - np.mean(group4_og_coverage)
    mu_group1 = np.mean(group1_og_coverage)
    sigma_group1 = np.std(group1_og_coverage)
    z_group1 = group1_og_coverage - mu_group1
    z_group3 = group3_og_coverage - np.mean(group3_og_coverage)
    z_values = np.concatenate([z_group1, z_group3])
    x_bins = np.linspace(np.min(z_values) - epsilon, np.max(z_values) + epsilon, num_bins)
    plot_presence_distribution(ax, z_group1, x_bins=x_bins, alpha=0.5, label=r'$\alpha$ only', ylabel='cumulative', cumulative=True, histtype='step', lw=2)
    plot_presence_distribution(ax, z_group3, x_bins=x_bins, fit_params=(0, sigma_group1), alpha=0.5, xlabel='mean-centered coverage', ylabel='cumulative', label=r' $\alpha-\beta$ mixed', label_fs=16, legend_fs=10, fit_label=r'$\alpha$ only fit', cumulative=True, histtype='step', lw=2)
    ax.set_xlim(x_bins[0] - 0.1, x_bins[-1])
    ax.set_ylim(-0.02, ymax)
    #ax.set_ylim(0.0, ymax)

    ks_stat, ks_pvalue = stats.kstest(z_group1, z_group3)
    print(f'OS05 vs mixed-species samples MSOG coverage: {ks_stat:.4f}, {ks_pvalue:.1e}')
    utils.print_break()


    plt.tight_layout()
    plt.savefig(savefig)
    plt.close()

 

def partition_ogs_by_clusters(og_ids, pangenome_map):
    og_subtable = pangenome_map.og_table.loc[pangenome_map.og_table['parent_og_id'].isin(og_ids), :]
    sorted_og_ids = {}
    for o in og_ids:
        clusters = ','.join(np.sort(og_subtable.loc[og_subtable['parent_og_id'] == o, 'sequence_cluster'].values))
        if clusters not in sorted_og_ids:
            sorted_og_ids[clusters] = [o]
        else:
            sorted_og_ids[clusters].append(o)
    return sorted_og_ids


def plot_block_distributions_sample_comparisons(input_dir, savefig):
    pure_syna_sample_block_stats = pd.read_csv(f'{input_dir}pure_syna_sample_4D_sites_hybrid_linkage_block_stats.tsv', sep='\t', index_col=0)
    mixed_syna_sample_block_stats = pd.read_csv(f'{input_dir}mixed_syna_sample_4D_sites_hybrid_linkage_block_stats.tsv', sep='\t', index_col=0)
    bootstrap_results = read_bootstrap_results(input_dir)

    # Set up figure
    nrows = 1
    ncols = 3
    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    gspec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    ax = plt.subplot(gspec[0, 0])
    ax_label_hratio = 1.15
    ymax = 0.04
    ax.text(80, ax_label_hratio * ymax, 'A', fontweight='bold', fontsize=14)
    num_blocks_values = bootstrap_results['number of blocks'].values
    x_bins = np.arange(np.min(num_blocks_values) - 1, np.max(num_blocks_values) + 2, 4)
    plot_presence_distribution(ax, num_blocks_values[1:], x_bins=x_bins, alpha=1.0, xlabel='number of blocks', label_fs=14)
    ax.set_ylim(0, ymax)
    ax.annotate('', xy=(num_blocks_values[0], 0.01), xycoords='data', 
            xytext=(num_blocks_values[0], 0.035),
            arrowprops=dict(facecolor='black', width=3, headwidth=8),
            horizontalalignment='center', verticalalignment='top',
            )

    ax = plt.subplot(gspec[0, 1])
    ymax = 0.3
    ax.text(-5, ax_label_hratio * ymax, 'B', fontweight='bold', fontsize=16)
    group1_lengths = pure_syna_sample_block_stats['num_snps'].values
    group2_lengths = mixed_syna_sample_block_stats['num_snps'].values
    length_values = np.concatenate([group1_lengths, group2_lengths])
    x_bins = np.arange(5, np.max(length_values) + 2)
    plot_presence_distribution(ax, group1_lengths, x_bins=x_bins, alpha=0.5, xlabel='block length (SNPs)', ylabel=None, label=r'$\alpha$ only', label_fs=14)
    plot_presence_distribution(ax, group2_lengths, x_bins=x_bins, alpha=0.5, xlabel='block length (SNPs)', ylabel=None, label=r'$\alpha-\beta$ mixed', label_fs=14, legend_fs=8)
    ax.set_xticks(np.arange(5, x_bins[-1], 5))
    ax.set_ylim(0, ymax)

    ks_stat, ks_pvalue = stats.kstest(group1_lengths, group2_lengths)
    print(f'OS05 vs mixed-species samples block lengths: {ks_stat:.4f}, {ks_pvalue:.1e}')
    print(np.mean(group1_lengths), np.std(group1_lengths))
    print(np.mean(group2_lengths), np.std(group2_lengths))
    print('\n')


    ax = plt.subplot(gspec[0, 2])
    ymax = 4 
    ax.text(-0.15, ax_label_hratio * ymax, 'C', fontweight='bold', fontsize=16)
    group1_dS = pure_syna_sample_block_stats['dS_b'].values
    group2_dS = mixed_syna_sample_block_stats['dS_b'].values
    dS_values = np.concatenate([group1_dS, group2_dS])
    epsilon = 0.1
    num_bins = 23
    x_bins = np.linspace(0, np.max(dS_values) + epsilon, num_bins)
    plot_presence_distribution(ax, group1_dS, x_bins=x_bins, alpha=0.5, ylabel=None, label=r'$\alpha$ only', label_fs=14)
    plot_presence_distribution(ax, group2_dS, x_bins=x_bins, alpha=0.5, xlabel='hapl. divergence, $d_S^{(b)}$', ylabel=None, label=r'$\alpha-\beta$ mixed', label_fs=14, legend_fs=8)
    ax.set_xlim(0, 1.2)
    #ax.set_ylim(0, ymax)

    ks_stat, ks_pvalue = stats.kstest(group1_dS, group2_dS)
    print(f'OS05 vs mixed-species samples haplotype divergences: {ks_stat:.4f}, {ks_pvalue:.1e}')
    print(np.mean(group1_dS), np.std(group1_dS))
    print(np.mean(group2_dS), np.std(group2_dS))
    utils.print_break()

    plt.tight_layout()
    plt.savefig(savefig)
    plt.close()


def read_bootstrap_results(input_dir):
    # Read bootstrap results
    num_replicas = 100
    bootstrap_samples_idx = [f'sample{i}' for i in range(1, num_replicas+1)]
    bootstrap_results = pd.DataFrame(index=['test_sample'] + bootstrap_samples_idx, columns=['number of blocks', 'mean block length', 'mean dS_b'])
    pure_syna_sample_block_stats = pd.read_csv(f'{input_dir}pure_syna_sample_4D_sites_hybrid_linkage_block_stats.tsv', sep='\t', index_col=0)
    bootstrap_results.loc['test_sample', 'number of blocks'] = len(pure_syna_sample_block_stats)
    bootstrap_results.loc['test_sample', 'mean block length'] = pure_syna_sample_block_stats['num_snps'].mean()
    bootstrap_results.loc['test_sample', 'mean dS_b'] = pure_syna_sample_block_stats['dS_b'].mean()

    for i in range(1, num_replicas+1):
        sample_block_stats = pd.read_csv(f'{input_dir}mixed_syna_sample{i}_4D_sites_hybrid_linkage_block_stats.tsv', sep='\t', index_col=0)
        bootstrap_results.loc[f'sample{i}', 'number of blocks'] = len(sample_block_stats)
        bootstrap_results.loc[f'sample{i}', 'mean block length'] = sample_block_stats['num_snps'].mean()
        bootstrap_results.loc[f'sample{i}', 'mean dS_b'] = sample_block_stats['dS_b'].mean()

    return bootstrap_results


def make_linkage_block_figures(args, fig_count=19):
    plot_dNdS_scatter(args, fig_count)
    fig_count += 1
    plot_dNdS_histograms(args, fig_count, num_bins=50, p_cutoff=0.70)
    fig_count += 1
    plot_block_length_histogram(args, fig_count, num_bins=50)
    plot_block_diversity_histogram(args, fig_count)
    plot_block_haplotype_frequency_spectrum(args, fig_count)

    return fig_count + 1
    
def plot_dNdS_scatter(args, fig_count):
    fig = plt.figure(figsize=(double_col_width, 0.9 * single_col_width))

    # Analyze main allele divergences
    ax_labels = ['A', 'B']
    for i, species in enumerate(['A', 'Bp']):
        f_haplotypes = f'{args.data_dir}{species}_all_sites_hybrid_linkage_block_stats.tsv'
        block_diversity_stats = pd.read_csv(f_haplotypes, sep='\t', index_col=0)
        block_dNdS = block_diversity_stats[['dN_b', 'dS_b']].dropna()

        ax = fig.add_subplot(1, 2, i + 1)
        ax.set_xlabel('synonymous divergence, $d_S$', fontsize=14)
        ax.set_ylabel('nonsynonymous divergence, $d_N$', fontsize=14)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

        # Get unsaturated blocks
        p_cutoff = 0.6
        dS = align_utils.calculate_divergence(block_dNdS.loc[(block_dNdS['dS_b'] < p_cutoff) & (block_dNdS['dN_b'] < p_cutoff), 'dS_b'].values)
        dN = align_utils.calculate_divergence(block_dNdS.loc[(block_dNdS['dS_b'] < p_cutoff) & (block_dNdS['dN_b'] < p_cutoff), 'dN_b'].values)
        #ax.scatter(dS, dN, s=2**2, alpha=0.1, c='k', label='haplotype divergence')
        ax.scatter(dS, dN, s=2**2, alpha=0.1, c='k')

        dx = np.linspace(0, np.max(dS), 30)
        dN_mean = [np.mean(dN[(dS > dx[i]) & (dS <= dx[i + 1])]) for i in range(len(dx) - 1)]
        ax.plot(dx[:-1], dN_mean, lw=1, c=open_colors['blue'][4], label=r'$\langle d_N \rangle$ vs $d_S$')

        ax.text(-0.2, 1.05 * np.max(dN), ax_labels[i], fontsize=14, fontweight='bold', va='bottom')
        ax.legend(fontsize=10)
        print(len(dS), len(block_dNdS))

        dc = 0.05
        slope, intercept, rvalue, pvalue, _ = stats.linregress(dS[dN > dc], dN[dN > dc])
        print(slope, rvalue, pvalue)

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_block_dNdS_scatter.pdf')
       

def plot_dNdS_histograms(args, fig_count, num_bins=50, p_cutoff=0.6, legend_fs=12):
    fig = plt.figure(figsize=(double_col_width, single_col_width))

    # Analyze main allele divergences
    ax_labels = ['A', 'B']
    for i, species in enumerate(['A', 'Bp']):
        f_haplotypes = f'{args.data_dir}{species}_all_sites_hybrid_linkage_block_stats.tsv'
        block_diversity_stats = pd.read_csv(f_haplotypes, sep='\t', index_col=0)
        block_dNdS = block_diversity_stats[['dN_b', 'dS_b']].dropna()

        ax = fig.add_subplot(1, 2, i + 1)
        ax.set_xlabel('divergence', fontsize=14)
        ax.set_ylabel('counts', fontsize=14)

        # Get unsaturated blocks
        dS = align_utils.calculate_divergence(block_dNdS.loc[(block_dNdS['dS_b'] < p_cutoff) & (block_dNdS['dN_b'] < p_cutoff), 'dS_b'].values)
        dN = align_utils.calculate_divergence(block_dNdS.loc[(block_dNdS['dS_b'] < p_cutoff) & (block_dNdS['dN_b'] < p_cutoff), 'dN_b'].values)

        x_bins = np.linspace(0, max([np.max(dS), np.max(dN)]), num_bins)
        ax.hist(dN, bins=x_bins, alpha=0.6, label=f'$d_N$')
        ax.hist(dS, bins=x_bins, alpha=0.6, label=f'$d_S$')

        # Get max histogram value
        dN_hist, _ = np.histogram(dN, bins=x_bins)
        dS_hist, _ = np.histogram(dS, bins=x_bins)
        ax.text(-0.2, 1.1 * max(np.max(dS_hist), np.max(dN_hist)), ax_labels[i], fontsize=14, fontweight='bold', va='bottom')
        ax.legend(frameon=False, fontsize=legend_fs)

        print(f'{species}, <dN> = {np.mean(dN)}, <dS> = {np.mean(dS)}')

        #print(np.sum(dS < 0.001), len(dS))

    print('\n')

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_block_dNdS_hist.pdf')


def plot_block_length_histogram(args, fig_count, num_bins=50, legend_fs=12):
    fig = plt.figure(figsize=(double_col_width, single_col_width))

    # Analyze main allele divergences
    species_colors = {'A':'tab:orange', 'Bp':'tab:blue'}
    species_labels = {'A':r'$\alpha$', 'Bp':r'$\beta$'}

    ax = fig.add_subplot(121)
    ax.set_xlabel('block length (bp)', fontsize=14)
    ax.set_ylabel('histogram', fontsize=14)
    ax.set_yscale('log')
    x_bins = np.linspace(0, 250, num_bins)
    for i, species in enumerate(['A', 'Bp']):
        f_haplotypes = f'{args.data_dir}{species}_all_sites_hybrid_linkage_block_stats.tsv'
        block_diversity_stats = pd.read_csv(f_haplotypes, sep='\t', index_col=0)
        block_lengths = (block_diversity_stats['x_end'] - block_diversity_stats['x_start'] + 1).values
        ax.hist(block_lengths, bins=x_bins, alpha=0.5, color=species_colors[species], label=species_labels[species], density=True)
        print(species, np.mean(block_lengths), np.std(block_lengths))
    ax.text(-0.2, 1.05, 'A', fontsize=14, fontweight='bold', va='bottom', transform=ax.transAxes)
    ax.legend(frameon=False, fontsize=legend_fs)


    ax = fig.add_subplot(122)
    ax.set_xlabel('block SNPs', fontsize=14)
    ax.set_ylabel('histogram', fontsize=14)
    ax.set_yscale('log')
    x_bins = np.arange(0, 40)
    for i, species in enumerate(['A', 'Bp']):
        f_haplotypes = f'{args.data_dir}{species}_all_sites_hybrid_linkage_block_stats.tsv'
        block_diversity_stats = pd.read_csv(f_haplotypes, sep='\t', index_col=0)
        ax.hist(block_diversity_stats['num_snps'], bins=x_bins, alpha=0.5, color=species_colors[species], label=species_labels[species], density=True)
        print(species, np.mean(block_diversity_stats['num_snps']), np.std(block_diversity_stats['num_snps']), np.sum(block_diversity_stats['num_snps']))
    ax.text(-0.2, 1.05, 'B', fontsize=14, fontweight='bold', va='bottom', transform=ax.transAxes)
    ax.legend(frameon=False, fontsize=legend_fs)


    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_block_length_hist.pdf')


def plot_block_diversity_histogram(args, fig_count, num_bins=30, legend_fs=12):
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))

    # Analyze main allele divergences
    species_colors = {'A':'tab:orange', 'Bp':'tab:blue'}
    species_labels = {'A':r'$\alpha$', 'Bp':r'$\beta$'}
    species_markers = {'A':'o', 'Bp':'s'}

    ax = fig.add_subplot(111)
    ax.set_xlabel(r'block diversity, $\pi$', fontsize=14)
    ax.set_xscale('log')
    ax.set_ylabel(r'reversed cumulative', fontsize=14)
    ax.set_yscale('log')
    ax.set_ylim(1E-4, 1.1)
    x_bins = np.geomspace(1E-4, 2E-1, num_bins)
    for i, species in enumerate(['A', 'Bp']):
        f_haplotypes = f'{args.data_dir}{species}_all_sites_hybrid_linkage_block_stats.tsv'
        block_diversity_stats = pd.read_csv(f_haplotypes, sep='\t', index_col=0)
        pi_values = np.concatenate(block_diversity_stats[['pi1', 'pi2']].values.T)
        y = np.array([np.sum(pi_values > x) / len(pi_values) for x in x_bins])
        #ax.hist(pi_values + 1E-5, bins=x_bins, alpha=0.5, color=species_colors[species], label=species_labels[species], density=True)
        ax.plot(x_bins, y, f'-{species_markers[species]}', lw=1, ms=5, color=species_colors[species], mfc='none', mec=species_colors[species], label=species_labels[species])
        print(species, np.sum(pi_values > 1E-5), len(pi_values))
    ax.legend(frameon=False, fontsize=legend_fs)

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_block_diversity_hist.pdf')


def plot_block_haplotype_frequency_spectrum(args, fig_count, num_bins=50, label_fs=14, legend_fs=10):
    color_dict = {'A':'tab:orange', 'Bp':'tab:blue'}
    label_dict = {'A':r'$\alpha$', 'Bp':r'$\beta$'}

    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('sample depth, $n$', fontsize=label_fs)
    ax1.set_ylabel('histogram', fontsize=label_fs)
    ax1.text(-0.05, 1.05, 'A', fontsize=14, fontweight='bold', va='bottom', transform=ax1.transAxes)

    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('minor allele frequency, $f_m$', fontsize=label_fs)
    ax2.set_ylabel('histogram', fontsize=label_fs)
    ax2.set_ylim(0, 6)
    ax2.text(-0.05, 1.05, 'B', fontsize=14, fontweight='bold', va='bottom', transform=ax2.transAxes)

    n_bins = np.linspace(0, 110, num_bins)
    f_bins = np.linspace(0, 0.5, 20)
    for i, species in enumerate(['A', 'Bp']):
        f_block_stats = f'{args.data_dir}{species}_all_sites_hybrid_linkage_block_stats.tsv'
        species_block_stats = pd.read_csv(f_block_stats, sep='\t')
        species_block_stats['major_allele_freq'] = species_block_stats['haplotype_frequencies'].str.split(';').str[0].astype(int)
        species_block_stats['minor_allele_freq'] = species_block_stats['haplotype_frequencies'].str.split(';').str[1].astype(int)
        species_block_stats['num_seqs'] = species_block_stats[['major_allele_freq', 'minor_allele_freq']].sum(axis=1)
        ax1.hist(species_block_stats['num_seqs'], bins=n_bins, color=color_dict[species], alpha=0.5, label=label_dict[species])

        n_avg = np.mean(species_block_stats['num_seqs'])
        n_std = np.sqrt(n_avg)
        ax1.axvline(n_avg - n_std, lw=2, c=color_dict[species])
        ax1.axvline(n_avg + n_std, lw=2, c=color_dict[species])
        #filtered_idx = species_block_stats.index.values
        filtered_idx = species_block_stats.index.values[(species_block_stats['num_seqs'] > n_avg - n_std) & (species_block_stats['num_seqs'] < n_avg + n_std)]
        print(len(species_block_stats), len(filtered_idx))
        maf = species_block_stats.loc[filtered_idx, 'minor_allele_freq'] / species_block_stats.loc[filtered_idx, 'num_seqs']
        ax2.hist(maf, bins=f_bins, color=color_dict[species], alpha=0.5, label=label_dict[species], density=True)

    f_arr = np.array([np.mean(f_bins[i:i+2]) for i in range(len(f_bins) - 1)])
    #ax2.plot(f_bins[1:], 6 * f_bins[1] / f_bins[1:], '-o', c='k')
    #ax2.plot(f_arr, 0.4 / f_arr, '-o', c='k')
    f0 = np.mean(f_bins[0:2])
    f_arr = np.linspace(f0, 1 - f0, 40)
    folded_sfs = 1. / f_arr + 1. / f_arr[::-1]
    ax2.plot(f_arr[:20], 0.35 * folded_sfs[:20], '-o', c='k', label='neutral model')

    ax1.legend(fontsize=legend_fs, frameon=False)
    ax2.legend(fontsize=legend_fs, frameon=False)
    
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_block_frequencies.pdf')


def make_synteny_figures(args, fig_count=20):
    color_dict = {'A':'tab:orange', 'Bp':'tab:blue'}
    ref_dict = {'A':'CP000239', 'Bp':'CP000240'}

    # Read SAG IDs and sort by species
    f_sags = f'{args.results_dir}supplement/synteny_analysis_sag_ids.txt'
    sag_ids = np.loadtxt(f_sags, dtype=str)
    metadata = MetadataMap()
    species_sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')

    fig = plt.figure(figsize=(double_col_width, 2.0 * single_col_width))
    for i, species in enumerate(['A', 'Bp']):
        ax = fig.add_subplot(2, 1, i + 1)

        results_dict = {}
        for sag_id in species_sorted_sag_ids[species]:
            f_blast = f'{args.results_dir}blast_alignment/{sag_id}_whole_contig_blast_results.tab'
            raw_results = seq_utils.read_blast_results(f_blast)
            blast_results = raw_results.loc[raw_results['sseqid'] == ref_dict[species], :].copy()
            results_dict[sag_id] = blast_results

        if species == 'A':
            xlabel = 'OS-A genome position (kbp)'
            color = 'tab:orange'
        elif species == 'Bp':
            xlabel = "OS-B' genome position (kbp)"
            color = 'tab:blue'
        #plot_multisag_synteny_color_map(results_dict, species, f'{args.figures_dir}S{fig_count}_{species}_synteny_color_map.pdf', lw=28)
        plot_multisag_synteny_color_map(results_dict, species, ax=ax, xlabel=xlabel, lw=20)

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_synteny_color_map.pdf')
    plt.close()


def plot_multisag_synteny_color_map(blast_results_dict, sag_species, ax=None, savefig=None, xlabel='ref genome location (kbp)', cmap='plasma', xscale=1000, num_bins=1000, min_length=1000, num_bins_segment=10, lw=12):
    # Set up axes
    if ax is None:
        fig = plt.figure(figsize=(double_col_width, 1.5 * single_col_width))
        ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Plot reference genome
    if sag_species == 'A':
        x = np.linspace(1, utils.osa_genome_size / xscale, num_bins)
        ref_name = "OS-A"
    else:
        x = np.linspace(1, utils.osbp_genome_size / xscale, num_bins)
        ref_name = "OS-B'"
    y_ref = np.zeros(len(x))
    y0 = 0

    norm = plt.Normalize(x.min(), x.max())
    lc = make_colored_line_collection(x, y_ref, norm, values=x, lw=lw)
    ref_line = ax.add_collection(lc)

    yticklabels = [ref_name]
    for sag_id in blast_results_dict:
        blast_results = blast_results_dict[sag_id]
        plot_color_map_queries(ax, blast_results, norm, y0=y0, lw=lw, xscale=xscale, num_bins_segment=num_bins_segment)
        #sag_gold_id = blast_results['qseqid'].str.split('_').str[0].unique()[0]
        yticklabels.append(utils.strip_sample_name(sag_id, replace=True))
        y0 += 1

    ax.set_xlim(-0.05 * x.max(), 1.05 * x.max())
    ax.set_ylim(-0.5, y0 + 0.5)
    ax.set_yticks(np.arange(y0 + 1))
    ax.set_yticklabels(yticklabels)

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()


def plot_synteny_color_map(blast_results, sag_species, savefig, xlabel='ref genome location (kbp)', cmap='plasma', xscale=1000, num_bins=1000, min_length=1000, num_bins_segment=10, lw=12):
    # Set up axes
    fig = plt.figure(figsize=(double_col_width, 0.5 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel, fontsize=14)
    #ax.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Plot reference genome
    if sag_species == 'A':
        x = np.linspace(1, utils.osa_genome_size / xscale, num_bins)
        ref_name = "OS-A"
    else:
        x = np.linspace(1, utils.osbp_genome_size / xscale, num_bins)
        ref_name = "OS-B'"
    #x = np.linspace(5E5, 1E6, num_bins)
    y_ref = np.zeros(len(x))

    norm = plt.Normalize(x.min(), x.max())
    lc = make_colored_line_collection(x, y_ref, norm, values=x, lw=lw)
    ref_line = ax.add_collection(lc)

    # Plot contigs
    '''
    queries = np.unique(blast_results['qseqid'].values)
    for q in queries:
        q_results = blast_results.loc[blast_results['qseqid'] == q, :].copy()
        #q_results['mapped_length'] = np.abs(q_results['qend'] - q_results['qstart'])
        #q_results = q_results.sort_values('mapped_length', ascending=False)
        #approximate_size = np.max(q_results[['qstart', 'qend']].values) - np.min(q_results[['qstart', 'qend']].values) + 1
        #q_results['cumulative_mapped'] = [np.sum(q_results['mapped_length'].values[:i]) / approximate_size for i in range(1, len(q_results) + 1)]
        #filtered_idx = (q_results['cumulative_mapped'] < 0.95)
        q_results['segment_length'] = np.abs(q_results['qend'] - q_results['qstart']) + 1
        #filtered_idx = (q_results['evalue'] < 1E-180) & (q_results['segment_length'] > min_length)
        filtered_idx = (q_results['evalue'] < 1E-100)
        x0 = np.mean(q_results.loc[filtered_idx, ['sstart', 'send']].values)
        L = np.max(q_results[['qstart', 'qend']].values) - np.min(q_results[['qstart', 'qend']].values) + 1

        for i, row in q_results.loc[filtered_idx, :].iterrows():
            xq0 = row['qstart'] - L / 2 + x0
            xq1 = row['qend'] - L / 2 + x0
            xq = np.linspace(xq0, xq1, num_bins_segment) / xscale
            xs = np.linspace(row['sstart'], row['send'], num_bins_segment) / xscale
            yq = np.ones(num_bins_segment)

            lc = make_colored_line_collection(xq, yq, norm, values=xs, lw=lw)
            ax.add_collection(lc)
    '''
    plot_color_map_queries(ax, blast_results, norm, lw=lw, xscale=xscale, num_bins_segment=num_bins_segment)

    #fig.colorbar(ref_line, ax=ax)
    ax.set_xlim(-0.05 * x.max(), 1.05 * x.max())
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])

    sag_gold_id = blast_results['qseqid'].str.split('_').str[0].unique()[0]
    ax.set_yticklabels([ref_name, sag_gold_id])

    plt.tight_layout()
    plt.savefig(savefig)
    plt.close()


def plot_color_map_queries(ax, blast_results, norm, y0=0, lw=12, xscale=1000, num_bins_segment=10, evalue_cutoff=1E-100):
    queries = np.unique(blast_results['qseqid'].values)
    x_min = 1000000
    x_max = 0
    for q in queries:
        q_results = blast_results.loc[blast_results['qseqid'] == q, :].copy()
        q_results['segment_length'] = np.abs(q_results['qend'] - q_results['qstart']) + 1
        filtered_idx = (q_results['evalue'] < evalue_cutoff)
        #x0 = np.mean(q_results.loc[filtered_idx, ['sstart', 'send']].values)
        x0 = calculate_median_mapped_location(q_results.loc[filtered_idx, ['sstart', 'send']].values)
        #print(x0, q_results['segment_length'].sum())
        #print('\n\n')
        L = np.max(q_results[['qstart', 'qend']].values) - np.min(q_results[['qstart', 'qend']].values) + 1

        for i, row in q_results.loc[filtered_idx, :].iterrows():
            xq0 = row['qstart'] - L / 2 + x0
            xq1 = row['qend'] - L / 2 + x0
            xq = np.linspace(xq0, xq1, num_bins_segment) / xscale
            xs = np.linspace(row['sstart'], row['send'], num_bins_segment) / xscale
            yq = y0 + np.ones(num_bins_segment)

            lc = make_colored_line_collection(xq, yq, norm, values=xs, lw=lw)
            ax.add_collection(lc)

            if xq0 < x_min:
                x_min = xq0
            if xq1 > x_max:
                x_max = xq1

    print(x_min, x_max)

def calculate_median_mapped_location(mapped_segments):
    # Get mapped segment coordinates
    temp = []
    for s in mapped_segments:
        step = (-1)**(s[1] < s[0]) # set direction of segment along refernce
        mapped_values = np.arange(s[0], s[1] + step, step)
        temp.append(mapped_values)
    x_mapped_values = np.sort(np.concatenate(temp))
    return np.median(x_mapped_values)


def make_colored_line_collection(x, y, norm, values=None, cmap='plasma', lw=12):
    '''
    Copied from Matplotlib website.
    '''
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    lc = LineCollection(segments, cmap=cmap, norm=norm)

    # Set the values used for colormapping
    if values is None:
        lc.set_array(x)
    else:
        lc.set_array(values)
    lc.set_linewidth(lw)

    return lc


###########################################################
# Genetic diversity analysis
###########################################################

def make_genetic_diversity_figures(pangenome_map, args, fig_count, low_diversity_cutoff=0.05):
    metadata = MetadataMap()
    alignments_dir = '../results/single-cell/alignments/core_ogs_cleaned_4D_sites/'

    # Get single-site statistics tables
    #syna_num_site_alleles, syna_mutation_spectra = get_single_site_statistics(pangenome_map, metadata, alignments_dir, species='A', main_cloud=False)
    syna_num_site_alleles, syna_mutation_spectra = get_single_site_statistics(pangenome_map, metadata, alignments_dir, args, species='A', main_cloud=False)
    low_diversity_ogs = np.array(syna_num_site_alleles.loc[syna_num_site_alleles['fraction_polymorphic'] < low_diversity_cutoff, :].index)
    high_diversity_ogs = np.array([o for o in syna_num_site_alleles.index if o not in low_diversity_ogs])
    syna_num_site_alleles['num_snps'] = syna_num_site_alleles[['2', '3', '4']].sum(axis=1)

    syna_mean_diversity = syna_num_site_alleles.loc[low_diversity_ogs, "piS"].mean()

    print(f'Low-diversity OGs: {syna_num_site_alleles.loc[low_diversity_ogs, "num_snps"].sum():.0f} 4D SNPs; {syna_num_site_alleles.loc[low_diversity_ogs, "L"].sum():.0f} 4D sites; piS = {syna_num_site_alleles.loc[low_diversity_ogs, "piS"].mean()}; {len(low_diversity_ogs)} loci')
    print(f'High-diversity OGs: {syna_num_site_alleles.loc[high_diversity_ogs, "num_snps"].sum():.0f} 4D SNPs; {syna_num_site_alleles.loc[high_diversity_ogs, "L"].sum():.0f} 4D sites; piS = {syna_num_site_alleles.loc[high_diversity_ogs, "piS"].mean()}; {len(high_diversity_ogs)} loci')

    #syna_num_all_site_alleles, syna_all_site_mutation_spectra = get_single_site_statistics(pangenome_map, metadata, f'{args.results_dir}alignments/core_ogs_cleaned/', species='A', main_cloud=False, sites='all_sites', aln_ext='cleaned_aln.fna')
    syna_num_all_site_alleles, syna_all_site_mutation_spectra = get_single_site_statistics(pangenome_map, metadata, f'{args.results_dir}alignments/core_ogs_cleaned/', args, species='A', main_cloud=False, sites='all_sites', aln_ext='cleaned_aln.fna')
    syna_num_all_site_alleles['num_snps'] = syna_num_all_site_alleles[['2', '3', '4']].sum(axis=1)

    f_block_stats = f'{args.results_dir}main_figures_data/A_all_sites_hybrid_linkage_block_stats.tsv'
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

    plot_genomic_trench_diversity(syna_num_site_alleles, synbp_num_site_alleles, low_diversity_ogs, args, rng)

    print(f'Beta OGs: {synbp_num_site_alleles["num_snps"].sum()} 4D SNPs; {synbp_num_site_alleles["L"].sum()} 4D sites; piS = {synbp_num_site_alleles["piS"].mean()}; {len(synbp_num_site_alleles)} loci')
    synbp_mean_diversity = synbp_num_site_alleles["piS"].mean()

    #synbp_num_all_site_alleles, synbp_all_site_mutation_spectra = get_single_site_statistics(pangenome_map, metadata, f'{args.results_dir}alignments/core_ogs_cleaned/', species='Bp', main_cloud=False, sites='all_sites', aln_ext='cleaned_aln.fna')
    synbp_num_all_site_alleles, synbp_all_site_mutation_spectra = get_single_site_statistics(pangenome_map, metadata, f'{args.results_dir}alignments/core_ogs_cleaned/', args, species='Bp', main_cloud=False, sites='all_sites', aln_ext='cleaned_aln.fna')
    synbp_num_all_site_alleles['num_snps'] = synbp_num_all_site_alleles[['2', '3', '4']].sum(axis=1)

    f_block_stats = f'{args.results_dir}main_figures_data/Bp_all_sites_hybrid_linkage_block_stats.tsv'
    block_diversity_stats = pd.read_csv(f_block_stats, sep='\t')
    num_total_snps = synbp_num_all_site_alleles.loc[:, "num_snps"].sum()
    num_block_snps = block_diversity_stats.loc[block_diversity_stats.index.values[np.isin(block_diversity_stats['og_id'].values, synbp_num_all_site_alleles.index.values)], 'num_snps'].sum()   
    print(f'\t{num_total_snps:.0f} SNPs; {num_block_snps:.0f} SNPs in blocks; {synbp_num_all_site_alleles.loc[:, "L"].sum():.0f} sites; fraction of block SNPs {num_block_snps / num_total_snps:.3f}')

    plot_hybrid_gene_diversity(pangenome_map, metadata, syna_num_site_alleles, syna_mean_diversity, synbp_num_site_alleles, synbp_mean_diversity, rng, args)


    print('\n\n')

    plot_alpha_spring_low_diversity(pangenome_map, metadata, low_diversity_ogs, args, rng, num_bins=30, legend_fs=8)


    plot_gamma_alignment_results(pangenome_map, metadata, fig_count, args)

    return fig_count + 1



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
                    aln_main_cloud = read_main_cloud_alignment(f_aln, pangenome_map, metadata)
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


def get_single_site_statistics_v1(pangenome_map, metadata, alignments_dir, hybridization_dir='../results/single-cell/hybridization/', main_cloud=False):
    f_num_site_alleles = f'{args.output_dir}syna_num_site_alleles_4D_all_seqs.tsv'
    f_mutation_spectra = f'{args.output_dir}syna_mutation_spectra_4D_all_seqs.tsv'

    if os.path.exists(f_num_site_alleles) and os.path.exists(f_mutation_spectra):
        num_site_alleles = pd.read_csv(f_num_site_alleles, sep='\t', index_col=0)
        mutation_spectra = pd.read_csv(f_mutation_spectra, sep='\t', index_col=0)
    else:
        species_cluster_genomes = pd.read_csv(f'{hybridization_dir}sscs_labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)
        temp = np.array(species_cluster_genomes.loc[species_cluster_genomes['core_A'] == 'Yes', :].index)
        syna_core_ogs = temp[['rRNA' not in o for o in temp]]
        species_sorted_sags = metadata.sort_sags(pangenome_map.get_sag_ids(), by='species')
        syna_sag_ids = species_sorted_sags['A']

        num_site_alleles = pd.DataFrame(index=syna_core_ogs, columns=['1', '2', '3', '4', 'n'])
        mutation_spectra = pd.DataFrame(index=syna_core_ogs, columns=['A', 'C', 'G', 'T', 'A<>C', 'A<>G', 'A<>T', 'C<>G', 'C<>T', 'G<>T'])
        min_num_seqs = 20
        for o in syna_core_ogs:
            f_aln = f'{alignments_dir}{o}_4D_aln.fna'
            if os.path.exists(f_aln):
                if main_cloud == False:
                    syna_aln = pangenome_map.read_sags_og_alignment(f_aln, o, syna_sag_ids)
                else:
                    aln_main_cloud = read_main_cloud_alignment(f_aln, pangenome_map, metadata)
                    filtered_gene_ids = pangenome_map.get_og_gene_ids(o, sag_ids=syna_sag_ids)
                    syna_aln = align_utils.get_subsample_alignment(aln_main_cloud, filtered_gene_ids)
                if len(syna_aln) > min_num_seqs:
                    num_site_alleles.loc[o, ['1', '2', '3', '4']] = calculate_site_alleles_histogram(syna_aln)
                    num_site_alleles.loc[o, 'n'] = len(syna_aln)
                    mutation_spectra.loc[o, :] = calculate_allele_mutation_frequencies(syna_aln)
        num_site_alleles['L'] = num_site_alleles[['1', '2', '3', '4']].sum(axis=1)
        num_site_alleles['fraction_polymorphic'] = num_site_alleles[['2', '3', '4']].sum(axis=1) / num_site_alleles['L']

        # Add synonymous diversity
        piS, piS_idx = calculate_synonymous_diversity(num_site_alleles.index.values, syna_sag_ids, pangenome_map, args)
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
    ax.set_xticks([1E-5, 1E-4, 1E-3, 1E-2, 1E-1])

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
            y_null, x_null = generate_polymorphic_sites_null(low_diversity_ogs, syna_sag_ids, alignments_dir, theta, rng, x_bins=x_bins_inset)
            ax_inset.plot(x_null, y_null, '-o', c=null_color, ms=ms, lw=1)

    ax.legend(fontsize=10, frameon=False)



def generate_polymorphic_sites_null(og_ids, sag_ids, alignments_dir, theta, rng, x_bins=None):
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


def plot_alpha_loci_diversity(ax, num_site_alleles, low_diversity_ogs, high_diversity_ogs, epsilon=1E-5, xlim=(9E-6, 0.25), num_bins=50, legend=True):
    ax.set_xscale('log')
    #ax.set_yscale('log')
    #x_bins = np.linspace(*xlim, num_bins)
    x_bins = np.geomspace(*xlim, num_bins)
    ax.hist(num_site_alleles.loc[low_diversity_ogs, 'piS'] + epsilon, bins=x_bins, density=False, label='non-hybrid loci', alpha=0.5)
    ax.hist(num_site_alleles.loc[high_diversity_ogs, 'piS'] + epsilon, bins=x_bins, density=False, label='hybrid loci', alpha=0.5)

    if legend:
        ax.legend(fontsize=10, frameon=False)


def plot_genomic_trench_diversity(syna_num_site_alleles, synbp_num_site_alleles, control_loci, args, rng, num_bins=20, label_fs=14, epsilon=1E-5):
    genomic_troughs_df = pd.read_csv(f'{args.results_dir}hybridization/genomic_trench_loci_annotations.tsv', sep='\t', index_col=0)
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
    ax.set_xscale('log')
    ax.set_xlabel('synonymous diversity, $\pi_S$', fontsize=label_fs)
    ax.set_xticks([1E-5, 1E-4, 1E-3, 1E-2, 1E-1])
    ax.set_ylabel('counts', fontsize=label_fs)

    syna_gt_idx = [g for g in genomic_troughs_df.index if g in syna_num_site_alleles.index.values]
    syna_control_og_list = [g for g in control_loci if (g not in syna_gt_idx) and (g in syna_num_site_alleles.index.values)]
    syna_control_idx = rng.choice(syna_control_og_list, size=len(syna_gt_idx), replace=False)

    x_bins = np.geomspace(1E-5, 1, num_bins)
    ax.hist(syna_num_site_alleles.loc[syna_gt_idx, 'piS'] + epsilon, bins=x_bins, color='tab:purple', alpha=0.5, label=r'$\alpha$ troughs')
    ax.hist(syna_num_site_alleles.loc[syna_control_idx, 'piS'] + epsilon, bins=x_bins, color='tab:orange', alpha=0.5, label=r'$\alpha$ non-hybrid')
    ax.legend(loc='upper left', frameon=False)

    print(syna_num_site_alleles.loc[syna_gt_idx, 'piS'].mean(), syna_num_site_alleles.loc[syna_control_idx, 'piS'].mean())

    synbp_gt_idx = [g for g in genomic_troughs_df.index if g in synbp_num_site_alleles.index.values]
    synbp_control_og_list = [g for g in synbp_num_site_alleles.index.values if g not in syna_gt_idx]
    synbp_control_idx = rng.choice(synbp_control_og_list, size=len(synbp_gt_idx), replace=False)

    ax = fig.add_subplot(122)
    ax.set_xscale('log')
    ax.set_xlabel('synonymous diversity, $\pi_S$', fontsize=label_fs)
    ax.set_xticks([1E-5, 1E-4, 1E-3, 1E-2, 1E-1])
    ax.set_ylabel('counts', fontsize=label_fs)
    ax.hist(synbp_num_site_alleles.loc[synbp_gt_idx, 'piS'] + epsilon, bins=x_bins, color='tab:purple', alpha=0.5, label=r'$\beta$ troughs')
    ax.hist(synbp_num_site_alleles.loc[synbp_control_idx, 'piS'] + epsilon, bins=x_bins, color='tab:blue', alpha=0.5, label=r'$\beta$ core')
    ax.legend(loc='upper left', frameon=False)

    print(synbp_num_site_alleles.loc[synbp_gt_idx, 'piS'].mean(), synbp_num_site_alleles.loc[synbp_control_idx, 'piS'].mean())

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_genomic_trough_diversity.pdf')
    plt.close()


def plot_hybrid_gene_diversity(pangenome_map, metadata, syna_num_site_alleles, syna_mean_diversity, synbp_num_site_alleles, synbp_mean_diversity, rng, args, x0=0, dx1=0.25, dx2=0.2):
    species_cluster_genomes = pd.read_csv(f'{args.results_dir}hybridization/sscs_labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)
    syna_hybrid_donor_frequency_table = main_figs.make_donor_frequency_table(species_cluster_genomes, 'A', pangenome_map, metadata)
    synbp_hybrid_donor_frequency_table = main_figs.make_donor_frequency_table(species_cluster_genomes, 'Bp', pangenome_map, metadata)

    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    print('\n\n')

    label_fs = 14
    colors_dict = {'A':'tab:orange', 'Bp':'tab:blue', 'C':'tab:green', 'O':'tab:purple'}

    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(121)
    ax.set_xticks([0, 1, 2])
    #ax.set_xticklabels([r'\beta', r'\gamma', 'X'])
    ax.set_xticklabels([r'Bp', r'C', 'X'], fontsize=label_fs)
    ax.set_ylabel(r'synonymous diversity, $\pi_S$', fontsize=label_fs)
    ax.set_ylim(8E-6, 1)
    ax.set_yscale('log')
    ax.axhline(syna_mean_diversity, lw=2, color=colors_dict['A'], alpha=0.5)

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
            y = np.mean(pS_values) + 1E-5
            x = np.array([x0, x0 + dx1 / 2 + dx2]) + rng.uniform(-dx1 / 2, dx1 / 2, size=2)
            ax.scatter(x[0], y, 16, marker='o', color=colors_dict[hybrid_cluster])
            if (g in synbp_num_site_alleles.index.values) and (hybrid_cluster == 'Bp'):
                #print(g, np.mean(pS_values), synbp_num_site_alleles.loc[g, 'piS'], len(aln_hybrids))
                yc = synbp_num_site_alleles.loc[g, 'piS']
                ax.scatter(x[1], yc, 16, marker='s', color=colors_dict[hybrid_cluster])
                ax.plot(x, [y, yc], c='k', lw=0.75, alpha=0.5)
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
    ax.set_yticks([1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1])

    counter = np.array([0, 0])
    ax = fig.add_subplot(122)
    ax.set_xticks([0, 1, 2])
    #ax.set_xticklabels([r'\beta', r'\gamma', 'X'])
    ax.set_xticklabels([r'A', r'C', 'X'], fontsize=label_fs)
    ax.set_ylabel(r'synonymous diversity, $\pi_S$', fontsize=label_fs)
    ax.set_ylim(8E-6, 1)
    ax.set_yscale('log')
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
            y = np.mean(pS_values) + 1E-5
            x = np.array([x0, x0 + dx1 / 2 + dx2]) + rng.uniform(-dx1 / 2, dx1 / 2, size=2)
            ax.scatter(x[0], y, 16, marker='o', color=colors_dict[hybrid_cluster])
            if (g in syna_num_site_alleles.index.values) and (hybrid_cluster == 'A'):
                #print(g, np.mean(pS_values), syna_num_site_alleles.loc[g, 'piS'], len(aln_hybrids))
                yc = syna_num_site_alleles.loc[g, 'piS']
                ax.scatter(x[1], yc, 16, marker='s', color=colors_dict[hybrid_cluster])
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
    ax.set_yticks([1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1])

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_hybrid_gene_diversity.pdf')
    plt.close()


def plot_alpha_spring_low_diversity(pangenome_map, metadata, og_ids, args, rng, label_fs=14, num_bins=50, legend_fs=10, ms=4):
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

    xlim = (1E-6, 1E-2)
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('mean pair divergence, $\pi_{ij}$', fontsize=label_fs)
    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ax.set_ylabel('cumulative', fontsize=label_fs)
    ax.set_yscale('log')
    ax.set_ylim(8E-1, len(syna_sag_ids)**2)

    x_idx, y_idx = np.where(pdist_mean < 3E-4)
    #for i in range(len(x_idx)):
    #    print(syna_sag_ids[x_idx[i]], syna_sag_ids[y_idx[i]], pdist_mean.loc[syna_sag_ids[x_idx[i]], syna_sag_ids[y_idx[i]]])

    pdist_values = utils.get_matrix_triangle_values(pdist_mean.values, k=1)
    x_bins = np.geomspace(*xlim, num_bins)
    #y = np.array([np.sum(pdist_values > x) / len(pdist_values) for x in x_bins])
    #y = np.array([np.sum(pdist_values <= x) for x in x_bins])
    #ax.plot(x_bins, y, f'-o', lw=1, ms=3, color='tab:gray', alpha=0.5, mfc='none', label=r'all $\alpha$')
    print('all alpha', np.nanmean(pdist_values))

    spring_colors = {'OS':'tab:cyan', 'MS':'tab:purple'}
    for spring in syna_sorted_sag_ids:
        spring_sag_ids = syna_sorted_sag_ids[spring]
        pdist_values = utils.get_matrix_triangle_values(pdist_mean.loc[spring_sag_ids, spring_sag_ids].values, k=1)
        #y = np.array([np.sum(pdist_values > x) / len(pdist_values) for x in x_bins])
        y = np.array([np.sum(pdist_values <= x) for x in x_bins])
        n = len(syna_sorted_sag_ids[spring])
        ax.plot(x_bins, y, f'-s', lw=1, ms=ms, color=spring_colors[spring], alpha=0.5, mfc='none', label=f'{spring} (n={n:d})')

        if spring == 'OS':
            subsampled_sag_ids = rng.choice(spring_sag_ids, size=len(syna_sorted_sag_ids['MS']))
            pdist_values = utils.get_matrix_triangle_values(pdist_mean.loc[subsampled_sag_ids, subsampled_sag_ids].values, k=1)
            #y = np.array([np.sum(pdist_values > x) / len(pdist_values) for x in x_bins])
            y = np.array([np.sum(pdist_values <= x) for x in x_bins])
            ax.plot(x_bins, y, f'-^', lw=1, ms=ms, color=spring_colors[spring], alpha=0.5, mfc='none', label=f'{spring} subsampled')

        print(spring, np.nanmean(pdist_values))

    # Between springs comparison
    pdist_values = pdist_mean.loc[syna_sorted_sag_ids['OS'],syna_sorted_sag_ids['MS']].values.flatten()
    #y = np.array([np.sum(pdist_values > x) / len(pdist_values) for x in x_bins])
    y = np.array([np.sum(pdist_values <= x) for x in x_bins])
    ax.plot(x_bins, y, f'-D', lw=1, ms=3, color='tab:red', alpha=0.5, mfc='none', label='OS vs MS')
    print('OS vs MS', np.nanmean(pdist_values))

    ax.legend(frameon=False, fontsize=legend_fs)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_alpha_low_diversity_location_cumul.pdf')
    plt.close()

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


def plot_gamma_alignment_results(pangenome_map, metadata, fig_count, args, num_bins=50):
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


###########################################################
# Genetic diversity analysis
###########################################################


def make_metagenome_recruitment_figures(pangenome_map, args, fig_count, num_loci=100):

    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    species_abundances = main_figs.read_species_abundances(args, num_loci)
    species_abundances.loc[['A', 'Bp', 'C'], :] = species_abundances.loc[['A', 'Bp', 'C'], :] / species_abundances.loc['total_average_depth', :]
    plot_species_frequency_across_samples(ax, species_abundances, fig, ms=6, ax_label='')
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_species_abundances.pdf')
    plt.close()
    fig_count += 1


    fig_count = plot_correlation_across_loci(num_loci, args, fig_count)
    '''
    f_recruitment = f'{args.metagenome_dir}{species}_Allele_pident95.xlsx'
    alleles_df = pd.read_excel(f_recruitment, sheet_name=0, index_col=0)
    sample_ids = np.array([c for c in alleles_df.columns if 'Hot' in c])
    alleles_df['average_depth'] = [np.nan, np.nan] + [np.nanmean(row.astype(float)) for row in alleles_df[sample_ids].values[2:]]
    alleles_df = alleles_df.sort_values('average_depth', ascending=False)
    loci_depth_df = alleles_df.loc[alleles_df.index[:-2], np.concatenate([sample_ids, ['average_depth']])]
    plot_abundant_target_counts(loci_depth_df, f'{args.figures_dir}{species}_loci_depths_across_samples.pdf', ylabel='read depth', num_targets=num_loci, lw=0.75, alpha=0.5)
    '''

    return fig_count 


def plot_correlation_across_loci(num_loci, args, fig_count):
    species_labels = {'A':r'$\alpha$', 'B':r'$\beta$', 'C':r'$\gamma$'}

    fig = plt.figure(figsize=(double_col_width, 2.4 * single_col_width))
    for i, species in enumerate(['A', 'B', 'C']):
        ax = fig.add_subplot(3, 1, i + 1)
        ax.set_title(f'{species_labels[species]} reference', fontsize=14)
        f_recruitment = f'{args.metagenome_dir}{species}_Allele_pident95.xlsx'
        alleles_df = pd.read_excel(f_recruitment, sheet_name=0, index_col=0)
        sample_ids = np.array([c for c in alleles_df.columns if 'Hot' in c])
        alleles_df['average_depth'] = [np.nan, np.nan] + [np.nanmean(row.astype(float)) for row in alleles_df[sample_ids].values[2:]]
        alleles_df = alleles_df.sort_values('average_depth', ascending=False)
        loci_depth_df = alleles_df.loc[alleles_df.index[:-2], np.concatenate([sample_ids, ['average_depth']])]
        plot_abundant_target_counts(ax, loci_depth_df, ylabel='read depth', num_targets=num_loci, lw=0.75, alpha=0.5)
        ax.set_yticks([1, 1E1, 1E2, 1E3])

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_read_depth_correlation.pdf')
    plt.close()
    return fig_count + 1

def plot_abundant_target_counts(ax, abundance_df, ylabel='reads recruited', num_targets=10, lw=1.0, alpha=1.0, yticks=[1, 1E1, 1E2, 1E3]):

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

    sample_columns = abundance_df.columns[:-1]
    x_labels = [strip_sample_id(sid) for sid in sample_columns]
    x = np.arange(len(sample_columns))

    ax.set_xticks(x)
    ax.set_xticklabels([x_labels[i] for i in x], fontsize=8, rotation=90)
    ax.set_ylabel(ylabel)
    ax.set_yticks(yticks)
    ax.set_yscale('log')
    ax.set_ylim(9E-1, 1.5 * np.nanmax(abundance_df[sample_columns].values))
    for i in range(num_targets):
        ax.plot(x, abundance_df.iloc[i, :-1], lw=lw, alpha=alpha, label=strip_target_id(abundance_df.index[i]))
    #ax.legend()



def plot_species_frequency_across_samples(ax, species_abundances, fig, lw=0.8, tick_size=12, label_size=14, legend_size=10, ms=4, sag_samples=['HotsprSampleMSe4', 'HotsprSampleOSM3'], clean_borders=True, ax_label=None, yticks=[1E-4, 1E-3, 1E-2, 1E-1, 1], num_loci=100):
    sample_relative_abundances = main_figs.process_species_abundance_table(species_abundances)
    metagenome_dir = '../data/metagenome/recruitment_v4/'
    sample_columns = sample_relative_abundances.columns.values
    
    markers = ['o', 's', 'D']
    colors = ['tab:orange', 'tab:blue', 'tab:green']

    for i, species in enumerate(['A', 'B', 'C']):
        x = np.arange(sample_relative_abundances.shape[1])
        y = sample_relative_abundances.loc[species, :].values
        
        if i == 0:
            #sample_columns = loci_depth_df.columns[:-1]
            x_labels = [strip_sample_id(sid) for sid in sample_columns]
            xlabel_fontsize = 8
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=8, rotation=90, ha='center')
            ax.set_yscale('log')
        ax.plot(x, y, f'-{markers[i]}', c=colors[i], label=species)

    '''
    x = np.array([50, 55, 60, 65])
    for i, species in enumerate(['A', 'B', 'C']):
        y = plot_abundances.loc[species, :].values
        ax.plot(x, y, f'-{markers[i]}', c=colors[i], label=species)

    ax.set_xlim(48, 67)
    ax.set_xticks(x)
    ax.set_xlabel('temperature ($^\circ$C)', fontsize=label_size)
    '''
    ax.set_ylim(1E-4, 1.5)
    ax.set_yticks(yticks)
    ax.set_yscale('log')
    ax.set_ylabel('relative abundance', fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.plot(x, np.ones(len(x)) / 48, ls='--', c=open_colors['gray'][8])
    #y_min = 1. / species_abundances.loc['total_average_depth', :].values
    y_min = 1. / species_abundances.loc['loci_average_depth', :].values
    ax.plot(x, y_min, ls='--', c=open_colors['gray'][5])

    #ax.legend(fontsize=10)

    if clean_borders:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    if ax_label is not None:
        trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
        ax.text(-0.02, 1.02, ax_label, transform=ax.transAxes + trans, fontsize=14, fontweight='bold', va='bottom')


def make_revised_linkage_figures(pangenome_map, args, fig_count, avg_length_fraction=0.75, ms=5, low_diversity_cutoff=0.05, ax_label_size=12, tick_size=12):
    color_dict = {'A':'tab:orange', 'Bp':'tab:blue', 'Bp_subsampled':'gray', 'population':'k'}
    label_dict = {'A':r'$\alpha$', 'Bp':r'$\beta$', 'Bp_subsampled':r'$\beta$ (subsampled)', 'population':r'whole population'}
    cloud_dict = {'A':0.05, 'Bp':0.05, 'Bp_subsampled':0.05, 'population':0.1}
    marker_dict = {'A':'o', 'Bp':'s', 'Bp_subsampled':'x', 'population':'D'}

    rng = np.random.default_rng(args.random_seed)
    random_gene_linkage = calculate_random_gene_linkage(args, rng, cloud_dict)

    #cloud_radius = 0.1
    #random_gene_linkage = pickle.load(open(f'{args.results_dir}linkage_disequilibria/sscs_core_ogs_random_gene_linkage_c{cloud_radius}.dat', 'rb'))


    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(121)
    set_up_linkage_curve_axis(ax, xlim=(0.8, 2E4), ylim=(5E-3, 1.5E0), ax_label='', ax_label_fs=16, xticks=[1, 1E1, 1E2, 1E3], ylabel='linkage disequilibrium')
    for species in ['A', 'Bp', 'population']:
        #linkage_results = pickle.load(open(f'{args.results_dir}linkage_disequilibria/sscs_core_ogs_{species}_linkage_curves_c{cloud_radius}.dat', 'rb'))
        if species != 'population':
            cloud_radius = cloud_dict[species]
            #linkage_results = pickle.load(open(f'{args.output_dir}sscs_core_ogs_cleaned_{species}_linkage_curves_c{cloud_radius}.dat', 'rb'))
            linkage_results = pickle.load(open(f'{args.output_dir}sscs_core_ogs_cleaned_{species}_linkage_curves_c{cloud_radius}_all_sites.dat', 'rb'))
        else:
            #linkage_results = pickle.load(open(f'{args.output_dir}sscs_core_ogs_cleaned_{species}_linkage_curves.dat', 'rb'))
            linkage_results = pickle.load(open(f'{args.output_dir}sscs_core_ogs_cleaned_{species}_linkage_curves_all_sites.dat', 'rb'))
        #sigmad2 = plt_linkage.average_sigmad_sq(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
        #sigmad2_cg, x_cg = plt_linkage.coarse_grain_linkage_array(sigmad2)
        x_arr, sigmad2 = average_linkage_curves(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
        x_cg, sigmad2_cg = coarse_grain_distances(x_arr, sigmad2)

        #ax.plot(x_cg[:-5], sigmad2_cg[:-5], f'-{marker_dict[species]}', ms=ms, mec='none', lw=1, alpha=1.0, c=color_dict[species], label=label_dict[species]) # exclude last points with low depth
        ax.plot(x_cg[:-5], sigmad2_cg[:-5], f'-{marker_dict[species]}', ms=ms, mfc='none', lw=1, alpha=1.0, c=color_dict[species], label=label_dict[species]) # exclude last points with low depth
        if species in random_gene_linkage:
            #sigmad2_random_genes, og_pairs = random_gene_linkage[species]
            #ax.scatter(1.5E4, sigmad2_random_genes, s=20, fc='none', ec=color_dict[species], marker=marker_dict[species])

            gene_pair_linkage, og_arr = pickle.load(open(f'{args.output_dir}{species}_random_gene_linkage_c{cloud_radius}.dat', 'rb'))
            linkage_avg = np.nanmean(gene_pair_linkage[:, 0, :], axis=0)
            control_avg = np.nanmean(gene_pair_linkage[:, 1, :], axis=0)
            ax.scatter(1.5E4, linkage_avg[0], s=20, fc='none', ec=color_dict[species], marker=marker_dict[species])
            ax.scatter(1.5E4, control_avg[0], s=40, fc='none', ec=color_dict[species], marker='_', lw=2) # plot control

    ax.axvline(1E4, ls='--', c='k')
    ax.legend(fontsize=12, frameon=False)

    
    metadata = MetadataMap()
    ax = fig.add_subplot(122)
    f_divergences = f'../results/single-cell/locus_diversity/core_ogs_species_consensus_divergence_table.tsv'
    consensus_divergence_table = pd.read_csv(f_divergences, sep='\t', index_col=0)
    consensus_divergence_table['average'] = consensus_divergence_table.mean(axis=1)
    rrna_aln = main_figs.read_rrna_alignment(pangenome_map)
    rrna_consensus_divergences = main_figs.calculate_locus_consensus_divergence(rrna_aln, metadata)
    rrna_sag_ids = np.array(rrna_consensus_divergences.index)

    x = rrna_consensus_divergences.values
    y = consensus_divergence_table.loc[rrna_sag_ids, 'average'].values
    main_figs.plot_consensus_divergence_loci_comparisons(ax, x, y, rrna_sag_ids, metadata, fig, ax_label='', label_size=ax_label_size, tick_size=tick_size)

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_linkage_decay.pdf')
    fig_count += 1

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

    # Low vs high diversity alpha OGs
    metadata = MetadataMap()
    alignments_dir = '../results/single-cell/alignments/core_ogs_cleaned_4D_sites/'
    syna_num_site_alleles, syna_mutation_spectra = get_single_site_statistics(pangenome_map, metadata, alignments_dir, args, species='A', main_cloud=False)
    low_diversity_ogs = np.array(syna_num_site_alleles.loc[syna_num_site_alleles['fraction_polymorphic'] < low_diversity_cutoff, :].index)
    high_diversity_ogs = np.array([og for og in syna_num_site_alleles.index if og not in low_diversity_ogs])
    #linkage_arr, og_arr = calculate_species_gene_pair_linkage(pangenome_map, 'A', 0.05, rng, species_core_ogs=low_diversity_ogs)
    #pickle.dump((linkage_arr, og_arr), open(f'{args.output_dir}A_low_diversity_random_gene_linkage_c0.05.dat', 'wb'))
    #linkage_arr, og_arr = calculate_species_gene_pair_linkage(pangenome_map, 'A', 0.05, rng, species_core_ogs=high_diversity_ogs)
    #pickle.dump((linkage_arr, og_arr), open(f'{args.output_dir}A_high_diversity_random_gene_linkage_c0.05.dat', 'wb'))

    fig_count = plot_alpha_loci_linkage(low_diversity_ogs, high_diversity_ogs, args, fig_count, cloud_dict, avg_length_fraction=avg_length_fraction)


    # Compare A with B' and subsampled B'
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    set_up_linkage_curve_axis(ax, xlim=(1, 2E4), ylim=(5E-3, 1.5E0))
    for species in ['A', 'Bp', 'Bp_subsampled']:
        cloud_radius = cloud_dict[species]
        #linkage_results = pickle.load(open(f'{args.output_dir}sscs_core_ogs_cleaned_{species}_linkage_curves_c{cloud_radius}.dat', 'rb'))
        #sigmad2 = plt_linkage.average_sigmad_sq(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
        #sigmad2_cg, x_cg = plt_linkage.coarse_grain_linkage_array(sigmad2)
        linkage_results = pickle.load(open(f'{args.output_dir}sscs_core_ogs_cleaned_{species}_linkage_curves_c{cloud_radius}_all_sites.dat', 'rb'))
        x_arr, sigmad2 = average_linkage_curves(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
        x_cg, sigmad2_cg = coarse_grain_distances(x_arr, sigmad2)

        #ax.plot(x_cg[:-5], sigmad2_cg[:-5], '-o', ms=3, mec='none', lw=1, alpha=1.0, c=color_dict[species], label=label_dict[species]) # exclude last points with low depth
        ax.plot(x_cg[:-5], sigmad2_cg[:-5], f'-{marker_dict[species]}', ms=ms, mfc='none', mew=1.5, lw=1.5, alpha=1.0, c=color_dict[species], label=label_dict[species])
        if species in random_gene_linkage:
            #ax.scatter(1.5E4, random_gene_linkage[species], s=20, ec='none', fc=color_dict[species])
            sigmad2_random_genes, og_pairs = random_gene_linkage[species]
            ax.scatter(1.5E4, sigmad2_random_genes, s=20, fc='none', ec=color_dict[species], marker=marker_dict[species])
    ax.axvline(1.0E4, ls='--', c='k')
    ax.legend(fontsize=10, frameon=False)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_linkage_depth_validation.pdf')
    fig_count += 1


    # Plot curves for different cloud cutoffs
    avg_length_fraction = 0.75
    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    for i, species in enumerate(['A', 'Bp']):
        ax = fig.add_subplot(1, 2, i + 1)
        set_up_linkage_curve_axis(ax, xlim=(1, 3E3), ylim=(5E-2, 1.5E0), ax_label='', ax_label_fs=16, xticks=[1, 1E1, 1E2, 1E3], ylabel='linkage disequilibrium')
        for c in [0.03, 0.05, 0.1, 0.2, 1.0]:
            linkage_results = pickle.load(open(f'{args.output_dir}sscs_core_ogs_cleaned_{species}_linkage_curves_c{c}_all_sites.dat', 'rb'))
            x, sigmad2 = average_linkage_curves(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
            x_cg, sigmad2_cg = coarse_grain_distances(x, sigmad2)
            ax.plot(x_cg[:-6], sigmad2_cg[:-6], f'-{marker_dict[species]}', ms=ms, mfc='none', lw=1, alpha=1.0, label=f'c={c}') # exclude last points with low depth
        ax.legend(fontsize=10, frameon=False)
        #break
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_linkage_main_cloud.pdf')
    fig_count += 1


    # Plot rsq figure
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    set_up_linkage_curve_axis(ax, xlim=(0.8, 2E3), ylim=(2E-5, 1.0E-1), ax_label='', linkage_metric=r'$r^2$', ax_label_fs=16, xticks=[1, 1E1, 1E2, 1E3], ylabel='linkage disequilibrium', yticks=[1E-4, 1E-3, 1E-2, 1E-1])
    for species in ['A', 'Bp', 'population']:
        if species == 'Bp':
            #continue
            pass
        if species != 'population':
            cloud_radius = cloud_dict[species]
            linkage_results = pickle.load(open(f'{args.output_dir}sscs_core_ogs_cleaned_{species}_linkage_curves_c{cloud_radius}_all_sites.dat', 'rb'))
        else:
            linkage_results = pickle.load(open(f'{args.output_dir}sscs_core_ogs_cleaned_{species}_linkage_curves_all_sites.dat', 'rb'))
        x, rsq = average_linkage_curves(linkage_results, metric='r_sq', average_length_fraction=avg_length_fraction)
        x_cg, rsq_cg = coarse_grain_distances(x, rsq)
        ax.plot(x_cg[:-5], rsq_cg[:-5], f'-{marker_dict[species]}', ms=ms, mfc='none', lw=1, alpha=1.0, c=color_dict[species], label=label_dict[species]) # exclude last points with low depth

    #ax.axvline(1E4, ls='--', c='k')
    ax.legend(fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_rsq_decay.pdf')
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


def calculate_alpha_low_diversity_random_gene_linkage(low_diversity_ogs, args, rng, sites_ext='_all_sites', min_sample_size=20, min_coverage=0.9, sample_size=1000, cloud_radius=0.05):
    species = 'A'
    gene_pair_results = pickle.load(open(f'{args.output_dir}sscs_core_ogs_cleaned_{species}_gene_pair_linkage_c{cloud_radius}{sites_ext}.dat', 'rb'))

    gene_pair_linkage = gene_pair_results['sigmad_sq']
    sample_sizes = gene_pair_results['sample_sizes']

    # Draw random gene pairs
    g1_idx, g2_idx = np.where((sample_sizes.values >= min_sample_size) & (gene_pair_linkage.notna().values))
    og_ids = sample_sizes.index.values
    low_diversity_idx = np.arange(len(og_ids))[np.isin(og_ids, low_diversity_ogs)]
    low_diversity_g1_idx = g1_idx[np.isin(g1_idx, low_diversity_idx) & np.isin(g2_idx, low_diversity_idx)]
    low_diversity_g2_idx = g2_idx[np.isin(g1_idx, low_diversity_idx) & np.isin(g2_idx, low_diversity_idx)]
    random_sample_idx = rng.choice(len(low_diversity_g1_idx), size=sample_size, replace=False)
    gene_pair_array = np.array([og_ids[low_diversity_g1_idx[random_sample_idx]], og_ids[low_diversity_g2_idx[random_sample_idx]]]).T
    print(gene_pair_linkage)
    return (np.mean(gene_pair_linkage.values[low_diversity_g1_idx, low_diversity_g2_idx]), gene_pair_array)


def plot_alpha_loci_linkage(low_diversity_ogs, high_diversity_ogs, args, fig_count, cloud_dict, avg_length_fraction=0.75, ms=5):
    color_dict = {'A':'tab:orange', 'Bp':'tab:blue', 'Bp_subsampled':'gray', 'population':'k'}
    label_dict = {'A':r'$\alpha$', 'Bp':r'$\beta$', 'Bp_subsampled':r'$\beta$ (subsampled)', 'population':r'whole population'}
    #cloud_dict = {'A':0.05, 'Bp':0.1, 'Bp_subsampled':0.1, 'population':0.1}
    marker_dict = {'A':'o', 'Bp':'s', 'Bp_subsampled':'x', 'population':'D'}
    species = 'A'
    cloud_radius = cloud_dict[species]


    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    set_up_linkage_curve_axis(ax, xlim=(0.7, 1E4), ylim=(5E-3, 1.5E0), ax_label='', ax_label_fs=16, xticks=[1, 1E1, 1E2, 1E3], ylabel='linkage disequilibrium')
    linkage_results = pickle.load(open(f'{args.output_dir}sscs_core_ogs_cleaned_A_linkage_curves_c{cloud_radius}_all_sites.dat', 'rb'))
    low_diversity_gene_linkage, og_arr = pickle.load(open(f'{args.output_dir}A_low_diversity_random_gene_linkage_c0.05.dat', 'rb'))
    plot_diversity_comparisons(ax, species, linkage_results, low_diversity_gene_linkage, low_diversity_ogs, high_diversity_ogs, avg_length_fraction, ms, marker_dict)


    ax.legend(fontsize=10, frameon=False)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_alpha_linkage.pdf')

    return fig_count + 1


def plot_diversity_comparisons(ax, species, linkage_results, low_diversity_gene_linkage, low_diversity_ogs, high_diversity_ogs, avg_length_fraction, ms, marker_dict, x_wg=6E3):
    x_all, sigmad_all = average_linkage_curves(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
    x_all_cg, sigmad_all_cg = coarse_grain_distances(x_all, sigmad_all)
    #ax.plot(x_all_cg[:-5], sigmad_all_cg[:-5], f'-{marker_dict[species]}', ms=ms, mfc='none', lw=1, alpha=1.0, label='all core') # exclude last points with low depth

    lowd_linkage = {}
    highd_linkage = {}
    for og in linkage_results:
        if og in low_diversity_ogs:
            lowd_linkage[og] = linkage_results[og]
        elif og in high_diversity_ogs:
            highd_linkage[og] = linkage_results[og]

    x_lowd, sigmad_lowd = average_linkage_curves(lowd_linkage, metric='sigmad_sq', average_length_fraction=avg_length_fraction, min_depth=1000)
    x_lowd_cg, sigmad_lowd_cg = coarse_grain_distances(x_lowd, sigmad_lowd)
    x_lowd_cg = x_lowd_cg[sigmad_lowd_cg > 0]
    sigmad_lowd_cg = sigmad_lowd_cg[sigmad_lowd_cg > 0]
    #ax.plot(x_lowd_cg[:-5], sigmad_lowd_cg[:-5], f'-X', ms=ms, mec='none', lw=1, alpha=1.0, c='tab:red', label='non-hybrid core') # exclude last points with low depth
    ax.plot(x_lowd_cg[1:-2], sigmad_lowd_cg[1:-2], f'-X', ms=ms, mec='none', lw=1, alpha=1.0, c='tab:red', label='non-hybrid core') # exclude last points with low depth

    x_highd, sigmad_highd = average_linkage_curves(highd_linkage, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
    x_highd_cg, sigmad_highd_cg = coarse_grain_distances(x_highd, sigmad_highd)
    x_highd_cg = x_highd_cg[sigmad_highd_cg > 0]
    sigmad_highd_cg = sigmad_highd_cg[sigmad_highd_cg > 0]
    #ax.plot(x_highd_cg[:-5], sigmad_highd_cg[:-5], f'-{marker_dict[species]}', ms=ms, mfc='none', lw=1, alpha=1.0, c='tab:orange', label='hybrid core') # exclude last points with low depth
    ax.plot(x_highd_cg[1:-2], sigmad_highd_cg[1:-2], f'-{marker_dict[species]}', ms=ms, mfc='none', lw=1, alpha=1.0, c='tab:orange', label='hybrid core') # exclude last points with low depth

    # Plot whole-genome linkage
    ax.axvline(3E3, ls='--', c='k')

    gene_pair_linkage, og_arr = pickle.load(open(f'{args.output_dir}A_high_diversity_random_gene_linkage_c0.05.dat', 'rb'))
    hd_linkage_avg = np.nanmean(gene_pair_linkage[:, 0, :], axis=0)
    hd_control_avg = np.nanmean(gene_pair_linkage[:, 1, :], axis=0)
    ax.scatter(x_wg, hd_linkage_avg[0], s=ms**2, fc='none', ec='tab:orange', marker='o', zorder=1)
    ax.scatter(x_wg, hd_control_avg[0], s=60, fc='none', ec='tab:orange', marker='_', lw=2, zorder=2) # plot control

    low_diversity_gene_linkage, og_arr = pickle.load(open(f'{args.output_dir}A_low_diversity_random_gene_linkage_c0.05.dat', 'rb'))
    ld_linkage_avg = np.nanmean(low_diversity_gene_linkage[:, 0, :], axis=0)
    ld_control_avg = np.nanmean(low_diversity_gene_linkage[:, 1, :], axis=0)
    ax.scatter(x_wg, ld_linkage_avg[0], s=ms**2, fc='tab:red', marker='x', zorder=1)
    ax.scatter(x_wg, ld_control_avg[0], s=60, fc='none', ec='tab:red', marker='_', lw=2, zorder=2) # plot control

    print(f'Genomewide linkage (high diversity): {hd_linkage_avg[0]:.4f}/{hd_control_avg[0]:.4f}')
    print(f'Genomewide linkage (low diversity): {ld_linkage_avg[0]:.4f}/{ld_control_avg[0]:.4f}')


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


def check_hybrid_allele_haplotypes(pangenome_map, args):
    hybridization_dir = f'{args.results_dir}hybridization/'
    species_cluster_genomes = pd.read_csv(f'{hybridization_dir}sscs_labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)
    metadata = MetadataMap()
    rng = np.random.default_rng(args.random_seed)

    syna_hybrid_donor_frequency_table = main_figs.make_donor_frequency_table(species_cluster_genomes, 'A', pangenome_map, metadata)
    synbp_hybrid_donor_frequency_table = main_figs.make_donor_frequency_table(species_cluster_genomes, 'Bp', pangenome_map, metadata)

    syna_genotypes = print_hybrid_allele_stats('A', syna_hybrid_donor_frequency_table, species_cluster_genomes, pangenome_map, metadata)
    syna_genotypes_num = convert_genotype_to_numeric(syna_genotypes, biallelic=True)
    Dsq, denom = align_utils.calculate_ld_matrices_vectorized(syna_genotypes_num.values.T, convert_to_numeric=False, unbiased=False)
    rsq = Dsq / (denom + (denom == 0.))
    rsq_values = utils.get_matrix_triangle_values(rsq, k=1)
    print('Test: ', np.mean(rsq_values), np.std(rsq_values))

    random_genotype_arr = randomize_genotype_alleles(syna_genotypes_num.values, rng)
    Dsq, denom = align_utils.calculate_ld_matrices_vectorized(random_genotype_arr.T, convert_to_numeric=False, unbiased=False)
    rsq_control = Dsq / (denom + (denom == 0.))
    rsq_control_values = utils.get_matrix_triangle_values(rsq_control, k=1)
    print('Control: ', np.mean(rsq_control_values), np.std(rsq_control_values))
    print('\n\n')

    synbp_genotypes = print_hybrid_allele_stats('Bp', synbp_hybrid_donor_frequency_table, species_cluster_genomes, pangenome_map, metadata)
    synbp_genotypes_num = convert_genotype_to_numeric(synbp_genotypes, biallelic=True)
    Dsq, denom = align_utils.calculate_ld_matrices_vectorized(synbp_genotypes_num.values.T, convert_to_numeric=False, unbiased=False)
    rsq = Dsq / (denom + (denom == 0.))
    rsq_values = utils.get_matrix_triangle_values(rsq, k=1)
    print('Test: ', np.mean(rsq_values), np.std(rsq_values))

    random_genotype_arr = randomize_genotype_alleles(synbp_genotypes_num.values, rng)
    Dsq, denom = align_utils.calculate_ld_matrices_vectorized(random_genotype_arr.T, convert_to_numeric=False, unbiased=False)
    rsq_control = Dsq / (denom + (denom == 0.))
    rsq_control_values = utils.get_matrix_triangle_values(rsq_control, k=1)
    print('Control: ', np.mean(rsq_control_values), np.std(rsq_control_values))

def print_hybrid_allele_stats(species, hybrid_donor_frequency_table, species_cluster_genomes, pangenome_map, metadata):
    if species == 'A':
        donor_species = ['Bp', 'C', 'O']
    else:
        donor_species = ['A', 'C', 'O']

    hybrid_og_ids = hybrid_donor_frequency_table.loc[(hybrid_donor_frequency_table[donor_species] > 0).any(axis=1), :].index.values
    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')
    num_hybrid_sags = 0
    for s in species_sorted_sag_ids[species]:
        alleles, freq = utils.sorted_unique(species_cluster_genomes.loc[hybrid_og_ids, s].dropna())
        print(s, alleles, freq)
        if len(alleles) > 1:
            num_hybrid_sags += 1
    print(f'{num_hybrid_sags} hybrid SAGs out of {len(species_sorted_sag_ids[species])} total.')
    print('\n\n')

    return species_cluster_genomes.loc[hybrid_og_ids, species_sorted_sag_ids[species]]


def convert_genotype_to_numeric(genotypes, biallelic=False):
    loci_idx = genotypes.index.values
    genotypes_num = pd.DataFrame(0, index=loci_idx, columns=genotypes.columns, dtype=int)

    if biallelic == False:
        for l in loci_idx:
            alleles, freq = utils.sorted_unique(genotypes.loc[l, :].dropna(), sort='ascending', sort_by='tag')
            for k, a in enumerate(alleles):
                genotypes_num.loc[l, genotypes.loc[l, :] == a] = k + 1
    else:
        for l in loci_idx:
            alleles, freq = utils.sorted_unique(genotypes.loc[l, :].dropna())
            genotypes_num.loc[l, genotypes.loc[l, :] == alleles[0]] = 1
            genotypes_num.loc[l, np.isin(genotypes.loc[l, :], alleles[1:])] = 2

    return genotypes_num


def randomize_genotype_alleles(genotypes, rng, random_seed=None):
    g_arr = np.array(genotypes)
    shuffled_arr = []
    for i in range(g_arr.shape[0]):
        nonzero_idx = g_arr[i, :] > 0
        shuffled_row = np.zeros(g_arr.shape[1], dtype=int)
        shuffled_row[nonzero_idx] = rng.permutation(g_arr[i, nonzero_idx])
        shuffled_arr.append(shuffled_row)
    shuffled_arr = np.array(shuffled_arr)
    return shuffled_arr 


def calculate_gene_pair_linkage(pangenome_map, args, dM=0.05, sample_size=1000):
    rng = np.random.default_rng(args.random_seed)
    metadata = MetadataMap()
    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')
    core_ogs_dict = pangenome_map.get_core_og_ids(metadata, og_type='parent_og_id', output_type='dict')
    #for s in ['A', 'Bp', 'population']:
    for s in ['population']:
        if s != 'population':
            species_core_ogs = np.concatenate([core_ogs_dict[s], core_ogs_dict['M']])
            species_sag_ids = species_sorted_sag_ids[s]
        else:
            common_core_ogs = core_ogs_dict['Bp'][np.isin(core_ogs_dict['Bp'], core_ogs_dict['A'])]
            species_core_ogs = np.unique(np.concatenate([common_core_ogs, core_ogs_dict['M']]))
            species_sag_ids = np.concatenate([species_sorted_sag_ids['A'], species_sorted_sag_ids['Bp']]) 
            print(species_core_ogs, len(species_core_ogs))
            print(species_sag_ids, len(species_sag_ids))

        og_arr = rng.choice(species_core_ogs, size=(2, sample_size))
        linkage_values = []
        for i in range(sample_size):
            valid_pair = True
            gene_pair_alns = []
            control_alns = []
            for j in range(2):
                f_aln = f'{args.results_dir}alignments/core_ogs_cleaned/{og_arr[j, i]}_cleaned_aln.fna'
                aln = seq_utils.read_alignment_and_map_sag_ids(f_aln, pangenome_map) # use SAG IDs for easy concatenation of gene pair alignments

                if s != 'population':
                    species_aln = qlink.get_species_main_cloud_alignment(aln, s, species_sag_ids, dM, trimming='none')
                else:
                    species_aln = aln

                if (len(species_aln) < 2) or (species_aln.get_alignment_length() < 2):
                    valid_pair = False
                    continue

                species_snp_aln = seq_utils.get_snps(species_aln)
                gene_pair_alns.append(species_snp_aln)

                # Check alignment length
                if (species_snp_aln.get_alignment_length() < 2) or (len(species_snp_aln) < 2):
                    valid_pair = False
                else:
                    control_snp_aln = align_utils.randomize_alignment_snps(species_snp_aln)
                    control_alns.append(control_snp_aln)

            if valid_pair:
                linkage_results = qlink.calculate_average_gene_pair_linkage(*gene_pair_alns, return_sample_size=True)
                control_results = qlink.calculate_average_gene_pair_linkage(*control_alns, return_sample_size=True)
            else:
                linkage_results = [np.nan, np.nan, np.nan]
                control_results = [np.nan, np.nan, np.nan]
            linkage_values.append(np.array([np.array(linkage_results), np.array(control_results)]))

            print(i, og_arr[:, i], linkage_results, control_results)
        linkage_values = np.array(linkage_values)

        pickle.dump((linkage_values, og_arr), open(f'{args.output_dir}{s}_random_gene_linkage_c{dM}.dat', 'wb'))


def calculate_species_gene_pair_linkage(pangenome_map, s, dM, rng, species_core_ogs=None, sample_size=1000):
    metadata = MetadataMap()
    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')

    if species_core_ogs is None:
        core_ogs_dict = pangenome_map.get_core_og_ids(metadata, og_type='parent_og_id', output_type='dict')
        if s != 'population':
            species_core_ogs = np.concatenate([core_ogs_dict[s], core_ogs_dict['M']])
        else:
            common_core_ogs = core_ogs_dict['Bp'][np.isin(core_ogs_dict['Bp'], core_ogs_dict['A'])]
            species_core_ogs = np.unique(np.concatenate([common_core_ogs, core_ogs_dict['M']]))

    if s != 'population':
        species_sag_ids = species_sorted_sag_ids[s]
    else:
        species_sag_ids = np.concatenate([species_sorted_sag_ids['A'], species_sorted_sag_ids['Bp']]) 

    og_arr = rng.choice(species_core_ogs, size=(2, sample_size))
    linkage_values = []
    for i in range(sample_size):
        valid_pair = True
        gene_pair_alns = []
        control_alns = []
        for j in range(2):
            f_aln = f'{args.results_dir}alignments/core_ogs_cleaned/{og_arr[j, i]}_cleaned_aln.fna'
            aln = seq_utils.read_alignment_and_map_sag_ids(f_aln, pangenome_map) # use SAG IDs for easy concatenation of gene pair alignments

            if s != 'population':
                species_aln = qlink.get_species_main_cloud_alignment(aln, s, species_sag_ids, dM, trimming='none')
            else:
                species_aln = aln

            if (len(species_aln) < 2) or (species_aln.get_alignment_length() < 2):
                valid_pair = False
                continue

            species_snp_aln = seq_utils.get_snps(species_aln)
            gene_pair_alns.append(species_snp_aln)

            # Check alignment length
            if (species_snp_aln.get_alignment_length() < 2) or (len(species_snp_aln) < 2):
                valid_pair = False
            else:
                control_snp_aln = align_utils.randomize_alignment_snps(species_snp_aln)
                control_alns.append(control_snp_aln)

        if valid_pair:
            linkage_results = qlink.calculate_average_gene_pair_linkage(*gene_pair_alns, return_sample_size=True)
            control_results = qlink.calculate_average_gene_pair_linkage(*control_alns, return_sample_size=True)
        else:
            linkage_results = [np.nan, np.nan, np.nan]
            control_results = [np.nan, np.nan, np.nan]
        linkage_values.append(np.array([np.array(linkage_results), np.array(control_results)]))

        print(i, og_arr[:, i], linkage_results, control_results)
    linkage_values = np.array(linkage_values)
    
    return linkage_values, og_arr



def compare_gene_pair_linkage(pangenome_map, args, dM=0.05):
    for s in ['A', 'Bp', 'population']:
        gene_pair_linkage, og_arr = pickle.load(open(f'{args.output_dir}{s}_random_gene_linkage_c{dM}.dat', 'rb'))
        linkage_avg = np.nanmean(gene_pair_linkage[:, 0, :], axis=0)
        control_avg = np.nanmean(gene_pair_linkage[:, 1, :], axis=0)

        print(s)
        print(linkage_avg, linkage_avg.shape)
        print(control_avg, control_avg.shape)
        print('\n\n')


def make_alignment_figures(pangenome_map, args, fig_count=33):
    metadata = MetadataMap()
    rng = np.random.default_rng(args.random_seed)
    fig_count = plot_mixed_cluster_ogs(pangenome_map, metadata, rng, args, fig_count)

    fig_count = plot_rrna_alignments(pangenome_map, metadata, args, fig_count)


def plot_mixed_cluster_ogs(pangenome_map, metadata, rng, args, fig_count):
    merged_donor_frequency_table = read_merged_donor_frequency_table(pangenome_map, metadata, args)
    mosaic_og_ids = merged_donor_frequency_table.loc[merged_donor_frequency_table['mosaic hybrid'] > 0, :].index.values
    #mixed_cluster_ogs = ['YSG_1571', 'YSG_0215c', 'YSG_0405', 'YSG_0270', 'YSG_1280']
    mixed_cluster_ogs = ['YSG_1519', 'YSG_0498', 'YSG_0215c', 'YSG_1364', 'YSG_0291']
    #mixed_cluster_ogs = rng.choice(mosaic_og_ids, size=5)

    i_ascii = 65
    fig = plt.figure(figsize=(double_col_width, 1.2 * double_col_width))
    for i, og_id in enumerate(mixed_cluster_ogs):
        ax = fig.add_subplot(5, 1, i + 1)
        f_aln = f'{args.results_dir}alignments/core_ogs_cleaned/{og_id}_cleaned_aln.fna'
        #f_aln = f'{args.results_dir}sscs_pangenome/_aln_results/{og_id}_aln.fna'
        aln = seq_utils.read_alignment(f_aln)
        species_grouping = align_utils.sort_aln_rec_ids(aln, pangenome_map, metadata)
        lw = aln.get_alignment_length() / 100
        plot_alignment(aln, annotation=species_grouping, annotation_style='lines', annot_lw=lw, reference=1, fig_dpi=1000, ax=ax)
        ax.text(-0.05, 1.05, chr(i_ascii  + i), fontsize=14, fontweight='bold', va='bottom', transform=ax.transAxes)

        print(i, og_id)

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_msog_alignments.pdf')

    return fig_count + 1


def plot_rrna_alignments(pangenome_map, metadata, args, fig_count, fig_dpi=1000):
    fig = plt.figure(figsize=(double_col_width, 1.4 * single_col_width))
    gspec = gridspec.GridSpec(2, 2, width_ratios=[1, 1.0])

    i_ascii = 65

    ax = plt.subplot(gspec[0, :])
    f_16S_aln = f'{args.results_dir}supplement/16S_rRNA_aln.fna'
    aln = seq_utils.read_alignment(f_16S_aln)
    aln_plot = aln[:, 0:801]
    species_grouping = align_utils.sort_aln_rec_ids(aln_plot, pangenome_map, metadata)
    lw = aln_plot.get_alignment_length() / 100
    plot_alignment(aln_plot, annotation=species_grouping, annotation_style='lines', annot_lw=lw, reference=0, ax=ax)
    ax.text(-0.05, 1.05, chr(i_ascii), fontsize=14, fontweight='bold', va='bottom', transform=ax.transAxes)

    for i, og_id in enumerate(['YSG_0713', 'YSG_1007']):
        ax = plt.subplot(gspec[1, i])
        f_aln = f'{args.results_dir}supplement/{og_id}_manual_aln.fna'
        aln = seq_utils.read_alignment(f_aln)

        if i == 0:
            aln_plot = aln
        else:
            aln_plot = aln[:78, 0:301]

        species_grouping = align_utils.sort_aln_rec_ids(aln_plot, pangenome_map, metadata)
        lw = aln_plot.get_alignment_length() / 50
        plot_alignment(aln_plot, annotation=species_grouping, annotation_style='lines', annot_lw=lw, reference=0, ax=ax)
        ax.text(-0.05, 1.05, chr(i_ascii + i + 1), fontsize=14, fontweight='bold', va='bottom', transform=ax.transAxes)


    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_rrna_alignments.pdf', dpi=fig_dpi)

    return fig_count + 1


def make_sag_coverage_figures(pangenome_map, args):
    # Get alpha SAG IDs
    species_sorted_sags = metadata.sort_sags(pangenome_map.get_sag_ids(), by='species')
    species_sag_ids = species_sorted_sags[species]
    label_dict = {'A':r'$\alpha$ core genes', 'Bp':r'$\beta$ core genes'}

    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(121)
    ax.set_xlabel('Total contig length', fontsize=label_fs)
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
    ax.set_xticks([1E-5, 1E-4, 1E-3, 1E-2, 1E-1])

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_{species}_diversity_partition.pdf')
    plt.close()

    return fig_count + 1


if __name__ == '__main__':
    # Default variables
    figures_dir = '../figures/paper_panels/supplement/'
    pangenome_dir = '../results/single-cell/sscs_pangenome/'
    results_dir = '../results/single-cell/'
    output_dir = '../results/single-cell/supplement/'
    metagenome_dir = '../data/metagenome/recruitment_v4/'
    annotations_dir = '../data/single-cell/filtered_annotations/sscs/'
    f_orthogroup_table = f'{pangenome_dir}filtered_orthogroups/sscs_annotated_single_copy_orthogroup_presence.tsv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--data_dir', default='../results/single-cell/main_figures_data/', help='Main results directory.')
    parser.add_argument('-F', '--figures_dir', default=figures_dir, help='Directory where figures are saved.')
    parser.add_argument('-M', '--metagenome_dir', default=metagenome_dir, help='Directory with results for metagenome.')
    parser.add_argument('-N', '--annotations_dir', default=annotations_dir, help='Directory with annotation files.')
    parser.add_argument('-P', '--pangenome_dir', default=pangenome_dir, help='Pangenome directory.')
    parser.add_argument('-R', '--results_dir', default=results_dir, help='Main results directory.')
    parser.add_argument('-O', '--output_dir', default=output_dir, help='Directory in which supplemental data is saved.')
    parser.add_argument('-g', '--orthogroup_table', default=f_orthogroup_table, help='File with orthogroup table.')
    parser.add_argument('-r', '--random_seed', default=12345, type=int, help='Seed for RNG.')
    args = parser.parse_args()


    pangenome_map = pg_utils.PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    #fig_count = make_genome_clusters_figures(pangenome_map, args)
    #fig_count = 4
    #fig_count = make_linkage_figures(pangenome_map, args, fig_count=fig_count)
    #fig_count = 8
    #fig_count = make_recent_transfer_figures(pangenome_map, args, fig_count=fig_count)
    #fig_count = 9
    #fig_count = make_species_full_sweep_figures(pangenome_map, args, fig_count=fig_count)
    #fig_count = 14
    #fig_count = make_hybridization_qc_figures(pangenome_map, args, fig_count=fig_count)
    #fig_count = 19
    #fig_count = make_linkage_block_figures(args, fig_count=fig_count)
    #fig_count = 21
    #fig_count = make_synteny_figures(args, fig_count=fig_count)
    #fig_count = 22
    #fig_count = make_genetic_diversity_figures(pangenome_map, args, fig_count)
    #fig_count = 24
    #fig_count = make_metagenome_recruitment_figures(pangenome_map, args, fig_count)
    #fig_count = 26
    #make_revised_linkage_figures(pangenome_map, args, fig_count)
    #check_hybrid_allele_haplotypes(pangenome_map, args)
    #calculate_gene_pair_linkage(pangenome_map, args)
    #compare_gene_pair_linkage(pangenome_map, args)
    #fig_count = 33
    #fig_count = make_alignment_figures(pangenome_map, args, fig_count=fig_count)
    fig_count = 35
    make_sag_coverage_figures(pangenome_map, args)

    #metadata = MetadataMap()
    #synbp_num_site_alleles, synbp_mutation_spectra = get_single_site_statistics(pangenome_map, metadata, f'{args.results_dir}alignments/core_ogs_cleaned/', args, species='A', main_cloud=False, sites='all_sites', aln_ext='cleaned_aln.fna')
    #print(synbp_num_site_alleles)
    #plot_alpha_spring_low_diversity(pangenome_map, metadata, core_og_ids[:10], args)
    #fig_count = 24
    #plot_gamma_alignment_results(pangenome_map, metadata, fig_count, args)
