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
import make_revision_main_figures as main_figs
import er2
import scipy.stats as stats
import matplotlib.transforms as mtransforms
import mcorr_fit
from analyze_metagenome_reads import strip_sample_id, strip_target_id, plot_abundant_target_counts
from syn_homolog_map import SynHomologMap
from metadata_map import MetadataMap
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from plot_utils import *

fig_count = 1

def make_genome_clusters_figures(pangenome_map, rng, args, cloud_radius=0.1):
    global fig_count

    metadata = MetadataMap()
    num_triple_patterns = 5

    # Plot gene triplet example patterns
    fig = plt.figure(figsize=(double_col_width, 0.75 * double_col_width))
    gs = gridspec.GridSpec(3, 5, hspace=0.35, wspace=0.05)
    example_sag_ids = ['UncmicOctRedA1J9_FD', 'UncmicMRedA02H14_2_FD', 'UncmicOcRedA3L13_FD']
    gene_triplet_divergences_dict = pickle.load(open(f'{args.results_dir}v1_data/sscs_gene_triple_divergence_tables.dat', 'rb'))
    ax_labels = ['A', 'B', 'C']
    for i, sag_id in enumerate(example_sag_ids):
        ax_objs = []
        for j in range(num_triple_patterns):
            ax_objs.append(fig.add_subplot(gs[i, j]))
        plot_sag_gene_triplet_aggregates(ax_objs, gene_triplet_divergences_dict[sag_id], rng, ax_label=ax_labels[i])
    plt.savefig(f'{args.figures_dir}S{fig_count}_gene_triplet_examples.pdf')
    fig_count += 1

    # Plot SAG fingerprints
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    triplet_pattern_table = pd.read_csv(f'{args.results_dir}v1_data/gene_triplet_table_cloud_radius{cloud_radius}.tsv', sep='\t', index_col=0)
    plot_sag_fingerprints(ax, triplet_pattern_table, metadata, lw=0.5, cloud_radius=cloud_radius)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_gene_triplet_fingerprints.pdf')
    fig_count += 1

    # Plot 16S tree
    rrna_tree = Tree(f'{args.results_dir}v1_data/rrna_UPGMA_tree.nwk')
    plot_16s_tree(rrna_tree, pangenome_map, metadata, args, fig_count=fig_count)
    fig_count += 1

    # Plot Cyanobacteria tree
    concatenated_tree = Tree(f'{args.results_dir}v1_data/cyanobacteria_marker_genes.tree')
    ref_table = read_cyanobacteria_ref_table(f'{args.results_dir}supplement/Chen_et_al_2020_metadata.xlsx', concatenated_tree)
    leaf_annotations = make_cyanobacteria_tree_annotation(ref_table, concatenated_tree, pangenome_map)
    plot_cyanobacteria_tree(concatenated_tree, ref_table, leaf_annotations, prune_leaves=leaf_annotations['Thermal springs'], draw_mode='r', savefig=f'{args.figures_dir}S{fig_count}_cyano_tree.pdf', width=120)
    fig_count += 1

    # Plot Gamma alignment results
    plot_gamma_alignment_results(pangenome_map, metadata, args, savefig=f'{args.figures_dir}S{fig_count}_gamma_blast.pdf')
    fig_count += 1


def plot_sag_gene_triplet_aggregates(ax_objs, sag_triplet_divergences_df, rng, triplet_patterns=['XA-B', 'XAB', 'X-A-B', 'X-AB', 'XB-A'], R=0.3, ax_label=None, label_fs=14):
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
            plot_triplets(divergence_values, ax, rng=rng, center='X', R=R, ms=10)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if i == 0 and ax_label is not None:
            ax.text(-0.2, 1.05, ax_label, transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)
            #ax.text(-1.5 * R, 1.0 * R, ax_label, fontweight='bold', fontsize=label_fs)

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


def plot_triplets(triplets, ax, rng=None, center='X', cloud_radius=0.1, ms=40, R=0.4, alpha=0.4):
    ax.set_xlim(-R, R)
    ax.set_xticks([])
    ax.set_ylim(-R, R)
    ax.set_yticks([])
    #ax.set_axis_off()

    if center == 'X':
        circle = plt.Circle((0,0), cloud_radius, fc='none', ec='k', lw=1.0, zorder=1)
        ax.add_patch(circle)
        ax.scatter([0], [0], s=0.75*ms, marker='o', edgecolor='none', facecolor='k', zorder=1)

        for triplet in triplets:
            #print(triplet)
            if (0 not in triplet) and (1 not in triplet):
                if rng is not None:
                    rotation = rng.uniform(0, 2 * np.pi)
                else:
                # Apply random rotation
                    rotation = 0.

                triangle_coordinates = calculate_triangle_coords(triplet, rotation=rotation)
                if ~np.isnan(triangle_coordinates[2][1]):
                    triangle = plt.Polygon(triangle_coordinates, facecolor='none', edgecolor='gray', lw=0.5, alpha=0.35, zorder=0)
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
            if rng is not None:
                rotation = rng.uniform(0, 2 * np.pi)
            else:
                # Apply random rotation
                rotation = 0.

            triangle_coordinates = calculate_triangle_coords(triplet, center=center, rotation=rotation)
            if ~np.isnan(triangle_coordinates[2][1]):
                triangle = plt.Polygon(triangle_coordinates, facecolor='none', edgecolor='gray', lw=0.5, alpha=0.35, zorder=0)
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


def calculate_triangle_coords(triplet, center='X', rotation=0):
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
    coords = np.array(coords)

    # Apply rotation
    R_mat = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])

    return np.matmul(R_mat, coords.T).T


def plot_sag_fingerprints(ax, triplet_pattern_table, metadata, sag_ids=None, x_labels=['XA-B', 'XAB', 'X-A-B', 'X-AB', 'XB-A'], lw=1, alpha=0.5, cloud_radius=10, clean_borders=True):
    color_dict = {'A':'tab:orange', 'Bp':'tab:blue', 'C':'tab:green'}
    species_added = {'A':False, 'Bp':False, 'C':False}
    species_labels = {'A':r'$\alpha$', 'Bp':r'$\beta$', 'C':r'$\gamma$'}
    x = np.arange(len(x_labels))

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(-0.02, 1.2)
    ax.set_ylabel('Gene fraction')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

    if clean_borders:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

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

    ax.legend(fontsize=12, loc='upper center', frameon=False)


def plot_16s_tree(rrna_tree, pangenome_map, metadata, args, fig_count=1):
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


def read_cyanobacteria_ref_table(f_metadata, concatenated_tree):
    metadata_book = pd.read_excel(f_metadata, sheet_name=None, header=1)
    ref_table = metadata_book['TableS1']

    leaf_names = [leaf.name for leaf in concatenated_tree.iter_leaves()]
    ref_table = ref_table.loc[ref_table['strainID'].isin(leaf_names), :]
    ref_table = ref_table.set_index('strainID')
     
    return ref_table

def make_cyanobacteria_tree_annotation(ref_table, concatenated_tree, pangenome_map):
    leaf_annotations = {'Thermal springs':[], 'Freshwater':[], 'Terrestrial':[], 'A':['Synechococcus_sp._JA-3-3Ab'], 'Bp':['Synechococcus_sp._JA-2-3Ba2-13']}
    for leaf in concatenated_tree.iter_leaves():
        species_name = leaf.name

        # Rename Synecho. C cell
        candidate_sag_id = pangenome_map.get_gene_sag_id(species_name)
        if species_name != candidate_sag_id:
            species_name = utils.strip_sample_name(candidate_sag_id, replace=True)
            leaf.name = species_name
            leaf_annotations['C'] = species_name
            leaf_annotations['Thermal springs'].append(species_name)
            continue

        environment = ref_table.loc[species_name, 'Ecosystem']
        leaf_annotations[environment].append(species_name)

    return leaf_annotations


def plot_cyanobacteria_tree(tree, ref_table, leaf_annotations, prune_leaves=None, draw_mode='c', width=110, scale=0.1, fs=12, pad=12, savefig=None):
    # Set parameters
    branch_thickness = 2

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

    ts = ete3.TreeStyle()
    ts.mode = draw_mode
    ts.draw_guiding_lines = False
    ts.scale_length = scale # Set scale
    ts.show_leaf_name = False

    if prune_leaves is not None:
        tree.prune(prune_leaves)


    for n in tree.traverse():
        nstyle = NodeStyle()
        nstyle['fgcolor'] = 'black'
        nstyle['size'] = 0
        n.set_style(nstyle)

    # Modified from Kat Holt package
    for node in tree.traverse():
        nstyle = NodeStyle()

        if node.is_leaf():
            if node.name in leaf_annotations['Freshwater']:
                nstyle['fgcolor'] = open_colors['blue'][4]
            elif node.name in leaf_annotations['Terrestrial']:
                nstyle['fgcolor'] = open_colors['green'][4]
            elif node.name in leaf_annotations['Thermal springs']:
                nstyle['fgcolor'] = open_colors['red'][4]

            nstyle['size'] = 8
            node.set_style(nstyle)

            # Remove name endings
            if 'GCA' in node.name:
                node.name = node.name.split('_GCA')[0]
            if '_cyanobac' in node.name:
                node.name = node.name.split('_cyanobac')[0]

            name_face = TextFace(node.name, fsize=fs)
            name_face.margin_left = pad
            if node.name in leaf_annotations['A']:
                name_face.background.color = open_colors['orange'][6]
            elif node.name in leaf_annotations['Bp']:
                name_face.background.color = open_colors['blue'][6]
            elif node.name in leaf_annotations['C']:
                name_face.background.color = open_colors['green'][6]

            node.add_face(name_face, column=0)


        else:
            nstyle['fgcolor'] = 'black'
            nstyle['size'] = 0
            node.set_style(nstyle)

        node.img_style['hz_line_width'] = branch_thickness
        node.img_style['vt_line_width'] = branch_thickness

    tree.dist = 0 # Set root distance to zero

    if savefig is None:
        tree.render(f'{args.figures_dir}cyanobacteria_tree.pdf', w=width, units='mm', dpi=300, tree_style=ts)
    else:
        tree.render(savefig, w=width, units='mm', dpi=300, tree_style=ts)


def plot_gamma_alignment_results(pangenome_map, metadata, args, savefig, num_bins=50):
    alignment_files = glob.glob(f'{args.results_dir}v1_data/alignments/Ga*_gamma_blast.tab')
    x_bins = np.linspace(0, 0.5, num_bins)

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\gamma$ divergence', fontsize=14)
    ax.set_ylabel('Genomes', fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

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
        ax.plot(x, hist, alpha=0.5, lw=1, c='tab:gray')

    plt.tight_layout()
    plt.savefig(savefig)
    plt.close()


###########################################################
# Metagenome figures
###########################################################

def make_metagenome_recruitment_figures(pangenome_map, args, num_loci=100):
    global fig_count

    # Plot metagenome loci correlations
    plot_correlation_across_loci(num_loci, args, f'{args.figures_dir}S{fig_count}_read_depth_correlation.pdf')
    fig_count += 1

    # Plot metagenome species abundances
    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    species_abundances = main_figs.read_species_abundances(args, num_loci)
    species_abundances.loc[['A', 'Bp', 'C'], :] = species_abundances.loc[['A', 'Bp', 'C'], :] / species_abundances.loc['total_average_depth', :]
    plot_species_frequency_across_samples(ax, species_abundances, fig, ms=6, ax_label='')
    ax.set_yticks([1E-4, 1E-3, 1E-2, 1E-1, 1])
    plt.tight_layout(pad=1.5)
    plt.savefig(f'{args.figures_dir}S{fig_count}_species_abundances.pdf')
    plt.close()
    fig_count += 4


def plot_correlation_across_loci(num_loci, args, savefig):
    species_labels = {'A':r'$\alpha$', 'B':r'$\beta$', 'C':r'$\gamma$'}

    fig = plt.figure(figsize=(double_col_width, 2.4 * single_col_width))
    for i, species in enumerate(['A', 'B', 'C']):
        ax = fig.add_subplot(3, 1, i + 1)
        ax.set_title(f'{species_labels[species]} reference', fontsize=14)
        ax.set_yticks([1, 1E1, 1E2, 1E3])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        f_recruitment = f'{args.metagenome_dir}{species}_Allele_pident95.xlsx'
        alleles_df = pd.read_excel(f_recruitment, sheet_name=0, index_col=0)
        sample_ids = np.array([c for c in alleles_df.columns if 'Hot' in c])
        alleles_df['average_depth'] = [np.nan, np.nan] + [np.nanmean(row.astype(float)) for row in alleles_df[sample_ids].values[2:]]
        alleles_df = alleles_df.sort_values('average_depth', ascending=False)
        loci_depth_df = alleles_df.loc[alleles_df.index[:-2], np.concatenate([sample_ids, ['average_depth']])]
        plot_abundant_target_counts(ax, loci_depth_df, ylabel='Read depth', num_targets=num_loci, lw=0.75, alpha=0.5)

    plt.tight_layout()
    plt.savefig(savefig)
    plt.close()

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

    ax.tick_params(labelsize=tick_size)
    ax.set_ylim(1E-4, 1.5)
    ax.set_yticks(yticks)
    ax.set_yscale('log')
    ax.set_ylabel('Relative abundance', fontsize=label_size)
    ax.plot(x, np.ones(len(x)) / 48, ls='--', c=open_colors['gray'][8])
    y_min = 1. / species_abundances.loc['loci_average_depth', :].values
    ax.plot(x, y_min, ls='--', c=open_colors['gray'][5])

    #ax.legend(fontsize=10)

    if clean_borders:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    if ax_label is not None:
        #trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
        #ax.text(-0.02, 1.02, ax_label, transform=ax.transAxes + trans, fontsize=14, fontweight='bold', va='bottom')
        ax.text(-0.2, 1.05, ax_label, transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)


###########################################################
# Linkage figures
###########################################################

def make_linkage_figures(pangenome_map, args, avg_length_fraction=0.75, ax_label_size=14, ms=5):
    global fig_count

    metadata = MetadataMap()

    # Set up style dicts
    color_dict = {'A':'tab:orange', 'Bp':'tab:blue', 'Bp_subsampled':'gray', 'population':'k'}
    label_dict = {'A':r'$\alpha$', 'Bp':r'$\beta$', 'Bp_subsampled':r'$\beta$ (subsampled)', 'population':r'whole population'}
    cloud_dict = {'A':0.05, 'Bp':0.05, 'Bp_subsampled':0.05, 'population':0.1}
    marker_dict = {'A':'o', 'Bp':'s', 'Bp_subsampled':'x', 'population':'D'}

    plot_linkage_depth_control(cloud_dict, color_dict, label_dict, marker_dict, args,
            avg_length_fraction=avg_length_fraction, ms=ms,
            savefig=f'{args.figures_dir}S{fig_count}_linkage_depth_validation.pdf')
    fig_count += 1


    # Plot curves for different cloud cutoffs
    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax_labels = ['A', 'B']
    for i, species in enumerate(['A', 'Bp']):
        ax = fig.add_subplot(1, 2, i + 1)
        main_figs.set_up_linkage_curve_axis(ax, xlim=(1, 3E3), ylim=(5E-2, 1.5E0), ax_label='', ax_label_fs=16, xticks=[1, 1E1, 1E2, 1E3], ylabel='Linkage disequilibrium', clean_borders=True)
        for c in [0.03, 0.05, 0.1, 0.2, 1.0]:
            linkage_results = pickle.load(open(f'{args.results_dir}v1_data/sscs_core_ogs_cleaned_{species}_linkage_curves_c{c}_all_sites.dat', 'rb'))
            x, sigmad2 = main_figs.average_linkage_curves(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
            x_cg, sigmad2_cg = main_figs.coarse_grain_distances(x, sigmad2)
            ax.plot(x_cg[:-6], sigmad2_cg[:-6], f'-{marker_dict[species]}', ms=ms, mfc='none', lw=1, alpha=1.0, label=f'c={c}') # exclude last points with low depth
        ax.legend(fontsize=10, frameon=False)
        ax.text(-0.2, 1.05, ax_labels[i], transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)
    plt.tight_layout(pad=1)
    plt.savefig(f'{args.figures_dir}S{fig_count}_linkage_main_cloud.pdf')
    fig_count += 1


    # Plot sigmad_sq vs rsq comparison
    #plot_linkage_metric_comparison(cloud_dict, color_dict, label_dict, marker_dict, args, savefig=f'{args.figures_dir}S{fig_count}_rsq_decay.pdf')
    #fig_count += 1


    # Plot A vs Bp vs population
    random_gene_linkage = main_figs.calculate_random_gene_linkage(args, rng, cloud_dict)

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    main_figs.set_up_linkage_curve_axis(ax, xlim=(0.8, 1E4), ylim=(5E-3, 1.5E0), ax_label='', ax_label_fs=16, xticks=[1, 1E1, 1E2, 1E3], ylabel='Linkage disequilibrium', yticks=[1E-2, 1E-1, 1], clean_borders=True)
    for species in ['A', 'Bp', 'population']:
        if species != 'population':
            cloud_radius = cloud_dict[species]
            linkage_results = pickle.load(open(f'{args.results_dir}supplement/sscs_core_ogs_cleaned_{species}_linkage_curves_c{cloud_radius}_all_sites.dat', 'rb'))
        else:
            linkage_results = pickle.load(open(f'{args.results_dir}supplement/sscs_core_ogs_cleaned_{species}_linkage_curves_all_sites.dat', 'rb'))

        x, rsq = main_figs.average_linkage_curves(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
        x_cg, rsq_cg = main_figs.coarse_grain_distances(x, rsq)
        ax.plot(x_cg[:-5], rsq_cg[:-5], f'-{marker_dict[species]}', ms=ms, mfc='none', lw=1, alpha=1.0, c=color_dict[species], label=label_dict[species]) # exclude last points with low depth

        if species in random_gene_linkage:
            gene_pair_linkage, og_arr = pickle.load(open(f'{args.results_dir}supplement/{species}_random_gene_linkage_c{cloud_radius}.dat', 'rb'))
            linkage_avg = np.nanmean(gene_pair_linkage[:, 0, :], axis=0)
            control_avg = np.nanmean(gene_pair_linkage[:, 1, :], axis=0)
            ax.scatter(7.0E3, linkage_avg[0], s=20, fc='none', ec=color_dict[species], marker=marker_dict[species])
            ax.scatter(7.0E3, control_avg[0], s=40, fc='none', ec=color_dict[species], marker='_', lw=2) # plot control

    ax.axvline(3E3, ls='--', c='k')
    ax.legend(fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_population_linkage.pdf')
    plt.close()
    fig_count += 1


    # Comparison to neutral model
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    main_figs.set_up_linkage_curve_axis(ax, ax_label='', xlim=(0.02, 3E2), xticks=[1E-1, 1E0, 1E1, 1E2], ylim=(1E-2, 1.5E0), xlabel=r'rescaled separation, $\rho x$', ylabel='linkage disequilibrium', x_ax_label=3E-3)
    rho_fit = {'A':0.03, 'Bp':0.12}
    theta = 0.03
    lmax = 2000
    x_theory = np.arange(1, lmax)

    for species in ['A', 'Bp']:
        cloud_radius = cloud_dict[species]
        linkage_results = pickle.load(open(f'{args.output_dir}sscs_core_ogs_cleaned_{species}_linkage_curves_c{cloud_radius}_all_sites.dat', 'rb'))
        x_arr, sigmad2 = main_figs.average_linkage_curves(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
        x_cg, sigmad2_cg = main_figs.coarse_grain_distances(x_arr, sigmad2)
        y0 = sigmad2_cg[1]

        # Plot theory
        rho = rho_fit[species]
        y_theory = er2.sigma2_theory(rho * x_theory, theta)
        ax.plot(rho * x_cg[:-5], sigmad2_cg[:-5], f'-{marker_dict[species]}', ms=ms, mfc='none', mew=1.5, lw=1.0, alpha=1.0, c=color_dict[species], label=label_dict[species])

    x_theory = np.geomspace(0.01, 200, 100)
    y_theory = er2.sigma2_theory(x_theory, theta)
    ax.plot(x_theory, y_theory, lw=1.5, ls='-', c='k', label=f'neutral theory')

    ax.legend(fontsize=10, frameon=False, loc='lower left')
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_linkage_collapse.pdf')
    fig_count += 1


    # SAG coverage figure
    plot_sag_coverages(pangenome_map, metadata, savefig=f'{args.figures_dir}S{fig_count}_sag_coverage.pdf')
    fig_count += 1

    # ClonalFrameML results
    recomb_df = pd.read_csv(f'{args.results_dir}supplement/clonal_frame_ml_Bp_results.importation_status.txt', sep='\t')
    params = pd.read_csv(f'{args.results_dir}supplement/clonal_frame_ml_Bp_results.em.txt', sep='\t', index_col=0)
    plot_recombination_length_distribution(recomb_df, params, savefig=f'{args.figures_dir}S{fig_count}_clonal_frame_recomb_lengths.pdf')
    fig_count += 1

    # mcorr results
    A_corr_df = pd.read_csv(f'{args.results_dir}supplement/A_correlation_profile_b150.csv') # best fit A profile
    Bp_corr_df = pd.read_csv(f'{args.results_dir}supplement/Bp_correlation_profile_b835.csv') # best fit Bp profile
    print(A_corr_df)

    fig = plt.figure(figsize=(double_col_width, single_col_width))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0)
    gssub = gs[:, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0)
    plot_mcorr_results(A_corr_df, gssub, ec='tab:orange', label=r'$\alpha$ data', ax_label='A')
    gssub = gs[:, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0)
    plot_mcorr_results(Bp_corr_df, gssub, ec='tab:blue', label=r'$\beta$ data', ax_label='B')
    fig.tight_layout()
    fig.savefig(f'{args.figures_dir}S{fig_count}_mcorr_fit.pdf')
    plt.close()
    fig_count += 1


def plot_linkage_depth_control(cloud_dict, color_dict, label_dict, marker_dict, args, avg_length_fraction=0.75, ms=5, savefig=None):
    # Set axes limits
    xlim = (0.8, 3E3)
    ylim = (5E-3, 1.5E0)

    # Read random gene linkage
    #random_gene_linkage = pickle.load(open(f'{args.results_dir}v1_data/sscs_core_ogs_random_gene_linkage_c{cloud_radius}.dat', 'rb'))
    #random_gene_linkage = get_random_gene_linkage(args, rng, cloud_dict)
    random_gene_linkage = {} # Don't plot random genes

    # Linkage depth validation
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    main_figs.set_up_linkage_curve_axis(ax, xlim=xlim, ylim=ylim, ax_label='', ax_label_fs=14, xticks=[1, 1E1, 1E2, 1E3], ylabel='Linkage disequilibrium', clean_borders=True)
    for species in ['A', 'Bp', 'Bp_subsampled']:
        cloud_radius = cloud_dict[species]
        linkage_results = pickle.load(open(f'{args.results_dir}v1_data/sscs_core_ogs_cleaned_{species}_linkage_curves_c{cloud_radius}_all_sites.dat', 'rb'))
        x_arr, sigmad2 = main_figs.average_linkage_curves(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
        x_cg, sigmad2_cg = main_figs.coarse_grain_distances(x_arr, sigmad2)
        ax.plot(x_cg[:-5], sigmad2_cg[:-5], f'-{marker_dict[species]}', ms=ms, mfc='none', mew=1.0, lw=1.0, alpha=1.0, c=color_dict[species], label=label_dict[species])
        if species in random_gene_linkage:
            gene_linkage_avg, og_pairs = random_gene_linkage[species]
            ax.scatter(7.0E3, gene_linkage_avg, s=20, fc='none', ec=color_dict[species], marker=marker_dict[species])

    ax.legend(fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_linkage_depth_validation.pdf')


def plot_linkage_metric_comparison(cloud_dict, color_dict, label_dict, marker_dict, args, avg_length_fraction=0.75, ms=5, savefig=None):
    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(121)
    main_figs.set_up_linkage_curve_axis(ax, xlim=(0.8, 2E3), ylim=(5E-3, 1.5E0), ax_label='', ax_label_fs=16, xticks=[1, 1E1, 1E2, 1E3], ylabel='Linkage disequilibrium', yticks=[1E-2, 1E-1, 1], clean_borders=True)
    for species in ['A', 'Bp', 'population']:
        if species == 'Bp':
            #continue
            pass
        if species != 'population':
            cloud_radius = cloud_dict[species]
            linkage_results = pickle.load(open(f'{args.results_dir}v1_data/sscs_core_ogs_cleaned_{species}_linkage_curves_c{cloud_radius}_all_sites.dat', 'rb'))
        else:
            linkage_results = pickle.load(open(f'{args.results_dir}v1_data/sscs_core_ogs_cleaned_{species}_linkage_curves_all_sites.dat', 'rb'))

        x, rsq = main_figs.average_linkage_curves(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
        x_cg, rsq_cg = main_figs.coarse_grain_distances(x, rsq)
        ax.plot(x_cg[:-5], rsq_cg[:-5], f'-{marker_dict[species]}', ms=ms, mfc='none', lw=1, alpha=1.0, c=color_dict[species], label=label_dict[species]) # exclude last points with low depth
    ax.legend(fontsize=12, frameon=False)
    ax.text(-0.2, 1.1, 'A', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)

    ax = fig.add_subplot(122)
    main_figs.set_up_linkage_curve_axis(ax, xlim=(0.8, 2E3), ylim=(2E-5, 1.0E-1), ax_label='', linkage_metric=r'$r^2$', ax_label_fs=16, xticks=[1, 1E1, 1E2, 1E3], ylabel='Linkage disequilibrium', yticks=[1E-4, 1E-3, 1E-2, 1E-1], clean_borders=True)
    for species in ['A', 'Bp', 'population']:
        if species == 'Bp':
            #continue
            pass
        if species != 'population':
            cloud_radius = cloud_dict[species]
            linkage_results = pickle.load(open(f'{args.results_dir}v1_data/sscs_core_ogs_cleaned_{species}_linkage_curves_c{cloud_radius}_all_sites.dat', 'rb'))
        else:
            linkage_results = pickle.load(open(f'{args.results_dir}v1_data/sscs_core_ogs_cleaned_{species}_linkage_curves_all_sites.dat', 'rb'))

        x, rsq = main_figs.average_linkage_curves(linkage_results, metric='r_sq', average_length_fraction=avg_length_fraction)
        x_cg, rsq_cg = main_figs.coarse_grain_distances(x, rsq)
        ax.plot(x_cg[:-5], rsq_cg[:-5], f'-{marker_dict[species]}', ms=ms, mfc='none', lw=1, alpha=1.0, c=color_dict[species], label=label_dict[species]) # exclude last points with low depth
    ax.text(-0.2, 1.1, 'B', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)

    ax.legend(fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_rsq_decay.pdf')


def get_random_gene_linkage(args, rng, cloud_dict, min_sample_size=20, sample_size=1000, sites_ext='_all_sites'):
    random_gene_linkage = {}
    for species in ['A', 'Bp', 'Bp_subsampled', 'population']:
        c = cloud_dict[species]
        if species != 'population':
            gene_pair_results = pickle.load(open(f'{args.results_dir}v1_data/sscs_core_ogs_cleaned_{species}_gene_pair_linkage_c{c}{sites_ext}.dat', 'rb'))
        else:
            gene_pair_results = pickle.load(open(f'{args.results_dir}v1_data/sscs_core_ogs_cleaned_{species}_gene_pair_linkage{sites_ext}.dat', 'rb'))
        mean_linkage, gene_pair_array = main_figs.calculate_random_gene_linkage_values(gene_pair_results, rng, sample_size, min_sample_size)
        random_gene_linkage[species] = (mean_linkage, gene_pair_array)
    return random_gene_linkage


def plot_sag_coverages(pangenome_map, metadata, savefig=None, f_cutoff=0.75):
    sag_ids = pangenome_map.get_sag_ids()

    # Get total coverage for SAGs
    bp_covered = []
    for s in sag_ids:
        contigs = pangenome_map.cell_contigs[s]
        bp = 0
        for c in contigs:
            c_rec = pangenome_map.contig_records[c]
            bp += len(c_rec)
        bp_covered.append(bp)

    coverage_df = pd.Series(bp_covered, index=sag_ids)
    species_sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')
    species_labels = {'A':r'$\alpha$', 'Bp':r'$\beta$', 'C':r'$\gamma$'}
    species_colors = {'A':'tab:orange', 'Bp':'tab:blue', 'C':'tab:green'}
    species_genome_sizes = {'A':utils.osa_genome_size, 'Bp':utils.osbp_genome_size}
    high_coverage_sags = {}

    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('SAG index', fontsize=14)
    ax.set_ylabel('SAG coverage (bp)', fontsize=14)
    ax.set_xlim(-1, len(sag_ids) + 1)
    #ax.set_ylim(0, 1.2 * species_genome_sizes['Bp'])

    i0 = 0
    x = np.arange(len(sag_ids))
    for sp in ['A', 'Bp', 'C']:
        sp_sags = species_sorted_sag_ids[sp]
        n_sp = len(sp_sags)
        ax.bar(x[i0:i0 + n_sp], np.sort(coverage_df[sp_sags])[::-1], color=species_colors[sp], label=species_labels[sp])
        i0 += len(sp_sags)

        if sp in species_genome_sizes:
            L = species_genome_sizes[sp]
            ax.axhline(f_cutoff * L, lw=2, ls='--', c=species_colors[sp])
            high_coverage_sags[sp] = coverage_df[sp_sags].index.values[coverage_df[sp_sags] > f_cutoff * L]
    print(coverage_df)
    print(f'Coverage:\n\t{(np.mean(coverage_df.astype(float))/3e+6):.3f} +- {(np.std(coverage_df.astype(float))/3e+6):.3f}\n\t{(np.min(coverage_df.astype(float))/3e+6):.3f}-{(np.max(coverage_df.astype(float))/3e+6):.3f}')
    #print(f'Coverage:\n\t{(np.mean(coverage_df.astype(float))/3e+6):.3f} +- {(np.std(coverage_df.astype(float))/3e+6):.3f)}')
    #print(f'Coverage:\n\t{(np.mean(coverage_df.astype(float))/3e+6):.3f}')
    #print(f'Coverage:\n\t{(np.std(coverage_df.astype(float))/3e+6):3f}')

    ax.axhline(species_genome_sizes['A'], lw=1, c=species_colors['A'], label='OS-A')
    ax.axhline(species_genome_sizes['Bp'], lw=1, c=species_colors['Bp'], label="OS-B'")

    ax.legend(loc='center right', fontsize=10, frameon=True)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)

    return high_coverage_sags

def plot_recombination_length_distribution(recomb_df, params, savefig, bins=100):
    recomb_df['Length'] = recomb_df['End'] - recomb_df['Beg'] + 1

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Inferred segment length', fontsize=14)
    ax.set_ylabel('Probability density', fontsize=14)
    ax.set_yscale('log')
    ax.hist(recomb_df['Length'].values, bins=bins, density=True)

    delta = 1. / params.loc['1/delta', 'Posterior Mean']
    x = np.linspace(0, 10 * delta, 100)
    ax.plot(x, np.exp(-x / delta) / delta, '-k', lw=1)
    print(delta, recomb_df['Length'].mean())

    plt.tight_layout()
    plt.savefig(savefig)


def plot_mcorr_results(corr_df, gs, ec='k', label='data', ax_label=None):
    b = corr_df.loc[0, 'b']
    d_s = corr_df.loc[0, 'm']
    xvalues, yvalues = corr_df.loc[1:, ['l', 'm']].values.T
    fit_res = mcorr_fit.fit_model(xvalues, yvalues, d_s, mcorr_fit.const_r1)
    fit_data = FitData(xvalues, yvalues, d_s)
    plot_mcorr_fit(fit_data, fit_res, gs, ec=ec, label=label, ax_label=ax_label)

class FitData:
    '''Base class for collecting mcorr results'''
    def __init__(self, xvalues=None, yvalues=None, d_sample=None):
        self.xvalues = xvalues
        self.yvalues = yvalues
        self.d_sample = d_sample


def plot_mcorr_fit(fitdata, fitres, gs, title=None, ec='k', label='data', ax_label=None):
    """Fit all row data and do plotting for the full-recombination model"""
    xvalues = fitdata.xvalues
    yvalues = fitdata.yvalues

    ax1 = plt.subplot(gs[0, 0])
    #ax1 = plt.subplot(gs[0])
    ax1.scatter(xvalues, yvalues, s=20, facecolors='none', edgecolors=ec, label=label)
    predictions = yvalues + fitres.residual
    ax1.plot(xvalues, predictions, 'k', label='mcorr fit')
    ax1.set_ylabel(r'Correlation profile, $P$', fontsize=12)
    if np.min(yvalues) != np.max(yvalues):
        ax1.set_ylim([np.min(yvalues)*0.9, np.max(yvalues)*1.1])
    ax1.locator_params(axis='x', nbins=5)
    ax1.locator_params(axis='y', nbins=5)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.legend(frameon=False, fontsize=12)

    ax1.text(-0.2, 1.1, ax_label, transform=ax1.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)
    if title: plt.title(title, loc="left")

    ax2 = plt.subplot(gs[1, 0])
    #ax2 = plt.subplot(gs[1])
    markerline, _, _ = ax2.stem(xvalues,
                                fitres.residual,
                                linefmt='k-',
                                basefmt='r-',
                                markerfmt='ko')
    ax2.set_xlabel(r'Distance, $l$', fontsize=12)
    ax2.set_ylabel("Residual", fontsize=12)
    ax2.locator_params(axis='x', nbins=5)
    plt.setp(markerline, "markersize", 4)



###########################################################
# Gene hybridization figures
###########################################################

def make_gene_hybridization_figures(pangenome_map, args):
    global fig_count
    print(f'Printing gene hybridization supplemental figures...')

    hybrid_counts_table = pd.read_csv(f'{args.results_dir}main_figures_data/hybridization_counts_table.tsv', sep='\t', index_col=0)
    #main_figs.plot_hybridization_pie_chart(hybrid_counts_table, savefig=f'{args.figures_dir}S{fig_count}hybridization_pie_chart.pdf')
    #plot_hybridization_pie_chart(hybrid_counts_table, savefig=f'{args.figures_dir}S{fig_count}_hybridization_pie_chart.pdf')

    hybrid_counts_table = hybrid_counts_table.fillna(0)
    nonhybrid_og_ids = hybrid_counts_table.index.values[hybrid_counts_table[['total_transfers', 'M']].sum(axis=1) == 0]
    mosaic_og_ids = hybrid_counts_table.index.values[hybrid_counts_table['M'] > 0]
    singleton_hybrid_og_ids = hybrid_counts_table.index.values[(hybrid_counts_table['total_transfers'] == 1) & (hybrid_counts_table['M'] == 0)]
    nonsingleton_hybrid_og_ids = hybrid_counts_table.index.values[(hybrid_counts_table['total_transfers'] > 1) & (hybrid_counts_table['M'] == 0)]

    bins = [len(nonhybrid_og_ids), len(mosaic_og_ids), len(singleton_hybrid_og_ids), len(nonsingleton_hybrid_og_ids)]
    bin_labels = [f'no gene\nhybrids ({bins[0]})', f'mixed\nclusters ({bins[1]})', f'singleton\nhybrids ({bins[2]})', f'non-singleton\nhybrids ({bins[3]})']
    print(f'Random choice of non-hybrid OG IDs: {np.random.choice(nonhybrid_og_ids, 10)}')

    fig = plt.figure(figsize=(single_col_width, single_col_width))
    ax = fig.add_subplot(111)
    plot_labeled_pie_axis(ax, bins, bin_labels)
    ax.set_title(r'$\alpha-\beta$ core genes', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_hybridization_pie_chart.pdf')
    fig_count += 1


    # Plot A low and high diversity hybrid pie
    low_diversity_cutoff = 0.05
    syna_num_site_alleles = pd.read_csv(f'{args.results_dir}main_figures_data/A_num_site_alleles_4D.tsv', sep='\t', index_col=0)
    low_diversity_ogs = np.array(syna_num_site_alleles.loc[syna_num_site_alleles['fraction_polymorphic'] < low_diversity_cutoff, :].index)
    high_diversity_ogs = np.array([o for o in syna_num_site_alleles.index if o not in low_diversity_ogs])
    syna_hybrid_donor_frequency_table = hybrid_counts_table[['A', 'Bp->A', 'C->A', 'O->A', 'total_transfers', 'M']].fillna(0).astype(int)
    syna_hybrid_donor_frequency_table['total_transfers'] = syna_hybrid_donor_frequency_table[['Bp->A', 'C->A', 'O->A']].sum(axis=1)

    fig = plt.figure(figsize=(double_col_width, single_col_width))

    # High-diversity OGs
    ax = fig.add_subplot(121)
    high_diversity_og_hybrids = syna_hybrid_donor_frequency_table.loc[high_diversity_ogs, :]
    nonhybrid_og_ids = high_diversity_og_hybrids.index.values[high_diversity_og_hybrids[['total_transfers', 'M']].sum(axis=1) == 0]
    mosaic_og_ids = high_diversity_og_hybrids.index.values[high_diversity_og_hybrids['M'] > 0]
    singleton_hybrid_og_ids = high_diversity_og_hybrids.index.values[(high_diversity_og_hybrids['total_transfers'] == 1) & (high_diversity_og_hybrids['M'] == 0)]
    nonsingleton_hybrid_og_ids = high_diversity_og_hybrids.index.values[(high_diversity_og_hybrids['total_transfers'] > 1) & (high_diversity_og_hybrids['M'] == 0)]
    bins = [len(nonhybrid_og_ids), len(mosaic_og_ids), len(singleton_hybrid_og_ids), len(nonsingleton_hybrid_og_ids)]
    bin_labels = [f'no gene\nhybrids ({bins[0]})', f'mixed\nclusters ({bins[1]})', f'singleton\nhybrids ({bins[2]})', f'non-singleton\nhybrids ({bins[3]})']
    plot_labeled_pie_axis(ax, bins, bin_labels)
    ax.set_title(r'$\alpha$ hybrid genes', fontsize=16)
    ax.text(-0.2, 1.1, 'A', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)

    # Low-diversity OGs
    ax = fig.add_subplot(122)
    low_diversity_og_hybrids = syna_hybrid_donor_frequency_table.loc[low_diversity_ogs, :]
    nonhybrid_og_ids = low_diversity_og_hybrids.index.values[low_diversity_og_hybrids[['total_transfers', 'M']].sum(axis=1) == 0]
    mosaic_og_ids = low_diversity_og_hybrids.index.values[low_diversity_og_hybrids['M'] > 0]
    singleton_hybrid_og_ids = low_diversity_og_hybrids.index.values[(low_diversity_og_hybrids['total_transfers'] == 1) & (low_diversity_og_hybrids['M'] == 0)]
    nonsingleton_hybrid_og_ids = low_diversity_og_hybrids.index.values[(low_diversity_og_hybrids['total_transfers'] > 1) & (low_diversity_og_hybrids['M'] == 0)]
    bins = [len(nonhybrid_og_ids), len(mosaic_og_ids), len(singleton_hybrid_og_ids), len(nonsingleton_hybrid_og_ids)]
    bin_labels = [f'no gene\nhybrids ({bins[0]})', f'mixed\nclusters({bins[1]})', '', f'non-singleton\nhybrids ({bins[3]})']
    plot_labeled_pie_axis(ax, bins, bin_labels, startangle=0)
    ax.set_title(r'$\alpha$ backbone genes', fontsize=16)
    ax.text(-0.2, 1.1, 'B', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_A_hybrid_pie.pdf')
    fig_count += 1


    # Plot cluster pdist distribution
    og_table = pangenome_map.og_table
    clustered_og_table = pd.read_csv(f'{args.pangenome_dir}filtered_low_copy_clustered_orthogroup_table.tsv', sep='\t', index_col=0)
    #plot_pdist_distributions(og_table, args, f'{args.figures_dir}S{fig_count}A_orthogroup_pdist_distribution.pdf')
    #plot_species_clusters_distribution(clustered_og_table, args, f'{args.figures_dir}S{fig_count}B_species_clusters.pdf')
    #fig_count += 1

    plot_species_cluster_figure(og_table, clustered_og_table, args, f'{args.figures_dir}S{fig_count}_orthogroup_clusters.pdf')
    fig_count += 1


    main_figs.print_break()


def plot_hybridization_pie_chart(hybrid_counts_table, savefig=None, min_mosaic_fraction=0.5):
    hybrid_counts_table = hybrid_counts_table.fillna(0)
    nonhybrid_og_ids = hybrid_counts_table.index.values[hybrid_counts_table[['total_transfers', 'M']].sum(axis=1) == 0]
    mosaic_og_ids = hybrid_counts_table.index.values[hybrid_counts_table['M'] > 0]
    singleton_hybrid_og_ids = hybrid_counts_table.index.values[(hybrid_counts_table['total_transfers'] == 1) & (hybrid_counts_table['M'] == 0)]
    nonsingleton_hybrid_og_ids = hybrid_counts_table.index.values[(hybrid_counts_table['total_transfers'] > 1) & (hybrid_counts_table['M'] == 0)]

    bins = [len(nonhybrid_og_ids), len(mosaic_og_ids), len(singleton_hybrid_og_ids), len(nonsingleton_hybrid_og_ids)]
    bin_labels = [f'no gene\nhybrids ({bins[0]})', f'mosaic hybrids\nand other\nmixed clusters({bins[1]})', f'singleton\nhybrids\n({bins[2]})', f'non-singleton\nhybrids ({bins[3]})']

    text_props = {'size':10, 'color':'k'}
    text_fmt = r'%1.0f\%%'
    fig = plt.figure(figsize=(single_col_width, single_col_width))
    ax = fig.add_subplot(111)
    ax.pie(bins, labels=bin_labels, autopct=text_fmt, textprops=text_props, labeldistance=1.2)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    
def plot_labeled_pie_axis(ax, data, labels, startangle=-50):
    text_props = {'size':9, 'color':'w'}
    text_fmt = r'%1.0f\%%'
    wedges, texts, _ = ax.pie(data, startangle=startangle, autopct=text_fmt, textprops=text_props, pctdistance=0.75)
    #wedges, texts = ax.pie(data, startangle=0)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})

        if ang > 0.:
            ax.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                        horizontalalignment=horizontalalignment, fontsize=9, **kw)

def plot_pdist_distributions(og_table, args, savefig):
    pdist_values = []
    for o in og_table['parent_og_id'].unique():
        pdist_df = pickle.load(open(f'{args.pangenome_dir}pdist/{o}_trimmed_pdist.dat', 'rb'))
        pdist_values.append(utils.get_matrix_triangle_values(pdist_df.values, k=1))

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Pairwise distance, $\pi_{ij}$', fontsize=14)
    ax.set_ylabel('Probability density', fontsize=14)
    ax.hist(np.concatenate(pdist_values), bins=100, density=True)
    ax.axvline(0.075, lw=2, ls='--', c='tab:red')

    # Remove axis borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(savefig)
    plt.close()

    print(f'Number of unique orthogroups: {len(og_table.parent_og_id.unique())}')


def plot_species_cluster_figure(og_table, clustered_og_table, args, savefig):
    pdist_values = []
    for o in og_table['parent_og_id'].unique():
        pdist_df = pickle.load(open(f'{args.pangenome_dir}pdist/{o}_trimmed_pdist.dat', 'rb'))
        pdist_values.append(utils.get_matrix_triangle_values(pdist_df.values, k=1))

    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    
    # Orthogroup pdist distribution
    ax = fig.add_subplot(121)
    ax.text(-0.2, 1.05, 'A', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)
    ax.set_xlabel('Pairwise distance, $\pi_{ij}$', fontsize=14)
    ax.set_ylabel('Probability density', fontsize=14)
    ax.hist(np.concatenate(pdist_values), bins=100, density=True)
    ax.axvline(0.075, lw=2, ls='--', c='tab:red')

    # Remove axis borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Number of species clusters
    parent_ids, parent_counts = utils.sorted_unique(clustered_og_table['parent_og_id'].values)
    x, y = utils.sorted_unique(parent_counts, sort='ascending')
    ax = fig.add_subplot(122)
    ax.text(-0.2, 1.05, 'B', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)
    ax.set_xlabel('Species clusters', fontsize=14)
    ax.set_ylabel('Orthogroups', fontsize=14)
    ax.set_yscale('log')
    ax.set_ylim(8E-1, 1.5 * np.max(y))
    ax.plot(x, y, '-o', lw=2, ms=6, mfc='none', mec='tab:blue')

    # Remove axis borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if args.verbose:
        print(f'Number of unique orthogroups: {len(og_table.parent_og_id.unique())}')
        print('Species cluster distribution:')
        counts, hist = utils.sorted_unique(parent_counts, sort='ascending', sort_by='tag')
        print(f'Counts: {counts}')
        print(f'Number of clusters: {hist}; total: {np.sum(hist)}')
        print(f'Fraction of clusters: {hist / np.sum(hist)}')
        print('\n')

    plt.tight_layout()
    plt.savefig(savefig)
    plt.close()


def plot_species_clusters_distribution(clustered_og_table, args, savefig):
    parent_ids, parent_counts = utils.sorted_unique(clustered_og_table['parent_og_id'].values)


    # Plot OG species clusters
    x, y = utils.sorted_unique(parent_counts, sort='ascending')
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('species clusters', fontsize=14)
    ax.set_ylabel('number of orthogroups', fontsize=14)
    ax.set_yscale('log')
    ax.set_ylim(8E-1, 1.5 * np.max(y))
    ax.plot(x, y, '-o', lw=2, ms=6, mfc='none', mec='tab:blue')
    plt.tight_layout()
    #plt.savefig(f'{args.figures_dir}{args.fhead}species_clusters_distribution.pdf')
    plt.savefig(savefig)
    plt.close()

    if args.verbose:
        print('Species cluster distribution:')
        counts, hist = utils.sorted_unique(parent_counts, sort='ascending', sort_by='tag')
        print(f'Counts: {counts}')
        print(f'Number of clusters: {hist}; total: {np.sum(hist)}')
        print(f'Fraction of clusters: {hist / np.sum(hist)}')
        print('\n')



###########################################################
# Linkage block figures
###########################################################

def make_linkage_block_figures(pangenome_map, args):
    global fig_count
    print(f'Plotting linkage block supplemental figures...')

    metadata = MetadataMap()

    plot_dNdS_histograms(metadata, args, num_bins=50, p_cutoff=0.70, savefig=f'{args.figures_dir}S{fig_count}_block_dNdS.pdf')
    fig_count += 1

    plot_block_length_histogram(args, num_bins=50, savefig=f'{args.figures_dir}S{fig_count}_block_lengths.pdf')
    fig_count += 1

    plot_rrna_alignments(pangenome_map, metadata, args, fig_count)
    fig_count += 1

    main_figs.print_break()
       

def plot_dNdS_histograms(metadata, args, num_bins=50, p_cutoff=0.6, legend_fs=12, savefig=None):
    # Calculate species average pN and pS
    d_arr_raw, core_og_ids, sag_ids = pickle.load(open(f'{args.results_dir}main_figures_data/pS_array.dat', 'rb'))
    species_sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')
    A_idx = np.arange(len(sag_ids))[np.isin(sag_ids, species_sorted_sag_ids['A'])]
    Bp_idx = np.arange(len(sag_ids))[np.isin(sag_ids, species_sorted_sag_ids['Bp'])]
    pS_ABp = np.nanmean(d_arr_raw[:, A_idx, :][:, :, Bp_idx])
    dS_ABp = align_utils.calculate_divergence(pS_ABp)

    d_arr_raw, core_og_ids, sag_ids = pickle.load(open(f'{args.results_dir}main_figures_data/pN_array.dat', 'rb'))
    A_idx = np.arange(len(sag_ids))[np.isin(sag_ids, species_sorted_sag_ids['A'])]
    Bp_idx = np.arange(len(sag_ids))[np.isin(sag_ids, species_sorted_sag_ids['Bp'])]
    pN_ABp = np.nanmean(d_arr_raw[:, A_idx, :][:, :, Bp_idx])
    dN_ABp = align_utils.calculate_divergence(pN_ABp)


    fig = plt.figure(figsize=(double_col_width, single_col_width))

    # Analyze main allele divergences
    ax_labels = ['A', 'B']
    for i, species in enumerate(['A', 'Bp']):
        #f_haplotypes = f'{args.data_dir}{species}_all_sites_hybrid_linkage_block_stats.tsv'
        f_haplotypes = f'{args.results_dir}main_figures_data/{species}_core_snp_block_stats.tsv'
        block_diversity_stats = pd.read_csv(f_haplotypes, sep='\t', index_col=0)
        block_dNdS = block_diversity_stats[['dN_b', 'dS_b']].dropna()

        ax = fig.add_subplot(1, 2, i + 1)
        ax.set_xlabel('Divergence', fontsize=14)
        ax.set_ylabel('SNP blocks', fontsize=14)

        # Get unsaturated blocks
        dS = align_utils.calculate_divergence(block_dNdS.loc[(block_dNdS['dS_b'] < p_cutoff) & (block_dNdS['dN_b'] < p_cutoff), 'dS_b'].values)
        dN = align_utils.calculate_divergence(block_dNdS.loc[(block_dNdS['dS_b'] < p_cutoff) & (block_dNdS['dN_b'] < p_cutoff), 'dN_b'].values)

        x_bins = np.linspace(0, max([np.max(dS), np.max(dN)]), num_bins)
        ax.hist(dN, bins=x_bins, alpha=0.6, label=f'$d_N$')
        ax.hist(dS, bins=x_bins, alpha=0.6, label=f'$d_S$')
        #ax.hist(dN, bins=x_bins, histtype='step', lw=2, alpha=1.0, label=f'$d_N$')
        #ax.hist(dS, bins=x_bins, histtype='step', lw=2, alpha=1.0, label=f'$d_S$')

        # Get max histogram value
        dN_hist, _ = np.histogram(dN, bins=x_bins)
        dS_hist, _ = np.histogram(dS, bins=x_bins)
        #ax.text(-0.2, 1.1 * max(np.max(dS_hist), np.max(dN_hist)), ax_labels[i], fontsize=14, fontweight='bold', va='bottom')
        ax.text(-0.2, 1.05, ax_labels[i], transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)

        ax.axvline(dN_ABp, lw=2.0, color='tab:blue', label=r'$\alpha-\beta$ $d_N$')
        ax.axvline(dS_ABp, lw=2.0, color='tab:orange', label=r'$\alpha-\beta$ $d_S$')
        ax.legend(frameon=False, fontsize=legend_fs)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        print(f'{species}, <dN> = {np.mean(dN)}, <dS> = {np.mean(dS)}')

    print('\n')

    plt.tight_layout()
    plt.savefig(savefig)


def plot_block_length_histogram(args, num_bins=50, legend_fs=12, savefig=None):
    fig = plt.figure(figsize=(double_col_width, single_col_width))

    # Analyze main allele divergences
    species_colors = {'A':'tab:orange', 'Bp':'tab:blue'}
    species_labels = {'A':r'$\alpha$', 'Bp':r'$\beta$'}

    ax = fig.add_subplot(121)
    ax.set_xlabel('Block length (bp)', fontsize=14)
    ax.set_ylabel('SNP blocks', fontsize=14)
    ax.set_yscale('log')
    x_bins = np.linspace(0, 250, num_bins)
    for i, species in enumerate(['A', 'Bp']):
        #f_haplotypes = f'{args.data_dir}{species}_all_sites_hybrid_linkage_block_stats.tsv'
        f_haplotypes = f'{args.results_dir}main_figures_data/{species}_core_snp_block_stats.tsv'
        block_diversity_stats = pd.read_csv(f_haplotypes, sep='\t', index_col=0)
        block_lengths = (block_diversity_stats['x_end'] - block_diversity_stats['x_start'] + 1).values
        #ax.hist(block_lengths, bins=x_bins, alpha=0.5, color=species_colors[species], label=species_labels[species], density=True)
        ax.hist(block_lengths, bins=x_bins, alpha=0.5, color=species_colors[species], label=species_labels[species])
        print(species, np.mean(block_lengths), np.std(block_lengths))
    ax.text(-0.2, 1.05, 'A', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)
    ax.legend(frameon=False, fontsize=legend_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax = fig.add_subplot(122)
    ax.set_xlabel('Block length (SNPs)', fontsize=14)
    ax.set_ylabel('SNP blocks', fontsize=14)
    ax.set_yscale('log')
    x_bins = np.arange(0, 40)
    for i, species in enumerate(['A', 'Bp']):
        #f_haplotypes = f'{args.data_dir}{species}_all_sites_hybrid_linkage_block_stats.tsv'
        f_haplotypes = f'{args.results_dir}main_figures_data/{species}_core_snp_block_stats.tsv'
        block_diversity_stats = pd.read_csv(f_haplotypes, sep='\t', index_col=0)
        #ax.hist(block_diversity_stats['num_snps'], bins=x_bins, alpha=0.5, color=species_colors[species], label=species_labels[species], density=True)
        ax.hist(block_diversity_stats['num_snps'], bins=x_bins, alpha=0.5, color=species_colors[species], label=species_labels[species])
        print(species, np.mean(block_diversity_stats['num_snps']), np.std(block_diversity_stats['num_snps']), np.sum(block_diversity_stats['num_snps']))
    ax.text(-0.2, 1.05, 'B', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)
    ax.legend(frameon=False, fontsize=legend_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    plt.tight_layout()
    #plt.savefig(f'{args.figures_dir}S{fig_count}_block_length_hist.pdf')
    plt.savefig(savefig)

def plot_rrna_alignments(pangenome_map, metadata, args, fig_count, fig_dpi=1000):
    fig = plt.figure(figsize=(double_col_width, 1.4 * single_col_width))
    gspec = gridspec.GridSpec(2, 2, width_ratios=[1, 1.0])

    i_ascii = 65

    ax = plt.subplot(gspec[0, :])
    #f_16S_aln = f'{args.results_dir}supplement/16S_rRNA_aln.fna'
    f_16S_aln = f'{args.results_dir}main_figures_data/16S_rRNA_manual_aln.fna'
    aln = seq_utils.read_alignment(f_16S_aln)
    #aln_plot = aln[:, 0:801]
    aln_plot = aln[:, 190:1080]
    species_grouping = align_utils.sort_aln_rec_ids(aln_plot, pangenome_map, metadata)
    lw = aln_plot.get_alignment_length() / 100
    plot_alignment(aln_plot, annotation=species_grouping, annotation_style='lines', marker_size=lw, reference=0, ax=ax)
    ax.text(-0.05, 1.05, chr(i_ascii), fontsize=10, fontweight='bold', va='center', transform=ax.transAxes)

    xticks = np.arange(10, aln_plot.get_alignment_length() + 1, 100)
    xticklabels = xticks + 190
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # Manually highlight mosaic regions
    block_idx = np.arange(len(aln_plot))
    main_figs.highlight_block_haplotype(ax, block_idx, 150, 390, -0.5, open_colors['grape'][4], 0.1)
    main_figs.highlight_block_haplotype(ax, block_idx, 870, 890, -0.5, open_colors['grape'][4], 0.1)

    for i, og_id in enumerate(['YSG_0713', 'YSG_1007']):
        ax = plt.subplot(gspec[1, i])
        f_aln = f'{args.results_dir}supplement/{og_id}_manual_aln.fna'
        aln = seq_utils.read_alignment(f_aln)

        if i == 0:
            aln_plot = aln
            block_idx = np.arange(len(aln_plot))
            main_figs.highlight_block_haplotype(ax, block_idx, 190, 288, -0.5, open_colors['grape'][4], 0.1)
        else:
            aln_plot = aln[:78, 0:301]
            block_idx = np.arange(len(aln_plot))
            main_figs.highlight_block_haplotype(ax, block_idx, 150, 205, -0.5, open_colors['grape'][4], 0.1)


        species_grouping = align_utils.sort_aln_rec_ids(aln_plot, pangenome_map, metadata)
        lw = aln_plot.get_alignment_length() / 50
        plot_alignment(aln_plot, annotation=species_grouping, annotation_style='lines', marker_size=lw, reference=0, ax=ax)
        ax.text(-0.05, 1.05, chr(i_ascii + i + 1), fontsize=10, fontweight='bold', va='center', transform=ax.transAxes)


    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_rrna_alignments.pdf', dpi=fig_dpi)

    return fig_count + 1


###########################################################
# Sample variation figures
###########################################################

def make_sample_variation_figures(pangenome_map, args, label_fs=14, num_bins=50, ms=4, legend_fs=12):
    global fig_count
    print(f'Plotting sample variation supplemental figures...')

    # Calculate A backbone pdist
    metadata = MetadataMap()

    # Get A backbone OGs
    low_diversity_cutoff = 0.05
    syna_num_site_alleles = pd.read_csv(f'{args.results_dir}main_figures_data/A_num_site_alleles_4D.tsv', sep='\t', index_col=0)
    low_diversity_ogs = np.array(syna_num_site_alleles.loc[syna_num_site_alleles['fraction_polymorphic'] < low_diversity_cutoff, :].index)
    high_diversity_ogs = np.array([o for o in syna_num_site_alleles.index if o not in low_diversity_ogs])

    # Calculate backbone pdist
    #d_arr_raw, core_og_ids, sag_ids = pickle.load(open(f'{args.results_dir}main_figures_data/pdist_array.dat', 'rb'))
    d_arr_raw, core_og_ids, sag_ids = pickle.load(open(f'{args.results_dir}main_figures_data/pS_array.dat', 'rb'))
    A_backbone_idx = np.array([i for i in range(d_arr_raw.shape[0]) if core_og_ids[i] in low_diversity_ogs])
    d_backbone_arr = d_arr_raw[A_backbone_idx]

    pdist_mean = pd.DataFrame(np.nanmean(d_backbone_arr, axis=0), index=sag_ids, columns=sag_ids)
    species_sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')
    syna_sag_ids = np.array(species_sorted_sag_ids['A'])
    syna_sorted_sag_ids = metadata.sort_sags(syna_sag_ids, by='location')

    pdist_mean = pdist_mean.loc[syna_sag_ids, syna_sag_ids]
    high_divergence_idx = np.where(pdist_mean.values > 0.01)

    # Alpha low-diversity comparison between springs
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Mean pair divergence, $\pi_{ij}$', fontsize=label_fs)
    ax.set_xscale('log')
    ax.set_ylabel('Reverse cumulative', fontsize=label_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    xlim = (1E-4, 2E-1)
    pdist_values = utils.get_matrix_triangle_values(pdist_mean.values, k=1)
    x_bins = np.geomspace(*xlim, num_bins)

    spring_colors = {'OS':'tab:cyan', 'MS':'tab:purple'}
    for spring in syna_sorted_sag_ids:
        spring_sag_ids = syna_sorted_sag_ids[spring]
        pdist_values = utils.get_matrix_triangle_values(pdist_mean.loc[spring_sag_ids, spring_sag_ids].values, k=1)
        y = np.array([np.sum(pdist_values > x) / len(pdist_values) for x in x_bins])
        n = len(syna_sorted_sag_ids[spring])
        ax.plot(x_bins, y, f'-s', lw=1, ms=ms, color=spring_colors[spring], alpha=0.5, mfc='none', label=f'{spring} (n={n:d})')

        '''
        if spring == 'OS':
            subsampled_sag_ids = rng.choice(spring_sag_ids, size=len(syna_sorted_sag_ids['MS']))
            pdist_values = utils.get_matrix_triangle_values(pdist_mean.loc[subsampled_sag_ids, subsampled_sag_ids].values, k=1)
            y = np.array([np.sum(pdist_values > x) / len(pdist_values) for x in x_bins])
            ax.plot(x_bins, y, f'-^', lw=1, ms=ms, color=spring_colors[spring], alpha=0.5, mfc='none', label=f'{spring} subsampled')
        '''
        #print(spring, x_bins[y < 0.1], np.sum(y < 0.1))

    # Between springs comparison
    pdist_values = pdist_mean.loc[syna_sorted_sag_ids['OS'],syna_sorted_sag_ids['MS']].values.flatten()
    y = np.array([np.sum(pdist_values > x) / len(pdist_values) for x in x_bins])
    ax.plot(x_bins, y, f'-D', lw=1, ms=3, color='tab:red', alpha=0.5, mfc='none', label='OS vs MS')

    ax.legend(frameon=False, fontsize=legend_fs)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_alpha_location_diversity.pdf')
    plt.close()
    fig_count += 1


    # Simple hybrids per sample
    species_cluster_genomes = pd.read_csv(f'{args.results_dir}hybridization/labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)
    transfer_type_distribution = calculate_hybridization_spectrum(species_cluster_genomes, pangenome_map, metadata)
    plot_hybridization_spectra(transfer_type_distribution, pangenome_map, metadata, args, savefig=f'{args.figures_dir}S{fig_count}_simple_hybrids_by_sample.pdf')
    fig_count += 1

    main_figs.print_break()


def plot_hybridization_spectra(transfer_type_distribution, pangenome_map, metadata, args, savefig):
    sag_ids = pangenome_map.get_sag_ids()
    sample_sorted_sags = metadata.sort_sags(sag_ids, by='sample')
    sample_ids = np.sort(list(sample_sorted_sags.keys()))
    sag_species = ['A', 'Bp', 'C']
    all_transfer_cols = ['Bp->A', 'C->A', 'O->A', 'total->A', 'A->Bp', 'C->Bp', 'O->Bp', 'total->Bp']
    donor_label_dict = {'A':r'$\alpha$', 'Bp':r'$\beta$', 'C':r'$\gamma$', 'O':'X', 'total':'All'}
    #transfer_label = f'{donor_label_dict[s]}$\\rightarrow${donor_label_dict[species]}'

    # Plot species composition by sample
    ax_labels = ['A', 'B']
    line_styles = ['-o', '--s', '-.x', ':D']
    colors = ['tab:purple', 'tab:cyan']
    x = np.arange(3)

    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))

    for i, host in enumerate(['A', 'Bp']):
        transfer_cols = [col for col in all_transfer_cols if f'->{host}' in col]
        x = np.arange(len(transfer_cols))
        ax = fig.add_subplot(1, 2, i + 1)
        ax.set_xticks(x)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Make tick labels
        xticklabels = []
        for c in transfer_cols:
            d, h = c.split('->')
            xticklabels.append(f'{donor_label_dict[d]}$\\rightarrow${donor_label_dict[h]}')
        ax.set_xticklabels(xticklabels, fontsize=14)

        if i == 0:
            ax.set_ylabel('Simple hybrid transfers', fontsize=14)
        ax.set_ylim(-0.9, 1.2 * np.max(transfer_type_distribution[transfer_cols].values))
        #ax.set_ylim(-0.9, 1.2 * np.max(transfer_type_distribution[allele_transfers].values))

        for j, sample in enumerate(sample_ids):
            #y = transfer_type_distribution.loc[sample, allele_transfers].values
            y = transfer_type_distribution.loc[sample, transfer_cols].values
            line_idx = j % 4
            color_idx = j // 4
            ax.plot(x, y, line_styles[line_idx], color=colors[color_idx], ms=4, label=sample)

        ax.text(-0.2, 1.05, ax_labels[i], transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)
        ax.legend(ncol=2, fontsize=8, frameon=False, handlelength=3)

    plt.tight_layout()
    plt.savefig(savefig)
    plt.close()

    #plt.savefig(f'{args.figures_dir}S{fig_count}_simple_hybrids_by_sample.pdf')
    #plt.close()



def calculate_hybridization_spectrum(species_cluster_genomes, pangenome_map, metadata):
    sag_ids = pangenome_map.get_sag_ids()
    sample_sorted_sags = metadata.sort_sags(sag_ids, by='sample')
    sample_ids = np.sort(list(sample_sorted_sags.keys()))

    donor_list = ['A', 'Bp', 'C', 'O']

    # Break down gene hybridizations by sample and species
    sag_species = ['A', 'Bp', 'C']
    allele_transfers = ['Bp->A', 'C->A', 'O->A', 'total->A', 'A->Bp', 'C->Bp', 'O->Bp', 'total->Bp']
    transfer_type_distribution = pd.DataFrame(index=sample_ids, columns=sag_species + allele_transfers)

    genome_table_stats_columns = np.array([col for col in species_cluster_genomes.columns if 'Uncmic' not in col])
    for sample in sample_ids:
        sample_cluster_genomes = species_cluster_genomes[np.concatenate([genome_table_stats_columns, sample_sorted_sags[sample]])]
        species_sorted_sags = metadata.sort_sags(sample_sorted_sags[sample], by='species')
        for s in ['A', 'Bp', 'C']:
            transfer_type_distribution.loc[sample, s] = len(species_sorted_sags[s])

        for species in ['A', 'Bp']:
            species_hybrid_donor_frequency_table = make_donor_frequency_table(sample_cluster_genomes, species, pangenome_map, metadata)
            species_donors = [s for s in donor_list if s != species]

            for d in species_donors:
                transfer_type_distribution.loc[sample, f'{d}->{species}'] = species_hybrid_donor_frequency_table[d].sum()

            transfer_type_distribution.loc[sample, f'total->{species}'] = transfer_type_distribution.loc[sample, [f'{d}->{species}' for d in species_donors]].sum()

    return transfer_type_distribution


def make_donor_frequency_table(species_cluster_genomes, species, pangenome_map, metadata):
    if species == 'A':
        species_core_genome_clusters = species_cluster_genomes.loc[species_cluster_genomes['osa_location'].dropna().index, :].sort_values('osa_location')
        species_core_genome_clusters = species_core_genome_clusters.loc[species_core_genome_clusters['core_A'] == 'Yes', :]
    elif species == 'Bp':
        species_core_genome_clusters = species_cluster_genomes.loc[species_cluster_genomes['osbp_location'].dropna().index, :].sort_values('osbp_location')
        species_core_genome_clusters = species_core_genome_clusters.loc[species_core_genome_clusters['core_Bp'] == 'Yes', :]

    # Initialize frequency table
    sag_ids = np.array([col for col in species_cluster_genomes.columns if 'Uncmic' in col])
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    donor_freq_table = pd.DataFrame(index=species_core_genome_clusters.index, columns=['CYA_tag', 'CYB_tag', 'osa_location', 'osbp_location', 'A', 'Bp', 'C', 'O'])
    donor_freq_table[['CYA_tag', 'CYB_tag', 'osa_location', 'osbp_location']] = species_core_genome_clusters[['CYA_tag', 'CYB_tag', 'osa_location', 'osbp_location']].values
    donor_freq_table[['A', 'Bp', 'C', 'O']] = 0

    #Fill table
    for o in donor_freq_table.index:
        #genome_clusters = species_core_genome_clusters.loc[o, species_sorted_sags[species]].dropna().replace({'a':'A', 'b':'Bp'})
        #unique_clusters, cluster_counts = utils.sorted_unique(genome_clusters)
        #donor_freq_table.loc[o, unique_clusters] = cluster_counts
        #seq_clusters = np.concatenate([[utils.split_alphanumeric_string(s)[0] for s in c.split(',')] for c in species_core_genome_clusters.loc[o, species_sorted_sags[species]].dropna().replace({'a':'A', 'b':'Bp'}).values])
        gene_clusters = species_core_genome_clusters.loc[o, species_sorted_sags[species]].dropna().replace({'a':'A', 'b':'Bp'}).values
        if len(gene_clusters) > 0:
            seq_clusters = np.concatenate([[utils.split_alphanumeric_string(s)[0] for s in c.split(',')] for c in gene_clusters])
            unique_clusters, cluster_counts = utils.sorted_unique(seq_clusters)
            donor_freq_table.loc[o, unique_clusters] = cluster_counts

    return donor_freq_table


###########################################################
# Hybridization quality control
###########################################################

def make_hybridization_qc_figures(pangenome_map, args, low_diversity_cutoff=0.05):
    global fig_count
    print(f'Plotting QC supplemental figures...')

    metadata = MetadataMap()
    sag_ids = pangenome_map.get_sag_ids()
    syna_sags = metadata.sort_sags(sag_ids, by='species')['A']
    pure_syna_sample_sags, mixed_syna_sample_sags = make_syna_test_samples(pangenome_map, metadata)

    # Split OGs by diversity
    syna_num_site_alleles = pd.read_csv(f'{args.results_dir}main_figures_data/A_num_site_alleles_4D.tsv', sep='\t', index_col=0)
    low_diversity_ogs = np.array(syna_num_site_alleles.loc[syna_num_site_alleles['fraction_polymorphic'] < low_diversity_cutoff, :].index)
    high_diversity_ogs = np.array([o for o in syna_num_site_alleles.index if o not in low_diversity_ogs])
    og_ids = np.concatenate([low_diversity_ogs, high_diversity_ogs])

    plot_species_composition_timeseries(metadata, f'{args.figures_dir}S{fig_count}_species_composition.pdf')
    fig_count += 1

    plot_og_presence_model_validation(pangenome_map, low_diversity_ogs, pure_syna_sample_sags, f'{args.figures_dir}S{fig_count}_og_presence_model_fit.pdf')
    fig_count += 1

    plot_og_presence_tests(pangenome_map, (low_diversity_ogs, high_diversity_ogs), (pure_syna_sample_sags, mixed_syna_sample_sags), f'{args.figures_dir}S{fig_count}_og_presence_tests.pdf')
    fig_count += 1
    '''
    '''

    plot_block_distributions_sample_comparisons('../results/single-cell/supplement/quality_control/', f'{args.figures_dir}S{fig_count}_block_distribution_comparison.pdf')
    fig_count += 1

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
    ax.set_xlabel('Year', fontsize=label_fs)
    ax.set_xticks(xticks)
    ax.set_ylabel(r'$\alpha$ relative abundance', fontsize=label_fs)
    ax.set_yticks(yticks)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    p, p_err = estimate_binomial_fraction(os_num)
    ax.errorbar(os_t, 1 - p[:, 1], yerr=p_err[::-1], fmt='-o', mec='none', mfc='tab:cyan', c='tab:cyan', elinewidth=1, capsize=3, label='OS')
    p, p_err = estimate_binomial_fraction(ms_num)
    ax.errorbar(ms_t, 1 - p[:, 1], yerr=p_err[::-1], fmt='-o', mec='none', mfc='tab:purple', c='tab:purple', elinewidth=1, capsize=3, label='MS')
    ax.legend(fontsize=legend_fs, frameon=False)
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
    w_ratio = 15
    #ratio = 1
    fig = plt.figure(figsize=(double_col_width, (2 / (1 + h_ratio)) * double_col_width), constrained_layout=True)
    gspec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig, height_ratios=[h_ratio, 1], width_ratios=[w_ratio, w_ratio, 1])

    ax = plt.subplot(gspec[0, 0])
    ax.text(-0.2, 1.05, 'A', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)
    ax.imshow(presence_df.loc[sorted_sag_ids, sorted_og_ids].values, aspect='auto', vmin=0, vmax=1, cmap='Greys', interpolation='nearest')
    ax.set_xlabel('Gene rank', fontsize=12)
    ax.set_ylabel('Cell rank', fontsize=12)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.xaxis.set_label_position('top') 

    ax = plt.subplot(gspec[1, 0])
    ax.text(-0.2, 1.05, 'C', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)
    og_coverage = np.sum(presence_df.values, axis=0) / N
    plot_presence_distribution(ax, og_coverage, fit_params=(mu_coverage, sigma_coverage), label='data')
    ax.legend(fontsize=10, frameon=False)

    ax = plt.subplot(gspec[0, 1])
    ax.text(-0.2, 1.05, 'B', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)
    cmap = plt.get_cmap('coolwarm')
    #cmap = plt.get_cmap('bwr')
    norm = mpl.colors.TwoSlopeNorm(vmin=np.min(covariance_df.values), vcenter=0, vmax=np.max(covariance_df.values))
    im = ax.imshow(covariance_df.loc[sorted_og_ids, sorted_og_ids].values, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    #im = ax.imshow(covariance_df.loc[sorted_og_ids, sorted_og_ids].values, aspect='auto', cmap=cmap, interpolation='nearest')
    ticks = np.arange(0, L, 100)
    #ticks = np.arange(0, 801, 200)
    ax.set_xlabel('Gene rank', fontsize=12)
    ax.set_xticks(ticks)
    ax.set_ylabel('Gene rank', fontsize=12)
    ax.set_yticks(ticks)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.xaxis.set_label_position('top') 

    ax = plt.subplot(gspec[0, 2])
    plt.colorbar(im, cax=ax, ticks=[-0.1, -0.07, -0.03, 0, 0.1, 0.2, 0.3], label='Covariance')

    ax = plt.subplot(gspec[1, 1])
    ax.text(-0.2, 1.05, 'D', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)
    ax.set_xlim(-0.1, 0.5)
    ax.set_ylim(-0.02, 30) 
    ax.set_xlabel('Gene coverage covariance')
    ax.set_ylabel('Probability density')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

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
    ax.plot(x_bins, y_aa, color='tab:orange', label='$C_{aa}$ null model')

    ax.hist(C_ab_values, bins=x_bins, color='tab:blue', alpha=0.6, density=True)
    y_ab = stats.norm.pdf(x_bins, loc=0, scale=sigma_ab)
    ax.plot(x_bins, y_ab, color='tab:green', label='$C_{ab}$ null model')
    ax.legend(fontsize=10, frameon=False)

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

def plot_presence_distribution(ax, og_coverage, fit_params=None, x_bins=None, num_bins=47, alpha=1.0, xlabel='Gene coverage probability', ylabel='Probability density', label='', fit_label='fit', label_fs=12, legend_fs=12, **hist_kwargs):
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

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
        ax.legend(fontsize=legend_fs, frameon=False)


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
    #ax.text(-0.1, 1.1 * ymax, 'A', fontweight='bold', fontsize=16)
    #plot_presence_distribution(ax, group1_og_coverage, alpha=0.5, label='mixed species OGs')
    #plot_presence_distribution(ax, group2_og_coverage, alpha=0.5, label='pure species OGs', label_fs=14, legend_fs=10)
    coverage_values = np.concatenate([group1_og_coverage, group2_og_coverage])
    x_bins = np.linspace(np.min(coverage_values) - epsilon, np.max(coverage_values) + epsilon, num_bins)

    plot_presence_distribution(ax, group1_og_coverage, alpha=0.5, label='mixed species OGs', x_bins=x_bins, cumulative=True, histtype='step', lw=2)
    plot_presence_distribution(ax, group2_og_coverage, alpha=0.5, label='pure species OGs', ylabel='Cumulative', label_fs=16, legend_fs=10, x_bins=x_bins, cumulative=True, histtype='step', lw=2)
    #ax.set_xlim(0, 1.1)
    ax.set_ylim(-0.02, ymax)
    ax.set_xlim(x_bins[0] - 0.1, x_bins[-1])
    #ax.set_ylim(0.0, ymax)
    ax.legend(loc='upper left', fontsize=10, frameon=False)
    ax.text(-0.1, 1.1 * ymax, 'A', fontweight='bold', fontsize=10)

    ax = plt.subplot(gspec[0, 1])
    ymax = 1.2
    #ax.text(-0.40, 1.1 * ymax, 'B', fontweight='bold', fontsize=16)
    #z_group1 = group1_og_coverage - np.mean(group2_og_coverage)
    #z_group3 = group3_og_coverage - np.mean(group4_og_coverage)
    mu_group1 = np.mean(group1_og_coverage)
    sigma_group1 = np.std(group1_og_coverage)
    z_group1 = group1_og_coverage - mu_group1
    z_group3 = group3_og_coverage - np.mean(group3_og_coverage)
    z_values = np.concatenate([z_group1, z_group3])
    x_bins = np.linspace(np.min(z_values) - epsilon, np.max(z_values) + epsilon, num_bins)
    plot_presence_distribution(ax, z_group1, x_bins=x_bins, alpha=0.5, label=r'$\alpha$ only', ylabel='', cumulative=True, histtype='step', lw=2)
    plot_presence_distribution(ax, z_group3, x_bins=x_bins, fit_params=(0, sigma_group1), alpha=0.5, xlabel='Mean-centered coverage', ylabel='', label=r' $\alpha-\beta$ mixed', label_fs=16, legend_fs=10, fit_label=r'$\alpha$ only fit', cumulative=True, histtype='step', lw=2)
    ax.set_xlim(x_bins[0] - 0.1, x_bins[-1])
    ax.set_ylim(-0.02, ymax)
    ax.text(-0.40, 1.1 * ymax, 'B', fontweight='bold', fontsize=10)
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
    #pure_syna_sample_block_stats = pd.read_csv(f'{input_dir}pure_syna_sample_4D_sites_hybrid_linkage_block_stats.tsv', sep='\t', index_col=0)
    #mixed_syna_sample_block_stats = pd.read_csv(f'{input_dir}mixed_syna_sample_4D_sites_hybrid_linkage_block_stats.tsv', sep='\t', index_col=0)
    pure_syna_sample_block_stats = pd.read_csv(f'{input_dir}pure_syna_sample_snp_block_stats.tsv', sep='\t', index_col=0)
    mixed_syna_sample_block_stats = pd.read_csv(f'{input_dir}mixed_syna_sample_snp_block_stats.tsv', sep='\t', index_col=0)
    bootstrap_results = read_bootstrap_results(input_dir)
    print(bootstrap_results)

    # Set up figure
    nrows = 1
    ncols = 3
    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    #gspec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    #ax = plt.subplot(gspec[0, 0])
    ax = fig.add_subplot(131)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.text(-0.2, 1.1, 'A', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)

    ax_label_hratio = 1.15
    ymax = 0.04
    num_blocks_values = bootstrap_results['number of blocks'].values
    x0 = np.min(num_blocks_values) - 1
    x1 = np.max(num_blocks_values) + 2
    #x_bins = np.arange(np.min(num_blocks_values) - 1, np.max(num_blocks_values) + 2, 4)
    #x_bins = np.arange(np.min(num_blocks_values) - 1, np.max(num_blocks_values) + 2, int((x1 - x0) / 15))
    x_bins = np.linspace(np.min(num_blocks_values) - 1, np.max(num_blocks_values) + 2, 15)
    plot_presence_distribution(ax, num_blocks_values[1:], x_bins=x_bins, alpha=1.0, xlabel='SNP blocks', label_fs=14)
    ax.set_ylim(0, ymax)
    ax.annotate('', xy=(num_blocks_values[0], 0.02), xycoords='data', 
            xytext=(num_blocks_values[0], 0.03),
            arrowprops=dict(facecolor='black', width=3, headwidth=8),
            horizontalalignment='center', verticalalignment='top',
            )

    #ax = plt.subplot(gspec[0, 1])
    ax = fig.add_subplot(132)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.text(-0.2, 1.1, 'B', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)

    ymax = 0.3
    #ax.text(-5, ax_label_hratio * ymax, 'B', fontweight='bold', fontsize=16)
    group1_lengths = pure_syna_sample_block_stats['num_snps'].values
    group2_lengths = mixed_syna_sample_block_stats['num_snps'].values
    length_values = np.concatenate([group1_lengths, group2_lengths])
    x_bins = np.arange(5, np.max(length_values) + 2)
    plot_presence_distribution(ax, group1_lengths, x_bins=x_bins, alpha=0.5, xlabel='Block length (SNPs)', ylabel=None, label=r'$\alpha$ only', label_fs=14)
    plot_presence_distribution(ax, group2_lengths, x_bins=x_bins, alpha=0.5, xlabel='Block length (SNPs)', ylabel=None, label=r'$\alpha-\beta$ mixed', label_fs=14, legend_fs=10)
    #ax.set_xticks(np.arange(5, x_bins[-1], 5))
    ax.set_xticks(np.arange(5, x_bins[-1], 10))
    ax.set_ylim(0, ymax)

    ks_stat, ks_pvalue = stats.kstest(group1_lengths, group2_lengths)
    print(f'OS05 vs mixed-species samples block lengths: {ks_stat:.4f}, {ks_pvalue:.1e}')
    print(np.mean(group1_lengths), np.std(group1_lengths))
    print(np.mean(group2_lengths), np.std(group2_lengths))
    print('\n')


    #ax = plt.subplot(gspec[0, 2])
    ax = fig.add_subplot(133)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.text(-0.2, 1.1, 'C', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center', usetex=False)

    ymax = 4 
    #ax.text(-0.15, ax_label_hratio * ymax, 'C', fontweight='bold', fontsize=16)
    group1_dS = pure_syna_sample_block_stats['dS_b'].values
    group2_dS = mixed_syna_sample_block_stats['dS_b'].values
    dS_values = np.concatenate([group1_dS, group2_dS])
    epsilon = 0.1
    num_bins = 23
    x_bins = np.linspace(0, np.max(dS_values) + epsilon, num_bins)
    plot_presence_distribution(ax, group1_dS, x_bins=x_bins, alpha=0.5, ylabel=None, label=r'$\alpha$ only', label_fs=14)
    plot_presence_distribution(ax, group2_dS, x_bins=x_bins, alpha=0.5, xlabel=r'Hapl. divergence, $d_{\alpha_1 \alpha_2}$', ylabel=None, label=r'$\alpha-\beta$ mixed', label_fs=14, legend_fs=10)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, ymax)

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
    #pure_syna_sample_block_stats = pd.read_csv(f'{input_dir}pure_syna_sample_4D_sites_hybrid_linkage_block_stats.tsv', sep='\t', index_col=0)
    pure_syna_sample_block_stats = pd.read_csv(f'{input_dir}pure_syna_sample_snp_block_stats.tsv', sep='\t', index_col=0)
    #pure_syna_sample_block_stats = pd.read_csv(f'{input_dir}pure_syna_sample_snp_block_stats_v1.tsv', sep='\t', index_col=0)
    bootstrap_results.loc['test_sample', 'number of blocks'] = len(pure_syna_sample_block_stats)
    bootstrap_results.loc['test_sample', 'mean block length'] = pure_syna_sample_block_stats['num_snps'].mean()
    bootstrap_results.loc['test_sample', 'mean dS_b'] = pure_syna_sample_block_stats['dS_b'].mean()

    for i in range(1, num_replicas+1):
        #sample_block_stats = pd.read_csv(f'{input_dir}mixed_syna_sample{i}_4D_sites_hybrid_linkage_block_stats.tsv', sep='\t', index_col=0)
        sample_block_stats = pd.read_csv(f'{input_dir}mixed_syna_sample_{i}_snp_block_stats.tsv', sep='\t', index_col=0)
        bootstrap_results.loc[f'sample{i}', 'number of blocks'] = len(sample_block_stats)
        bootstrap_results.loc[f'sample{i}', 'mean block length'] = sample_block_stats['num_snps'].mean()
        bootstrap_results.loc[f'sample{i}', 'mean dS_b'] = sample_block_stats['dS_b'].mean()

    return bootstrap_results


if __name__ == '__main__':
    # Default variables
    figures_dir = '../figures/supplement/'
    pangenome_dir = '../results/single-cell/sscs_pangenome_v2/'
    results_dir = '../results/single-cell/'
    output_dir = '../results/single-cell/supplement/'
    metagenome_dir = f'../results/metagenome/'
    #annotations_dir = '../data/single-cell/filtered_annotations/sscs/'
    annotations_dir = '../data/single-cell/contig_annotations/'
    f_orthogroup_table = f'{pangenome_dir}filtered_low_copy_clustered_core_mapped_labeled_cleaned_orthogroup_table.tsv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--figures_dir', default=figures_dir, help='Directory where figures are saved.')
    parser.add_argument('-L', '--linkage_dir', default='../results/single-cell/v1_data/')
    parser.add_argument('-M', '--metagenome_dir', default=metagenome_dir, help='Directory with results for metagenome.')
    parser.add_argument('-N', '--annotations_dir', default=annotations_dir, help='Directory with annotation files.')
    parser.add_argument('-P', '--pangenome_dir', default=pangenome_dir, help='Pangenome directory.')
    parser.add_argument('-R', '--results_dir', default=results_dir, help='Main results directory.')
    parser.add_argument('-O', '--output_dir', default=output_dir, help='Directory in which supplemental data is saved.')
    parser.add_argument('-g', '--orthogroup_table', default=f_orthogroup_table, help='File with orthogroup table.')
    parser.add_argument('-r', '--random_seed', default=12345, type=int, help='Seed for RNG.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    args = parser.parse_args()


    pangenome_map = pg_utils.PangenomeMap(gff_files_dir=args.annotations_dir, f_orthogroup_table=args.orthogroup_table)
    rng = np.random.default_rng(args.random_seed)
    make_genome_clusters_figures(pangenome_map, rng, args)
    #fig_count = 6
    make_metagenome_recruitment_figures(pangenome_map, args)
    #fig_count = 11
    make_linkage_figures(pangenome_map, args)
    make_gene_hybridization_figures(pangenome_map, args)
    #fig_count = 17
    make_linkage_block_figures(pangenome_map, args)
    #fig_count = 19
    make_sample_variation_figures(pangenome_map, args)
    #fig_count = 21
    make_hybridization_qc_figures(pangenome_map, args)

