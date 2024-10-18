import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ete3
import utils
import pickle
import gatc
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pandas as pd
import scipy.sparse as sparse
import scipy.cluster.hierarchy as hclust
import matplotlib.gridspec as gridspec
import copy
import scipy.stats as stats
from sklearn.neighbors import KernelDensity
from ete3 import Tree, TreeStyle, TextFace, NodeStyle, Face, ClusterTree, ProfileFace
from syn_homolog_map import SynHomologMap
from alignment_tools import calculate_consensus_distance
#from seq_processing_utils import VisualGene
from matplotlib.transforms import Bbox, TransformedBbox, blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector, BboxConnectorPatch


# Configure matplotlib environment
helvetica_scale_factor = 0.92 # rescale Helvetica to other fonts of same size
mpl.rcParams['font.size'] = 12 * helvetica_scale_factor
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['axes.titlesize'] = 12 * helvetica_scale_factor
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

# Make figure background transparent
mpl.rcParams['figure.facecolor'] = (0, 0, 0, 0.)

cm = plt.get_cmap('tab10')
color_dict = {'blue':cm(0), 'orange':cm(1), 'green':cm(2), 'red':cm(3), 'pink':cm(6), 'olive':cm(8), 'cyan':cm(9)}

# Define open-colors dictionary
open_colors = {
    'gray': {
        0: '#f8f9fa',
        1: '#f1f3f5',
        2: '#e9ecef',
        3: '#dee2e6',
        4: '#ced4da',
        5: '#adb5bd',
        6: '#868e96',
        7: '#495057',
        8: '#343a40',
        9: '#212529',
    },
    'red': {
        0: '#fff5f5',
        1: '#ffe3e3',
        2: '#ffc9c9',
        3: '#ffa8a8',
        4: '#ff8787',
        5: '#ff6b6b',
        6: '#fa5252',
        7: '#f03e3e',
        8: '#e03131',
        9: '#c92a2a',
    },
    'pink': {
        0: '#fff0f6',
        1: '#ffdeeb',
        2: '#fcc2d7',
        3: '#faa2c1',
        4: '#f783ac',
        5: '#f06595',
        6: '#e64980',
        7: '#d6336c',
        8: '#c2255c',
        9: '#a61e4d',
    },
    'grape': {
        0: '#f8f0fc',
        1: '#f3d9fa',
        2: '#eebefa',
        3: '#e599f7',
        4: '#da77f2',
        5: '#cc5de8',
        6: '#be4bdb',
        7: '#ae3ec9',
        8: '#9c36b5',
        9: '#862e9c',
    },
    'violet': {
        0: '#f3f0ff',
        1: '#e5dbff',
        2: '#d0bfff',
        3: '#b197fc',
        4: '#9775fa',
        5: '#845ef7',
        6: '#7950f2',
        7: '#7048e8',
        8: '#6741d9',
        9: '#5f3dc4',
    },
    'indigo': {
        0: '#edf2ff',
        1: '#dbe4ff',
        2: '#bac8ff',
        3: '#91a7ff',
        4: '#748ffc',
        5: '#5c7cfa',
        6: '#4c6ef5',
        7: '#4263eb',
        8: '#3b5bdb',
        9: '#364fc7',
    },
    'blue': {
        0: '#e7f5ff',
        1: '#d0ebff',
        2: '#a5d8ff',
        3: '#74c0fc',
        4: '#4dabf7',
        5: '#339af0',
        6: '#228be6',
        7: '#1c7ed6',
        8: '#1971c2',
        9: '#1864ab',
    },
    'cyan': {
        0: '#e3fafc',
        1: '#c5f6fa',
        2: '#99e9f2',
        3: '#66d9e8',
        4: '#3bc9db',
        5: '#22b8cf',
        6: '#15aabf',
        7: '#1098ad',
        8: '#0c8599',
        9: '#0b7285',
    },
    'teal': {
        0: '#e6fcf5',
        1: '#c3fae8',
        2: '#96f2d7',
        3: '#63e6be',
        4: '#38d9a9',
        5: '#20c997',
        6: '#12b886',
        7: '#0ca678',
        8: '#099268',
        9: '#087f5b',
    },
    'green': {
        0: '#ebfbee',
        1: '#d3f9d8',
        2: '#b2f2bb',
        3: '#8ce99a',
        4: '#69db7c',
        5: '#51cf66',
        6: '#40c057',
        7: '#37b24d',
        8: '#2f9e44',
        9: '#2b8a3e',
    },
    'lime': {
        0: '#f4fce3',
        1: '#e9fac8',
        2: '#d8f5a2',
        3: '#c0eb75',
        4: '#a9e34b',
        5: '#94d82d',
        6: '#82c91e',
        7: '#74b816',
        8: '#66a80f',
        9: '#5c940d',
    },
    'yellow': {
        0: '#fff9db',
        1: '#fff3bf',
        2: '#ffec99',
        3: '#ffe066',
        4: '#ffd43b',
        5: '#fcc419',
        6: '#fab005',
        7: '#f59f00',
        8: '#f08c00',
        9: '#e67700',
    },
    'orange': {
        0: '#fff4e6',
        1: '#ffe8cc',
        2: '#ffd8a8',
        3: '#ffc078',
        4: '#ffa94d',
        5: '#ff922b',
        6: '#fd7e14',
        7: '#f76707',
        8: '#e8590c',
        9: '#d9480f',
    },
}

single_col_width = 3.43 # = 8.7 cm
intermediate_col_width = 4.76 # = 12.1 cm; Science two-column figures
double_col_width = 7.01 # = 17.8 cm

main_figures_dir = '../figures/'
analysis_figures_dir = '../figures/analysis/'

def plot_small_multipanel(data_list, titles=None, orientation='portrait',
                          xlabel='', x_lim=None, xticks=None,
                          ylabel='', y_lim=None, yticks=None,
                          n_wide_axis=12, n_narrow_axis=8,
                          save_file=None):
    '''
    Takes list of data tuples (x_i, y_i) and makes page-sized small multipanel plot with 126 panels.
    '''
    if orientation == 'landscape':
        fig_size = (1.4 * double_col_width, double_col_width)
        n_rows = n_narrow_axis
        n_cols = n_wide_axis
    else:
        fig_size = (double_col_width, 1.4 * double_col_width)
        n_rows = n_wide_axis
        n_cols = n_narrow_axis

    plt.rc('font', size=5)
    fig = plt.figure(figsize=fig_size)
    axes = []

    for x in range(n_cols):
        for y in range(n_rows):
            i = x + y * n_cols
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            axes.append(ax)

            # Add labels to outer panels to save space
            if y == n_rows - 1:
                ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel('')
            if i % n_cols == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel('')

            if x_lim is not None:
                ax.set_xlim(x_lim)
            if y_lim is not None:
                ax.set_ylim(y_lim)

            if xticks is not None:
                ax.set_xticks(xticks)
            if yticks is not None:
                ax.set_yticks(yticks)

            if i < len(data_list):
                ax.plot(data_list[i][0], data_list[i][1])

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(f'{save_file}')
        plt.close()

    return fig, axes

def make_small_multipanel(panels=(12, 8), figsize=None, titles=None,
                          xlabel='', x_lim=None, xticks=None, xticklabels=None,
                          ylabel='', y_lim=None, yticks=None):
    '''
    Makes figure and axes for small multi-panel figure.
    '''
    (n_cols, n_rows) = panels

    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    elif n_cols > n_rows:
        #fig = plt.figure(figsize=(1.4 * double_col_width, double_col_width))
        fig = plt.figure(figsize=(1.8 * double_col_width, 1.3 * double_col_width))
    elif n_cols == n_rows:
        fig = plt.figure(figsize=(double_col_width, double_col_width))
    else:
        fig = plt.figure(figsize=(double_col_width, 1.4 * double_col_width))
    plt.rc('font', size=6)

    axes = []
    for y in range(n_rows):
        for x in range(n_cols):
            i = x + n_cols * y
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            axes.append(ax)

            if titles is not None and i < len(titles):
                ax.set_title(titles[i], fontsize=5)

            # Add labels to outer panels to save space
            if y == n_rows - 1:
                ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel('')
            if i % n_cols == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel('')

            if x_lim is not None:
                ax.set_xlim(x_lim)
            if y_lim is not None:
                ax.set_ylim(y_lim)

            if xticks is not None:
                ax.set_xticks(xticks)
            if yticks is not None:
                ax.set_yticks(yticks)

            if xticklabels is not None:
                ax.set_xticklabels(xticklabels)

    plt.tight_layout()
    return fig, axes

def plot_gene_sequence_panel(gene_seq, cm, ax, x0=0, y0=0, vrange=None, fontsize=8):
    ax.set_yticks([])
    color_map = generate_color_map(cm, vrange)
    for gene in gene_seq:
        if gene[2] == 0:
            color = 'gray'
        else:
            color = color_map.to_rgba(gene[2])
        ax.arrow(x0 + gene[0], y0, gene[1], 0, color=color, width=0.4, head_width=0.6, head_length=0.15*abs(gene[1]), length_includes_head=True)
        ax.text(x0 + gene[0], y0 + 0.5, gene[3], rotation=45, horizontalalignment='left', verticalalignment='center', fontweight='bold', fontsize=fontsize)

def generate_color_map(cm, vrange):
    if vrange is None:
        color_map = mpl.cm.ScalarMappable(cmap=cm)
    else:
        norm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1])
        color_map = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    return color_map


def plot_correlation(x_data, y_data, xlabel='', ylabel='', save_file=None):
    fig = plt.figure(figsize=(single_col_width, single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter(x_data, y_data)

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(f'{analysis_figures_dir}{save_file}')
        plt.close()


def plot_3panel_distribution(values, bins=30, xlabel='', xcuml_lim=None, save_file=None):
    x, cuml = utils.cumulative_distribution(values)
    fig = plt.figure(figsize=(double_col_width, double_col_width / 3))
    ax = fig.add_subplot(131)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('pdf')
    ax.hist(values, bins=bins, density=True)

    ax = fig.add_subplot(132)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('count')
    ax.set_yscale('log')
    ax.hist(values, bins=bins)
    counts, x_bins = np.histogram(values, bins=bins)
    y_min = np.min(counts[counts.nonzero()])
    y_max = np.max(counts)
    ax.set_ylim(0.5 * y_min, 2 * y_max)

    ax = fig.add_subplot(133)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('cumulative')
    if xcuml_lim is not None:
        ax.set_xlim(xcuml_lim)
    ax.plot(x, cuml)

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(f'{analysis_figures_dir}{save_file}')
        plt.close()


def plot_distribution(data, xlim=None, xlabel='', title='',
                      bins=30, histyscale=None, histylim=None,
                      cumlxlim=None, cumlylim=None, cumlxscale='log', cumlyscale='log',
                      fontsize=10, save_fig=None):
    fig = plt.figure(figsize=(double_col_width, single_col_width))
    ax = fig.add_subplot(121)
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel('counts', fontsize=fontsize)
    if histylim is not None:
        ax.set_ylim(histylim)
    if histyscale is not None:
        ax.set_yscale(histyscale)
    if xlim is not None:
        ax.set_xlim(xlim)
        #ax.hist(data, bins=bins, range=xlim)
    #else:
    ax.hist(data, bins=bins)

    ax = fig.add_subplot(122)
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel('complementary cumulative', fontsize=fontsize)
    ax.set_xscale(cumlxscale)
    ax.set_yscale(cumlyscale)
    if cumlxlim is not None:
        ax.set_xlim(cumlxlim)
        x, cumulative = utils.cumulative_distribution(data, x_min=cumlxlim[0])
    else:
        x, cumulative = utils.cumulative_distribution(data)
    if cumlylim is not None:
        ax.set_ylim(cumlylim)

    ax.plot(x, 1 - cumulative, lw=1, c='k')
    plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig)
        plt.close()


def plot_pdf(data, bins=30, title='', data_label='', xlabel='', xlim=None, xscale='linear', 
        ylabel='counts', ylim=None, yscale='linear',
        alpha=1, color='tab:blue', density=False, align='mid', savefig=None):
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xscale(xscale)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.hist(data, bins=bins, density=density, align=align, alpha=alpha, color=color, label=data_label)
    if data_label != '':
        ax.legend(fontsize=8)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    else:
        return ax


def plot_length_clusters_distribution(edge_matrix, sorted_lengths, savefig=None):
    '''
    Plots distribution of gene lengths sorted by connected components
    from graph given by edge_matrix
    '''
    num_components, components = sparse.csgraph.connected_components(edge_matrix)
    x0 = sorted_lengths[0] - 100
    x1 = sorted_lengths[-1] + 100

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('sequence')
    ax.set_xlim(x0, x1)
    ax.set_ylabel('counts')

    for i_component in range(num_components):
        ax.hist(sorted_lengths[components == i_component], bins=30, range=(x0, x1), label=f'{i_component}')
    ax.legend(fontsize=6)
    plt.tight_layout()

    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        return ax


def plot_along_genome(x, y, xlim=None, ylim=None, ylabel='', region=None, coarse_grain=None, title='', save_fig=None):
    fig = plt.figure(figsize=(double_col_width, double_col_width / 3))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('genome position, $x$')
    ax.set_ylabel('number of samples')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if coarse_grain is not None:
        x = utils.coarse_grain_distribution(x, coarse_grain)
        y = utils.coarse_grain_distribution(y, coarse_grain)
    ax.plot(x, y, lw=1, c='k')

    plt.tight_layout()
    if save_fig is not None:
        plt.savefig(analysis_figures_dir + save_fig)


def construct_gene_sequence(contig_genes):
    # TODO: Add strand information to gene_seq
    gene_seqs = {'osa':[], 'osbp':[]}
    for i, row in contig_genes.iterrows():
        gene_location = row['location']
        gene_info = [gene_location[0], gene_location[1] - gene_location[0] + 1]
        locus_tags = row['Syn_homolog_tags']
        if locus_tags is None:
            gene_info.append(0)
            gene_info.append('')
            gene_seqs['osa'].append(gene_info)
            gene_seqs['osbp'].append(gene_info)
        else:
            locus_tags = np.array(locus_tags)
            seq_idents = np.array(row['Syn_alignment_identities'])
            osa_hits = np.array(['CYA' in tag for tag in locus_tags])
            osbp_hits = np.array(['CYB' in tag for tag in locus_tags])

            # TODO: Hack to get some information from genes with multiple hits; need to update filter_contigs.py to include info for all hits
            if len(locus_tags) == 2:
                osa_ident = seq_idents[osa_hits][0]
                osa_tag = locus_tags[osa_hits][0]

                osbp_ident = seq_idents[osbp_hits][0]
                osbp_tag = locus_tags[osbp_hits][0]
            else:
                osa_ident = seq_idents[osa_hits[-2:]][0]
                osa_tag = locus_tags[osa_hits][-1]

                osbp_ident = seq_idents[osbp_hits[-2:]][0]
                osbp_tag = locus_tags[osbp_hits][-1]

            gene_seqs['osa'].append(gene_info + [osa_ident, osa_tag])
            gene_seqs['osbp'].append(gene_info + [osbp_ident, osbp_tag])

    return gene_seqs

def draw_alignment_sequences(aln, savefig=None):

    fig = plt.figure(figsize=(2 * double_col_width, double_col_width))

    ax_dendro = plt.subplot2grid((20, 20), (1, 0), rowspan=19, colspan=1, fig=fig)
    ax_dendro.axis('off')
    ax_consensus = plt.subplot2grid((20, 20), (0, 1), rowspan=1, colspan=19, fig=fig)
    ax_consensus.axis('off')
    ax = plt.subplot2grid((20, 20), (1, 1), rowspan=19, colspan=19, fig=fig)
    ax.axis('off')

    # Print consensus

    aln_arr = np.array(aln)
    consensus_seq = [utils.sorted_unique(aln_col)[0][0] for aln_col in aln_arr.T]
    print(consensus_seq)

    x = 1
    y = 1
    i = 1
    for base in consensus_seq:
        gatc.letter_at(base, x, y, xscale=0.5, yscale=0.8, ax=ax_consensus)
        if i%3 == 0:
            x += 1
        else:
            x += 0.6
        i += 1
    ax_consensus.set_xlim(0, x + 1)
    ax_consensus.set_ylim(0.5, 2.5)


    # Print rest
    i = 1
    x = 1
    for aln_col in aln_arr.T:
        y = -1
        if len(np.unique(aln_col[aln_col != '-'])) == 1:
            for base in aln_col:
                gatc.letter_at('.', x, y, yscale=1, ax=ax)
                y -= 1

            if i%3 == 0:
                x += 1
            else:
                x += 0.4
        else:
            for base in aln_col:
                gatc.letter_at(base, x, y, yscale=0.8, ax=ax)
                y -= 1

            if i%3 == 0:
                x += 1
            else:
                x += 0.6
        i += 1
    ax.set_xlim(0, x + 1)
    ax.set_ylim(y - 1, 0)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        return ax, ax_dendro, ax_consensus


def plot_tree(tree, f_aln=None, rotation=0, savefig=None):
    ts = ete3.TreeStyle()
    ts.show_branch_support = True
    ts.show_branch_length = True
    ts.show_leaf_name = False
    ts.rotation = rotation

    if f_aln:
        tree.link_to_alignment(f_aln, aln_format='fasta')

    if savefig is None:
        tree.show(tree_style=ts)
    else:
        tree.render(savefig, w=178, units='mm', tree_style=ts)


def plot_fancy_tree(tree, branch_thickness=1.0, color_dict=None, savefig=None):
    ts = ete3.TreeStyle()
    ts.mode = 'c' # Make tree circular
    ts.draw_guiding_lines = False

    # Modified from Kat Holt package
    for node in tree.traverse():
        nstyle = NodeStyle()
        nstyle['fgcolor'] = 'black'
        #if color_dict is not None:
        #    if node.name in color_dict:
        #        nstyle['bgcolor'] = color_dict[node.name]
        nstyle['size'] = 0
        node.set_style(nstyle)

        node.img_style['hz_line_width'] = branch_thickness
        node.img_style['vt_line_width'] = branch_thickness

    tree.dist = 0 # Set root distance to zero

    if savefig is None:
        tree.show(tree_style=ts)
    else:
        tree.render(savefig, w=178, units='mm', dpi=300, tree_style=ts)


def plot_rank_abundance(abundance, title='', data_label='', xlabel='rank', xlim=None, xscale='log', 
        ylabel='abundance', ylim=None, yscale='log',
        alpha=1, color='tab:blue', savefig=None):

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_xscale(xscale)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_yscale(yscale)

    sorted_abundance = np.sort(abundance)[::-1]
    rank = np.arange(len(sorted_abundance)) + 1
    ax.plot(rank, sorted_abundance, label=data_label)
    if data_label != '':
        ax.legend(fontsize=10)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    else:
        return ax


def plot_ridgelines(data_list, y_aspect_factor=1.5, title='', xlim=(0, 1), num_xticks=6, ylim=(0, 15.0), xlabel='', ylabels=[], color=open_colors['blue'][4], savefig=None):
    '''
    Script for drawing multiple distributions on ridgeline plot. 
        Implementation is adapted from https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
    '''

    gs = (gridspec.GridSpec(len(data_list), 1))
    fig = plt.figure(figsize=(double_col_width, y_aspect_factor * double_col_width))
    ax_objs = []

    for i, x in enumerate(data_list):
        x_d = np.linspace(xlim[0], xlim[1], 1000)
        kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
        kde.fit(x[:, None])
        logprob = kde.score_samples(x_d[:, None])

        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
        ax_objs[-1].plot(x_d, np.exp(logprob), color='white', lw=1)
        ax_objs[-1].fill_between(x_d, np.exp(logprob), alpha=1.0, color=color)

        # setting uniform x and y lims
        ax_objs[-1].set_xlim(xlim)
        ax_objs[-1].set_ylim(ylim)

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticks([])
        ax_objs[-1].set_yticklabels([])

        if i == len(data_list) - 1:
            ax_objs[-1].set_xlabel(xlabel, fontsize=12)
            ax_objs[-1].set_xticks(np.linspace(xlim[0], xlim[1], num_xticks))
        else:
            ax_objs[-1].set_xticks([])
            ax_objs[-1].set_xticklabels([])

        spines = ['top', 'right', 'left', 'bottom']
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)

        if i < len(ylabels):
            ax_objs[-1].text(xlim[0] - 0.02, 0, ylabels[i], fontweight='normal', fontsize=6, ha='right')

    gs.update(hspace=-0.7)
    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    else:
        fig, gs, ax_objs


def plot_pdist_clustermap2(pdist_df, linkage=None, sort_matrix=True, location_sort=None, sample_sort=None, species_sort=None, temperature_sort=None, log=False, pseudocount=1E-1, cmap='Blues', grid=True, cbar_label='SNPs', savefig=None):
    '''
    Outdated method for plotting clustered divergence matrices. Copied from test_mixed_og_subclustering.py.
    '''

    # Set up a colormap:
    # use copy so that we do not mutate the global colormap instance
    palette = copy.copy(plt.get_cmap(cmap))
    palette.set_bad((0.5, 0.5, 0.5))

    if linkage is not None and sort_matrix == True:
        dn = hclust.dendrogram(linkage, no_plot=True)
        sample_ids = np.array(pdist_df.index)
        ordered_sags = list(sample_ids[dn['leaves']])
    else:
        ordered_sags = list(pdist_df.index)

    plot_df = pdist_df.reindex(index=ordered_sags, columns=ordered_sags)
    num_seqs = plot_df.shape[0]

    #fig = plt.figure(figsize=(double_col_width, double_col_width))
    fig = plt.figure(figsize=(single_col_width, single_col_width))
    ax_dendro = plt.subplot2grid((10, 10), (0, 0), rowspan=1, colspan=9)
    ax_dendro.axis('off')
    ax_cbar = plt.subplot2grid((10, 10), (2, 9), rowspan=8, colspan=1)
    ax = plt.subplot2grid((10, 10), (1, 0), rowspan=9, colspan=9)
    #ax = fig.add_subplot(111)
    if log == True:
        im = ax.imshow(np.log10(pseudocount + plot_df), cmap=palette, aspect='equal')
    else:
        im = ax.imshow(plot_df.astype(float), cmap=palette, aspect='equal')

    ax.set_xlim(-1, num_seqs)
    ticks = []
    tick_labels = []
    for ref in ["OS-A", "OS-B'"]:
        if ref in ordered_sags:
            ticks.append(ordered_sags.index(ref))
            tick_labels.append(ref)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(-1, num_seqs)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(plot_df.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(plot_df.shape[0] + 1) - 0.5, minor=True)
    ax.tick_params(which='minor', bottom=False, left=False, right=False, top=False)
    if grid == True:
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.1)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    #fig.colorbar(im, shrink=0.75)
    cbar = fig.colorbar(im, shrink=0.75, cax=ax_cbar, label=cbar_label)

    if linkage is not None:
        mpl.rcParams['lines.linewidth'] = 0.5
        hclust.dendrogram(linkage, ax=ax_dendro, color_threshold=0, above_threshold_color='k')

    # Add annotations
    if location_sort is not None:
        # Annotate locations
        x_marks = np.arange(num_seqs)
        y_marks = -1.5 * np.ones(num_seqs)
        colors = []
        for sag_id in ordered_sags:
            if sag_id in location_sort['OS']:
                colors.append('cyan')
            elif sag_id in location_sort['MS']:
                colors.append('magenta')
            else:
                colors.append('none')
        ax.scatter(x_marks, y_marks, s=10, ec='none', fc=colors, clip_on=False)
        ax.scatter(y_marks, x_marks, s=10, ec='none', fc=colors, clip_on=False)

    elif species_sort is not None:
        # Annotate species
        x_marks = np.arange(num_seqs)
        y_marks = -1.5 * np.ones(num_seqs)
        cmap = plt.get_cmap('tab10')
        colors = []
        for sag_id in ordered_sags:
            if sag_id in species_sort['syna']:
                colors.append(cm(0))
            elif sag_id in species_sort['synbp']:
                colors.append(cm(1))
            else:
                colors.append('none')
        ax.scatter(x_marks, y_marks, s=10, ec='none', fc=colors, clip_on=False)
        ax.scatter(y_marks, x_marks, s=10, ec='none', fc=colors, clip_on=False)

    elif sample_sort is not None:
        # Annotate sample
        x_marks = np.arange(num_seqs)
        y_marks = -1.5 * np.ones(num_seqs)
        cmap = plt.get_cmap('tab10')
        colors = []
        sag_samples = sorted(list(sample_sort.keys()))
        sample_dict = {}
        for i, sample_id in enumerate(sag_samples):
            sample_dict[sample_id] = i

        samples = []
        for sag_id in ordered_sags:
            color = 'none'
            for sample_id in sample_sort:
                if sag_id in sample_sort[sample_id]:
                    color = cmap(sample_dict[sample_id])
                    samples.append(sample_id)
                    break
            colors.append(color)

        print('\n')
        print([(sample_id, sample_dict[sample_id]) for sample_id in sag_samples])
        for sample_id in sag_samples:
            print(sample_id, len(sample_sort[sample_id]), sample_sort[sample_id])
        print('\n\n')
        ax.scatter(x_marks, y_marks, s=10, ec='none', fc=colors, clip_on=False)
        ax.scatter(y_marks, x_marks, s=10, ec='none', fc=colors, clip_on=False)

    elif temperature_sort is not None:
        # Annotate locations
        x_marks = np.arange(num_seqs)
        y_marks = -1.5 * np.ones(num_seqs)
        colors = []
        if 'T55' in temperature_sort:
            for sag_id in ordered_sags:
                if sag_id in location_sort['T55.0']:
                    colors.append('blue')
                elif sag_id in location_sort['T59.0'] or sag_id in location_sort['T60.0']:
                    colors.append('red')
                else:
                    colors.append('none')
        ax.scatter(x_marks, y_marks, s=10, ec='none', fc=colors, clip_on=False)
        ax.scatter(y_marks, x_marks, s=10, ec='none', fc=colors, clip_on=False)


    plt.tight_layout(h_pad=-1.5, pad=0.2)
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    else:
        return ax


def plot_pdist_clustermap(pdist_df, linkage=None, sort_matrix=True, log=False, pseudocount=1E-1, cmap='Blues', grid=True, cbar_label='SNPs', savefig=None):
    '''
    Updated method for plotting clustered divergence matrices. Use instead of `plot_pdist_clustermap2`.
    '''

    # Set up a colormap:
    # use copy so that we do not mutate the global colormap instance
    palette = copy.copy(plt.get_cmap(cmap))
    palette.set_bad((0.5, 0.5, 0.5))

    if linkage is not None and sort_matrix == True:
        dn = hclust.dendrogram(linkage, no_plot=True)
        sample_ids = np.array(pdist_df.index)
        ordered_sags = list(sample_ids[dn['leaves']])
    else:
        ordered_sags = list(pdist_df.index)

    plot_df = pdist_df.reindex(index=ordered_sags, columns=ordered_sags)
    num_seqs = plot_df.shape[0]

    #fig = plt.figure(figsize=(double_col_width, double_col_width))
    fig = plt.figure(figsize=(single_col_width, single_col_width))
    ax_dendro = plt.subplot2grid((10, 10), (0, 0), rowspan=1, colspan=9)
    ax_dendro.axis('off')
    ax_cbar = plt.subplot2grid((10, 10), (2, 9), rowspan=8, colspan=1)
    ax = plt.subplot2grid((10, 10), (1, 0), rowspan=9, colspan=9)
    #ax = fig.add_subplot(111)
    if log == True:
        im = ax.imshow(np.log10(pseudocount + plot_df), cmap=palette, aspect='equal')
    else:
        im = ax.imshow(plot_df.astype(float), cmap=palette, aspect='equal')

    ax.set_xlim(-1, num_seqs)
    ticks = []
    tick_labels = []
    for ref in ["OS-A", "OS-B'"]:
        if ref in ordered_sags:
            ticks.append(ordered_sags.index(ref))
            tick_labels.append(ref)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(-1, num_seqs)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(plot_df.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(plot_df.shape[0] + 1) - 0.5, minor=True)
    ax.tick_params(which='minor', bottom=False, left=False, right=False, top=False)
    if grid == True:
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.1)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    #fig.colorbar(im, shrink=0.75)
    cbar = fig.colorbar(im, shrink=0.75, cax=ax_cbar, label=cbar_label)

    if linkage is not None:
        mpl.rcParams['lines.linewidth'] = 0.5
        hclust.dendrogram(linkage, ax=ax_dendro, color_threshold=0, above_threshold_color='k')

    plt.tight_layout(h_pad=-1.5, pad=0.2)
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    else:
        return ax


def plot_heatmap(data, xlabel='', ylabel='', cmap='Blues', grid=True, cbar_label='SNPs', cbar_ticks=None, savefig=None):
    # Set up color map
    palette = copy.copy(plt.get_cmap(cmap))
    palette.set_bad((0.5, 0.5, 0.5))

    fig = plt.figure(figsize=(single_col_width, 0.9 * single_col_width))
    w1 = 20
    w2 = 1
    gspec = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[w1, w2])

    ax = plt.subplot(gspec[0, 0], aspect=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    im = ax.imshow(data.astype(float), cmap=palette, aspect='equal')

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    #ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    #ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which='minor', bottom=False, left=False, right=False, top=False)
    if grid == True:
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.1)

    #cax = plt.subplot(gspec[0, 1], aspect=w1)
    cax = plt.subplot(gspec[0, 1])
    cb = plt.colorbar(im, cax=cax)
    cb.ax.set_ylabel(cbar_label, rotation=90, fontsize=12, labelpad=10)
    cb.ax.tick_params(labelsize=12)

    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)


    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    else:
        return ax


def plot_loci_genome_location(og_ids, pangenome_map, ref_genbank_file='../data/reference_genomes/CP000239.genbank', ax=None, savefig=None):
    # Add OS-A and OS-B' mapped ids
    og_table = pangenome_map.og_table
    ref_mapped_ids = []
    for pid in og_ids:
        for oid in og_table.loc[og_table['parent_og_id'] == pid, :].index:
            if 'CYA' in oid:
                ref_mapped_ids.append(oid)
                break
            elif 'YSG' not in oid:
                ref_mapped_ids.append(oid)
                break
    print(ref_mapped_ids)

    '''
    # Map locus tags to reference genomes
    raw_ref_cds_dict = utils.read_genbank_cds(ref_genbank_file)
    ref_cds_dict = add_gene_id_map(raw_ref_cds_dict)
    locus_tags = [[map_ssog_id(ssog_id, ref_cds_dict) for ssog_id in group_ids if map_ssog_id(ssog_id, ref_cds_dict) is not None] for group_ids in ref_mapped_ids]

    # Get mapping for all OS-A and OS-B' homologs
    keep_locus = [[True] * len(locus_tags[0]), [True] * len(locus_tags[1])]
    syn_homolog_map = SynHomologMap(build_maps=True)
    mapped_locus_tags = []
    for i, tag_list in enumerate(locus_tags):
        mapped_tags = [syn_homolog_map.get_ortholog(tag) if 'CYB' in tag else tag for tag in tag_list] 
        keep_locus[i] = [tag != 'none' for tag in mapped_tags]
        mapped_locus_tags.append(np.array(mapped_tags))

    alt_mapped_locus_tags = []
    for i, tag_list in enumerate(locus_tags):
        mapped_tags = [syn_homolog_map.get_ortholog(tag) if 'CYA' in tag else tag for tag in tag_list] 
        alt_mapped_locus_tags.append(np.array(mapped_tags))
        for j, tag in enumerate(mapped_tags):
            if tag == 'none':
                keep_locus[i][j] = False

    for i in range(2):
        mapped_locus_tags[i] = mapped_locus_tags[i][keep_locus[i]]
        alt_mapped_locus_tags[i] = alt_mapped_locus_tags[i][keep_locus[i]]
        print(len(locus_tags[i]), len(mapped_locus_tags[i]), len(alt_mapped_locus_tags[i]))

    # Get color values
    scale_factor = 1 / (2932766) # normalize by OS-A genome length
    x_osa = []
    for tag_list in mapped_locus_tags:
        x_osa.append([scale_factor * np.mean((ref_cds_dict[tag].location.start, ref_cds_dict[tag].location.end)) for tag in tag_list])

    raw_ref_cds_dict = utils.read_genbank_cds('../data/reference_genomes/CP000240.genbank')
    alt_ref_cds_dict = add_gene_id_map(raw_ref_cds_dict)
    osbp_scale_factor = 1 / 3046682 # normalize by OS-B' genome length
    x_osbp = []
    for tag_list in alt_mapped_locus_tags:
        x_osbp.append([osbp_scale_factor * np.mean((alt_ref_cds_dict[tag].location.start, alt_ref_cds_dict[tag].location.end)) for tag in tag_list])

    y0 = -0.05
    offset = 0.008
    offset2 = 0.012
    markers = ['o', 'D']
    locus_labels = ['genomic trench', 'typical locus']

    # Plot position according to OS-A
    fig = plt.figure(figsize=(double_col_width, 0.5 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'genome location')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.1, 0.1)
    ax.get_yaxis().set_visible(False)
    ax.axhline(y0, lw=1, c='k')

    cmap = plt.get_cmap('Blues')
    for i in range(2):
        y = y0 + ((-1)**i) * offset * np.ones(len(x_osa[i]))
        ax.plot(x_osa[i], y, marker='|', mec='k', ls='', mew=0.1, zorder=2)
        ax.scatter(x_osa[i][0], y[0] + ((-1)**i) * offset2, marker=markers[i], s=15, fc=cmap(x_osbp[i][0]), ec='white', lw=0.05, zorder=3, label=locus_labels[i])
        for j in range(1, len(y)):
            ax.scatter(x_osa[i][j], y[j] + ((-1)**i) * offset2, marker=markers[i], s=15, fc=cmap(x_osbp[i][j]), ec='white', lw=0.05, zorder=3)

    ax.legend(loc='upper right', fontsize=8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1) 
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}genomic_trenches_osa_genome_locations.pdf')
    plt.close()


    # Plot position according to OS-B'
    fig = plt.figure(figsize=(double_col_width, 0.5 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'genome location')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.1, 0.1)
    ax.get_yaxis().set_visible(False)
    ax.axhline(y0, lw=1, c='k')

    cmap = plt.get_cmap('Oranges')
    for i in range(2):
        y = y0 + ((-1)**i) * offset * np.ones(len(x_osbp[i]))
        ax.plot(x_osbp[i], y, marker='|', mec='k', ls='', mew=0.1, zorder=2)
        ax.scatter(x_osbp[i][0], y[0] + ((-1)**i) * offset2, marker=markers[i], s=15, fc=cmap(x_osa[i][0]), ec='white', lw=0.05, zorder=3, label=locus_labels[i])
        for j in range(1, len(y)):
            ax.scatter(x_osbp[i][j], y[j] + ((-1)**i) * offset2, marker=markers[i], s=15, fc=cmap(x_osa[i][j]), ec='white', lw=0.05, zorder=3)

    ax.legend(loc='upper right', fontsize=8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1) 
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}genomic_trenches_osbp_genome_locations.pdf')
    plt.close()
    '''

def test_histogram_density(random_seed=None):
    rng = np.random.default_rng(random_seed)
    figures_dir = '../figures/analysis/tests/statistics/'

    # Draw lognormal random variables
    mu = 0
    sigma = 1
    num_samples = 10000 
    z = np.random.lognormal(mu, sigma, num_samples)

    # Plot linear scale without density
    #ax = plot_pdf(z, bins=x_bins, xscale='log')
    x_bins = np.linspace(0, 1.2 * np.max(z), 100)
    ax = plot_pdf(z, bins=x_bins)
    C = 1.1 * num_samples / 1.5 # Fit constant for histogram
    y_theory = C * stats.lognorm.pdf(x_bins, sigma, loc=mu)
    ax.plot(x_bins, y_theory, '-k')
    plt.tight_layout()
    plt.savefig(f'{figures_dir}lognormal_histogram_linx.pdf')


    # Plot linear scale with density
    ax = plot_pdf(z, bins=x_bins, density=True)
    y_theory = stats.lognorm.pdf(x_bins, sigma, loc=mu)
    ax.plot(x_bins, y_theory, '-k')
    plt.tight_layout()
    plt.savefig(f'{figures_dir}lognormal_pdf_linx.pdf')


    # Plot log scale with density
    x_bins = np.geomspace(0.5 * np.min(z), 2 * np.max(z), 100)
    ax = plot_pdf(z, bins=x_bins, xscale='log', density=False)
    C = 500
    y_theory = C * stats.lognorm.pdf(x_bins, sigma / np.log(10), loc=(mu / np.log(10))) # log base 10 changes scale of lognormal
    #y_theory = C * stats.lognorm.pdf(x_bins, sigma, loc=mu) # log base 10 changes scale of lognormal
    ax.plot(x_bins, y_theory, '-k')
    plt.tight_layout()
    plt.savefig(f'{figures_dir}lognormal_histogram_logx.pdf')

    ax = plot_pdf(z, bins=x_bins, xscale='log', density=True)
    y_theory = stats.lognorm.pdf(x_bins, sigma, loc=mu)
    ax.plot(x_bins, y_theory, '-k')
    plt.tight_layout()
    plt.savefig(f'{figures_dir}lognormal_pdf_logx.pdf')


    # Plot log-transformed variables
    logz = np.log10(z)
    mu_y = mu / np.log(10)
    sigma_y = sigma / np.log(10)
    x_bins = np.linspace(1.2 * np.min(logz), 1.2 * np.max(logz), 100)
    ax = plot_pdf(logz, bins=x_bins, density=True)
    y_theory = stats.norm.pdf(x_bins, loc=mu_y, scale=sigma_y)
    ax.plot(x_bins, y_theory, '-k')
    plt.tight_layout()
    plt.savefig(f'{figures_dir}lognormal_logtransform_pdf.pdf')

nucl_color_dict = {'-':'#ffffff', 'A':open_colors['red'][5], 'C':open_colors['orange'][5], 'G':open_colors['cyan'][4], 'T':open_colors['blue'][5]}
#nucl_colors = [open_colors['gray'][4], open_colors['red'][6], open_colors['orange'][5], open_colors['cyan'][5], open_colors['blue'][7]]
#nucl_colors = [open_colors['gray'][4], open_colors['red'][6], open_colors['orange'][5], open_colors['yellow'][3], open_colors['blue'][7]]
nucl_colors = [open_colors['gray'][1], open_colors['teal'][8], open_colors['red'][7], open_colors['yellow'][3], open_colors['blue'][7]]
#nucl_colors = [open_colors['gray'][4], '#009988', '#EE7733', '#EE3377', '#0077BB'] # taken from https://personal.sron.nl/~pault/
#nucl_colors = [open_colors['gray'][4], '#009988', '#EE7733', '#EE3377', '#33BBEE'] # taken from https://personal.sron.nl/~pault/
nucl_aln_cmap = mpl.colors.LinearSegmentedColormap.from_list('nucl_aln', nucl_colors, N=len(nucl_colors))


def plot_alignment(aln, yticks='auto', annotation=None, marker_size=2, aln_type='nucl', reference=None, h0=100, w0=1000, fig_dpi=1200, color_scheme='nucleotides', annotation_style='dots', annot_lw=4, aspect='auto', ax=None, savefig=None):
    # Convert to numerical values 
    if aln_type == 'nucl':
        if reference == 'closest_to_consensus':
            d_consensus = calculate_consensus_distance(aln)
            reference = int(np.argmin(d_consensus))
        aln_numeric = convert_nucleotides_to_numbers(aln, reference, conversion_type=color_scheme)
        vmax = np.nanmax(aln_numeric)

    w = aln.get_alignment_length()
    h = len(aln)
    stretch_factor = max(0.5, (np.round(2 * (w / w0)) / 2))
    fig_width = stretch_factor * double_col_width # stretch width for long genes to preserve resolution
    fig_height = 0.75 * single_col_width
    
    if ax is None:
        fig = plt.figure(figsize=(fig_width, 0.75 * fig_height))
        ax = fig.add_subplot(111)

    ax.imshow(aln_numeric, aspect=aspect, cmap=nucl_aln_cmap, vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_xlim(-0.2, aln_numeric.shape[1] - 0.8)

    if yticks == 'auto':
        ax.set_yticks([])
    else:
        ax.set_yticks(yticks)

    if annotation is not None:
        color_dict = {'A':'tab:orange', 'Bp':'tab:blue', 'C':'tab:green'}
        rec_ids = [rec.id for rec in aln]
        annotation_idx = {}
        for label in annotation:
            annotation_idx[label] = [rec_ids.index(g) for g in annotation[label]]

            if annotation_style == 'dots':
                x = -1.5 * np.ones(len(annotation_idx[label]))
                ax.scatter(x, annotation_idx[label], s=marker_size**2, ec='none', fc=color_dict[label], clip_on=False)
            elif annotation_style == 'lines':
                x0 = -annot_lw
                x1 = 0
                yoff = 0.5
                for y in annotation_idx[label]:
                    ax.fill_between([x0, x1], y - yoff, y - yoff + 1, capstyle='butt', ec='none', fc=color_dict[label], clip_on=False)
                    ax.plot([x0, x1], [y - yoff, y - yoff], lw=0.2, c='w', clip_on=False)
                    if y == len(aln) - 1:
                        ax.plot([x0, x1], [y - yoff + 1, y - yoff + 1], lw=0.25, c='k', clip_on=False)

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig, dpi=fig_dpi)
        plt.close()
    else:
        return ax


def convert_nucleotides_to_numbers(aln, reference, conversion_type='nucleotides'):
    '''
    Params
    -------

    conversion_type : sets mapping from characters to numbers ['nucleotides', 'synonymous_nonsynonymous'];
                        only 'nucleotides' mapping is currently implemented
    '''
    nucleotides = ['-', 'A', 'C', 'G', 'T']
    
    aln_numeric = np.array(aln)
    if reference is None:
        aln_numeric = convert_seq_array_to_numeric(aln_numeric, nucleotides)

    elif type(reference) == int:
        ref_seq_arr = aln_numeric[reference]
        ref_based_aln_arr = [ref_seq_arr]
        for i, seq_arr in enumerate(aln_numeric):
            if i != reference:
                ref_based_aln_arr.append(convert_seq_array_to_numeric(seq_arr, nucleotides, reference=ref_seq_arr))
        ref_based_aln_arr[0] = convert_seq_array_to_numeric(ref_seq_arr, nucleotides)
        aln_numeric = np.array(ref_based_aln_arr)
            
    return aln_numeric


def convert_seq_array_to_numeric(seq_arr, characters, offset=0, reference=None, gap_value=np.nan):
    seq_numeric = np.zeros(seq_arr.shape)
    for i, c in enumerate(characters):
        if c == '-':
            seq_numeric[np.where(seq_arr == c)] = gap_value
        else:
            seq_numeric[np.where(seq_arr == c)] = i + offset

    if reference is not None:
        seq_numeric[np.where(seq_arr == reference)] = 0

    return seq_numeric.astype(float)


def test_alignment_plot(figures_dir='../figures/analysis/tests/', results_dir='../results/tests/alignment/'):
    test_og_id = 'YSG_1000'
    f_aln = f'../results/single-cell/sscs_pangenome/_aln_results/{test_og_id}_aln.fna'
    aln = seq_utils.read_alignment(f_aln)
    trimmed_aln, _ = align_utils.trim_alignment_and_remove_gap_codons(aln)

    '''
    ref = 0
    aln_syn_masked = seq_utils.mask_alignment(trimmed_aln, '4D')
    plot_alignment(aln_syn_masked, reference=ref, savefig=f'{figures_dir}{test_og_id}_4D_masked_aln.pdf')
    align_utils.write_alignment(aln_syn_masked, f'{results_dir}{test_og_id}_4D_masked_aln.fna')

    aln_non4d_masked = seq_utils.mask_alignment(trimmed_aln, 'non-4D')
    aln_4d_snps, x_4d_snps = seq_utils.get_snps(aln_non4d_masked, return_x=True)
    #plot_alignment(aln_non4d_masked[:, :1100], reference=ref, savefig=f'{figures_dir}{test_og_id}_non-4D_masked_aln.pdf')
    plot_alignment(aln_non4d_masked, reference=ref, savefig=f'{figures_dir}{test_og_id}_non-4D_masked_aln.pdf')
    align_utils.write_alignment(aln_non4d_masked, f'{results_dir}{test_og_id}_non-4D_masked_aln.fna')

    aln_1d_masked = seq_utils.mask_alignment(trimmed_aln, '1D')
    plot_alignment(aln_1d_masked, reference=ref, savefig=f'{figures_dir}{test_og_id}_1D_masked_aln.pdf')
    align_utils.write_alignment(aln_1d_masked, f'{results_dir}{test_og_id}_1D_masked_aln.fna')

    aln_4d_snps = seq_utils.get_snps(aln_non4d_masked) 
    plot_alignment(aln_4d_snps, reference=ref, savefig=f'{figures_dir}{test_og_id}_4D_SNPs_aln.pdf')
    align_utils.write_alignment(aln_4d_snps, f'{results_dir}{test_og_id}_4D_SNPs_aln.fna')

    plot_alignment(aln, savefig=f'{figures_dir}{test_og_id}_aln.pdf')

    '''

    ref_idx = 100
    plot_alignment(aln, reference=ref_idx, savefig=f'{figures_dir}{test_og_id}_aln_ref{ref_idx}.pdf', color_scheme='nucleotides')
    plot_alignment(aln, reference=ref_idx, savefig=f'{figures_dir}{test_og_id}_aln_ref{ref_idx}_NSS_colors.pdf', color_scheme='synonymous_nonsynonymous')
    #for ref_idx in [0, 100]:
    #    plot_alignment(aln, reference=ref_idx, savefig=f'{figures_dir}{test_og_id}_aln_ref{ref_idx}.pdf')

    '''
    plot_alignment(aln, reference='closest_to_consensus', savefig=f'{figures_dir}{test_og_id}_aln_ref_consensus_min.pdf')

    plot_alignment(trimmed_aln, savefig=f'{figures_dir}{test_og_id}_aln.pdf')
    for ref_idx in [0, 100]:
        plot_alignment(trimmed_aln, reference=ref_idx, savefig=f'{figures_dir}{test_og_id}_aln_ref{ref_idx}.pdf')
    plot_alignment(trimmed_aln, reference='closest_to_consensus', savefig=f'{figures_dir}{test_og_id}_aln_ref_consensus_min.pdf')

    aln_4d, x_4d = seq_utils.get_synonymous_sites(trimmed_aln, return_x=True)
    x_ns = seq_utils.get_nonsynonymous_snps(trimmed_aln)
    aln_snps, x_snps = seq_utils.get_snps(trimmed_aln, return_x=True)
    #print(x_4d, len(x_4d))
    #print(x_ns, len(x_ns))
    #print(x_snps, len(x_snps))
    '''


def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):
    if prop_patches is None:
        prop_patches = {
            **prop_lines,
            "alpha": prop_lines.get("alpha", 1) * 0.2,
            "clip_on": False,
        }

    c1 = BboxConnector(
        bbox1, bbox2, loc1=loc1a, loc2=loc2a, clip_on=False, **prop_lines)
    c2 = BboxConnector(
        bbox1, bbox2, loc1=loc1b, loc2=loc2b, clip_on=False, **prop_lines)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    p = BboxConnectorPatch(bbox1, bbox2,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           clip_on=False,
                           **prop_patches)

    return c1, c2, bbox_patch1, bbox_patch2, p


def zoom_effect01(ax1, ax2, xmin, xmax, **kwargs):
    """
    Connect *ax1* and *ax2*. The *xmin*-to-*xmax* range in both axes will
    be marked.

    Parameters
    ----------
    ax1
        The main axes.
    ax2
        The zoomed axes.
    xmin, xmax
        The limits of the colored area in both plot axes.
    **kwargs
        Arguments passed to the patch constructor.
    """

    bbox = Bbox.from_extents(xmin, 0, xmax, 1)

    mybbox1 = TransformedBbox(bbox, ax1.get_xaxis_transform())
    mybbox2 = TransformedBbox(bbox, ax2.get_xaxis_transform())

    prop_patches = {**kwargs, "ec": "none", "alpha": 0.2}

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
        mybbox1, mybbox2,
        loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        prop_lines=kwargs, prop_patches=prop_patches)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


def zoom_effect02(ax1, ax2, **kwargs):
    """
    ax1 : the main axes
    ax1 : the zoomed axes

    Similar to zoom_effect01.  The xmin & xmax will be taken from the
    ax1.viewLim.
    """

    tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
    trans = blended_transform_factory(ax2.transData, tt)

    mybbox1 = ax1.bbox
    mybbox2 = TransformedBbox(ax1.viewLim, trans)

    prop_patches = {**kwargs, "ec": "none", "alpha": 0.2}

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
        mybbox1, mybbox2,
        loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        prop_lines=kwargs, prop_patches=prop_patches)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


def connect_bbox2(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):
    if prop_patches is None:
        prop_patches = {**prop_lines, 'alpha': prop_lines.get('alpha', 1) * 0.2, 'clip_on': False}

    c1 = BboxConnector(
        bbox1, bbox2, loc1=loc1a, loc2=loc2a, clip_on=False, **prop_lines)
    c2 = BboxConnector(
        bbox1, bbox2, loc1=loc1b, loc2=loc2b, clip_on=False, **prop_lines)

    bbox_patch1 = BboxPatch(bbox1, fc='none', **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, fc='tab:red', **prop_patches)

    p = BboxConnectorPatch(bbox1, bbox2,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           clip_on=False,
                           **prop_patches)

    return c1, c2, bbox_patch1, bbox_patch2, p


def zoom_effect03(ax1, ax2, **kwargs):
    """
    ax1 : the main axes
    ax1 : the zoomed axes

    Similar to zoom_effect01.  The xmin & xmax will be taken from the
    ax1.viewLim.
    """

    tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
    trans = blended_transform_factory(ax2.transData, tt)

    mybbox1 = ax1.bbox
    mybbox2 = TransformedBbox(ax1.viewLim, trans)
    print(mybbox1)
    print(mybbox2)

    prop_patches = {**kwargs, "ec": "none", "alpha": 0.2}

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox2(
        mybbox1, mybbox2,
        loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        prop_lines=kwargs, prop_patches=prop_patches)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


def test_axes_zoom():
    axs = plt.figure().subplot_mosaic([
        #["zoom1", "zoom2"],
        ["zoom2", "zoom2"],
        ["main", "main"],
    ])

    axs["main"].set(xlim=(0, 5))
    #zoom_effect01(axs["zoom1"], axs["main"], 0.2, 0.8)
    axs["zoom2"].set(xlim=(2.1, 2.2))
    zoom_effect02(axs["zoom2"], axs["main"])

    plt.show()


if __name__ == '__main__':
    #print('No tests implemented...')
    #test_histogram_density(12345)
    test_alignment_plot()
    #test_axes_zoom()

