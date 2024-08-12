import argparse
import os
import glob
import numpy as np 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import scipy.cluster.hierarchy as hclust
import scipy.spatial.distance as distance
import scipy.sparse as sparse
from metadata_map import MetadataMap
from Bio import SeqIO
from Bio import AlignIO
from Bio.SeqRecord import SeqRecord
from plot_utils import *
from pangenome_utils import PangenomeMap
from test_gene_clustering import plot_pdist_clustermap

def plot_divergence_matrix(snp_pdist, highlighted_seqs=[], cluster_threshold=0, linkage_method='average', cbar_label='SNPs', cmap='Blues', savefig=None):
    gene_ids = np.array(list(snp_pdist.index))
    pdist = snp_pdist.fillna(0)
    pdist_squareform = distance.squareform(pdist.values)
    Z = hclust.linkage(pdist_squareform, method=linkage_method, optimal_ordering=True)
    if cluster_threshold > 0:
        clusters = hclust.fcluster(Z, cluster_threshold, criterion='distance')
        print(clusters)
    dn = hclust.dendrogram(Z, no_plot=True)
    sags_sorting = list(gene_ids[dn['leaves']])
    print(sags_sorting)
    plot_gene_divergence_matrix(pdist, linkage=Z, sags_sorting=sags_sorting, highlighted_seqs=highlighted_seqs, cbar_label=cbar_label, cmap=cmap, savefig=savefig)
    return Z

def plot_gene_divergence_matrix(pdist_df, linkage=None, sags_sorting=None, highlighted_seqs=['UncmicOcRedA3L13_FD'], location_sort=None, species_sort=None, log=False, pseudocount=1E-1, cmap='Blues', grid=True, cbar_label='SNPs', savefig=None):
    # Set up a colormap:
    # use copy so that we do not mutate the global colormap instance
    palette = plt.get_cmap(cmap)
    palette.set_bad((0.5, 0.5, 0.5))

    if sags_sorting is None:
        ordered_sags = list(pdist_df.index)
    else:
        ordered_sags = list(sags_sorting)

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
        im = ax.imshow(plot_df, cmap=palette, aspect='equal')

    ax.set_xlim(-1, num_seqs)
    ticks = []
    tick_labels = []
    for ref in ["OS-A", "OS-B'"]:
        if ref in ordered_sags:
            ticks.append(ordered_sags.index(ref))
            tick_labels.append(ref)
    for seq_id in highlighted_seqs:
        if seq_id in ordered_sags:
            ticks.append(ordered_sags.index(seq_id))
            tick_labels.append(utils.strip_sample_name(seq_id))

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=6)
    ax.set_ylim(-1, num_seqs)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=6)

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

    plt.tight_layout(h_pad=-1.5, pad=0.2)
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    else:
        return ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alignment_file', help='Sequence alignments.')
    parser.add_argument('-c', '--convert_mafft_aln', action='store_true', help='Convert input file from MAFFT format to regular FASTA.')
    parser.add_argument('-d', '--divergence_file', default=None, help='Pre-computed pairwise divergences.')
    parser.add_argument('-o', '--output_file', default=None, help='Figure file.')
    parser.add_argument('-t', '--tag', default=None, help='Locus tag used in divergence dictionary.')
    parser.add_argument('--cluster_threshold', default=0, type=float ,help='Threshold for defining clusters.')
    parser.add_argument('--highlight_seqs', default='', type=str, help='Sequence IDs to add to tick labels.')
    parser.add_argument('--cmap', default='Blues', type=str, help='Colormap.')
    parser.add_argument('--cbar_label', default='SNPs', type=str, help='Colorbar label.')
    parser.add_argument('--trim_alignment_gaps', action='store_true')
    args = parser.parse_args()


    if args.divergence_file is not None:
        if args.tag is not None:
            pdist_dict = pickle.load(open(args.divergence_file, 'rb'))
            pdist = pdist_dict[args.tag]
        else:
            pdist = pickle.load(open(args.divergence_file, 'rb'))
    else:
        if args.convert_mafft_aln:
            aln = seq_utils.read_mafft_alignment(args.alignment_file)
            AlignIO.write(aln, args.alignment_file, 'fasta')
        else:
            aln = AlignIO.read(args.alignment_file, 'fasta')

        if args.trim_alignment_gaps:
            aln = align_utils.trim_alignment_gaps(aln)

        pdist = align_utils.calculate_pairwise_distances(aln, metric='divergences')

    highlighted_seqs = args.highlight_seqs.split(',')
    plot_divergence_matrix(pdist, highlighted_seqs=highlighted_seqs, cluster_threshold=args.cluster_threshold, cmap=args.cmap, cbar_label=args.cbar_label, savefig=args.output_file)
    if args.output_file is None:
        plt.show()
