import argparse
import numpy as np
import pandas as pd
import glob
import os
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
import matplotlib.pyplot as plt
from metadata_map import MetadataMap
from plot_utils import *

mpl.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def test_all(pangenome_map, metadata, og_id='YSG_0879'):
    # Plot alignment
    f_aln = f'../results/single-cell/alignments/core_ogs_main_clusters/{og_id}_cleaned_aln.fna'
    aln = seq_utils.read_alignment(f_aln)
    species_grouping = align_utils.sort_aln_rec_ids(aln, pangenome_map, metadata)
    plot_alignment(aln, annotation=species_grouping, marker_size=3, reference='closest_to_consensus', annotation_style='lines', fig_dpi=1000, savefig=f'../figures/analysis/tests/{og_id}_main_clusters_aln.pdf')

    f_aln = f'../results/single-cell/sscs_pangenome/_aln_results/{og_id}_aln.fna'
    aln = seq_utils.read_alignment(f_aln)
    species_grouping = align_utils.sort_aln_rec_ids(aln, pangenome_map, metadata)
    plot_alignment(aln, annotation=species_grouping, marker_size=3, reference='closest_to_consensus', annotation_style='lines', fig_dpi=1000, savefig=f'../figures/analysis/tests/{og_id}_raw_aln.pdf')



if __name__ == '__main__':
    pangenome_dir = '../results/single-cell/sscs_pangenome/'
    f_orthogroup_table = f'{pangenome_dir}filtered_orthogroups/sscs_annotated_single_copy_orthogroup_presence.tsv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fasta_file', help='FASTA file to be parsed.')
    parser.add_argument('-g', '--orthogroup_table', default=f_orthogroup_table, help='File with orthogroup table.')
    parser.add_argument('-m', '--marker_size', type=int, default=3)
    parser.add_argument('-o', '--output_file', default=None)
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('--no_species_annotation', action='store_true', help='Remove SAG species annotation from alignment margins.')
    args = parser.parse_args()

    pangenome_map = pg_utils.PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    metadata = MetadataMap()

    if args.test == True:
        test_all(pangenome_map, metadata)
    else:
        aln = seq_utils.read_alignment(args.fasta_file)

        if args.no_species_annotation == False:
            species_grouping = align_utils.sort_aln_rec_ids(aln, pangenome_map, metadata)
        else:
            species_grouping = None
        plot_alignment(aln, annotation=species_grouping, marker_size=args.marker_size, reference='closest_to_consensus', annotation_style='lines', fig_dpi=1000, savefig=args.output_file)

        if args.output_file is None:
            plt.show()


