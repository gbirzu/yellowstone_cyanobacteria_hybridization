import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import pickle
import calculate_linkage_disequilibria as ld
import er2
import alignment_tools as align_utils
import os
from metadata_map import MetadataMap
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
#from analyze_allele_frequencies import plot_pdist_clustermap
from test_gene_clustering import plot_pdist_clustermap
from plot_utils import *


def plot_haplotypes(linkage_data, min_depth=1, save_file=None):
    fig = plt.figure(figsize=(double_col_width, double_col_width/2))
    ax = fig.add_subplot(121)
    ax.set_title('OS-A reference')
    ax.set_xticks([])
    ax.set_ylabel('sample')
    if 'OS-A' in linkage_data.keys():
        osa_allele_table = linkage_data['OS-A']['allele_table']
        add_haplotype_map(osa_allele_table, min_depth, ax)

    ax = fig.add_subplot(122)
    ax.set_title('OS-B\' reference')
    ax.set_xticks([])
    ax.set_yticks([])
    if 'OS-Bp' in linkage_data.keys():
        osbp_allele_table = linkage_data['OS-Bp']['allele_table']
        add_haplotype_map(osbp_allele_table, min_depth, ax)

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(analysis_figures_dir + save_file)

def add_haplotype_map(allele_table, min_depth, ax):
    sample_cols = ld.get_sample_columns(allele_table)
    haplotypes = allele_table.loc[allele_table['locus_depth'] > min_depth, sample_cols].values.astype(float).T
    cmap = colors.ListedColormap(['gray', 'red', 'blue'])
    cmap.set_under('gray')
    cmap.set_over('cyan')
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(haplotypes, aspect='auto', cmap=cmap)

def plot_locus_depth(linkage_data, save_file=None):
    fig = plt.figure(figsize=(double_col_width, double_col_width/2))
    ax = fig.add_subplot(121)
    ax.set_title('OS-A reference')
    ax.set_xlabel('position along OS-A genome')
    ax.set_ylabel('number of samples aligned at locus')
    if 'OS-A' in linkage_data.keys():
        osa_allele_table = linkage_data['OS-A']['allele_table']
        ax.plot(osa_allele_table.index, osa_allele_table['locus_depth'])
        print(f'OS-A max depth: {max(osa_allele_table["locus_depth"])}')

    ax = fig.add_subplot(122)
    ax.set_title('OS-B\' reference')
    ax.set_xlabel('position along OS-A genome')
    ax.set_ylabel('number of samples aligned at locus')
    if 'OS-Bp' in linkage_data.keys():
        osbp_allele_table = linkage_data['OS-Bp']['allele_table']
        ax.plot(osbp_allele_table.index, osbp_allele_table['locus_depth'])
        print(f'OS-B\' max depth: {max(osbp_allele_table["locus_depth"])}')

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(analysis_figures_dir + save_file)

def plot_r2(linkage_data, binned=False, title='', save_file=None):
    fig = plt.figure(figsize=(double_col_width, double_col_width/2))
    ax = fig.add_subplot(121)
    ax.set_title('OS-A reference')
    ax.set_xlabel('SNP separation (nt)')
    ax.set_ylabel(r'1/$\langle r^2 \rangle$')
    ax.set_yscale('log')
    ax.set_ylim([1E-3, 1])
    if 'OS-A' in linkage_data.keys():
        ld_stats = ld_data['OS-A']['linkage_stats']
        add_ld_panel(ld_stats, ax, binned)

    #ax.set_ylim([0, 0.04])

    ax = fig.add_subplot(122)
    ax.set_title('OS-B\' reference')
    ax.set_xlabel('SNP separation (nt)')
    ax.set_ylabel(r'1/$\langle r^2 \rangle$')
    ax.set_yscale('log')
    ax.set_ylim([1E-3, 1])
    if 'OS-Bp' in linkage_data.keys():
        ld_stats = ld_data['OS-Bp']['linkage_stats']
        add_ld_panel(ld_stats, ax, binned)

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(analysis_figures_dir + save_file)

def add_ld_panel(ld_stats, ax, binned):
    #x_r2 = np.arange(len(ld_stats['r^2'][np.where(ld_stats['r^2'] != 0)[0]]))
    x_r2 = np.arange(len(ld_stats['r^2']))
    if binned == True:
        bins = 25
        bin_size = len(x_r2) // bins
        r2 = np.array([np.mean(ld_stats['r^2'][i:i+bin_size]) for i in range(0, len(x_r2), bin_size)])
        x_r2 = [np.mean(x_r2[i:i+bin_size]) for i in range(0, len(x_r2), bin_size)]
    else:
        r2 = ld_stats['r^2']

    print(len(x_r2), len(r2))
    ax.scatter(x_r2, r2, c='r', label='$r^2$')
    #ax.scatter(x_r2, 1 / ld_stats['r^2'][np.where(ld_stats['r^2'])[0]], c='r', label='$r^2$')
    #x_sigma2 = np.arange(len(ld_stats['r^2'][np.where(ld_stats['sigma^2'] != 0)[0]]))
    #ax.scatter(x_sigma2, 1 / ld_stats['sigma^2'][np.where(ld_stats['sigma^2'])[0]], c='b', label='$sigma^2$')

def average_over_distance(r_sq, denom=2, coarse_grain=False, num_cg_points=20):
    x_max = int(r_sq.shape[0] / denom)
    r2_avg = np.zeros(x_max)
    for k in range(x_max):
        r2_avg[k] = np.trace(r_sq, offset=k) / len(np.diag(r_sq, k=k))
    if coarse_grain == True:
        x_bin = (np.log10(2 * x_max) - np.log10(0.5)) / num_cg_points
        x_log = np.geomspace(11, 2 * x_max, num_cg_points)
        r2_cg = np.zeros(len(x_log) + 11)
        r2_cg[:11] = r2_avg[:11]
        r2_std = np.zeros(len(x_log) + 11)
        r2_std[:11] = 0
        for i, x in enumerate(x_log):
            idx = i + 11
            jl = int(np.floor(x))
            jr = int(np.ceil(10**(np.log10(x) + x_bin)))
            r2_cg[idx] = np.mean(r2_avg[jl:jr])
            r2_std[idx] = np.std(r2_avg[jl:jr])
        x_cg = np.concatenate([np.arange(11), x_log])
        return x_cg, r2_cg, r2_std
    else:
        return r2_avg


def test_species_linkage():
    locus1_files = ['../results/single-cell/sscs/osa_alignments/CYA_1459_aln_MUSCLE.fasta',
                        '../results/single-cell/sscs/osbp_alignments/CYB_0652_aln_MUSCLE.fasta',
                        '../results/single-cell/sscs/os-merged_alignments/CYB_0652_aln_MUSCLE.fasta']
    locus2_files = ['../results/single-cell/sscs/osa_alignments/CYA_2611_aln_MUSCLE.fasta',
                        '../results/single-cell/sscs/osbp_alignments/CYB_2270_aln_MUSCLE.fasta',
                        '../results/single-cell/sscs/os-merged_alignments/CYB_2270_aln_MUSCLE.fasta']

    paths = ['../results/single-cell/sscs/osa_alignments/',
                        '../results/single-cell/sscs/osbp_alignments/',
                        '../results/single-cell/sscs/os-merged_alignments/']

    locus_dict = {'CYB_0652':'CYA_1459', 'CYB_2270':'CYA_2611'}
    species = ['syna', 'synbp', 'syn']
    #for locus in ['CYB_0652', 'CYB_2270']:
    for locus in ['CYB_0652']:
        fig = plt.figure(figsize=(double_col_width, double_col_width / 3.5))

        r2_list = []
        for i in range(3):
            if i == 0:
                f_path = f'{paths[i]}{locus_dict[locus]}_aln_MUSCLE.fasta'
            else:
                f_path = f'{paths[i]}{locus}_aln_MUSCLE.fasta'
            aln = AlignIO.read(f_path, 'fasta')
            r_sq = ld.calculate_rsquared(aln)
            r2_avg = average_over_distance(r_sq)
            r2_list.append(r2_avg)
            ax = fig.add_subplot(1, 3, i + 1)
            im = ax.imshow(np.log10(r_sq + 1E-6), cmap='Greys', aspect='equal')

        fig.colorbar(im)
        plt.tight_layout()
        #plt.savefig(f'../figures/analysis/tests/linkage/{locus}_r2_matrix.pdf')

        fig = plt.figure(figsize=(single_col_width, single_col_width))
        ax = fig.add_subplot(111)
        ax.set_xlabel('separation, x')
        ax.set_xscale('log')
        ax.set_xlim(3E-1, 2E3)
        #ax.set_xlim(0, 6E2)
        ax.set_ylabel(r'linkage disequilibrium, $r^2$')
        ax.set_yscale('log')

        rho_list = [0.1, 0.03, 1E-3]
        ls = [':', '-', '--']
        for i in range(3):
            r2_avg = r2_list[i]
            x = 0.5 + np.arange(len(r2_avg))
            ax.scatter(x, r2_avg, s=10, label=species[i])

            if i != 0:
                rho = rho_list[i]
                theta = r2_avg[0]
                x_theory = np.arange(0, len(r2_avg))
                r2_theory_alt = [er2.calculate_r2_fast(rho * x, theta, l_max=20) for x in x_theory]
                ax.plot(0.4 + x_theory, r2_theory_alt, c='k', ls=ls[i], lw=2, label=f'$\\rho={rho}, \\theta={theta:.2f}$')

        #ax.plot([60, 1000], [1 / 60, 1E-3], c='k', ls='--', lw=2, label=r'$\sim 1/x$')
        #x_asymp = np.geomspace(60, 600, 100)
        #y_asymp = np.geomspace(1. / 60, 1. / 600, 100)
        #ax.plot(x_asymp, y_asymp, c='k', ls='--', lw=2, label=r'$\sim 1/x$')

        #theta = r2_avg[0]
        #rho = 10.0
        #r2_theory = [er2.calculate_r2_fast(rho * x, theta, l_max=20) for x in x_theory]
        #ax.plot(0.4 + x_theory, r2_theory, c='k', ls='-', lw=2, label=f'$\\rho={rho}$')
        #r2_theory_alt = [er2.calculate_r2_fast(rho * x, 4 * theta) for x in x_theory]
        #ax.plot(0.4 + x_theory, r2_theory_alt, c='k', ls='--', lw=2, label=f'$\\rho={rho}$')

        theta = 0.1
        rho = 10
        r2_theory_alt = [er2.calculate_r2_fast(rho * x, 100, l_max=20) for x in x_theory]
        ax.plot(0.4 + x_theory, r2_theory_alt, c='k', ls=':', lw=2, label=f'$\\rho={rho}, \\theta={theta:.2f}$')

        ax.legend(fontsize=8)
        plt.tight_layout()
        #plt.savefig(f'../figures/analysis/tests/linkage/{locus}_r2_avg_loglin.pdf')
        plt.savefig(f'../figures/analysis/tests/linkage/{locus}_r2_avg_loglog.pdf')

def make_r2_panel(r2_avg, x, ax, label=None, xlim=(3E-1, 5E3), ylim=(1E-6, 1E0), theory=True):
    ax.set_xlabel('separation, x')
    ax.set_xscale('log')
    if xlim:
        ax.set_xlim(xlim)
    ax.set_ylabel(r'linkage disequilibrium, $r^2$')
    ax.set_yscale('log')
    if ylim:
        ax.set_ylim(ylim)

    ax.scatter(x, r2_avg, s=10, label=label)

    if theory == True:
        ax.plot([60, 1000], [1 / 60, 1E-3], c='k', ls='--', lw=2, label=r'$\sim 1/x$')

    ax.legend(fontsize=8)
    plt.tight_layout()


def test_allele_linkage():
    allele_table = pickle.load(open('../results/tests/CYA_2238_allele_table.dat', 'rb'))
    #allele_table = pickle.load(open('../results/tests/CYB_0652_allele_table.dat', 'rb'))
    #allele_table = pickle.load(open('../results/tests/CYB_2270_allele_table.dat', 'rb'))
    print(allele_table)
    genotypes, genotypes_inverse, genotype_counts = np.unique(allele_table.values, return_inverse=True, return_counts=True, axis=0)
    allele_table['genotype_counts'] = 0
    for i in range(len(genotypes)):
        index = genotypes_inverse[i]
        if 0 in genotypes[index]:
            allele_table.iloc[i, -1] = 0
        else:
            allele_table.iloc[i, -1] = genotype_counts[index]
    print(allele_table)


    #allele_table['genotype_label'] = allele_table['CYB_0651'] + (100 * allele_table['CYB_0652']) + (10000 * allele_table['CYB_0653'])
    #sorted_table = allele_table.sort_values('genotype_label')
    sorted_table = allele_table.sort_values('genotype_counts', ascending=False)
    #sorted_genotypes = sorted_table.iloc[:, :3]
    sorted_genotypes = sorted_table.iloc[:, :2]
    print(sorted_table)

    myMatrix = np.ma.masked_where(sorted_genotypes.values < 1, sorted_genotypes.values)
    fig = plt.figure(figsize=(single_col_width, double_col_width))
    ax = fig.add_subplot(111)
    #cmap = plt.get_gmap('gist_rainbow')
    #im = ax.imshow(sorted_genotypes, vmin=1, cmap='gist_rainbow', aspect='auto')
    ax.set_xticks([0.15, 1.15, 2.15])
    ax.set_xticklabels(sorted_table.columns, fontsize=8, ha='center', rotation=30)
    ax.set_yticks(0.25 + np.arange(sorted_table.shape[0]))
    ax.set_yticklabels(sorted_table.index, fontsize=6)
    im = ax.imshow(myMatrix[sorted_table['genotype_counts'] > 0], vmin=1, cmap='gist_rainbow', aspect='auto')
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig('../figures/analysis/tests/linkage/CYA_2238_allele_linkage.pdf')
    #plt.savefig('../figures/analysis/tests/linkage/CYB_0652_allele_linkage.pdf')
    #plt.savefig('../figures/analysis/tests/linkage/CYB_2270_allele_linkage.pdf')

def test_syntey_linkage():
    colinear_loci = ['CYB_2597', 'CYB_2598']
    colinear_dx = [0, 51]
    #rearranged_loci = ['CYB_2138', 'CYB_2139']
    #rearranged_dx = [0, 547]
    rearranged_loci = ['CYB_1493', 'CYB_1494']
    rearranged_dx = [0, 19]

    colinear_alnmts = []
    for locus in colinear_loci:
        locus_aln = AlignIO.read(f'../results/single-cell/sscs/osbp_alignments/{locus}_aln_MUSCLE.fasta', 'fasta')
        colinear_alnmts.append(locus_aln)

    colinear_aln, x_colinear = merge_alignments(colinear_alnmts, colinear_dx)
    x_colinear = x_colinear + 0.5
    print(x_colinear)
    print(colinear_aln, '\n\n')
    r2_colinear = ld.calculate_rsquared(colinear_aln)
    r2_avg_colinear = average_over_distance(r2_colinear)
    #r2_avg = average_over_distance(r2[:657, :][:, :657])
    #r2_avg = average_over_distance(r2[657:, :][:, 657:])

    rearranged_alnmts = []
    for locus in rearranged_loci:
        locus_aln = AlignIO.read(f'../results/single-cell/sscs/osbp_alignments/{locus}_aln_MUSCLE.fasta', 'fasta')
        filtered_aln = seq_utils.filter_main_cloud(locus_aln)
        #rearranged_alnmts.append(locus_aln)
        rearranged_alnmts.append(filtered_aln)

    rearranged_aln, x_rearranged = merge_alignments(rearranged_alnmts, rearranged_dx)
    #r2_rearranged = ld.calculate_rsquared(rearranged_alnmts[1], gap_threshold=0.01)
    r2_rearranged = ld.calculate_rsquared(rearranged_aln)
    print(r2_rearranged.shape)
    r2_avg_rearranged = average_over_distance(r2_rearranged)
    #r2_avg_rearranged = average_over_distance(r2_rearranged[:615, :][:, :615])
    #r2_avg_rearranged = average_over_distance(r2_rearranged[615:, :][:, 615:])
    x_r2 = np.arange(len(r2_avg_rearranged)) + 0.5
    #x_r2 = x_rearranged[:615][:len(r2_avg_rearranged)]
    #x_r2 = x_rearranged[615:] - 615 - rearranged_dx[1]
    #x_r2 = x_r2[:len(r2_avg_rearranged)]

    # Print filtered alignment to file
    #filtered_rearranged = align_utils.filter_alignment(rearranged_alnmts[1], gap_threshold=0.1)
    #print(filtered_rearranged)
    #align_utils.write_alleles_alignment(filtered_rearranged, '../results/tests/CYB_2139_alleles_aln_summary.txt')
    #align_utils.write_alleles_alignment(filtered_rearranged, '../results/tests/CYB_2139_alleles_aln1840-1940_summary.txt', aln_range=(1840, 1940))

    fig = plt.figure(figsize=(single_col_width, single_col_width))
    ax = fig.add_subplot(111)
    make_r2_panel(r2_avg_colinear, x_colinear[:len(r2_avg_colinear)], ax, label='colinear', ylim=(1E-6, 1E0))
    make_r2_panel(r2_avg_rearranged, x_r2, ax, label='rearranged', ylim=(1E-6, 1E0), theory=False)
    plt.savefig(f'../figures/analysis/tests/linkage/synteny_linkage_test_filtered.pdf')

def merge_alignments(alignment_list, dx, gap_threshold=0.25):
    # Filter seqs with many gaps and sort alignment by ID
    filtered_alignments = []
    for aln in alignment_list:
        filtered_aln = align_utils.filter_alignment(aln, gap_threshold)
        filtered_aln.sort()
        filtered_alignments.append(filtered_aln)

    sag_ids = find_common_ids(filtered_alignments)
    merged_records, x = merge_records(filtered_alignments, sag_ids, dx)

    return merged_records, x

def find_common_ids(alnmt_list):
    common_ids = set([record.id for record in alnmt_list[0]])
    for i in range(1, len(alnmt_list)):
        aln_ids = [record.id for record in alnmt_list[i]]
        common_ids = common_ids.intersection(set(aln_ids))
    return sorted(list(common_ids))

def merge_records(alnmt_list, sag_ids, dx=0):
    merged_alnmt = []
    filtered_alnmts = filter_alnmt_ids(alnmt_list, sag_ids)
    for i, sag_id in enumerate(sag_ids):
        merged_record = None
        for alnmt in filtered_alnmts:
            assert alnmt[i].id == sag_id
            if merged_record:
                merged_record = merged_record + alnmt[i]
            else:
                merged_record = alnmt[i]
        merged_alnmt.append(merged_record)

    # Define distance along genome
    if dx == 0:
        total_len = 0
        for alnmt in filtered_alnmts:
            alnmt_len = alnmt.get_alignment_length()
            total_len += alnmt_len
        x = np.arange(total_len)
    else:
        x = []
        for i, alnmt in enumerate(filtered_alnmts):
            alnmt_len = alnmt.get_alignment_length()
            if i == 0:
                x.append(np.arange(dx[i], dx[i] + alnmt_len))
            else:
                x0 = x[-1][-1]
                x.append(np.arange(x0 + dx[i], x0 + dx[i] + alnmt_len))
        x = np.concatenate(x)

    return MultipleSeqAlignment(merged_alnmt), x

def filter_alnmt_ids(alnmt_list, sag_ids):
    filtered_alnmts = []
    for alnmt in alnmt_list:
        filtered_aln = MultipleSeqAlignment([record for record in alnmt if record.id in sag_ids])
        filtered_aln.sort()
        filtered_alnmts.append(filtered_aln)
    return filtered_alnmts

def plot_ribosomal_linkage(f_loci='../results/tests/ribisomal_loci_tags.txt', main_cloud_radius=0.1):
    rp_loci = np.loadtxt(f_loci, dtype='U10')
    print(rp_loci)

    loci_plot = []
    r2_data = []
    r2_matrices = []
    for locus in rp_loci:
        #f_path = f'../results/single-cell/sscs/osbp_alignments/{locus}_aln_MUSCLE.fasta'
        f_path = f'../results/single-cell/sscs/os-merged_alignments/{locus}_codon_aln_MUSCLE.fasta'
        if os.path.exists(f_path):
            aln = AlignIO.read(f_path, 'fasta')
            synbp_aln = seq_utils.filter_species_alignment(aln, 'synbp')
            filtered_aln = seq_utils.filter_main_cloud(synbp_aln, radius=main_cloud_radius)
            if len(filtered_aln) > 50:
                r_sq = ld.calculate_rsquared(filtered_aln)
                r2_matrices.append(r_sq)
                x_cg, r2_avg, r2_std = average_over_distance(r_sq, coarse_grain=True)
                r2_data.append((x_cg, r2_avg, r2_std))
                loci_plot.append(f'{locus} ({len(aln[0])})')
            else:
                print(locus, aln)

    print(loci_plot, len(loci_plot))
    print(r2_data, len(r2_data))

    num_columns = 7
    nr = len(rp_loci) // num_columns
    num_rows =  nr + ((num_columns * nr) < len(rp_loci))

    fig, axes = make_small_multipanel(panels=(num_rows, num_columns))
    for i, r2 in enumerate(r2_matrices):
        axes[i].set_title(loci_plot[i], fontsize=6)
        axes[i].set_xscale('log')
        axes[i].set_xlim(8E-1, 5E3)
        axes[i].set_xticks([1, 1E1, 1E2, 1E3])
        axes[i].set_yscale('log')
        axes[i].set_ylim(1E-6, 1.5)
        axes[i].set_yticks([1E-5, 1E-3, 1E-1])

        r2_avg = average_over_distance(r2)
        x = np.arange(len(r2_avg))
        axes[i].scatter(x, r2_avg, s=10)
    plt.tight_layout()
    plt.savefig(f'../figures/analysis/linkage/ribosomal_proteins/osbp_rp_main_cloud{main_cloud_radius}_r2avg.pdf')

    fig, axes = make_small_multipanel(panels=(num_rows, num_columns))
    for i, r2_tuple in enumerate(r2_data):
        axes[i].set_title(loci_plot[i], fontsize=6)
        axes[i].set_xscale('log')
        axes[i].set_xlim(8E-1, 5E3)
        axes[i].set_xticks([1, 1E1, 1E2, 1E3])
        axes[i].set_yscale('log')
        axes[i].set_ylim(1E-6, 1.5)
        axes[i].set_yticks([1E-5, 1E-3, 1E-1])

        x, r2, r2_std = r2_tuple
        axes[i].plot(x, r2, '-o', ms=2, lw=1)
        axes[i].fill_between(x, r2 - r2_std / 2, r2 + r2_std / 2, alpha=0.5, ec='none')
    plt.tight_layout()
    plt.savefig(f'../figures/analysis/linkage/ribosomal_proteins/osbp_rp_main_cloud{main_cloud_radius}_r2avg_coarse-grained.pdf')

    r2_loci_avg = average_over_loci(r2_matrices)
    fig = plt.figure(figsize=(single_col_width, single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('separation, x')
    ax.set_xscale('log')
    ax.set_xlim(4E-1, 5E3)
    ax.set_ylabel(r'linkage disequilibrium, $r^2$')
    ax.set_yscale('log')
    ax.set_ylim(1E-6, 1.5E0)

    x_cg, r2_avg, r2_std = average_over_distance(r2_loci_avg, coarse_grain=True)
    ax.plot(x_cg + 0.5, r2_avg, '-o', lw=1, ms=3)
    ax.fill_between(x_cg, r2_avg - r2_std / 2, r2_avg + r2_std / 2, alpha=0.5, ec='none')
    plt.tight_layout()
    plt.savefig(f'../figures/analysis/linkage/ribosomal_proteins/osbp_rp_main_cloud{main_cloud_radius}_r2locus_avg.pdf')


def plot_random_rho_theory(rho0=0.03, L=1E3, num_samples=10, sigma=10, output_file='../figures/analysis/linkage/random_rho_theory.pdf'):
    np.random.seed(12345)
    theta = 0.05
    x = np.arange(1, L + 1, 2)
    r2_constant = np.array([er2.calculate_r2_fast(rho0 * xi, theta, l_max=20) for xi in x])

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'SNP separation, $x$')
    ax.set_xscale('log')
    ax.set_ylabel(r'$\langle r^2 \rangle$')
    ax.set_yscale('log')
    ax.plot(x, r2_constant, c='k', ls='--', lw=1, label=f'$\\rho={rho0}, \\theta={theta:.2f}$')

    r2_array = np.zeros((num_samples, len(r2_constant)))
    exp_min = np.log(0.1)
    exp_max = np.log(10)
    for i, drho in enumerate(np.random.uniform(exp_min, exp_max, size=num_samples)):
        rho = np.exp(drho) * rho0
        r2_array[i] = np.array([er2.calculate_r2_fast(rho * xi, theta, l_max=20) for xi in x])
        print(rho)
        ax.plot(x, r2_array[i], c='gray', alpha=0.5, ls='-', lw=1)

    ax.plot(x, np.mean(r2_array, axis=0), c='k', ls='-', lw=2, label='mean')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_file)

def average_over_loci(r2_list):
    max_sites = 0
    max_index = 0
    for i in range(len(r2_list)):
        r2 = r2_list[i]
        if r2.shape[0] > max_sites:
            max_sites = r2.shape[0]
            max_index = i
    r2_avg = np.zeros((max_sites, max_sites))
    counts = np.zeros((max_sites, max_sites))
    for r2 in r2_list:
        num_sites = r2.shape[0]
        r2_avg[:num_sites, :][:, :num_sites] += r2
        counts[:num_sites, :][:, :num_sites] += 1
    r2_avg /= counts
    return r2_avg


def test_sequence_filtering():
    cmap = plt.get_cmap('tab10')
    raw_aln = AlignIO.read('../results/single-cell/sscs/osbp_alignments/CYB_1493_aln_MUSCLE.fasta', 'fasta')
    r2_raw = ld.calculate_rsquared(raw_aln)
    r2_raw_avg = average_over_distance(r2_raw)

    sag_ids = [rec.id for rec in raw_aln]
    merged_raw_aln = AlignIO.read('../results/single-cell/sscs/os-merged_alignments/CYB_1493_aln_MUSCLE.fasta', 'fasta')
    merged_aln_filtered = MultipleSeqAlignment([rec for rec in merged_raw_aln if rec.id in sag_ids])
    r2_merged = ld.calculate_rsquared(merged_aln_filtered)
    r2_merged_avg = average_over_distance(r2_merged)

    raw_codon_aln = AlignIO.read('../results/single-cell/sscs/os-merged_alignments/CYB_1493_codon_aln_MUSCLE.fasta', 'fasta')
    codon_aln = MultipleSeqAlignment([rec for rec in raw_codon_aln if rec.id in sag_ids])
    r2_codon = ld.calculate_rsquared(codon_aln)
    r2_codon_avg = average_over_distance(r2_codon, denom=1.5)
    x_cg, r2_codon_cg, _ = average_over_distance(r2_codon, denom=1.5, coarse_grain=True, num_cg_points=20)

    # Plot r^2 for different alignments
    fig = plt.figure(figsize=(single_col_width, single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('separation, x')
    ax.set_xscale('log')
    ax.set_xlim(4E-1, 5E3)
    ax.set_ylabel(r'linkage disequilibrium, $r^2$')
    ax.set_yscale('log')
    ax.set_ylim(1E-6, 1E0)

    x_codon = 0.5 + np.arange(len(r2_codon_avg))
    ax.scatter(x_codon, r2_codon_avg, s=20, lw=1, marker='o', fc='none', ec='k', label='codon aln')

    x_raw = 0.5 + np.arange(len(r2_raw_avg))
    ax.scatter(x_raw, r2_raw_avg, s=20, lw=1, marker='+', label='raw B\' aln')

    x_merged = 0.5 + np.arange(len(r2_merged_avg))
    #ax.scatter(x_merged, r2_merged_avg, alpha=0.7, label='raw merged aln')
    ax.scatter(x_merged, r2_merged_avg, s=20, lw=1, marker='x', label='raw merged aln')

    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'../figures/analysis/tests/linkage/seq_filtering/gap_filtered_alnmts_comparison.pdf')

    '''
    fig = plt.figure(figsize=(double_col_width, single_col_width))
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('separation, x')
    ax1.set_xscale('log')
    ax1.set_xlim(4E-1, 5E3)
    ax1.set_ylabel(r'linkage disequilibrium, $r^2$')
    ax1.set_yscale('log')
    ax1.set_ylim(1E-6, 1E0)
    ax1.scatter(x_codon, r2_codon_avg, s=20, lw=1, marker='o', fc='none', ec='k', label='codon aln')

    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('separation, x')
    ax2.set_xscale('log')
    ax2.set_xlim(4E-1, 5E3)
    ax2.set_ylabel(r'linkage disequilibrium, $r^2$')
    ax2.set_yscale('log')
    ax2.set_ylim(1E-6, 1E0)
    #ax2.scatter(x_cg, r2_codon_cg, s=20, lw=1, marker='o', fc='none', ec='k', label='codon aln')
    ax2.plot(x_cg, r2_codon_cg, '-o', lw=1, c='k', ms=3, mec='none', mfc='k', label='codon aln')

    markers = ['^', 'v',  'D', 's']
    for i, pi_c in enumerate([0.1, 0.03, 0.01]):
        aln_main_cloud = seq_utils.filter_main_cloud(codon_aln, radius=pi_c)
        r2_mc = ld.calculate_rsquared(aln_main_cloud)
        r2_mc_avg = average_over_distance(r2_mc, denom=1.5)
        ax1.scatter(x_codon, r2_mc_avg, s=20, lw=1, marker=markers[i], fc='none', ec=cmap(i + 1), label=f'B\' cloud ($\pi={pi_c}$)')

        x_cg, r2_mc_cg, _ = average_over_distance(r2_mc, denom=1.5, coarse_grain=True, num_cg_points=20)
        #ax2.scatter(x_cg, r2_mc_cg, s=20, lw=1, marker=markers[i], fc='none', ec=cmap(i + 1), label=f'B\' cloud ($\pi={pi_c}$)')
        ax2.plot(x_cg, r2_mc_cg, f'-{markers[i]}', lw=1, c=cmap(i + 1), ms=3, mec='none', mfc=cmap(i + 1), label=f'B\' cloud ($\pi={pi_c}$)')

    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'../figures/analysis/tests/linkage/seq_filtering/codon_alnmts_divergence_comparison.pdf')
    '''
    # 4D sites
    codon_aln = MultipleSeqAlignment([rec for rec in raw_codon_aln if rec.id in sag_ids])
    filtered_codon_aln = seq_utils.filter_alignment_gaps(codon_aln, 0.25)
    r2_codon = ld.calculate_rsquared(filtered_codon_aln)
    r2_codon_avg = average_over_distance(r2_codon, denom=1.5)
    x_cg, r2_codon_cg, _ = average_over_distance(r2_codon, denom=1.5, coarse_grain=True, num_cg_points=20)

    syn_sites_aln = seq_utils.get_synonymous_sites(filtered_codon_aln)
    r2_syn = ld.calculate_rsquared(syn_sites_aln)
    r2_syn_avg = average_over_distance(r2_syn, denom=1.5)
    x_syn_cg, r2_syn_cg, _ = average_over_distance(r2_syn, denom=1.5, coarse_grain=True, num_cg_points=20)

    fig = plt.figure(figsize=(double_col_width, single_col_width))
    for i in range(2):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.set_xlabel('separation, x')
        ax.set_xscale('log')
        ax.set_xlim(4E-1, 5E3)
        ax.set_ylabel(r'linkage disequilibrium, $r^2$')
        ax.set_yscale('log')
        ax.set_ylim(1E-6, 1E0)

        if i == 0:
            ax.scatter(x_codon, r2_codon_avg, s=20, lw=1, marker='o', fc='none', ec='k', label='all sites')
            ax.scatter(0.5 + 6 * np.arange(len(r2_syn_avg)), r2_syn_avg, s=20, lw=1, marker='^', fc='none', ec=cmap(2), label=f'4D sites (est.separation)')
        else:
            ax.plot(x_cg, r2_codon_cg, '-o', lw=1, c='k', ms=3, mec='none', mfc='k', label='all sites')
            ax.plot(6 * x_syn_cg, r2_syn_cg, f'-^', lw=1, c=cmap(i + 1), ms=3, mec='none', mfc=cmap(2), label=f'4D sites (est. separation)')
            for pi_c in [0.1, 0.03]:
                mc_aln = seq_utils.filter_main_cloud(filtered_codon_aln, radius=pi_c)
                syn_mc_aln = seq_utils.get_synonymous_sites(mc_aln)
                print(pi_c, syn_mc_aln)
                r2_syn_mc = ld.calculate_rsquared(syn_mc_aln)
                r2_syn_mc_avg = average_over_distance(r2_syn_mc, denom=1.5)
                x_syn_mc_cg, r2_syn_mc_cg, _ = average_over_distance(r2_syn_mc, denom=1.5, coarse_grain=True, num_cg_points=20)
                ax.plot(6 * x_syn_mc_cg, r2_syn_mc_cg, f'-s', lw=1, ms=3, mec='none', label=f'4D sites ($Ks < {pi_c}$)')

    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'../figures/analysis/tests/linkage/seq_filtering/sites_comparisons.pdf')

def test_locus_averaging(output_dir='../figures/analysis/tests/linkage/locus_averaging/'):
    metadata_map = MetadataMap()
    #syn_diverse_loci = np.loadtxt('../results/tests/syn_os_common_high_diversity_loci.txt', dtype='U10')
    syn_diverse_loci = np.loadtxt('../results/tests/syn_os_common_high_diversity_loci_v1.txt', dtype='U10')
    species_r2 = {'A':[], 'Bp':[], 'Syn. OS':[]}
    for locus in syn_diverse_loci:
        aln = AlignIO.read(f'../results/single-cell/sscs/os-merged_alignments/{locus}_codon_aln_MUSCLE.fasta', 'fasta')
        sag_ids = [rec.id for rec in aln]
        species_sorted_sags = metadata_map.sort_sags(sag_ids, by='species')
        print(species_sorted_sags)
        species_sorted_sags['Syn. OS'] = sag_ids

        for species in species_sorted_sags:
            species_aln = MultipleSeqAlignment([rec for rec in aln if rec.id in species_sorted_sags[species]])
            if species == 'A':
                filtered_species_aln = seq_utils.filter_main_cloud(species_aln, anchor_id="OS-A", radius=0.1)
                filtered_species_aln = seq_utils.filter_alignment_gaps(filtered_species_aln, gap_threshold=0.25)
            elif species == 'Bp':
                filtered_species_aln = seq_utils.filter_main_cloud(species_aln, anchor_id="OS-B'", radius=0.1)
                filtered_species_aln = seq_utils.filter_alignment_gaps(filtered_species_aln, gap_threshold=0.25)
            else:
                filtered_species_aln = seq_utils.filter_alignment_gaps(species_aln, gap_threshold=0.25)
                for rec in filtered_species_aln:
                    if rec.id in ["OS-A", "OS-B'"]:
                        print(rec.seq, rec.id)
            syn_aln = seq_utils.get_synonymous_sites(filtered_species_aln)
            r_sq = ld.calculate_rsquared(syn_aln)
            species_r2[species].append(r_sq)
    r2_avg_dict = {}
    for species in species_r2:
        r2_list = species_r2[species]
        r2_avg = average_over_loci(r2_list)
        r2_avg_dict[species] = r2_avg
    print(r2_avg_dict)

    '''
    fig = plt.figure(figsize=(double_col_width, single_col_width))
    ax = fig.add_subplot(121)
    ax.set_title('4D sites (est. separation)')
    ax.set_xlabel('separation, x')
    ax.set_xscale('log')
    ax.set_xlim(4E-1, 5E3)
    ax.set_ylabel(r'linkage disequilibrium, $r^2$')
    ax.set_yscale('log')
    ax.set_ylim(1E-4, 1E0)

    k = 0
    markers = ['-o', '-^', '-s']
    for species in species_r2:
        r2_list = species_r2[species]
        for i in range(len(r2_list))[:2]:
            locus = syn_diverse_loci[i]
            x_cg, r2_cg, _ = average_over_distance(r2_list[i], denom=1.5, coarse_grain=True, num_cg_points=20)
            ax.plot(6 * x_cg, r2_cg, markers[k], lw=1, ms=3, alpha=0.6, label=f'{locus} {species}')
        k += 1
    ax.legend(fontsize=8)

    ax = fig.add_subplot(122)
    ax.set_title('4D sites (est. separation)')
    ax.set_xlabel('separation, x')
    ax.set_xscale('log')
    ax.set_xlim(4E-1, 5E3)
    ax.set_ylabel(r'linkage disequilibrium, $r^2$')
    ax.set_yscale('log')
    ax.set_ylim(1E-4, 1E0)

    k = 0
    for species in species_r2:
        r2_avg = r2_avg_dict[species]
        x_cg, r2_cg, _ = average_over_distance(r2_avg, denom=1.5, coarse_grain=True, num_cg_points=20)
        ax.plot(6 * x_cg, r2_cg, markers[k], lw=1, ms=3, label=f'{species} (loci average)')
        k += 1
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'../figures/analysis/tests/linkage/locus_averaging/syn_sites_loci_averaged.pdf')
    '''

    # Test site choice
    sites_average = {'all':[], 'synonymous':[], 'polymorphic':[], 'synonymous_polymorphic':[]}
    x_conversion = {'all':1, 'synonymous':1, 'polymorphic':1, 'synonymous_polymorphic':1}
    for locus in syn_diverse_loci:
        aln = AlignIO.read(f'../results/single-cell/sscs/os-merged_alignments/{locus}_codon_aln_MUSCLE.fasta', 'fasta')
        sag_ids = [rec.id for rec in aln]
        species_sorted_sags = metadata_map.sort_sags(sag_ids, by='species')
        print(species_sorted_sags)
        species = 'Bp'
        species_aln = MultipleSeqAlignment([rec for rec in aln if rec.id in species_sorted_sags[species]])
        filtered_species_aln = seq_utils.filter_main_cloud(species_aln, anchor_id="OS-B'", radius=0.1)
        filtered_species_aln = seq_utils.filter_alignment_gaps(filtered_species_aln, gap_threshold=0.25)

        # All sites
        r_sq = ld.calculate_rsquared(filtered_species_aln)
        sites_average['all'].append(r_sq)
        x_all = r_sq.shape[0]

        # Synonymous
        syn_aln = seq_utils.get_synonymous_sites(filtered_species_aln)
        r_sq = ld.calculate_rsquared(syn_aln)
        sites_average['synonymous'].append(r_sq)
        x_conversion['synonymous'] += x_all / r_sq.shape[0] / len(syn_diverse_loci)

        # polymorphic
        snps_aln = seq_utils.get_snps(filtered_species_aln)
        r_sq = ld.calculate_rsquared(snps_aln)
        sites_average['polymorphic'].append(r_sq)
        x_conversion['polymorphic'] += x_all / r_sq.shape[0] / len(syn_diverse_loci)

        # Synonymous polymorphic
        syn_aln = seq_utils.get_synonymous_sites(filtered_species_aln)
        syn_snps = seq_utils.get_snps(syn_aln)
        r_sq = ld.calculate_rsquared(syn_snps)
        sites_average['synonymous_polymorphic'].append(r_sq)
        x_conversion['synonymous_polymorphic'] += x_all / r_sq.shape[0] / len(syn_diverse_loci)

    r2_sites_avg = {}
    for sites in sites_average:
        r2_sites_avg[sites] = average_over_loci(sites_average[sites])
    print(r2_sites_avg)
    print(x_conversion)


    fig = plt.figure(figsize=(1.2 * single_col_width, single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('separation, x')
    ax.set_xscale('log')
    ax.set_xlim(4E-1, 5E3)
    ax.set_ylabel(r'linkage disequilibrium, $r^2$')
    ax.set_yscale('log')
    ax.set_ylim(1E-4, 1.5E0)

    for sites in r2_sites_avg:
        x_cg, r2_cg, _ = average_over_distance(r2_sites_avg[sites], denom=1.5, coarse_grain=True, num_cg_points=20)
        ax.plot(x_conversion[sites] * x_cg + 0.5, r2_cg, '-o', lw=1, ms=3, alpha=0.6, label=f'{sites}')

    rho = 1
    theta = np.mean(np.diag(r2_sites_avg['all']))
    x_theory = np.arange(0, 1000)
    r2_theory = [er2.calculate_r2_fast(rho * x, theta, l_max=20) for x in x_theory]
    ax.plot(0.5 + x_theory, r2_theory, c='k', ls='-', lw=1, label=f'$\\rho={rho}, \\theta={theta:.2f}$')

    ax.legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'../figures/analysis/tests/linkage/locus_averaging/loci_averaged_different_sites.pdf')


def find_highly_diverse_loci():
    metadata_map = MetadataMap()
    # Find test loci
    #locus_divergences = pickle.load(open('../results/tests/sag_filtered_alnmts_locus_nucleotide_divergences.dat', 'rb'))
    locus_divergences = pickle.load(open('../results/single-cell/sscs/sag_os-merged_alnmts_locus_nucleotide_divergences.dat', 'rb'))
    sag_ids = locus_divergences['SAGs']
    dijk = locus_divergences['nucleotide_divergences']
    l_locus = np.array(locus_divergences['locus_lengths'])
    locus_tags = np.array(locus_divergences['loci'])
    species_sorted_sags = metadata_map.sort_sags(sag_ids, by='species')
    high_diversity_loci = {}
    for species in species_sorted_sags:
        species_index = [sag_id in species_sorted_sags[species] for sag_id in sag_ids]
        d_median = np.nanmedian(dijk[species_index, :, :][:, species_index, :], axis=(0, 1))
        print(species, d_median.shape)
        mean_snps = d_median * l_locus
        print(dijk[species_index, species_index, :])
        print(np.isfinite(dijk[species_index, species_index, :]))
        locus_depths = np.sum(np.isfinite(dijk[species_index, species_index, :]), axis=0)
        print(locus_depths[np.argsort(locus_depths)[::-1]])
        filtered_loci = (d_median > 0.01) * (l_locus > 500) * (l_locus < 3000) * (np.sum(np.isfinite(dijk[species_index, species_index, :]), axis=0) > 50)
        filtered_mean_snps = mean_snps[filtered_loci]
        print(filtered_mean_snps[np.argsort(filtered_mean_snps)[::-1]])
        print(locus_tags[filtered_loci][np.argsort(filtered_mean_snps)[::-1]])
        print('\n')
        high_diversity_loci[species] = {'ordered_loci':locus_tags[filtered_loci][np.argsort(filtered_mean_snps)[::-1]], 'ordered_mean_snps':filtered_mean_snps[np.argsort(filtered_mean_snps)[::-1]], 'locus_lengths':l_locus[filtered_loci][np.argsort(filtered_mean_snps)[::-1]]}

    # Write loci to files
    for species in high_diversity_loci:
        loci = high_diversity_loci[species]['ordered_loci']
        mean_snps = high_diversity_loci[species]['ordered_mean_snps']
        print(mean_snps)
        filtered_loci = loci[mean_snps > 20]
        with open(f'../results/tests/syn_OS{species}_high_diversity_loci.txt', 'w') as f_out:
            for locus in filtered_loci:
                f_out.write(f'{locus}\n')

    common_diverse_loci = []
    for locus in high_diversity_loci['Bp']['ordered_loci'][:100]:
        if locus in high_diversity_loci['A']['ordered_loci'][:100]:
            common_diverse_loci.append(locus)
    print(common_diverse_loci)
    with open('../results/tests/syn_os_common_high_diversity_loci.txt', 'w') as f_out:
        for locus in common_diverse_loci:
            f_out.write(f'{locus}\n')

def plot_intergenic_linkage(f_loci='../results/tests/syn_os_common_high_diversity_loci.txt', main_cloud_radius=0.1):
    loci = np.loadtxt(f_loci, dtype='U10')
    print(len(loci))

    #num_loci = len(loci)
    num_loci = 20

    # Read loci alignments
    #alnmt_dict = read_loci_alignments(loci[:num_loci], '../results/single-cell/sscs/os-merged_alignments/')
    alnmt_dict = {}
    added_loci = 0
    i = 0
    while i < len(loci) and added_loci < num_loci:
        locus = loci[i]
        aln = AlignIO.read(f'../results/single-cell/sscs/os-merged_alignments/{locus}_codon_aln_MUSCLE.fasta', 'fasta')
        filtered_aln = seq_utils.filter_alignment_gaps(aln, gap_threshold=0.25)
        synbp_aln = seq_utils.filter_species_alignment(aln, 'synbp')
        main_cloud_aln = seq_utils.filter_main_cloud(synbp_aln, radius=main_cloud_radius)
        if len(main_cloud_aln) > 20:
            alnmt_dict[locus] = main_cloud_aln
            added_loci += 1
        i += 1
    print(alnmt_dict)
    print(len(alnmt_dict))
    alnmt_loci = sorted(list(alnmt_dict.keys()))
    r2_intergenic = pd.DataFrame(np.nan, index=alnmt_loci, columns=alnmt_loci)

    num_loci = 10
    num_panels = num_loci * (num_loci - 1) / 2
    num_columns = int(np.sqrt(num_panels)) - 1
    nr = num_panels // num_columns
    num_rows =  int(nr + ((num_columns * nr) < num_panels))
    fig, axes = make_small_multipanel(panels=(num_rows, num_columns))

    ax_idx = 0
    for i in range(num_loci):
        for j in range(i):
            l1 = alnmt_loci[i]
            l2 = alnmt_loci[j]
            aln_list = []
            for locus in [l1, l2]:
                aln = alnmt_dict[locus]
                aln_snps = seq_utils.get_snps(aln)
                if len(aln) > 20:
                    aln_list.append(aln_snps)
                else:
                    print(aln)
                    print(filtered_aln)
                    print(aln_snps)
            if len(aln_list) == 2:
                merged_aln = seq_utils.merge_alignments(aln_list)
                merged_aln = seq_utils.filter_alignment_gaps(merged_aln, gap_threshold=0.1)
                if len(merged_aln) > 10:
                    axes[ax_idx].set_title(f'{l1}-{l2}', fontsize=6)
                    print(l1, l2)
                    print(merged_aln)
                    f_sites1 = len(aln_list[0][0]) / (len(aln_list[0][0]) + len(aln_list[1][0]))
                    merged_snps = seq_utils.get_snps(merged_aln)

                    r2_ij = ld.calculate_rsquared(merged_snps)
                    num_sites = len(r2_ij)
                    axes[ax_idx].set_xticks([f_sites1 * num_sites])
                    axes[ax_idx].set_yticks([f_sites1 * num_sites])
                    im = axes[ax_idx].imshow(np.log10(r2_ij + 1E-5), origin='lower', cmap='Blues', aspect='equal')
                    axes[ax_idx].plot([0, num_sites], 2 * [f_sites1 * num_sites], lw=1, c='k')
                    axes[ax_idx].plot(2 * [f_sites1 * num_sites], [0, num_sites], lw=1, c='k')
                    ax_idx += 1

                    n11 = int(f_sites1 * num_sites)
                    if abs(num_sites - n11) > 10:
                        r2_11 = np.mean(utils.get_matrix_triangle_values(r2_ij[:n11, :n11], k=1))
                        r2_22 = np.mean(utils.get_matrix_triangle_values(r2_ij[n11:, n11:], k=1))
                        r2_12 = np.mean(r2_ij[:n11, n11:])
                        r2_intergenic.loc[l1, l1] = r2_11
                        r2_intergenic.loc[l1, l2] = r2_12
                        r2_intergenic.loc[l2, l1] = r2_12
                        r2_intergenic.loc[l2, l2] = r2_22
            print('\n')

    plt.tight_layout()
    plt.savefig(f'../figures/analysis/linkage/synbp_diverse_loci_intergenic_linkage_multipanel.pdf')

    fig = plt.figure(figsize=(double_col_width, double_col_width))
    ax = fig.add_subplot(111)

    r2_plot = r2_intergenic.dropna(axis=0, how='all')
    r2_plot = r2_plot.dropna(axis=1, how='all')
    print(r2_plot)
    ax.set_xticks(np.arange(len(r2_plot)))
    ax.set_xticklabels(r2_plot.index, fontsize=8, rotation=45)
    ax.set_yticks(np.arange(len(r2_plot)))
    ax.set_yticklabels(r2_plot.columns, fontsize=8, rotation=45)

    im = ax.imshow(r2_plot.values, origin='lower', cmap='Blues', aspect='equal')
    #fig.colorbar(im, label='$log_{10}r^2$')
    fig.colorbar(im, label=r'$\langle r^2 \rangle$')

    plt.tight_layout()
    plt.savefig(f'../figures/analysis/linkage/synbp_diverse_loci_intergenic_linkage_map.pdf')


def read_loci_alignments(loci, input_dir):
    alnmt_dict = {}
    for locus in loci:
        aln = AlignIO.read(f'{input_dir}{locus}_codon_aln_MUSCLE.fasta', 'fasta')
        alnmt_dict[locus] = aln
    return alnmt_dict

def test_interspecies_recombination(output_dir='../figures/analysis/tests/linkage/interspecies_recombination/', random_seed=12345):
    metadata = MetadataMap()
    candidate_loci = np.loadtxt('../results/tests/interspecies_recombination_test_loci.txt', dtype='U10')
    pdist_dict = pickle.load(open('../results/tests/interspecies_recombination_test_loci_divergences.dat', 'rb'))
    print(len(pdist_dict))

    #syna_cutoff = 0.15
    #synbp_cutoff = 0.15
    pS_cutoff = 0.15
    pN_cutoff = 0.05
    locus_pi_dict = {'low_pi':[], 'medium_pi':[], 'high_pi':[]}
    low_pi_values = []
    high_pi_values = []
    for locus in pdist_dict:
        pN, pS = pdist_dict[locus]
        sag_ids = list(pS.index)
        if "OS-A" not in sag_ids or "OS-B'" not in sag_ids:
            continue
        species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
        pS_aa = pS.loc[species_sorted_sags['A'], species_sorted_sags['A']]
        #core_syna_ids = list(pS_aa.loc[pS_aa["OS-A"] < pS_cutoff, :].index)
        core_syna_ids = list(pN.loc[species_sorted_sags['A'], :].loc[pN.loc[species_sorted_sags['A'], "OS-A"] < pN_cutoff, :].index)
        if len(core_syna_ids) < 10:
            continue
        pS_aa = pS_aa.loc[core_syna_ids, core_syna_ids]
        aa_values = np.array(utils.get_matrix_triangle_values(pS_aa.values, k=1), dtype=np.float64)
        piS_aa = np.nanmedian(aa_values)
        piN_aa = np.nanmedian(utils.get_matrix_triangle_values(np.array(pN.loc[core_syna_ids, core_syna_ids].values, dtype=np.float64), k=1))

        pS_bpbp = pS.loc[species_sorted_sags['Bp'], species_sorted_sags['Bp']]
        #core_synbp_ids = list(pS_bpbp.loc[pS_bpbp["OS-B'"] < pS_cutoff, :].index)
        core_synbp_ids = list(pN.loc[species_sorted_sags['Bp'], :].loc[pN.loc[species_sorted_sags['Bp'], "OS-B'"] < pN_cutoff, :].index)
        if len(core_synbp_ids) < 10:
            continue
        pS_bpbp = pS_bpbp.loc[core_synbp_ids, core_synbp_ids]
        bpbp_values = np.array(utils.get_matrix_triangle_values(pS_bpbp.values, k=1), dtype=np.float64)
        piS_bpbp = np.nanmedian(bpbp_values)
        piN_bpbp = np.nanmedian(utils.get_matrix_triangle_values(np.array(pN.loc[core_synbp_ids, core_synbp_ids].values, dtype=np.float64), k=1))

        pS_abp = pS.loc[core_syna_ids, core_synbp_ids]
        piS_abp = np.nanmedian(np.array(pS_abp.values, dtype=np.float64))
        piN_abp = np.nanmedian(np.array(pN.loc[core_syna_ids, core_synbp_ids].values, dtype=np.float64))

        if piS_abp < 0.15:
            locus_pi_dict['low_pi'].append({'locus':locus, 'syna_sag_ids':core_syna_ids, 'synbp_sag_ids':core_synbp_ids})
            low_pi_values.append([piS_aa, piS_bpbp, piS_abp, piN_aa, piN_bpbp, piN_abp])
        elif piS_abp > 0.2:
            locus_pi_dict['high_pi'].append({'locus':locus, 'syna_sag_ids':core_syna_ids, 'synbp_sag_ids':core_synbp_ids})
            high_pi_values.append([piS_aa, piS_bpbp, piS_abp, piN_aa, piN_bpbp, piN_abp])
        else:
            locus_pi_dict['medium_pi'].append(locus)
            print(locus, piS_aa, piS_bpbp, piS_abp)
    low_pi_values = np.array(low_pi_values)
    high_pi_values = np.array(high_pi_values)
    print(len(locus_pi_dict['low_pi']))
    print(len(locus_pi_dict['high_pi']))

    # Plot divergence distribution across loci
    np.random.seed(random_seed)
    cmap = plt.get_cmap('tab10')

    fig = plt.figure(figsize=(double_col_width, 0.4 * double_col_width))
    ax = fig.add_subplot(121)
    ax.set_title('low divergence')
    ax.set_xlabel('median pairwise divergence')
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.7)
    ax.set_yticks([])
    y = np.random.uniform(0, 0.1, size=len(low_pi_values))
    ax.scatter(low_pi_values[:, 0], y, s=20, marker='o', fc='none', ec=cmap(1), label=r"$\pi_S^{AA}$")
    y = np.random.uniform(0.1, 0.2, size=len(low_pi_values))
    ax.scatter(low_pi_values[:, 1], y, s=20, marker='o', fc='none', ec=cmap(0), label=r"$\pi_S^{B'B'}$")
    y = np.random.uniform(0.2, 0.3, size=len(low_pi_values))
    ax.scatter(low_pi_values[:, 2], y, s=20, marker='o', fc='none', ec=cmap(2), label=r"$\pi_S^{AB'}$")

    y = np.random.uniform(0.3, 0.4, size=len(low_pi_values))
    ax.scatter(low_pi_values[:, 3], y, s=20, marker='x', fc=cmap(1), label=r"$\pi_N^{AA}$")
    y = np.random.uniform(0.4, 0.5, size=len(low_pi_values))
    ax.scatter(low_pi_values[:, 4], y, s=20, marker='x', fc=cmap(0), label=r"$\pi_N^{B'B'}$")
    y = np.random.uniform(0.5, 0.6, size=len(low_pi_values))
    ax.scatter(low_pi_values[:, 5], y, s=20, marker='x', fc=cmap(2), label=r"$\pi_N^{AB'}$")

    ax.legend(fontsize=6)

    ax = fig.add_subplot(122)
    ax.set_title('typical divergence')
    ax.set_xlabel('median pairwise divergence')
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.7)
    ax.set_yticks([])
    y = np.random.uniform(0, 0.1, size=len(high_pi_values))
    ax.scatter(high_pi_values[:, 0], y, s=20, marker='o', fc='none', ec=cmap(1), label=r"$\pi_S^{AA}$")
    y = np.random.uniform(0.1, 0.2, size=len(high_pi_values))
    ax.scatter(high_pi_values[:, 1], y, s=20, marker='o', fc='none', ec=cmap(0), label=r"$\pi_S^{B'B'}$")
    y = np.random.uniform(0.2, 0.3, size=len(high_pi_values))
    ax.scatter(high_pi_values[:, 2], y, s=20, marker='o', fc='none', ec=cmap(2), label=r"$\pi_S^{AB'}$")

    y = np.random.uniform(0.3, 0.4, size=len(high_pi_values))
    ax.scatter(high_pi_values[:, 3], y, s=20, marker='x', fc=cmap(1), label=r"$\pi_N^{AA}$")
    y = np.random.uniform(0.4, 0.5, size=len(high_pi_values))
    ax.scatter(high_pi_values[:, 4], y, s=20, marker='x', fc=cmap(0), label=r"$\pi_N^{B'B'}$")
    y = np.random.uniform(0.5, 0.6, size=len(high_pi_values))
    ax.scatter(high_pi_values[:, 5], y, s=20, marker='x', fc=cmap(2), label=r"$\pi_N^{AB'}$")

    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(f'{output_dir}interspecies_recombination_test_loci_divergences.pdf')


    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(121)
    ax.set_xlabel(r'median synonymous divergence, $K_S$')
    #ax.set_xlim(0, 0.6)
    ax.set_xlim(1E-3, 1)
    ax.set_xscale('log')
    ax.set_ylabel(r'median non-synonymous divergence, $K_a$')
    #ax.set_ylim(0, 0.25)
    ax.set_ylim(1E-4, 1)
    ax.set_yscale('log')

    ax.scatter(high_pi_values[:, 0], high_pi_values[:, 3], s=20, marker='o', fc='none', ec=cmap(1), label=r"typical A")
    ax.scatter(high_pi_values[:, 1], high_pi_values[:, 4], s=20, marker='o', fc='none', ec=cmap(0), label=r"typical B'")
    ax.scatter(high_pi_values[:, 2], high_pi_values[:, 5], s=20, marker='o', fc='none', ec=cmap(2), label=r"typical A-B'")

    ax.scatter(low_pi_values[:, 0], low_pi_values[:, 3], s=20, marker='x', fc=cmap(1), label=r"low A")
    ax.scatter(low_pi_values[:, 1], low_pi_values[:, 4], s=20, marker='x', fc=cmap(0), label=r"low B'")
    ax.scatter(low_pi_values[:, 2], low_pi_values[:, 5], s=20, marker='x', fc=cmap(2), label=r"low A-B'")

    #ax.plot([0.01, 0.1], [0.01, 0.1], ls='--', c='k')
    ax.legend(fontsize=6)

    ax = fig.add_subplot(122)
    ax.set_xlabel(r'median synonymous divergence, $K_S$')
    #ax.set_xlim(0, 0.6)
    ax.set_xlim(1E-3, 1)
    ax.set_xscale('log')
    ax.set_ylabel(r'$K_a/K_S$')
    #ax.set_ylim(0, 0.25)
    ax.set_ylim(0, 1.5)

    dNdS = high_pi_values[:, [3, 4, 5]] / (high_pi_values[:, [0, 1, 2]] + (high_pi_values[:, [0, 1, 2]] == 0))
    ax.scatter(high_pi_values[:, 0], dNdS[:, 0], s=20, marker='o', fc='none', ec=cmap(1), label=r"typical A")
    ax.scatter(high_pi_values[:, 1], dNdS[:, 1], s=20, marker='o', fc='none', ec=cmap(0), label=r"typical B'")
    ax.scatter(high_pi_values[:, 2], dNdS[:, 2], s=20, marker='o', fc='none', ec=cmap(2), label=r"typical A-B'")

    dNdS = low_pi_values[:, [3, 4, 5]] / (low_pi_values[:, [0, 1, 2]] + (low_pi_values[:, [0, 1, 2]] == 0))
    ax.scatter(low_pi_values[:, 0], dNdS[:, 0], s=20, marker='x', fc=cmap(1), label=r"low A")
    ax.scatter(low_pi_values[:, 1], dNdS[:, 1], s=20, marker='x', fc=cmap(0), label=r"low B'")
    ax.scatter(low_pi_values[:, 2], dNdS[:, 2], s=20, marker='x', fc=cmap(2), label=r"low A-B'")

    #ax.plot([0.01, 0.1], [0.01, 0.1], ls='--', c='k')

    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(f'{output_dir}interspecies_recombination_test_loci_dNdS.pdf')

    # Calculate linkage
    #calculate_interspecies_r2(locus_pi_dict)
    r2_dict = pickle.load(open('../results/tests/interspecies_recombination_test_loci_r2_matrices.dat', 'rb'))
    print(r2_dict)

    # Plot r2 curves
    fig = plt.figure(figsize=(double_col_width, 0.4 * double_col_width))
    ax = fig.add_subplot(121)
    ax.set_title('typical divergence loci')
    ax.set_xlabel('separation, x')
    ax.set_xscale('log')
    ax.set_xlim(4E-1, 5E3)
    ax.set_ylabel(r'linkage disequilibrium, $r^2$')
    ax.set_yscale('log')
    ax.set_ylim(1E-4, 1.5E0)

    r2_aa = average_over_loci(r2_dict['high_pi']['aa'])
    x_cg, r2_cg, _ = average_over_distance(r2_aa, denom=1.5, coarse_grain=True, num_cg_points=20)
    ax.plot(x_cg + 0.5, r2_cg, '-o', lw=1, ms=3, alpha=1.0, c=cmap(1), label=f'Syn. A')

    r2_bpbp = average_over_loci(r2_dict['high_pi']['bpbp'])
    x_cg, r2_cg, _ = average_over_distance(r2_bpbp, denom=1.5, coarse_grain=True, num_cg_points=20)
    ax.plot(x_cg + 0.5, r2_cg, '-o', lw=1, ms=3, alpha=1.0, c=cmap(0), label=f'Syn. B\'')

    r2_abp = average_over_loci(r2_dict['high_pi']['abp'])
    x_cg, r2_cg, _ = average_over_distance(r2_abp, denom=1.5, coarse_grain=True, num_cg_points=20)
    ax.plot(x_cg + 0.5, r2_cg, '-o', lw=1, ms=3, alpha=1.0, c=cmap(2), label=f'A-B\'')
    ax.legend(fontsize=8)


    ax = fig.add_subplot(122)
    ax.set_title('low divergence loci')
    ax.set_xlabel('separation, x')
    ax.set_xscale('log')
    ax.set_xlim(4E-1, 5E3)
    ax.set_ylabel(r'linkage disequilibrium, $r^2$')
    ax.set_yscale('log')
    ax.set_ylim(1E-4, 1.5E0)

    r2_aa = average_over_loci(r2_dict['low_pi']['aa'])
    x_cg, r2_cg, _ = average_over_distance(r2_aa, denom=1.5, coarse_grain=True, num_cg_points=20)
    ax.plot(x_cg + 0.5, r2_cg, '-o', lw=1, ms=3, alpha=1.0, c=cmap(1), label=f'Syn. A')

    r2_bpbp = average_over_loci(r2_dict['low_pi']['bpbp'])
    x_cg, r2_cg, _ = average_over_distance(r2_bpbp, denom=1.5, coarse_grain=True, num_cg_points=20)
    ax.plot(x_cg + 0.5, r2_cg, '-o', lw=1, ms=3, alpha=1.0, c=cmap(0), label=f'Syn. B\'')

    r2_abp = average_over_loci(r2_dict['low_pi']['abp'])
    x_cg, r2_cg, _ = average_over_distance(r2_abp, denom=1.5, coarse_grain=True, num_cg_points=20)
    ax.plot(x_cg + 0.5, r2_cg, '-o', lw=1, ms=3, alpha=1.0, c=cmap(2), label=f'A-B\'')

    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'../figures/analysis/tests/linkage/interspecies_recombination/interspecies_r2.pdf')

def calculate_interspecies_r2(locus_pi_dict, output_file='../results/tests/interspecies_recombination_test_loci_r2_matrices.dat'):
    r2_dict = {}
    for locus_type in locus_pi_dict:
        locus_type_dict = {'aa':[], 'bpbp':[], 'abp':[], 'loci':[]}
        for locus_dict in locus_pi_dict[locus_type]:
            locus = locus_dict['locus']

            aln = AlignIO.read(f'../results/single-cell/sscs/os-merged_alignments/{locus}_codon_aln_MUSCLE.fasta', 'fasta')
            aa_aln = MultipleSeqAlignment([rec for rec in aln if rec.id in locus_dict['syna_sag_ids']])
            aa_aln = seq_utils.filter_alignment_gaps(aa_aln, gap_threshold=0.25)
            #aa_syn_aln = seq_utils.get_synonymous_sites(aa_aln)
            #r2_aa = ld.calculate_rsquared(aa_syn_aln)
            r2_aa = ld.calculate_rsquared(aa_aln)

            bpbp_aln = MultipleSeqAlignment([rec for rec in aln if rec.id in locus_dict['synbp_sag_ids']])
            bpbp_aln = seq_utils.filter_alignment_gaps(bpbp_aln, gap_threshold=0.25)
            #bpbp_syn_aln = seq_utils.get_synonymous_sites(bpbp_aln)
            #r2_bpbp = ld.calculate_rsquared(bpbp_syn_aln)
            r2_bpbp = ld.calculate_rsquared(bpbp_aln)

            merged_sag_ids = locus_dict['syna_sag_ids'] + locus_dict['synbp_sag_ids']
            abp_aln = MultipleSeqAlignment([rec for rec in aln if rec.id in merged_sag_ids])
            abp_aln = seq_utils.filter_alignment_gaps(abp_aln, gap_threshold=0.25)
            #abp_syn_aln = seq_utils.get_synonymous_sites(abp_aln)
            #r2_abp = ld.calculate_rsquared(abp_syn_aln)
            r2_abp = ld.calculate_rsquared(abp_aln)

            locus_type_dict['loci'].append(locus)
            locus_type_dict['aa'].append(r2_aa)
            locus_type_dict['bpbp'].append(r2_bpbp)
            locus_type_dict['abp'].append(r2_abp)

        r2_dict[locus_type] = locus_type_dict
    pickle.dump(r2_dict, open(output_file, 'wb'))


def plot_species_linkage_comparison(output_dir='../figures/analysis/linkage/', random_seed=12345):
    metadata = MetadataMap()
    candidate_loci = np.loadtxt('../results/tests/other_tests/interspecies_recombination_test_loci.txt', dtype='U10')
    pdist_dict = pickle.load(open('../results/tests/other_tests/interspecies_recombination_test_loci_divergences.dat', 'rb'))
    print(len(pdist_dict))

    #syna_cutoff = 0.15
    #synbp_cutoff = 0.15
    pS_cutoff = 0.15
    pN_cutoff = 0.05
    locus_pi_dict = {'low_pi':[], 'medium_pi':[], 'high_pi':[]}
    low_pi_values = []
    high_pi_values = []
    for locus in pdist_dict:
        pN, pS = pdist_dict[locus]
        sag_ids = list(pS.index)
        if "OS-A" not in sag_ids or "OS-B'" not in sag_ids:
            continue
        species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
        pS_aa = pS.loc[species_sorted_sags['A'], species_sorted_sags['A']]
        #core_syna_ids = list(pS_aa.loc[pS_aa["OS-A"] < pS_cutoff, :].index)
        core_syna_ids = list(pN.loc[species_sorted_sags['A'], :].loc[pN.loc[species_sorted_sags['A'], "OS-A"] < pN_cutoff, :].index)
        if len(core_syna_ids) < 10:
            continue
        pS_aa = pS_aa.loc[core_syna_ids, core_syna_ids]
        aa_values = np.array(utils.get_matrix_triangle_values(pS_aa.values, k=1), dtype=np.float64)
        piS_aa = np.nanmedian(aa_values)
        piN_aa = np.nanmedian(utils.get_matrix_triangle_values(np.array(pN.loc[core_syna_ids, core_syna_ids].values, dtype=np.float64), k=1))

        pS_bpbp = pS.loc[species_sorted_sags['Bp'], species_sorted_sags['Bp']]
        #core_synbp_ids = list(pS_bpbp.loc[pS_bpbp["OS-B'"] < pS_cutoff, :].index)
        core_synbp_ids = list(pN.loc[species_sorted_sags['Bp'], :].loc[pN.loc[species_sorted_sags['Bp'], "OS-B'"] < pN_cutoff, :].index)
        if len(core_synbp_ids) < 10:
            continue
        pS_bpbp = pS_bpbp.loc[core_synbp_ids, core_synbp_ids]
        bpbp_values = np.array(utils.get_matrix_triangle_values(pS_bpbp.values, k=1), dtype=np.float64)
        piS_bpbp = np.nanmedian(bpbp_values)
        piN_bpbp = np.nanmedian(utils.get_matrix_triangle_values(np.array(pN.loc[core_synbp_ids, core_synbp_ids].values, dtype=np.float64), k=1))

        pS_abp = pS.loc[core_syna_ids, core_synbp_ids]
        piS_abp = np.nanmedian(np.array(pS_abp.values, dtype=np.float64))
        piN_abp = np.nanmedian(np.array(pN.loc[core_syna_ids, core_synbp_ids].values, dtype=np.float64))

        if piS_abp < 0.15:
            locus_pi_dict['low_pi'].append({'locus':locus, 'syna_sag_ids':core_syna_ids, 'synbp_sag_ids':core_synbp_ids})
            low_pi_values.append([piS_aa, piS_bpbp, piS_abp, piN_aa, piN_bpbp, piN_abp])
        elif piS_abp > 0.2:
            locus_pi_dict['high_pi'].append({'locus':locus, 'syna_sag_ids':core_syna_ids, 'synbp_sag_ids':core_synbp_ids})
            high_pi_values.append([piS_aa, piS_bpbp, piS_abp, piN_aa, piN_bpbp, piN_abp])
        else:
            locus_pi_dict['medium_pi'].append(locus)
            print(locus, piS_aa, piS_bpbp, piS_abp)
    low_pi_values = np.array(low_pi_values)
    high_pi_values = np.array(high_pi_values)
    print(len(locus_pi_dict['low_pi']))
    print(len(locus_pi_dict['high_pi']))


    # Calculate linkage
    #calculate_interspecies_r2(locus_pi_dict)
    r2_dict = pickle.load(open('../results/tests/other_tests/interspecies_recombination_test_loci_r2_matrices.dat', 'rb'))
    print(r2_dict)

    # Plot r2 curves
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'separation, $x$')
    ax.set_xscale('log')
    #ax.set_xlim(4E-1, 5E3)
    ax.set_xlim(8E-1, 5E3)
    ax.set_ylabel(r'linkage disequilibrium, $r^2$')
    ax.set_yscale('log')
    ax.set_ylim(1E-4, 1.5E0)

    r2_aa = average_over_loci(r2_dict['high_pi']['aa'])
    x_cg, r2_cg, _ = average_over_distance(r2_aa, denom=1.5, coarse_grain=True, num_cg_points=20)
    #ax.plot(x_cg + 0.5, r2_cg, '-o', lw=1, ms=3, alpha=1.0, c='tab:orange', label=r'$\alpha$')
    ax.plot(x_cg[1:], r2_cg[1:], '-o', lw=1, ms=3, alpha=1.0, c='tab:orange', label=r'$\alpha$')

    r2_bpbp = average_over_loci(r2_dict['high_pi']['bpbp'])
    x_cg, r2_cg, _ = average_over_distance(r2_bpbp, denom=1.5, coarse_grain=True, num_cg_points=20)
    #ax.plot(x_cg + 0.5, r2_cg, '-o', lw=1, ms=3, alpha=1.0, c='tab:blue', label=r'$\beta$')
    ax.plot(x_cg[1:], r2_cg[1:], '-o', lw=1, ms=3, alpha=1.0, c='tab:blue', label=r'$\beta$')

    r2_abp = average_over_loci(r2_dict['high_pi']['abp'])
    x_cg, r2_cg, _ = average_over_distance(r2_abp, denom=1.5, coarse_grain=True, num_cg_points=20)
    #ax.plot(x_cg + 0.5, r2_cg, '-o', lw=1, ms=3, alpha=1.0, c='k', label=r'$\alpha$ vs $\beta$')
    ax.plot(x_cg[1:], r2_cg[1:], '-o', lw=1, ms=3, alpha=1.0, c='k', label=r'$\alpha$ vs $\beta$')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'../figures/analysis/linkage/species_rsq_comparison.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()

    if args.test == True:
        #test_species_linkage()
        #test_syntey_linkage()
        #test_allele_linkage()
        #plot_ribosomal_linkage()
        #plot_random_rho_theory(rho0=0.01)
        #test_sequence_filtering()
        #find_highly_diverse_loci()
        #test_locus_averaging()
        #plot_intergenic_linkage(f_loci='../results/tests/syn_OSBp_high_diversity_loci.txt')
        test_interspecies_recombination()
    else:
        print('Plotting linkage disequilibria...')
        plot_species_linkage_comparison()


