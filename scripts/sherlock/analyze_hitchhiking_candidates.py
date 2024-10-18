import argparse
import os
import glob
import time
import numpy as np
import pandas as pd
import pickle
import copy
import utils
import matplotlib.pyplot as plt
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
import scipy.stats as stats
import scipy.cluster.hierarchy as hclust
import scipy.spatial.distance as distance
import calculate_linkage_disequilibria as ld
import plot_linkage_disequilibrium as plt_ld
from pangenome_utils import PangenomeMap
from metadata_map import MetadataMap
from syn_homolog_map import SynHomologMap
from plot_utils import *
from Bio import AlignIO
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

def contruct_hitchhiking_allele_haplotypes(hitchhiking_candidates, loci_alignments):
    haplotype_df = initialize_haplotype_table(loci_alignments)
    aln_dict = construct_loci_alignments_dict(loci_alignments)
    for locus in hitchhiking_candidates:
        hitchhiking_dict = hitchhiking_candidates[locus]
        core_sites_aln = extract_sweeping_allele_alignments(aln_dict[locus], hitchhiking_dict['segments'])
        sag_alleles = group_sags_by_alleles(core_sites_aln)

        for allele_dict in sag_alleles:
            sag_ids = np.concatenate(list(allele_dict.values()))
            for allele_id in allele_dict:
                haplotype_df.loc[allele_dict[allele_id], locus] = allele_id

    return haplotype_df


def initialize_haplotype_table(loci_alignments):
    loci = list([data_tuple[0] for data_tuple in loci_alignments])
    sag_ids = []
    for locus, aln in loci_alignments:
        for rec in aln:
            if rec.id not in sag_ids:
                sag_ids.append(rec.id)
    table = pd.DataFrame(0, index=sag_ids, columns=loci)
    return table


def construct_loci_alignments_dict(loci_alignments):
    aln_dict = {}
    for locus, aln in loci_alignments:
        aln_dict[locus] = aln
    return aln_dict


def group_sags_by_alleles(core_sites_aln):
    sag_alleles = []
    for aln in core_sites_aln:
        allele_dict = {}
        sag_ids = np.array([rec.id for rec in aln])
        aln_seqs = np.array(aln)
        alleles, allele_counts = utils.sorted_unique(aln_seqs)
        for i, allele in enumerate(alleles):
            allele_dict[i + 1] = sag_ids[np.concatenate(aln_seqs) == allele]
        sag_alleles.append(allele_dict)
    return sag_alleles

def calculate_hitchhiking_allele_linkage(locus_data, hitchhiking_segments, min_samples=20):
    site_ids = label_hitchhiking_sites(locus_data, hitchhiking_segments)
    num_loci = len(locus_data)

    allele_alnmts = []
    for i in range(num_loci):
        alnmts = extract_sweeping_allele_alignments(locus_data[i][1], hitchhiking_segments[i])
        allele_alnmts += alnmts

    # Calculate linkage for each pair separately to maximize sample size per locus
    rsq_df = pd.DataFrame(index=site_ids, columns=site_ids)
    for j in range(len(site_ids)):
        for i in range(j):
            merged_aln = seq_utils.merge_alignments([allele_alnmts[i], allele_alnmts[j]])
            if len(merged_aln) > min_samples:
                Dsq, denom = align_utils.calculate_ld_matrices_vectorized(merged_aln, unbiased=False)
                rsq = Dsq / (denom + (denom == 0))
                rsq_df.loc[[site_ids[i], site_ids[j]], [site_ids[i], site_ids[j]]] = rsq
                if rsq[0, 1] != rsq[1, 0]:
                    print(sites_ids[i], sites_ids[j])
                    print(rsq)
                    print('\n')
            else:
                rsq_df.loc[site_ids[i], site_ids[j]] = np.nan
                rsq_df.loc[site_ids[j], site_ids[i]] = np.nan
    return rsq_df


def label_hitchhiking_sites(locus_data, hitchhiking_segments):
    site_ids = []
    for i, data in enumerate(locus_data):
        locus = data[0]
        segments = [hitchhiking_segments[i]]
        for segment in segments:
            i_core = np.mean(segment, dtype=int)
            site_ids.append(f'{locus}_xc{i_core}')
    return site_ids

def extract_sweeping_allele_alignments(aln, hitchhiking_segments):
    trimmed_aln = align_utils.trim_alignment_gaps(aln, start_gap_perc=0.05)
    aln_snps, x_snps = seq_utils.get_snps(trimmed_aln, return_x=True)
    core_sites_alnmts = []
    for segment in hitchhiking_segments:
        i_core = int(np.mean(segment, dtype=int))
        allele_aln = MultipleSeqAlignment([SeqRecord(Seq(rec[i_core]), id=rec.id) for rec in aln_snps])
        core_sites_alnmts.append(allele_aln)
    return core_sites_alnmts

def read_locus_alignments(locus_ids, pangenome_map, alignments_dir):
    aln_data = []
    for locus_id in locus_ids:
        f_aln = f'{alignments_dir}{locus_id}_aln.fna'
        aln_mapped = seq_utils.read_alignment_and_map_sag_ids(f_aln, pangenome_map)
        if aln_mapped is not None:
            aln_data.append([locus_id, aln_mapped])
    return aln_data


def plot_rsq_map(rsq_df, cmap='Blues', cluster=False, log_scale=False, epsilon=1E-6, label_fontsize=3, savefig=None):
    palette = copy.copy(plt.get_cmap(cmap))
    palette.set_bad((0.5, 0.5, 0.5))

    if cluster:
        idx = np.array(list(rsq_df.dropna().index))
        pdist = distance.pdist(rsq_df.values)
        Z = hclust.linkage(pdist, method='average', optimal_ordering=True)
        dn = hclust.dendrogram(Z, no_plot=True)
        sorted_idx = list(idx[dn['leaves']])
        rsq_df = rsq_df.reindex(index=sorted_idx, columns=sorted_idx)

    fig = plt.figure(figsize=(double_col_width, double_col_width))
    ax = fig.add_subplot(111)

    ax.set_xticks(np.arange(len(rsq_df)))
    ax.set_xticklabels(rsq_df.index, fontsize=label_fontsize, rotation=90)
    ax.set_yticks(np.arange(len(rsq_df)))
    ax.set_yticklabels(rsq_df.columns, fontsize=label_fontsize, rotation=0)

    if log_scale == False:
        im = ax.imshow(rsq_df.astype(float).values, origin='lower', cmap=palette, aspect='equal')
    else:
        log_rsq = np.log10(rsq_df.astype(float).values + epsilon)
        im = ax.imshow(log_rsq, origin='lower', cmap=palette, aspect='equal')
    fig.colorbar(im, label=r'$\langle r^2 \rangle$')

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()

    if cluster:
        return Z

def plot_rsq_vs_osbp_distance(rsq_df, correlated_loci, osbp_cds_dict, max_tag_number=2942, savefig=None):
    l_half = max_tag_number // 2

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('approximate OS-B\' genome distance (kb)')
    ax.set_ylabel(r'$r^2$')

    ax.set_xscale('log')
    ax.set_xlim(5E-1, 2 * l_half)
    #ax.set_yscale('log')
    #ax.set_xlim(5E-1, 1.1 * l_half)
    ax.set_ylim([0, 1.1])

    x = []
    y = []
    rsq_index = np.array(list(rsq_df.index))
    og_ids = np.array([og_id.split('_xc')[0] for og_id in rsq_index])
    print(og_ids, len(og_ids))
    rsq_og_idx = [og_id not in correlated_loci for og_id in og_ids]
    og_ids = np.array([og_id for og_id in og_ids if og_id not in correlated_loci])
    print(og_ids, len(og_ids))
    for j in range(len(og_ids)):
        locus_tag_j = map_ssog_id(og_ids[j], osbp_cds_dict)
        if locus_tag_j is not None:
            for i in range(j):
                locus_tag_i = map_ssog_id(og_ids[i], osbp_cds_dict)
                if locus_tag_i is not None:
                    xi = int(locus_tag_i.split('_')[-1])
                    xj = int(locus_tag_j.split('_')[-1])
                    x.append((abs(xi - xj) + 1) % l_half)
                    i_rsq = rsq_index[rsq_og_idx][i]
                    j_rsq = rsq_index[rsq_og_idx][j]
                    y.append(rsq_df.loc[i_rsq, j_rsq])
    x = np.array(x)
    y = np.array(y)
    x = x[np.isfinite(y)]
    y = y[np.isfinite(y)]
    print(x, y)
    print(sum((y + 1E-6) <= 0))
    print('Pearson correlation: ', stats.pearsonr(np.log(x + 1E-3), np.log(y + 1E-6)))
    print('Spearman correlation: ', stats.spearmanr(np.log(x + 1E-3), np.log(y + 1E-6)))
    ax.scatter(x, y, s=10, fc='none', ec='tab:blue', lw=0.7)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()


def plot_gene_pair_linkage_matrix(r2_ij, boundary_idx, cmap='Blues', savefig=None):
    fig = plt.figure(figsize=(single_col_width, single_col_width))
    ax = fig.add_subplot(111)

    im = ax.imshow(r2_ij, origin='lower', cmap=cmap, aspect='equal')
    ax.axhline(boundary_idx, lw=1, c='k')
    ax.axvline(boundary_idx, lw=1, c='k')

    fig.colorbar(im)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()

def plot_gene_pair_average_linkage_comparison(rsq_dict, savefig=None):
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([r'$\langle r^2 \rangle_{11}$', r'$\langle r^2 \rangle_{12}$', r'$\langle r^2 \rangle_{22}$'])
    ax.set_ylim(1E-4, 1.2)

    cmap = plt.get_cmap('tab10')

    x = np.arange(3)
    counter = 0
    for gene_pair in rsq_dict:
        rsq, boundary_idx = rsq_dict[gene_pair]
        rsq_locus1 = rsq[:boundary_idx + 1, :boundary_idx + 1]
        rsq_locus2 = rsq[boundary_idx:, boundary_idx:]
        rsq_locus12 = rsq[boundary_idx:, :boundary_idx + 1]
        rsq_mean = [np.mean(rsq_locus1), np.mean(rsq_locus12), np.mean(rsq_locus2)]
        rsq_std = [np.std(rsq_locus1), np.std(rsq_locus12), np.std(rsq_locus2)]
        ax.plot(x, rsq_mean, '-o', c=cmap(counter), mec=cmap(counter), mfc='none', label=f'{gene_pair[0]}-{gene_pair[1]}')
        ax.errorbar(x, rsq_mean, yerr=rsq_std, c=cmap(counter))
        counter += 1

    ax.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()

def plot_linkage_curve(rsq_avg, savefig=None):
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('genome separation, $x$')
    ax.set_xscale('log')
    ax.set_xlim(5E-1, 2 * len(rsq_avg))
    ax.set_ylabel(r'$\langle r^2 \rangle$')
    ax.set_ylim(1E-4, 2.0)
    ax.set_yscale('log')

    x = np.arange(len(rsq_avg))
    ax.plot(x, rsq_avg, '-o', ms=4)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()


def plot_species_composition(locus_alns, metadata, savefig=None):
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('OS-B\' cell fraction')
    ax.set_ylabel('hitchhiking events')

    synbp_fractions = []
    for locus, aln in locus_alns:
        sag_ids = [rec.id for rec in aln]
        species_sag_ids = metadata.sort_sags(sag_ids, by='species')
        synbp_fractions.append(len(species_sag_ids['Bp']) / len(sag_ids))
    ax.hist(synbp_fractions, range=(0, 1), bins=50, alpha=0.8)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()


def clean_rsq_matrix(rsq_df, high_freq_na=0.25, max_na_count=0):
    # Remove sites with high NA frequency
    clean_sites = list(rsq_df.index)
    high_na_sites = []
    for site_id in clean_sites:
        if np.sum(rsq_df[site_id].isna()) > high_freq_na * len(clean_sites):
            high_na_sites.append(site_id)

    #for site_id in ['CYB_1769_xc6', 'CYB_0582_xc10']:
    #    clean_sites.remove(site_id)
    for site_id in high_na_sites:
        clean_sites.remove(site_id)
    rsq_clean = rsq_df.loc[clean_sites, clean_sites]

    # Remove remaining sites with NAs
    sites_to_remove = []
    for site_id in clean_sites:
        if np.sum(rsq_clean[site_id].isna()) > max_na_count:
            sites_to_remove.append(site_id)

    for site_id in sites_to_remove:
        clean_sites.remove(site_id)
    rsq_clean = rsq_df.loc[clean_sites, clean_sites]
    #for site_id in rsq_clean.index:
    #    rsq_clean.loc[site_id, site_id] = 1

    return rsq_clean

def plot_haplotype_map(allele_table, min_presence=0.5, cmap='Set1', cluster=False, cell_labels=False, savefig=None):
    plot_table = allele_table.loc[np.sum(allele_table > 0, axis=1) > min_presence * allele_table.shape[1], :]
    if cluster:
        idx = np.array(list(plot_table.dropna().index))
        pdist = distance.pdist(plot_table.values, metric='hamming')
        Z = hclust.linkage(pdist, method='average', optimal_ordering=True)
        dn = hclust.dendrogram(Z, no_plot=True)
        sorted_idx = list(idx[dn['leaves']])
        plot_table = plot_table.reindex(index=sorted_idx)
    plot_table = plot_table.replace(0, np.nan)

    palette = plt.get_cmap(cmap)
    #palette = copy.copy(plt.get_cmap(cmap))
    #palette.set_bad([0.5, 0.5, 0.5, 0.5])

    fig = plt.figure(figsize=(double_col_width, 0.8 * double_col_width))
    ax = fig.add_subplot(111)
    ax.set_xticks(np.arange(plot_table.shape[1]) - 0.25)
    ax.set_xticklabels(list(plot_table.columns), fontsize=6, rotation=90)
    if cell_labels == True:
        ax.set_yticks(np.arange(plot_table.shape[0]))
        ax.set_yticklabels(list(plot_table.index), fontsize=6)
    else:
        ax.set_yticks([])
    im = ax.imshow(plot_table, origin='lower', cmap=cmap, aspect='auto', vmin=1, vmax=9)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()

    if cluster:
        return idx, Z

def add_gene_id_map(cds_dict):
    updated_dict = copy.copy(cds_dict)
    for cds in cds_dict:
        annot = cds_dict[cds]
        if 'gene' in annot.qualifiers:
            gene_id = annot.qualifiers['gene'][0]
            updated_dict[gene_id] = cds
    return updated_dict

def find_ssog_reference_neighbors(ssog_list, ref_cds_dict, candidate_neighbors=None, max_distance=3):
    genome_neighbors_dict = {}
    if candidate_neighbors is not None:
        for ssog_id in ssog_list:
            gene_id = map_ssog_id(ssog_id, ref_cds_dict)
            neighbors = find_genome_neighbors(gene_id, candidate_neighbors, ref_cds_dict, max_distance)
            genome_neighbors_dict[ssog_id] = neighbors
    return genome_neighbors_dict

def map_ssog_id(ssog_id, ref_cds_dict):
    if 'CYB_' in ssog_id or 'CYA_' in ssog_id:
        gene_id = ssog_id
    elif 'YSG' not in ssog_id:
        if ssog_id in ref_cds_dict:
            gene_id = ref_cds_dict[ssog_id]
        elif ssog_id.split('-')[0] in ref_cds_dict:
            gene_id = ref_cds_dict[ssog_id.split('-')[0]]
        else:
            gene_id = None
    else:
        gene_id = None
    return gene_id


def find_genome_neighbors(gene_id, candidate_neighbors, ref_cds_dict, max_distance):
    genome_neighbors = []
    head, idx_str = gene_id.split('_')
    tag_idx = int(idx_str)

    # Look for genome neighbors within `max_distance` from target
    counter = 1
    while counter < max_distance + 1:
        for i in [1, -1]:
            candidate_tag = 'CYB_' + str(tag_idx + i * counter).zfill(4)
            if candidate_tag in candidate_neighbors:
                genome_neighbors.append(candidate_tag)
            elif candidate_tag in ref_cds_dict:
                annot = ref_cds_dict[candidate_tag]
                if 'gene' in annot.qualifiers:
                    candidate_gene_id = annot.qualifiers['gene'][0]
                    if candidate_gene_id in candidate_neighbors:
                        genome_neighbors.append(candidate_gene_id)
                    else:
                        for j in range(1, 4):
                            candidate_var = f'{candidate_gene_id}-{j}'
                            if candidate_var in candidate_neighbors:
                                genome_neighbors.append(candidate_var)
        counter += 1
    return genome_neighbors

def update_sorted_loci(sorted_loci, osbp_cds_dict, synbp_strain_ssog_ids):
    synbp_hitchhiking_neighbors = find_ssog_reference_neighbors(sorted_loci['correlated'], osbp_cds_dict, candidate_neighbors=synbp_strain_ssog_ids)
    sorted_loci['correlated_neighbors'] = list(np.unique([ssog_id for ssog_id in np.concatenate(list(synbp_hitchhiking_neighbors.values())) if ssog_id not in sorted_loci['correlated'] and ssog_id not in sorted_loci['uncorrelated']]))
    atypical_loci = list(np.concatenate(list(sorted_loci.values())))
    sorted_loci['typical'] = [ssog_id for ssog_id in synbp_strain_ssog_ids if ssog_id not in atypical_loci]
    return sorted_loci

def plot_strain_FST(strain_differences, sorted_loci, return_control=False, savefig=None):
    # Sample typical loci as control
    sample_size = len(sorted_loci['correlated'])
    control_loci = np.random.choice(sorted_loci['typical'], sample_size)
    print(control_loci, len(control_loci))

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlim(-1, 4)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(['correlated', 'correlated\nOS-B\' neighbors', 'uncorrelated', 'control'], fontsize=8, rotation=45)
    ax.set_ylabel(r'$F_{ST}$', fontweight='bold')

    locus_groups = [sorted_loci['correlated'], sorted_loci['correlated_neighbors'], sorted_loci['uncorrelated'], control_loci]
    for x0, loci in enumerate(locus_groups):
        FST_values = [strain_differences[locus]['F_ST'] for locus in loci if locus in strain_differences]
        print(x0, FST_values, len(FST_values))
        x = np.random.uniform(x0 - 0.25, x0 + 0.25, size=len(FST_values))
        ax.scatter(x, FST_values, s=10, ec='none', fc='tab:blue')
        FST_mean = np.mean(FST_values)
        ax.plot([x0 - 0.25, x0 + 0.25], [FST_mean, FST_mean], lw=1, c='tab:blue')

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()

    if return_control:
        return control_loci

def calculate_rsq_curves(strain_differences, loci, group='all'):
    results = {'curves':[]}
    rsq_list = []
    for locus in loci:
        if locus in strain_differences:
            rsq_matrix = strain_differences[locus][f'rsq_{group}']
            x, rsq_mean, rsq_std = plt_ld.average_over_distance(rsq_matrix, denom=1.5, coarse_grain=True, num_cg_points=25)
            results['curves'].append([x, rsq_mean])
            rsq_list.append(rsq_matrix)
            print(locus, rsq_mean[:3])

    rsq_loci_avg = plt_ld.average_over_loci(rsq_list)
    x, rsq_mean, rsq_std = plt_ld.average_over_distance(rsq_loci_avg, denom=1.5, coarse_grain=True, num_cg_points=25)
    results['mean'] = [x, rsq_mean]
    return results


def plot_rsq_curves(rsq_curve_dict, savefig=None):
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('genome distance (bp)')
    ax.set_xscale('log')
    ax.set_ylabel(r'$r^2$', fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim([1E-4, 1])

    x_mean, rsq_mean = rsq_curve_dict['mean']
    ax.plot(x_mean[np.isfinite(rsq_mean)], rsq_mean[np.isfinite(rsq_mean)], lw=2.5, c='k', label=r'$\langle r^2 \rangle$', zorder=3)

    for i, locus_data in enumerate(rsq_curve_dict['curves']):
        x, rsq = locus_data
        if i == 0:
            ax.plot(x[np.isfinite(rsq)], rsq[np.isfinite(rsq)], lw=1, c='gray', alpha=0.6, label='individual loci')
        else:
            ax.plot(x[np.isfinite(rsq)], rsq[np.isfinite(rsq)], lw=1, c='gray', alpha=0.6, label='individual loci')

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()

def plot_FST_distribution(strain_differences, sorted_loci, bins=50, cumulative=False, savefig=None):
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$F_{ST}$')
    ax.set_ylabel('counts')

    FST_values = np.array([strain_differences[locus]['F_ST'] for locus in strain_differences])
    print(f'F_ST < 0.25: {sum(FST_values < 0.25)}\n0.25 < F_ST < 0.5: {sum((FST_values > 0.25) * (FST_values < 0.5))}\nF_ST > 0.5: {sum(FST_values > 0.5)}\n\n')
    ax.hist(FST_values, bins=bins, alpha=0.8, cumulative=cumulative, label='all shared')

    if cumulative == False:
        for loci_group in ['correlated', 'uncorrelated']:
            loci = sorted_loci[loci_group]
            group_FST = [strain_differences[locus]['F_ST'] for locus in loci if locus in strain_differences]
            ax.hist(group_FST, bins=50, alpha=0.6, label=f'{loci_group} sweeps')
    ax.legend(fontsize=8)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()


def plot_osbp_genome_locations(sorted_loci, ref_cds_dict, ref_genome='osbp', max_tag_number=2942, randomize=False, savefig=None):
    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'genome location')
    ax.set_xlim(0, max_tag_number + 1)
    ax.set_ylim(-0.5, 0.5)
    ax.get_yaxis().set_visible(False)

    #loci_categories = ['correlated', 'uncorrelated', 'typical', 'high_FST']
    loci_categories = ['correlated', 'high_FST', 'uncorrelated']

    cmap = plt.get_cmap('tab10')
    delta = 0.05
    offset = 0.025
    ax.axhline(0, lw=2, c='k')
    for i, locus_type in enumerate(loci_categories):
        locus_tags = [map_ssog_id(ssog_id, ref_cds_dict) for ssog_id in sorted_loci[locus_type]]
        if ref_genome == 'osa':
            syn_homolog_map = SynHomologMap(build_maps=True)
            locus_tags = [syn_homolog_map.get_ortholog(tag) for tag in locus_tags if tag is not None]
            locus_tags = [tag if tag != 'none' else None for tag in locus_tags]

        if randomize:
            tag_numbers = np.random.choice(max_tag_number, size=len(locus_tags))
        else:
            tag_numbers = [int(tag.split('_')[-1]) for tag in locus_tags if tag is not None]
        print(locus_type)
        print(locus_tags)
        print(tag_numbers, len(tag_numbers), len(sorted_loci[locus_type]))
        print('\n')
        if i < 2:
            y0 = i * delta
            ax.scatter(tag_numbers, offset + y0 * np.ones(len(tag_numbers)), marker='v', s=20, ec=cmap(i), fc=cmap(i), label=locus_type)
        else:
            y0 = -delta
            ax.scatter(tag_numbers, offset + y0 * np.ones(len(tag_numbers)), marker='^', s=20, ec=cmap(i), fc=cmap(i), label=locus_type)

    ax.legend()

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--data_dir', default='../results/single-cell/sscs_pangenome/', help='Directory with output of previous clustering steps.')
    parser.add_argument('-O', '--output_dir', help='Directory in which results are saved.')
    parser.add_argument('-F', '--figures_dir', help='Directory in which figures are saved.')
    parser.add_argument('-b', '--bprime_strain_differences', help='Path to file with B\' strain difference metrics.')
    parser.add_argument('-c', '--candidates_file', help='File with candidate hitchhiking sites.')
    parser.add_argument('-g', '--orthogroup_table', help='File with orthogroup table.')
    parser.add_argument('-o', '--output_file', default=None, help='File path where to store r^2 results.')
    parser.add_argument('-r', '--rsq_file', default=None, help='File with r^2 results.')
    parser.add_argument('-s', '--sag_species_file', default=None, help='File path for updating species SAG sorting.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    parser.add_argument('-t', '--test_mode', action='store_true', help='Run in test mode.')
    parser.add_argument('--osbp_genbank_file', default='../data/reference_genomes/CP000240.genbank', help='Path to OS-B\' genbank file.')
    parser.add_argument('--analyze_within_gene_linkage', action='store_true', help='Make different within gene linkage plots.')
    parser.add_argument('--analyze_sweep_linkage', action='store_true', help='Make plots with linkage between sweeps.')
    parser.add_argument('--analyze_allele_haplotypes', action='store_true', help='Make allele haplotype plots.')
    parser.add_argument('--analyze_strain_divergence', action='store_true', help='Make strain genomewide divergence plots.')
    parser.add_argument('--calculate_within_gene_linkage', action='store_true', help='Calculate r^2 matrix for hitchhiking genes.')
    parser.add_argument('--test_gene_pair_linkage', action='store_true', help='Make test plots for gene pair linkage.')
    parser.add_argument('--random_seed', type=int, default=12345)
    args = parser.parse_args()

    metadata = MetadataMap()
    pangenome_map = PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    hitchhiking_candidates = pickle.load(open(args.candidates_file, 'rb'))
    print(f'{len(hitchhiking_candidates)} candidate hitchhiking loci')
    for locus in hitchhiking_candidates:
        hh_segments = hitchhiking_candidates[locus]['segments']
        print(f'{locus} segments:\n\tlengths: {hh_segments[:, 1] - hh_segments[:, 0]}\n\tstart coordinates: {hh_segments[:, 0]}\n')
    print('\n')

    raw_osbp_cds_dict = utils.read_genbank_cds(args.osbp_genbank_file)
    osbp_cds_dict = add_gene_id_map(raw_osbp_cds_dict)

    if args.test_mode:
        candidate_loci = ['psaI', 'psaL-1', 'psaF']
    else:
        candidate_loci = list(hitchhiking_candidates.keys())

    locus_alns = read_locus_alignments(candidate_loci, pangenome_map, alignments_dir=f'{args.data_dir}_aln_results/')
    allele_haplotype_df = contruct_hitchhiking_allele_haplotypes(hitchhiking_candidates, locus_alns)
    print(allele_haplotype_df)
    haplotype_sag_ids = list(allele_haplotype_df.index)
    species_sorted_sag_ids = metadata.sort_sags(haplotype_sag_ids, by='species')
    print(species_sorted_sag_ids)
    allele_haplotype_df = allele_haplotype_df.reindex(index=species_sorted_sag_ids['Bp'])
    print(allele_haplotype_df)
    print('\n\n')
    
    hitchhiking_segments = [hitchhiking_candidates[locus]['segments'] for locus in candidate_loci]
    plot_species_composition(locus_alns, metadata, savefig=f'{args.figures_dir}hitchhiking_clusters_species_composition.pdf')


    #######################################################################
    # Linkage between sweeping alleles
    #######################################################################

    if args.analyze_sweep_linkage or args.analyze_allele_haplotypes:
        if args.rsq_file:
            rsq_df = pickle.load(open(args.rsq_file, 'rb'))
        else:
            rsq_df = calculate_hitchhiking_allele_linkage(locus_alns, hitchhiking_segments)

        if args.output_file:
            pickle.dump(rsq_df, open(args.output_file, 'wb'))

        print(rsq_df)
        rsq_clean = clean_rsq_matrix(rsq_df)
        plot_rsq_map(rsq_df, savefig=f'{args.figures_dir}core_site_rsq_matrix.pdf')
        Z_loci = plot_rsq_map(rsq_clean, cluster=True, savefig=f'{args.figures_dir}core_site_rsq_matrix_clustered.pdf')

        clusters = hclust.fcluster(Z_loci, 3, criterion='distance')
        cluster_idx, idx_count = utils.sorted_unique(clusters)
        loci = np.array(list(rsq_clean.index))
        correlated_loci = [site_tag.split('_xc')[0] for site_tag in loci[clusters == cluster_idx[1]]]
        print(correlated_loci, len(correlated_loci))

        plot_rsq_vs_osbp_distance(rsq_df, correlated_loci, osbp_cds_dict, savefig=f'{args.figures_dir}core_site_rsq_vs_osbp_genome_distance.pdf')


    #######################################################################
    # Candidate strain differentiation loci haplotypes
    #######################################################################

    if args.analyze_allele_haplotypes:
        correlated_haplotype_df = allele_haplotype_df.loc[np.sum(allele_haplotype_df[correlated_loci] > 0, axis=1) > 0, correlated_loci]
        plot_haplotype_map(correlated_haplotype_df, min_presence=0.0, cluster=True, savefig=f'{args.figures_dir}correlated_loci_haplotypes_clustered_all.pdf')
        strain_idx, Z_strains = plot_haplotype_map(correlated_haplotype_df, min_presence=0.5, cluster=True, savefig=f'{args.figures_dir}correlated_loci_haplotypes_clustered_presence-0.5.pdf')
        strain_clusters = hclust.fcluster(Z_strains, 2, criterion='maxclust')
        strain_dict = {'Bp1':strain_idx[strain_clusters == 1], 'Bp2':strain_idx[strain_clusters == 2]}

        # Update SAG species sorted IDs
        if args.sag_species_file:
            sag_species = pickle.load(open(args.sag_species_file, 'rb'))
            sag_species['hitchhiking_alleles'] = strain_dict
            pickle.dump(sag_species, open(args.sag_species_file, 'wb'))

            # Test strain sorting
            sag_ids = list(correlated_haplotype_df.index)
            synbp_strains_sag_ids = metadata.sort_sags(sag_ids, by='Bp_strains')
            for strain in synbp_strains_sag_ids:
                print(strain, len(synbp_strains_sag_ids[strain]))
                print(synbp_strains_sag_ids[strain])
                print('\n')
            print('\n\n')

        # Check strain-sample correlates
        print('\n\n')
        sag_ids = list(correlated_haplotype_df.index)
        synbp_strains_sag_ids = metadata.sort_sags(sag_ids, by='Bp_strains')
        for sorting in ['location', 'sample', 'temperature']:
            for strain in strain_dict:
                sorted_strains = metadata.sort_sags(synbp_strains_sag_ids[strain], by=sorting)
                print(f'{strain} cells sorted by {sorting}:')
                output_str = [f'{len(sorted_strains[key])} {key}' for key in sorted_strains]
                print(f'\t{"; ".join(output_str)}')
            print('\n')


    #######################################################################
    # Linkage within strain differentiation genes
    #######################################################################

    aln_dict = construct_loci_alignments_dict(locus_alns)
    if args.calculate_within_gene_linkage:
        sorted_loci = {'correlated':correlated_loci, 'uncorrelated':[locus for locus in aln_dict if locus not in correlated_loci]}
        within_gene_rsq = {}
        for loci_type in sorted_loci:
            loci = sorted_loci[loci_type]
            rsq_list = []
            for locus in loci:
                print(f'Calculating r^2 at {locus}')
                aln = aln_dict[locus]
                trimmed_aln = align_utils.trim_alignment_gaps(aln, start_gap_perc=0.05)
                rsq_list.append(ld.calculate_rsquared(trimmed_aln))
            within_gene_rsq[loci_type] = rsq_list
            print('\n')
        pickle.dump((sorted_loci, within_gene_rsq), open(f'{args.output_dir}hitchhiking_genes_rsq.dat', 'wb'))
    else:
        sorted_loci, within_gene_rsq = pickle.load(open(f'{args.output_dir}hitchhiking_genes_rsq.dat', 'rb'))


    if args.bprime_strain_differences is not None:
        #strain_differences = pickle.load(open(args.bprime_strain_differences, 'rb'))
        FST_dict = pickle.load(open(args.bprime_strain_differences, 'rb'))
        strain_differences = {}
        for locus in FST_dict:
            #strain_differences[locus] = {'F_ST':FST_dict[locus]}
            strain_differences[locus] = {'F_ST':FST_dict[locus]['F_ST']}

        synbp_strain_ssog_ids = sorted([ssog_id for ssog_id in strain_differences])
        sorted_loci = update_sorted_loci(sorted_loci, osbp_cds_dict, synbp_strain_ssog_ids)

        np.random.seed(args.random_seed)
        control_loci = plot_strain_FST(strain_differences, sorted_loci, return_control=True, savefig=f'{args.figures_dir}synbp_strain_FST.pdf')
        plot_FST_distribution(strain_differences, sorted_loci, bins=100, savefig=f'{args.figures_dir}synbp_strain_FST_distribution_bins100.pdf')
        plot_FST_distribution(strain_differences, sorted_loci, cumulative=True, savefig=f'{args.figures_dir}synbp_strain_FST_distribution_cumulative.pdf')

        sorted_loci['high_FST'] = [locus for locus in strain_differences if strain_differences[locus]['F_ST'] > 0.5 and locus in synbp_strain_ssog_ids]
        sorted_loci['low_FST'] = [locus for locus in strain_differences if strain_differences[locus]['F_ST'] < 0.25 and locus in synbp_strain_ssog_ids]
        pickle.dump(sorted_loci, open(f'{args.output_dir}synbp_strains_sorted_loci.dat', 'wb'))

        print('\n\n')
        for locus_type in sorted_loci:
            print(locus_type)
            print(sorted_loci[locus_type], len(sorted_loci[locus_type]))
            print('\n')
        print('\n')

        plot_osbp_genome_locations(sorted_loci, osbp_cds_dict, savefig=f'{args.figures_dir}osbp_strain_loci_locations.pdf')
        plot_osbp_genome_locations(sorted_loci, osbp_cds_dict, ref_genome='osa', savefig=f'{args.figures_dir}osa_strain_loci_locations.pdf')
        plot_osbp_genome_locations(sorted_loci, osbp_cds_dict, randomize=True, savefig=f'{args.figures_dir}randomized_strain_loci_locations.pdf')


        if args.analyze_within_gene_linkage:
            print('Correlated loci...')
            rsq_correlated_loci = calculate_rsq_curves(strain_differences, sorted_loci['correlated'], group='all')
            plot_rsq_curves(rsq_correlated_loci, savefig=f'{args.figures_dir}linkage_measures/synbp_strain_correlated_loci_rsq_all.pdf')
            print('\n\n')
            rsq_correlated_loci = calculate_rsq_curves(strain_differences, sorted_loci['correlated'], group='strain1')
            plot_rsq_curves(rsq_correlated_loci, savefig=f'{args.figures_dir}linkage_measures/synbp_strain_correlated_loci_rsq_strain1.pdf')
            print('\n\n')
            rsq_correlated_loci = calculate_rsq_curves(strain_differences, sorted_loci['correlated'], group='strain2')
            plot_rsq_curves(rsq_correlated_loci, savefig=f'{args.figures_dir}linkage_measures/synbp_strain_correlated_loci_rsq_strain2.pdf')
            print('\n\n\n')

            rsq_correlated_neighbor_loci = calculate_rsq_curves(strain_differences, sorted_loci['correlated_neighbors'], group='all')
            plot_rsq_curves(rsq_correlated_neighbor_loci, savefig=f'{args.figures_dir}linkage_measures/synbp_strain_correlated_neighbor_loci_rsq_all.pdf')
            rsq_correlated_neighbor_loci = calculate_rsq_curves(strain_differences, sorted_loci['correlated_neighbors'], group='strain1')
            plot_rsq_curves(rsq_correlated_neighbor_loci, savefig=f'{args.figures_dir}linkage_measures/synbp_strain_correlated_neighbor_loci_rsq_strain1.pdf')
            rsq_correlated_neighbor_loci = calculate_rsq_curves(strain_differences, sorted_loci['correlated_neighbors'], group='strain2')
            plot_rsq_curves(rsq_correlated_neighbor_loci, savefig=f'{args.figures_dir}linkage_measures/synbp_strain_correlated_neighbor_loci_rsq_strain2.pdf')

            rsq_uncorrelated_loci = calculate_rsq_curves(strain_differences, sorted_loci['uncorrelated'], group='all')
            plot_rsq_curves(rsq_uncorrelated_loci, savefig=f'{args.figures_dir}linkage_measures/synbp_strain_uncorrelated_loci_rsq_all.pdf')
            rsq_uncorrelated_loci = calculate_rsq_curves(strain_differences, sorted_loci['uncorrelated'], group='strain1')
            plot_rsq_curves(rsq_uncorrelated_loci, savefig=f'{args.figures_dir}linkage_measures/synbp_strain_uncorrelated_loci_rsq_strain1.pdf')
            rsq_uncorrelated_loci = calculate_rsq_curves(strain_differences, sorted_loci['uncorrelated'], group='strain2')
            plot_rsq_curves(rsq_uncorrelated_loci, savefig=f'{args.figures_dir}linkage_measures/synbp_strain_uncorrelated_loci_rsq_strain2.pdf')

            rsq_control_loci = calculate_rsq_curves(strain_differences, control_loci, group='all')
            plot_rsq_curves(rsq_control_loci, savefig=f'{args.figures_dir}linkage_measures/synbp_strain_control_loci_rsq_all.pdf')
            rsq_control_loci = calculate_rsq_curves(strain_differences, control_loci, group='strain1')
            plot_rsq_curves(rsq_control_loci, savefig=f'{args.figures_dir}linkage_measures/synbp_strain_control_loci_rsq_strain1.pdf')
            rsq_control_loci = calculate_rsq_curves(strain_differences, control_loci, group='strain2')
            plot_rsq_curves(rsq_control_loci, savefig=f'{args.figures_dir}linkage_measures/synbp_strain_control_loci_rsq_strain2.pdf')


        # Gene pair linkage metrics tests
        if args.test_gene_pair_linkage:
            test_pairs = np.random.choice(sorted_loci['low_FST'], size=(10, 2))
            for og_id1, og_id2 in test_pairs:
                f_aln1 = f'{args.data_dir}_aln_results/{og_id1}_aln.fna'
                aln1 = seq_utils.read_alignment_and_map_sag_ids(f_aln1, pangenome_map)
                f_aln2 = f'{args.data_dir}_aln_results/{og_id2}_aln.fna'
                aln2 = seq_utils.read_alignment_and_map_sag_ids(f_aln2, pangenome_map)

                #aln1 = seq_utils.read_alignment_and_map_sag_ids(og_id1, f'{args.data_dir}_aln_results/', pangenome_map)
                #aln2 = seq_utils.read_alignment_and_map_sag_ids(og_id2, f'{args.data_dir}_aln_results/', pangenome_map)

                if (aln1 is not None) and (aln2 is not None):
                    print(og_id1, og_id2)
                    print(aln1)
                    print(aln2)

                    start_time = time.time()
                    rsq, boundary_idx = align_utils.calculate_gene_pair_snp_linkage([aln1, aln2])
                    print(utils.timeit(start_time))
                    plot_gene_pair_linkage_matrix(rsq, boundary_idx, savefig=f'{args.figures_dir}gene_pair_plots/{og_id1}-{og_id2}_rsq_matrix.pdf')
                    rsq_between = rsq[boundary_idx:, :boundary_idx + 1].flatten()
                    plot_distribution(rsq_between, xlabel=r'$r^2$', xlim=(1E-2, 1.05), histyscale='log', bins=100, cumlxlim=(1E-4, 2), save_fig=f'{args.figures_dir}gene_pair_plots/rsq_distributions/{og_id1}-{og_id2}_rsq_distribution.pdf')
                    print(f'r^2 = {np.mean(rsq_between)}')

                print('\n')


    # Calculate strain genomewide divergence
    if args.analyze_strain_divergence:
        divergence_files = sorted(glob.glob(f'{args.data_dir}_aln_results/sscs_orthogroups_*.dat'))
        for f_divergence in divergence_files:
            print(f_divergence)
            gene_divergence = pickle.load(open(f_divergence, 'rb'))
            print(gene_divergence)
            break




    '''

    rsq_dict = {}
    rsq_list = []

    # Make individual linkage matrices for gene pairs
    for j in range(len(candidate_loci)):
        for i in range(j):
            locus_i, aln_i = locus_alns[i]
            locus_j, aln_j = locus_alns[j]
            rsq, boundary_idx = align_utils.calculate_gene_pair_snp_linkage([aln_i, aln_j])
            plot_gene_pair_linkage_matrix(rsq, boundary_idx, savefig=f'{args.figures_dir}gene_pair_plots/{locus_i}-{locus_j}_rsq_matrix.pdf')
            rsq_dict[(locus_i, locus_j)] = (rsq, boundary_idx)
        locus_j, aln_j = locus_alns[j]
        rsq_list.append(ld.calculate_rsquared(aln_j))
    plot_gene_pair_average_linkage_comparison(rsq_dict, savefig=f'{args.figures_dir}gene_pair_plots/psa_genes_average_rsq_comparisons.pdf')
    rsq_matrix_avg = plt_ld.average_over_loci(rsq_list)
    rsq_avg = plt_ld.average_over_distance(rsq_matrix_avg)
    plot_linkage_curve(rsq_avg, savefig=f'{args.figures_dir}psa_genes_rsq_vs_x_averaged.pdf')


    if args.verbose:
        print('\n\n')
        for locus, aln in locus_alns:
            print(locus)
            print(aln)
            print('\n')
        print('\n\n')

        print(allele_haplotype_df)
        print(correlated_loci)
        print(allele_haplotype_df[correlated_loci])

    '''
