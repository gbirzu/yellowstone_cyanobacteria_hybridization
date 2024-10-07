import argparse
import os
import numpy as np
import pandas as pd
import pickle
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
import matplotlib.pyplot as plt
import plot_linkage_disequilibrium as plt_ld
from pangenome_utils import PangenomeMap
from metadata_map import MetadataMap
from analyze_hitchhiking_candidates import add_gene_id_map, map_ssog_id
from plot_utils import *

fig_count = 1

def get_genomic_trenches_diversity_stats(og_stats, args, pi_cutoff=0.05, length_cutoff=200, min_num_seqs=10, make_figures=True):
    index_filter = (og_stats['A_seqs'] >= min_num_seqs) & (og_stats['Bp_seqs'] >= min_num_seqs) & (og_stats['avg_length'] > length_cutoff)
    filtered_stats_table = og_stats.loc[index_filter, :]
    genomic_trenches_stats = filtered_stats_table.loc[(filtered_stats_table['pi_ABp'] < pi_cutoff) & (filtered_stats_table['num_duplicates'] == 0), :] # Exclude OGs with duplicates

    if make_figures:
        # Plot pi_ABp distributions
        plot_pdf(filtered_stats_table['pi_ABp'].values, bins=50, xlabel=r"nucleotide diversity, $\pi$", alpha=1.0, savefig=f'{args.figures_dir}synabp_mean_pi_distribution.pdf')
        ax = plot_pdf(filtered_stats_table['pi_ABp'].values, bins=50, xlabel=r"nucleotide diversity, $\pi$", alpha=1.0)
        ax.axvline(pi_cutoff, c='r', lw=2)
        plt.savefig(f'{args.figures_dir}synabp_mean_pi_distribution_gt_cutoff.pdf')

        plot_pdf(filtered_stats_table['pi_A'].dropna().values, bins=50, xlabel=r"nucleotide diversity, $\pi$", alpha=1.0, savefig=f'{args.figures_dir}syna_pi_distribution.pdf')
        plot_pdf(filtered_stats_table['pi_Bp'].dropna().values, bins=50, xlabel=r"nucleotide diversity, $\pi$", alpha=1.0, savefig=f'{args.figures_dir}synbp_pi_distribution.pdf')

        ax = plot_pdf(filtered_stats_table['pi_ABp'].values, bins=50, xlabel=r"nucleotide diversity, $\pi$", color='tab:purple', alpha=0.5, data_label="A vs B'")
        ax.hist(filtered_stats_table['pi_A'].dropna().values, bins=50, color='tab:orange', alpha=0.5, label='A')
        ax.hist(filtered_stats_table['pi_Bp'].dropna().values, bins=50, color='tab:blue', alpha=0.5, label="B'")
        ax.legend()
        plt.savefig(f'{args.figures_dir}synabp_mean_pi_distribution_overlapped.pdf')

    return genomic_trenches_stats

def choose_Sbar_genomic_trench_loci(og_stats, args, make_plots=True, Sbarstar=0.1, Sbar_ratio=1.0):
    # Calculate derived statistics
    og_stats['Sbar'] = og_stats['S'] - og_stats['singletons']
    og_stats['Sbar*'] = og_stats['Sbar'] / og_stats['trimmed_aln_length']
    og_stats['Sbar_A'] = og_stats['S_A'] - og_stats['A_singletons']
    og_stats['Sbar*_A'] = og_stats['Sbar_A'] / og_stats['trimmed_aln_length']
    og_stats['Sbar_Bp'] = og_stats['S_Bp'] - og_stats['Bp_singletons']
    og_stats['Sbar*_Bp'] = og_stats['Sbar_Bp'] / og_stats['trimmed_aln_length']
    og_stats['Sbar_ratio'] = (og_stats['Sbar_A'] + og_stats['Sbar_Bp']) / og_stats['Sbar']
    print(og_stats)

    # Drop single-species OGs
    filtered_og_stats = og_stats.loc[(og_stats['A_seqs'] >= args.min_num_seqs) & (og_stats['Bp_seqs'] >= args.min_num_seqs), :]
    print(filtered_og_stats)

    if make_plots:
        # Make figures with diversity distributions
        plot_distribution(filtered_og_stats['S'].values, xlabel='number of SNPs, $S$', save_fig=f'{args.figures_dir}S_distribution.pdf')
        plot_distribution(filtered_og_stats['Sbar'].values, xlabel='number of SNPs (no singletons), $\overline{S}$', save_fig=f'{args.figures_dir}Sbar_distribution.pdf')
        plot_distribution(filtered_og_stats['Sbar*'].values, bins=50, xlabel='number of SNPs (no singletons) per bp, $\overline{S}^*$', save_fig=f'{args.figures_dir}Sbarstar_distribution.pdf')
        plot_distribution(np.log2(filtered_og_stats.loc[filtered_og_stats['Sbar_ratio'] > 0, 'Sbar_ratio'].values), bins=30, xlabel=r"$\log_2 \left[ (\overline{S}_A + \overline{S}_{B'}) / \overline{S} \right]$", save_fig=f'{args.figures_dir}Sbar_ratio_distribution.pdf')
        plot_distribution(np.log10(filtered_og_stats.loc[filtered_og_stats['H'] > 0, 'H'].values), bins=30, xlabel=r"$\log_{10} \left[ H \right]$", cumlxscale='linear', save_fig=f'{args.figures_dir}H_distribution.pdf')
        plot_distribution(np.log10(filtered_og_stats.loc[filtered_og_stats['Sbar*'] > 0, 'Sbar*'].values), bins=30, xlabel=r"$\log_{10} \left[ \overline{S}^* \right]$", cumlxscale='linear', save_fig=f'{args.figures_dir}Sbarstar_distribution_logx.pdf')

    
    utils.print_break()
    key_stats_columns = ['trimmed_aln_length', 'A_seqs', 'Bp_seqs', 'H', 'Sbar*', 'Sbar*_A', 'Sbar*_Bp', 'Sbar_ratio']
    print(filtered_og_stats.loc[filtered_og_stats['Sbar*'] < Sbarstar, key_stats_columns])

    utils.print_break()
    refined_filter = (filtered_og_stats['Sbar*'] < Sbarstar) & (filtered_og_stats['Sbar_ratio'] > Sbar_ratio)
    return filtered_og_stats.loc[refined_filter, :]


def choose_control_loci(pangenome_map, gt_loci, args, gt_ids_type='sog'):
    parent_og_species_counts = calculate_parent_og_species_counts(pangenome_map, args)
    print(parent_og_species_counts)

    # Find parent OGs present in both species
    presence_filter = (parent_og_species_counts['A'] >= args.min_num_seqs) & (parent_og_species_counts['Bp'] >= args.min_num_seqs)
    shared_parent_ogs = list(parent_og_species_counts.loc[presence_filter, :].index)

    # Exclude test loci from possible control list
    if gt_ids_type == 'sog':
        og_table = pangenome_map.og_table
        test_og_ids = get_og_ids(og_table.loc[gt_loci, :])
    else:
        test_og_ids = gt_loci
    nontest_og_ids = [oid for oid in shared_parent_ogs if oid not in test_og_ids]

    return np.random.choice(nontest_og_ids, size=len(gt_loci))


def calculate_parent_og_species_counts(pangenome_map, args):
    og_table = pangenome_map.og_table
    og_ids = get_og_ids(og_table)
    og_table['parent1_og_id'] = og_table['parent_og_id'].str.split('-').str[0]
    species_counts_df = pangenome_map.og_species_counts_table
    parent_og_species_counts = pd.DataFrame(index=og_ids, columns=['A', 'Bp', 'C'])
    
    for pid in og_ids:
        parent_og_subtable = og_table.loc[og_table['parent1_og_id'] == pid, :]
        parent_og_species_counts.loc[pid, :] = species_counts_df.loc[list(parent_og_subtable.index), :].sum(axis=0).values
    return parent_og_species_counts

def get_og_ids(og_table):
    raw_og_ids = og_table['parent_og_id'].values
    parent_og_ids = [oid.split('-')[0] for oid in raw_og_ids]
    return np.unique(parent_og_ids)


def calculate_og_stats(og_ids, og_table, synabp_sag_ids, args, index_type='parent_og'):
    og_stats = pd.DataFrame(index=og_ids, columns=['Sbar*', 'pi', 'K'])

    if index_type == 'parent_og':
        og_table['parent1_og_id'] = og_table['parent_og_id'].str.split('-').str[0]

    for oid in og_ids:
        f_aln = f'{args.alignment_dir}{oid}_aln.fna'
        if os.path.exists(f_aln):
            aln = seq_utils.read_alignment(f_aln)
            filtered_aln, x_filtered = align_utils.trim_alignment_and_remove_gaps(aln, max_edge_gaps=0.0)

            if index_type == 'parent_og':
                og_subtable = og_table.loc[og_table['parent1_og_id'] == oid, np.concatenate(synabp_sag_ids)]
            else:
                og_subtable = og_table.loc[[oid], np.concatenate(synabp_sag_ids)]

            # Get A and B' gene IDs
            synabp_gene_ids = []
            for species_sag_ids in synabp_sag_ids:
                synabp_gene_ids.append(np.concatenate([pg_utils.read_gene_ids(subtable_row[species_sag_ids], drop_none=True) for i, subtable_row in og_subtable.iterrows()]))
            synabp_aln = align_utils.get_subsample_alignment(filtered_aln, np.concatenate(synabp_gene_ids))
            
            # Update A and B' gene IDs present in alignment
            aln_synab_gene_ids = []
            for gene_ids in synabp_gene_ids:
                aln_synab_gene_ids.append([rec.id for rec in synabp_aln if rec.id in gene_ids])
                print(gene_ids, len(gene_ids))
                print(aln_synab_gene_ids[-1], len(aln_synab_gene_ids[-1]))
                print('\n')

            snp_frequencies = align_utils.calculate_snp_frequencies(synabp_aln)
            f_singleton = 1.001 / len(filtered_aln) # add epsilon to be sure
            Sbar = np.sum(snp_frequencies > f_singleton)
            og_stats.loc[oid, 'Sbar*'] = Sbar / synabp_aln.get_alignment_length()

            pdist_df = align_utils.calculate_fast_pairwise_divergence(synabp_aln)
            #og_stats.loc[oid, 'pi'] = np.mean(pdist_df.loc[synabp_gene_ids[0], synabp_gene_ids[1]].values)
            og_stats.loc[oid, 'pi'] = np.mean(pdist_df.loc[aln_synab_gene_ids[0], aln_synab_gene_ids[1]].values)

            aln_snps, x_snps = seq_utils.get_snps(synabp_aln, return_x=True)
            alleles, allele_counts = align_utils.group_alignment_alleles(aln_snps)
            og_stats.loc[oid, 'K'] = len(allele_counts)
            print('\n\n')
    return og_stats


#def make_diversity_comparison_figures(genomic_trenches_ogs, control_loci_ogs, og_stats, og_table, synabp_sag_ids, args):
def make_diversity_comparison_figures(genomic_trenches_stats, control_loci_stats, og_table, args):
    ax = plot_pdf(genomic_trenches_stats['Sbar*'].values, bins=15, xlabel=r'non-singleton SNPs per bp, $\overline{S}^*$', color='tab:purple', alpha=0.5, data_label='genomic trenches')
    ax.hist(control_loci_stats['Sbar*'].values, bins=15, alpha=0.5, color='gray', label='typical loci')
    plt.savefig(f'{args.figures_dir}Sbarstar_distribution_GT_vs_TL.pdf')

    ax = plot_pdf(genomic_trenches_stats['pi'].values, bins=15, xlabel=r'nucleotide diversity, $\pi$', color='tab:purple', alpha=0.5, data_label='genomic trenches')
    ax.hist(control_loci_stats['pi'].values, bins=15, alpha=0.5, color='gray', label='typical loci')
    plt.savefig(f'{args.figures_dir}pi_distribution_GT_vs_TL.pdf')

    ax = plot_pdf(genomic_trenches_stats['K'].values, bins=15, xlabel=r'number of alleles, $K$', color='tab:purple', alpha=0.5, data_label='genomic trenches')
    ax.hist(control_loci_stats['K'].values, bins=15, alpha=0.5, color='gray', label='typical loci')
    plt.savefig(f'{args.figures_dir}K_distribution_GT_vs_TL.pdf')

    
    raw_osbp_cds_dict = utils.read_genbank_cds('../data/reference_genomes/CP000240.genbank')
    osbp_cds_dict = add_gene_id_map(raw_osbp_cds_dict)
    sorted_loci = {'genomic_trenches':list(genomic_trenches_stats.index)}

    # Add control OG IDs
    og_table['parent1_og_id'] = og_table['parent_og_id'].str.split('-').str[0]
    control_og_ids = []
    for pid in control_loci_stats.index:
        for oid in og_table.loc[og_table['parent1_og_id'] == pid, :].index:
            if 'CYB' in oid:
                control_og_ids.append(oid)
                break
            elif 'YSG' not in oid:
                control_og_ids.append(oid)
                break
        print(pid, control_og_ids[-1])
    sorted_loci['control'] = control_og_ids
    print(sorted_loci)

    plot_osbp_genome_locations(sorted_loci, osbp_cds_dict, savefig=f'{args.figures_dir}GT_vs_TL_osbp_genome_location.pdf')
    plot_osbp_genome_locations(sorted_loci, osbp_cds_dict, ref_genome='osa', savefig=f'{args.figures_dir}GT_vs_TL_osa_genome_location.pdf')


def plot_loci_genome_location(genomic_trenches_ogs, control_loci_ogs, og_table, args, ref_genbank_file='../data/reference_genomes/CP000239.genbank', savefig=None):
    # Add OS-A and OS-B' mapped ids
    og_table['parent1_og_id'] = og_table['parent_og_id'].str.split('-').str[0]
    ref_mapped_ids = [[], []]
    for i, og_list in enumerate([genomic_trenches_ogs, control_loci_ogs]):
        for pid in og_list:
            for oid in og_table.loc[og_table['parent1_og_id'] == pid, :].index:
                if 'CYA' in oid:
                    ref_mapped_ids[i].append(oid)
                    break
                elif 'YSG' not in oid:
                    ref_mapped_ids[i].append(oid)
                    break


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


def plot_osbp_genome_locations(sorted_loci, ref_cds_dict, ref_genome='osbp', max_tag_number=2942, randomize=False, savefig=None):
    fig = plt.figure(figsize=(double_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'genome location')
    ax.set_xlim(0, max_tag_number + 1)
    ax.set_ylim(-0.5, 0.5)
    ax.get_yaxis().set_visible(False)

    #loci_categories = ['correlated', 'uncorrelated', 'typical', 'high_FST']
    loci_categories = ['genomic_trenches', 'control']

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
        if i < 1:
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

def make_linkage_disequilibrium_figures(genomic_trenches_ogs, control_loci_ogs, og_table, synabp_sag_ids, args, species_genomewide_rsq=None):
    #rsq_list = calculate_linkage_matrices(genomic_trenches_ogs, og_table, synabp_sag_ids, args, index_type='og')
    rsq_list = calculate_linkage_matrices(genomic_trenches_ogs, og_table, synabp_sag_ids, args, index_type='parent_og')
    rsq_avg = plt_ld.average_over_loci(rsq_list)
    x_gt, r2_gt, _ = plt_ld.average_over_distance(rsq_avg, denom=1.5, coarse_grain=True, num_cg_points=20)

    rsq_list = calculate_linkage_matrices(control_loci_ogs, og_table, synabp_sag_ids, args, index_type='parent_og')
    rsq_avg = plt_ld.average_over_loci(rsq_list)
    x_ctrl, r2_ctrl, _ = plt_ld.average_over_distance(rsq_avg, denom=1.5, coarse_grain=True, num_cg_points=20)

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('separation, x')
    ax.set_xscale('log')
    ax.set_xlim(4E-1, 5E3)
    ax.set_ylabel(r'linkage disequilibrium, $r^2$')
    ax.set_yscale('log')
    ax.set_ylim(1E-4, 1.5E0)

    ax.plot(x_gt, r2_gt, '-o', lw=1, ms=3, alpha=1.0, c='tab:purple', label='genomic trenches')
    ax.plot(x_ctrl, r2_ctrl, '-o', lw=1, ms=3, alpha=1.0, c='k', label='typical loci')

    if species_genomewide_rsq is not None:
        # Add species genomewide averages
        species = ['A', "B'"]
        colors = ['tab:orange', 'tab:blue']
        for i, rsq in enumerate(species_rsq_avg):
            x_cg, r2_cg, _ = plt_ld.average_over_distance(rsq, denom=2.0, coarse_grain=True, num_cg_points=20)
            ax.plot(x_cg, r2_cg, '-o', lw=1, ms=3, alpha=1.0, c=colors[i], label=f'genomewide {species[i]}')

    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}genomic_trenches_linkage_comparisons.pdf')


def calculate_linkage_matrices(og_ids, og_table, synabp_sag_ids, args, index_type='parent_og'):
    if index_type == 'parent_og':
        og_table['parent1_og_id'] = og_table['parent_og_id'].str.split('-').str[0]

    rsq_matrix_list = []
    for oid in og_ids:
        f_aln = f'{args.alignment_dir}{oid}_aln.fna'
        if os.path.exists(f_aln):
            aln = seq_utils.read_alignment(f_aln)
            filtered_aln, x_filtered = align_utils.trim_alignment_and_remove_gaps(aln, max_edge_gaps=0.0)

            if index_type == 'parent_og':
                og_subtable = og_table.loc[og_table['parent1_og_id'] == oid, np.concatenate(synabp_sag_ids)]
            else:
                og_subtable = og_table.loc[[oid], np.concatenate(synabp_sag_ids)]

            # Get A and B' gene IDs
            synabp_gene_ids = []
            for species_sag_ids in synabp_sag_ids:
                synabp_gene_ids.append(np.concatenate([pg_utils.read_gene_ids(subtable_row[species_sag_ids], drop_none=True) for i, subtable_row in og_subtable.iterrows()]))
            synabp_aln = align_utils.get_subsample_alignment(filtered_aln, np.concatenate(synabp_gene_ids))

            Dsq, denom = align_utils.calculate_ld_matrices_vectorized(synabp_aln, reference=0, unbiased=False)
            rsq = Dsq / (denom + (denom == 0))
            rsq_matrix_list.append(rsq)

            # Update A and B' gene IDs present in alignment
            aln_synab_gene_ids = []
            for gene_ids in synabp_gene_ids:
                aln_synab_gene_ids.append([rec.id for rec in synabp_aln if rec.id in gene_ids])

    return rsq_matrix_list


def plot_nucleotide_diversity_comparison(og_stats, genomic_trenches_ogs, control_loci_ogs, args, length_cutoff=200):
    index_filter = (og_stats['A_seqs'] >= args.min_num_seqs) & (og_stats['Bp_seqs'] >= args.min_num_seqs) & (og_stats['avg_length'] > length_cutoff)
    filtered_stats_table = og_stats.loc[index_filter, :]

    x = np.arange(1, 5)
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlim(0.2, 4.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["A", "B'", "genomic\ntrenches", "typical\nloci"], fontsize=10)
    ax.set_ylabel(r'nucleotide diversity, $\pi$')

    #pi_values = [og_stats.loc[genomic_trenches_ogs, 'pi_ABp'].values, og_stats.loc[control_loci_ogs, 'pi_ABp'].values]
    pi_values = []
    pi_values.append(filtered_stats_table['pi_A'].dropna().values)
    pi_values.append(filtered_stats_table['pi_Bp'].dropna().values)
    pi_values.append(og_stats.loc[genomic_trenches_ogs, 'pi_ABp'].values)
    pi_values.append(og_stats.loc[control_loci_ogs, 'pi_ABp'].dropna().values)
    print(x)
    print(pi_values)

    violins = ax.violinplot(pi_values, positions=x, widths=0.5, showextrema=False)
    colors = ['tab:orange', 'tab:blue', 'tab:purple', 'grey']
    for i, pc in enumerate(violins['bodies']):
        pc.set_facecolor(colors[i])

    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}nucleotide_diversity_comparison_violins.pdf')

def read_genomic_trench_loci(genomic_trench_loci_file):
    og_ids = []
    with open(genomic_trench_loci_file, 'r') as in_handle:
        for line in in_handle.readlines():
            if line[0] == '#':
                og_ids.append([])
            else:
                oid = line.strip()
                og_ids[-1].append(oid)
    return og_ids

def construct_trench_loci_table(parent_og_ids, og_table, metadata, osa_cds_dict, osbp_cds_dict):
    sites_df = pd.DataFrame(index=parent_og_ids, columns=['OG_ID', 'CYB_tag', 'genome_position', 'AA_length', 'annotation', 'potential_operon'])

    for pid in sites_df.index:
        candidate_og_ids = og_table.index[og_table['parent_og_id'] == pid].values
        annotated_og_ids = [oid for oid in candidate_og_ids if 'YSG' not in oid]
        if len(annotated_og_ids) > 0:
            sites_df.loc[pid, 'OG_ID'] = annotated_og_ids[0]
        else:
            sites_df.loc[pid, 'OG_ID'] = candidate_og_ids[0]


    # Add CYB locus tags
    cyb_tags = []
    for locus_id in sites_df['OG_ID']:
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
            og_id = sites_df.loc[idx, 'OG_ID']
            if 'CYA' in og_id:
                locus_tag = og_id
                annotation = osa_cds_dict[locus_tag]
            else:
                aln_length = og_table.loc[og_id, 'avg_length']
                sites_df.loc[idx, 'AA_length'] = int(aln_length / 3)
                continue
        else:
            locus_tag = cyb_tag
            annotation = osbp_cds_dict[locus_tag]

        loc = annotation.location
        pos_str = f'{loc.start}-{loc.end}'
        sites_df.loc[idx, 'genome_position'] = pos_str

        qualifiers = annotation.qualifiers
        aa_length = len(qualifiers['translation'][0])
        sites_df.loc[idx, 'AA_length'] = aa_length

        if 'product' in qualifiers:
            sites_df.loc[idx, 'annotation'] = qualifiers['product'][0]

    return sites_df


def plot_genomic_trenches_panel(species_cluster_genomes, pangenome_map, metadata, dx=2, w=5, min_og_presence=0.2, min_length=200, savefig=None):
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
    species_divergence_table = pd.DataFrame(index=filtered_idx, columns=['genome_position', 'species_divergence'])
    species_divergence_table['genome_position'] = species_cluster_genomes.loc[filtered_idx, 'osbp_location']
    mean_divergence, og_ids = pangenome_map.calculate_mean_divergence_between_groups(filtered_idx, species_sorted_sag_ids['A'], species_sorted_sag_ids['Bp'])
    species_divergence_table.loc[og_ids, 'species_divergence'] = mean_divergence
    species_divergence_table = species_divergence_table.loc[species_divergence_table['species_divergence'].notnull(), :]

    # Choose random high-coverage A and B' SAGs for individual pair comparisons
    sample_size = 10
    gene_presence_cutoff = 1000
    high_coverage_syna_sag_ids = list(np.array(species_sorted_sag_ids['A'])[(og_table[species_sorted_sag_ids['A']].notna().sum(axis=0) > gene_presence_cutoff).values])
    syna_sample_sag_ids = np.random.choice(high_coverage_syna_sag_ids, size=10)
    high_coverage_synbp_sag_ids = list(np.array(species_sorted_sag_ids['Bp'])[(og_table[species_sorted_sag_ids['Bp']].notna().sum(axis=0) > gene_presence_cutoff).values])
    synbp_sample_sag_ids = np.random.choice(high_coverage_synbp_sag_ids, size=10)
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
        x_pair_divergences.append(species_divergence_table.loc[og_id, 'genome_position'])
    pair_divergence_values = np.array(pair_divergence_values)
    x_pair_divergences = np.array(x_pair_divergences)

    fig = plt.figure(figsize=(double_col_width, 0.5 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel("OS-B' genome position (Mb)")
    ax.set_ylabel(r'$\alpha-\beta$ divergence')
    ax.set_ylim(0, 0.35)

    for i, pair_divergences in enumerate(pair_divergence_values.T):
        y_smooth = np.array([np.nanmean(pair_divergences[j:j + w]) if np.sum(np.isfinite(pair_divergences[j:j + w])) > 0 else np.nan for j in range(0, len(pair_divergences) - w, dx)])
        x_smooth = np.array([np.mean(x_pair_divergences[j:j + w]) for j in range(0, len(species_divergence_table) - w, dx)])
        ax.plot(x_smooth, y_smooth, lw=0.25, c='gray', alpha=0.3)

    y_smooth = np.array([np.mean(species_divergence_table['species_divergence'].values[j:j + w]) for j in range(0, len(species_divergence_table) - w, dx)])
    x_smooth = np.array([np.mean(species_divergence_table['genome_position'].values[j:j + w]) for j in range(0, len(species_divergence_table) - w, dx)])
    ax.plot(x_smooth, y_smooth, c='k', lw=1.5)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)

    print(species_divergence_table.loc[species_divergence_table['species_divergence'] < 0.03, :])
    typical_locus_filter = (species_divergence_table['species_divergence'] > 0.1) & (species_divergence_table['species_divergence'] < 0.2) & (species_divergence_table['genome_position'] > 0.6) & (species_divergence_table['genome_position'] < 0.7)
    print(species_divergence_table.loc[typical_locus_filter, :])



def plot_species_diversity_along_genome(species_cluster_genomes, pangenome_map, metadata, species='A', dx=2, w=5, min_og_presence=0.2, min_length=200, savefig=None):
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
    print(og_ids, len(og_ids))
    print(mean_divergence, len(mean_divergence))
    species_divergence_table.loc[og_ids, 'species_diversity'] = mean_divergence
    print(species_divergence_table.loc[species_divergence_table['species_diversity'].notnull(), :])
    species_divergence_table = species_divergence_table.loc[species_divergence_table['species_diversity'].notnull(), :]

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
    print(pair_divergence_values)
    x_pair_divergences = np.array(x_pair_divergences)


    # Plot results
    ylabel = r'nucleotide diversity, $\pi$'

    fig = plt.figure(figsize=(double_col_width, 0.5 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_ylim(0, 0.15)
    ax.set_ylim(1E-3, 3E-1)
    ax.set_yscale('log')

    for i, pair_divergences in enumerate(pair_divergence_values.T):
        y_smooth = np.array([np.nanmean(pair_divergences[j:j + w]) if np.sum(np.isfinite(pair_divergences[j:j + w])) > 0 else np.nan for j in range(0, len(pair_divergences) - w, dx)])
        x_smooth = np.array([np.mean(x_pair_divergences[j:j + w]) for j in range(0, len(species_divergence_table) - w, dx)])
        #ax.plot(x_smooth, y_smooth, lw=0.25, c='gray', alpha=0.3)
        ax.plot(x_smooth, y_smooth, lw=0.25, c=sample_color, alpha=0.3)

    y_smooth = np.array([np.mean(species_divergence_table['species_diversity'].values[j:j + w]) for j in range(0, len(species_divergence_table) - w, dx)])
    x_smooth = np.array([np.mean(species_divergence_table['genome_position'].values[j:j + w]) for j in range(0, len(species_divergence_table) - w, dx)])
    #ax.plot(x_smooth, y_smooth, c='k', lw=1.5)
    ax.plot(x_smooth, y_smooth, c=mean_color, lw=1.5)

    plt.tight_layout()
    if savefig is not None:
        print(f'Saving figure to {savefig}')
        plt.savefig(savefig)

    print(species_divergence_table.loc[species_divergence_table['species_diversity'] < 0.005, :])
    typical_locus_filter = (species_divergence_table['species_diversity'] > 0.01) & (species_divergence_table['species_diversity'] < 0.035) 
    print(species_divergence_table.loc[typical_locus_filter, :])


def analyze_pairwise_divergences(pangenome_map, args, label_fs=14, num_bins=50, ms=4, legend_fs=12):
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
    syna_backbone_idx = np.array([i for i in range(d_arr_raw.shape[0]) if core_og_ids[i] in low_diversity_ogs])
    syna_hybrid_idx = np.array([i for i in range(d_arr_raw.shape[0]) if core_og_ids[i] not in low_diversity_ogs])
    d_backbone_arr = d_arr_raw[syna_backbone_idx]
    d_hybrid_arr = d_arr_raw[syna_hybrid_idx]

    backbone_pdist = pd.DataFrame(np.nanmean(d_backbone_arr, axis=0), index=sag_ids, columns=sag_ids)
    hybrid_pdist = pd.DataFrame(np.nanmean(d_hybrid_arr, axis=0), index=sag_ids, columns=sag_ids)
    genome_pdist = pd.DataFrame(np.nanmean(d_arr_raw, axis=0), index=sag_ids, columns=sag_ids)
    species_sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')
    syna_sag_ids = np.array(species_sorted_sag_ids['A'])
    syna_sorted_sag_ids = metadata.sort_sags(syna_sag_ids, by='location')

    syna_backbone_pdist = backbone_pdist.loc[syna_sag_ids, syna_sag_ids]
    syna_hybrid_pdist = hybrid_pdist.loc[syna_sag_ids, syna_sag_ids]
    syna_genome_pdist = genome_pdist.loc[syna_sag_ids, syna_sag_ids]

    # Alpha low-diversity comparison between springs
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Mean pair divergence, $\pi_{ij}$', fontsize=label_fs)
    ax.set_xscale('log')
    ax.set_ylabel('Reverse cumulative', fontsize=label_fs)

    xlim = (1E-4, 2E-1)
    x_bins = np.geomspace(*xlim, num_bins)

    os_sag_ids = syna_sorted_sag_ids['OS']
    markers = ['o', 's', 'D']
    labels = ['backbone', 'hybrid', 'whole-genome']
    for i, segment_pdist in enumerate([syna_backbone_pdist, syna_hybrid_pdist, syna_genome_pdist]):
        pdist_values = utils.get_matrix_triangle_values(segment_pdist.loc[os_sag_ids, os_sag_ids].values, k=1)
        y = np.array([np.sum(pdist_values > x) / len(pdist_values) for x in x_bins])
        ax.plot(x_bins, y, f'-{markers[i]}', lw=1, ms=ms, alpha=0.5, mfc='none', label=f'{labels[i]}')

    ax.legend(frameon=False, fontsize=legend_fs)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_alpha_os_pdist_hist.pdf')
    plt.close()
    fig_count += 1

    os_idx = np.array([i for i in range(d_arr_raw.shape[1]) if sag_ids[i] in os_sag_ids])
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Nucleotide diversity')
    ax.set_xscale('log')
    ax.set_ylabel('Density')
    for i, idx in enumerate([syna_backbone_idx, syna_hybrid_idx]):
        pi_values = np.nanmean(d_arr_raw[idx][:, os_idx, :][:, :, os_idx], axis=(1, 2))
        ax.hist(pi_values, bins=x_bins, lw=2, histtype='step', label=f'{labels[i]}')
        print(pi_values)
    ax.legend(frameon=False, fontsize=legend_fs)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}S{fig_count}_alpha_genome_pi_hist.pdf')
    plt.close()
    fig_count += 1



if __name__ == '__main__':
    # Default parameters
    alignment_dir = '../results/single-cell/sscs_pangenome/_aln_results/'
    figures_dir = '../figures/analysis/tests/locus_diversity/'
    results_dir = '../results/single-cell/'
    #f_diversity_stats = f'../results/single-cell/sscs_pangenome/filtered_orthogroups/sscs_orthogroup_diversity_stats.tsv'
    f_diversity_stats = f'../results/single-cell/sscs_pangenome/filtered_orthogroups/sscs_parent_orthogroup_diversity_stats.tsv'
    orthogroup_table = '../results/single-cell/sscs_pangenome/filtered_orthogroups/sscs_annotated_single_copy_orthogroup_presence.tsv'


    parser = argparse.ArgumentParser()
    parser.add_argument('-A', '--alignment_dir', default=alignment_dir, help='Directory with orthogroup alignment files.')
    parser.add_argument('-F', '--figures_dir', default=figures_dir, help='Directory in which figures are saved.')
    parser.add_argument('-O', '--output_dir', help='Directory in which results are saved.')
    parser.add_argument('-R', '--results_dir', default=results_dir, help='Main results directory.')
    parser.add_argument('-d', '--diversity_stats_table', default=f_diversity_stats)
    parser.add_argument('-g', '--orthogroup_table', default=orthogroup_table, help='File with orthogroup table.')
    parser.add_argument('-n', '--min_num_seqs', default=10, type=int, help='Minimum number of sequences in each species cluster per OG.')
    parser.add_argument('-p', '--parent_og_diversity_stats_table', default=None)
    parser.add_argument('-r', '--random_seed', default=12345, type=int, help='RNG seed.')
    parser.add_argument('--data_stats_file', default=None)
    args = parser.parse_args()

    # Configuration
    np.random.seed(args.random_seed)
    pd.set_option('display.max_rows', 100)

    pangenome_map = PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    metadata = MetadataMap()

    analyze_pairwise_divergences(pangenome_map, args, legend_fs=10)

    '''
    og_stats = pd.read_csv(args.diversity_stats_table, sep='\t', index_col=0)
    genomic_trenches_stats = choose_Sbar_genomic_trench_loci(og_stats, args, make_plots=True, Sbarstar=0.1, Sbar_ratio=1.0)
    genomic_trenches_ogs = list(genomic_trenches_stats.index)
    print(genomic_trenches_stats[['trimmed_aln_length', 'A_seqs', 'Bp_seqs', 'S', 'Sbar', 'Sbar*', 'K', 'H']])

    # Choose controls
    pangenome_map.count_og_species_composition(metadata)
    control_loci_ogs = choose_control_loci(pangenome_map, list(genomic_trenches_stats.index), args)
    print(f'Control OG IDs: {control_loci_ogs}')
    '''

    '''
    # Get A and B' SAG IDs
    og_table = pangenome_map.og_table
    sag_ids = [col for col in og_table.columns if 'Uncmic' in col]
    sag_species_dict = metadata.sort_sags(sag_ids, by='species')
    synabp_sag_ids = [sag_species_dict['A'], sag_species_dict['Bp']]

    # Plot diversity comparions
    utils.print_break()

    if args.data_stats_file is None:
        genomic_trenches_stats = calculate_og_stats(genomic_trenches_ogs, og_table, synabp_sag_ids, args, index_type='og')
        print(genomic_trenches_stats)
        control_loci_stats = calculate_og_stats(control_loci_ogs, og_table, synabp_sag_ids, args)
        print(control_loci_stats)
        pickle.dump({'trench_stats':genomic_trenches_stats, 'control_stats':control_loci_stats}, open(f'{args.output_dir}genomic_trenches_stats.dat', 'wb'))
    else:
        data_stats = pickle.load(open(args.data_stats_file, 'rb'))
        genomic_trenches_stats = data_stats['trench_stats']
        control_loci_stats = data_stats['control_stats']

    #make_diversity_comparison_figures(genomic_trenches_ogs, control_loci_ogs, og_stats, og_table, synabp_sag_ids, args)
    make_diversity_comparison_figures(genomic_trenches_stats, control_loci_stats, og_table, args)
    '''

    '''
    og_table = pangenome_map.og_table
    og_stats = pd.read_csv(args.diversity_stats_table, sep='\t', index_col=0)
    genomic_trench_loci_file = f'{args.output_dir}genomic_trench_loci.txt'
    if os.path.exists(genomic_trench_loci_file):
        genomic_trenches_ogs, control_loci_ogs = read_genomic_trench_loci(genomic_trench_loci_file)
    else:
        genomic_trenches_stats = get_genomic_trenches_diversity_stats(og_stats, args)
        genomic_trenches_ogs = list(genomic_trenches_stats.index)

        pangenome_map.count_og_species_composition(metadata)
        control_loci_ogs = choose_control_loci(pangenome_map, list(genomic_trenches_stats.index), args, gt_ids_type='og')

        with open(genomic_trench_loci_file, 'w') as out_handle:
            out_handle.write('# Genomic trenches\n')
            for oid in genomic_trenches_ogs:
                out_handle.write(f'{oid}\n')

            out_handle.write('# Control\n')
            for oid in control_loci_ogs:
                out_handle.write(f'{oid}\n')

    # Export genomic trench loci table
    raw_osa_cds_dict = utils.read_genbank_cds('../data/reference_genomes/CP000239.genbank')
    osa_cds_dict = add_gene_id_map(raw_osa_cds_dict)
    raw_osbp_cds_dict = utils.read_genbank_cds('../data/reference_genomes/CP000240.genbank')
    osbp_cds_dict = add_gene_id_map(raw_osbp_cds_dict)
    genomic_trench_loci_table = construct_trench_loci_table(genomic_trenches_ogs, og_table, metadata, osa_cds_dict, osbp_cds_dict)
    genomic_trench_loci_table.to_csv(f'{args.output_dir}genomic_trench_loci_annotations.tsv', sep='\t')


    genomic_trenches_stats = og_stats.loc[genomic_trenches_ogs, :]
    print(genomic_trenches_stats.loc[genomic_trenches_ogs, :], len(genomic_trenches_ogs))
    print(f'Control OG IDs: {control_loci_ogs}')

    linkage_results_dir = '../results/single-cell/linkage_disequilibria/'
    rsq_avg_fname = f'{linkage_results_dir}sscs_rsq_avg.dat'
    if os.path.exists(rsq_avg_fname):
        species_rsq_avg = pickle.load(open(rsq_avg_fname, 'rb'))
    else:
        species_rsq_avg = []
        for i, species in enumerate(['A', 'Bp']):
            print(f'Averaging {species} loci...')
            fname_head = f'sscs_{species}_linkage_matrices'
            batch_map_fname = linkage_results_dir + '_'.join(fname_head.split('_')[:-1]) + '_batch_map.dat'
            if os.path.exists(batch_map_fname):
                linkage_table = pg_utils.LinkageTable(linkage_results_dir, locus_batch_map=batch_map_fname)
            else:
                linkage_table = pg_utils.LinkageTable(linkage_results_dir, fname_head=fname_head)
                linkage_table.save_batch_map_file(batch_map_fname)

            # Minimum number of seqs = 20% of maximum
            min_num_seqs = 0.2 * len(synabp_sag_ids[i])
            #min_num_seqs = 0.33 * len(synabp_sag_ids[i])
            
            og_ids = list(og_stats.loc[(og_stats[f'{species}_seqs'] - og_stats['num_duplicates']) >= min_num_seqs, :].index)
            print(f'{len(og_ids)} present in more than {min_num_seqs} cells found...')
            r2_matrix_avg = linkage_table.calculate_average_rsq(og_ids)
            species_rsq_avg.append(r2_matrix_avg)
        pickle.dump(species_rsq_avg, open(rsq_avg_fname, 'wb'))

    # Plot genomewide average r^2 curves for both species
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('separation, x')
    ax.set_xscale('log')
    ax.set_xlim(4E-1, 5E3)
    ax.set_ylabel(r'linkage disequilibrium, $r^2$')
    ax.set_yscale('log')
    ax.set_ylim(1E-4, 1.5E0)

    species = ['A', "B'"]
    colors = ['tab:orange', 'tab:blue']
    for i, rsq in enumerate(species_rsq_avg):
        x_cg, r2_cg, _ = plt_ld.average_over_distance(rsq, denom=2.0, coarse_grain=True, num_cg_points=20)
        ax.plot(x_cg, r2_cg, '-o', lw=1, ms=3, alpha=1.0, c=colors[i], label=species[i])

    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}genomewide_species_rsq_avg.pdf')

    make_linkage_disequilibrium_figures(genomic_trenches_ogs, control_loci_ogs, og_table, synabp_sag_ids, args, species_rsq_avg)

    plot_nucleotide_diversity_comparison(og_stats, genomic_trenches_ogs, control_loci_ogs, args)
    utils.print_break()
    plot_loci_genome_location(genomic_trenches_ogs, control_loci_ogs, og_table, args, savefig=f'{args.figures_dir}genomic_trenches_reference_genome_locations.pdf')
    '''

    '''
    species_cluster_genomes = pd.read_csv(f'{args.results_dir}hybridization/sscs_labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)
    osa_scale_factor = 1E-6 * 2932766 / 2905 # approximate gene position in Mb
    species_cluster_genomes['osa_location'] = species_cluster_genomes['CYA_tag'].str.split('_').str[-1].astype(float) * osa_scale_factor
    osbp_scale_factor = 1E-6 * 3046682 / 2942  # approximate gene position in Mb
    species_cluster_genomes['osbp_location'] = species_cluster_genomes['CYB_tag'].str.split('_').str[-1].astype(float) * osbp_scale_factor

    non_sag_columns = [col for col in species_cluster_genomes.columns if 'Uncmic' not in col]
    sag_columns = [col for col in species_cluster_genomes.columns if 'Uncmic' in col]
    species_cluster_genomes = species_cluster_genomes[non_sag_columns + sag_columns]
    species_cluster_genomes.to_csv(f'{args.results_dir}hybridization/sscs_labeled_sequence_cluster_genomes.tsv', sep='\t')

    divergence_files = [f'{args.alignment_dir}sscs_orthogroup_{j}_divergence_matrices.dat' for j in range(10)]
    pangenome_map.read_pairwise_divergence_results(divergence_files)
    #plot_genomic_trenches_panel(species_cluster_genomes, pangenome_map, metadata, savefig=f'{args.figures_dir}core_gene_species_divergences.pdf')
    for species in ['A', 'Bp']:
        plot_species_diversity_along_genome(species_cluster_genomes, pangenome_map, metadata, species=species, savefig=f'{args.figures_dir}{species}_core_gene_diversity.pdf')

    '''
