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
from scipy import stats
from pangenome_utils import PangenomeMap
from metadata_map import MetadataMap
from analyze_hitchhiking_candidates import add_gene_id_map, map_ssog_id
from plot_utils import *


def make_genomic_troughs_table(pangenome_map, metadata, args, tidx):
    og_table = pangenome_map.og_table

    genomic_trough_loci_file = f'{args.output_dir}genomic_trough_loci.txt'
    if os.path.exists(genomic_trough_loci_file):
        genomic_troughs_ogs, control_loci_ogs = read_genomic_trough_loci(genomic_trough_loci_file)
    else:
        genomic_troughs_stats = get_genomic_troughs_diversity_stats(og_stats, args)
        genomic_troughs_ogs = list(genomic_troughs_stats.index)

        pangenome_map.count_og_species_composition(metadata)
        control_loci_ogs = choose_control_loci(pangenome_map, list(genomic_troughs_stats.index), args, gt_ids_type='og')

        with open(genomic_trough_loci_file, 'w') as out_handle:
            out_handle.write('# Genomic trenches\n')
            for oid in genomic_troughs_ogs:
                out_handle.write(f'{oid}\n')

            out_handle.write('# Control\n')
            for oid in control_loci_ogs:
                out_handle.write(f'{oid}\n')

    # Export genomic trench loci table
    raw_osa_cds_dict = utils.read_genbank_cds('../data/reference_genomes/CP000239.genbank')
    osa_cds_dict = add_gene_id_map(raw_osa_cds_dict)
    raw_osbp_cds_dict = utils.read_genbank_cds('../data/reference_genomes/CP000240.genbank')
    osbp_cds_dict = add_gene_id_map(raw_osbp_cds_dict)
    genomic_trough_loci_table = construct_trench_loci_table(genomic_troughs_ogs, og_table, metadata, osa_cds_dict, osbp_cds_dict)
    genomic_trough_loci_table.to_csv(f'{args.supplement_dir}Table_S{tidx}_genomic_troughs.tsv', sep='\t')

    return tidx + 1


def read_genomic_trough_loci(genomic_trough_loci_file):
    og_ids = []
    with open(genomic_trough_loci_file, 'r') as in_handle:
        for line in in_handle.readlines():
            if line[0] == '#':
                og_ids.append([])
            else:
                oid = line.strip()
                og_ids[-1].append(oid)
    return og_ids


def get_genomic_troughs_diversity_stats(og_stats, args, pi_cutoff=0.05, length_cutoff=200, min_num_seqs=10, make_figures=True, tidx=1):
    index_filter = (og_stats['A_seqs'] >= min_num_seqs) & (og_stats['Bp_seqs'] >= min_num_seqs) & (og_stats['avg_length'] > length_cutoff)
    filtered_stats_table = og_stats.loc[index_filter, :]
    genomic_troughs_stats = filtered_stats_table.loc[(filtered_stats_table['pi_ABp'] < pi_cutoff) & (filtered_stats_table['num_duplicates'] == 0), :] # Exclude OGs with duplicates

    if make_figures:
        # Plot pi_ABp distributions
        plot_pdf(filtered_stats_table['pi_ABp'].values, bins=50, xlabel=r"nucleotide diversity, $\pi$", alpha=1.0, savefig=f'{args.figures_dir}synabp_mean_pi_distribution.pdf')
        ax = plot_pdf(filtered_stats_table['pi_ABp'].values, bins=50, xlabel=r"nucleotide diversity, $\pi$", alpha=1.0)
        ax.axvline(pi_cutoff, c='r', lw=2)
        plt.savefig(f'{args.figures_dir}Table_S{tidx}_total_diversity.pdf')

        '''
        plot_pdf(filtered_stats_table['pi_A'].dropna().values, bins=50, xlabel=r"nucleotide diversity, $\pi$", alpha=1.0, savefig=f'{args.figures_dir}syna_pi_distribution.pdf')
        plot_pdf(filtered_stats_table['pi_Bp'].dropna().values, bins=50, xlabel=r"nucleotide diversity, $\pi$", alpha=1.0, savefig=f'{args.figures_dir}synbp_pi_distribution.pdf')

        ax = plot_pdf(filtered_stats_table['pi_ABp'].values, bins=50, xlabel=r"nucleotide diversity, $\pi$", color='tab:purple', alpha=0.5, data_label="A vs B'")
        ax.hist(filtered_stats_table['pi_A'].dropna().values, bins=50, color='tab:orange', alpha=0.5, label='A')
        ax.hist(filtered_stats_table['pi_Bp'].dropna().values, bins=50, color='tab:blue', alpha=0.5, label="B'")
        ax.legend()
        plt.savefig(f'{args.figures_dir}synabp_mean_pi_distribution_overlapped.pdf')
        '''

    return genomic_troughs_stats


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

def get_og_ids(og_table):
    raw_og_ids = og_table['parent_og_id'].values
    parent_og_ids = [oid.split('-')[0] for oid in raw_og_ids]
    return np.unique(parent_og_ids)


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

def construct_trench_loci_table(parent_og_ids, og_table, metadata, osa_cds_dict, osbp_cds_dict):
    sites_df = pd.DataFrame(index=parent_og_ids, columns=['OG_ID', 'CYB_tag', 'genome_position', 'AA_length', 'annotation'])

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

def make_sample_association_tables(pangenome_map, rng, args, tidx):
    tidx = analyze_linkage_block_correlations(pangenome_map, metadata, args, rng, tidx)
    return tidx


def analyze_hybrid_gene_sample_correlations(pangenome_map, args, rng, tidx):
    hybridization_dir = f'{args.results_dir}hybridization/'
    species_cluster_genomes = pd.read_csv(f'{hybridization_dir}sscs_labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)
    metadata = MetadataMap()

    # Make test and control groups for species composition
    pure_syna_sample_sags, control_sample_sags = make_syna_test_samples(pangenome_map, metadata)
    test_sample_sags = rng.choice(pure_syna_sample_sags, size=len(control_sample_sags), replace=False)

    combined_sag_ids = rng.permutation(np.concatenate([test_sample_sags, control_sample_sags]))
    randomized_test_sags, randomized_control_sags = combined_sag_ids[:len(test_sample_sags)], combined_sag_ids[len(test_sample_sags):] 

    donor_list = ['A', 'Bp', 'C', 'O']
    results_dict = {}
    for species in ['A', 'Bp']:
        species_hybrid_donor_frequency_table = make_donor_frequency_table(species_cluster_genomes, species, pangenome_map, metadata)
        species_donors = [s for s in donor_list if s != species]
        species_results = pd.DataFrame(index=species_hybrid_donor_frequency_table.index, columns=np.concatenate([[f'{s}_OR', f'{s}_pvalue', f'{s}_randomized_pvalue'] for s in species_donors]))

        for s in species_donors:
            contingency_tables = construct_contingency_tables(species_cluster_genomes, species_hybrid_donor_frequency_table, test_sample_sags, control_sample_sags, species, s)
            randomized_contingency_tables = construct_contingency_tables(species_cluster_genomes, species_hybrid_donor_frequency_table, randomized_test_sags, control_sample_sags, species, s)
            for o in contingency_tables:
                odds_ratio, pvalue = stats.fisher_exact(contingency_tables[o])
                species_results.loc[o, [f'{s}_OR', f'{s}_pvalue']] = odds_ratio, pvalue
                randomized_or, randomized_pvalue = stats.fisher_exact(randomized_contingency_tables[o])
                species_results.loc[o, f'{s}_randomized_pvalue'] = randomized_pvalue

        results_dict[species] = species_results
        break

    for species in results_dict:
        species_results = results_dict[species]
        for col in species_results.columns:
            print(species, col, utils.sorted_unique(species_results[col].dropna(), sort='ascending', sort_by='tags'))
        print('\n\n')


def get_species_composition_sag_ids(pangenome_map, metadata, species):
    sag_ids = pangenome_map.get_sag_ids()
    species_sag_ids = metadata.sort_sags(sag_ids, by='species')[species]
    sample_sorted_sag_ids = metadata.sort_sags(species_sag_ids, by='sample')

    if species == 'A':
        pure_species_sag_ids = np.array(sample_sorted_sag_ids['OS2005'])
        mixed_species_sag_ids = np.concatenate([sample_sorted_sag_ids[sample] for sample in ['MS2004', 'MS2006', 'OS2009']])
    elif species == 'Bp':
        pure_species_sag_ids = np.array(sample_sorted_sag_ids['MS2005'])
        mixed_species_sag_ids = np.concatenate([sample_sorted_sag_ids[sample] for sample in ['MS2004', 'MS2006']])
    else:
        print('Invalid species in get_species_composition_sag_ids. Returning None values.')
        pure_species_sag_ids = None
        mixed_species_sag_ids = None

    return pure_species_sag_ids, mixed_species_sag_ids


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
        gene_clusters = species_core_genome_clusters.loc[o, species_sorted_sags[species]].dropna().replace({'a':'A', 'b':'Bp'}).values
        if len(gene_clusters) > 0:
            seq_clusters = np.concatenate([[utils.split_alphanumeric_string(s)[0] for s in c.split(',')] for c in gene_clusters])
            unique_clusters, cluster_counts = utils.sorted_unique(seq_clusters)
            donor_freq_table.loc[o, unique_clusters] = cluster_counts

    return donor_freq_table

def construct_contingency_tables(species_cluster_genomes, species_hybrid_donor_frequency_table, test_sample_sags, control_sample_sags, species, s):
    hybridized_og_ids = species_hybrid_donor_frequency_table.index[species_hybrid_donor_frequency_table[s] > 0]
    test_alleles = species_cluster_genomes.loc[hybridized_og_ids, test_sample_sags]
    control_alleles = species_cluster_genomes.loc[hybridized_og_ids, control_sample_sags]

    n_test_species = pd.Series(np.nansum(np.array([test_alleles[sag_id].str.contains(species).values for sag_id in test_alleles.columns]).T, axis=1), index=hybridized_og_ids)
    n_test_hybrid = pd.Series(np.nansum(np.array([test_alleles[sag_id].str.contains(s).values for sag_id in test_alleles.columns]).T, axis=1), index=hybridized_og_ids)
    n_control_species = pd.Series(np.nansum(np.array([control_alleles[sag_id].str.contains(species).values for sag_id in control_alleles.columns]).T, axis=1), index=hybridized_og_ids)
    n_control_hybrid = pd.Series(np.nansum(np.array([control_alleles[sag_id].str.contains(s).values for sag_id in control_alleles.columns]).T, axis=1), index=hybridized_og_ids)

    contingency_tables = {}
    for o in hybridized_og_ids:
        contingency_tables[o] = np.array([[n_test_species[o], n_control_species[o]], [n_test_hybrid[o], n_control_hybrid[o]]])

    return contingency_tables


def analyze_linkage_block_correlations(pangenome_map, metadata, args, rng, tidx, min_haplotype_frequency=5):
    alignments_dir = f'{args.results_dir}alignments/core_ogs_cleaned/'
    syn_homolog_map = SynHomologMap(build_maps=True)
    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')

    for species in ['A', 'Bp']:
        # Species composition test
        test_sag_ids, control_sag_ids = get_species_composition_sag_ids(pangenome_map, metadata, species)
        species_composition_table = make_association_table(test_sag_ids, control_sag_ids, species, rng, min_haplotype_frequency, args)
        species_composition_table.to_csv(f'{args.supplement_dir}Table_S{tidx}_{species}_blocks_composition_association.tsv', sep='\t')
        tidx += 1

        # Sample location test
        location_sorted_sags = metadata.sort_sags(species_sorted_sags[species], by='location')
        test_sag_ids = location_sorted_sags['MS']
        control_sag_ids = location_sorted_sags['OS']
        location_table = make_association_table(test_sag_ids, control_sag_ids, species, rng, min_haplotype_frequency, args)
        location_table.to_csv(f'{args.supplement_dir}Table_S{tidx}_{species}_blocks_location_association.tsv', sep='\t')
        tidx += 1

    return tidx


def make_association_table(test_sag_ids, control_sag_ids, species, rng, min_haplotype_frequency, args):
    # Filter blocks with low-frequency haplotypes
    merged_sag_ids = np.concatenate([test_sag_ids, control_sag_ids])
    #species_block_haplotypes = pd.read_csv(f'{args.results_dir}snp_blocks/{species}_all_sites_hybrid_linkage_block_haplotypes.tsv', sep='\t', index_col=0)
    species_block_haplotypes = pd.read_csv(f'{args.results_dir}main_figures_data/{species}_core_snp_block_haplotypes.tsv', sep='\t', index_col=0)
    species_block_haplotypes = species_block_haplotypes.loc[merged_sag_ids, :]
    minor_haplotype_frequency = (species_block_haplotypes >= 20).sum(axis=0)
    filtered_block_ids = np.array(minor_haplotype_frequency.index[minor_haplotype_frequency.values >= min_haplotype_frequency])
    results_table = calculate_fisher_exact_tests(species_block_haplotypes, test_sag_ids, control_sag_ids, filtered_block_ids)
    print(species_block_haplotypes.dropna(axis=0, how='all'))
    print(results_table)

    # Calculate control with randomized groups
    randomized_sag_ids = rng.permutation(merged_sag_ids)
    randomized_test_sag_ids = randomized_sag_ids[:len(test_sag_ids)]
    randomized_control_sag_ids = randomized_sag_ids[len(test_sag_ids):]
    control_table = calculate_associations(species_block_haplotypes, randomized_test_sag_ids, randomized_control_sag_ids, filtered_block_ids)

    # Add control pvalues
    results_table['pvalue_randomized'] = control_table.loc[results_table.index.values, 'pvalue']
    results_table['pvalue_ratio'] = results_table['pvalue_randomized'].min() / results_table['pvalue']
    results_table['significant_association'] = 'no'
    results_table.loc[results_table['pvalue_ratio'] > 1., 'significant_association'] = 'yes'

    return results_table.sort_values('pvalue_ratio', ascending=False)

def calculate_associations(species_block_haplotypes, test_sag_ids, control_sag_ids, filtered_block_ids):
    n00_arr = (species_block_haplotypes.loc[test_sag_ids, filtered_block_ids] < 20).sum(axis=0)
    n01_arr = (species_block_haplotypes.loc[control_sag_ids, filtered_block_ids] < 20).sum(axis=0)
    n10_arr = (species_block_haplotypes.loc[test_sag_ids, filtered_block_ids] >= 20).sum(axis=0)
    n11_arr = (species_block_haplotypes.loc[control_sag_ids, filtered_block_ids] >= 20).sum(axis=0)
    contingency_tables = np.array([[n00_arr, n01_arr], [n10_arr, n11_arr]])

    results_table = pd.DataFrame(index=filtered_block_ids, columns=['coverage', 'num_hapl1_MS', 'num_hapl1_OS', 'num_hapl2_MS', 'num_hapl2_OS', 'odds_ratio', 'pvalue'])
    results_table['coverage'] = np.sum(contingency_tables, axis=(0, 1))
    for i, block in enumerate(filtered_block_ids):
        block_table = contingency_tables[:, :, i]
        results_table.loc[block, ['odds_ratio', 'pvalue']] = stats.fisher_exact(block_table)
        results_table.loc[block, ['num_hapl1_MS', 'num_hapl1_OS', 'num_hapl2_MS', 'num_hapl2_OS']] = block_table[0, 0], block_table[0, 1], block_table[1, 0], block_table[1, 1]

    return results_table

def calculate_fisher_exact_tests(species_block_haplotypes, test_sag_ids, control_sag_ids, filtered_block_ids):
    n00_arr = (species_block_haplotypes.loc[test_sag_ids, filtered_block_ids] < 20).sum(axis=0)
    n01_arr = (species_block_haplotypes.loc[control_sag_ids, filtered_block_ids] < 20).sum(axis=0)
    n10_arr = (species_block_haplotypes.loc[test_sag_ids, filtered_block_ids] >= 20).sum(axis=0)
    n11_arr = (species_block_haplotypes.loc[control_sag_ids, filtered_block_ids] >= 20).sum(axis=0)
    contingency_tables = np.array([[n00_arr, n01_arr], [n10_arr, n11_arr]])

    results_table = pd.DataFrame(index=filtered_block_ids, columns=['coverage', 'num_hapl1_test', 'num_hapl1_control', 'num_hapl2_test', 'num_hapl2_control', 'odds_ratio', 'pvalue'])
    results_table['coverage'] = np.sum(contingency_tables, axis=(0, 1))
    for i, block in enumerate(filtered_block_ids):
        block_table = contingency_tables[:, :, i]
        results_table.loc[block, ['odds_ratio', 'pvalue']] = stats.fisher_exact(block_table)
        results_table.loc[block, ['num_hapl1_test', 'num_hapl1_control', 'num_hapl2_test', 'num_hapl2_control']] = block_table[0, 0], block_table[0, 1], block_table[1, 0], block_table[1, 1]

    return results_table


def add_annotations_to_significant_blocks(results_table, control_table, syn_homolog_map, pangenome_map):
    # Get annotations for significant associations
    control_min_pvalue = control_table['pvalue'].min()
    significant_subtable = results_table.loc[results_table['pvalue'] < control_min_pvalue, :]
    print(results_table.loc[results_table['pvalue'] < control_min_pvalue / 10, :])

    og_table = pangenome_map.og_table
    significant_ogs = np.array([s.split('_block')[0] for s in significant_subtable.index])
    for i, og in enumerate(significant_ogs):
        # Get ref. genome tag if available
        locus_tags = og_table.loc[og_table['parent_og_id'] == og, 'locus_tag'].dropna().unique()
        synbp_idx = ['CYB' in t for t in locus_tags]
        if np.sum(synbp_idx) > 0:
            ref_tag = locus_tags[synbp_idx][0]
        elif len(locus_tags) > 0:
            ref_tag = locus_tags[0]
        else:
            print(og, locus_tags)
            continue

        block_id = significant_subtable.index[i]
        results_table.loc[block_id, 'gene_product'] = syn_homolog_map.get_cds_annotation(ref_tag).qualifiers['product'][0]

    return results_table.sort_values('pvalue')

if __name__ == '__main__':
    # Default parameters
    alignment_dir = '../results/single-cell/sscs_pangenome/_aln_results/'
    figures_dir = '../figures/supplement/'
    output_dir = '../results/single-cell/supplement/'
    results_dir = '../results/single-cell/'
    supplement_dir = '../doc/paper_drafts/supplement/'
    f_diversity_stats = f'../results/single-cell/sscs_pangenome/filtered_orthogroups/sscs_parent_orthogroup_diversity_stats.tsv'
    orthogroup_table = '../results/single-cell/sscs_pangenome/filtered_orthogroups/sscs_annotated_single_copy_orthogroup_presence.tsv'


    parser = argparse.ArgumentParser()
    parser.add_argument('-A', '--alignment_dir', default=alignment_dir, help='Directory with orthogroup alignment files.')
    parser.add_argument('-F', '--figures_dir', default=figures_dir, help='Directory in which figures are saved.')
    parser.add_argument('-O', '--output_dir', default=output_dir, help='Directory in which results are saved.')
    parser.add_argument('-R', '--results_dir', default=results_dir, help='Main results directory.')
    parser.add_argument('-S', '--supplement_dir', default=supplement_dir, help='Directory with supplemental tables.')
    parser.add_argument('-d', '--diversity_stats_table', default=f_diversity_stats)
    parser.add_argument('-g', '--orthogroup_table', default=orthogroup_table, help='File with orthogroup table.')
    parser.add_argument('-n', '--min_num_seqs', default=10, type=int, help='Minimum number of sequences in each species cluster per OG.')
    parser.add_argument('-r', '--random_seed', default=238759, type=int, help='RNG seed.')
    args = parser.parse_args()

    # Configuration
    rng = np.random.default_rng(args.random_seed)

    pangenome_map = PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    metadata = MetadataMap()
    og_stats = pd.read_csv(args.diversity_stats_table, sep='\t', index_col=0)
    tidx = 1

    #tidx = make_genomic_troughs_table(pangenome_map, metadata, args, tidx)
    tidx = 2
    tidx = make_sample_association_tables(pangenome_map, rng, args, tidx)

