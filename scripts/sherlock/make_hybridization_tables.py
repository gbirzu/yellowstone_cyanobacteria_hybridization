import argparse
import pandas as pd
import numpy as np
import pickle
import os
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
import find_hybridization_events as find_hybrids
from pangenome_utils import PangenomeMap
from metadata_map import MetadataMap
from syn_homolog_map import SynHomologMap


'''
This script generates the tables used to make the gene-level hybridization figures. 
'''


def make_hybridization_counts_table(hybridization_table, pangenome_map, metadata, min_og_frequency=0.2, og_type='parent_og_id'):
    og_table = pangenome_map.og_table
    og_ids = np.sort(og_table['parent_og_id'].dropna().unique())
    hybridization_counts_df = pd.DataFrame(index=og_ids, columns=['CYA_tag', 'CYB_tag', 'core_A', 'core_Bp', 'Bp->A', 'C->A', 'O->A', 'A->Bp', 'C->Bp', 'O->Bp', 'total_transfers'])

    # Add core OGs
    hybridization_counts_df = add_core_og_labels(hybridization_counts_df, og_table)

    # Add locus tags
    hybridization_counts_df = add_locus_tags(hybridization_counts_df, og_table)
    hybridization_counts_df = sort_hybridization_events_by_type(hybridization_counts_df, pangenome_map, metadata)

    return hybridization_counts_df

def add_core_og_labels(locus_table, og_table):
    core_df = og_table[['parent_og_id', 'core_A', 'core_Bp']].replace('Yes', 1).groupby('parent_og_id').sum()
    core_df.loc[core_df['core_A'] > 0, 'core_A'] = 'Yes'
    core_df.loc[core_df['core_Bp'] > 0, 'core_Bp'] = 'Yes'
    locus_table.loc[:, ['core_A', 'core_Bp']] = core_df.loc[locus_table.index.values, ['core_A', 'core_Bp']]
    return locus_table

def add_locus_tags(locus_table, og_table):
    for og_id in locus_table.index:
        for col in ['CYA_tag', 'CYB_tag']:
            tags = og_table.loc[og_table['parent_og_id'] == og_id, col].dropna().values
            if len(tags) > 0:
                locus_table.loc[og_id, col] = tags[0]
    return locus_table

def sort_hybridization_events_by_type(hybridization_counts_df, pangenome_map, metadata):
    og_table = pangenome_map.og_table
    gene_cluster_mismatches = find_gene_cell_cluster_mismatches(og_table, metadata)
    mismatches_idx = list(gene_cluster_mismatches.index)
    transfer_columns = [col for col in hybridization_counts_df.columns if '->' in col] + ['total_transfers']
    hybridization_counts_df[transfer_columns] = 0 # initialize transfer type columns

    for i, og_id in enumerate(og_table['parent_og_id'].unique()):
        sog_ids = [idx for idx in og_table.loc[og_table['parent_og_id'] == og_id, :].index if idx in mismatches_idx]
        if len(sog_ids) == 1:
            mismatches_subtable = gene_cluster_mismatches.loc[sog_ids[0], :].dropna()
            mismatched_sag_ids = mismatches_subtable.index[mismatches_subtable == True].values
            donor_cluster = og_table.loc[sog_ids[0], 'sequence_cluster']
            host_clusters = [metadata.get_sag_species(sag_id) for sag_id in mismatched_sag_ids]
            for host_species in host_clusters:
                transfer_type = f'{donor_cluster}->{host_species}'
                hybridization_counts_df.loc[og_id, transfer_type] += 1
                hybridization_counts_df.loc[og_id, 'total_transfers'] += 1

        else:
            mismatches_subtable = gene_cluster_mismatches.loc[sog_ids, :].dropna(axis=1, how='all')
            mismatched_sag_ids = mismatches_subtable.columns[(mismatches_subtable == True).sum(axis=0) > 0].values
            donor_cluster = og_table.loc[sog_ids, 'sequence_cluster']
            host_clusters = [[metadata.get_sag_species(sag_id) for sag_id in mismatched_sag_ids[mismatches_subtable.loc[sog_id, mismatched_sag_ids] == True]] for sog_id in sog_ids] 
            for j, donor in enumerate(donor_cluster):
                if len(host_clusters[j]) > 0:
                    for host_species in host_clusters[j]:
                        transfer_type = f'{donor}->{host_species}'
                        hybridization_counts_df.loc[og_id, transfer_type] += 1
                        hybridization_counts_df.loc[og_id, 'total_transfers'] += 1

    return hybridization_counts_df


def find_gene_cell_cluster_mismatches(og_table, metadata):
    relevant_cluster_labels = ['A', 'Bp', 'C', 'O']

    # Make mismatched SAG IDs dict
    #   maps each SOG cluster to SAGs that would be mismatched if they contained the SOG
    sag_ids = [col for col in og_table.columns if 'Uncmic' in col]
    sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')
    mismatched_sags_dict = {}
    for cluster_label in relevant_cluster_labels:
        mismatched_sags_dict[cluster_label] = np.concatenate([sorted_sag_ids[l] for l in sorted_sag_ids if l != cluster_label])

    # Add mixed SOG subclusters
    mismatched_sags_dict['a'] = mismatched_sags_dict['A']
    mismatched_sags_dict['b'] = mismatched_sags_dict['Bp']
    relevant_cluster_labels += ['a', 'b']

    # Make mismatch table
    relevant_sog_ids = list(og_table.loc[og_table['sequence_cluster'].isin(relevant_cluster_labels), :].index)
    og_cluster_mismatch_table = pd.DataFrame(index=relevant_sog_ids, columns = sag_ids)
    for cluster_label in mismatched_sags_dict:
        mismatched_sag_ids = mismatched_sags_dict[cluster_label]
        subtable = og_table.loc[og_table['sequence_cluster'] == cluster_label, mismatched_sag_ids]
        og_cluster_mismatch_table.loc[list(subtable.index), mismatched_sag_ids] = subtable.notna()

    return og_cluster_mismatch_table


def initialize_sequence_cluster_haplotypes(pangenome_map, metadata, syn_homolog_map):
    og_table = pangenome_map.og_table
    og_ids = np.sort(og_table['parent_og_id'].dropna().unique())
    metadata = MetadataMap()
    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    synabp_sag_ids = np.concatenate((species_sorted_sags['A'], species_sorted_sags['Bp']))
    species_cluster_genomes = pd.DataFrame(index=og_ids, columns=np.concatenate((['CYA_tag', 'CYB_tag', 'core_A', 'core_Bp', 'osa_location', 'osbp_location'], sag_ids)) )

    # Add species core OGs
    species_cluster_genomes = add_core_og_labels(species_cluster_genomes, og_table)

    # Add locus tags
    species_cluster_genomes = add_locus_tags(species_cluster_genomes, og_table)

    # Add genome positions
    osa_scale_factor = 1E-6 * 2932766 / 2905 # approximate gene position in Mb
    species_cluster_genomes['osa_location'] = species_cluster_genomes['CYA_tag'].str.split('_').str[-1].astype(float) * osa_scale_factor
    osbp_scale_factor = 1E-6 * 3046682 / 2942  # approximate gene position in Mb
    species_cluster_genomes['osbp_location'] = species_cluster_genomes['CYB_tag'].str.split('_').str[-1].astype(float) * osbp_scale_factor

    return species_cluster_genomes


def annotate_haplotype_gene_clusters(species_cluster_genomes, pangenome_map, metadata, excluded_contigs=[]):
    og_table = pangenome_map.og_table.copy()
    sag_ids = pangenome_map.get_sag_ids()
    sag_species = ['A', 'Bp', 'C']
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')

    # Map species cluster labels
    og_table['sequence_cluster'] = og_table['sequence_cluster'].map({'A':'A', 'Bp':'Bp', 'M':'M', 'C':'C', 'O':'O', 'a':'A', 'b':'Bp'}, na_action='ignore')
    core_og_ids = np.array([o for o in species_cluster_genomes.index[(species_cluster_genomes[['core_A', 'core_Bp']] == 'Yes').any(axis=1)] if 'rRNA' not in o])

    num_multiple_alleles = 0
    num_filtered_multiple_alleles = 0
    for o in core_og_ids:
        og_idx = np.array(og_table.loc[og_table['parent_og_id'] == o, :].index)
        og_clusters = og_table.loc[og_idx, 'sequence_cluster'].copy()

        # Assign unique ID for multiple alleles from same OG
        if len(og_clusters.unique()) < len(og_clusters):
            cluster_labels, label_count = utils.sorted_unique(og_clusters.values)
            for c in cluster_labels[label_count > 1]:
                new_labels = [f'{c}{i+1}' for i in range(label_count[cluster_labels == c][0])]
                og_clusters[og_clusters == c] = new_labels

        # Check if any SAGs have more than one cluster
        num_genes = np.array([len(og_table.loc[og_idx, s].dropna()) for s in sag_ids])
        for s in sag_ids[num_genes > 1]:
            # Get index for covered OGs on valid contigs
            sag_allele_contigs = og_table.loc[og_idx, s].dropna().str.split('_').str[:2].str.join('_')
            filtered_idx = sag_allele_contigs.index[~sag_allele_contigs.isin(excluded_contigs)]

            # Assign cluster label
            s_alleles = og_clusters[filtered_idx].values
            species_cluster_genomes.loc[o, s] = ','.join(np.sort(s_alleles))

            num_multiple_alleles += 1
            if len(s_alleles) > 1:
                num_filtered_multiple_alleles += 1

        #for idx, c in og_clusters.iteritems():
        for idx, c in og_clusters.items():
            cluster_sag_ids = np.array(og_table.loc[idx, sag_ids[num_genes == 1]].dropna().index)
            species_cluster_genomes.loc[o, cluster_sag_ids] = c

            # Filter out OGs on excluded contigs
            covered_sag_ids = np.array(og_table.loc[idx, sag_ids[num_genes == 1]].dropna().index)
            sag_contig_ids = og_table.loc[idx, covered_sag_ids].str.split('_').str[:2].str.join('_') # Get contig IDs for covered SAGs
            filtered_contig_sag_ids = sag_contig_ids.index[sag_contig_ids.isin(excluded_contigs)]
            if len(filtered_contig_sag_ids) > 0:
                species_cluster_genomes.loc[o, filtered_contig_sag_ids] = np.nan

    return species_cluster_genomes

def count_gene_hybrids(species_cluster_genomes, species_sorted_sags, host_species=['A', 'Bp'], donor_species=['A', 'Bp', 'C', 'O']):
    '''
    Calculates distribution of hybridization events from species cluster labeled genomes.
    '''

    # Initialize table
    metadata_columns = ['CYA_tag', 'CYB_tag', 'core_A', 'core_Bp']
    transfer_columns = []
    for h in host_species:
        for d in donor_species:
            if h != d:
                transfer_columns.append(f'{d}->{h}')
    columns = metadata_columns + transfer_columns
    hybridization_counts_df = pd.DataFrame(0, index=species_cluster_genomes.index.values, columns=columns)
    hybridization_counts_df.loc[:, metadata_columns] = species_cluster_genomes.loc[:, metadata_columns]

    # Count hybrids
    species_cluster_genomes = species_cluster_genomes.fillna('')
    for host in host_species:
        host_sags = species_sorted_sags[host]
        host_donors = [s for s in donor_species if s != host]

        for donor in host_donors:
            col = f'{donor}->{host}'
            for s in host_sags:
                hybrids = species_cluster_genomes[s].str.contains(donor).astype(int)
                hybridization_counts_df.loc[:, col] += hybrids

    hybridization_counts_df['total_transfers'] = hybridization_counts_df[transfer_columns].sum(axis=1)

    return hybridization_counts_df


def sort_sags_by_species(sag_ids, metadata):
    '''
    Sorts SAG IDs by species and returns dict with just species present.
    '''
    temp = metadata.sort_sags(sag_ids, by='species')
    species_sorted_sags = {}
    for s in temp.keys():
        # Remove empty categories
        if len(temp[s]) > 0:
            species_sorted_sags[s] = temp[s]
    return species_sorted_sags


if __name__ == '__main__':
    # Define default variables
    hybridization_table = '../results/single-cell/hybridization/sscs_hybridization_events.tsv'
    orthogroup_table = '../results/single-cell/sscs_pangenome/filtered_orthogroups/sscs_annotated_single_copy_orthogroup_presence.tsv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--annotations_dir', default=None, help='Directory with SAG GFF annotations.')
    parser.add_argument('-O', '--output_dir', required=True, help='Directory in which results are saved.')
    parser.add_argument('-g', '--orthogroup_table', default=orthogroup_table, help='File with orthogroup table.')
    parser.add_argument('-t', '--hybridization_table', default=hybridization_table, help='File with contigs containing hybrid genes.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    parser.add_argument('--validate_counts', action='store_true', help='Recalculate transfer counts from cluster labeled genomes for validation.')
    parser.add_argument('--contig_length_cutoff', type=int, default=0, help='Filter hybrid contigs with this or fewer number of genes.')
    parser.add_argument('--min_og_presence', type=float, default=0.2, help='Minimum species presence for core OGs.')
    parser.add_argument('--min_length', type=int, default=200, help='Minimum alignment length for genomic trenches analysis.')
    parser.add_argument('--random_seed', default=12345, type=int, help='Random seed for reproducibility.')
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    pd.set_option('display.max_rows', 150)

    hybridization_table = utils.read_hybridization_table(args.hybridization_table, length_cutoff=args.contig_length_cutoff)
    fully_hybrid_contigs = np.array(hybridization_table.loc[hybridization_table['num_matches'] == 0, :].index) # filter out contigs without species alleles

    # Save hybrid contigs
    with open(f'{args.output_dir}fully_hybrid_contigs.txt', 'w') as f_out:
        for c in fully_hybrid_contigs:
            f_out.write(f'{c}\n')

    pangenome_map = PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    og_table = pangenome_map.og_table
    metadata = MetadataMap()
    syn_homolog_map = SynHomologMap(build_maps=True)

    filtered_hybridization_table = hybridization_table.loc[hybridization_table.index[~hybridization_table.index.isin(fully_hybrid_contigs)], :].copy()
    hybridization_counts = make_hybridization_counts_table(filtered_hybridization_table, pangenome_map, metadata)
    hybridization_counts.to_csv(f'{args.output_dir}hybridization_counts_table.tsv', sep='\t')

    if args.verbose:
        print('Full hybridization table:')
        print(hybridization_table)
        print('\n')
        print('Hybridization table excluding contings without same species genes:')
        print(filtered_hybridization_table)
        print('\n')
        print('OGs with simple transfers:')
        print(hybridization_counts.loc[hybridization_counts['total_transfers'] > 0, :])
        print('\n\n')

    species_cluster_genomes = initialize_sequence_cluster_haplotypes(pangenome_map, metadata, syn_homolog_map)
    species_cluster_genomes = annotate_haplotype_gene_clusters(species_cluster_genomes, pangenome_map, metadata, excluded_contigs=fully_hybrid_contigs)
    species_cluster_genomes.to_csv(f'{args.output_dir}labeled_sequence_cluster_genomes.tsv', sep='\t')

    if args.verbose:
        print(species_cluster_genomes)
        print('\n\n')

    if args.validate_counts:
        # Validate hybridization counts
        sag_ids = pangenome_map.get_sag_ids()
        species_sorted_sags = sort_sags_by_species(sag_ids, metadata)
        counts_validation = count_gene_hybrids(species_cluster_genomes, species_sorted_sags)
        counts_validation.to_csv(f'{args.output_dir}hybridization_counts_table_validation.tsv', sep='\t')

        if args.verbose:
            print('Counts validation:')
            print(hybridization_counts)
            print(counts_validation)
            print(counts_validation)

