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
This scripts generates all of the tables used to make the gene-level hybridization figures. 
The code is based on `make_hybridization_figures.py`, which was used in previous versions.
'''


def catalog_hybridization_events(filtered_hybridization_table):
    # Initialize counts dict
    cluster_labels = ['A', 'B', 'C', 'O']
    transfer_type_counts = {}
    for cl1 in cluster_labels:
        non_cluster_labels = [label for label in cluster_labels if label not in [cl1, 'O']] # host can only be A, B, or C, but must be different from donor
        for cl2 in non_cluster_labels:
            transfer_type_counts[f'{cl1}->{cl2}'] = 0

    for contig, contig_row in filtered_hybridization_table.iterrows():
        # Get cluster labels for all hybrid genes
        sag_species_label = contig_row['sag_species'][0]
        non_cluster_labels = [label for label in cluster_labels if label != sag_species_label]
        contig_cluster_labels = np.array(list(contig_row['cluster_labels_sequence']))
        filtered_cluster_labels = contig_cluster_labels[np.isin(contig_cluster_labels, non_cluster_labels)]

        #print(contig, sag_species_label)
        #print(contig_cluster_labels, filtered_cluster_labels, contig_row['num_hybrid_genes'])
        if len(filtered_cluster_labels) != contig_row['num_hybrid_genes']:
            print(contig)
            print(contig_row)
            print(filtered_cluster_labels)
            print('\n')

        for cl in filtered_cluster_labels:
            transfer_type_counts[f'{cl}->{sag_species_label}'] += 1

    return transfer_type_counts


def sort_hybridization_events_by_type(hybridization_table):
    # Initialize counts dict
    cluster_labels = ['A', 'B', 'C', 'O']
    transfer_type_counts = {}
    for cl1 in cluster_labels:
        non_cluster_labels = [label for label in cluster_labels if label not in [cl1, 'O']] # host can only be A, B, or C, but must be different from donor
        for cl2 in non_cluster_labels:
            transfer_type_counts[f'{cl1}->{cl2}'] = []

    for contig, contig_row in hybridization_table.iterrows():
        # Get cluster labels for all hybrid genes
        sag_species_label = contig_row['sag_species'][0]
        non_cluster_labels = [label for label in cluster_labels if label != sag_species_label]
        contig_cluster_labels = np.array(list(contig_row['cluster_labels_sequence']))
        filtered_cluster_labels = contig_cluster_labels[np.isin(contig_cluster_labels, non_cluster_labels)]
        print(contig, filtered_cluster_labels)

        if len(filtered_cluster_labels) != contig_row['num_hybrid_genes']:
            print(contig)
            print(contig_row)
            print(filtered_cluster_labels)
            print('\n')

        for cl in filtered_cluster_labels:
            #transfer_type_counts[f'{cl}->{sag_species_label}'] += 1
            continue

    return transfer_type_counts


def translate_transfer_label(transfer_label):
    label_dict = {'A':'$\\alpha$', 'B':'$\\beta$', 'C':'$\gamma$', 'O':'X'}
    donor, host = transfer_label.split('->')
    return f'{label_dict[donor]}->{label_dict[host]}'


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



def make_species_core_hybridization_table(pangenome_map, syn_homolog_map, species, hybridized_og_ids, og_hybridization_counts, args, og_type='og_id'):
    mixed_core_og_table = make_species_core_og_table(pangenome_map, 'M', min_core_presence=args.min_og_presence)
    species_core_og_table = make_species_core_og_table(pangenome_map, species, min_core_presence=args.min_og_presence)
    if og_type == 'og_id':
        species_core_og_ids = np.concatenate([list(species_core_og_table.index), list(mixed_core_og_table.index)])
    elif og_type == 'parent_og_id':
        species_core_og_ids = np.unique(np.concatenate([species_core_og_table['parent_og_id'].values, mixed_core_og_table['parent_og_id'].values]))
    species_core_hybridization_table = pd.DataFrame(index=species_core_og_ids, columns=['CYA_tag', 'CYB_tag', 'num_hybrid_cells'])

    for og_id in species_core_hybridization_table.index:
        if og_type == 'og_id':
            locus_tag = og_table.loc[og_id, 'locus_tag']
            orthologous_tag = syn_homolog_map.get_ortholog(locus_tag)
            if type(locus_tag) != str:
                species_core_hybridization_table.loc[og_id, 'num_hybrid_cells'] = 0
                continue

            if 'CYA' in locus_tag:
                species_core_hybridization_table.loc[og_id, 'CYA_tag'] = locus_tag
                if 'CYB' in orthologous_tag:
                    species_core_hybridization_table.loc[og_id, 'CYB_tag'] = orthologous_tag
            elif 'CYB' in locus_tag:
                species_core_hybridization_table.loc[og_id, 'CYB_tag'] = locus_tag
                if 'CYA' in orthologous_tag:
                    species_core_hybridization_table.loc[og_id, 'CYA_tag'] = orthologous_tag

        elif og_type == 'parent_og_id':
            locus_tags = og_table.loc[og_table['parent_og_id'] == og_id, 'locus_tag'].dropna().values
            if len(locus_tags) > 0:
                cya_tags = [tag for tag in locus_tags if 'CYA' in tag]
                cyb_tags = [tag for tag in locus_tags if 'CYB' in tag]

                if len(cya_tags) > 0:
                    species_core_hybridization_table.loc[og_id, 'CYA_tag'] = cya_tags[0]
                    if len(cyb_tags) == 0:
                        orthologous_tag = syn_homolog_map.get_ortholog(cya_tags[0])
                        if 'CYB' in orthologous_tag:
                            species_core_hybridization_table.loc[og_id, 'CYB_tag'] = orthologous_tag
                    else:
                        species_core_hybridization_table.loc[og_id, 'CYB_tag'] = cyb_tags[0]

                elif len(cyb_tags) > 0:
                    species_core_hybridization_table.loc[og_id, 'CYB_tag'] = cyb_tags[0]
                    if len(cya_tags) == 0:
                        orthologous_tag = syn_homolog_map.get_ortholog(cyb_tags[0])
                        if 'CYA' in orthologous_tag:
                            species_core_hybridization_table.loc[og_id, 'CYA_tag'] = orthologous_tag
                    else:
                        species_core_hybridization_table.loc[og_id, 'CYA_tag'] = cya_tags[0]

        if og_id in hybridized_og_ids:
            species_core_hybridization_table.loc[og_id, 'num_hybrid_cells'] = og_hybridization_counts[hybridized_og_ids == og_id][0]
        else:
            species_core_hybridization_table.loc[og_id, 'num_hybrid_cells'] = 0


    return species_core_hybridization_table


def make_species_core_og_table(pangenome_map, species, min_core_presence=0.2):
    og_table = pangenome_map.og_table
    if species == 'A':
        species_labels = ['A', 'a']
        species_og_table = og_table.loc[og_table['sequence_cluster'].isin(species_labels), :]
    elif species == 'Bp':
        species_labels = ['Bp', 'b']
        species_og_table = og_table.loc[og_table['sequence_cluster'].isin(species_labels), :]
    else:
        species_og_table = og_table.loc[og_table['sequence_cluster'] == species, :]

    # Get core gene cutoff
    metadata = MetadataMap()
    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
    num_species_cells = calculate_number_species_cells(species_sorted_sags)
    min_species_og_counts = min_core_presence * num_species_cells[species]

    return species_og_table.loc[species_og_table['num_seqs'] >= min_species_og_counts, :]

def calculate_number_species_cells(species_sorted_sags):
    num_species_cells = {}
    for species in ['A', 'Bp', 'C']:
        num_species_cells[species] = len(species_sorted_sags[species])
    num_species_cells['M'] = np.sum([num_species_cells[species] for species in num_species_cells]) # Add extra for total
    return num_species_cells


def make_hybridization_counts_table(hybridization_table, pangenome_map, metadata, min_og_frequency=0.2, og_type='parent_og_id'):
    og_table = pangenome_map.og_table
    og_ids = np.sort(og_table['parent_og_id'].dropna().unique())
    hybridization_counts_df = pd.DataFrame(index=og_ids, columns=['CYA_tag', 'CYB_tag', 'core_A', 'core_Bp', 'Bp->A', 'C->A', 'O->A', 'A->Bp', 'C->Bp', 'O->Bp', 'total_transfers'])

    # Add core OGs
    hybridization_counts_df = add_core_og_labels(hybridization_counts_df, og_table)

    # Add locus tags
    hybridization_counts_df = add_locus_tags(hybridization_counts_df, og_table)
    print(hybridization_table)
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
        #print(sog_ids, [idx for idx in og_table.loc[og_table['parent_og_id'] == og_id, :].index])
        if len(sog_ids) == 1:
            mismatches_subtable = gene_cluster_mismatches.loc[sog_ids[0], :].dropna()
            mismatched_sag_ids = mismatches_subtable.index[mismatches_subtable == True].values
            donor_cluster = og_table.loc[sog_ids[0], 'sequence_cluster']
            host_clusters = [metadata.get_sag_species(sag_id) for sag_id in mismatched_sag_ids]
            #print('Single cluster OG')
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


def is_valid_hybridization(transfer_type):
    valid_transfers = ['Bp->A', 'C->A', 'O->A', 'A->Bp', 'C->Bp', 'O->Bp', 'A->C', 'Bp->C', 'O->C']
    return (transfer_type in valid_transfers)

def translate_transfer_label(transfer_label):
    label_dict = {'A':'$\\alpha$', 'B':'$\\beta$', 'Bp':'$\\beta$', 'C':'$\gamma$', 'O':'X'}
    donor, host = transfer_label.split('->')
    return f'{label_dict[donor]}->{label_dict[host]}'


def make_species_painted_genome_tables(hybridization_table, pangenome_map, syn_homolog_map, metadata, min_og_frequency=0.2):
    species_cluster_genomes = initialize_sequence_cluster_haplotypes(pangenome_map, metadata, syn_homolog_map, min_og_frequency)
    species_cluster_genomes = assign_gene_species_clusters(species_cluster_genomes, pangenome_map, metadata)
    return species_cluster_genomes


def initialize_sequence_cluster_haplotypes(pangenome_map, metadata, syn_homolog_map, min_og_frequency):
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


def assign_gene_species_clusters(species_cluster_genomes, pangenome_map, metadata):
    og_table = pangenome_map.og_table
    gene_cluster_mismatches = find_gene_cell_cluster_mismatches(og_table, metadata)
    mismatches_idx = list(gene_cluster_mismatches.index)
    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sags = metadata.sort_sags(sag_ids, by='species')

    for i, og_id in enumerate(og_table['parent_og_id'].unique()):
        sog_ids = [idx for idx in og_table.loc[og_table['parent_og_id'] == og_id, :].index if idx in mismatches_idx]
        if len(sog_ids) == 1:
            mismatches_subtable = gene_cluster_mismatches.loc[sog_ids[0], :].dropna()
            mismatched_sag_ids = mismatches_subtable.index[mismatches_subtable == True].values
            donor_cluster = og_table.loc[sog_ids[0], 'sequence_cluster']
            species_cluster_genomes.loc[og_id, mismatched_sag_ids] = donor_cluster
            
            syna_matched_subtable = gene_cluster_mismatches.loc[sog_ids[0], species_sorted_sags['A']].dropna()
            #print(syna_matched_subtable)
            syna_matched_sag_ids = syna_matched_subtable.index[syna_matched_subtable == False].values
            species_cluster_genomes.loc[og_id, syna_matched_sag_ids] = 'A'
            synbp_matched_subtable = gene_cluster_mismatches.loc[sog_ids[0], species_sorted_sags['Bp']].dropna()
            synbp_matched_sag_ids = synbp_matched_subtable.index[synbp_matched_subtable == False].values
            species_cluster_genomes.loc[og_id, synbp_matched_sag_ids] = 'Bp'


        else:
            mismatches_subtable = gene_cluster_mismatches.loc[sog_ids, :].dropna(axis=1, how='all')
            mismatched_sag_ids = mismatches_subtable.columns[(mismatches_subtable == True).sum(axis=0) > 0].values
            donor_clusters = og_table.loc[sog_ids, 'sequence_cluster']
            for j, donor in enumerate(donor_clusters):
                if donor == 'a':
                    donor = 'A'
                elif donor == 'b':
                    donor = 'Bp'
                mismatches_subtable = gene_cluster_mismatches.loc[sog_ids[j], :].dropna()
                mismatched_sag_ids = mismatches_subtable.index[mismatches_subtable == True].values
                species_cluster_genomes.loc[og_id, mismatched_sag_ids] = donor

                syna_matched_subtable = gene_cluster_mismatches.loc[sog_ids[j], species_sorted_sags['A']].dropna()
                syna_matched_sag_ids = syna_matched_subtable.index[syna_matched_subtable == False].values
                species_cluster_genomes.loc[og_id, syna_matched_sag_ids] = 'A'
                synbp_matched_subtable = gene_cluster_mismatches.loc[sog_ids[j], species_sorted_sags['Bp']].dropna()
                synbp_matched_sag_ids = synbp_matched_subtable.index[synbp_matched_subtable == False].values
                species_cluster_genomes.loc[og_id, synbp_matched_sag_ids] = 'Bp'

                #matched_sag_ids = mismatches_subtable.index[mismatches_subtable == False].values
                #species_cluster_genomes.loc[og_id, matched_sag_ids] = donor

    return species_cluster_genomes


def annotate_haplotype_gene_clusters(species_cluster_genomes, pangenome_map, metadata, min_og_frequency, excluded_contigs=[]):
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
            print(sag_allele_contigs, filtered_idx)

            # Assign cluster label
            s_alleles = og_clusters[filtered_idx].values
            species_cluster_genomes.loc[o, s] = ','.join(np.sort(s_alleles))
            print(o, s, s_alleles)
            print('\n')

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

    print(num_multiple_alleles, num_filtered_multiple_alleles)

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

    species_cluster_genomes = initialize_sequence_cluster_haplotypes(pangenome_map, metadata, syn_homolog_map, args.min_og_presence)
    species_cluster_genomes = annotate_haplotype_gene_clusters(species_cluster_genomes, pangenome_map, metadata, args.min_og_presence, excluded_contigs=fully_hybrid_contigs)
    #species_cluster_genomes = annotate_haplotype_gene_clusters(species_cluster_genomes.iloc[:10], pangenome_map, metadata, args.min_og_presence)
    #species_cluster_genomes = annotate_haplotype_gene_clusters(species_cluster_genomes, pangenome_map, metadata, args.min_og_presence)
    species_cluster_genomes.to_csv(f'{args.output_dir}labeled_sequence_cluster_genomes.tsv', sep='\t')

    species_cluster_genomes = pd.read_csv(f'{args.output_dir}labeled_sequence_cluster_genomes.tsv', sep='\t', index_col=0)

    # Get SAG IDs sorted by species
    sag_ids = pangenome_map.get_sag_ids()
    species_sorted_sags = sort_sags_by_species(sag_ids, metadata)

    '''
    # Make test table
    test_genomes = species_cluster_genomes.fillna('')
    labels, label_counts = utils.sorted_unique(test_genomes[species_sorted_sags['A']].values)
    label_list = []
    for i in range(len(labels)):
        label_list += label_counts[i] * [labels[i]]
    label_list = np.random.permutation(label_list)
    L = test_genomes.shape[0]
    for i, s in enumerate(species_sorted_sags['A']):
        test_genomes.loc[:, s] = label_list[i * L:(i + 1) * L]
    test_genomes.loc[:, species_sorted_sags['Bp']] = 'Bp'

    print(labels, label_counts, np.sum(label_counts[3:]))
    print(counts_validation)
    print(counts_validation.iloc[:, 4:].sum(axis=0))
    '''
    
    counts_validation = count_gene_hybrids(species_cluster_genomes, species_sorted_sags)
    counts_validation.to_csv(f'{args.output_dir}hybridization_counts_table_validation.tsv', sep='\t')
    print(hybridization_counts)
    print(counts_validation)

    if args.verbose:
        print(species_cluster_genomes)
        print(counts_validation)
