import argparse
import subprocess
import glob
import pickle
import natsort
import numpy as np
import pandas as pd
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
from Bio import SeqIO
from Bio import AlignIO
from Bio.SeqRecord import SeqRecord


class PangenomeMap:
    '''
    Class for quick access to gene annotations and
    contig sequences from SAGs.
    '''
    def __init__(self, gff_files_dir=None, f_orthogroup_table=None, major_og_cluster_threshold=0.9):
        if gff_files_dir:
            self.gff_files = glob.glob(f'{gff_files_dir}/*.gff')
            self.read_gff_files()
        if f_orthogroup_table:
            self.og_table = pd.read_csv(f_orthogroup_table, sep='\t', index_col=0, low_memory=False)
            self.make_cell_gene_map()
        self.major_og_cluster_threshold = major_og_cluster_threshold

    def read_gff_files(self):
        annotations_list = []
        self.cell_contigs = {}
        self.contig_records = {}
        for fname in self.gff_files:
            annotations, contig_seqs = utils.read_annotation_file(fname, format='gff-fasta')
            annotations_list.append(annotations)
            self.contig_records = {**self.contig_records, **contig_seqs}
            cell_id = self.extract_gff_cell_id(fname)
            self.cell_contigs[cell_id] = sorted(list(contig_seqs.keys()))
        self.annotations = pd.concat(annotations_list)

    def extract_gff_cell_id(self, fname):
        return fname.split('/')[-1].replace('.gff', '')


    def make_cell_gene_map(self):
        sag_ids = [col for col in self.og_table.columns if 'Uncmic' in col]
        cell_gene_map = {}
        for sag_id in sag_ids:
            gene_ids = np.concatenate([gene_str.split(';') for gene_str in self.og_table[sag_id].dropna()])
            gene_prefixes = [gene_id.split('_')[0] for gene_id in gene_ids]
            unique_prefixes = np.unique(gene_prefixes)

            # Make sure all genes are correctly assigned
            assert len(unique_prefixes) == 1

            prefix = unique_prefixes[0]
            cell_gene_map[sag_id] = prefix
            cell_gene_map[prefix] = sag_id
        self.cell_gene_map = cell_gene_map


    def export_sag_protein_seqs(self, output_dir):
        for sag_id in self.cell_contigs:
            sag_contig_ids = self.cell_contigs[sag_id]

            sag_records = []
            cds_annotations = self.annotations.loc[self.annotations['type'] == 'CDS', :]
            cds_ids = list(cds_annotations.loc[cds_annotations['contig'].isin(sag_contig_ids), :].index)
            for cds_id in cds_ids:
                sag_records.append(self.extract_gene_record(cds_id, type='prot'))
            SeqIO.write(sag_records, f'{output_dir}{sag_id}.faa', 'fasta')

    def export_protein_seqs(self, output_file):
        seq_records = []
        cds_ids = list(self.annotations.loc[self.annotations['type'] == 'CDS', :].index)
        for cds_id in cds_ids:
            seq_records.append(self.extract_gene_record(cds_id, type='prot'))
        SeqIO.write(seq_records, output_file, 'fasta')

    def extract_gene_record(self, gene_id, type='nucl'):
        contig_id = self.annotations.loc[gene_id, 'contig']
        contig_seq = self.contig_records[contig_id]
        x_start, x_end = self.annotations.loc[gene_id, ['start_coord', 'end_coord']].astype(int)
        gene_seq = contig_seq.seq[x_start - 1:x_end]
        if self.annotations.loc[gene_id, 'strand'] == '-':
            gene_seq = gene_seq.reverse_complement()
        if type == 'prot':
            gene_seq = gene_seq.translate(table=11)
        gene_record = SeqRecord(gene_seq, id=gene_id, description='')
        return gene_record

    def export_rrna_seqs(self, output_dir):
        rrna_records = {'5S':[], '16S':[], '23S':[]}
        rrna_ids = list(self.annotations.loc[self.annotations['type'] == 'rRNA', :].index)
        for rrna_id in rrna_ids:
            product = self.get_gene_product(rrna_id)
            rrna_type = product.split(' ')[0]
            rrna_records[rrna_type].append(self.extract_gene_record(rrna_id, type='nucl'))

        for rrna_type in rrna_records:
            SeqIO.write(rrna_records[rrna_type], f'{output_dir}{rrna_type}_rRNA.fna', 'fasta')

    def get_gene_product(self, gene_id):
        attr = self.annotations.loc[gene_id, 'attributes']
        attr_dict = utils.make_gff_attribute_dict(attr)
        if 'product' in attr_dict:
            product = attr_dict['product']
        else:
            product = None
        return product

    def export_cell_protein_seqs(self, output_dir):
        for cell_id in self.cell_contigs:
            contig_ids = self.cell_contigs[cell_id]
            cds_ids = list(self.annotations.loc[self.annotations['contig'].isin(contig_ids).values * (self.annotations['type'] == 'CDS').values, :].index)

            seq_records = []
            for cds_id in cds_ids:
                seq_records.append(self.extract_gene_record(cds_id, type='prot'))
            SeqIO.write(seq_records, f'{output_dir}{cell_id}_CDS.faa', 'fasta')

    def get_gene_id_map(self):
        gene_id_map = {}
        for cell_id in self.cell_contigs:
            for contig_id in self.cell_contigs[cell_id]:
                contig_gene_ids = list(self.annotations.index[self.annotations['contig'] == contig_id])
                for gene_id in contig_gene_ids:
                    gene_id_map[gene_id] = cell_id
        return gene_id_map

    def get_single_copy_ogs(self, max_copies_per_cell=1.05):
        return list(self.og_table.loc[self.og_table['seqs_per_cell'] <= max_copies_per_cell, :].index)

    def get_high_confidence_og_table(self, high_presence_threshold=100, max_copies_per_cell=1.1, subset_sag_ids=None):
        scog_ids = self.get_single_copy_ogs(max_copies_per_cell=max_copies_per_cell)
        scog_table = self.og_table.reindex(index=scog_ids)

        if subset_sag_ids is not None:
            data_columns = [col for col in scog_table if 'Uncmic' not in col]
            new_columns = data_columns + list(subset_sag_ids)
            scog_table = scog_table.reindex(columns=new_columns)

            # Update OG statistics
            scog_table['num_cells'] = scog_table[subset_sag_ids].notna().sum(axis=1).values
            scog_table = scog_table.reindex(index=list(scog_table.loc[scog_table['num_cells'] > 0, :].index))
            for og in scog_table.index:
                num_seqs_per_sag = [len(gene_ids_str.split(';')) for gene_ids_str in scog_table.loc[og, subset_sag_ids].dropna()]
                scog_table.loc[og, 'num_seqs'] = np.sum(num_seqs_per_sag)
                scog_table.loc[og, 'seqs_per_cell'] = np.sum(num_seqs_per_sag) / scog_table.at[og, 'num_cells']

                gene_lengths = np.concatenate([[int(gene_str.split('_')[-1]) - int(gene_str.split('_')[-2]) + 1 for gene_str in gene_ids_str.split(';')] for gene_ids_str in scog_table.loc[og, subset_sag_ids].dropna()])
                scog_table.loc[og, 'avg_length'] = np.mean(gene_lengths)

        return scog_table.loc[scog_table['num_cells'] >= high_presence_threshold, :]

    def get_gene_sag_id(self, gene_id):
        prefix = gene_id.split('_')[0]
        if prefix in self.cell_gene_map:
            return self.cell_gene_map[prefix]
        else:
            return gene_id

    def read_orthogroup_mapping(self, f_map):
        self.orthogroup_map = pickle.load(open(f_map, 'rb'))

    def get_mapped_og_id(self, og_id):
        if og_id in self.orthogroup_map:
            mapped_id = self.orthogroup_map[og_id]
        else:
            mapped_id = None
        return mapped_id

    def make_gene_id_to_orthogroup_map(self):
        gene_id_map = {}
        sag_ids = [col for col in self.og_table.columns if 'Uncmic' in col]
        for sid in sag_ids:
            sag_genes = self.og_table[sid].dropna()
            for oid, gene_ids_str in sag_genes.iteritems():
                gene_ids = gene_ids_str.split(';')
                for gid in gene_ids:
                    gene_id_map[gid] = oid
        return gene_id_map 

    def make_og_clusters_dict(self, max_num_subclusters=np.inf):
        sog_arr = self.og_table['og_id'].values
        parent_ogs = [sog.split('-')[0] for sog in sog_arr]
        unique_parent_ogs, parent_og_counts = utils.sorted_unique(parent_ogs)

        og_clusters_dict = {}
        for parent_og in unique_parent_ogs:
            sog_list = [sog for sog in sog_arr if parent_og in sog]
            if len(sog_list) <= max_num_subclusters:
                og_clusters_dict[parent_og] = sog_list
        return og_clusters_dict

    def add_parent_og_ids(self):
        og_table = self.og_table.copy()
        data_columns = [col for col in og_table if 'Uncmic' not in col]

        if 'og_id' in data_columns:
            og_table['parent_og_id'] = og_table['og_id'].str.split('-').str[0]

            # Reorder columns
            sag_columns = [col for col in og_table if 'Uncmic' in col]
            #data_columns.append('parent_og_id')
            data_columns = data_columns[:2] + ['parent_og_id'] + data_columns[2:]
            new_columns = data_columns + sag_columns
            self.og_table = og_table.reindex(columns=new_columns)
        else:
            print(f'"og_id" column not found. Cannot add parent OG IDs.')

    def bin_ogs_by_species_composition(self, metadata, parent_og_ids=None, label_dict={'A':'A', 'Bp':'Bp', 'C':'C', 'O':'O', 'M':'M'}, reorder_columns=True):
        og_table = self.og_table
        og_table = og_table.reindex(index=np.sort(np.unique(og_table.index)))
        sag_ids = [col for col in og_table if 'Uncmic' in col]

        if parent_og_ids is None:
            parent_og_ids = np.unique(og_table['parent_og_id'].values)

        # Get species counts for each SOG
        self.count_og_species_composition(metadata)

        for pid in parent_og_ids:
            og_num_seqs = og_table.loc[og_table['parent_og_id'] == pid, 'num_seqs'].sort_values(ascending=False)

            if len(og_num_seqs) == 1:
                sid = og_num_seqs.index[0]
                og_table.loc[sid, 'sequence_cluster'] = self.label_major_og_cluster(sid, label_dict=label_dict)
            else:
                labeled_major_clusters = {'A':False, 'Bp':False}
                for i, sid in enumerate(og_num_seqs.index):
                    are_major_clusters_labeled = labeled_major_clusters['A'] & labeled_major_clusters['Bp']
                    if are_major_clusters_labeled == False:
                        cluster_label = self.label_major_og_cluster(sid, label_dict=label_dict)
                        og_table.loc[sid, 'sequence_cluster'] = cluster_label
                        labeled_major_clusters[cluster_label] = True
                    else:
                        cluster_label = self.label_minor_og_cluster(sid, label_dict=label_dict)
                        og_table.loc[sid, 'sequence_cluster'] = cluster_label

        # Reorder columns and return table
        data_columns = [col for col in og_table if 'Uncmic' not in col]
        sag_columns = [col for col in og_table if 'Uncmic' in col]
        if reorder_columns:
            new_columns = data_columns[:3] + [data_columns[-1]] + data_columns[3:-1] + sag_columns
        else:
            new_columns = data_columns + sag_columns

        return og_table.reindex(columns=new_columns)

    def count_og_species_composition(self, metadata):
        og_ids = list(self.og_table.index)
        sag_ids = [col for col in self.og_table if 'Uncmic' in col]
        species_sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')
        species_labels = ['A', 'Bp', 'C']
        species_counts_df = pd.DataFrame(index=og_ids, columns=species_labels)
        for i, species in enumerate(species_labels):
            species_sag_ids = species_sorted_sag_ids[species]
            for sid in og_ids:
                species_gene_ids = read_gene_ids(self.og_table.loc[sid, species_sorted_sag_ids[species]], drop_none=True)
                species_counts_df.loc[sid, species] = len(species_gene_ids)
        self.og_species_counts_table = species_counts_df

    def label_major_og_cluster(self, sog_id, label_dict={'A':'A', 'Bp':'Bp', 'M':'M'}):
        og_counts = self.og_species_counts_table.loc[sog_id, :]
        f_A = og_counts['A'] / og_counts.sum()
        f_Bp = og_counts['Bp'] / og_counts.sum()

        is_A_cluster = (f_A > self.major_og_cluster_threshold) & (f_Bp < self.major_og_cluster_threshold) & (og_counts['C'] < 1)
        is_Bp_cluster = (f_A < self.major_og_cluster_threshold) & (f_Bp > self.major_og_cluster_threshold) & (og_counts['C'] < 1)
        if is_A_cluster:
            #return 'A'
            return label_dict['A']
        elif is_Bp_cluster:
            #return 'Bp'
            return label_dict['Bp']
        else:
            #return 'M'
            return label_dict['M']

    def label_minor_og_cluster(self, sog_id, label_dict={'C':'C', 'O':'O'}):
        og_counts = self.og_species_counts_table.loc[sog_id, :]
        if og_counts['C'] > 0:
            #return 'C'
            return label_dict['C'] 
        else:
            #return 'O'
            return label_dict['O'] 

    def label_contig_sequence_clusters(self, output_file=None):
        og_table = self.og_table
        gene_annotations = self.annotations
        unique_contigs = np.sort(np.unique(self.annotations['contig'].values))
        contig_seqs_df = pd.DataFrame(index=unique_contigs, columns=['cluster_labels_sequence'])

        for cid in unique_contigs:
            sag_prefix, _ = cid.split('_')
            if sag_prefix in self.cell_gene_map:
                sag_id = self.cell_gene_map[sag_prefix]
                print(cid, sag_id)
            else:
                print(f'SAG ID for contig {cid} not found!')
                continue

            contig_genes = list(gene_annotations.loc[gene_annotations['contig'] == cid, :].index)
            contig_seq = []
            for gid in contig_genes:
                cluster_label = og_table.loc[og_table[sag_id] == gid, 'sequence_cluster'].dropna().values
                if len(cluster_label) > 0:
                    contig_seq.append(cluster_label[0][0])
                else:
                    contig_seq.append('N')
            contig_seqs_df.loc[cid, 'cluster_labels_sequence'] = ''.join(contig_seq)

        if output_file is not None:
            contig_seqs_df.to_csv(output_file, sep='\t')
        else:
            return contig_seqs_df

    def get_sag_ids(self):
        return np.array([col for col in self.og_table.columns if 'Uncmic' in col])

    def get_og_contig_location(self, og_id=None, gene_id=None, sag_id=None):
        if og_id is not None:
            gene_id = self.og_table.loc[og_id, sag_id]

        if gene_id is not None:
            loc_str = gene_id.split('_')[-2:]
            loc = [int(x) for x in loc_str]
        else:
            loc = None
        return loc

    def read_pairwise_divergence_results(self, results_files, key_type='og'):
        # Note only key_type='og' is implemented; assumes pdist matrices are at the OG level
        self.make_divergence_results_file_map(results_files) # avoid loading large number of matrices into memory
        self.add_divergence_key_to_og_table(key_type)

    def make_divergence_results_file_map(self, results_files):
        file_dict = {}
        key_dict = {}
        for f_results in results_files:
            pdist_dict = pickle.load(open(f_results, 'rb'))
            key_ids = np.array(list(pdist_dict.keys()))
            key_dict[f_results] = key_ids
            for key in key_ids:
                file_dict[key] = f_results
        self._divergence_file_map = file_dict
        self._divergence_key_map = key_dict

    def add_divergence_key_to_og_table(self, key_type):
        if key_type == 'og':
            key_labels = [p if p in self._divergence_file_map else None for p in self.og_table['parent_og_id']]
            self.og_table['pdist_key'] = key_labels

    def get_sog_pairwise_divergences(self, sog_id, sag_ids=None):
        # Find file with pdist matrix
        og_id = self.og_table.loc[sog_id, 'parent_og_id']
        if og_id in self._divergence_file_map:
            f_results = self._divergence_file_map[og_id]
            results_dict = pickle.load(open(f_results, 'rb'))
            pdist_df = results_dict[og_id]
            
            # Get SAGs subset
            if sag_ids is not None:
                gene_ids = self.get_sog_gene_ids(sog_id, sag_ids)
                filtered_gene_ids = [g for g in gene_ids if g in list(pdist_df.index)]
                pdist_df = pdist_df.loc[filtered_gene_ids, filtered_gene_ids]
        else:
            pdist_df = None
        return pdist_df

    def get_sog_gene_ids(self, sog_id, sag_ids=None):
        if sag_ids is None:
            sag_ids = self.get_sag_ids()
        return read_gene_ids(self.og_table.loc[sog_id, sag_ids], drop_none=True)

    def get_og_gene_ids(self, og_id, sag_ids=None):
        if sag_ids is None:
            sag_ids = self.get_sag_ids()
        sog_ids = np.array(self.og_table.loc[self.og_table['parent_og_id'] == og_id, :].index)
        return np.concatenate([read_gene_ids(self.og_table.loc[sog_id, sag_ids], drop_none=True) for sog_id in sog_ids])

    def calculate_mean_pairwise_divergence(self, input_og_ids, sag_ids):
        # Get divergence results files
        results_files = np.unique([self._divergence_file_map[og_id] for og_id in input_og_ids if og_id in self._divergence_file_map])
        results_dict = {}
        for f_results in results_files:
            divergences_matrices_dict = pickle.load(open(f_results, 'rb'))
            file_og_ids = [og_id for og_id in self._divergence_key_map[f_results] if og_id in input_og_ids]
            for og_id in file_og_ids:
                pdist_df = divergences_matrices_dict[og_id]

                # Filter gene IDs
                sog_idx = np.array(self.og_table.loc[self.og_table['parent_og_id'] == og_id, :].index)
                og_gene_ids = np.concatenate([self.get_sog_gene_ids(sog_id, sag_ids) for sog_id in sog_idx])
                filtered_gene_ids = [g for g in og_gene_ids if g in list(pdist_df.index)]
                pdist_df = pdist_df.loc[filtered_gene_ids, filtered_gene_ids]
                results_dict[og_id] = np.mean(utils.get_matrix_triangle_values(pdist_df.values, k=1))

        return np.array([results_dict[oid] for oid in input_og_ids if oid in results_dict]), np.array([oid for oid in input_og_ids if oid in results_dict])

    def construct_pairwise_divergence_across_ogs(self, input_og_ids, sag_ids):
        # Initialize matrix
        pdist_matrix = np.zeros((len(sag_ids), len(sag_ids), len(input_og_ids)))
        pdist_matrix[:] = np.nan

        x_idx = np.array(sag_ids)
        y_idx = np.array(sag_ids)
        z_idx = np.array(input_og_ids)

        # Get divergence results files
        results_files = np.unique([self._divergence_file_map[og_id] for og_id in input_og_ids if og_id in self._divergence_file_map])
        results_dict = {}
        for f_results in results_files:
            divergences_matrices_dict = pickle.load(open(f_results, 'rb'))
            file_og_ids = [og_id for og_id in self._divergence_key_map[f_results] if og_id in input_og_ids]
            for og_id in file_og_ids:
                # Map gene IDs to SAG IDs
                pdist_df = divergences_matrices_dict[og_id]
                gene_ids = pdist_df.index.values
                reindex_dict = {}
                covered_sag_ids = []
                for g in gene_ids:
                    sag_id = self.get_gene_sag_id(g)
                    if sag_id not in covered_sag_ids:
                        # If multiple gene copies in the same cell arbitrarily chose first one
                        reindex_dict[g] = sag_id
                        covered_sag_ids.append(sag_id)
                    else:
                        pdist_df.drop(g, axis='index', inplace=True)
                        pdist_df.drop(g, axis='columns', inplace=True)
                pdist_df.rename(index=reindex_dict, inplace=True)
                pdist_df.rename(columns=reindex_dict, inplace=True)

                # Copy values
                filter_idx = np.arange(len(x_idx))[np.isin(x_idx, pdist_df.index.values)]
                filtered_sag_ids = x_idx[filter_idx]
                og_idx = np.arange(len(z_idx))[z_idx == og_id][0]

                for i in filter_idx:
                    pdist_matrix[filter_idx, i, og_idx] = pdist_df.loc[filtered_sag_ids, sag_ids[i]].copy().values

        return pdist_matrix

    def get_og_pairwise_divergence_matrices(self, input_og_ids, sag_ids):
        # Get divergence results files
        results_files = np.unique([self._divergence_file_map[og_id] for og_id in input_og_ids if og_id in self._divergence_file_map])
        results_dict = {}
        for f_results in results_files:
            divergences_matrices_dict = pickle.load(open(f_results, 'rb'))
            file_og_ids = [og_id for og_id in self._divergence_key_map[f_results] if og_id in input_og_ids]
            for og_id in file_og_ids:
                pdist_df = divergences_matrices_dict[og_id]

                # Filter gene IDs
                sog_idx = np.array(self.og_table.loc[self.og_table['parent_og_id'] == og_id, :].index)
                og_gene_ids = np.concatenate([self.get_sog_gene_ids(sog_id, sag_ids) for sog_id in sog_idx])
                filtered_gene_ids = [g for g in og_gene_ids if g in list(pdist_df.index)]
                pdist_df = pdist_df.loc[filtered_gene_ids, filtered_gene_ids]
                results_dict[og_id] = pdist_df

        return results_dict

    def get_sags_pairwise_divergences(self, sag_ids, input_og_ids=None):
        # Get divergence results files
        results_dict = {}
        if input_og_ids is None:
            divergence_files = list(self._divergence_key_map.keys())
            print(divergence_files)
            #for f_results in divergence_files:
            #    divergences_matrices_dict = pickle.load(open(f_results, 'rb'))

        else:
            results_files = np.unique([self._divergence_file_map[og_id] for og_id in input_og_ids if og_id in self._divergence_file_map])
            for f_results in results_files:
                divergences_matrices_dict = pickle.load(open(f_results, 'rb'))
                file_og_ids = [og_id for og_id in self._divergence_key_map[f_results] if og_id in input_og_ids]
                for og_id in file_og_ids:
                    pdist_df = divergences_matrices_dict[og_id]

                    # Map gene IDs to SAG IDs
                    mapped_gene_ids = [self.get_gene_sag_id(gene_id) for gene_id in pdist_df.index]
                    if len(np.unique(mapped_gene_ids)) < len(mapped_gene_ids):
                        # Skip OGs where requested SAG IDs have duplicates
                        #   Simplest way to avoid comparing paralogs
                        continue
                    pdist_df.index = mapped_gene_ids
                    pdist_df.columns = mapped_gene_ids

                    # Save results
                    mapped_pdist_df = pdist_df.reindex(index=sag_ids, columns=sag_ids)
                    results_dict[og_id] = mapped_pdist_df.values

            #divergence_matrix = np.concatenate([results_dict[og_id] for og_id in input_og_ids if og_id in results_dict])
            #ouput_og_ids = [og_id for og_id in input_og_ids if og_id in results_dict]

        #return divergence_matrix, output_og_ids
        return results_dict


    def calculate_mean_divergence_between_groups(self, input_og_ids, sag_ids1, sag_ids2):
        # Get divergence results files
        results_files = np.unique([self._divergence_file_map[og_id] for og_id in input_og_ids if og_id in self._divergence_file_map])
        results_dict = {}
        for f_results in results_files:
            divergences_matrices_dict = pickle.load(open(f_results, 'rb'))
            file_og_ids = [og_id for og_id in self._divergence_key_map[f_results] if og_id in input_og_ids]
            for og_id in file_og_ids:
                pdist_df = divergences_matrices_dict[og_id]

                # Filter gene IDs
                sog_idx = np.array(self.og_table.loc[self.og_table['parent_og_id'] == og_id, :].index)
                group1_gene_ids = np.concatenate([self.get_sog_gene_ids(sog_id, sag_ids1) for sog_id in sog_idx])
                filtered_group1_gene_ids = [g for g in group1_gene_ids if g in list(pdist_df.index)]
                group2_gene_ids = np.concatenate([self.get_sog_gene_ids(sog_id, sag_ids2) for sog_id in sog_idx])
                filtered_group2_gene_ids = [g for g in group2_gene_ids if g in list(pdist_df.index)]
                results_dict[og_id] = np.mean(pdist_df.loc[filtered_group1_gene_ids, filtered_group2_gene_ids].values)

        return np.array([results_dict[oid] for oid in input_og_ids if oid in results_dict]), np.array([oid for oid in input_og_ids if oid in results_dict])

    def calculate_group_divergence(self, input_og_ids, sag_ids1, sag_ids2, metric='mean'):
        # Get divergence results files
        results_files = np.unique([self._divergence_file_map[og_id] for og_id in input_og_ids if og_id in self._divergence_file_map])
        results_dict = {}
        for f_results in results_files:
            divergences_matrices_dict = pickle.load(open(f_results, 'rb'))
            file_og_ids = [og_id for og_id in self._divergence_key_map[f_results] if og_id in input_og_ids]
            for og_id in file_og_ids:
                pdist_df = divergences_matrices_dict[og_id]

                # Filter gene IDs
                sog_idx = np.array(self.og_table.loc[self.og_table['parent_og_id'] == og_id, :].index)
                group1_gene_ids = np.concatenate([self.get_sog_gene_ids(sog_id, sag_ids1) for sog_id in sog_idx])
                filtered_group1_gene_ids = [g for g in group1_gene_ids if g in list(pdist_df.index)]
                group2_gene_ids = np.concatenate([self.get_sog_gene_ids(sog_id, sag_ids2) for sog_id in sog_idx])
                filtered_group2_gene_ids = [g for g in group2_gene_ids if g in list(pdist_df.index)]

                if metric == 'mean':
                    results_dict[og_id] = np.mean(pdist_df.loc[filtered_group1_gene_ids, filtered_group2_gene_ids].values)
                    pdist_values = pdist_df.loc[filtered_group1_gene_ids, filtered_group2_gene_ids].values
                elif metric == 'median':
                    results_dict[og_id] = np.median(pdist_df.loc[filtered_group1_gene_ids, filtered_group2_gene_ids].values)
                elif metric == 'min':
                    results_dict[og_id] = np.min(pdist_df.loc[filtered_group1_gene_ids, filtered_group2_gene_ids].values)
                elif metric == 'max':
                    results_dict[og_id] = np.max(pdist_df.loc[filtered_group1_gene_ids, filtered_group2_gene_ids].values)
                elif metric == 'std':
                    results_dict[og_id] = np.std(pdist_df.loc[filtered_group1_gene_ids, filtered_group2_gene_ids].values)

        return np.array([results_dict[oid] for oid in input_og_ids if oid in results_dict]), np.array([oid for oid in input_og_ids if oid in results_dict])


    def get_core_mixed_species_og_ids(self, presence_fraction_cutoff=0.15):
        '''
        Returns core mixed species orthogroup IDs. Uses simple cutoff in total OG presence.
        '''
        og_table = self.og_table
        sag_ids = self.get_sag_ids()
        mixed_cluster_ogs = np.array(og_table.loc[og_table['sequence_cluster'] == 'M', :].index)
        presence_cutoff = int(presence_fraction_cutoff * len(sag_ids))
        core_mixed_ogs = mixed_cluster_ogs[(og_table.loc[mixed_cluster_ogs, 'num_cells'] > presence_cutoff).values]
        return core_mixed_ogs

    def get_species_core_og_ids(self, species, metadata, include_mixed_orthogroups=True, min_og_frequency=0.2, og_type='og_id', output_type='array'):
        '''
        Replaces different versions from analyze_hybridization_events.py, make_orthogroup_diversity_table.py, etc.
        '''

        og_table = self.og_table
        sag_ids = self.get_sag_ids()
        species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
        num_species_cells = self.calculate_number_species_cells(species_sorted_sags)

        #core_og_ids = []
        core_og_dict = {}
        for species in ['A', 'Bp', 'M']:
            if species == 'A':
                species_labels = ['A', 'a']
                species_og_table = og_table.loc[og_table['sequence_cluster'].isin(species_labels), :]
            elif species == 'Bp':
                species_labels = ['Bp', 'b']
                species_og_table = og_table.loc[og_table['sequence_cluster'].isin(species_labels), :]
            else:
                species_og_table = og_table.loc[og_table['sequence_cluster'] == species, :]

            min_species_og_presence = min_og_frequency * num_species_cells[species]
            species_og_table = species_og_table.loc[species_og_table['num_seqs'] >= min_species_og_presence, :]

            if og_type == 'og_id':
                #core_og_ids.append(species_og_table['og_id'].values)
                core_og_dict[species] = species_og_table['og_id'].values
            elif og_type == 'og_table_index':
                #core_og_ids.append(list(species_og_table.index))
                core_og_dict[species] = np.array(species_og_table.index)
            elif og_type == 'parent_og_id':
                #core_og_ids.append(species_og_table['parent_og_id'].values)
                #core_og_ids.append(species_og_table['parent_og_id'].unique())
                core_og_dict[species] = species_og_table['parent_og_id'].unique()

        #old_results = np.unique(np.concatenate(core_og_ids))
        #print(np.sum(np.isin(core_og_dict_values, old_results)), len(core_og_dict_values))
        #print(np.sum(np.isin(old_results, core_og_dict_values)), len(old_results))

        if output_type == 'array':
            return np.unique(np.concatenate(list(core_og_dict.values())))
        elif output_type == 'dict':
            return core_og_dict
        else:
            print(f'ERROR! Unrecognized output type PangenomeMap.get_core_og_ids()!')


    def get_core_og_ids(self, metadata, min_og_frequency=0.2, og_type='og_id', output_type='array'):
        '''
        Replaces different versions from analyze_hybridization_events.py, make_orthogroup_diversity_table.py, etc.
        '''

        og_table = self.og_table
        sag_ids = self.get_sag_ids()
        species_sorted_sags = metadata.sort_sags(sag_ids, by='species')
        num_species_cells = self.calculate_number_species_cells(species_sorted_sags)

        #core_og_ids = []
        core_og_dict = {}
        for species in ['A', 'Bp', 'M']:
            if species == 'A':
                species_labels = ['A', 'a']
                species_og_table = og_table.loc[og_table['sequence_cluster'].isin(species_labels), :]
            elif species == 'Bp':
                species_labels = ['Bp', 'b']
                species_og_table = og_table.loc[og_table['sequence_cluster'].isin(species_labels), :]
            else:
                species_og_table = og_table.loc[og_table['sequence_cluster'] == species, :]

            min_species_og_presence = min_og_frequency * num_species_cells[species]
            species_og_table = species_og_table.loc[species_og_table['num_seqs'] >= min_species_og_presence, :]

            if og_type == 'og_id':
                #core_og_ids.append(species_og_table['og_id'].values)
                core_og_dict[species] = species_og_table['og_id'].values
            elif og_type == 'og_table_index':
                #core_og_ids.append(list(species_og_table.index))
                core_og_dict[species] = np.array(species_og_table.index)
            elif og_type == 'parent_og_id':
                #core_og_ids.append(species_og_table['parent_og_id'].values)
                #core_og_ids.append(species_og_table['parent_og_id'].unique())
                core_og_dict[species] = species_og_table['parent_og_id'].unique()

        #old_results = np.unique(np.concatenate(core_og_ids))
        #print(np.sum(np.isin(core_og_dict_values, old_results)), len(core_og_dict_values))
        #print(np.sum(np.isin(old_results, core_og_dict_values)), len(old_results))

        if output_type == 'array':
            return np.unique(np.concatenate(list(core_og_dict.values())))
        elif output_type == 'dict':
            return core_og_dict
        else:
            print(f'ERROR! Unrecognized output type PangenomeMap.get_core_og_ids()!')

    def calculate_number_species_cells(self, species_sorted_sags):
        num_species_cells = {}
        for species in ['A', 'Bp', 'C']:
            num_species_cells[species] = len(species_sorted_sags[species])
        num_species_cells['M'] = np.sum([num_species_cells[species] for species in num_species_cells]) # Add extra for total
        return num_species_cells

    '''
    def get_og_ref_location(self, og_id, syn_homolog_map, ref_prefix='CYA', id_type='og_id', output_type='locus_tag'):
        if id_type == 'og_id':
            og_locus_tags = self.og_table.loc[self.og_table['parent_og_id'] == og_id, 'locus_tag'].dropna().values
            if len(og_locus_tags) > 0:
                has_ref_tag = [ref_prefix in locus_tag for locus_tag in og_locus_tags]
                if np.sum(has_ref_tag) > 0:
                    ref_tag = og_locus_tags[has_ref_tag][0]
                else:
                    # Choose random tag from other reference genome
                    alt_ref_tag = og_locus_tags[0]
    '''

    def read_species_og_alignment(self, og_id, species_sag_ids, aln_dir='../results/single-cell/sscs_pangenome/_aln_results/'):
        # Read alignment
        f_aln = f'{aln_dir}{og_id}_aln.fna'
        aln = seq_utils.read_alignment(f_aln)

        # Get species gene IDs
        filtered_gene_ids = self.get_og_gene_ids(og_id, sag_ids=species_sag_ids)

        return align_utils.get_subsample_alignment(aln, filtered_gene_ids)

    def read_sags_og_alignment(self, f_aln, og_id, sag_ids):
        aln = seq_utils.read_alignment(f_aln)
        filtered_gene_ids = self.get_og_gene_ids(og_id, sag_ids=sag_ids)
        return align_utils.get_subsample_alignment(aln, filtered_gene_ids)

    def get_species_cluster_gene_ids(self, og_id, species):
        if species == 'A':
            species_sequence_clusters = ['A', 'a']
        elif species == 'Bp':
            species_sequence_clusters = ['Bp', 'b']
        else:
            species_sequence_clusters = [species]

        og_subtable = self.og_table.loc[self.og_table['parent_og_id'] == og_id, :]

        sag_ids = self.get_sag_ids()
        gene_ids = []
        for idx, row in og_subtable.loc[og_subtable['sequence_cluster'].isin(species_sequence_clusters), sag_ids].iterrows():
            gene_ids += list(row.dropna())

        return gene_ids


def read_orthogroup_table(f_table):
    return pd.read_csv(f_table, sep='\t', index_col=0, low_memory=False)

def read_roary_clusters(f_in):
    cluster_dict = {}
    with open(f_in, 'r') as handle:
        for line in handle.readlines():
            head, tail = line.split(':')
            cluster_id = head.strip()
            gene_ids = tail.strip().split('\t')
            cluster_dict[cluster_id] = gene_ids
    return cluster_dict

def read_roary_gene_presence(f_in):
    gene_presence_absence = pd.read_csv(f_in, sep=',', index_col=0)
    return gene_presence_absence

def make_locus_map(gene_clusters, gene_id_map, sag_ids):
    cluster_ids = natsort.natsorted(list(gene_clusters.keys()))
    locus_map = pd.DataFrame(index=cluster_ids, columns=sag_ids)
    for cluster_id in cluster_ids:
        locus_map.loc[cluster_id, :] = [[] for sag_id in sag_ids]
        for gene_id in gene_clusters[cluster_id]:
            if gene_id in gene_id_map:
                locus_map.at[cluster_id, get_gene_sag_id(gene_id, gene_id_map)].append(gene_id)
    return locus_map

def make_sag_gene_id_map(data_tables):
    gene_id_map = {'CYA':'CP000239', 'CYB':'CP000240'}
    for sag_id in data_tables:
        gene_table = data_tables[sag_id]['genes']
        for gene_id in gene_table.index:
            gene_id_map[gene_id] = sag_id
        id_stem = gene_id.split('_')[0]
        gene_id_map[id_stem] = sag_id
    return gene_id_map

def get_gene_sag_id(gene_id, gene_id_map):
    stem = gene_id.split('_')[0]
    return gene_id_map[stem]

def calculate_gene_copy_numbers(locus_map):
    gene_copy_numbers = pd.DataFrame(index=locus_map.index, columns=locus_map.columns)
    for sag_id in gene_copy_numbers.columns:
        gene_copy_numbers[sag_id] = locus_map[sag_id].str.len()
    return gene_copy_numbers

def update_og_table_stats(og_table):
    #sag_columns = list(og_table.columns[4:])
    sag_columns = [col for col in og_table.columns if 'Uncmic' in col]
    for og_id in og_table.index:
        gene_ids = read_gene_ids(og_table.loc[og_id, sag_columns], drop_none=True)
        og_table.loc[og_id, 'num_seqs'] = len(gene_ids)
        sag_grouped_gene_ids = read_gene_ids(og_table.loc[og_id, sag_columns], drop_none=True, group_by_sag=True)
        num_cells = len(sag_grouped_gene_ids)
        og_table.loc[og_id, 'num_cells'] = num_cells
        og_table.loc[og_id, 'seqs_per_cell'] = len(gene_ids) / (num_cells + (num_cells == 0))
        og_table.loc[og_id, 'avg_length'] = calculate_mean_gene_length(gene_ids)
    return og_table

def read_gene_ids(table_row, group_by_sag=False, drop_none=False):
    gene_ids = []
    is_null = table_row.isnull().values
    for i, sag_entry in enumerate(table_row):
        if is_null[i]:
            gene_ids.append(None)
        elif group_by_sag:
            gene_ids.append(sag_entry.split(';'))
        else:
            gene_ids += sag_entry.split(';') 
    if drop_none:
        return [gene_id for gene_id in gene_ids if gene_id is not None]
    else:
        return gene_ids

def make_og_table_row(subcluster_id, gene_ids, og_table):
    sag_grouped_gene_ids = group_og_table_genes(gene_ids, og_table)
    
    row_df = pd.Series(index=og_table.columns)
    row_df['num_seqs'] = len(gene_ids)
    row_df['num_cells'] = len(sag_grouped_gene_ids)
    row_df['seqs_per_cell'] = len(gene_ids) / len(sag_grouped_gene_ids)
    row_df['avg_length'] = calculate_mean_gene_length(gene_ids)
    for sag_id in sag_grouped_gene_ids:
        row_df[sag_id] = ';'.join(sag_grouped_gene_ids[sag_id])
    return {subcluster_id:row_df}

def make_subcluster_og_table_row(subcluster_id, gene_ids, og_table):
    table_row = []
    table_row.append(subcluster_id)
    table_row.append(len(gene_ids))
    sag_grouped_gene_ids = group_og_table_genes(gene_ids, og_table)
    table_row.append(len(sag_grouped_gene_ids))
    table_row.append(len(gene_ids) / len(sag_grouped_gene_ids))
    table_row.append(calculate_mean_gene_length(gene_ids))

    sag_presence = {}
    for sag_id in sag_grouped_gene_ids:
        sag_presence[sag_id] = ';'.join(sag_grouped_gene_ids[sag_id])
    table_row.append(sag_presence)
    return table_row

def calculate_mean_gene_length(gene_ids):
    gene_lengths = get_gene_lengths(gene_ids)
    return np.mean(gene_lengths)

def get_gene_lengths(gene_ids):
    gene_lengths = []
    for gene_id in gene_ids:
        contig_id, loc = seq_utils.split_gene_id(gene_id)
        gene_lengths.append(loc[1] - loc[0] + 1)
    return gene_lengths

def group_gene_ids(gene_ids, gene_id_map):
    gene_sag_ids = np.array([gene_id_map[gene_id] for gene_id in gene_ids])
    unique_sag_ids = np.unique(gene_sag_ids)
    gene_ids_arr = np.array(gene_ids)
    sag_gene_id_dict = {}
    for sag_id in unique_sag_ids:
        sag_gene_id_dict[sag_id] = gene_ids_arr[gene_sag_ids == sag_id]
    return sag_gene_id_dict

def group_og_table_genes(gene_ids, og_table):
    #sag_columns = list(og_table.columns[4:])
    sag_columns = [col for col in og_table.columns if 'Uncmic' in col]
    prefix_sag_dict = make_prefix_dict(og_table)
    grouped_gene_ids = {}
    for gene_id in gene_ids:
        sag_id = prefix_sag_dict[gene_id.split('_')[0]]
        if sag_id in grouped_gene_ids:
            grouped_gene_ids[sag_id].append(gene_id)
        else:
            grouped_gene_ids[sag_id] = [gene_id]
    return grouped_gene_ids

def map_gene_ids_to_sags(gene_ids, gene_prefix_dict):
    gene_id_map = {}
    for gene_id in gene_ids:
        sag_id = gene_prefix_dict[gene_id.split('_')[0]]
        if sag_id not in gene_id_map:
            gene_id_map[gene_id] = sag_id
    return gene_id_map

def make_prefix_dict(og_table):
    #sag_columns = list(og_table.columns[4:])
    sag_columns = [col for col in og_table.columns if 'Uncmic' in col]
    prefix_sag_dict = {}
    for sag_id in sag_columns:
        gene_ids = og_table[sag_id].dropna().values[0].split(';')
        prefix = gene_ids[0].split('_')[0]
        prefix_sag_dict[prefix] = sag_id
    return prefix_sag_dict

def read_cluster_alignments(cluster_ids, data_dir):
    '''
    Reads alignment files in pan_genome_sequences/ with given cluster IDs.
    Returns list of all sequences.
    '''
    alignment_records = []
    for cluster_id in cluster_ids:
        aln = AlignIO.read(f'{data_dir}pan_genome_sequences/{cluster_id}.fa.aln', 'fasta')
        alignment_records += list(aln)
    return alignment_records

def calculate_pairwise_identities(in_fasta, blastp_bin='blastp', makeblastdb_bin='makeblastdb', word_size=3, evalue=1E-3, qcov_perc=75, min_ident=0, num_threads=1, ext='.fasta', output_dir='./.scratch/'):
    '''
    Reads protein sequences from file and calculates identities between each query-hit pair
    using BLASTP.
    '''
    subprocess.run(['mkdir', '-p', output_dir], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Make BLAST database
    file_stem = in_fasta.split('/')[-1].replace(ext, '')
    f_blastdb = f'{output_dir}{file_stem}.blastdb'
    makeblastdb_out = subprocess.run([makeblastdb_bin, '-dbtype', 'prot', '-out', f_blastdb, '-in', in_fasta], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Run pairwise search
    f_blast_results = f'{output_dir}{file_stem}_blast_results.tab'
    blastp_out = subprocess.run([blastp_bin, '-db', f_blastdb, '-num_threads', f'{num_threads}', '-word_size', f'{word_size}', '-evalue', f'{evalue}', '-outfmt', '6 std qlen slen', '-qcov_hsp_perc', f'{qcov_perc}', '-out', f_blast_results, '-query', in_fasta], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Filter non-reciprocal hits and .abc graph
    f_graph_abc = f_blast_results.replace('_blast_results.tab', '.abc')
    write_protein_graph(f_blast_results, f_graph_abc, min_ident=min_ident)

def write_protein_graph(f_blast_results, f_out, min_ident=0, weights='bitscore'):
    hits = np.loadtxt(f_blast_results, delimiter='\t', dtype=object)

    # TODO: Implement other edge weighting methods
    if weights == 'bitscore':
        idx = [0, 1, 11]
    else:
        idx = [0, 1, 11] # use bitscore if none supplied

    with open(f_out, 'w') as out_handle:
        for hit_arr in hits:
            # Check pident threshold
            if float(hit_arr[2]) >= min_ident:
                out_handle.write('\t'.join(hit_arr[idx]) + '\n')

def filter_reciprocal_best_hits(blast_results):
    query_ids = np.unique(blast_results['qseqid'])
    query_hits = {}
    for query_id in query_ids:
        query_hits[query_id] = list(blast_results.loc[blast_results['qseqid'] == query_id, 'sseqid'])

    rbh_idx = []
    for query in query_ids:
        query_bh = blast_results.loc[blast_results['qseqid'] == query, :]
        query_bh = query_bh.loc[query_bh['sseqid'] != query, :] # remove self-hits if present
        best_hits = query_bh.loc[query_bh['bitscore'] == query_bh['bitscore'].max(), 'sseqid'].values
        for best_hit in best_hits:
            if best_hit in query_hits and query in query_hits[best_hit]:
                reciprocal_bh = blast_results.loc[blast_results['qseqid'] == best_hit, :]
                reciprocal_bh = reciprocal_bh.loc[reciprocal_bh['sseqid'] != best_hit, :]
                max_reciprocal_bitscore = reciprocal_bh['bitscore'].max()
                if reciprocal_bh.loc[reciprocal_bh['sseqid'] == query, 'bitscore'].values[0] == max_reciprocal_bitscore:
                    rbh_idx.append(query_bh.loc[query_bh['sseqid'] == best_hit, :].index[0])
                    rbh_idx.append(reciprocal_bh.loc[reciprocal_bh['sseqid'] == query, :].index[0])

    unique_idx = np.unique(rbh_idx)
    reciprocal_best_hit_results = blast_results.reindex(index=unique_idx)
    return reciprocal_best_hit_results.reset_index(drop=True)

def run_mcl(in_abc, out_clusters, mcl_inflation=1.5, num_threads=1):
    mcl_out = subprocess.run(['mcl', in_abc, '--abc', '-I', f'{mcl_inflation}', '-te', f'{num_threads}', '-o', out_clusters], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    return mcl_out

def process_mcl_clusters(in_mci, idx=1, cluster_prefix='YSG'):
    clusters = []
    with open(in_mci, 'r') as in_handle:
        for line in in_handle.readlines():
            cluster_ids = line.strip().split('\t')
            clusters.append(np.array(cluster_ids))

    # Assign cluster name based on reference genomes; else use idx
    named_clusters = {}
    for cluster_ids in clusters:
        cluster_id, idx = make_cluster_id(cluster_ids, cluster_prefix, group_idx=idx)
        named_clusters[cluster_id] = cluster_ids
    return named_clusters, idx

def make_cluster_id(cluster_ids, prefix, group_idx=1):
    osbp_idx = ['CYB' in gene_id for gene_id in cluster_ids]
    osa_idx = ['CYA' in gene_id for gene_id in cluster_ids]
    if sum(osbp_idx) > 0:
        cluster_id = cluster_ids[osbp_idx][0]
    elif sum(osa_idx) > 0:
        cluster_id = cluster_ids[osa_idx][0]
    else:
        cluster_id = f'{prefix}_{group_idx:04d}'
        group_idx += 1
    return cluster_id, group_idx

def write_cluster_seqs_to_file(clusters, pangenome_map, output_dir):
    cluster_files = {}
    for cluster_id in clusters:
        gene_ids = clusters[cluster_id]
        gene_records = []
        pangenome_idx = list(pangenome_map.annotations.index)
        for gene_id in gene_ids:
            if gene_id in pangenome_idx:
                gene_records.append(pangenome_map.extract_gene_record(gene_id))
        f_out = f'{output_dir}{cluster_id}.fna'
        SeqIO.write(gene_records, f_out, 'fasta')
        cluster_files[cluster_id] = f_out
    return cluster_files

def write_protein_seqs_to_file(clusters, pangenome_map, output_dir):
    cluster_files = {}
    for cluster_id in clusters:
        gene_ids = clusters[cluster_id]
        gene_records = []
        pangenome_idx = list(pangenome_map.annotations.index)
        for gene_id in gene_ids:
            if gene_id in pangenome_idx:
                gene_record = pangenome_map.extract_gene_record(gene_id)
                gene_record.seq = gene_record.seq.translate(table=11)
                gene_records.append(gene_record)
        f_out = f'{output_dir}{cluster_id}.faa'
        SeqIO.write(gene_records, f_out, 'fasta')
        cluster_files[cluster_id] = f_out
    return cluster_files


def align_cluster_seqs(f_seqs, f_aln=None, num_threads=1, aligner='muscle', aligner_bin='muscle'):
    ext = f_seqs.split('.')[-1]
    if f_aln is None:
        f_aln = f_seqs.replace(f'.{ext}', f'_aln.{ext}')
    aln_out = subprocess.run(['python3', 'codon_aware_align.py', '-i', f_seqs, '-o', f_aln, '-a', aligner, '-b', aligner_bin, '-e', ext, '-n', num_threads], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    return aln_out

def align_proteins(f_seqs, f_aln=None, mafft_bin='mafft', num_threads=1, verbose=False):
    if f_aln is None:
        ext = f_seqs.split('.')[-1]
        f_aln = f_seqs.replace(f'.{ext}', f'_aln.{ext}')
    aln_out = subprocess.call(' '.join([mafft_bin, '--thread', f'{num_threads}', '--auto', f_seqs, '>', f_aln]), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if verbose == True:
        print(aln_out)
    return f_aln

def build_tree(f_aln, seq_type='prot', fast_tree_bin='FastTree', verbose=False):
    ext = f_aln.split('.')[-1]
    f_tree = f_aln.replace(f'.{ext}', '.nwk')
    #tree_out = subprocess.run([fast_tree_bin, '-nj', '-noml', f_aln, '>', f_tree], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    if seq_type == 'nucl':
        tree_out = subprocess.call(' '.join([fast_tree_bin, '-nj', '-noml', '-nt', f_aln, '>', f_tree]), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        tree_out = subprocess.call(' '.join([fast_tree_bin, '-nj', '-noml', f_aln, '>', f_tree]), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if verbose == True:
        print(tree_out)
    return f_tree

class LinkageTable:
    def __init__(self, results_dir, locus_batch_map=None, fname_head=''):
        self.results_dir = results_dir
        if locus_batch_map is None:
            self.make_locus_batch_maps(results_dir, fname_head)
        else:
            self.locus_batch_dict, self.batch_locus_dict = pickle.load(open(locus_batch_map, 'rb'))

    def make_locus_batch_maps(self, results_dir, fname_head):
        results_files = sorted(glob.glob(f'{results_dir}{fname_head}_batch*.dat'))
        loci_dict = {}
        batch_dict = {}
        for fname in results_files:
            batch_results = pickle.load(open(fname, 'rb'))
            batch_dict[fname] = list(batch_results.keys())
            for og_id in batch_results:
                loci_dict[og_id] = fname
        self.locus_batch_dict = loci_dict
        self.batch_locus_dict = loci_dict

    def save_batch_map_file(self, fout):
        maps = (self.locus_batch_dict, self.batch_locus_dict)
        pickle.dump(maps, open(fout, 'wb'))

    def get_linkage_matrices(self, og_id):
        if og_id in self.locus_batch_dict:
            batch_fname = self.locus_batch_dict[og_id]
            batch_results = pickle.load(open(batch_fname, 'rb'))
            Dsq, Z = batch_results[og_id]
        else:
            Dsq = None
            Z = None
        return Dsq, Z

    def calculate_average_rsq(self, og_ids, epsilon=1E-8):
        # Sort OGs by batch file
        batch_sorted_ids = {}
        for oid in og_ids:
            batch_fname = self.locus_batch_dict[oid]
            if batch_fname in batch_sorted_ids:
                batch_sorted_ids[batch_fname].append(oid)
            else:
                batch_sorted_ids[batch_fname] = [oid]

        batch_sum_list = []
        weights_list = []
        for batch_fname in batch_sorted_ids:
            batch_results = pickle.load(open(batch_fname, 'rb'))
            if len(batch_sorted_ids[batch_fname]) > 1:
                rsq_list = []
                for oid in batch_sorted_ids[batch_fname]:
                    Dsq, Z = batch_results[oid]
                    rsq = Dsq / (Z + (np.abs(Z) < epsilon))
                    rsq_list.append(rsq)
                rsq_sum, weights = sum_rsq_matrices(rsq_list)
                batch_sum_list.append(rsq_sum)
                weights_list.append(weights)
            else:
                oid = batch_sorted_ids[batch_fname][0]
                Dsq, Z = batch_results[oid]
                rsq = Dsq / (Z + (np.abs(Z) < epsilon))
                batch_sum_list.append(rsq)
                weights_list.append(np.ones(rsq.shape))

        rsq_sum, weights_sum = sum_rsq_matrices(batch_sum_list, weights_list)
        return rsq_sum / weights_sum


def sum_rsq_matrices(r2_list, weights_list=None):
    max_sites = 0
    max_index = 0
    for i in range(len(r2_list)):
        r2 = r2_list[i]
        if r2.shape[0] > max_sites:
            max_sites = r2.shape[0]
            max_index = i
    r2_sum = np.zeros((max_sites, max_sites))
    weights_sum = np.zeros((max_sites, max_sites))
    for i, r2 in enumerate(r2_list):
        num_sites = r2.shape[1]

        # Skip empty matrices
        if num_sites == 0:
            continue

        r2_sum[:num_sites, :][:, :num_sites] += r2
        if weights_list is None:
            weights_sum[:num_sites, :][:, :num_sites] += 1
        else:
            weights_sum[:num_sites, :][:, :num_sites] += weights_list[i]

    return r2_sum, weights_sum

def merge_table_rows(table_rows, sag_columns=None):
    if sag_columns is None:
        sag_columns = [col for col in table_rows.columns if 'Uncmic' in col]

    merged_row = pd.Series(index=list(table_rows.columns))
    for sag_id in sag_columns:
        if table_rows[sag_id].notna().sum() > 0:
            sag_genes_str = ';'.join([gene_str for gene_str in table_rows[sag_id].dropna()])
        else:
            sag_genes_str = None
        merged_row[sag_id] = sag_genes_str
    merged_row[['num_seqs', 'num_cells']] = table_rows[['num_seqs', 'num_cells']].sum(axis=0)
    merged_row['seqs_per_cell'] = merged_row['num_seqs'] / merged_row['num_cells']
    merged_row['avg_length'] = (table_rows['num_seqs'] * table_rows['avg_length']).sum() / table_rows['num_seqs'].sum()

    return merged_row


     

if __name__ == '__main__':
    #test_dir = '../data/single-cell/filtered_annotations/'
    #pangenome_map = PangenomeMap(test_dir)
    ##test_gene_id = 'Ga0393474_001_4947_5657'
    #test_gene_id = 'Ga0393474_001_476_1147'
    #gene_seq = pangenome_map.extract_gene_record(test_gene_id)
    #print(gene_seq)

    # Merge SOG entries
    test_og_table = '../results/tests/pangenome_construction/sscs_v3/filtered_orthogroups/sscs_filtered_orthogroup_presence.tsv'
    pangenome_map = PangenomeMap(f_orthogroup_table=test_og_table)
    og_table = pangenome_map.og_table
    print(og_table)

    sag_columns = pangenome_map.get_sag_ids()
    sog_ids = list(og_table.index)
    merged_og_ids = [f'YSG_100{i}' for i in range(10)]
    for og_id in merged_og_ids:
        og_sog_ids = [s for s in sog_ids if og_id in s]
        merged_row = merge_table_rows(og_table.loc[og_sog_ids, :], sag_columns=sag_columns)
        og_table = og_table.drop(og_sog_ids)
        og_table.loc[og_id, :] = merged_row

    og_table = og_table.reindex(sorted(list(og_table.index)))
    print(og_table)
    og_table.to_csv(test_og_table, sep='\t')
