import argparse
import subprocess
import glob
import pickle
import os
import utils
import numpy as np
import pandas as pd
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
import time
from pangenome_utils import PangenomeMap
from Bio import SeqIO
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def get_consensus_seq_records(og_table, aln_dir, ext='aln'):
    sag_columns = [col for col in og_table.columns if 'Uncmic' in col]
    consensus_seq_records = []
    for scog_id in og_table.index:
        # Get parent OG ID
        if '-' in scog_id:
            og_id, subcluster_id = scog_id.split('-')
        else:
            og_id = scog_id
        f_aln = f'{aln_dir}{og_id}_{ext}.fna'
        if os.path.exists(f_aln):
            print(scog_id, og_id)
            gene_ids = pg_utils.read_gene_ids(og_table.loc[scog_id, sag_columns], drop_none=True)
            aln = AlignIO.read(f_aln, 'fasta')
            scog_aln = align_utils.get_subsample_alignment(aln, gene_ids)
            consensus_seq_arr = seq_utils.get_consensus_seq(scog_aln, seq_type='codons', keep_gaps=False)
            seq_record = SeqRecord(Seq(''.join(consensus_seq_arr)), id=f'{scog_id}', description='consensus seq')
            consensus_seq_records.append(seq_record)
    return consensus_seq_records

def find_orthologous_groups_rbh(f_blast_results):
    blast_results = seq_utils.read_blast_results(f_blast_results, extra_columns=['qseqlen', 'sseqlen'])
    rbh_df = pg_utils.filter_reciprocal_best_hits(blast_results)
    return rbh_df

def make_rbh_dict(ref_rbh_files):
    rbh_dict = {}
    for f_results in ref_rbh_files:
        ref_id = f_results.split('/')[-1].strip('_merged_blast_results.tsv')
        ref_rbh_df = find_orthologous_groups_rbh(f_results)
        ref_rbh_dict = {}
        for i, row in ref_rbh_df.iterrows():
            ref_rbh_dict[row['qseqid']] = row['sseqid']
            ref_rbh_dict[row['sseqid']] = row['qseqid']
        rbh_dict[ref_id] = ref_rbh_dict
    return rbh_dict


def update_og_table(og_table, rbh_dict, ref_annotations):
    # Get sorted columns
    columns = ['locus_tag', 'gene', 'CYA_tag', 'CYB_tag'] + list(og_table.columns)
    col_name_dict = {'CP000239':'CYA_tag', 'CP000240':'CYB_tag'}

    for og_id in og_table.index:
        for ref in ['CP000239', 'CP000240']:
            ref_locus_tag, ref_id = get_ref_locus_tag(og_id, rbh_dict, ref) 
            if ref_locus_tag is not None:
                og_table.loc[og_id, col_name_dict[ref]] = ref_locus_tag
                gene = get_locus_tag_gene(ref_locus_tag, ref_annotations[ref_id])
                if gene is not None:
                    og_table.loc[og_id, 'gene'] = gene

    # Add locus_tag : in order of preference gene ID, CYB ID, CYA ID
    og_table.insert(0, 'locus_tag', og_table['gene'].values)
    og_table.loc[(og_table['locus_tag'].isnull() & og_table['CYB_tag'].notna()), 'locus_tag'] = og_table.loc[(og_table['locus_tag'].isnull() & og_table['CYB_tag'].notna()), 'CYB_tag']
    og_table.loc[og_table['locus_tag'].isnull(), 'locus_tag'] = og_table.loc[og_table['locus_tag'].isnull(), 'CYA_tag']

    return og_table.reindex(columns=columns)

def get_ref_locus_tag(og_id, rbh_dict, default_ref):
    if og_id in rbh_dict[default_ref]:
        locus_tag = rbh_dict[default_ref][og_id]
        ref_id = default_ref
    else:
        locus_tag = None
        ref_id = None
    return locus_tag, ref_id

def get_locus_tag_gene(locus_tag, ref_annotation):
    locus_annot = ref_annotation[locus_tag]
    if 'gene' in locus_annot.qualifiers:
        gene = locus_annot.qualifiers['gene'][0]
    else:
        gene = None
    return gene


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--data_dir', help='Directory with output of previous clustering steps.')
    parser.add_argument('-O', '--output_dir', help='Directory in which to store intermediate results.')
    parser.add_argument('-b', '--best_hits_file', default=None, help='File with paths to BLAST results files.')
    parser.add_argument('-e', '--extension', default='trimmed_aln', help='Alignment file extension.')
    parser.add_argument('-g', '--orthogroup_table', help='File with orthogroup table.')
    parser.add_argument('-o', '--output_file', default=None, help='Output file.')
    parser.add_argument('-p', '--prefix', default='sscs', help='Files prefix.')
    parser.add_argument('-r', '--references_file', default=None, help='File with paths to Genbank files for each reference.')
    parser.add_argument('-u', '--updates_file', default=None, help='File where mapping of OG ids is stored.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print mapping.')
    parser.add_argument('--genbank_extension', default='genbank', help='Extension for genbank files.')
    parser.add_argument('--random_seed', default=12345, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--export_ref_seqs', action='store_true', help='Export CDS seqs from list of reference genome files in Genbank format.')
    parser.add_argument('--export_og_consensus_seqs', action='store_true', help='Export consensus seq for OGs in OG table.')
    parser.add_argument('--map_rbh', action='store_true', help='Map genes to reference using RBH.')
    args = parser.parse_args()


    if args.export_ref_seqs == True:
        with open(args.references_file, 'r') as in_handle:
            for line in in_handle.readlines():
                f_ref = line.strip()
                ref_id = f_ref.split('/')[-1].strip(f'.{args.genbank_extension}')
                ref_cds_seqs = utils.extract_genbank_cds_seqs(f_ref, add_descriptions=False)
                seq_utils.write_seqs_dict(ref_cds_seqs, f'{args.output_dir}{ref_id}_cds.fna')

    elif args.export_og_consensus_seqs == True:
        np.random.seed(args.random_seed)
        pangenome_map = PangenomeMap(f_orthogroup_table=args.orthogroup_table)
        og_table = pangenome_map.og_table

        consensus_seq_records = get_consensus_seq_records(og_table, f'{args.data_dir}trimmed_aln/', ext=args.extension)
        SeqIO.write(consensus_seq_records, args.output_file, 'fasta')

    elif args.map_rbh == True:
        # Make RBH dict
        ref_rbh_files = []
        with open(args.best_hits_file, 'r') as in_handle:
            for line in in_handle.readlines():
                f_results = line.strip()
                ref_rbh_files.append(f_results)
        rbh_dict = make_rbh_dict(ref_rbh_files)

        # Read ref annotations
        ref_annotations = {}
        with open(args.references_file, 'r') as in_handle:
            for line in in_handle.readlines():
                f_ref = line.strip()
                ref_id = f_ref.split('/')[-1].strip(f'.{args.genbank_extension}')
                ref_cds = utils.read_genbank_cds(f_ref)
                ref_annotations[ref_id] = ref_cds

        # Update OG table
        pangenome_map = PangenomeMap(f_orthogroup_table=args.orthogroup_table)
        og_table = pangenome_map.og_table
        updated_og_table = update_og_table(og_table, rbh_dict, ref_annotations)
        updated_og_table.to_csv(args.output_file, sep='\t')
