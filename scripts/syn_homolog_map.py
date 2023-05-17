import argparse
import os
import re
import glob
import numpy as np
import pandas as pd
import pickle
import utils as analysis
import seq_processing_utils as seq_utils
from metadata_map import MetadataMap
from Bio import SeqIO
from Bio import SearchIO
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.SeqRecord import SeqRecord
from Bio.Blast import NCBIXML

class SynHomologMap:
    def __init__(self, blast_results_dir='../results/reference_genomes/',
                 osa_genbank='../data/reference_genomes/CP000239.genbank',
                 osbp_genbank='../data/reference_genomes/CP000240.genbank',
                 build_maps=False):
        self.read_ref_genomes(osa_genbank, osbp_genbank)
        self.read_blast_results(blast_results_dir)
        self.extract_top_hsps()
        self.map_query_ids()
        if build_maps == True:
            self.build_paralog_dicts()
            self.build_ortholog_dicts()

    def read_ref_genomes(self, osa_ref_fname, osbp_ref_fname):
        self.genome_length = {}
        self.genome_length['osa'] = self.get_genome_length(osa_ref_fname)
        self.genome_length['osbp'] = self.get_genome_length(osbp_ref_fname)
        osa_cds, osa_seq = self.make_genbank_cds_dict(osa_ref_fname)
        osbp_cds, osbp_seq = self.make_genbank_cds_dict(osbp_ref_fname)
        self.cds = {**osa_cds, **osbp_cds}
        self.num_cds = {'osa':len(osa_cds), 'osbp':len(osbp_cds)}
        self.genome_seqs = {'osa':osa_seq, 'osbp':osbp_seq}

    def get_genome_length(self, f_genbank):
        gb_record = SeqIO.read(f_genbank, 'genbank')
        return gb_record.features[0].location.end

    def make_genbank_cds_dict(self, f_genbank):
        genbank_record = SeqIO.read(f_genbank, 'genbank')
        genbank_cds_dict = {}
        for feature in genbank_record.features:
            if feature.type == 'CDS':
                locus_tag = feature.qualifiers['locus_tag'][0]
                genbank_cds_dict[locus_tag] = feature
        return genbank_cds_dict, genbank_record.seq

    def read_blast_results(self, blast_results_dir):
        self.blast_results = {}
        #self.blast_results['osa-osa'] = read_blast_results(f'{blast_results_dir}osa-osa_blast.xml')
        #self.blast_results['osa-osbp'] = read_blast_results(f'{blast_results_dir}osa-osbp_blast.xml')
        #self.blast_results['osbp-osa'] = read_blast_results(f'{blast_results_dir}osbp-osa_blast.xml')
        #self.blast_results['osbp-osbp'] = read_blast_results(f'{blast_results_dir}osbp-osbp_blast.xml')
        self.blast_results['osa-osa'] = read_blast_results(f'{blast_results_dir}osa_osa_blast.xml')
        self.blast_results['osa-osbp'] = read_blast_results(f'{blast_results_dir}osa_osbp_blast.xml')
        self.blast_results['osbp-osa'] = read_blast_results(f'{blast_results_dir}osbp_osa_blast.xml')
        self.blast_results['osbp-osbp'] = read_blast_results(f'{blast_results_dir}osbp_osbp_blast.xml')


    def extract_top_hsps(self):
        self.blast_hits = {}
        for alnmt, qresults in self.blast_results.items():
            self.blast_hits[alnmt] = self.extract_blast_hits(qresults)

    def extract_blast_hits(self, blast_qresults):
        blast_hits = {}
        for query in blast_qresults:
            hit_info = {'query_len':query.seq_len, 'hits':[]}
            if len(query) > 0:
                for hit in query:
                    top_hsp = hit.hsps[0]
                    hit_info['hits'].append(top_hsp)
            blast_hits[query.id] = hit_info
        return blast_hits

    def map_query_ids(self):
        self.tag_query_id = {}
        for alnmt in ['osa-osa', 'osbp-osbp']:
            for query_id in self.blast_hits[alnmt].keys():
                gene_tag = analysis.extract_syn_gene_tag_from_fasta_header(query_id)
                self.tag_query_id[gene_tag] = query_id

    def build_paralog_dicts(self):
        self.paralogs = {}
        for species in ['osa', 'osbp']:
            alnmt = f'{species}-{species}'
            paralog_map = self.find_paralogs(alnmt)
            self.paralogs[species] = {'paralog_map':paralog_map}
            self.paralogs[species]['paralog_sets'] = self.find_paralog_sets(paralog_map)
            self.build_paralog_set_map(species)
            self.map_paralog_identities(species)

    def find_paralogs(self, alnmt):
        query_hits = self.blast_hits[alnmt]
        paralog_dict = {}
        for query_id, hit_info in query_hits.items():
            query_tag = analysis.extract_syn_gene_tag_from_fasta_header(query_id)
            hit_tags = self.get_hit_tags(hit_info['hits'], query_tag)
            paralog_dict[query_tag] = sorted(hit_tags)
        return paralog_dict

    def get_hit_tags(self, hits, query_tag):
        hit_tags = []
        for hit in hits:
            hit_tag = analysis.extract_syn_gene_tag_from_fasta_header(hit.hit.id)
            if hit_tag != query_tag:
                hit_tags.append(hit_tag)
        return hit_tags

    def find_paralog_sets(self, paralog_map):
        sets = set()
        for gene_tag, paralogs in paralog_map.items():
            if len(paralogs) > 0:
                sets.add(tuple(sorted([gene_tag] + paralogs)))
        return list(sets)

    def build_paralog_set_map(self, species):
        set_map = {}
        for paralog_set in self.paralogs[species]['paralog_sets']:
            for paralog in paralog_set:
                set_map[paralog] = paralog_set
        self.paralogs[species]['paralog_set_map'] = set_map

    def map_paralog_identities(self, species):
        paralog_map = self.paralogs[species]['paralog_map']
        paralog_identities = {}

        for gene_tag, paralogs in paralog_map.items():
            if len(paralogs) > 0:
                query_id = self.tag_query_id[gene_tag]
                paralog_identities[gene_tag] = self.find_hit_identities(query_id, f'{species}-{species}')
        self.paralogs[species]['paralog_identities'] = paralog_identities

    def find_hit_identities(self, query_id, alnmt, exclude_first=True, top_hit=False):
        if exclude_first == True:
            homolog_hits = self.blast_hits[alnmt][query_id]['hits'][1:] # exclude top hit; for self-alignments only!
        else:
            homolog_hits = self.blast_hits[alnmt][query_id]['hits']
        if top_hit == False:
            hit_idents = []
            for hit in homolog_hits:
                ident = hit.ident_num / self.blast_hits[alnmt][query_id]['query_len']
                hit_idents.append(ident)
        else:
            hit = homolog_hits[0]
            hit_idents = hit.ident_num / self.blast_hits[alnmt][query_id]['query_len']

        return hit_idents

    def build_ortholog_dicts(self):
        ortholog_maps = {'ortholog_map':{}, 'ortholog_ident':{}}
        self.reciprocal_best_hits = {}
        for query_id, hit_info in self.blast_hits['osa-osbp'].items():
            if len(hit_info['hits']) > 0:
                best_hit = hit_info['hits'][0]
                reverse_hit_info = self.blast_hits['osbp-osa'][best_hit.hit.id]
                if len(reverse_hit_info['hits']) > 0:
                    reverse_best_hit = reverse_hit_info['hits'][0]
                    if reverse_best_hit.hit.id == query_id:
                        reverse_id = reverse_best_hit.query.id
                        self.reciprocal_best_hits[query_id] = reverse_id
                        self.reciprocal_best_hits[reverse_id] = query_id
                        query_tag = analysis.extract_syn_gene_tag_from_fasta_header(query_id)
                        reverse_tag = analysis.extract_syn_gene_tag_from_fasta_header(reverse_id)
                        ortholog_maps['ortholog_ident'][query_tag] = self.find_hit_identities(query_id, 'osa-osbp', exclude_first=False, top_hit=True)
                        ortholog_maps['ortholog_ident'][reverse_tag] = self.find_hit_identities(reverse_id, 'osbp-osa', exclude_first=False, top_hit=True)
                        ortholog_maps['ortholog_map'][query_tag] = reverse_tag
                        ortholog_maps['ortholog_map'][reverse_tag] = query_tag
        self.orthologs = ortholog_maps

    def find_lowest_tagged_homolog(self, gene_tag):
        if 'CYA' in gene_tag:
            osbp_ortholog = self.get_ortholog(gene_tag)
            if osbp_ortholog == 'none':
                homolog = self.find_lowest_tagged_gene_in_family(gene_tag)
            else:
                homolog = self.find_lowest_tagged_gene_in_family(osbp_ortholog)
        elif 'CYB' in gene_tag:
            homolog = self.find_lowest_tagged_gene_in_family(gene_tag)
        else:
            homolog = 'none'
        return homolog

    def find_lowest_tagged_gene_in_family(self, gene_tag):
        if 'CYA' in gene_tag:
            paralog_set_map = self.paralogs['osa']['paralog_set_map']
            if gene_tag in paralog_set_map:
                paralog = paralog_set_map[gene_tag][0]
            else:
                paralog = gene_tag
        elif 'CYB' in gene_tag:
            paralog_set_map = self.paralogs['osbp']['paralog_set_map']
            if gene_tag in paralog_set_map:
                paralog = paralog_set_map[gene_tag][0]
            else:
                paralog = gene_tag
        else:
            paralog = 'none'
        return paralog

    def get_cds_annotation(self, gene_tag):
        if gene_tag in self.cds:
            return self.cds[gene_tag]
        else:
            return 'none'

    def get_paralogs(self, gene_tag, species):
        if gene_tag in self.paralogs[species]['paralog_map']:
            return self.paralogs[species]['paralog_map'][gene_tag]
        else:
            return 'none'

    def get_paralog_identities(self, gene_tag, species):
        if gene_tag in self.paralogs[species]['paralog_identities']:
            return self.paralogs[species]['paralog_identities'][gene_tag]
        else:
            return [np.nan]

    def get_ortholog(self, gene_tag):
        if gene_tag in self.orthologs['ortholog_map']:
            return self.orthologs['ortholog_map'][gene_tag]
        else:
            return 'none'

    def get_ortholog_identity(self, gene_tag):
        if gene_tag in self.orthologs['ortholog_ident']:
            return self.orthologs['ortholog_ident'][gene_tag]
        else:
            return np.nan

    def is_single_copy_gene(self, gene_tag):
        if 'CYA' in gene_tag:
            if gene_tag in self.paralogs['osa']['paralog_set_map']:
                flag = False
            else:
                ortholog = self.get_ortholog(gene_tag)
                if ortholog != 'none' and ortholog in self.paralogs['osbp']['paralog_set_map']:
                    flag = False
                else:
                    flag = True
        elif 'CYB' in gene_tag:
            if gene_tag in self.paralogs['osbp']['paralog_set_map']:
                flag = False
            else:
                ortholog = self.get_ortholog(gene_tag)
                if ortholog != 'none' and ortholog in self.paralogs['osa']['paralog_set_map']:
                    flag = False
                else:
                    flag = True
        else:
            print(f'Gene tag {gene_tag} not found in Syn. homolog map!')
            flag = None
        return flag


def read_blast_results(results_file):
    results = SearchIO.parse(results_file, 'blast-xml')
    return list(results)

def test_all():
    homolog_map = SynHomologMap(blast_results_dir='../results/reference_genomes/evalue1E-100_ws6_output/')

    homolog_map.build_paralog_dicts()
    homolog_map.build_ortholog_dicts()

    print(homolog_map.paralogs['osa']['paralog_map'])
    print('\n')
    print(homolog_map.paralogs['osbp']['paralog_map'])
    print('\n')

    print(homolog_map.get_paralogs('CYA_1472', 'osa'))
    print(homolog_map.get_paralog_identities('CYA_1472', 'osa'))
    print(homolog_map.get_ortholog('CYA_2016'))
    print(homolog_map.get_ortholog_identity('CYA_2016'))
    print(homolog_map.get_cds_annotation('CYB_2811'))
    print('\n')

    print(homolog_map.get_paralogs('CYB_0081', 'osbp'))
    print(homolog_map.get_paralog_identities('CYB_0081', 'osbp'))

    print('\n')
    print(homolog_map.is_single_copy_gene('CYB_0081'))
    print(homolog_map.is_single_copy_gene('CYB_2845'))
    print(homolog_map.is_single_copy_gene('XXX_2845'))

    print(homolog_map.get_paralogs('CYA_0322', 'osa'))
    print(homolog_map.is_single_copy_gene('CYA_0322'))
    print(homolog_map.get_paralogs('CYA_2519', 'osa'))
    print(homolog_map.is_single_copy_gene('CYA_2519'))

    print(homolog_map.get_ortholog('CYA_1177'), homolog_map.get_ortholog('CYB_2598'))
    print(homolog_map.get_ortholog('CYA_1619'))
    print(homolog_map.get_ortholog('CYA_2812'), homolog_map.get_ortholog('CYB_2849'))

    print(sum(['CYA' in key for key in homolog_map.orthologs['ortholog_map']]))
    print(sum(['CYB' in key for key in homolog_map.orthologs['ortholog_map']]))


def export_its_sequences():
    syn_homolog_map = SynHomologMap(build_maps=True)
    its_segment_dict = {}
    its_segment_dict['osa'] = FeatureLocation(2311720, 2312275)
    its_segment_dict['osbp'] = FeatureLocation(1448896, 1449624)
    ref_id_dict = {'osa':'CP000239_ITS', 'osbp':'CP000240_ITS'}
    
    its_records = []
    for ref in ['osa', 'osbp']:
        ref_genome = syn_homolog_map.genome_seqs[ref]
        its_loc = its_segment_dict[ref]
        ref_its = its_loc.extract(ref_genome)
        its_records.append(SeqRecord(ref_its, id=ref_id_dict[ref]))
        print(ref, ref_its, len(ref_its))

    seq_utils.write_seqs(its_records, f'../data/reference_genomes/syn_os_its_seqs.fna')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--blast_results_dir', default='../results/reference_genomes/')
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()

    if args.test == True:
        test_all()
    else:
        export_its_sequences()


