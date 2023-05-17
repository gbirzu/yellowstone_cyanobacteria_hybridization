import argparse
import numpy as np
import pandas as pd
import pickle
import utils
import scipy.sparse as sparse
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
from pangenome_utils import PangenomeMap
from metadata_map import MetadataMap
from Bio import AlignIO


def calculate_rsquared(aln, i_ref=0, unbiased=False):
    Dsq, denom = align_utils.calculate_ld_matrices_vectorized(aln, reference=i_ref, unbiased=unbiased)
    r_sq = Dsq / (denom + (denom == 0))
    return r_sq

def calculate_rsquared_fast(aln, i_ref=0, unbiased=False):
    '''
    Calculates r^2 matrix just for SNPs and then uses SNP locations to add zero values.
    '''
    # Calculate values for SNPs only
    aln_snps, x_snps = seq_utils.get_snps(aln, return_x=True)
    Dsq_snps, denom_snps = align_utils.calculate_ld_matrices_vectorized(aln_snps, reference=i_ref, unbiased=unbiased)

    # Fill zeros
    x_nonsnps = [i for i in range(aln.get_alignment_length()) if i not in x_snps]
    Dsq = utils.fill_matrix_zeros(Dsq_snps, x_nonsnps)
    denom = utils.fill_matrix_zeros(denom_snps, x_nonsnps)
    return Dsq / (denom + (denom == 0))

def calculate_sigmad_matrices(aln, i_ref=0, unbiased=False):
    aln_snps, x_snps = seq_utils.get_snps(aln, return_x=True)
    Dsq_snps, denom_snps = align_utils.calculate_ld_matrices_vectorized(aln_snps, reference=i_ref, unbiased=unbiased)

    # Fill zeros
    x_nonsnps = [i for i in range(aln.get_alignment_length()) if i not in x_snps]
    Dsq = utils.fill_matrix_zeros(Dsq_snps, x_nonsnps)
    denom = utils.fill_matrix_zeros(denom_snps, x_nonsnps)
    return Dsq, denom


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


def average_matrix_over_distances(mat, denom=2, coarse_grain=False, num_cg_points=20, idx=None, min_depth=1):
    x_max = int(mat.shape[0] / denom)
    mat_marginal = np.zeros(x_max)
    depth = np.zeros(x_max)

    # Get site distance matrix
    if idx is None:
        idx = np.arange(mat.shape[0])
    dxy = np.array([idx for i in range(len(idx))])
    dxy = np.abs(dxy - dxy.T).astype(int)
    x_arr = np.unique(dxy[np.where(dxy <= x_max)])

    #for k in range(x_max):
    #    mat_marginal[k] = np.trace(mat, offset=k) / len(np.diag(mat, k=k))
    for d in x_arr:

        # Get indices for sites d distance apart 
        x_idx, y_idx = np.where(dxy == d)
        x = idx[x_idx]
        y = idx[y_idx]

        # Average over sites d distance apart
        mat_marginal[d] = np.mean(mat[x, y])
        depth[d] = len(x_idx)

    if coarse_grain == True:
        x_bin = (np.log10(2 * x_max) - np.log10(0.5)) / num_cg_points
        x_log = np.geomspace(11, 2 * x_max, num_cg_points)
        marginal_cg = np.zeros(len(x_log) + 11)
        marginal_cg[:11] = mat_marginal[:11]
        depth_cg = np.zeros(len(x_log) + 11)
        depth_cg[:11] = depth_cg[:11]
        for i, x in enumerate(x_log):
            i_cg = i + 11
            jl = int(np.floor(x))
            jr = int(np.ceil(10**(np.log10(x) + x_bin)))
            marginal_cg[i_cg] = np.mean(mat_marginal[jl:jr][depth[jl:jr] >= min_depth])
            depth_cg[i_cg] = np.sum(depth[jl:jr][depth[jl:jr] >= min_depth])
        x_cg = np.concatenate([np.arange(11), x_log])
        output = (x_cg, marginal_cg, depth_cg)
    else:
        output = (x_arr, mat_marginal, depth)

    return output


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

def average_linkage_curves(r2_list):
    max_sites = 0
    for r2 in r2_list:
        if len(r2) > max_sites:
            max_sites = len(r2)

    r2_avg = np.zeros(max_sites)
    counts = np.zeros(max_sites)
    for r2 in r2_list:
        num_sites = len(r2) 
        r2_avg[:num_sites] += r2
        counts[:num_sites] += 1
    r2_avg /= counts
    return r2_avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--ext', default='fna')
    parser.add_argument('-g', '--orthogroup_table', help='File with orthogroup table.')
    parser.add_argument('-l', '--aln_files_list', default=None)
    parser.add_argument('-m', '--min_aln_length', default=50)
    parser.add_argument('-o', '--output_file')
    parser.add_argument('--metric', default='r_sq')
    parser.add_argument('--trimming', default='default')
    parser.add_argument('--sites', default='all')
    parser.add_argument('--species', default='all')
    parser.add_argument('--average', default='none')
    args = parser.parse_args()

    aln_files = np.loadtxt(args.aln_files_list, dtype=object, ndmin=1)
    pangenome_map = PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    og_table = pangenome_map.og_table
    metadata = MetadataMap()
    species_sorted_sags = metadata.sort_sags(pangenome_map.get_sag_ids(), by='species')

    linkage_disequilibria = {}
    for f_aln in aln_files:
        og_id = f_aln.split('/')[-1].replace(f'_aln.{args.ext}', '')
        print(og_id)

        aln = seq_utils.read_alignment(f_aln)

        # Filter cells
        if args.species == 'all':
            filtered_aln = aln
        elif args.species == 'ABp':
            synabp_sag_ids = np.concatenate([species_sorted_sags[species] for species in ['A', 'Bp']])
            synabp_gene_ids = pg_utils.read_gene_ids(og_table.loc[og_id, synabp_sag_ids], drop_none=True)
            filtered_aln = align_utils.get_subsample_alignment(aln, synabp_gene_ids)
        elif args.species == 'A':
            syna_sag_ids = species_sorted_sags['A']
            syna_gene_ids = pg_utils.read_gene_ids(og_table.loc[og_id, syna_sag_ids], drop_none=True)
            filtered_aln = align_utils.get_subsample_alignment(aln, syna_gene_ids)
        elif args.species == 'Bp':
            synbp_sag_ids = species_sorted_sags['Bp']
            synbp_gene_ids = pg_utils.read_gene_ids(og_table.loc[og_id, synbp_sag_ids], drop_none=True)
            filtered_aln = align_utils.get_subsample_alignment(aln, synbp_gene_ids)


        # Trim alignments
        if args.trimming == 'default':
            trimmed_aln, x_trimmed = align_utils.trim_alignment_and_remove_gaps(filtered_aln)
        elif args.trimming == 'mixed_species_ogs':
            trimmed_aln, x_trimmed = align_utils.clean_mixed_orthogroup_alignments(filtered_aln)

        if trimmed_aln.get_alignment_length() > args.min_aln_length:
            if args.metric == 'r_sq':
                r_sq = calculate_rsquared(trimmed_aln)

                if args.average == 'none':
                    linkage_disequilibria[og_id] = r_sq
                elif args.average == 'distance':
                    r_sq_avg = average_over_distance(r_sq)
                    linkage_disequilibria[og_id] = r_sq_avg

            elif args.metric == 'sigma_sq':
                D_sq, denom = align_utils.calculate_ld_matrices_vectorized(trimmed_aln, reference=0, unbiased=False)
                linkage_disequilibria[og_id] = (D_sq, denom)

    pickle.dump(linkage_disequilibria, open(args.output_file, 'wb'))

