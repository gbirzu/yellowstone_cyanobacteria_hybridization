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
import scipy.spatial.distance as distance
from analyze_hitchhiking_candidates import add_gene_id_map, map_ssog_id
from metadata_map import MetadataMap
from plot_utils import *

def make_methods_validation_figures(args):
    # Plot curves for different cloud cutoffs
    avg_length_fraction = 0.75
    for species in ['A', 'Bp']:
        if species == 'A':
            ax = set_up_linkage_curve_axis(ylim=(1E-3, 1.5E0))
        else:
            ax = set_up_linkage_curve_axis()
        for c in [0.03, 0.05, 0.1, 0.15, 0.2, 1.0]:
            linkage_results = pickle.load(open(f'{args.input_dir}sscs_core_ogs_{species}_linkage_curves_c{c}.dat', 'rb'))
            sigmad2 = average_sigmad_sq(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
            sigmad2_cg, x_cg = coarse_grain_linkage_array(sigmad2)
            ax.plot(x_cg, sigmad2_cg, '-o', ms=3, mec='none', lw=1, alpha=0.6, label=f'c={c}')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{args.figures_dir}syn_{species}_sigmad_sq_main_cloud_cutoff_comparison.pdf')

    # Compare A vs B'
    ax = set_up_linkage_curve_axis(ylim=(1E-3, 1.5E0))
    syna_linkage = pickle.load(open(f'{args.input_dir}sscs_core_ogs_A_linkage_curves_c0.1.dat', 'rb'))
    syna_sigmad2 = average_sigmad_sq(syna_linkage, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
    syna_sigmad2_cg, x_cg = coarse_grain_linkage_array(syna_sigmad2)
    ax.plot(x_cg, syna_sigmad2_cg, '-o', ms=3, mec='none', lw=1, alpha=0.6, color='tab:orange', label=r'$\alpha$')

    synbp_subsampled_linkage = pickle.load(open(f'{args.input_dir}sscs_core_ogs_Bp_subsampled_linkage_curves_c0.1.dat', 'rb'))
    synbp_sigmad2 = average_sigmad_sq(synbp_subsampled_linkage, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
    synbp_sigmad2_cg, x_cg = coarse_grain_linkage_array(synbp_sigmad2)
    ax.plot(x_cg, synbp_sigmad2_cg, '-o', ms=3, mec='none', lw=1, alpha=0.6, color='tab:blue', label=r'subsampled $\beta$')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}syn_A_vs_Bp_subsampled_sigmad_sq.pdf')

    # Compare trimming
    f_population_results = f'{args.input_dir}sscs_core_ogs_population_linkage_curves_c0.1.dat'
    population_linkage_curves = pickle.load(open(f_population_results, 'rb'))
    f_population_old_trim = f'{args.input_dir}sscs_core_ogs_population_linkage_curves_c0.1_old_trimming.dat'
    population_linkage_curves_old_trim = pickle.load(open(f_population_old_trim, 'rb'))
    plot_aln_trimming_comparison(population_linkage_curves_old_trim, population_linkage_curves, savefig=f'{args.figures_dir}population_linkage_aln_trimming_comparison.pdf')

    f_synbp_results = f'{args.input_dir}sscs_core_ogs_Bp_linkage_curves_c0.1.dat'
    synbp_linkage_curves = pickle.load(open(f_synbp_results, 'rb'))
    f_synbp_old_trim = f'{args.input_dir}sscs_core_ogs_Bp_linkage_curves_c0.1_old_trimming.dat'
    synbp_linkage_curves_old_trim = pickle.load(open(f_synbp_old_trim, 'rb'))
    plot_aln_trimming_comparison(synbp_linkage_curves, synbp_linkage_curves_old_trim, ylim=(1E-3, 1.5E0), savefig=f'{args.figures_dir}syn_Bp_linkage_aln_trimming_comparison.pdf')


    # Test averaging length
    plot_averaging_length_comparisons(population_linkage_curves, savefig=f'{args.figures_dir}population_linkage_avg_length_comparions.pdf')
    plot_averaging_length_comparisons(synbp_linkage_curves, savefig=f'{args.figures_dir}syn_Bp_linkage_avg_length_comparions.pdf')


def average_sigmad_sq(linkage_results, metric='sigmad_sq', min_sample_size=20, average_length_fraction=1):
    Dsq_list = []
    denom_list = []
    for og_id in linkage_results:
        Dsq, denom, sample_size = linkage_results[og_id]
        if sample_size >= min_sample_size:
            length_cutoff_idx = int(average_length_fraction * len(Dsq)) + 1
            Dsq_list.append(Dsq[:length_cutoff_idx])
            denom_list.append(denom[:length_cutoff_idx])
    if metric == 'sigmad_sq':
        Dsq_avg = ld.average_linkage_curves(Dsq_list)
        denom_avg = ld.average_linkage_curves(denom_list)
        return Dsq_avg / (denom_avg + (denom_avg == 0))

def coarse_grain_linkage_array(sigmad_sq, num_cg_points=20):
    x = np.arange(1, len(sigmad_sq) + 1)
    x_max = len(sigmad_sq) - 1
    x_bin = (np.log10(2 * x_max) - np.log10(0.5)) / num_cg_points
    x_log = np.geomspace(11, 2 * x_max, num_cg_points)
    sigmad2_cg = np.zeros(len(x_log) + 11)
    sigmad2_cg[:11] = sigmad_sq[:11]
    for i, x in enumerate(x_log):
        idx = i + 11
        jl = int(np.floor(x))
        jr = int(np.ceil(10**(np.log10(x) + x_bin)))
        sigmad2_cg[idx] = np.mean(sigmad_sq[jl:jr])
    x_cg = np.concatenate([np.arange(11), x_log])
    return sigmad2_cg, x_cg


def plot_aln_trimming_comparison(f_default, f_mixed_trimming, savefig=None):
    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'separation, $x$')
    ax.set_xscale('log')
    #ax.set_xlim(4E-1, 5E3)
    ax.set_xlim(8E-1, 1E4)
    ax.set_ylabel(r'linkage disequilibrium, $\sigma_d^2$')
    ax.set_yscale('log')
    ax.set_ylim(5E-3, 1.5E0)

    default_trimming_results = pickle.load(open(f_default, 'rb'))
    sigmad2 = average_sigmad_sq(default_trimming_results, metric='sigmad_sq')
    sigmad2_cg, x_cg = coarse_grain_linkage_array(sigmad2)
    ax.plot(x_cg, sigmad2_cg, '-o', ms=3, mec='none', lw=1, alpha=0.6, label=f'old trimming')

    default_trimming_results = pickle.load(open(f_mixed_trimming, 'rb'))
    sigmad2 = average_sigmad_sq(default_trimming_results, metric='sigmad_sq')
    sigmad2_cg, x_cg = coarse_grain_linkage_array(sigmad2)
    ax.plot(x_cg, sigmad2_cg, '-o', ms=3, mec='none', lw=1, alpha=0.6, label=f'new trimming')

    ax.legend(fontsize=8)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)


def plot_aln_trimming_comparison(default_trimming_results, mixed_trimming_results, ylim=(5E-3, 1.5E0), savefig=None):
    ax = set_up_linkage_curve_axis(ylim=ylim)
    sigmad2 = average_sigmad_sq(default_trimming_results, metric='sigmad_sq')
    sigmad2_cg, x_cg = coarse_grain_linkage_array(sigmad2)
    ax.plot(x_cg, sigmad2_cg, '-o', ms=3, mec='none', lw=1, alpha=0.6, label=f'old trimming')

    sigmad2 = average_sigmad_sq(mixed_trimming_results, metric='sigmad_sq')
    sigmad2_cg, x_cg = coarse_grain_linkage_array(sigmad2)
    ax.plot(x_cg, sigmad2_cg, '-o', ms=3, mec='none', lw=1, alpha=0.6, label=f'new trimming')

    ax.legend(fontsize=8)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)


def set_up_linkage_curve_axis(aspect=(1, 0.8), xlim=(8E-1, 1E4), xticks=None, ylim=(5E-3, 1.5E0), linkage_metric='$\sigma_d^2$'):
    fig = plt.figure(figsize=(aspect[0] * single_col_width, aspect[1] * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'separation, $x$', fontsize=14)
    ax.set_xscale('log')
    ax.set_xlim(xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_ylabel(f'linkage disequilibrium, {linkage_metric}', fontsize=14)
    ax.set_yscale('log')
    ax.set_ylim(ylim)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    return ax


def plot_averaging_length_comparisons(population_linkage_curves, savefig=None):
    ax = set_up_linkage_curve_axis()
    for avg_length_fraction in [1, 0.75, 0.5, 0.3, 0.25, 0.1]:
        sigmad2 = average_sigmad_sq(population_linkage_curves, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
        sigmad2_cg, x_cg = coarse_grain_linkage_array(sigmad2)
        ax.plot(x_cg, sigmad2_cg, '-o', ms=3, mec='none', lw=1, alpha=0.6, label=f'f={avg_length_fraction}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)

def subsample_alleles(allele_table, sample_sizes):
    subsampled_allele_table = allele_table.copy()
    for og_id in sample_sizes.index:
        og_sag_ids = list(subsampled_allele_table[og_id].dropna().index)
        if len(og_sag_ids) >= sample_sizes[og_id]:
            num_removed = len(og_sag_ids) - sample_sizes[og_id]
            removed_sag_ids = np.random.choice(og_sag_ids, size=num_removed, replace=False)
            subsampled_allele_table.loc[removed_sag_ids, og_id] = np.nan

    return subsampled_allele_table

def calculate_heterozygosities(filtered_allele_table):
    het_list = []
    for og_id in filtered_allele_table.columns:
        allele_ids, allele_counts = utils.sorted_unique(filtered_allele_table[og_id].dropna())
        allele_frequencies = allele_counts / np.sum(allele_counts)
        heterozygosity = 1 - np.sum(allele_frequencies**2)
        het_list.append(heterozygosity)
    return np.array(het_list)


def calculate_allele_pairwise_distances(filtered_allele_table, min_overlap=50):
    sag_ids = list(filtered_allele_table.index) 
    allele_pdist = pd.DataFrame(index=sag_ids, columns=sag_ids)
    print(allele_pdist)

    for idx1, sag1 in enumerate(sag_ids):
        allele_pdist.loc[sag1, sag1] = 0
        for idx2 in range(len(sag_ids) - idx1):
            sag2 = sag_ids[idx2]
            genotype1 = filtered_allele_table.loc[sag1, :].values
            genotype2 = filtered_allele_table.loc[sag2, :].values
            overlap_idx = (~np.isnan(genotype1)) & (~np.isnan(genotype2))
            if np.sum(overlap_idx) >= min_overlap:
                d12 = np.sum(genotype1[overlap_idx] != genotype2[overlap_idx]) / np.sum(overlap_idx)
                allele_pdist.loc[sag1, sag2] = d12
                allele_pdist.loc[sag2, sag1] = d12
            else:
                continue
                print(overlap_idx)
                print(genotype1[overlap_idx])
                print(genotype2[overlap_idx])
                print('\n')
    return allele_pdist


def randomize_table_columns(filtered_allele_table, rng):
    randomized_table = pd.DataFrame(index=filtered_allele_table.index, columns=filtered_allele_table.columns)
    for og_id in filtered_allele_table.columns:
        alleles = filtered_allele_table[og_id].values
        randomized_table.loc[:, og_id] = rng.permutation(alleles)
    return randomized_table



if __name__ == '__main__':
    # Default parameters
    input_dir = '../results/single-cell/linkage_disequilibria/'
    figures_dir = '../figures/analysis/linkage/'

    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--figures_dir', default=figures_dir, help='Directory in which figures are saved.')
    parser.add_argument('-I', '--input_dir', default=input_dir, help='Directory with input files.')
    parser.add_argument('-i', '--input_file', default=None, help='Linkage data input file.')
    parser.add_argument('--validate_methods', action='store_true', help='Make validation figures.')
    parser.add_argument('--analyze_alleles', action='store_true', help='Make protein alleles analysis figures.')
    parser.add_argument('--plot_linkage_curves', action='store_true', help='Make linkage curve and matrix figures.')
    parser.add_argument('--random_seed', type=int, default=12345)
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)

    if args.validate_methods:
        make_methods_validation_figures(args)

    if args.analyze_alleles:
        min_coverage = 20
        for species in ['A', 'Bp']:
            allele_table = pd.read_csv(f'{args.input_dir}sscs_core_ogs_{species}_protein_allele_table.tsv', sep='\t', index_col=0)
            print(species)

            # Plot locus coverages
            locus_coverage = allele_table.notna().sum(axis=0)
            if species == 'A':
                sample_sizes = locus_coverage

            #plot_pdf(locus_coverage.values, xlabel='coverage', savefig=f'{args.figures_dir}syn_{species}_core_ogs_coverage.pdf')
            ax = plot_pdf(locus_coverage.values, xlabel='coverage')
            ax.axvline(min_coverage, c='r')
            plt.savefig(f'{args.figures_dir}syn_{species}_core_ogs_coverage.pdf')

            # Filter allele table by locus coverage
            filtered_allele_table = allele_table.loc[:, locus_coverage >= min_coverage]

            # Allele number distribution
            plot_pdf(filtered_allele_table.max(axis=1), xlabel='number of alleles, $K_i$', savefig=f'{args.figures_dir}syn_{species}_allele_numbers.pdf')

            # Allele heterozygosity distribution
            het_arr = calculate_heterozygosities(filtered_allele_table)
            plot_pdf(het_arr, xlabel='allele heterozygosity, $h_i$', savefig=f'{args.figures_dir}syn_{species}_allele_heterozgyosity.pdf')

            if species == 'Bp':
                # Use subsampled B' as control for lower A coverage
                subsampled_allele_table = subsample_alleles(allele_table[locus_coverage.index], sample_sizes)
                subsampled_coverage = subsampled_allele_table.notna().sum(axis=0)
                plot_pdf(subsampled_coverage.values, xlabel='coverage', savefig=f'{args.figures_dir}syn_{species}_subsampled_core_ogs_coverage.pdf')
                filtered_subsampled_table = subsampled_allele_table.loc[:, subsampled_coverage >= min_coverage]
                subsampled_het_arr = calculate_heterozygosities(filtered_subsampled_table)
                plot_pdf(subsampled_het_arr, xlabel='allele heterozygosity, $h_i$', savefig=f'{args.figures_dir}syn_{species}_subsampled_allele_heterozgyosity.pdf')

            # Calculate pairwise distances
            min_loci_overlap = 100
            allele_pdist = calculate_allele_pairwise_distances(filtered_allele_table, min_overlap=min_loci_overlap)
            pdist_values = utils.get_matrix_triangle_values(allele_pdist.values.astype(float), k=1)
            x_bins = np.linspace(0, 1, 100)
            ax = plot_pdf(pdist_values[~np.isnan(pdist_values)], bins=x_bins, density=True, xlabel='allele pairwise distances, $d_{ij}$', ylabel='probability density', data_label='data', alpha=0.7)

            control_allele_table = randomize_table_columns(filtered_allele_table, rng)
            control_allele_pdist = calculate_allele_pairwise_distances(control_allele_table, min_overlap=min_loci_overlap)
            control_pdist_values = utils.get_matrix_triangle_values(control_allele_pdist.values.astype(float), k=1)

            hist, x_bins = np.histogram(control_pdist_values[~np.isnan(control_pdist_values)], bins=x_bins, density=True)
            x = np.array([np.mean(x_bins[i:i + 2]) for i in range(len(x_bins) - 1)])
            ax.plot(x, hist, c='tab:red', label='unlinked loci')
            ax.legend(fontsize=8)
            plt.savefig(f'{args.figures_dir}syn_{species}_allele_pdist_distribution.pdf')

            ax = plot_pdf(pdist_values[~np.isnan(pdist_values)], bins=x_bins, density=True, xlabel='allele pairwise distances, $d_{ij}$', ylabel='probability density', data_label='data', alpha=0.7, yscale='log')
            ax.plot(x, hist, c='tab:red', label='unlinked loci')
            ax.legend(fontsize=8)
            plt.savefig(f'{args.figures_dir}syn_{species}_allele_pdist_distribution_logy.pdf')


            utils.print_break()

        f_simulations = '../../../prototypes/bacterial_gene_sweeps/data/infinite_alleles/infinite_alleles_model_genotypes.dat'
        simulation_results = pickle.load(open(f_simulations, 'rb'))
        
        for c in [0, 204.8]:
            genotypes = simulation_results[c]
            pdist = distance.squareform(distance.pdist(genotypes, metric='hamming'))
            pdist_values = utils.get_matrix_triangle_values(pdist, k=1)
            print(genotypes)
            print(c, np.mean(pdist_values))

    if args.plot_linkage_curves:
        c = 0.1
        avg_length_fraction = 0.75
        color_dict = {'A':'tab:orange', 'Bp':'tab:blue', 'Bp_subsampled':'gray', 'population':'k'}
        label_dict = {'A':r'$\alpha$', 'Bp':r'$\beta$', 'Bp_subsampled':r'$\beta$ (subsampled)', 'population':r'whole population'}


        random_gene_linkage = {}
        for species in ['A', 'Bp', 'population']:
            gene_pair_results = pickle.load(open(f'{args.input_dir}sscs_core_ogs_{species}_gene_pair_linkage_c{c}.dat', 'rb'))

            gene_pair_linkage = gene_pair_results['sigmad_sq']
            sample_sizes = gene_pair_results['sample_sizes']
            min_sample_size = 20
            min_coverage = 0.9
            pseudocount = 1E-5
            filtered_pair_linkage = gene_pair_linkage.mask(sample_sizes < min_sample_size, np.nan)
            sigmad2_values = utils.get_matrix_triangle_values(filtered_pair_linkage.values, k=1).astype(float)
            sigmad2_values = sigmad2_values[np.isfinite(sigmad2_values.astype(float))]
            random_gene_linkage[species] = np.mean(rng.choice(sigmad2_values, size=1000))

            og_coverage = filtered_pair_linkage.notna().sum(axis=1) / len(filtered_pair_linkage)
            filtered_pair_linkage = filtered_pair_linkage.loc[og_coverage >= min_coverage, og_coverage >= min_coverage]
            print(filtered_pair_linkage)
            plot_pdist_clustermap2(filtered_pair_linkage, cbar_label=r'$\langle \sigma_d^2 \rangle$', grid=False, savefig=f'{args.figures_dir}syn_{species}_gene_pair_linkage.pdf')
            x_bins = np.geomspace(1E-5, 1, 100)
            plot_pdf(sigmad2_values + pseudocount, bins=x_bins, xlabel=r'gene linkage, $\sigma_d^2$', density=True, xscale='log', savefig=f'{args.figures_dir}syn_{species}_gene_pair_linkage_distribution.pdf')
            plot_pdf(utils.get_matrix_triangle_values(sample_sizes.values, k=1), xlabel=r'gene pair sample size, $n_2$', density=True, savefig=f'{args.figures_dir}syn_{species}_gene_pair_sample_size_distribution.pdf')
        pickle.dump(random_gene_linkage, open(f'{args.input_dir}sscs_core_ogs_random_gene_linkage_c{c}.dat', 'wb'))

        # Plot all curves
        ax = set_up_linkage_curve_axis(xlim=(0.8, 2E4), xticks=[1E0, 1E1, 1E2, 1E3, 1E4], ylim=(5E-3, 1.5E0))
        for species in ['A', 'Bp', 'population']:
            linkage_results = pickle.load(open(f'{args.input_dir}sscs_core_ogs_{species}_linkage_curves_c{c}.dat', 'rb'))
            sigmad2 = average_sigmad_sq(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
            sigmad2_cg, x_cg = coarse_grain_linkage_array(sigmad2)
            ax.plot(x_cg[:-5], sigmad2_cg[:-5], '-o', ms=3, mec='none', lw=1, alpha=1.0, c=color_dict[species], label=label_dict[species])
            if species in random_gene_linkage:
                ax.scatter(1.5E4, random_gene_linkage[species], s=20, ec='none', fc=color_dict[species])

        ax.axvline(1E4, ls='--', c='k')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{args.figures_dir}syn_sigmad_sq_curves.pdf')


        # Plot theory comparisons
        ax = set_up_linkage_curve_axis(xlim=(0.8, 2E4), xticks=[1E0, 1E1, 1E2, 1E3, 1E4], ylim=(5E-3, 1.5E0))
        species = 'A'
        linkage_results = pickle.load(open(f'{args.input_dir}sscs_core_ogs_{species}_linkage_curves_c{c}.dat', 'rb'))
        sigmad2 = average_sigmad_sq(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
        sigmad2_cg, x_cg = coarse_grain_linkage_array(sigmad2)
        ax.plot(x_cg[:-5], sigmad2_cg[:-5], '-o', ms=3, mec='none', lw=1, alpha=1.0, c=color_dict[species], label=label_dict[species])
        if species in random_gene_linkage:
            ax.scatter(1.5E4, random_gene_linkage[species], s=20, ec='none', fc=color_dict[species])
        lmax = 2000
        rho = 0.02
        theta = 0.03
        x_theory = np.arange(1, lmax)
        y_theory = er2.sigma2_theory(rho * x_theory, theta)
        ax.plot(x_theory, y_theory, lw=1, ls='-', c='k', label=f'theory ($\\rho={rho}$)')
        ax.plot(x_theory, 2 * y_theory, lw=1, ls='--', c='k', label=f'theory (intercept fit)')
        ax.axvline(1E4, ls='--', c='gray')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{args.figures_dir}syna_sigmad_sq_neutral_fit.pdf')

        ax = set_up_linkage_curve_axis(xlim=(0.8, 2E4), xticks=[1E0, 1E1, 1E2, 1E3, 1E4], ylim=(3E-3, 1.5E0))
        species = 'Bp'
        linkage_results = pickle.load(open(f'{args.input_dir}sscs_core_ogs_{species}_linkage_curves_c{c}.dat', 'rb'))
        sigmad2 = average_sigmad_sq(linkage_results, metric='sigmad_sq', average_length_fraction=avg_length_fraction)
        sigmad2_cg, x_cg = coarse_grain_linkage_array(sigmad2)
        ax.plot(x_cg[:-5], sigmad2_cg[:-5], '-o', ms=3, mec='none', lw=1, alpha=1.0, c=color_dict[species], label=label_dict[species])
        if species in random_gene_linkage:
            ax.scatter(1.5E4, random_gene_linkage[species], s=20, ec='none', fc=color_dict[species])
        rho = 0.1
        x_theory = np.arange(1, lmax)
        y_theory = er2.sigma2_theory(rho * x_theory, theta)
        ax.plot(x_theory, y_theory, lw=1, ls='-', c='k', label=f'theory ($\\rho={rho}$)')
        ax.plot(x_theory, 2 * y_theory, lw=1, ls='--', c='k', label=f'theory (intercept fit)')
        ax.axvline(1E4, ls='--', c='gray')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{args.figures_dir}synbp_sigmad_sq_neutral_fit.pdf')

